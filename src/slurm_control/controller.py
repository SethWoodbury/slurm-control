"""A command ledger and Slurm submitter; no node availability forecasting."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid
import io
import zipfile
import hashlib

from .common import atomic_json, digest, read_json, utcnow
from . import ledger
from .policy import recommend_memory, estimate_runtime
from .slurm import Slurm, SlurmSubmissionRejected


def load_config(path):
    config = read_json(path)
    for key in ("state", "tasks_path"):
        expanded = os.path.expanduser(os.path.expandvars(config[key]))
        if "$" in expanded:
            raise ValueError(f"Unresolved environment variable in {key}: {config[key]}")
        config[key] = str((Path(path).resolve().parent / expanded).resolve())
    if not config.get("pools") or not config.get("workloads"):
        raise ValueError("pools and workloads are required")
    if len({p["id"] for p in config["pools"]}) != len(config["pools"]):
        raise ValueError("Duplicate pool IDs")
    return config


def initialize(db, config):
    tasks = read_json(config["tasks_path"])
    if isinstance(tasks, dict):
        tasks = tasks["tasks"]
    ledger.register(db, tasks, config["budget_gpu_hours"], prior_gpu_hours=config.get("prior_gpu_hours", 0))


def memory_for(db, workload_key, settings, tasks, *, eligible_models=None):
    """Apply the shared additive policy; preserve device and retry-floor evidence."""
    records = [json.loads(r[0]) for r in db.execute("SELECT sample FROM samples ORDER BY rowid")]
    override = None
    if "override_memory_gib" in settings:
        override = {"memory_gib": settings["override_memory_gib"], "reason": settings.get("override_reason")}
    receipt = recommend_memory(
        records, workload=workload_key, policy=settings, current_gib=settings.get("memory_gib"),
        eligible_models=eligible_models, retry_floor_gib=max((t["min_memory_gib"] for t in tasks), default=0),
        override=override,
    )
    if receipt["action"] != "ready":
        raise ValueError(f"{workload_key}: memory admission held: {receipt['reason']}")
    return receipt["memory_gib"], receipt


def preview_batches(db, config, *, limit=None):
    """Pure selection. Partition caps refer to this campaign's outstanding jobs."""
    maximum = config.get("max_outstanding", 256)
    active = [dict(r) for r in db.execute("SELECT * FROM batches WHERE state NOT IN ('TERMINAL','REJECTED')")]
    remaining_slots = max(0, maximum - len(active))
    slots = min(remaining_slots, config.get("max_admissions_per_tick", 32), limit or maximum)
    remaining = ledger.status(db)["remaining_gpu_hours"]
    tasks = [dict(r) for r in db.execute("SELECT * FROM tasks WHERE state='PENDING' ORDER BY task_id")]
    measurements = [json.loads(r[0]) for r in db.execute("SELECT sample FROM samples ORDER BY rowid")]
    timing_cache = {}
    used = set()
    batches = []
    pool_counts = {p["id"]: sum(json.loads(a["spec"])["pool_id"] == p["id"] for a in active) for p in config["pools"]}
    # Round robin configured pools; no changing vacancy snapshots or model quotas.
    while slots and len(used) < len(tasks):
        progressed = False
        for pool in config["pools"]:
            if pool.get("enabled", True) is False or pool_counts[pool["id"]] >= pool.get("max_outstanding", maximum):
                continue
            for workload_key, settings in config["workloads"].items():
                if pool.get("workloads") and workload_key not in pool["workloads"]:
                    continue
                candidates = [t for t in tasks if t["task_id"] not in used and json.loads(t["spec"])["workload_key"] == workload_key]
                if not candidates:
                    continue
                gpus = pool.get("gpus", 1)
                if gpus not in (0, 1):
                    raise ValueError("This first version supports CPU or one-GPU allocations")
                wall = int(pool["walltime_seconds"])
                if wall <= pool.get("reserve_seconds", 120):
                    raise ValueError("Walltime must exceed shutdown reserve")
                # Remaining budget may still fund a short tail allocation.
                if gpus:
                    wall = min(wall, math.floor(remaining * 3600 / gpus))
                startup = pool.get("startup_seconds", 120)
                reserve = pool.get("reserve_seconds", 120)
                margin = settings.get("time_margin_fraction", 0.25)
                backlog = pool.get("backlog_factor", 1.2)
                for name, value in (("startup_seconds", startup), ("reserve_seconds", reserve), ("time_margin_fraction", margin)):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"{name} must be finite and nonnegative")
                if not isinstance(backlog, (int, float)) or not math.isfinite(backlog) or not 1 <= backlog <= pool.get("max_backlog_factor", 1.5):
                    raise ValueError("backlog_factor is outside configured bounds")
                if wall <= startup + reserve or wall < pool.get("min_walltime_seconds", 600):
                    continue
                models = sorted(set(pool.get("allowed_models", ["cpu"])) & set(settings.get("allowed_models", pool.get("allowed_models", ["cpu"]))))
                if not models:
                    continue
                selected = []
                mean_total = 0
                for row in candidates:
                    task = json.loads(row["spec"])
                    estimates = task.get("runtime_estimates", {})
                    fallback = task.get("fallback_seconds", settings.get("fallback_seconds"))
                    timing = {}
                    for model in models:
                        initial = estimates.get(model, fallback)
                        key = (workload_key, model, initial)
                        if key not in timing_cache:
                            timing_cache[key] = estimate_runtime(measurements, workload=workload_key,
                                                                gpu_model=model, policy=settings, fallback=initial)
                        timing[model] = timing_cache[key]
                    times = [timing[m].get("mean_seconds") if timing[m]["action"] == "ready" else None for m in models]
                    if any(t is None or not math.isfinite(t) or t <= 0 for t in times):
                        continue
                    if any(timing[m]["next_command_seconds"] > wall - startup - reserve for m in models):
                        continue  # Cannot start this command on every allowed GPU.
                    planning_time = sum(times) / len(times)
                    usable = wall - startup - reserve
                    capacity = usable / (1 + margin) * backlog
                    if mean_total + planning_time > capacity:
                        continue
                    selected.append((row, dict(task, fallback_seconds=fallback,
                                              runtime_estimates={m: timing[m]["next_command_seconds"] for m in models})))
                    mean_total += planning_time
                    if len(selected) >= pool.get("max_commands", 256):
                        break
                if not selected:
                    continue
                memory, basis = memory_for(db, workload_key, settings, [r for r, _ in selected], eligible_models=models)
                # Shorten tail batches; the planner never asks for eight idle hours.
                # A mean across broad hardware must not shorten a one-command
                # tail below the slowest allowed device's deadline estimate.
                first_command_bound = max(selected[0][1]["runtime_estimates"].values())
                expected = max(mean_total * (1 + margin), first_command_bound) + startup + reserve
                actual_wall = min(wall, max(pool.get("min_walltime_seconds", 600), math.ceil(expected / 60) * 60))
                bid = "b-" + uuid.uuid4().hex[:16]
                spec = {"batch_id": bid, "pool_id": pool["id"], "tasks": [t for _, t in selected],
                        "partition": pool["partition"], "account": pool.get("account"), "gres": pool.get("gres"),
                        "gpus": gpus, "cpus": pool.get("cpus", 2), "allowed_models": models,
                        "memory_gib": memory, "memory_decision": basis, "walltime_seconds": actual_wall,
                        "reserve_seconds": pool.get("reserve_seconds", 120), "config_digest": digest(config),
                        "worker_python": config.get("worker_python", sys.executable),
                        "planned_at": utcnow()}
                batches.append(spec)
                used.update(t["task_id"] for _, t in selected)
                remaining -= actual_wall * gpus / 3600
                pool_counts[pool["id"]] += 1
                slots -= 1
                progressed = True
                break
            if not slots:
                break
        if not progressed:
            break
    return batches


def write_batch(state, spec):
    root = Path(state) / "batches" / spec["batch_id"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "batch.json"
    if path.exists() and digest(read_json(path)) != digest(spec):
        raise ValueError("Immutable batch path already contains different content")
    atomic_json(path, spec)
    # Pending jobs must not import a changing checkout. Freeze the small stdlib
    # package for each distinct source revision; all batches reuse its hash path.
    source = Path(__file__).resolve().parent
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(source.glob("*.py")):
            info = zipfile.ZipInfo("slurm_control/" + file.name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, file.read_bytes())
    content = buffer.getvalue()
    identity = hashlib.sha256(content).hexdigest()
    frozen = Path(state) / "sources" / (identity + ".zip")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        frozen.write_bytes(content)
    elif hashlib.sha256(frozen.read_bytes()).hexdigest() != identity:
        raise ValueError("Frozen source archive changed")
    atomic_json(root / "source.json", {"sha256": identity, "path": str(frozen)})
    argv = [spec.get("worker_python", sys.executable), "-m", "slurm_control.worker", "--batch", str(path), "--state", str(Path(state).resolve())]
    script = root / "worker.sh"
    script.write_text("#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH=" + shlex.quote(str(frozen)) + "\nexec " + shlex.join(argv) + "\n")
    script.chmod(0o700)
    return script, root


def submit_planned(db, state, scheduler, *, limit=32):
    results = []
    for row in list(db.execute("SELECT * FROM batches WHERE state='PLANNED' ORDER BY created_at LIMIT ?", (limit,))):
        spec = json.loads(row["spec"])
        script, root = write_batch(state, spec)
        ledger.mark_intent(db, row["batch_id"])
        try:
            job = scheduler.submit(spec, script, row["token"], root)
        except SlurmSubmissionRejected as exc:
            ledger.mark_rejected(db, row["batch_id"], str(exc))
            results.append({"batch_id": row["batch_id"], "status": "REJECTED", "error": str(exc)})
            break  # Configuration errors must not burn the rest of the queue.
        except Exception as exc:
            ledger.mark_ambiguous(db, row["batch_id"], exc)
            results.append({"batch_id": row["batch_id"], "status": "AMBIGUOUS", "error": str(exc)})
            break  # Avoid multiplying a scheduler outage into many unknown jobs.
        ledger.mark_submitted(db, row["batch_id"], job)
        results.append({"batch_id": row["batch_id"], "job_id": job, "status": "SUBMITTED"})
    return results


def recover_command_receipts(state, spec, job_id):
    """Preserve outcomes written before a killed worker's final batch receipt.

    This is read-only and called only after matching terminal Slurm evidence.
    Unknown commands remain unknown; a missing file is not proof of no start.
    """
    from .common import task_digest
    reports = []
    fingerprint = digest(spec)
    for task in spec["tasks"]:
        candidates = []
        root = Path(state) / "attempts" / task["task_id"]
        for attempt in sorted(root.glob("attempt*")):
            final = attempt / "receipt.json"
            start = attempt / "start.json"
            path = final if final.exists() else start
            if not path.exists():
                continue
            report = read_json(path)
            if report.get("batch_id") != spec["batch_id"] or str(report.get("job_id")) != str(job_id):
                continue
            if report.get("batch_digest") != fingerprint or report.get("task_id") != task["task_id"] or report.get("task_digest") != task_digest(task):
                raise ValueError("Per-command receipt identity differs from terminal allocation")
            if final.exists():
                if report.get("status") not in {"COMPLETE", "FAILED"}:
                    raise ValueError("Unknown per-command terminal status")
                if report["status"] == "COMPLETE" and report.get("returncode") != 0:
                    raise ValueError("Successful command receipt requires returncode zero")
            else:
                report = dict(report, status="STARTED", reason="final_command_receipt_missing")
            candidates.append(dict(report, recovered_from=str(path)))
        if len(candidates) > 1:
            raise ValueError("Multiple command attempts within one non-requeued allocation")
        reports.extend(candidates)
    if not reports:
        return None
    return {"batch_id": spec["batch_id"], "batch_digest": fingerprint, "job_id": str(job_id),
            "reason": "recovered_per_command_receipts", "tasks": reports,
            "partial_recovery": True, "host_scope": "unavailable", "host_peak_gib": None}


def reconcile(db, state, scheduler, config):
    rows = list(db.execute("SELECT * FROM batches WHERE state NOT IN ('TERMINAL','REJECTED','PLANNED')"))
    if not rows:
        return []
    queue = scheduler.queue()  # Failures propagate; missing scheduler data is not terminal.
    for row in rows:
        if row["job_id"] is None:
            matches = [job for job, info in queue.items() if info["token"] == row["token"]]
            if len(matches) == 1:
                ledger.mark_submitted(db, row["batch_id"], matches[0])
            elif len(matches) > 1:
                raise RuntimeError("Duplicate Slurm jobs share a submission token")
    rows = list(db.execute("SELECT * FROM batches WHERE state IN ('SUBMITTED','RUNNING')"))
    accounting = scheduler.accounting([r["job_id"] for r in rows if r["job_id"] not in queue])
    completed = []
    for row in rows:
        if row["job_id"] in queue:
            continue
        record = accounting.get(row["job_id"])
        if not record or not record["terminal"]:
            continue
        if record.get("token") != row["token"]:
            raise ValueError("Slurm accounting token differs from submission intent")
        receipts = []
        for path in (Path(state) / "receipts" / row["batch_id"]).glob("*.json"):
            candidate = read_json(path)
            if str(candidate.get("job_id")) == row["job_id"]:
                receipts.append((path, candidate))
        if len(receipts) > 1:
            raise ValueError("Multiple final receipts for a single allocation")
        receipt = receipts[0][1] if receipts else recover_command_receipts(state, json.loads(row["spec"]), row["job_id"])
        settings = config["workloads"][json.loads(row["spec"])["tasks"][0]["workload_key"]]
        ledger.finalize(db, row["batch_id"], receipt, record, max_attempts=config.get("max_attempts", 3),
                        oom_step_gib=settings.get("oom_step_gib", 4), max_memory_gib=settings.get("max_memory_gib", 128))
        completed.append({"batch_id": row["batch_id"], "state": record["state"]})
    return completed
