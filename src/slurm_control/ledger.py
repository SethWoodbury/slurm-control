"""Single-writer durable ledger. Workers publish receipts, never open this DB."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import sqlite3

from .common import digest, positive_number, task_digest, utcnow

TERMINAL = {"TERMINAL", "REJECTED"}


@contextmanager
def locked_ledger(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(state / "ledger.sqlite", timeout=20)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks (
              task_id TEXT PRIMARY KEY, digest TEXT NOT NULL, spec TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT 'PENDING', batch_id TEXT,
              attempts INTEGER NOT NULL DEFAULT 0, min_memory_gib REAL NOT NULL DEFAULT 0,
              last_reason TEXT);
            CREATE TABLE IF NOT EXISTS batches (
              batch_id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
              job_id TEXT, token TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
              reserved_gpu_hours REAL NOT NULL, actual_gpu_hours REAL,
              receipt_path TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (batch_id TEXT PRIMARY KEY, sample TEXT NOT NULL);
            ''')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()


def event(db, kind, payload):
    db.execute("INSERT INTO events(at,kind,payload) VALUES(?,?,?)", (utcnow(), kind, json.dumps(payload, sort_keys=True)))


def register(db, tasks, budget_gpu_hours, *, prior_gpu_hours=0):
    positive_number(budget_gpu_hours, "budget_gpu_hours")
    if isinstance(prior_gpu_hours, bool) or not isinstance(prior_gpu_hours, (float, int)) or not math.isfinite(prior_gpu_hours) or prior_gpu_hours < 0 or prior_gpu_hours > budget_gpu_hours:
        raise ValueError("Invalid prior GPU hours")
    for key, value in {"budget_gpu_hours": budget_gpu_hours, "prior_gpu_hours": prior_gpu_hours}.items():
        old = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if old and json.loads(old[0]) != value:
            raise ValueError(f"Existing {key} differs; use explicit override")
        db.execute("INSERT OR IGNORE INTO settings VALUES(?,?)", (key, json.dumps(value)))
    seen = set()
    for task in tasks:
        tid = task["task_id"]
        if not isinstance(tid, str) or not tid or tid in seen:
            raise ValueError("Task IDs must be unique nonempty strings")
        seen.add(tid)
        if not isinstance(task.get("argv"), list) or not task["argv"] or any(not isinstance(a, str) for a in task["argv"]):
            raise ValueError(f"{tid}: argv must be a nonempty string list")
        if not task.get("workload_key"):
            raise ValueError(f"{tid}: workload_key required")
        identity = task_digest(task)
        old = db.execute("SELECT digest FROM tasks WHERE task_id=?", (tid,)).fetchone()
        if old and old[0] != identity:
            raise ValueError(f"Task {tid} changed; use a new ID")
        db.execute("INSERT OR IGNORE INTO tasks(task_id,digest,spec) VALUES(?,?,?)", (tid, identity, json.dumps(task, sort_keys=True)))
    event(db, "register", {"tasks": len(tasks), "budget_gpu_hours": budget_gpu_hours})


def status(db):
    settings = {r[0]: json.loads(r[1]) for r in db.execute("SELECT * FROM settings")}
    actual = db.execute("SELECT COALESCE(SUM(actual_gpu_hours),0) FROM batches").fetchone()[0]
    reserved = db.execute("SELECT COALESCE(SUM(reserved_gpu_hours),0) FROM batches WHERE state NOT IN ('TERMINAL','REJECTED')").fetchone()[0]
    prior = settings.get("prior_gpu_hours", 0)
    return {"at": utcnow(), "tasks": {r[0]: r[1] for r in db.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state")},
            "batches": {r[0]: r[1] for r in db.execute("SELECT state,COUNT(*) FROM batches GROUP BY state")},
            "budget_gpu_hours": settings.get("budget_gpu_hours"), "actual_gpu_hours": actual,
            "prior_gpu_hours": prior, "reserved_gpu_hours": reserved,
            "remaining_gpu_hours": settings.get("budget_gpu_hours", 0) - actual - prior - reserved}


def reserve_batch(db, spec):
    tid = [t["task_id"] for t in spec["tasks"]]
    if not tid or len(set(tid)) != len(tid):
        raise ValueError("A batch requires distinct tasks")
    if len({t["workload_key"] for t in spec["tasks"]}) != 1:
        raise ValueError("A batch cannot mix workload families")
    seconds = positive_number(spec["walltime_seconds"], "walltime_seconds")
    gpus = spec.get("gpus", 1)
    if isinstance(gpus, bool) or gpus not in (0, 1):
        raise ValueError("Allocation must explicitly request zero or one GPU")
    positive_number(spec["memory_gib"], "memory_gib")
    hours = seconds * gpus / 3600
    if hours > status(db)["remaining_gpu_hours"] + 1e-9:
        raise ValueError("GPU-hour budget exhausted")
    for task in spec["tasks"]:
        row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        if row is None or row["state"] != "PENDING" or row["digest"] != task_digest(task):
            raise ValueError(f"Task unavailable or changed: {task['task_id']}")
        if spec["memory_gib"] < row["min_memory_gib"]:
            raise ValueError("Batch memory is below retry floor")
    bid = spec["batch_id"]
    token = "sc-" + digest(spec)[:24]
    db.execute("INSERT INTO batches(batch_id,spec,state,token,created_at,reserved_gpu_hours) VALUES(?,?,'PLANNED',?,?,?)",
               (bid, json.dumps(spec, sort_keys=True), token, utcnow(), hours))
    db.executemany("UPDATE tasks SET state='RESERVED',batch_id=? WHERE task_id=?", [(bid, x) for x in tid])
    event(db, "reserve", {"batch_id": bid, "gpu_hours": hours, "tasks": len(tid)})


def mark_intent(db, bid):
    row = db.execute("SELECT * FROM batches WHERE batch_id=?", (bid,)).fetchone()
    if row is None or row["state"] != "PLANNED":
        raise ValueError("Only a never-submitted batch can be submitted")
    db.execute("UPDATE batches SET state='SUBMITTING' WHERE batch_id=?", (bid,))
    event(db, "submission_intent", {"batch_id": bid, "token": row["token"]})
    db.commit()  # Durable before invoking sbatch, including lost-response cases.
    return dict(row)


def mark_submitted(db, bid, job_id):
    if not str(job_id).isdigit():
        raise ValueError("Expected a Slurm job ID")
    changed = db.execute("UPDATE batches SET state='SUBMITTED',job_id=? WHERE batch_id=? AND state IN ('SUBMITTING','AMBIGUOUS') AND job_id IS NULL", (str(job_id), bid)).rowcount
    if changed != 1:
        raise ValueError("Unexpected submission state")
    event(db, "submitted", {"batch_id": bid, "job_id": str(job_id)})
    db.commit()


def mark_ambiguous(db, bid, error):
    db.execute("UPDATE batches SET state='AMBIGUOUS',error=? WHERE batch_id=?", (str(error)[:4000], bid))
    event(db, "submission_ambiguous", {"batch_id": bid, "error": str(error)[:1000]})
    db.commit()


def mark_rejected(db, bid, reason):
    """Release a proven rejected submission; hold tasks until configuration fixes.

    The transport must distinguish an explicit Slurm rejection from timeout or
    any other ambiguous outcome before calling this function.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A proven submission rejection requires a reason")
    row = db.execute("SELECT * FROM batches WHERE batch_id=?", (bid,)).fetchone()
    if row is None or row["state"] not in {"SUBMITTING", "AMBIGUOUS"} or row["job_id"] is not None:
        raise ValueError("Only an unbound submission can be rejected")
    tasks = json.loads(row["spec"])["tasks"]
    for task in tasks:
        owner = db.execute("SELECT state,batch_id FROM tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        if owner is None or owner["state"] != "RESERVED" or owner["batch_id"] != bid:
            raise ValueError("Task owner changed before submission rejection")
    db.execute("UPDATE batches SET state='REJECTED',error=? WHERE batch_id=?", (reason[:4000], bid))
    db.executemany("UPDATE tasks SET state='HELD',batch_id=NULL,last_reason=? WHERE task_id=?",
                   [("submission_rejected: " + reason[:1000], task["task_id"]) for task in tasks])
    event(db, "submission_rejected", {"batch_id": bid, "reason": reason})
    db.commit()


def finalize(db, bid, receipt, accounting, *, max_attempts=3, oom_step_gib=4, max_memory_gib=128):
    """Only called after Slurm proves terminal; allocated time includes failures."""
    row = db.execute("SELECT * FROM batches WHERE batch_id=?", (bid,)).fetchone()
    if row is None:
        raise ValueError("Unknown batch")
    if row["state"] in TERMINAL:
        return False
    if not accounting.get("terminal") or str(accounting.get("job_id")) != str(row["job_id"]):
        raise ValueError("Matching terminal Slurm evidence required")
    spec = json.loads(row["spec"])
    seconds = accounting.get("elapsed_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Actual allocated elapsed time is required")
    if receipt and (receipt.get("batch_id") != bid or receipt.get("batch_digest") != digest(spec)):
        raise ValueError("Receipt identity mismatch")
    if receipt and receipt.get("job_id") is not None and str(receipt["job_id"]) != str(row["job_id"]):
        raise ValueError("Receipt belongs to a different Slurm job")
    reports = (receipt or {}).get("tasks", [])
    reported = {r["task_id"]: r for r in reports}
    if len(reported) != len(reports) or not set(reported) <= {task["task_id"] for task in spec["tasks"]}:
        raise ValueError("Receipt contains duplicate or unknown task IDs")
    for value, name in ((max_attempts, "max_attempts"),):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    positive_number(oom_step_gib, "oom_step_gib")
    positive_number(max_memory_gib, "max_memory_gib")
    state = accounting.get("state", "UNKNOWN")
    for task in spec["tasks"]:
        report = reported.get(task["task_id"], {})
        if report and report.get("task_digest") != task_digest(task):
            raise ValueError("Task receipt identity mismatch")
        old = db.execute("SELECT * FROM tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        if old["batch_id"] != bid or old["state"] != "RESERVED":
            raise ValueError("Task owner changed")
        reason = report.get("reason") or (receipt or {}).get("reason") or state
        started = report.get("status") in {"COMPLETE", "FAILED", "STARTED"} and not report.get("skipped")
        attempts = old["attempts"] + int(started)
        floor = old["min_memory_gib"]
        if report.get("status") == "COMPLETE":
            next_state = "COMPLETE"
        elif state.startswith("CANCELLED"):
            next_state = "HELD"
        elif report.get("status") == "UNSTARTED":
            next_state = "PENDING" if str(reason).lower() in {"insufficient_time", "preempted", "time_limit", "timeout", "node_fail", "boot_fail", "out_of_memory", "command_failed", "host_oom", "gpu_oom", "completed"} else "HELD"
        elif reason in {"gpu_oom", "GPU_OOM"}:
            next_state = "HELD"  # Greater device VRAM needs an eligible pool override.
        elif state == "OUT_OF_MEMORY" or reason in {"host_oom", "HOST_OOM"}:
            floor = max(floor, spec["memory_gib"] + oom_step_gib)
            attempts = max(attempts, old["attempts"] + 1)
            next_state = "PENDING" if attempts < max_attempts and floor <= max_memory_gib else "HELD"
        elif state in {"PREEMPTED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL"} or reason in {"preempted", "time_limit"}:
            attempts = max(attempts, old["attempts"] + int(not report))
            next_state = "PENDING" if attempts < max_attempts else "HELD"
        else:
            next_state = "HELD"  # Unknown/application failures require a reasoned override.
        db.execute("UPDATE tasks SET state=?,attempts=?,min_memory_gib=?,last_reason=?,batch_id=NULL WHERE task_id=?",
                   (next_state, attempts, floor, reason, task["task_id"]))
    actual = seconds * spec.get("gpus", 1) / 3600
    db.execute("UPDATE batches SET state='TERMINAL',actual_gpu_hours=? WHERE batch_id=?", (actual, bid))
    if receipt and receipt.get("host_peak_gib"):
        clean = (state == "COMPLETED" and receipt.get("reason", "completed") in {"completed", "insufficient_time"}
                 and any(r.get("status") == "COMPLETE" and not r.get("skipped") for r in reports)
                 and all(r.get("status") in {"COMPLETE", "UNSTARTED"} for r in reports))
        sample = {"batch_id": bid, "workload_key": spec["tasks"][0]["workload_key"],
                  "device": receipt.get("device"), "host_peak_gib": receipt["host_peak_gib"],
                  "host_scope": receipt.get("host_scope"), "status": state if clean or state != "COMPLETED" else "FAILED",
                  "reason": receipt.get("reason"),
                  "task_count": len(spec["tasks"]), "elapsed_seconds": seconds,
                  "walltime_seconds": spec["walltime_seconds"], "memory_gib": spec["memory_gib"],
                  "tasks": (receipt or {}).get("tasks", [])}
        db.execute("INSERT OR REPLACE INTO samples VALUES(?,?)", (bid, json.dumps(sample, sort_keys=True)))
    event(db, "finalized", {"batch_id": bid, "slurm_state": state, "actual_gpu_hours": actual})
    return True


def override_tasks(db, task_ids, *, action, reason, memory_gib=None):
    if not reason or not reason.strip():
        raise ValueError("An override reason is required")
    if action not in {"hold", "release", "memory"}:
        raise ValueError("Unknown override")
    for tid in task_ids:
        row = db.execute("SELECT * FROM tasks WHERE task_id=?", (tid,)).fetchone()
        if row is None or row["state"] in {"RESERVED", "COMPLETE"}:
            raise ValueError("Cannot override a live/reserved or completed task")
        if action == "memory":
            positive_number(memory_gib, "memory_gib")
            db.execute("UPDATE tasks SET min_memory_gib=? WHERE task_id=?", (memory_gib, tid))
        else:
            db.execute("UPDATE tasks SET state=? WHERE task_id=?", ("HELD" if action == "hold" else "PENDING", tid))
    event(db, "operator_override", {"action": action, "tasks": task_ids, "reason": reason, "memory_gib": memory_gib})
