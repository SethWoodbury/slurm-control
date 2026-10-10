"""Pure, auditable resource decisions. Slurm, not this module, places jobs.

Measurements must already have passed receipt/identity validation by the caller.
The functions neither read state nor submit work. All returned values are JSON
serializable; ``action='hold'`` means no defensible request could be made.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics


WHOLE_ALLOCATION_SCOPES = {"whole_allocation_cgroup", "allocation_cgroup"}


def _number(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def workload_key(spec):
    """Identify comparable jobs without splitting evidence by motif/task identity.

    Caller chooses a useful size class (e.g. 160-280 residues) and explicitly
    supplies *every* resource-affecting option, such as beam width or replica
    count. Runtime/checkpoint identity and strategy are mandatory.
    """
    identity = {key: spec[key] for key in ("runtime_id", "strategy", "size_class")}
    if any(not isinstance(value, str) or not value.strip() for value in identity.values()):
        raise ValueError("runtime_id, strategy and size_class must be nonempty strings")
    options = spec.get("resource_options", {})
    if not isinstance(options, dict):
        raise ValueError("resource_options must be a mapping")
    identity["resource_options"] = options
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "workload-" + hashlib.sha256(raw.encode()).hexdigest()


def device_model(sample):
    """Accept explicit model strings or the worker's actual-device receipt."""
    device = sample.get("device")
    if isinstance(device, str):
        return device
    if isinstance(device, dict):
        return device.get("model") or device.get("name")
    return sample.get("gpu_model")


def _independent(samples, key, models=None):
    """Deduplicate allocations, refusing contradictory receipts for one job."""
    chosen = {}
    for sample in samples:
        if sample.get("workload_key") != key:
            continue
        if models is not None and device_model(sample) not in models:
            continue
        allocation = sample.get("batch_id") or sample.get("allocation_id")
        if not isinstance(allocation, str) or not allocation:
            raise ValueError("Comparable measurements require an independent allocation ID")
        encoded = json.dumps(sample, sort_keys=True, allow_nan=False)
        if allocation in chosen and chosen[allocation][0] != encoded:
            raise ValueError(f"Conflicting measurements for allocation {allocation}")
        chosen[allocation] = (encoded, sample)
    # Preserve caller's stable chronological order; the ledger uses insertion order.
    return [entry[1] for entry in chosen.values()]


def _ceil_gib(value, step):
    return round(math.ceil(value / step) * step, 8)


def recommend_memory(samples, *, workload, policy=None, current_gib=None,
                     eligible_models=None, retry_floor_gib=0, override=None):
    """Recommend host RAM using comparable whole-allocation peaks + 2 GiB.

    The default adaptation cohort is 256 independent successful allocations.
    A deliberately smaller pilot is configured with ``adapt_every``; a device
    needs ``min_independent_allocations`` (default 3) before its evidence can
    lower the fallback. Broad eligibility takes the maximum recommendation over
    eligible devices. Unknown devices retain an explicit current/fallback.
    Host OOMs establish a persistent additive retry floor immediately. Failed
    measurements never provide evidence for a lower request.
    """
    policy = dict(policy or {})
    margin = _number(policy.get("memory_margin_gib", 2), "memory_margin_gib", zero=True)
    step = _number(policy.get("round_gib", 1), "round_gib")
    minimum = _number(policy.get("min_memory_gib", 1), "min_memory_gib")
    maximum = _number(policy.get("max_memory_gib", 128), "max_memory_gib")
    if minimum > maximum:
        raise ValueError("Minimum host memory exceeds maximum")
    cohort = _integer(policy.get("adapt_every", 256), "adapt_every")
    independent_min = _integer(policy.get("min_independent_allocations", 3), "min_independent_allocations")
    oom_step = _number(policy.get("oom_step_gib", 4), "oom_step_gib")
    floor = _number(retry_floor_gib, "retry_floor_gib", zero=True)
    fallback = current_gib if current_gib is not None else policy.get("memory_gib")
    if fallback is not None:
        fallback = _number(fallback, "current/fallback memory")
    models = sorted(set(eligible_models)) if eligible_models is not None else None
    if models == []:
        raise ValueError("eligible_models cannot be empty")
    rows = _independent(samples, workload, models)
    accepted = []
    rejected = []
    failure_floor = 0.0
    for sample in rows:
        identifier = sample.get("batch_id") or sample.get("allocation_id")
        # Confirmed scheduler/cgroup OOM establishes a floor even when the
        # worker died before collecting a usable whole-allocation peak.
        status = sample.get("status")
        kind = sample.get("oom_kind") or sample.get("reason")
        device_oom = kind in {"gpu_oom", "GPU_OOM", "device", "device_oom"}
        host_oom = not device_oom and (status in {"OUT_OF_MEMORY", "HOST_OOM"} or kind in {"host", "host_oom", "HOST_OOM"})
        if host_oom:
            if sample.get("memory_gib") is None:
                return {"action": "hold", "reason": "host_oom_missing_granted_memory", "memory_gib": None}
            failure_floor = max(failure_floor, _number(sample["memory_gib"], "measured allocation memory_gib") + oom_step)
        if sample.get("host_scope") not in WHOLE_ALLOCATION_SCOPES:
            rejected.append({"allocation_id": identifier, "reason": "not_whole_allocation"})
            continue
        peak = sample.get("host_peak_gib")
        if peak is not None:
            peak = _number(peak, "host_peak_gib")
        if status == "COMPLETED" and peak is not None:
            accepted.append(sample)
            continue
        # Device OOM is a VRAM eligibility problem, never a host RAM escalation.
        if device_oom:
            rejected.append({"allocation_id": identifier, "reason": "device_oom"})
            continue
        requested = sample.get("memory_gib")
        if requested is not None:
            requested = _number(requested, "measured allocation memory_gib")
        if host_oom:
            if requested is None:
                return {"action": "hold", "reason": "host_oom_missing_granted_memory", "memory_gib": None}
            failure_floor = max(failure_floor, requested + oom_step)
        # An application failure is not evidence for lowering, but its granted
        # RAM alone is not a memory requirement. Only a measured peak or a
        # confirmed host OOM can establish an additional safety floor.
        if peak is not None:
            failure_floor = max(failure_floor, peak + margin)
        rejected.append({"allocation_id": identifier, "reason": "not_successful"})
    floor = max(floor, failure_floor, minimum)
    usable = len(accepted) // cohort * cohort
    evidence = accepted[:usable]
    if models is None:
        models = sorted({device_model(row) or "unknown" for row in rows}) or ["unknown"]
    recommendations = {}
    missing = []
    for model in models:
        comparable = [r for r in evidence if (device_model(r) or "unknown") == model]
        if len(comparable) >= independent_min:
            recommendations[model] = {
                "basis": "completed_cohort", "independent_allocations": len(comparable),
                "peak_gib": max(r["host_peak_gib"] for r in comparable),
                "memory_gib": _ceil_gib(max(r["host_peak_gib"] for r in comparable) + margin, step),
            }
        elif fallback is not None:
            recommendations[model] = {"basis": "explicit_fallback", "independent_allocations": len(comparable),
                                      "memory_gib": fallback}
        else:
            missing.append(model)
    receipt = {"action": "ready", "workload_key": workload, "memory_gib": None,
               "margin_gib": margin, "round_gib": step, "cohort_size": cohort,
               "successful_independent_allocations": len(accepted), "cohort_allocations_used": usable,
               "minimum_independent_allocations": independent_min, "retry_floor_gib": floor,
               "per_model": recommendations, "missing_models": missing, "excluded": rejected,
               "evidence_allocation_ids": [r.get("batch_id") or r.get("allocation_id") for r in evidence]}
    if override is not None:
        if not isinstance(override.get("reason"), str) or not override["reason"].strip():
            raise ValueError("A fixed memory override requires a reason")
        requested = _number(override.get("memory_gib"), "override memory_gib")
        if requested < floor:
            raise ValueError("Fixed memory override is below the retry/failure floor")
        receipt["override"] = dict(override)
        receipt["basis"] = "manual_override"
    elif missing:
        return dict(receipt, action="hold", reason="missing_explicit_memory_fallback")
    else:
        requested = max(floor, *(r["memory_gib"] for r in recommendations.values()))
        receipt["basis"] = "completed_cohort" if any(r["basis"] == "completed_cohort" for r in recommendations.values()) else "explicit_fallback"
    requested = _ceil_gib(requested, step)
    if requested > maximum:
        return dict(receipt, action="hold", reason="memory_ceiling_exceeded", required_memory_gib=requested,
                    max_memory_gib=maximum)
    return dict(receipt, memory_gib=requested)


def oom_memory_request(requested_gib, *, kind, retry_number, policy=None):
    """A bounded per-attempt host OOM increase; GPU OOM never changes host RAM."""
    policy = dict(policy or {})
    requested = _number(requested_gib, "requested_gib")
    attempt = _integer(retry_number, "retry_number")
    maximum_attempts = _integer(policy.get("max_oom_retries", 2), "max_oom_retries")
    maximum = _number(policy.get("max_memory_gib", 128), "max_memory_gib")
    if kind in {"device", "gpu", "gpu_oom"}:
        return {"action": "hold", "reason": "requires_larger_device_vram", "memory_gib": requested}
    if kind not in {"host", "host_oom"}:
        raise ValueError("OOM kind must distinguish host RAM from device VRAM")
    target = _ceil_gib(requested + _number(policy.get("oom_step_gib", 4), "oom_step_gib"),
                       _number(policy.get("round_gib", 1), "round_gib"))
    if attempt > maximum_attempts or target > maximum:
        return {"action": "hold", "reason": "oom_retry_bound", "memory_gib": requested,
                "required_memory_gib": target, "retry_number": attempt}
    return {"action": "retry", "reason": "bounded_host_oom_increase", "memory_gib": target,
            "retry_number": attempt}


def estimate_runtime(samples, *, workload, gpu_model, policy=None, fallback=None):
    """Mean for planning and p90 for next-command admission on the actual GPU.

    Successful command durations inside independently identified allocations are
    used; allocation walltime is never divided by planned command count (which
    may include an unstarted tail). A scalar ``command_seconds`` is also allowed
    for a one-command allocation. Unknown models need an explicit fallback.
    """
    policy = dict(policy or {})
    required = _integer(policy.get("min_independent_allocations", 3), "min_independent_allocations")
    cohort = _integer(policy.get("adapt_every", 256), "adapt_every")
    planning_margin = _number(policy.get("time_margin_fraction", 0.25), "time_margin_fraction", zero=True)
    deadline_margin = _number(policy.get("deadline_margin_fraction", 0.25), "deadline_margin_fraction", zero=True)
    rows = _independent(samples, workload, {gpu_model})
    measured = []
    for row in rows:
        durations = []
        if row.get("status") == "COMPLETED" and row.get("command_seconds") is not None:
            durations.append(_number(row["command_seconds"], "command_seconds"))
        for command in row.get("tasks", []):
            if command.get("status") == "COMPLETE" and not command.get("skipped") and not command.get("reused") and command.get("elapsed_seconds") is not None:
                durations.append(_number(command["elapsed_seconds"], "command elapsed_seconds"))
        if durations:
            measured.append((row.get("batch_id") or row.get("allocation_id"), durations))
    usable = len(measured) // cohort * cohort
    independent = [identifier for identifier, _ in measured[:usable]]
    values = [duration for _, durations in measured[:usable] for duration in durations]
    receipt = {"action": "ready", "workload_key": workload, "gpu_model": gpu_model,
               "independent_allocations": len(measured), "cohort_allocations_used": usable,
               "cohort_size": cohort, "command_count": len(values),
               "evidence_allocation_ids": independent}
    if len(independent) >= required:
        mean = statistics.fmean(values)
        p90 = sorted(values)[max(0, math.ceil(0.9 * len(values)) - 1)]
        basis = "measured_actual_device"
    elif fallback is not None:
        if isinstance(fallback, dict):
            mean = _number(fallback.get("mean_seconds"), "fallback mean_seconds")
            p90 = _number(fallback.get("p90_seconds", mean), "fallback p90_seconds")
        else:
            mean = p90 = _number(fallback, "fallback seconds")
        basis = "explicit_fallback"
    else:
        return dict(receipt, action="hold", reason="insufficient_actual_device_evidence")
    return dict(receipt, basis=basis, mean_seconds=mean, p90_seconds=p90,
                planning_seconds=mean * (1 + planning_margin),
                next_command_seconds=max(mean, p90) * (1 + deadline_margin),
                planning_margin_fraction=planning_margin, deadline_margin_fraction=deadline_margin)


def plan_batch(commands, timing, *, walltime_seconds, policy=None):
    """Select a fixed homogeneous batch plus optional modest surplus.

    ``timing`` is a ready runtime recommendation for this workload and a chosen
    planning device. For broad pools callers supply the explicitly chosen
    reference-device plan; workers must re-evaluate on the actual GPU. Surplus
    changes assigned command count only, never the requested allocation time.
    """
    policy = dict(policy or {})
    wall = _number(walltime_seconds, "walltime_seconds")
    startup = _number(policy.get("startup_seconds", 120), "startup_seconds", zero=True)
    reserve = _number(policy.get("reserve_seconds", 120), "reserve_seconds", zero=True)
    surplus = _number(policy.get("backlog_factor", 1.2), "backlog_factor")
    maximum_surplus = _number(policy.get("max_backlog_factor", 1.5), "max_backlog_factor")
    if surplus < 1 or surplus > maximum_surplus:
        raise ValueError("backlog_factor must be between 1 and max_backlog_factor")
    if wall <= startup + reserve:
        raise ValueError("Walltime must cover startup and shutdown reserve")
    if timing.get("action") != "ready":
        return {"action": "hold", "reason": "runtime_evidence_not_ready", "tasks": []}
    duration = _number(timing.get("planning_seconds"), "planning_seconds")
    commands = list(commands)
    keys = {command.get("workload_key") for command in commands}
    ids = [command.get("task_id") for command in commands]
    if len(keys) > 1 or (keys and next(iter(keys)) != timing.get("workload_key")):
        raise ValueError("Batch must be homogeneous and match its timing workload")
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Batch task IDs must be nonempty and unique")
    max_commands = _integer(policy.get("max_commands", 256), "max_commands")
    # Never enqueue a command that cannot individually fit an empty allocation.
    if _number(timing.get("next_command_seconds"), "next_command_seconds") > wall - startup - reserve:
        return {"action": "hold", "reason": "individual_command_exceeds_walltime", "tasks": []}
    count = min(len(commands), max_commands, math.floor((wall - startup - reserve) * surplus / duration))
    return {"action": "ready" if count else "hold", "reason": "fixed_homogeneous_batch" if count else "no_fitting_work",
            "tasks": commands[:count], "deferred_task_ids": ids[count:], "walltime_seconds": wall,
            "estimated_command_seconds": count * duration, "startup_seconds": startup,
            "reserve_seconds": reserve, "backlog_factor": surplus, "planning_gpu_model": timing.get("gpu_model")}
