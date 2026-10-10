import copy
import json
import math

import pytest

from slurm_control import policy


def sample(i, *, model="A100", peak=22.01, workload="gradient:r1:280", **kwargs):
    return {"batch_id": f"b{i}", "workload_key": workload,
            "device": {"model": model}, "host_peak_gib": peak,
            "host_scope": "whole_allocation_cgroup", "status": "COMPLETED",
            "memory_gib": 32, "command_seconds": 100, **kwargs}


def ram(rows, **kwargs):
    return policy.recommend_memory(rows, workload="gradient:r1:280", **kwargs)


def test_workload_key_is_resource_identity_not_each_motif():
    spec = {"runtime_id": "checkpoint-sha", "strategy": "beam", "size_class": "160-280",
            "resource_options": {"width": 4, "thinking": "medium"}, "motif": "pose1"}
    assert policy.workload_key(spec) == policy.workload_key(dict(spec, motif="pose2", task_id="new"))
    assert policy.workload_key(spec) != policy.workload_key(dict(spec, runtime_id="different-model"))
    assert policy.workload_key(spec) != policy.workload_key(dict(spec, strategy="gradient"))
    assert policy.workload_key(spec) != policy.workload_key(dict(spec, resource_options={"width": 8}))
    with pytest.raises(ValueError):
        policy.workload_key(dict(spec, resource_options={"width": float("nan")}))


def test_default_is_additive_peak_plus_two_no_hidden_tier_or_percent():
    result = ram([sample(i) for i in range(256)], current_gib=32, eligible_models=["A100"])
    assert result["memory_gib"] == 25  # ceil(22.01 + 2), not 29/32 from 30% headroom.
    assert result["basis"] == "completed_cohort"
    assert result["cohort_allocations_used"] == 256
    json.dumps(result, allow_nan=False)


def test_fixed_cohort_updates_only_when_another_cohort_finishes():
    initial = [sample(i, peak=8) for i in range(256)]
    almost = initial + [sample(i + 256, peak=18) for i in range(255)]
    assert ram(almost, current_gib=32)["memory_gib"] == 10
    assert ram(almost + [sample(511, peak=18)], current_gib=32)["memory_gib"] == 20


def test_missing_evidence_explicit_fallback_and_configured_pilot():
    rows = [sample(i, peak=6) for i in range(3)]
    assert ram(rows)["action"] == "hold"
    assert ram(rows, current_gib=32)["memory_gib"] == 32
    assert ram(rows, policy={"adapt_every": 3}, current_gib=32)["memory_gib"] == 8
    assert ram(rows[:2], policy={"adapt_every": 1}, current_gib=32)["memory_gib"] == 32


def test_allocations_are_independent_and_conflicting_duplicates_refused():
    row = sample(1, peak=4)
    result = ram([row] * 256, current_gib=32)
    assert result["successful_independent_allocations"] == 1
    assert result["memory_gib"] == 32
    with pytest.raises(ValueError, match="Conflicting"):
        ram([row, dict(row, host_peak_gib=5)], current_gib=32)
    with pytest.raises(ValueError, match="allocation ID"):
        ram([dict(row, batch_id=None)], current_gib=32)


def test_other_workload_and_per_process_measurements_cannot_lower_request():
    rows = [sample(i, peak=1, workload="beam:r1:280") for i in range(256)]
    rows += [sample(256 + i, peak=1, host_scope="process_rss") for i in range(256)]
    result = ram(rows, current_gib=32)
    assert result["memory_gib"] == 32
    assert result["successful_independent_allocations"] == 0


def test_broad_models_use_max_and_unknown_model_keeps_explicit_fallback():
    rows = [sample(i, model="A100", peak=8) for i in range(3)]
    rows += [sample(i + 3, model="L40", peak=14) for i in range(3)]
    config = {"adapt_every": 3}
    assert ram(rows, policy=config, eligible_models=["A100", "L40"], current_gib=32)["memory_gib"] == 16
    result = ram(rows, policy=config, eligible_models=["A100", "L40", "H200"], current_gib=32)
    assert result["memory_gib"] == 32
    assert result["per_model"]["H200"]["basis"] == "explicit_fallback"
    assert ram(rows, policy=config, eligible_models=["H200"])["action"] == "hold"


def test_failed_and_oom_records_create_persistent_floor_not_low_successes():
    rows = [sample(i, peak=2) for i in range(3)]
    failed = sample(3, peak=3, memory_gib=20, status="FAILED")
    assert ram(rows + [failed], current_gib=32, policy={"adapt_every": 3})["memory_gib"] == 5
    oom = sample(4, peak=31, memory_gib=32, status="OUT_OF_MEMORY")
    result = ram(rows + [oom], current_gib=32, policy={"adapt_every": 3})
    assert result["memory_gib"] == 36
    # Re-evaluating the same OOM does not add another four GiB every tick.
    assert ram(rows + [oom], current_gib=36, policy={"adapt_every": 3})["memory_gib"] == 36
    assert ram(rows + [dict(oom, memory_gib=None)], current_gib=32)["action"] == "hold"


def test_unrelated_failure_does_not_preserve_large_grant_after_clean_cohort():
    rows = [sample(i, peak=23) for i in range(256)]
    failed = sample(256, peak=20, memory_gib=32, status="FAILED", reason="command_failed")
    result = ram(rows + [failed], current_gib=32)
    assert result["memory_gib"] == 25
    assert result["successful_independent_allocations"] == 256
    assert result["cohort_allocations_used"] == 256
    # Failure cannot justify a lower request, but a genuinely higher observed
    # peak still protects the request even outside the completed cohort.
    higher = dict(failed, host_peak_gib=28.2)
    assert ram(rows + [higher], current_gib=32)["memory_gib"] == 31
    assert ram([failed], current_gib=32)["memory_gib"] == 32


def test_device_oom_does_not_raise_host_ram_and_memory_ceiling_holds():
    row = sample(1, status="FAILED", oom_kind="device", peak=60, memory_gib=64)
    assert ram([row], current_gib=16)["memory_gib"] == 16
    result = ram([sample(2, status="OUT_OF_MEMORY", peak=126, memory_gib=128)], current_gib=128)
    assert result["action"] == "hold"
    assert result["required_memory_gib"] == 132


def test_confirmed_host_oom_without_peak_scope_still_raises_floor():
    row = sample(1, status="OUT_OF_MEMORY", host_scope="unavailable", host_peak_gib=None, memory_gib=32)
    assert ram([row], current_gib=32)["memory_gib"] == 36


def test_cached_completion_is_never_new_runtime_evidence():
    rows = [sample(i, command_seconds=None, tasks=[
        {"status": "COMPLETE", "elapsed_seconds": 0, "skipped": True},
        {"status": "COMPLETE", "elapsed_seconds": 1, "reused": True},
        {"status": "COMPLETE", "elapsed_seconds": 100},
    ]) for i in range(3)]
    result = policy.estimate_runtime(rows, workload="gradient:r1:280", gpu_model="A100", policy={"adapt_every": 3})
    assert result["mean_seconds"] == 100
    assert result["command_count"] == 3


def test_manual_fixed_override_requires_reason_and_honors_bounds_and_floor():
    assert ram([], override={"memory_gib": 12, "reason": "Measured pilot"})["memory_gib"] == 12
    with pytest.raises(ValueError, match="reason"):
        ram([], override={"memory_gib": 12})
    with pytest.raises(ValueError, match="floor"):
        ram([], retry_floor_gib=20, override={"memory_gib": 12, "reason": "too small"})
    assert ram([], override={"memory_gib": 140, "reason": "too large"})["action"] == "hold"


@pytest.mark.parametrize("change", [{"memory_margin_gib": -1}, {"round_gib": 0}, {"adapt_every": 0},
                                   {"memory_gib": math.inf}, {"min_memory_gib": 129}])
def test_invalid_policy_values_refused(change):
    with pytest.raises(ValueError):
        ram([], policy=change)


def test_oom_retry_increment_and_retry_count_bounded():
    assert policy.oom_memory_request(20, kind="host", retry_number=1)["memory_gib"] == 24
    assert policy.oom_memory_request(24, kind="host", retry_number=2)["memory_gib"] == 28
    assert policy.oom_memory_request(28, kind="host", retry_number=3)["action"] == "hold"
    assert policy.oom_memory_request(126, kind="host", retry_number=1)["action"] == "hold"
    assert policy.oom_memory_request(20, kind="gpu", retry_number=1)["memory_gib"] == 20
    with pytest.raises(ValueError):
        policy.oom_memory_request(20, kind="unknown", retry_number=1)


def test_runtime_uses_actual_device_completed_commands_not_full_planned_batch_count():
    rows = [sample(i, command_seconds=None, tasks=[
        {"status": "COMPLETE", "elapsed_seconds": duration},
        {"status": "UNSTARTED", "elapsed_seconds": 1},
        {"status": "FAILED", "elapsed_seconds": 1}]) for i, duration in enumerate([100, 200, 300])]
    rows += [sample(5, model="L40", command_seconds=1), sample(6, workload="beam:r1:280", command_seconds=1)]
    result = policy.estimate_runtime(rows, workload="gradient:r1:280", gpu_model="A100", policy={"adapt_every": 3})
    assert result["mean_seconds"] == 200
    assert result["p90_seconds"] == 300
    assert result["planning_seconds"] == 250
    assert result["next_command_seconds"] == 375
    assert result["independent_allocations"] == 3
    assert result["command_count"] == 3
    assert policy.estimate_runtime(rows, workload="gradient:r1:280", gpu_model="H200")["action"] == "hold"


def test_runtime_cohort_holds_until_threshold_and_can_use_explicit_fallback():
    rows = [sample(i, command_seconds=100) for i in range(3)]
    result = policy.estimate_runtime(rows, workload="gradient:r1:280", gpu_model="A100", fallback=900)
    assert result["basis"] == "explicit_fallback"
    assert result["planning_seconds"] == 1125
    assert result["cohort_allocations_used"] == 0
    assert policy.estimate_runtime(rows, workload="gradient:r1:280", gpu_model="A100")["action"] == "hold"


def test_runtime_completed_commands_in_partial_job_retained_but_job_not_replicated():
    row = sample(1, status="TIMEOUT", command_seconds=None,
                 tasks=[{"status": "COMPLETE", "elapsed_seconds": 100}] * 10)
    result = policy.estimate_runtime([row] * 3, workload="gradient:r1:280", gpu_model="A100",
                                     policy={"adapt_every": 1})
    assert result["action"] == "hold"
    assert result["independent_allocations"] == 1
    assert result["command_count"] == 10


def test_fixed_homogeneous_batch_surplus_does_not_extend_walltime_or_mutate_input():
    timing = policy.estimate_runtime([], workload="gradient:r1:280", gpu_model="A100", fallback=100)
    commands = [{"task_id": str(i), "workload_key": "gradient:r1:280", "argv": ["true"]} for i in range(20)]
    original = copy.deepcopy(commands)
    result = policy.plan_batch(commands, timing, walltime_seconds=1240)
    assert len(result["tasks"]) == 9  # 1000 useful seconds * 1.2 / 125 planning seconds.
    assert result["walltime_seconds"] == 1240
    assert result["deferred_task_ids"] == [str(i) for i in range(9, 20)]
    assert commands == original
    with pytest.raises(ValueError, match="homogeneous"):
        policy.plan_batch(commands + [{"task_id": "b", "workload_key": "beam:r1:280"}], timing, walltime_seconds=1240)
    with pytest.raises(ValueError, match="unique"):
        policy.plan_batch(commands + commands, timing, walltime_seconds=1240)
    assert policy.plan_batch(commands, timing, walltime_seconds=300)["action"] == "hold"


def test_policy_never_mutates_input_evidence():
    rows = [sample(i) for i in range(256)]
    original = copy.deepcopy(rows)
    ram(rows, current_gib=32)
    assert rows == original


def test_controller_memory_bridge_uses_broad_models_and_task_retry_floor(tmp_path):
    from slurm_control import ledger
    from slurm_control.controller import memory_for
    with ledger.locked_ledger(tmp_path) as db:
        for row in [sample(i, peak=8) for i in range(3)]:
            db.execute("INSERT INTO samples VALUES(?,?)", (row["batch_id"], json.dumps(row)))
        settings = {"memory_gib": 32, "adapt_every": 3}
        chosen, receipt = memory_for(db, "gradient:r1:280", settings,
                                     [{"min_memory_gib": 14}], eligible_models=["A100"])
        assert chosen == 14
        assert receipt["retry_floor_gib"] == 14
        assert memory_for(db, "gradient:r1:280", settings, [], eligible_models=["A100", "L40"])[0] == 32
        with pytest.raises(ValueError, match="floor"):
            memory_for(db, "gradient:r1:280", dict(settings, override_memory_gib=8, override_reason="unsafe"),
                       [{"min_memory_gib": 14}], eligible_models=["A100"])
