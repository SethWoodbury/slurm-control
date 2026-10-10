import json
from pathlib import Path
import sys

import pytest

from slurm_control import controller, ledger
from slurm_control.common import atomic_json, digest, task_digest
from slurm_control.slurm import SlurmSubmissionRejected


def test_config_expands_environment_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CONTROL_TEST_ROOT", str(tmp_path / "scratch"))
    path = tmp_path / "campaign.json"
    atomic_json(path, {"state": "${SLURM_CONTROL_TEST_ROOT}/state", "tasks_path": "tasks.json",
                      "pools": [{"id": "cpu"}], "workloads": {"cpu:v1": {}}})
    loaded = controller.load_config(path)
    assert loaded["state"] == str(tmp_path / "scratch" / "state")
    assert loaded["tasks_path"] == str(tmp_path / "tasks.json")
    monkeypatch.delenv("SLURM_CONTROL_TEST_ROOT")
    with pytest.raises(ValueError, match="Unresolved environment variable"):
        controller.load_config(path)


def task(i="t1", **kwargs):
    return {"task_id": i, "argv": [sys.executable, "-c", "print('done')"], "workload_key": "gradient:v1:280",
            "runtime_estimates": {"A100": 100, "L40": 1000}, **kwargs}


def config(**kwargs):
    return {"max_outstanding": 4, "max_admissions_per_tick": 4,
            "pools": [{"id": "broad", "partition": "gpu", "gpus": 1, "allowed_models": ["A100", "L40"],
                       "walltime_seconds": 8 * 3600, "cpus": 2}],
            "workloads": {"gradient:v1:280": {"memory_gib": 16, "fallback_seconds": 1000}}, **kwargs}


def setup_batch(db, tasks, **kwargs):
    ledger.register(db, tasks, 100)
    batch = {"batch_id": "b1", "tasks": tasks, "walltime_seconds": 3600,
             "memory_gib": 16, "gpus": 1, "cpus": 2, "allowed_models": ["A100"], "partition": "gpu", **kwargs}
    ledger.reserve_batch(db, batch)
    ledger.mark_intent(db, "b1")
    ledger.mark_submitted(db, "b1", "123")
    return batch


def test_broad_tail_walltime_fits_first_command_on_every_allowed_gpu(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, [task()], 10)
        batch, = controller.preview_batches(db, config())
        maximum = max(batch["tasks"][0]["runtime_estimates"].values())
        assert batch["walltime_seconds"] >= maximum + 240
        assert batch["walltime_seconds"] < 8 * 3600
        assert batch["gres"] == "gpu:1"


@pytest.mark.parametrize("gpus,gres", [(0, "gpu:large:1"), (1, "gpu:large:2")])
def test_planner_rejects_gpu_request_accounting_mismatch_even_without_tasks(tmp_path, gpus, gres):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, [], 10)
        cfg = config()
        cfg["pools"][0].update(gpus=gpus, gres=gres)
        with pytest.raises(ValueError):
            controller.preview_batches(db, cfg)


def test_remaining_budget_can_fund_short_tail_instead_of_stranding_it(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, [task()], 1)
        batch, = controller.preview_batches(db, config())
        assert batch["walltime_seconds"] <= 3600
        assert batch["walltime_seconds"] >= 1490


def test_individually_oversized_command_is_not_returned_unstarted_forever(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, [task(runtime_estimates={"A100": 40000, "L40": 40000})], 10)
        assert controller.preview_batches(db, config()) == []


def test_separate_workloads_never_mix_and_surplus_is_bounded(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, [task(), task("t2", workload_key="beam:v1:280")], 100)
        cfg = config()
        cfg["workloads"]["beam:v1:280"] = {"memory_gib": 20, "fallback_seconds": 1000}
        planned = controller.preview_batches(db, cfg)
        assert len(planned) == 2
        assert all(len({t["workload_key"] for t in b["tasks"]}) == 1 for b in planned)
        cfg["pools"][0]["backlog_factor"] = 10
        with pytest.raises(ValueError, match="bounds"):
            controller.preview_batches(db, cfg)


def test_proven_submission_rejection_releases_budget_but_holds_tasks(tmp_path):
    class Reject:
        def submit(self, *args):
            raise SlurmSubmissionRejected("Batch job submission failed: invalid resource request")
    with ledger.locked_ledger(tmp_path) as db:
        tasks = [task()]
        ledger.register(db, tasks, 2)
        batch = {"batch_id": "b1", "tasks": tasks, "walltime_seconds": 3600, "memory_gib": 16, "gpus": 1}
        ledger.reserve_batch(db, batch)
        result = controller.submit_planned(db, tmp_path, Reject())
        assert result[0]["status"] == "REJECTED"
        status = ledger.status(db)
        assert status["reserved_gpu_hours"] == 0
        assert status["tasks"] == {"HELD": 1}
        assert controller.submit_planned(db, tmp_path, Reject()) == []


def test_recover_completion_when_worker_dies_before_final_batch_receipt(tmp_path):
    class Terminal:
        def queue(self):
            return {}
        def accounting(self, jobs):
            return {"123": {"job_id": "123", "terminal": True, "state": "TIMEOUT", "elapsed_seconds": 3600, "token": self.token}}
    with ledger.locked_ledger(tmp_path) as db:
        tasks = [task("first"), task("inflight"), task("unknown")]
        batch = setup_batch(db, tasks)
        for current, filename, status, rc in [(tasks[0], "receipt.json", "COMPLETE", 0), (tasks[1], "start.json", "FAILED", None)]:
            report = {"batch_id": "b1", "batch_digest": digest(batch), "job_id": "123", "task_id": current["task_id"],
                      "task_digest": task_digest(current), "status": status, "returncode": rc, "elapsed_seconds": 10}
            atomic_json(tmp_path / "attempts" / current["task_id"] / "attempt000001" / filename, report)
        scheduler = Terminal()
        scheduler.token = db.execute("SELECT token FROM batches").fetchone()[0]
        assert controller.reconcile(db, tmp_path, scheduler, config()) == [{"batch_id": "b1", "state": "TIMEOUT"}]
        states = {r["task_id"]: (r["state"], r["attempts"]) for r in db.execute("SELECT * FROM tasks")}
        assert states == {"first": ("COMPLETE", 1), "inflight": ("PENDING", 1), "unknown": ("PENDING", 1)}
        assert ledger.status(db)["actual_gpu_hours"] == 1


def test_recovery_refuses_mismatched_task_digest(tmp_path):
    current = task()
    batch = {"batch_id": "b1", "tasks": [current]}
    atomic_json(tmp_path / "attempts/t1/attempt000001/receipt.json", {
        "batch_id": "b1", "batch_digest": digest(batch), "job_id": "123", "task_id": "t1", "task_digest": "wrong",
        "status": "COMPLETE", "returncode": 0})
    with pytest.raises(ValueError, match="identity"):
        controller.recover_command_receipts(tmp_path, batch, "123")


def test_worker_setup_failure_holds_instead_of_infinite_unstarted_retry(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        current = task()
        batch = setup_batch(db, [current])
        receipt = {"batch_id": "b1", "batch_digest": digest(batch), "job_id": "123", "reason": "worker_error",
                   "host_peak_gib": 1, "host_scope": "whole_allocation_cgroup", "device": {"model": "A100"},
                   "tasks": [{"task_id": "t1", "task_digest": task_digest(current), "status": "UNSTARTED", "reason": "worker_error"}]}
        ledger.finalize(db, "b1", receipt, {"job_id": "123", "terminal": True, "state": "FAILED", "elapsed_seconds": 4})
        assert ledger.status(db)["tasks"] == {"HELD": 1}
        assert json.loads(db.execute("SELECT sample FROM samples").fetchone()[0])["status"] == "FAILED"


def test_gpu_oom_does_not_raise_host_floor_and_unstarted_tail_survives(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        first, tail = task("first"), task("tail")
        batch = setup_batch(db, [first, tail])
        receipt = {"batch_id": "b1", "batch_digest": digest(batch), "reason": "gpu_oom", "tasks": [
            {"task_id": "first", "task_digest": task_digest(first), "status": "FAILED", "reason": "gpu_oom"},
            {"task_id": "tail", "task_digest": task_digest(tail), "status": "UNSTARTED", "reason": "gpu_oom"}]}
        ledger.finalize(db, "b1", receipt, {"job_id": "123", "terminal": True, "state": "FAILED", "elapsed_seconds": 4})
        rows = {r["task_id"]: dict(r) for r in db.execute("SELECT * FROM tasks")}
        assert rows["first"]["state"] == "HELD"
        assert rows["first"]["min_memory_gib"] == 0
        assert rows["tail"]["state"] == "PENDING"
        assert rows["tail"]["attempts"] == 0


def test_freeze_preserves_source_archive_after_checkout_changes(tmp_path):
    spec = {"batch_id": "b1", "tasks": [task()], "walltime_seconds": 3600, "memory_gib": 16}
    script, root = controller.write_batch(tmp_path, spec)
    source = json.loads((root / "source.json").read_text())
    assert Path(source["path"]).is_file()
    assert source["sha256"] in script.read_text()
    # The written script names its immutable archive, never the mutable src dir.
    assert "/src" not in script.read_text()
    with pytest.raises(ValueError, match="Immutable"):
        controller.write_batch(tmp_path, dict(spec, memory_gib=20))
