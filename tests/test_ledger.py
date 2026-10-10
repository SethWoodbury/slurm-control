import json
import pytest

from slurm_control import ledger
from slurm_control.common import digest, task_digest
from slurm_control.controller import submit_planned


def tasks():
    return [{"task_id": str(i), "argv": ["python", "-c", "print('done')"], "workload_key": "gradient:v1:280"} for i in range(2)]


def batch():
    return {"batch_id": "b1", "tasks": tasks(), "walltime_seconds": 3600, "memory_gib": 25, "gpus": 1}


def test_completed_tasks_cannot_be_reregistered_or_retried(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, tasks(), 10)
        ledger.reserve_batch(db, batch())
        ledger.mark_intent(db, "b1")
        ledger.mark_submitted(db, "b1", "123")
        receipt = {"batch_id": "b1", "batch_digest": digest(batch()), "tasks": [
            {"task_id": "0", "task_digest": task_digest(tasks()[0]), "status": "COMPLETE"},
            {"task_id": "1", "task_digest": task_digest(tasks()[1]), "status": "UNSTARTED"}]}
        ledger.finalize(db, "b1", receipt, {"job_id": "123", "terminal": True, "state": "TIMEOUT", "elapsed_seconds": 3600})
        ledger.register(db, tasks(), 10)
        assert ledger.status(db)["tasks"] == {"COMPLETE": 1, "PENDING": 1}
        assert ledger.status(db)["actual_gpu_hours"] == 1
        assert not ledger.finalize(db, "b1", receipt, {})
        with pytest.raises(ValueError):
            ledger.reserve_batch(db, dict(batch(), batch_id="b2"))
        changed = tasks()
        changed[0]["argv"] = ["false"]
        with pytest.raises(ValueError, match="changed"):
            ledger.register(db, changed, 10)


def test_lost_submission_response_is_reserved_and_cannot_resubmit(tmp_path):
    class Lost:
        def submit(self, *args):
            raise TimeoutError("Response lost after Slurm may have accepted job")
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, tasks(), 10)
        ledger.reserve_batch(db, batch())
        result = submit_planned(db, tmp_path, Lost())
        assert result[0]["status"] == "AMBIGUOUS"
        assert ledger.status(db)["reserved_gpu_hours"] == 1
        assert submit_planned(db, tmp_path, Lost()) == []
        with pytest.raises(ValueError):
            ledger.mark_intent(db, "b1")


def test_budget_reservations_include_pending_jobs(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, tasks(), 0.5)
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve_batch(db, batch())
        assert ledger.status(db)["reserved_gpu_hours"] == 0


def test_host_oom_additive_retry_and_cancel_hold(tmp_path):
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, tasks(), 10)
        ledger.reserve_batch(db, batch())
        ledger.mark_intent(db, "b1")
        ledger.mark_submitted(db, "b1", "123")
        ledger.finalize(db, "b1", None, {"job_id": "123", "terminal": True, "state": "OUT_OF_MEMORY", "elapsed_seconds": 120}, oom_step_gib=4)
        assert all(r[0] == 29 for r in db.execute("SELECT min_memory_gib FROM tasks"))
        b2 = dict(batch(), batch_id="b2", memory_gib=29)
        ledger.reserve_batch(db, b2)
        ledger.mark_intent(db, "b2")
        ledger.mark_submitted(db, "b2", "124")
        ledger.finalize(db, "b2", None, {"job_id": "124", "terminal": True, "state": "CANCELLED", "elapsed_seconds": 60})
        assert ledger.status(db)["tasks"] == {"HELD": 2}


def test_mixed_workloads_rejected(tmp_path):
    ts = tasks()
    ts[1]["workload_key"] = "beam:v1:280"
    with ledger.locked_ledger(tmp_path) as db:
        ledger.register(db, ts, 10)
        with pytest.raises(ValueError, match="mix"):
            ledger.reserve_batch(db, dict(batch(), tasks=ts))
