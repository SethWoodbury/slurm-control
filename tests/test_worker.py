import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from slurm_control.common import atomic_json, digest, read_json, task_digest
from slurm_control.worker import HostMemory, detect_device, exclusive, model_name, run_batch


@pytest.fixture
def root(tmp_path):
    return tmp_path


def task(root, name="t1", code="print('hello')", **overrides):
    return dict({"task_id": name, "argv": [sys.executable, "-c", code],
                 "cwd": str(root), "env": {}, "workload_key": "cpu:test:v1",
                 "runtime_estimates": {"cpu": 0.05}, "fallback_seconds": 0.05}, **overrides)


def batch(root, tasks=None, name="b1", **overrides):
    value = dict({"batch_id": name, "tasks": tasks or [task(root)], "gpus": 0,
                  "allowed_models": ["cpu"], "walltime_seconds": 30,
                  "reserve_seconds": 0.2}, **overrides)
    path = root / (name + ".json")
    atomic_json(path, value)
    return path


def test_real_command_receipt_logs_and_no_sqlite(root):
    path = batch(root, [task(root, code="import os; from pathlib import Path; Path(os.environ['SLURM_CONTROL_ATTEMPT_DIR'],'output.txt').write_text('saved')")])
    result = run_batch(path, root / "state")
    row = result["tasks"][0]
    assert result["reason"] == "completed" and row["status"] == "COMPLETE"
    assert row["batch_digest"] == digest(read_json(path))
    assert Path(row["attempt_dir"], "output.txt").read_text() == "saved"
    assert read_json(root / "state/receipts/b1/attempt000001.json") == result
    assert not list(root.rglob("*.sqlite*"))


def test_partial_retry_preserves_success_and_science_ignores_new_estimates(root):
    first = task(root, "first", "from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')")
    second = task(root, "second", "from pathlib import Path; import sys; p=Path('once'); existed=p.exists(); p.write_text('partial'); sys.exit(0 if existed else 2)")
    original = run_batch(batch(root, [first, second]), root / "state")
    assert [t["status"] for t in original["tasks"]] == ["COMPLETE", "FAILED"]
    changed = dict(first, runtime_estimates={"cpu": 0.06}, fallback_seconds=0.2)
    assert task_digest(changed) == task_digest(first)
    retry = run_batch(batch(root, [changed, second], name="b2"), root / "state")
    assert retry["reason"] == "completed"
    assert retry["tasks"][0]["skipped"] and retry["tasks"][0]["elapsed_seconds"] == 0
    assert retry["tasks"][1]["attempt_dir"].endswith("attempt000002")
    assert (root / "counter").read_text() == "1"
    assert Path(original["tasks"][1]["attempt_dir"], "receipt.json").exists()


def test_completion_mismatch_and_corruption_do_not_skip_or_run(root):
    run_batch(batch(root), root / "state")
    changed = run_batch(batch(root, [task(root, code="print('different')")], name="b2"), root / "state")
    assert changed["reason"] == "worker_error"
    assert changed["tasks"][0]["status"] == "UNSTARTED"
    index = read_json(root / "state/completions/t1.json")
    receipt = Path(index["receipt_path"])
    atomic_json(receipt, dict(read_json(receipt), returncode=99))
    corrupt = run_batch(batch(root, name="b3"), root / "state")
    assert corrupt["reason"] == "worker_error"
    assert "immutable receipt" in corrupt["error"]


def test_manifest_identity_and_duplicate_task_rejected(root):
    path = batch(root)
    run_batch(path, root / "state")
    atomic_json(path, dict(read_json(path), reserve_seconds=1))
    with pytest.raises(ValueError, match="immutable manifest"):
        run_batch(path, root / "state")
    with pytest.raises(ValueError, match="Duplicate task"):
        run_batch(batch(root, [task(root), task(root)], name="dup"), root / "state")


def test_batch_and_task_locks_prevent_duplicate_execution(root):
    path = batch(root)
    with exclusive(root / "state/locks/batch-b1.lock"):
        with pytest.raises(BlockingIOError):
            run_batch(path, root / "state")
    with exclusive(root / "state/locks/task-t1.lock"):
        result = run_batch(path, root / "state")
    assert result["reason"] == "worker_error"
    assert result["tasks"][0]["status"] == "UNSTARTED"
    assert not (root / "state/attempts/t1").exists()


def test_actual_device_estimate_controls_tail_and_unknown_device_rejected(root):
    bound = lambda: {"model": "L40", "uuid": "GPU-test", "source": "qualification"}
    tasks = [task(root, runtime_estimates={"A100": 0.01, "L40": 40})]
    result = run_batch(batch(root, tasks, allowed_models=["L40", "A100"], gpus=1), root / "state", binding=bound)
    assert result["reason"] == "insufficient_time"
    assert result["tasks"][0]["status"] == "UNSTARTED"
    rejected = run_batch(batch(root, name="b2", allowed_models=["A100"], gpus=1), root / "state", binding=bound)
    assert rejected["reason"] == "worker_error"
    assert "outside allowed_models" in rejected["error"]


def test_missing_runtime_estimate_does_not_guess(root):
    item = task(root, runtime_estimates={}, fallback_seconds=None)
    result = run_batch(batch(root, [item]), root / "state")
    assert result["reason"] == "missing_runtime_estimate"
    assert result["tasks"][0]["status"] == "UNSTARTED"


def test_nonzero_stops_batch_and_preserves_logs(root):
    result = run_batch(batch(root, [task(root, code="import sys; print('bad',file=sys.stderr);sys.exit(3)"), task(root, "t2")]), root / "state")
    assert result["reason"] == "command_failed"
    assert [t["status"] for t in result["tasks"]] == ["FAILED", "UNSTARTED"]
    assert Path(result["tasks"][0]["attempt_dir"], "stderr.log").read_text() == "bad\n"


def test_device_oom_is_distinct_from_host_oom(root):
    result = run_batch(batch(root, [task(root, code="import sys; print('CUDA out of memory',file=sys.stderr);sys.exit(1)")]), root / "state")
    assert result["tasks"][0]["reason"] == "gpu_oom"
    killed = run_batch(batch(root, [task(root, "kill", "import os,signal; os.kill(os.getpid(),signal.SIGKILL)")], name="b2"), root / "state")
    assert killed["tasks"][0]["reason"] == "command_failed"


def test_cgroup_peak_scope_components_and_oom_delta(root):
    cg = root / "cgroup"
    cg.mkdir()
    (cg / "memory.current").write_text(str(2 * 1024 ** 3))
    (cg / "memory.peak").write_text(str(3 * 1024 ** 3))
    (cg / "memory.stat").write_text(f"anon {1024 ** 3}\nfile {1024 ** 3}\n")
    (cg / "memory.events").write_text("oom_kill 0\n")
    memory = HostMemory(cg, "whole_allocation_cgroup")
    code = f"from pathlib import Path; import sys; Path({str(cg / 'memory.events')!r}).write_text('oom_kill 1\\n'); sys.exit(137)"
    result = run_batch(batch(root, [task(root, code=code)]), root / "state", memory=memory)
    assert result["host_peak_gib"] == 3
    assert result["host_scope"] == "whole_allocation_cgroup"
    assert result["host_measurement"]["current_components"] == {"anon_gib": 1, "file_gib": 1}
    assert result["tasks"][0]["reason"] == "host_oom"
    (cg / "memory.peak").unlink()
    assert HostMemory(cg, "whole_allocation_cgroup").receipt()["host_scope"] == "sampled_process_cgroup"


def test_command_deadline_preserves_partial_and_stops(root):
    code = "from pathlib import Path; import time; Path('partial').write_text('before timeout'); time.sleep(20)"
    result = run_batch(batch(root, [task(root, code=code)], walltime_seconds=0.4, reserve_seconds=0), root / "state")
    assert result["reason"] == "time_limit"
    assert result["tasks"][0]["status"] == "FAILED"
    assert (root / "partial").read_text() == "before timeout"
    assert result["elapsed_seconds"] < 7


def test_real_usr1_terminates_process_group_and_writes_final_receipt(root):
    code = "import os,time,subprocess,sys; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path('child_pid').write_text(str(child.pid)); time.sleep(60)"
    path = batch(root, [task(root, code=code), task(root, "tail")], walltime_seconds=120)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    proc = subprocess.Popen([sys.executable, "-m", "slurm_control.worker", "--batch", str(path), "--state", str(root / "state")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not (root / "child_pid").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (root / "child_pid").exists()
        child = int((root / "child_pid").read_text())
        proc.send_signal(signal.SIGUSR1)
        assert proc.wait(timeout=15) == 1
        result = read_json(root / "state/receipts/b1/attempt000001.json")
        assert result["reason"] == "preempted"
        assert [r["status"] for r in result["tasks"]] == ["FAILED", "UNSTARTED"]
        child_stat = Path(f"/proc/{child}/stat")
        assert not child_stat.exists() or child_stat.read_text().split()[2] == "Z"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


def test_worker_locks_not_inherited_by_real_child(root):
    code = "import os; from pathlib import Path; links=[os.readlink(p) for p in Path('/proc/self/fd').glob('*') if p.exists()]; assert not any('batch-b1.lock' in x or 'task-t1.lock' in x for x in links),links"
    assert run_batch(batch(root, [task(root, code=code)]), root / "state")["reason"] == "completed"


def test_changed_input_manifest_stops_later_commands(root):
    code = "import json; from pathlib import Path; p=Path('b1.json'); b=json.loads(p.read_text()); b['reserve_seconds']=1;p.write_text(json.dumps(b))"
    result = run_batch(batch(root, [task(root, code=code), task(root, "t2")]), root / "state")
    assert result["reason"] == "worker_error"
    assert [r["status"] for r in result["tasks"]] == ["COMPLETE", "UNSTARTED"]


def test_model_names_do_not_confuse_l40s_and_l40():
    assert model_name("NVIDIA L40S") == "L40S"
    assert model_name("NVIDIA L40") == "L40"
    assert model_name("NVIDIA RTX 4000 Ada Generation") == "4000Ada"
    assert model_name("NVIDIA A100-SXM4-80GB") == "A100"
    assert model_name("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition") == "B6000Q"
    assert model_name("NVIDIA RTX PRO 6000 Blackwell Workstation Edition") == "B6000"


def test_task_cannot_change_detected_binding(root):
    with pytest.raises(ValueError, match="device/job binding"):
        run_batch(batch(root, [task(root, env={"CUDA_VISIBLE_DEVICES": "1"})]), root / "state")


def test_cuda_binding_uses_visible_ordinal_and_rejects_ambiguity(monkeypatch):
    import slurm_control.worker as worker
    calls = []

    class Driver:
        count = 1
        def cuInit(self, flags):
            return 0
        def cuDeviceGetCount(self, value):
            value._obj.value = self.count
            return 0
        def cuDeviceGet(self, device, ordinal):
            calls.append(ordinal)
            device._obj.value = 0
            return 0
        def cuDeviceGetName(self, buf, length, device):
            buf.value = b"NVIDIA L40"
            return 0
        def cuDeviceGetUuid(self, buf, device):
            for i in range(16):
                buf._obj[i] = i
            return 0
        def cuDeviceTotalMem_v2(self, buf, device):
            buf._obj.value = 48 * 1024 ** 3
            return 0

    driver = Driver()
    monkeypatch.setattr(worker.ctypes, "CDLL", lambda name: driver)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_GPUS", "5")
    actual = detect_device()
    assert actual["model"] == "L40" and actual["memory_gib"] == 48 and calls == [0]
    assert actual["uuid"] == "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    driver.count = 2
    with pytest.raises(RuntimeError, match="found 2"):
        detect_device()
    driver.count = 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")
    with pytest.raises(RuntimeError, match="disagrees"):
        detect_device()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError, match="assignment"):
        detect_device()
