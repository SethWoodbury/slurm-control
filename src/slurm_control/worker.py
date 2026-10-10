"""Run immutable commands sequentially; publish receipts, never edit a ledger.

An interrupted command starts a new attempt. Only a verified successful command
with the same scientific digest can be skipped. Commands may use
SLURM_CONTROL_ATTEMPT_DIR for outputs that must survive retries separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

from .common import atomic_json, digest, positive_number, read_json, task_digest, utcnow


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise ValueError("Identifiers must be safe nonempty path components")
    return value


@contextmanager
def exclusive(path):
    """Nonblocking lock; its descriptor is never inherited by commands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        os.set_inheritable(stream.fileno(), False)
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_batch(batch):
    _identifier(batch["batch_id"])
    positive_number(batch["walltime_seconds"], "walltime_seconds")
    reserve = batch.get("reserve_seconds", 120)
    if isinstance(reserve, bool) or not isinstance(reserve, (float, int)) or not math.isfinite(reserve) or not 0 <= reserve < batch["walltime_seconds"]:
        raise ValueError("reserve_seconds must be finite, nonnegative and less than walltime")
    models = batch.get("allowed_models")
    if not isinstance(models, list) or not models or any(not isinstance(x, str) or not x for x in models):
        raise ValueError("An explicit allowed_models list is required")
    tasks = batch.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A nonempty task list is required")
    seen = set()
    for task in tasks:
        tid = _identifier(task["task_id"])
        if tid in seen:
            raise ValueError("Duplicate task ID")
        seen.add(tid)
        if not isinstance(task.get("argv"), list) or not task["argv"] or any(not isinstance(x, str) or "\0" in x for x in task["argv"]):
            raise ValueError("argv must be a nonempty list of strings")
        if not isinstance(task.get("workload_key"), str) or not task["workload_key"]:
            raise ValueError("A workload_key is required")
        if task.get("cwd") is not None and (not isinstance(task["cwd"], str) or not Path(task["cwd"]).is_absolute()):
            raise ValueError("cwd must be an absolute path")
        env = task.get("env", {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k or "=" in k or "\0" in k + v for k, v in env.items()):
            raise ValueError("env must contain valid string keys and values")
        if {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"} & env.keys():
            raise ValueError("Task env cannot override the allocation's device/job binding")
        estimates = task.get("runtime_estimates", {})
        if not isinstance(estimates, dict):
            raise ValueError("runtime_estimates must be a model-to-seconds mapping")
        for value in estimates.values():
            positive_number(value, "runtime estimate")
        if task.get("fallback_seconds") is not None:
            positive_number(task["fallback_seconds"], "fallback_seconds")
    if len({t["workload_key"] for t in tasks}) != 1:
        raise ValueError("A batch must have a homogeneous workload_key")
    digest(batch)  # Reject nonfinite or non-JSON metadata as well.


def model_name(name):
    """Canonicalize known families without treating L40S as L40, etc."""
    for pattern, model in [(r"\bRTX PRO 6000 Blackwell.*Max[- ]Q", "B6000Q"),
                           (r"\bRTX PRO 6000 Blackwell", "B6000"),
                           (r"\bRTX PRO 4000 Blackwell", "B4000"),
                           (r"\bRTX 4000 Ada", "4000Ada"),
                           (r"\b(A100|H100|H200|L40S|L40|A6000|A4000)\b", None)]:
        match = re.search(pattern, name, re.I)
        if match:
            return model or match.group(1).upper()
    return name


def detect_device():
    """Ask CUDA for its one visible device, not nvidia-smi's node inventory.

    CUDA applies CUDA_VISIBLE_DEVICES before assigning local ordinal zero. The
    driver UUID therefore identifies the actual GPU even when Slurm's global
    index and the process-local ordinal differ. MIG/multiple GPUs need a separate
    qualified adapter and are deliberately not inferred from host totals.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len(visible.split(",")) != 1 or visible in {"-1", "NoDevFiles"}:
        raise RuntimeError("Exactly one CUDA_VISIBLE_DEVICES assignment is required")
    if visible.startswith("MIG-"):
        raise RuntimeError("MIG binding requires a separately qualified adapter")
    lib = ctypes.CDLL("libcuda.so.1")

    def call(name, *args):
        code = getattr(lib, name)(*args)
        if code:
            raise RuntimeError(f"CUDA {name} failed with code {code}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError(f"Expected one CUDA-visible GPU, found {count.value}")
    device = ctypes.c_int()
    call("cuDeviceGet", ctypes.byref(device), 0)
    name = ctypes.create_string_buffer(256)
    call("cuDeviceGetName", name, len(name), device)
    identifier = (ctypes.c_ubyte * 16)()
    call("cuDeviceGetUuid", ctypes.byref(identifier), device)
    memory = ctypes.c_size_t()
    call("cuDeviceTotalMem_v2", ctypes.byref(memory), device)
    actual = "GPU-" + str(uuid.UUID(bytes=bytes(identifier)))
    if visible.startswith("GPU-") and not actual.startswith(visible):
        raise RuntimeError("CUDA UUID disagrees with its explicit visible assignment")
    text = name.value.decode()
    return {"model": model_name(text), "name": text, "uuid": actual,
            "memory_gib": memory.value / 1024 ** 3,
            "source": "cuda_visible_device", "cuda_visible_devices": visible}


class HostMemory:
    """Kernel cgroup peak; only an identified Slurm job ancestor is whole-job."""
    def __init__(self, path=None, scope=None):
        self.path = Path(path) if path else None
        self.scope = scope or "unavailable"
        self.peak = None
        self.stat = {}
        if self.path is None:
            try:
                entries = Path("/proc/self/cgroup").read_text().splitlines()
                relative = next(line.split(":", 2)[2] for line in entries if line.startswith("0::"))
                path = Path("/sys/fs/cgroup") / relative.lstrip("/")
                self.scope = "process_cgroup"
                job = os.environ.get("SLURM_JOB_ID")
                if job and job.isdigit():
                    for parent in (path, *path.parents):
                        if re.fullmatch(r"job_" + job + r"(?:\.scope)?", parent.name):
                            path, self.scope = parent, "whole_allocation_cgroup"
                            break
                if (path / "memory.current").exists():
                    self.path = path
                else:
                    self.scope = "unavailable"
            except (OSError, StopIteration):
                pass
        self.initial_oom = self.oom_count()
        self.sample()

    def oom_count(self):
        try:
            entries = dict(line.split() for line in (self.path / "memory.events").read_text().splitlines())
            return int(entries.get("oom_kill", 0))
        except (OSError, TypeError, ValueError):
            return None

    def sample(self):
        if self.path is None:
            return
        try:
            # Without a kernel peak this is only a lower bound, unsuitable for
            # reducing a whole-allocation memory request.
            if not (self.path / "memory.peak").exists():
                self.scope = "sampled_process_cgroup"
                filename = "memory.current"
            else:
                filename = "memory.peak"
            value = int((self.path / filename).read_text()) / 1024 ** 3
            self.peak = max(self.peak or 0, value)
            entries = dict(line.split() for line in (self.path / "memory.stat").read_text().splitlines())
            self.stat = {key + "_gib": int(entries[key]) / 1024 ** 3 for key in ("anon", "file") if key in entries}
        except (OSError, ValueError):
            pass

    def receipt(self):
        self.sample()
        return {"host_peak_gib": self.peak, "host_scope": self.scope,
                "host_measurement": {"path": str(self.path) if self.path else None,
                                     "current_components": self.stat,
                                     "oom_kill_start": self.initial_oom,
                                     "oom_kill_end": self.oom_count()}}


def _new_attempt(root):
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1000000):
        path = root / f"attempt{number:06d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Attempt index exhausted")


def _completion(state, task):
    path = state / "completions" / (task["task_id"] + ".json")
    if not path.exists():
        return None
    index = read_json(path)
    if index["task_digest"] != task_digest(task):
        raise ValueError("Existing completion has a different scientific identity")
    receipt = read_json(index["receipt_path"])
    if digest(receipt) != index["receipt_digest"] or receipt.get("task_id") != task["task_id"] or receipt.get("task_digest") != task_digest(task) or receipt.get("status") != "COMPLETE" or receipt.get("returncode") != 0:
        raise ValueError("Completion index does not bind a successful immutable receipt")
    return dict(receipt, reason="already_complete", elapsed_seconds=0,
                reused=True, skipped=True, completion_receipt=index["receipt_path"])


def _stop_group(process, grace=5):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _execute(task, attempt, deadline, stopped, memory, *, batch_id, batch_digest):
    start = time.monotonic()
    report = {"task_id": task["task_id"], "task_digest": task_digest(task),
              "status": "FAILED", "returncode": None, "reason": "command_launch_failed",
              "elapsed_seconds": 0, "attempt_dir": str(attempt), "started_at": utcnow(),
              "batch_id": batch_id, "batch_digest": batch_digest,
              "job_id": os.environ.get("SLURM_JOB_ID"), "skipped": False}
    atomic_json(attempt / "task.json", task)
    atomic_json(attempt / "start.json", report)
    environment = dict(os.environ, **task.get("env", {}))
    environment.update(SLURM_CONTROL_ATTEMPT_DIR=str(attempt), SLURM_CONTROL_TASK_ID=task["task_id"])
    before_oom = memory.oom_count()
    process = None
    try:
        with (attempt / "stdout.log").open("wb") as out, (attempt / "stderr.log").open("wb") as err:
            process = subprocess.Popen(task["argv"], cwd=task.get("cwd"), env=environment,
                                       stdout=out, stderr=err, start_new_session=True, close_fds=True)
            report["pid"] = process.pid
            atomic_json(attempt / "start.json", report)
            while process.poll() is None:
                memory.sample()
                if stopped[0] or time.monotonic() >= deadline:
                    report["reason"] = "preempted" if stopped[0] else "time_limit"
                    _stop_group(process)
                    break
                time.sleep(0.1)
            report["returncode"] = process.wait()
        if report["reason"] not in {"preempted", "time_limit"}:
            report["reason"] = "completed" if report["returncode"] == 0 else "command_failed"
            if report["returncode"] == 0:
                report["status"] = "COMPLETE"
            else:
                after_oom = memory.oom_count()
                if before_oom is not None and after_oom is not None and after_oom > before_oom:
                    report["reason"] = "host_oom"
                else:
                    # Preserve explicit device OOM distinction. Do not classify
                    # SIGKILL alone as host OOM.
                    with (attempt / "stderr.log").open("rb") as stream:
                        stream.seek(max(0, (attempt / "stderr.log").stat().st_size - 65536))
                        tail = stream.read().decode(errors="replace")
                    if re.search(r"CUDA out of memory|CUDA_ERROR_OUT_OF_MEMORY|torch\.cuda\.OutOfMemoryError", tail, re.I):
                        report["reason"] = "gpu_oom"
    except Exception as exc:
        if process is not None and process.poll() is None:
            _stop_group(process)
        report["error"] = f"{type(exc).__name__}: {exc}"
    report.update(elapsed_seconds=time.monotonic() - start, finished_at=utcnow())
    atomic_json(attempt / "receipt.json", report)
    return report


def run_batch(batch_path, state, *, device=None, binding=None, memory=None):
    """Run one batch. ``binding``/``memory`` are injected qualification seams."""
    batch_path, state = Path(batch_path).resolve(), Path(state).resolve()
    batch = read_json(batch_path)
    validate_batch(batch)
    fingerprint = digest(batch)
    bid = batch["batch_id"]
    with exclusive(state / "locks" / ("batch-" + bid + ".lock")):
        manifest = state / "manifests" / (bid + ".json")
        if manifest.exists() and digest(read_json(manifest)) != fingerprint:
            raise ValueError("Batch ID already binds a different immutable manifest")
        if not manifest.exists():
            atomic_json(manifest, batch)
        attempt = _new_attempt(state / "batch_attempts" / bid)
        start = time.monotonic()
        report = {"batch_id": bid, "batch_digest": fingerprint, "attempt_id": attempt.name,
                  "started_at": utcnow(), "job_id": os.environ.get("SLURM_JOB_ID"),
                  "device": None, "tasks": [], "reason": "setup_failed"}
        stopped = [None]
        previous = {}
        memory = memory or HostMemory()
        try:
            for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, lambda number, frame: stopped.__setitem__(0, number))
            if device not in {None, "cpu"}:
                raise ValueError("Only explicit CPU mode is supported; GPUs must be detected")
            cpu = device == "cpu" or (batch.get("gpus") == 0 and batch["allowed_models"] == ["cpu"])
            actual = {"model": "cpu", "name": "cpu", "uuid": None, "source": "explicit_cpu"} if cpu else (binding() if binding else detect_device())
            if actual.get("model") not in batch["allowed_models"]:
                raise ValueError("Actual device is outside allowed_models")
            report["device"] = actual
            deadline = start + batch["walltime_seconds"] - batch.get("reserve_seconds", 120)
            report["reason"] = "completed"
            for task in batch["tasks"]:
                if digest(read_json(batch_path)) != fingerprint:
                    raise ValueError("Input batch changed during execution")
                with exclusive(state / "locks" / ("task-" + task["task_id"] + ".lock")):
                    done = _completion(state, task)
                    if done:
                        report["tasks"].append(done)
                        continue
                    if stopped[0]:
                        report["reason"] = "preempted"
                        break
                    estimate = task.get("runtime_estimates", {}).get(actual["model"], task.get("fallback_seconds"))
                    if estimate is None:
                        report["reason"] = "missing_runtime_estimate"
                        break
                    if time.monotonic() + estimate > deadline:
                        report["reason"] = "insufficient_time"
                        break
                    task_attempt = _new_attempt(state / "attempts" / task["task_id"])
                    result = _execute(task, task_attempt, deadline, stopped, memory,
                                      batch_id=bid, batch_digest=fingerprint)
                    report["tasks"].append(result)
                    if result["status"] != "COMPLETE":
                        report["reason"] = result["reason"]
                        break
                    atomic_json(state / "completions" / (task["task_id"] + ".json"),
                                {"task_digest": task_digest(task), "receipt_path": str(task_attempt / "receipt.json"),
                                 "receipt_digest": digest(result)})
        except Exception as exc:
            report.update(reason="worker_error", error=f"{type(exc).__name__}: {exc}")
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            completed = {row["task_id"] for row in report["tasks"]}
            report["tasks"].extend({"task_id": task["task_id"], "task_digest": task_digest(task),
                                    "status": "UNSTARTED", "returncode": None, "reason": report["reason"],
                                    "elapsed_seconds": 0, "attempt_dir": None}
                                   for task in batch["tasks"] if task["task_id"] not in completed)
            report.update(finished_at=utcnow(), elapsed_seconds=time.monotonic() - start, **memory.receipt())
            atomic_json(state / "receipts" / bid / (attempt.name + ".json"), report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", choices=["cpu"])
    args = parser.parse_args(argv)
    report = run_batch(args.batch, args.state, device=args.device)
    print(json.dumps(report, sort_keys=True))
    # Unstarted tails are ordinary command-boundary stops. Real command/setup
    # failures remain nonzero, including device OOM and preemption.
    return 0 if report["reason"] in {"completed", "insufficient_time"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
