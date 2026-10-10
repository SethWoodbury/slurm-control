"""A bounded argv-only Slurm transport. Node placement remains Slurm's job."""
from __future__ import annotations

import getpass
import re
import subprocess

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"}


class SlurmSubmissionRejected(RuntimeError):
    """The server explicitly refused sbatch; no allocation was accepted."""


def validated_gpu_request(spec):
    """Return an unambiguous GRES request consistent with GPU-hour accounting.

    Version 0.1 supports CPU without GRES, or exactly one GPU. Missing GPU GRES
    defaults to ``gpu:1``. Additional/multiple resources need a separate adapter
    that accounts for them rather than silently trusting a conflicting string.
    """
    count = spec.get("gpus", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count not in (0, 1):
        raise ValueError("gpus must be integer 0 (CPU) or 1 (one GPU)")
    gres = spec.get("gres")
    if count == 0:
        if gres not in (None, ""):
            raise ValueError("CPU allocations cannot request GRES")
        return None
    if gres in (None, ""):
        return "gpu:1"
    if not isinstance(gres, str) or not re.fullmatch(r"gpu:(?:[A-Za-z0-9_.-]+:)?1", gres):
        raise ValueError("Single-GPU GRES must be gpu:1 or gpu:<class>:1")
    return gres


class Slurm:
    def __init__(self, timeout=30, runner=None):
        self.timeout = timeout
        self.runner = runner or subprocess.run

    def call(self, argv):
        result = self.runner(argv, text=True, capture_output=True, timeout=self.timeout, check=False, close_fds=True)
        if result.returncode:
            if (argv[0] == "sbatch" and not result.stdout.strip()
                    and re.search(r"^sbatch: error: Batch job submission failed: .+", result.stderr, re.MULTILINE)):
                raise SlurmSubmissionRejected(f"sbatch rejected ({result.returncode}): {result.stderr[-2000:]}")
            raise RuntimeError(f"{argv[0]} exited {result.returncode}: {result.stderr[-2000:]}")
        return result.stdout

    def submit(self, spec, script, token, logdir):
        gres = validated_gpu_request(spec)
        seconds = int(spec["walltime_seconds"])
        argv = ["sbatch", "--parsable", "--job-name", token, "--comment", token,
                "--partition", spec["partition"], "--nodes", "1", "--ntasks", "1",
                "--cpus-per-task", str(spec["cpus"]), "--mem", f"{int(spec['memory_gib'] * 1024)}M",
                "--time", f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}",
                "--no-requeue", "--signal", "B:USR1@120", "--open-mode", "append",
                "--output", str(logdir / "slurm-%j.out"), "--error", str(logdir / "slurm-%j.err")]
        if spec.get("account"):
            argv.extend(["--account", spec["account"]])
        if gres:
            argv.extend(["--gres", gres])
        if spec.get("allowed_models") and spec.get("gpus", 1):
            # These are trusted cluster feature names, not user commands.
            if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in spec["allowed_models"]):
                raise ValueError("Invalid GPU feature name")
            argv.extend(["--constraint", "|".join(spec["allowed_models"])])
        out = self.call(argv + [str(script)]).strip()
        if not re.fullmatch(r"\d+(;[A-Za-z0-9_.-]+)?", out):
            raise RuntimeError("Ambiguous sbatch response: " + out[:200])
        return out.split(";")[0]

    def queue(self):
        out = self.call(["squeue", "-r", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%P|%k"])
        result = {}
        for line in out.splitlines():
            if not line.strip():
                continue
            fields = line.split("|", 3)
            if len(fields) != 4:
                raise ValueError("Malformed Slurm queue response")
            job, state, partition, comment = fields
            if not job.isdigit():
                # Unrelated arrays are not owned single-GPU campaign jobs.
                continue
            result[job] = dict(job_id=job, state=state, partition=partition, token=comment)
        return result

    def accounting(self, job_ids):
        if not job_ids:
            return {}
        if any(not str(x).isdigit() for x in job_ids):
            raise ValueError("Invalid job ID")
        out = self.call(["sacct", "-X", "-n", "-P", "-j", ",".join(map(str, job_ids)),
                         "--format=JobIDRaw,State%40,ElapsedRaw,Comment,JobName%128,User"])
        result = {}
        for line in out.splitlines():
            fields = line.split("|")
            if len(fields) < 4 or fields[0] not in set(map(str, job_ids)):
                continue
            job, raw, elapsed, comment = fields[:4]
            name = fields[4] if len(fields) > 4 else ""
            owner = fields[5] if len(fields) > 5 else ""
            if owner and owner != getpass.getuser():
                raise ValueError("Slurm accounting owner differs from submitting user")
            state = raw.split()[0].rstrip("+")
            if not elapsed.isdigit():
                continue
            result[job] = {"job_id": job, "state": state, "elapsed_seconds": int(elapsed),
                           "terminal": state in TERMINAL_STATES, "token": comment}
            if not comment and name and owner == getpass.getuser():
                # Some sites omit Comment from accounting storage. We set both
                # JobName and Comment to the unique durable token at submission.
                # A nonempty wrong Comment is never hidden by this fallback.
                result[job].update(token=name, token_source="job_name_and_owner", owner=owner)
        return result
