from pathlib import Path
from types import SimpleNamespace
import subprocess
import getpass

import pytest

from slurm_control.slurm import Slurm, SlurmSubmissionRejected


def reply(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_explicit_server_rejection_distinct_from_ambiguous_or_accepted_output():
    message = "sbatch: error: The job time limit is too short; run longer jobs!\nsbatch: error: Batch job submission failed: Access/permission denied\n"
    with pytest.raises(SlurmSubmissionRejected):
        Slurm(runner=lambda *a, **kw: reply(stderr=message, returncode=1)).call(["sbatch", "worker.sh"])
    for output, error in [("123\n", message), ("", "transport connection reset")]:
        with pytest.raises(RuntimeError) as caught:
            Slurm(runner=lambda *a, **kw: reply(stdout=output, stderr=error, returncode=1)).call(["sbatch", "worker.sh"])
        assert not isinstance(caught.value, SlurmSubmissionRejected)


def test_submit_argv_broad_constraints_no_shell_and_exact_resources():
    calls = []
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return reply("123;cluster\n")
    scheduler = Slurm(runner=runner)
    spec = {"walltime_seconds": 43200, "partition": "gpu-train", "cpus": 2, "memory_gib": 25,
            "gres": "gpu:large:1", "allowed_models": ["A100", "L40"]}
    assert scheduler.submit(spec, Path("/some path/worker.sh"), "exact-token", Path("/logs with spaces")) == "123"
    argv, kwargs = calls[0]
    assert argv[argv.index("--constraint") + 1] == "A100|L40"
    assert argv[argv.index("--mem") + 1] == "25600M"
    assert argv[argv.index("--time") + 1] == "12:00:00"
    assert argv[-1] == "/some path/worker.sh"
    assert kwargs["close_fds"] is True and kwargs["timeout"] == 30
    assert not kwargs.get("shell")


def test_accounting_preserves_token_elapsed_and_cancellation_reason():
    output = "123|CANCELLED by 1000|3600|sc-token|\n124|COMPLETED|10|other-token|\n123.batch|COMPLETED|2|sc-token|\n"
    scheduler = Slurm(runner=lambda *a, **kw: reply(output))
    assert scheduler.accounting(["123"])["123"] == {
        "job_id": "123", "state": "CANCELLED", "elapsed_seconds": 3600, "terminal": True, "token": "sc-token"}


def test_queue_transport_error_not_treated_as_no_running_jobs():
    def runner(*a, **kw):
        raise subprocess.TimeoutExpired("squeue", 30)
    with pytest.raises(subprocess.TimeoutExpired):
        Slurm(runner=runner).queue()


def test_blank_accounting_comment_falls_back_to_exact_job_name_and_owner():
    output = f"123|COMPLETED|2||sc-exact-token|{getpass.getuser()}|\n"
    record = Slurm(runner=lambda *a, **kw: reply(output)).accounting(["123"])["123"]
    assert record["token"] == "sc-exact-token"
    assert record["token_source"] == "job_name_and_owner"
    assert record["owner"] == getpass.getuser()


def test_nonempty_wrong_comment_is_not_hidden_and_wrong_owner_refused():
    output = f"123|COMPLETED|2|wrong-token|sc-exact-token|{getpass.getuser()}|\n"
    record = Slurm(runner=lambda *a, **kw: reply(output)).accounting(["123"])["123"]
    assert record["token"] == "wrong-token"
    wrong = "123|COMPLETED|2||sc-exact-token|different-user|\n"
    with pytest.raises(ValueError, match="owner"):
        Slurm(runner=lambda *a, **kw: reply(wrong)).accounting(["123"])
    missing = "123|COMPLETED|2||sc-exact-token||\n"
    assert Slurm(runner=lambda *a, **kw: reply(missing)).accounting(["123"])["123"]["token"] == ""
