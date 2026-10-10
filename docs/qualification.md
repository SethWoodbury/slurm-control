# Initial qualification

The initial release passed **82 automated tests** on 2026-10-10 UTC.
They cover resource comparability, additive memory decisions, deadline packing,
budget reservations, duplicate prevention, submission ambiguity and rejection,
partial completion recovery, scheduler accounting, consistent GPU requests and
budget reservations, and worker failure handling.

Two small jobs also completed on a real Slurm cluster:

| Check | Result |
| --- | --- |
| CPU batch | Two sequential Python commands completed; immutable receipts reconciled into the ledger. |
| Single GPU batch | The worker identified its allocated A100, ran a Python command, and reconciled completion and actual allocated GPU time. |
| Host measurements | Both jobs recorded the whole-allocation cgroup memory peak. |
| Site submission rejection | A walltime below the site's minimum was rejected; commands were held and the reservation released after proving no submission occurred. |
| Accounting without comments | The site omits accounting comments; reconciliation verified the exact job-name token and owner as its fallback. |

The GPU check validates allocation binding and controller integration, not a
scientific application's performance, CUDA computation, or GPU memory needs.
Failure injection is automated; the qualification did not induce real cluster
OOM or preemption events. Applications and additional clusters need their own
bounded qualification before large production use.

Operational receipts, job IDs and campaign data stay outside this repository.
For another site, start with the CPU example, qualify one GPU command using
your actual application, and inspect its receipts before expanding the queue.
