# slurm-control

Keep this infrastructure independent of campaign chemistry and model code.
Use standard Python and Slurm commands; avoid adding another placement scheduler.
Slurm chooses nodes from a tested compatibility allowlist. Partition controls
queue policy, while the actual allocated GPU controls runtime estimates.

Resource policies use comparable whole-allocation measurements. Separate
workload families (for example gradient and search), runtime identity, size class,
and resource-affecting options. Default host RAM margin is an additive 2 GiB,
rounded to 1 GiB; never apply a hidden percentage or coarse memory tier.
Missing evidence requires an explicit fallback. OOM retries increase memory by
a bounded additive increment. Device VRAM OOM must not increase host RAM.

Single writer owns the campaign ledger. Workers write per-attempt receipts.
Record submission intent before sbatch; an ambiguous submission stays reserved
until reconciled. Never retry a possibly live allocation. Preserve completed
commands and immutable scientific inputs. Resume is between commands, unless a
task explicitly implements finer checkpointing. Run arbitrary commands as argv,
without a shell by default. Manual overrides must be explicit and recorded.

All heavy outputs, state, tests with generated artifacts and scratch work belong
in the user's configured scratch directory, not in this repository. Commit only code, small
configuration examples, tests and documentation. Do not copy private campaign
outputs or secrets into Git. This is a public GitHub repository under the MIT license.

Validate meaningful failure cases: duplicate submission, ambiguous sbatch,
partial completion, preemption, host/device OOM, budget limits, and resource
measurements from different workloads. Keep live campaign migration separate
from generic controller qualification; never claim prepared code is active.
