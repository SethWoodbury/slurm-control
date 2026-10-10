# Resource policy

`slurm_control.policy` contains pure functions. They accept validated measurements
and return JSON decisions; they do not submit jobs, select nodes, change task
ownership, or write campaign state. Slurm chooses an allocated GPU from the
configured compatibility list. A model being fast or currently idle is not an
admission requirement.

## Comparable evidence

`workload_key()` hashes runtime/checkpoint identity, strategy, size class and
explicit resource-affecting options. For example, beam width, number of starts
per command, thinking level and precision belong in those options. Motif and
task IDs do not. Callers are responsible for choosing a meaningful size class
and including options that affect resources; the policy cannot infer them from
an arbitrary command.

Measurements have independent `batch_id` (or `allocation_id`), `workload_key`,
actual `device` model, `status`, `host_scope`, and `host_peak_gib`. Device may be a
canonical model string or a dictionary with `model`/`name`. Model labels must
match the compatibility list; do not group different hardware under a guessed
name. Duplicate receipts count once; contradictory measurements for the same
allocation are rejected. The caller validates receipt hashes and job identity
before passing evidence to this module.

Host memory evidence must cover the **whole allocation**, including children and
containers. Accepted scopes are `whole_allocation_cgroup` and
`allocation_cgroup`; one process's RSS is not enough. Keep measurements in stable
chronological order. The controller reads the sample ledger in insertion order.

## Host RAM

The default successful-evidence recommendation is:

```text
ceil_to_1_GiB(maximum comparable whole-allocation peak + 2 GiB)
```

There is no hidden percentage or coarse memory tier. `memory_margin_gib` and
`round_gib` are explicit policy fields. The default adaptation cohort is 256
successful independent allocations (`adapt_every`), with at least three
independent allocations for each GPU model (`min_independent_allocations`).
An explicit pilot may lower the cohort size; repeated commands in one job do
not satisfy the independent-job requirement.

The recommendation uses completed cohorts cumulatively. For example, with a
256-job cohort, evidence from jobs 257–511 does not lower the request until job
512 completes. This intentionally retains previously observed high peaks.
Failed/partial measurements cannot justify a reduction. Their observed peaks
plus the margin remain upward safety floors, but the RAM granted to an unrelated
application failure does not itself establish a floor. A confirmed host OOM immediately creates
a floor of its granted RAM plus `oom_step_gib` (default 4 GiB). The same OOM is
not incremented again on every controller tick. Task retry floors are also
honored, including when an operator sets a fixed request.

A broad compatible pool takes the maximum recommendation across its eligible
GPU models. An unmeasured model retains the explicit initial request
(`memory_gib`/`current_gib`). With no adequate evidence and no fallback, the
decision is `hold`. A deliberate override requires both `memory_gib` and a
nonempty `reason`; it must satisfy the floor and `max_memory_gib` (default 128).
It is recorded separately from measured evidence. Failed floors must be
investigated and explicitly revised upstream if they were misclassified.

`oom_memory_request()` distinguishes host RAM and device VRAM. Host retries add
4 GiB by default, at most two OOM retries and no higher than the configured RAM
ceiling. Device OOM holds for a compatible device with more VRAM; increasing
host RAM does not fix it. The ledger also enforces the overall started-attempt
limit and requires terminal/cleanup evidence before retrying.

## Runtime and fixed batches

`estimate_runtime()` groups completed command durations by workload and the
**actual allocated GPU**. It reads each `tasks` entry with `status=COMPLETE` and
`elapsed_seconds`, including completed commands preserved in a partial job.
It never divides allocation walltime by the planned command count, since that
count may include an unstarted tail. A `command_seconds` scalar is supported
for one-command allocation fixtures; do not supply both forms for the same
command.

The default estimate updates after each full 256-allocation cohort, with at
least three independent jobs. It returns:

| Purpose | Default |
| --- | --- |
| Batch planning | Mean command duration × 1.25 |
| Next-command deadline check | max(mean, p90) × 1.25 |
| Unknown hardware / insufficient evidence | Explicit fallback, otherwise hold |

Both margins are configurable (`time_margin_fraction` and
`deadline_margin_fraction`). A p90 bound is an estimate, not a guarantee against
rare overruns. A worker still checks remaining walltime before every command
and preserves completed receipts if preempted or timed out.

`plan_batch()` takes a fixed, homogeneous task list and a chosen reference GPU's
planning estimate. Its default surplus is 20% (`backlog_factor=1.2`), bounded by
the explicit `max_backlog_factor=1.5`. Surplus assigns more commands; it never
extends the Slurm request or budget reservation. Startup and shutdown reserve
default to 120 seconds each. Individual commands must fit an empty allocation.
For broad hardware pools, the worker must use the actual GPU's deadline
estimate, stop before another command cannot fit, and return the unstarted tail
without counting another started attempt. The reusable planner does not invent
distributed work stealing or sub-command checkpoints.

## API

```python
from slurm_control.policy import recommend_memory, estimate_runtime, plan_batch

memory = recommend_memory(
    samples, workload=key, eligible_models=["A100", "L40"],
    current_gib=32, retry_floor_gib=0, policy={"adapt_every": 256},
)
timing = estimate_runtime(
    samples, workload=key, gpu_model="L40", fallback=900,
    policy={"adapt_every": 256},
)
batch = plan_batch(commands, timing, walltime_seconds=8 * 3600)
```

These are recommendations for future work. Changing a policy does not edit a
running allocation. Pending-job migration and live campaign adoption require a
separate audited bridge preserving task identities, completion records, Slurm
submission intents, and total allocated GPU-hour accounting.
