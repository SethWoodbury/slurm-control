# Why a small Slurm command layer

This package supplies command identity, receipts, budget accounting and measured
resource policy. Slurm remains the placement scheduler. A pool is a tested GPU
compatibility allowlist; the GPU actually assigned selects the worker's runtime
estimate. Host-memory tuning uses comparable whole-allocation evidence, with an
explicit additive margin and bounded retry policy.

Existing tools are useful, but address different boundaries:

| Option | Useful capabilities | Fit here |
| --- | --- | --- |
| [Submitit](https://github.com/facebookincubator/submitit) | Submit Python callables to Slurm, retrieve results and logs. Its [checkpoint interface](https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md) can supply a callable for requeue. | Good optional adapter for Python-native tasks. Our heterogeneous argv commands still need durable command identity, budget reservations and explicit resource evidence. |
| [simple-slurm](https://github.com/amq92/simple_slurm) | Construct and submit Slurm batch scripts from Python. | A lightweight alternative to our small argv submission adapter; it does not eliminate the campaign ledger or worker receipt contract. |
| [Parsl](https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.HighThroughputExecutor.html) | Worker pools, tasks, accelerator assignments, worker memory limits and drain periods; [providers](https://parsl.readthedocs.io/en/stable/userguide/configuration/config.html) provision scheduler resources. | Consider for a future Python task graph. A separate executor/manager layer adds lifecycle machinery to the current sequential-command use case. |
| [Snakemake Slurm executor](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) | File-based workflows, resource profiles, retries and Slurm submission. | Strong when outputs and dependencies form a reusable DAG. Our application can eventually be a rule or executor client; adopting a DAG engine solely for independent commands is unnecessary. |
| [Dask-jobqueue](https://jobqueue.dask.org/en/stable/generated/dask_jobqueue.SLURMCluster.html) | Slurm-backed distributed worker clusters with scaling and adaptive allocation. | Appropriate for distributed computations needing Dask workers. Long independent external commands do not currently require that scheduler/worker network. |

These fit judgments are architectural recommendations, not claims that another
project cannot be extended. We retain no dependency on these tools in the minimal
core. Adapters should reuse the same task manifests and receipts, so changing the
submission frontend does not change scientific identity.

## Deliberate boundaries

- The single controller owns SQLite, submission intents and the GPU-hour budget.
  A lost `sbatch` response remains an unresolved reserved allocation; it is not
  permission to submit another copy.
- The worker runs argv sequentially and writes durable receipts. It does not
  choose nodes, alter scientific parameters, query another campaign's ledger or
  automatically escalate memory.
- Resource comparisons separate workload strategy, runtime/checkpoint identity,
  size class and resource-affecting options. The controller may aggregate a
  configured cohort before changing requests. Explicit overrides retain a
  reason and are independent from scientific task identity.
- Retrying requires authoritative terminal scheduler evidence. A receipt or an
  idle GPU alone cannot prove that an old job has stopped.
- Initial native submissions use `--no-requeue`: the controller gives retries a
  new batch ID. Multiple final receipts for one allocation deliberately require
  investigation; native same-job requeue is not qualified in this version.

## Worker contract

Run `python -m slurm_control.worker --batch /absolute/batch.json --state
/absolute/state`. A CPU batch declares `gpus: 0` and `allowed_models: ["cpu"]`;
`--device cpu` is also available for a direct CPU smoke test. Production GPU
binding uses CUDA's visible-device API, which accounts for the local ordinal
remapping described in [Slurm's GPU documentation](https://slurm.schedmd.com/gres.html#GPU_Management).
Exactly one visible GPU is required. MIG and multi-GPU execution need their own
qualified binding adapter. Unknown model names remain exact names, requiring an
explicit compatibility entry and runtime fallback.

The batch contains `batch_id`, `tasks`, `allowed_models`, `walltime_seconds` and
`reserve_seconds`. Each task has `task_id`, `argv`, `cwd`, `env`, `workload_key`,
model-keyed `runtime_estimates`, and an optional explicit `fallback_seconds`.
Use absolute `cwd` and input/output paths for portable runs. Scientific identity
includes task ID, argv, cwd, environment, workload key, inputs and metadata;
changing only estimates does not invalidate completed work. A batch ID always
binds one exact immutable manifest.

The next command starts only if its estimate for the actual GPU fits before the
walltime reserve. SIGUSR1, SIGTERM and SIGINT terminate its process group. The
worker preserves an interrupted attempt and restarts that command from its
beginning on a later authorized allocation. It does not reconstruct optimizer,
trajectory or application-internal checkpoints.

Each command receives `SLURM_CONTROL_ATTEMPT_DIR` and `SLURM_CONTROL_TASK_ID`.
Its stdout/stderr and receipts are kept in the attempt directory. **Commands must
be idempotent, or write their outputs beneath the supplied attempt directory.**
Fresh logs cannot prevent an arbitrary command from overwriting an absolute
output path embedded in its argv. The worker never deletes partial outputs.

Small durable artifacts under the configured state directory:

- `manifests/{batch_id}.json`: immutable submitted manifest.
- `attempts/{task_id}/attemptNNNNNN/`: task manifest, start receipt, stdout,
  stderr and command outcome. Outcome includes batch/job/task identity.
- `completions/{task_id}.json`: scientific identity and hash-bound successful
  outcome reference. A retry can skip only an exact verified match; skipped
  rows explicitly have `skipped: true` and zero fresh command runtime.
- `receipts/{batch_id}/attemptNNNNNN.json`: final allocation-worker receipt,
  including actual device, task outcomes and host-memory measurement scope.

Kernel cgroup peaks are allocation-wide only when the sampler finds the exact
Slurm job ancestor. Other/unsupported cgroup layouts remain explicitly scoped or
unavailable and cannot justify reducing host RAM. Current anon/file components
are recorded separately from peak; they are not a peak anon/file decomposition.
A CUDA OOM is distinct from a host cgroup OOM. SIGKILL alone is not sufficient to
classify either. The Slurm accounting record remains the final authority when
an allocation dies before publishing its last receipt.
