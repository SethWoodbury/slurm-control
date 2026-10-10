# slurm-control

A small, inspectable controller for batches of command-line work on Slurm.
It keeps command identity, progress, resource measurements and a GPU-hour budget.
Slurm chooses the nodes and decides when queued jobs start.

Public, open-source software under the [MIT license](LICENSE).

```mermaid
flowchart LR
    N[Notebook or CLI] --> L[Worklist and budget ledger]
    L --> Q[Slurm queues]
    Q --> W[Sequential command worker]
    W --> R[Immutable completion receipts]
    R --> L
    R --> M[Independent metrics and review]
```

The core has no Python dependencies beyond the standard library. Cluster settings
and workload settings are data, and command execution uses argument lists rather
than shell interpolation. See [the alternatives review](docs/alternatives.md) for
the comparison with existing open-source tools.

## Resource decisions

- **Host RAM:** comparable measured allocation peak + **2 GiB**, rounded to
  1 GiB. Gradient and search use separate workload keys. CPU jobs work the same way.
- **OOM recovery:** proven host OOM increases the next request by a configurable
  fixed increment (default 4 GiB), with attempt and memory ceilings. GPU OOM is
  held for a pool with greater device VRAM; adding host RAM cannot fix it.
- **Timing:** plan from averages and a modest margin; workers use the actual
  allocated GPU for the next-command deadline check. Partition names do not
  imply a hardware speed.
- **Adaptation:** resource changes use completed cohorts (default 256 independent
  allocations, configurable for pilots), with missing evidence explicitly shown.
- **Overrides:** fixed RAM requests, pool limits, walltime, model allowlists,
  holds and retries are visible controls. Overrides require reasons; uncertainty
  about a submitted job never authorizes a duplicate submission.

The [resource policy](docs/resource-policy.md) defines measurement comparability,
sample limits and override behavior. [DIGS pool examples](examples/digs_gpu_pools.json)
show normal small/large, broad backfill and a modest train supplement.

## Start with the CPU example

From this checkout (or install with `pip install -e .`), first adjust the example's
partition, account and scratch directory for your cluster. Paths support
environment variables such as `${USER}` and `~`:

```bash
export PYTHONPATH="$PWD/src"
python -m slurm_control.cli --config examples/cpu_campaign.json init
python -m slurm_control.cli --config examples/cpu_campaign.json plan
python -m slurm_control.cli --config examples/cpu_campaign.json tick --submit
python -m slurm_control.cli --config examples/cpu_campaign.json tick
python -m slurm_control.cli --config examples/cpu_campaign.json status
```

`init` registers immutable commands. `plan` previews resource requests. Only
`tick --submit` or `run --submit` sends jobs to Slurm. The CPU example requests
one CPU, 1 GiB and ten minutes on `cpu-bf`; its files go to scratch. DIGS
rejects shorter job limits used in the initial qualification; the command batch
exits as soon as its work completes rather than consuming the full limit.

For automatic operation, run the following in a small CPU allocation:

```bash
python -m slurm_control.cli --config /path/to/campaign.json run --submit --interval 60
```

The configuration supplies `state`, `tasks_path`, `budget_gpu_hours`, a
`workloads` mapping and a `pools` list. Each task has a unique `task_id`, an `argv`
list, a `workload_key`, and explicit timing estimates/fallbacks. Include input
hashes and model/runtime versions in `inputs`/`metadata`. Change the task ID if
the scientific command changes. Resource estimates can change independently.

Keep state and generated output on scratch. A controller restart reuses the same
state directory. Frozen worker source archives are stored by hash with each run,
so queued jobs do not import a changing checkout.

## Recovery and operator controls

```bash
python -m slurm_control.cli --config campaign.json override hold --task task-001 --reason 'Inspect input geometry'
python -m slurm_control.cli --config campaign.json override memory --task task-001 --memory-gib 32 --reason 'Larger measured cold startup'
python -m slurm_control.cli --config campaign.json override release --task task-001 --reason 'Input audit complete'
```

Create `STATE/PAUSED` to pause submission. Removing it resumes ordinary ticks.
An ambiguous `sbatch` response retains its reservation. The controller adopts a
matching live Slurm comment token automatically; `bind --batch ID --job JOB
--reason TEXT` can recover a known completed submission after exact token checks.

Completed commands are never deliberately rerun. Interrupted commands restart
from a new attempt; the core does not checkpoint an application's internal
trajectory. Commands should be idempotent or write their generated output under
the worker's `SLURM_CONTROL_ATTEMPT_DIR`. Logs and completion receipts are
separate from scientific outputs. Application failures and manual cancellations
are held for inspection; preemption, timeout and proven host OOM have bounded
automatic recovery.

There is one ledger writer, one lock, and no distributed work-stealing service.
The worker never writes SQLite. Unknown scheduler state remains unknown. A
machine/kernel or shared-filesystem failure can still require an administrator.

## Development and scope

See [qualification and limits](docs/qualification.md) for tested behavior and the
scope of the real CPU/GPU smoke tests.

```bash
python -m pytest -q
```

Version 0.1 supports CPU or single-GPU command allocations. It is infrastructure,
not a chemistry or protein-design workflow engine. AF3 reuse, scientific filters,
candidate publication and application checkpointing belong in campaign adapters.
When adopting an existing campaign, keep a single authoritative ledger and never
register already queued commands in a second controller. Validate application
compatibility separately from infrastructure qualification.
