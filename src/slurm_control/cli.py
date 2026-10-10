"""Explicit preview, submit, reconcile and override controls for humans and agents."""
import argparse
import json
from pathlib import Path
import signal
import time

from .common import atomic_json, utcnow
from . import controller, ledger
from .slurm import Slurm


def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def tick(config, *, submit=False, initialize=False):
    with ledger.locked_ledger(config["state"]) as db:
        if initialize:
            controller.initialize(db, config)
        if Path(config["state"], "PAUSED").exists():
            return {"status": "paused", **ledger.status(db)}
        completed = controller.reconcile(db, config["state"], Slurm(), config)
        planned = controller.preview_batches(db, config)
        submitted = []
        if submit:
            for batch in planned:
                ledger.reserve_batch(db, batch)
            db.commit()
            submitted = controller.submit_planned(db, config["state"], Slurm(), limit=config.get("max_admissions_per_tick", 32))
        result = {"at": utcnow(), "completed": completed, "submitted": submitted,
                  "planned": [{k: v for k, v in b.items() if k != "tasks"} | {"command_count": len(b["tasks"])} for b in planned],
                  "status": ledger.status(db), "submission_enabled": submit}
    atomic_json(Path(config["state"]) / "status.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Register immutable tasks and the GPU-hour budget; never submits")
    sub.add_parser("plan", help="Preview new homogeneous batches; never submits")
    sub.add_parser("status", help="Read ledger status")
    p = sub.add_parser("tick", help="Reconcile and preview, or explicitly submit one bounded window")
    p.add_argument("--submit", action="store_true")
    p = sub.add_parser("run", help="Repeat ticks; requires explicit --submit to launch")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--cycles", type=int, default=0, help="0 means run until signalled")
    p = sub.add_parser("override", help="Hold/release unowned tasks or set a retry memory floor")
    p.add_argument("action", choices=["hold", "release", "memory"])
    p.add_argument("--task", action="append", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--memory-gib", type=float)
    p = sub.add_parser("bind", help="Reconcile a known ambiguous submission using its Slurm token")
    p.add_argument("--batch", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    config = controller.load_config(args.config)
    if args.command == "run":
        if args.interval < 10 or args.cycles < 0:
            parser.error("interval must be >=10 seconds and cycles nonnegative")
        stopped = False
        def stop(signum, frame):
            nonlocal stopped
            stopped = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        cycle = 0
        while not stopped and (not args.cycles or cycle < args.cycles):
            try:
                emit(tick(config, submit=args.submit))
            except Exception as exc:
                # Leave reservations intact and report failure; next cycle may recover transport.
                failure = {"at": utcnow(), "phase": "controller_error", "type": type(exc).__name__, "error": str(exc)}
                atomic_json(Path(config["state"]) / "last_error.json", failure)
                emit(failure)
            cycle += 1
            deadline = time.monotonic() + args.interval
            while not stopped and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        return
    if args.command == "tick":
        emit(tick(config, submit=args.submit))
        return
    with ledger.locked_ledger(config["state"]) as db:
        if args.command == "init":
            controller.initialize(db, config)
            emit(ledger.status(db))
        elif args.command == "plan":
            emit(controller.preview_batches(db, config))
        elif args.command == "status":
            emit(ledger.status(db))
        elif args.command == "override":
            ledger.override_tasks(db, args.task, action=args.action, reason=args.reason, memory_gib=args.memory_gib)
            emit(ledger.status(db))
        elif args.command == "bind":
            scheduler = Slurm()
            row = db.execute("SELECT * FROM batches WHERE batch_id=?", (args.batch,)).fetchone()
            if row is None or row["state"] not in {"SUBMITTING", "AMBIGUOUS"} or row["job_id"]:
                raise ValueError("Only an ambiguous unbound submission can be bound")
            record = scheduler.queue().get(args.job) or scheduler.accounting([args.job]).get(args.job)
            if not record or record["token"] != row["token"]:
                raise ValueError("The job does not match the durable submission token")
            ledger.mark_submitted(db, args.batch, args.job)
            ledger.event(db, "operator_bind", {"batch_id": args.batch, "job_id": args.job, "reason": args.reason})
            emit(ledger.status(db))


if __name__ == "__main__":
    main()
