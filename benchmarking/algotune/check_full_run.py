#!/usr/bin/env python3
"""Readiness check for an AlgoTune task manifest.

Without ``--smoke`` this imports each task module and constructs the task
(fast). With ``--smoke`` it also generates one problem at the manifest's
official ``n``, solves it with the reference implementation, and validates the
result. Missing dependencies or broken tasks are reported without stopping the
rest of the suite.

Examples:
    python benchmarking/algotune/check_full_run.py
    python benchmarking/algotune/check_full_run.py --smoke --start 0 --limit 10
    python benchmarking/algotune/check_full_run.py --tasks svm,tsp
"""

import argparse
import importlib
import json
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ALGOTUNE_ROOT = "/hx2-weka/home/nr1713/AlgoTune"


class TaskTimeout(RuntimeError):
    pass


def _with_timeout(seconds, func, *args, **kwargs):
    if not seconds or seconds <= 0:
        return func(*args, **kwargs)

    def _handler(signum, frame):
        raise TaskTimeout(f"timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algotune-root", default=DEFAULT_ALGOTUNE_ROOT)
    parser.add_argument("--manifest", default=str(HERE / "tasks_full.json"))
    parser.add_argument("--tasks", default=None, help="Comma-separated subset")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", default="1", help="First seed is used for smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.algotune_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from AlgoTuneTasks.factory import TaskFactory

    rows = json.loads(Path(args.manifest).read_text())
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        rows = [row for row in rows if row["name"] in wanted]
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]
    seed = int(str(args.seeds).split(",")[0])

    failures = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        name = row["name"]
        n = row.get("n")
        try:
            importlib.import_module(f"AlgoTuneTasks.{name}.{name}")
            task = TaskFactory(name)
            detail = "import+factory"
            if args.smoke:
                problem = _with_timeout(
                    args.timeout, task.generate_problem, n, random_seed=seed
                )
                solution = _with_timeout(args.timeout, task.solve, problem)
                ok = _with_timeout(
                    args.timeout, task.is_solution, problem, solution
                )
                detail = f"smoke ok={ok}"
                if not ok:
                    raise RuntimeError("reference solution failed is_solution")
            print(f"[{index}/{len(rows)}] OK   {name} n={n} ({detail})", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((name, exc))
            print(
                f"[{index}/{len(rows)}] FAIL {name} n={n}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.smoke:
                traceback.print_exc(limit=2)

    elapsed = time.perf_counter() - started
    print(
        f"\nChecked {len(rows)} tasks in {elapsed:.1f}s: "
        f"{len(rows) - len(failures)} ok, {len(failures)} failed"
    )
    if failures:
        print("Failures:")
        for name, exc in failures:
            print(f"  {name}: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
