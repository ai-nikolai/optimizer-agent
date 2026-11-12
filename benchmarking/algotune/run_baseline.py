#!/usr/bin/env python3
"""Evaluate the unoptimized AlgoTune baselines.

Each task's ``baseline.py`` delegates to the official reference solver. Running
it through the same evaluator used by the agent gives:

- reference correctness for every task
- ``runtime_original_ms`` / ``runtime_custom_ms`` reference timings
- a speedup sanity check (should be ~1.0x)

This needs no LLM and no GPU. Example:

    python benchmarking/algotune/run_baseline.py \
        --manifest benchmarking/algotune/dataset_full/manifest.json \
        --dataset-root benchmarking/algotune/dataset_full \
        --output experiments/algotune_baseline/baseline_results.csv
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_ALGOTUNE_ROOT = "/hx2-weka/home/nr1713/AlgoTune"
MARKER = "=== Evaluation Result ==="


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(HERE / "dataset/manifest.json"))
    parser.add_argument("--dataset-root", default=str(HERE / "dataset"))
    parser.add_argument("--tasks", default=None, help="Comma-separated subset")
    parser.add_argument("--task-list", default=None, help="Named official list")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--algotune-root", default=DEFAULT_ALGOTUNE_ROOT)
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--solve-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "experiments/algotune_baseline/baseline_results.csv")
    )
    return parser.parse_args()


def parse_result(stdout):
    marker = stdout.rfind(MARKER)
    if marker < 0:
        return None
    for line in stdout[marker + len(MARKER) :].splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main():
    args = parse_args()
    algotune_root = Path(args.algotune_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())

    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        manifest = [row for row in manifest if row["task"] in wanted]
    if args.task_list:
        lists = json.loads((HERE / "task_lists.json").read_text())
        if args.task_list not in lists:
            raise SystemExit(
                f"unknown task list '{args.task_list}'; "
                f"available: {', '.join(sorted(lists))}"
            )
        wanted = set(lists[args.task_list])
        manifest = [row for row in manifest if row["task"] in wanted]

    manifest = manifest[args.start :]
    if args.limit is not None:
        manifest = manifest[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "category",
        "n",
        "seeds",
        "valid",
        "compiled",
        "correctness",
        "runtime_custom_ms",
        "runtime_original_ms",
        "combined_score",
        "error",
    ]

    rows = []
    started = time.perf_counter()
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(manifest, 1):
            name = row["task"]
            baseline = dataset_root / "tasks" / name / "baseline.py"
            env = os.environ.copy()
            env.update(
                {
                    "ALGOTUNE_ROOT": str(algotune_root),
                    "ALGOTUNE_TASK": name,
                    "ALGOTUNE_N": str(row.get("n", 64)),
                    "ALGOTUNE_SEEDS": ",".join(
                        str(seed) for seed in row.get("seeds", [1])
                    ),
                    "ALGOTUNE_TIMING_RUNS": str(args.timing_runs),
                    "ALGOTUNE_WARMUPS": str(args.warmups),
                    "ALGOTUNE_SOLVE_TIMEOUT": str(args.solve_timeout),
                    "PYTHONPATH": os.pathsep.join(
                        [
                            str(REPO_ROOT),
                            str(algotune_root),
                            env.get("PYTHONPATH", ""),
                        ]
                    ),
                }
            )

            result = None
            error = ""
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "evaluate.py"),
                        "--solution",
                        str(baseline),
                        "--reference",
                        str(baseline),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    env=env,
                    cwd=str(HERE),
                )
                result = parse_result(proc.stdout)
                if result is None:
                    error = (proc.stderr or proc.stdout or "").strip()[-500:]
            except subprocess.TimeoutExpired:
                error = f"timeout after {args.timeout}s"
            except Exception as exc:  # noqa: BLE001 - one task must not kill the run
                error = f"{type(exc).__name__}: {exc}"

            if result is None:
                result = {}
            out = {
                "task": name,
                "category": row.get("category", ""),
                "n": row.get("n"),
                "seeds": ",".join(str(seed) for seed in row.get("seeds", [1])),
                "valid": result.get("valid"),
                "compiled": result.get("compiled"),
                "correctness": result.get("correctness"),
                "runtime_custom_ms": result.get("runtime_custom_ms"),
                "runtime_original_ms": result.get("runtime_original_ms"),
                "combined_score": result.get("combined_score"),
                "error": error or result.get("error", ""),
            }
            rows.append(out)
            writer.writerow(out)
            handle.flush()
            score = finite(out["combined_score"])
            print(
                f"[{index}/{len(manifest)}] {name:32s} "
                f"valid={out['valid']} correct={out['correctness']} "
                f"speedup={score if score is not None else 'n/a'}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    scores = [finite(row["combined_score"]) for row in rows]
    scores = [score for score in scores if score is not None]
    correct = sum(1 for row in rows if row["correctness"])
    print(
        f"\nBaseline: {len(rows)} tasks in {elapsed:.1f}s, "
        f"correct={correct}/{len(rows)}"
    )
    if scores:
        print(
            f"speedup mean={statistics.mean(scores):.4f} "
            f"median={statistics.median(scores):.4f} "
            f"min={min(scores):.4f} max={max(scores):.4f}"
        )
    failures = [row for row in rows if not row["correctness"]]
    if failures:
        print("Failures:")
        for row in failures:
            print(f"  {row['task']}: {row['error'][:200]}")
    print(f"Results: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
