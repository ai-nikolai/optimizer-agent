#!/usr/bin/env python3
"""Run OpenEvolve over the prepared AlgoTune tasks.

Each task directory contains OpenEvolve's own ``initial_program.py``,
``evaluator.py`` and ``config.yaml``. This driver runs them sequentially with
the local vLLM server and writes a comparison CSV.

Example:
    python benchmarking/openevolve/algotune/run_tasks.py \
        --iterations 100 \
        --output experiments/openevolve_algotune/results.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OPENEVOLVE_ROOT = Path("/hx2-weka/home/nr1713/openevolve")
DEFAULT_ALGOTUNE_ROOT = "/hx2-weka/home/nr1713/AlgoTune"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", default=str(HERE / "tasks"))
    parser.add_argument("--tasks", default=None, help="Comma-separated subset")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--openevolve-root", default=str(DEFAULT_OPENEVOLVE_ROOT))
    parser.add_argument("--algotune-root", default=DEFAULT_ALGOTUNE_ROOT)
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "experiments/openevolve_algotune/results.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_tasks(tasks_dir, wanted):
    tasks = []
    for path in sorted(tasks_dir.iterdir()):
        if not path.is_dir():
            continue
        if wanted and path.name not in wanted:
            continue
        if all((path / name).exists() for name in ("initial_program.py", "evaluator.py", "config.yaml")):
            tasks.append(path)
    return tasks


def read_usage(path):
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    if not Path(path).exists():
        return totals
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        for key in totals:
            totals[key] += int(record.get(key, 0) or 0)
    return totals


def parse_best(task_dir):
    best_dir = task_dir / "openevolve_output" / "best"
    info_path = best_dir / "best_program_info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text())
    except ValueError:
        return None
    evaluation = info.get("evaluation_result") or {}
    return {
        "best_score": info.get("score", evaluation.get("combined_score")),
        "best_speedup": info.get(
            "speedup_score", evaluation.get("speedup_score", evaluation.get("combined_score"))
        ),
        "best_path": str(best_dir / "best_program.py"),
        "info": info,
    }


def main():
    args = parse_args()
    tasks_dir = Path(args.tasks_dir)
    openevolve_root = Path(args.openevolve_root).resolve()
    wanted = None
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
    tasks = discover_tasks(tasks_dir, wanted)
    if not tasks:
        raise SystemExit(f"no OpenEvolve tasks found under {tasks_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "n",
        "iterations",
        "status",
        "best_score",
        "best_speedup",
        "runtime_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_calls",
        "best_path",
        "error",
    ]

    env = os.environ.copy()
    env.update(
        {
            "ALGOTUNE_PATH": args.algotune_root,
            "OPENAI_API_KEY": env.get("OPENAI_API_KEY", "dummy"),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(openevolve_root),
                    str(REPO_ROOT),
                    args.algotune_root,
                    env.get("PYTHONPATH", ""),
                ]
            ),
        }
    )

    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, task_dir in enumerate(tasks, 1):
            name = task_dir.name
            meta_path = task_dir / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if args.overwrite:
                out_dir = task_dir / "openevolve_output"
                if out_dir.exists():
                    import shutil

                    shutil.rmtree(out_dir)
                usage_path = task_dir / "token_usage.jsonl"
                if usage_path.exists():
                    usage_path.unlink()
            started = time.perf_counter()
            error = ""
            status = "ok"
            cmd = [
                sys.executable,
                str(openevolve_root / "openevolve-run.py"),
                "initial_program.py",
                "evaluator.py",
                "--config",
                "config.yaml",
                "--iterations",
                str(args.iterations),
                "--output",
                "openevolve_output",
            ]
            print(f"[{index}/{len(tasks)}] {name} n={meta.get('n')} -> {task_dir}", flush=True)
            task_env = env.copy()
            task_env["OPENEVOLVE_USAGE_LOG"] = str(task_dir / "token_usage.jsonl")
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(task_dir),
                    env=task_env,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                if proc.returncode != 0:
                    status = "failed"
                    error = (proc.stderr or proc.stdout or "").strip()[-1000:]
            except subprocess.TimeoutExpired:
                status = "timeout"
                error = f"timeout after {args.timeout}s"
            except Exception as exc:  # noqa: BLE001 - one task must not kill the suite
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

            runtime = time.perf_counter() - started
            best = parse_best(task_dir) if status == "ok" else None
            if status == "ok" and best is None:
                status = "no_result"
                error = "openevolve_output/best/best_program_info.json not found"
            best = best or {}
            usage = read_usage(task_dir / "token_usage.jsonl")
            row = {
                "task": name,
                "n": meta.get("n"),
                "iterations": args.iterations,
                "status": status,
                "best_score": best.get("best_score"),
                "best_speedup": best.get("best_speedup"),
                "runtime_seconds": f"{runtime:.1f}",
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "llm_calls": usage["calls"],
                "best_path": best.get("best_path", ""),
                "error": error,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"[{index}/{len(tasks)}] {name} status={status} "
                f"speedup={row['best_speedup']} runtime={runtime:.0f}s",
                flush=True,
            )

    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
