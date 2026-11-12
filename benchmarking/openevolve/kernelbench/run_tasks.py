#!/usr/bin/env python3
"""Run OpenEvolve over prepared KernelBench tasks.

Each task directory contains the reference problem, an initial ``ModelNew``,
the robust_kernelbench evaluator, and an OpenEvolve config. This driver runs
them sequentially against the local vLLM server and writes a comparison CSV.

Example:
    python benchmarking/openevolve/kernelbench/run_tasks.py \
        --tasks-dir benchmarking/openevolve/kernelbench/tasks/level_1 \
        --iterations 50 \
        --output experiments/openevolve_kernelbench/results.csv
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OPENEVOLVE_ROOT = Path("/hx2-weka/home/nr1713/openevolve")
DEFAULT_ROBUST_KB = "/hx2-weka/home/nr1713/robust_kernelbench"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-dir", default=str(HERE / "tasks" / "level_1")
    )
    parser.add_argument("--problem-ids", default=None, help="Comma-separated IDs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--openevolve-root", default=str(DEFAULT_OPENEVOLVE_ROOT))
    parser.add_argument("--robust-kb-root", default=DEFAULT_ROBUST_KB)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "experiments/openevolve_kernelbench/results.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tasks(tasks_dir, problem_ids):
    tasks = []
    for path in sorted(tasks_dir.iterdir()):
        if not path.is_dir() or not (path / "meta.json").exists():
            continue
        meta = json.loads((path / "meta.json").read_text())
        if problem_ids and int(meta["problem_id"]) not in problem_ids:
            continue
        tasks.append((path, meta))
    tasks.sort(key=lambda item: int(item[1]["problem_id"]))
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
    }


def main():
    args = parse_args()
    tasks_dir = Path(args.tasks_dir)
    problem_ids = None
    if args.problem_ids:
        problem_ids = {
            int(part) for part in args.problem_ids.split(",") if part.strip()
        }
    tasks = load_tasks(tasks_dir, problem_ids)
    tasks = tasks[args.start :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit(f"no OpenEvolve KernelBench tasks found under {tasks_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem_id",
        "name",
        "level",
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
            "ROBUST_KERNELBENCH_ROOT": args.robust_kb_root,
            "OPENAI_API_KEY": env.get("OPENAI_API_KEY", "dummy"),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(Path(args.openevolve_root).resolve()),
                    str(REPO_ROOT),
                    args.robust_kb_root,
                    env.get("PYTHONPATH", ""),
                ]
            ),
        }
    )

    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (task_dir, meta) in enumerate(tasks, 1):
            pid = int(meta["problem_id"])
            name = meta.get("name", f"problem_{pid}")
            if args.overwrite:
                out_dir = task_dir / "openevolve_output"
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                usage_path = task_dir / "token_usage.jsonl"
                if usage_path.exists():
                    usage_path.unlink()
            started = time.perf_counter()
            status = "ok"
            error = ""
            cmd = [
                sys.executable,
                str(Path(args.openevolve_root).resolve() / "openevolve-run.py"),
                "initial_program.py",
                "evaluator.py",
                "--config",
                "config.yaml",
                "--iterations",
                str(args.iterations),
                "--output",
                "openevolve_output",
            ]
            print(
                f"[{index}/{len(tasks)}] pid={pid} name={name} -> {task_dir}",
                flush=True,
            )
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
            except Exception as exc:  # noqa: BLE001 - one problem must not kill the suite
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
                "problem_id": pid,
                "name": name,
                "level": meta.get("level"),
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
                f"[{index}/{len(tasks)}] pid={pid} status={status} "
                f"speedup={row['best_speedup']} runtime={runtime:.0f}s",
                flush=True,
            )

    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
