#!/usr/bin/env python3
"""Run optimizer-agent over the AlgoTune task manifest.

Each task has a baseline ``Solver`` that delegates to the official AlgoTune
reference implementation. The evaluator imports the official task, checks
correctness, and compares candidate/reference runtimes. The agent then tries to
replace the baseline with a faster solver.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_ALGOTUNE_ROOT = os.environ.get(
    "ALGOTUNE_ROOT", "/hx2-weka/home/nr1713/AlgoTune"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(HERE / "dataset/manifest.json"))
    parser.add_argument("--tasks", default=None, help="Comma-separated task names")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument(
        "--selection",
        default="bestN_last",
        choices=["bestN_last", "bestN_all", "map_elites"],
    )
    parser.add_argument(
        "--map-elites-method",
        default="cluster",
        choices=["cluster", "fixed", "dynamic"],
    )
    parser.add_argument("--num-map-elites", type=int, default=1)
    parser.add_argument("--feedback-strategy", default="rules_llm")
    parser.add_argument("--feedback-profile", default="generic")
    parser.add_argument("--use-strategy", action="store_true", default=None)
    parser.add_argument("--use-debug", action="store_true", default=None)
    parser.add_argument("--num-debug-iterations", type=int, default=1)
    parser.add_argument("--algotune-root", default=DEFAULT_ALGOTUNE_ROOT)
    parser.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments/algotune"))
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--model-url", default="http://0.0.0.0:30000")
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--model-api-key", default=None)
    parser.add_argument("--model-params", type=json.loads, default={})
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--solve-timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=None)
    parser.add_argument("--test", action="store_true", help="Use a fake LLM server")
    return parser.parse_args()


def _slug(text):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_")


def main():
    args = parse_args()
    from optimizer_agent.main import Agent

    manifest = json.loads(Path(args.manifest).read_text())
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        manifest = [row for row in manifest if row["task"] in wanted]
    manifest = manifest[args.start :]
    if args.limit is not None:
        manifest = manifest[: args.limit]

    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.results_csv) if args.results_csv else experiments_dir / "results.csv"
    fieldnames = [
        "task",
        "category",
        "n",
        "rounds",
        "best_score",
        "best_speedup",
        "best_path",
        "scores",
        "stages",
        "statuses",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(manifest, 1):
            name = row["task"]
            task_dir = HERE / "dataset" / "tasks" / name
            baseline = task_dir / "baseline.py"
            reference = task_dir / "reference.txt"
            description = (task_dir / "description.txt").read_text()
            run_dir = experiments_dir / name
            print(
                f"[{index}/{len(manifest)}] {name} n={row.get('n')} "
                f"-> {run_dir}",
                flush=True,
            )

            # The evaluator and baseline both need the official task package.
            os.environ["ALGOTUNE_ROOT"] = str(Path(args.algotune_root).resolve())
            os.environ["ALGOTUNE_TASK"] = name
            os.environ["ALGOTUNE_N"] = str(row.get("n", 64))
            os.environ["ALGOTUNE_SEEDS"] = ",".join(
                str(seed) for seed in row.get("seeds", [1])
            )
            os.environ["ALGOTUNE_TIMING_RUNS"] = str(args.timing_runs)
            os.environ["ALGOTUNE_WARMUPS"] = str(args.warmups)
            os.environ["ALGOTUNE_SOLVE_TIMEOUT"] = str(args.solve_timeout)
            sys.path.insert(0, os.environ["ALGOTUNE_ROOT"])

            try:
                agent = Agent(
                    num_iterations=args.rounds,
                    num_samples=args.samples,
                    max_candidates=args.max_candidates,
                    predecessor_selection_method=args.selection,
                    num_map_elites=args.num_map_elites,
                    map_elites_method=args.map_elites_method,
                    prompt_config_file=str(HERE / "prompt_config.json"),
                    feedback_config_file=str(HERE / "feedback_config.json"),
                    feedback_strategy=args.feedback_strategy,
                    feedback_profile=args.feedback_profile,
                    response_format="code",
                    artifact_extension="py",
                    score_key="combined_score",
                    task_description=description,
                    model_name=args.model_name,
                    model_url=args.model_url,
                    model_api_key=args.model_api_key,
                    model_params=args.model_params,
                    eval_timeout=args.eval_timeout,
                    use_strategy=args.use_strategy,
                    use_debug=args.use_debug,
                    num_debug_iterations=args.num_debug_iterations,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    test=args.test,
                )
                history = agent.run(
                    baseline=str(baseline),
                    evaluate=str(HERE / "evaluate.py"),
                    reference=str(reference),
                    experiments_dir=str(run_dir),
                )
                best = (
                    max(
                        history,
                        key=lambda item: item.get(
                            "rank_score", item.get("score", float("-inf"))
                        ),
                    )
                    if history
                    else {}
                )
                row_data = {
                    "task": name,
                    "category": row.get("category", ""),
                    "n": row.get("n"),
                    "rounds": len(history),
                    "best_score": best.get("score"),
                    "best_speedup": best.get("score"),
                    "best_path": best.get("path", ""),
                    "scores": json.dumps([item.get("score") for item in history]),
                    "stages": json.dumps([item.get("failure_stage") for item in history]),
                    "statuses": json.dumps([item.get("status") for item in history]),
                    "error": "",
                }
            except Exception as exc:  # noqa: BLE001 - one task must not kill the suite
                print(f"FAILED {name}: {exc}", flush=True)
                row_data = {
                    "task": name,
                    "category": row.get("category", ""),
                    "n": row.get("n"),
                    "rounds": 0,
                    "best_score": "",
                    "best_speedup": "",
                    "best_path": "",
                    "scores": "[]",
                    "stages": "[]",
                    "statuses": "[]",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            writer.writerow(row_data)
            csv_handle.flush()

    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
