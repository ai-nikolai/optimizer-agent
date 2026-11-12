#!/usr/bin/env python3
"""Run optimizer-agent over a KernelBench level, one problem at a time.

This is the batch driver that makes optimizer-agent behave like
robust_kernelbench's test-time-scaling loop:

    round 1: generate from the reference PyTorch code
    round N: repair the *previous* round's code using its compile/run/
             correctness feedback, regardless of whether it scored better

The agent itself is unchanged; the KernelBench-specific pieces are the
evaluator adapter and the prompt/feedback JSON in this directory.

Example (model server already running on port 30000):

    python examples/kernelbench/run_level.py \
        --level 1 --rounds 3 \
        --model-url http://localhost:30000 \
        --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --experiments-dir experiments/opt_agent_kb_level1

For parallel GPUs, run several shards (each with its own CUDA_VISIBLE_DEVICES):

    --num-shards 4 --shard-index 0
    --num-shards 4 --shard-index 1
    ...
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

ROBUST_KB_ROOT = Path(
    os.environ.get(
        "ROBUST_KERNELBENCH_ROOT",
        "/hx2-weka/home/nr1713/robust_kernelbench",
    )
)
if str(ROBUST_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBUST_KB_ROOT))

from optimizer_agent.main import Agent  # noqa: E402


def _slug(text):
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_")
    return text[:80] or "problem"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3, help="Repair rounds per problem")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--filter", action="store_true", help="KernelBench representative subset")
    parser.add_argument("--baseline-dir", type=str, default=str(REPO_ROOT / "problems"))
    parser.add_argument(
        "--experiments-dir", type=str, default=str(REPO_ROOT / "experiments/opt_agent_kb")
    )
    parser.add_argument("--prompt-config", type=str, default=str(HERE / "prompt_config.json"))
    parser.add_argument(
        "--feedback-config",
        type=str,
        default=str(REPO_ROOT / "examples/feedback/kernel_rules_llm.json"),
    )
    parser.add_argument("--evaluate", type=str, default=str(HERE / "evaluate.py"))
    parser.add_argument("--model-url", type=str, default="http://0.0.0.0:30000")
    parser.add_argument("--model-name", type=str, default="default")
    parser.add_argument("--model-api-key", type=str, default=None)
    parser.add_argument("--model-params", type=json.loads, default={})
    parser.add_argument("--use-strategy", action="store_true", default=None)
    parser.add_argument("--use-debug", action="store_true", default=None)
    parser.add_argument("--num-debug-iterations", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--minimize", action="store_true", default=None)
    parser.add_argument("--diff-based", action="store_true", default=None)
    parser.add_argument("--diff-mode", type=str, default=None, choices=["unified-diff", "json"])
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=None,
        help="Resume completed problems (default: on).",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Rerun completed problems.",
    )
    parser.add_argument(
        "--predecessor-selection-method",
        type=str,
        default=None,
        choices=["bestN_last", "bestN_all", "map_elites"],
        help="Override the task config selection method.",
    )
    parser.add_argument(
        "--feedback-strategy",
        type=str,
        default=None,
        choices=["none", "prompt", "rules", "rules_llm", "llm", "evaluator"],
        help="Override the task feedback strategy.",
    )
    parser.add_argument(
        "--feedback-profile",
        type=str,
        default=None,
        choices=["generic", "kernel"],
        help="Override the feedback wording profile.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from robust_kernelbench.inference_common import load_dataset_for_level

    dataset = load_dataset_for_level(args.level, use_filter=args.filter)
    rows = list(dataset)
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    baseline_dir = Path(args.baseline_dir) / f"level_{args.level}"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    results_path = experiments_dir / f"results_shard{args.shard_index}.csv"
    fieldnames = [
        "problem_id",
        "name",
        "rounds",
        "best_score",
        "best_path",
        "scores",
        "stages",
        "statuses",
        "error",
    ]
    with open(results_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(rows):
            problem_id = row.get("problem_id")
            name = row.get("name") or f"problem_{problem_id}"
            run_dir = experiments_dir / f"level_{args.level}_pid_{problem_id}"
            print(
                f"[{index + 1}/{len(rows)}] pid={problem_id} name={name} -> {run_dir}",
                flush=True,
            )
            try:
                problem_path = baseline_dir / f"{problem_id}_{_slug(name)}.py"
                problem_path.write_text(row["code"])

                agent = Agent(
                    num_iterations=args.rounds,
                    num_samples=1,
                    # bestN_last + num_samples=1 repairs the previous round.
                    predecessor_selection_method=args.predecessor_selection_method,
                    max_candidates=args.max_candidates,
                    minimize=args.minimize,
                    diff_based=args.diff_based,
                    diff_mode=args.diff_mode,
                    prompt_config_file=args.prompt_config,
                    feedback_config_file=args.feedback_config,
                    feedback_strategy=args.feedback_strategy,
                    feedback_profile=args.feedback_profile,
                    model_name=args.model_name,
                    model_url=args.model_url,
                    model_api_key=args.model_api_key,
                    model_params=args.model_params,
                    use_strategy=args.use_strategy,
                    use_debug=args.use_debug,
                    num_debug_iterations=args.num_debug_iterations,
                    eval_timeout=args.eval_timeout,
                    resume=args.resume,
                    overwrite=args.overwrite,
                )
                history = agent.run(
                    baseline=str(problem_path),
                    evaluate=args.evaluate,
                    reference=str(problem_path),
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
                    "problem_id": problem_id,
                    "name": name,
                    "rounds": len(history),
                    "best_score": best.get("score"),
                    "best_path": best.get("path", ""),
                    "scores": json.dumps([item["score"] for item in history]),
                    "stages": json.dumps([item.get("failure_stage") for item in history]),
                    "statuses": json.dumps([item.get("status") for item in history]),
                    "error": "",
                }
            except Exception as exc:  # noqa: BLE001 - one problem must not kill the shard
                print(f"FAILED pid={problem_id}: {exc}", flush=True)
                row_data = {
                    "problem_id": problem_id,
                    "name": name,
                    "rounds": 0,
                    "best_score": "",
                    "best_path": "",
                    "scores": "[]",
                    "stages": "[]",
                    "statuses": "[]",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            writer.writerow(row_data)
            handle.flush()

    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
