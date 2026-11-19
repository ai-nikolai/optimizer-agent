#!/usr/bin/env python3
"""Aggregate benchmark results with de-duplicated token accounting.

Several OpenEvolve runs wrote their token usage to a per-task
``token_usage.jsonl`` that is *appended* by every attempt, so the per-attempt
CSV rows hold cumulative totals. Summing those rows over-counts. This script
counts each task once:

* score    -> the first attempt (lowest shard index) for that task
* tokens   -> cumulative total divided by the number of attempts

Usage:
    python benchmarking/aggregate_results.py
    python benchmarking/aggregate_results.py --markdown
"""

import argparse
import ast
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AKD = Path("/hx2-weka/home/nr1713/automatic_knowledge_discovery/kernel_eng2")
KERNELBENCH = Path("/hx2-weka/home/nr1713/robust_kernelbench")


def _speedup(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def summarize(name, scores, tokens, calls, tasks_total, valid, done=None):
    scores = sorted(scores)
    done = len(valid) if done is None else done
    return {
        "name": name,
        "tasks": f"{done}/{tasks_total}",
        "valid": len(scores),
        "median": statistics.median(scores) if scores else None,
        "fast": sum(s > 1 for s in scores),
        "tokens_in": tokens["prompt_tokens"],
        "tokens_out": tokens["completion_tokens"],
        "tokens_total": tokens["total_tokens"],
        "calls": calls,
    }


def akd_summaries(name, run_dir, tasks_total=100):
    """optimizer-agent KernelBench runs (AKD kernel_eng2 experiment dirs)."""
    scores, tokens, calls = [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0
    valid = []
    for path in sorted(glob.glob(str(run_dir / "*/summary.json"))):
        summary = json.loads(Path(path).read_text())
        usage = summary.get("token_usage") or {}
        for key in tokens:
            tokens[key] += int(usage.get(key) or 0)
        calls += int(usage.get("calls") or 0)
        scores_block = (summary.get("best") or {}).get("scores") or {}
        if not scores_block.get("valid"):
            continue
        custom = scores_block.get("runtime_custom_ms")
        original = scores_block.get("runtime_original_ms")
        if not custom or not original:
            continue
        speedup = _speedup(original / custom)
        if speedup:
            scores.append(speedup)
            valid.append(Path(path).parent.name)
    return summarize(name, scores, tokens, calls, tasks_total, valid, done=len(glob.glob(str(run_dir / "*/summary.json"))))


def algotune_csv(name, pattern, tasks_total=154):
    """optimizer-agent AlgoTune driver CSVs: exactly one row per task."""
    scores, tokens, calls = [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0
    valid = []
    for path in sorted(glob.glob(pattern)):
        for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
            for key in tokens:
                tokens[key] += int(float(row.get(key) or 0))
            calls += int(float(row.get("llm_calls") or 0))
            speedup = _speedup(row.get("best_speedup"))
            if speedup:
                scores.append(speedup)
                valid.append(row.get("task"))
    return summarize(name, scores, tokens, calls, tasks_total, valid)


def openevolve_csv(name, pattern, key, tasks_total, cumulative_usage=True, scale=1.0):
    """OpenEvolve driver CSVs, de-duplicated to one attempt per task.

    ``cumulative_usage`` is set when the per-task usage log is appended across
    attempts (AlgoTune), in which case the last row holds the sum of all
    attempts and is divided by the attempt count.
    """
    attempts = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        shard = int(path.rsplit("_shard", 1)[-1].split("/")[0]) if "_shard" in path else 0
        for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
            task = row.get(key) or row.get("task")
            attempts[task].append((shard, row))

    scores, tokens, calls = [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0
    valid, spent_total = [], 0
    for task, rows in attempts.items():
        rows.sort(key=lambda item: item[0])
        if cumulative_usage:
            counts = len(rows)
            for column in tokens:
                tokens[column] += max(float(row.get(column) or 0) for _, row in rows) / counts
            calls += max(float(row.get("llm_calls") or 0) for _, row in rows) / counts
            spent_total += max(float(row.get("total_tokens") or 0) for _, row in rows)
        else:
            for column in tokens:
                tokens[column] += sum(float(row.get(column) or 0) for _, row in rows)
            calls += sum(float(row.get("llm_calls") or 0) for _, row in rows)
        for _, row in rows:
            if row.get("status") != "ok":
                continue
            speedup = _speedup(row.get("best_speedup"))
            if speedup:
                scores.append(speedup)
                valid.append(task)
                break

    result = summarize(name, scores, tokens, calls, tasks_total, valid)
    result["tokens_all_attempts"] = spent_total * scale
    result["attempts"] = len(attempts)
    return result


def v11_summary(name, run_dir, tasks_total=100):
    best, tokens_in, tokens_out = {}, 0, 0
    for trial in sorted(run_dir.glob("trial_*")):
        evals = trial / "evaluations.csv"
        gens = trial / "generations.csv"
        if evals.exists():
            for row in csv.DictReader(open(evals, newline="", encoding="utf-8")):
                if str(row.get("correctness")).lower() != "true":
                    continue
                try:
                    runtime = float(row.get("runtime"))
                    reference = float(row.get("runtime_original"))
                except (TypeError, ValueError):
                    continue
                if runtime > 0 and reference > 0:
                    best[row["problem_id"]] = max(best.get(row["problem_id"], 0), reference / runtime)
        if gens.exists():
            for row in csv.DictReader(open(gens, newline="", encoding="utf-8")):
                tokens_in += int(float(row.get("in_tokens") or 0))
                try:
                    parsed = ast.literal_eval(row.get("out_tokens") or "{}")
                    tokens_out += int(float((parsed or {}).get("mean") or 0))
                except (TypeError, ValueError, SyntaxError):
                    pass
    tokens = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out}
    return summarize(name, list(best.values()), tokens, 0, tasks_total, list(best))


def fmt(value, digits=3):
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return "—" if value is None else str(value)


def render(rows):
    header = "| Run | Tasks | Valid | Median | fast@1 | Tokens in | Tokens out | Tokens total | Calls |"
    line = "|---|---|---|---|---|---|---|---|---|"
    out = [header, line]
    for row in rows:
        out.append(
            f"| {row['name']} | {row['tasks']} | {row['valid']} | {fmt(row['median'])} | "
            f"{row['fast']} | {fmt(row['tokens_in'], 0)} | {fmt(row['tokens_out'], 0)} | "
            f"{fmt(float(row['tokens_total']), 0)} | {fmt(float(row['calls']), 0)} |"
        )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="Only print the table")
    args = parser.parse_args()

    rows = [
        akd_summaries("AKD KernelBench sequential", AKD / "experiment_v5_1_robust_tokens_optagent"),
        akd_summaries("AKD KernelBench advanced", AKD / "experiment_v5_1_newsettings_tokens_optagent"),
        algotune_csv("AlgoTune sequential 10 rounds", str(REPO / "experiments/algotune_full_1689335_shard*/results.csv")),
        algotune_csv("AlgoTune advanced 4 rounds", str(REPO / "experiments/algotune_full_1689334_shard*/results.csv")),
        openevolve_csv(
            "OpenEvolve KernelBench",
            str(REPO / "experiments/openevolve_kernelbench_full_1670750_shard*/results.csv"),
            "problem_id",
            100,
            cumulative_usage=False,
        ),
        openevolve_csv(
            "OpenEvolve AlgoTune",
            str(REPO / "experiments/openevolve_algotune_full_1689375_shard*/results.csv"),
            "task",
            154,
        ),
        v11_summary("robust_kernelbench v11 (orig)", KERNELBENCH / "experiments/exp_local_L1_v11_0_Qwen3_Coder_30B_A3B_Instruct_kernelbench_API1"),
        v11_summary("robust_kernelbench v11 fair", KERNELBENCH / "experiments/exp_local_L1_v11_1_fair_Qwen3_Coder_30B_A3B_Instruct_kernelbench_API1"),
    ]
    print(render(rows))
    if not args.markdown:
        print()
        for row in rows:
            if row.get("tokens_all_attempts"):
                print(
                    f"* {row['name']}: table counts one attempt per task; "
                    f"all attempts together spent {row['tokens_all_attempts']:,.0f} tokens "
                    f"across {row['attempts']} tasks"
                )


if __name__ == "__main__":
    main()
