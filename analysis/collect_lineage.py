#!/usr/bin/env python3
"""Reconstruct candidate parent/child lineage from an optimizer-agent run.

The agent saves each candidate's prompt in ``solutions/<run>/core_prompt.json``
and the candidate artifact in ``core_response.py`` / ``core_response_debug.py``.
The prompt contains the parent implementation under ``## Current Implementation``.
We match that code against previous candidates to recover the parent link.

Output columns:
    task, child_iteration, child_sample, child_path, child_score,
    child_valid, child_tokens, parent_iteration, parent_sample, parent_path,
    parent_score, delta, match

Example:
    python analysis/collect_lineage.py \
        --experiments-dir experiments/algotune_adv8_1794491181 \
        --version adv8
"""

import argparse
import csv
import difflib
import json
import re
from pathlib import Path

MARKERS = ("## Current Implementation", "## Current implementation")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--version", default="optagent")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated subset")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Minimum fuzzy match ratio when exact match fails.",
    )
    return parser.parse_args()


def norm(text):
    return re.sub(r"\s+", "", text or "")


def extract_current(sol_dir):
    prompt_path = sol_dir / "core_prompt.json"
    if not prompt_path.exists():
        return None
    try:
        messages = json.loads(prompt_path.read_text())
    except (OSError, ValueError):
        return None
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        for marker in MARKERS:
            index = content.find(marker)
            if index < 0:
                continue
            rest = content[index + len(marker) :]
            end = rest.find("\n## ")
            return (rest[:end] if end >= 0 else rest).strip()
    return None


def candidate_sources(sol_dir):
    sources = []
    for pattern in ("core_response.py", "core_response_debug.py"):
        path = sol_dir / pattern
        if path.exists():
            sources.append(path)
    return sources


def load_entries(task_dir, task_name):
    summary_path = task_dir / "summary.json"
    if not summary_path.exists():
        return []
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return []
    entries = []
    for item in summary.get("history") or []:
        path = item.get("path") or ""
        sol_name = Path(path).parent.name
        sol_dir = task_dir / "solutions" / sol_name
        if not sol_dir.exists():
            continue
        codes = []
        for source in candidate_sources(sol_dir):
            try:
                codes.append(norm(source.read_text()))
            except OSError:
                pass
        entries.append(
            {
                "task": task_name,
                "iteration": item.get("iteration"),
                "sample": item.get("sample"),
                "path": str(sol_dir / "core_response.py"),
                "dir": sol_dir,
                "scores": item.get("scores") or {},
                "score": item.get("score"),
                "correctness": (item.get("scores") or {}).get("correctness"),
                "tokens": (item.get("token_usage") or {}).get("total_tokens", 0),
                "codes": codes,
            }
        )
    return entries


def score_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    args = parse_args()
    root = Path(args.experiments_dir).resolve()
    output = (
        Path(args.output)
        if args.output
        else root / f"lineage_{args.version}.csv"
    )
    wanted = None
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}

    all_entries = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "summary.json").exists():
            continue
        if wanted and task_dir.name not in wanted:
            continue
        all_entries.extend(load_entries(task_dir, task_dir.name))

    rows = []
    for child in all_entries:
        current = extract_current(child["dir"])
        if current is None:
            continue
        current_norm = norm(current)
        best_parent = None
        best_ratio = 0.0
        for parent in all_entries:
            if parent is child or parent["task"] != child["task"]:
                continue
            if parent["iteration"] is None or child["iteration"] is None:
                continue
            if parent["iteration"] >= child["iteration"]:
                continue
            for code in parent["codes"]:
                if code == current_norm:
                    best_parent, best_ratio = parent, 1.0
                    break
                ratio = difflib.SequenceMatcher(None, current_norm, code).ratio()
                if ratio > best_ratio:
                    best_parent, best_ratio = parent, ratio
            if best_ratio == 1.0:
                break

        if child["iteration"] == 0:
            parent_iteration = -1
            parent_sample = None
            parent_path = "baseline"
            parent_score = None
            match = "baseline"
        elif best_parent is not None and best_ratio >= args.threshold:
            parent_iteration = best_parent["iteration"]
            parent_sample = best_parent["sample"]
            parent_path = str(best_parent["dir"])
            parent_score = score_value(best_parent["score"])
            match = "exact" if best_ratio == 1.0 else f"fuzzy:{best_ratio:.3f}"
        else:
            parent_iteration = None
            parent_sample = None
            parent_path = ""
            parent_score = None
            match = "none"

        child_score = score_value(child["score"])
        delta = (
            child_score - parent_score
            if child_score is not None and parent_score is not None
            else None
        )
        rows.append(
            {
                "task": child["task"],
                "child_iteration": child["iteration"],
                "child_sample": child["sample"],
                "child_path": str(child["dir"]),
                "child_score": child_score,
                "child_valid": child["correctness"],
                "child_tokens": child["tokens"],
                "parent_iteration": parent_iteration,
                "parent_sample": parent_sample,
                "parent_path": parent_path,
                "parent_score": parent_score,
                "delta": delta,
                "match": match,
            }
        )

    fieldnames = [
        "task",
        "child_iteration",
        "child_sample",
        "child_path",
        "child_score",
        "child_valid",
        "child_tokens",
        "parent_iteration",
        "parent_sample",
        "parent_path",
        "parent_score",
        "delta",
        "match",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(1 for row in rows if row["match"] not in ("none",))
    print(f"Wrote {len(rows)} lineage rows to {output}")
    print(f"Parent linked: {matched}/{len(rows)}")


if __name__ == "__main__":
    main()
