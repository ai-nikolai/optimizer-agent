#!/usr/bin/env python3
"""Collect optimizer-agent experiment summaries into analysis CSVs.

Scans an experiments directory for ``summary.json`` files, flattens each
history entry into one row, then writes the files consumed by
``analysis/interactive.py``:

    experiment_results_<version>.csv
    experiment_results_summary_deb_<version>.csv
    experiment_results_summary_<version>.csv
    experiment_results_summary_agg_<version>.csv

The layout works for both optimizer-agent run layouts:

    experiments/<run>/P12/summary.json          # kernel_eng2 style
    experiments/<run>/problem_pid_12/summary.json
    experiments/<run>/level_1_pid_12/summary.json

Example:
    python analysis/collect_results.py \
        --experiments-dir experiments \
        --version v5_0
    python analysis/interactive.py \
        --data-dir experiments \
        --version v5_0
"""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

METHOD_FIELDS = (
    ("predecessor_selection_method", "met"),
    ("num_islands", "is"),
    ("num_samples", "samp"),
    ("num_map_elites", "el"),
    ("use_strategy", "strat"),
    ("use_feedback", "feed"),
)
SCORE_COLUMNS = [
    "compiled",
    "loading_success",
    "correctness",
    "runtime_custom_ms",
    "runtime_original_ms",
    "combined_score",
    "fast_1",
    "fast_2",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Root containing run directories with summary.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write CSVs (default: the experiments directory).",
    )
    parser.add_argument(
        "--version",
        default="optagent",
        help="Suffix for the generated CSVs (default: optagent).",
    )
    return parser.parse_args()


def parse_metadata(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().rstrip("=")
    return values


def method_name(summary_path):
    parent = summary_path.parent
    metadata = parse_metadata(parent / "metadata.txt")

    # Fall back to config.json when metadata.txt is missing.
    if not any(key in metadata for key, _ in METHOD_FIELDS):
        config_path = parent / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                for key, _ in METHOD_FIELDS:
                    if key in config and key not in metadata:
                        metadata[key] = str(config[key])
            except (OSError, ValueError):
                pass

    parts = []
    for key, short in METHOD_FIELDS:
        if key in metadata and metadata[key] != "":
            parts.append(f"{short}={metadata[key]}")
    if parts:
        return "__".join(parts)
    return parent.parent.name or "unknown"


def problem_id(summary_path):
    name = summary_path.parent.name
    match = re.search(r"[Pp](\d+)$", name) or re.search(r"pid[_-]?(\d+)", name)
    return match.group(1) if match else name


def finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def collect_rows(experiments_dir):
    rows = []
    summaries = sorted(experiments_dir.rglob("summary.json"))
    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"skipping unreadable {summary_path}: {exc}")
            continue

        history = summary.get("history") or []
        if not history:
            continue

        name = method_name(summary_path)
        pid = problem_id(summary_path)
        for entry in history:
            scores = entry.get("scores") or {}
            row = {
                "method_name": name,
                "problem_id": pid,
                "iteration": entry.get("iteration"),
                "island": entry.get("island"),
                "sample": entry.get("sample"),
                "compiled": scores.get("compiled"),
                "loading_success": scores.get("loading_success"),
                "correctness": scores.get("correctness"),
                "runtime_custom_ms": scores.get("runtime_custom_ms"),
                "runtime_original_ms": scores.get("runtime_original_ms"),
                "combined_score": scores.get("combined_score"),
            }
            rows.append(row)
    return rows, summaries


def build_tables(rows, version, output_dir):
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no summary history found under the experiments directory")

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.dropna(subset=["iteration"]).copy()
    df["iteration"] = df["iteration"].astype(int)

    for column in ("compiled", "loading_success", "correctness"):
        df[column] = df[column].fillna(False).astype(bool).astype(int)
    for column in ("runtime_custom_ms", "runtime_original_ms", "combined_score"):
        df[column] = df[column].apply(finite)

    df["fast_1"] = (
        (df["correctness"] == 1)
        & (df["runtime_custom_ms"] <= df["runtime_original_ms"])
    ).astype(float)
    df["fast_2"] = (
        (df["correctness"] == 1)
        & (2.0 * df["runtime_custom_ms"] <= df["runtime_original_ms"])
    ).astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"experiment_results_{version}.csv", index=False)

    # Best sample/island per problem and iteration.
    per_problem = (
        df.groupby(["method_name", "iteration", "problem_id"])[SCORE_COLUMNS]
        .max()
        .reset_index()
    )
    per_problem.to_csv(
        output_dir / f"experiment_results_summary_deb_{version}.csv", index=False
    )

    counts = (
        per_problem.groupby(["method_name", "iteration"])["problem_id"]
        .nunique()
        .reset_index(name="num_problems")
    )
    grouped = (
        per_problem.groupby(["method_name", "iteration"])[SCORE_COLUMNS]
        .sum()
        .reset_index()
        .merge(counts, on=["method_name", "iteration"], how="left")
    )
    for column in SCORE_COLUMNS:
        grouped[column] = grouped[column] / grouped["num_problems"]
    grouped["new_combined_score"] = (
        grouped["runtime_original_ms"] - grouped["runtime_custom_ms"]
    )
    grouped.drop(columns=["num_problems"]).to_csv(
        output_dir / f"experiment_results_summary_{version}.csv", index=False
    )

    cumulative = grouped.sort_values(["method_name", "iteration"]).copy()
    columns = SCORE_COLUMNS + ["new_combined_score"]
    cumulative[columns] = cumulative.groupby("method_name")[columns].cummax()
    cumulative.drop(columns=["num_problems"]).to_csv(
        output_dir / f"experiment_results_summary_agg_{version}.csv", index=False
    )

    return len(rows), df["method_name"].nunique()


def main():
    args = parse_args()
    experiments_dir = Path(args.experiments_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else experiments_dir
    rows, summaries = collect_rows(experiments_dir)
    n_rows, n_methods = build_tables(rows, args.version, output_dir)
    print(
        f"Read {len(summaries)} summary.json files, wrote {n_rows} rows "
        f"for {n_methods} method(s) to {output_dir}"
    )


if __name__ == "__main__":
    main()
