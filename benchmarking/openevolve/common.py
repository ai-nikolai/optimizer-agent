"""Shared helpers for the OpenEvolve benchmark drivers."""

import json
from pathlib import Path


def read_usage(path):
    """Sum token usage records written by the patched OpenEvolve LLM client."""
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
    """Read OpenEvolve's ``best_program_info.json`` and normalize the score."""
    best_dir = Path(task_dir) / "openevolve_output" / "best"
    info_path = best_dir / "best_program_info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text())
    except ValueError:
        return None
    evaluation = info.get("evaluation_result") or {}
    metrics = info.get("metrics") or {}
    return {
        "best_score": info.get(
            "score", metrics.get("combined_score", evaluation.get("combined_score"))
        ),
        "best_speedup": info.get(
            "speedup_score",
            metrics.get(
                "speedup_score",
                evaluation.get("speedup_score", evaluation.get("combined_score")),
            ),
        ),
        "best_path": str(best_dir / "best_program.py"),
        "info": info,
    }
