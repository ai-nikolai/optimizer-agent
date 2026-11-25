"""OpenEvolve evaluator adapter for AlgoTune.

Every AlgoTune agent (optimizer-agent, OpenEvolve) is scored by the same CLI,
``benchmarking/tasks/algotune/evaluate.py``, so trials, seeds, tolerances and
the error message are identical across agents.

This file is copied into each prepared task directory. ``meta.json`` supplies
the official task name, size ``n`` and seeds.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_evaluator() -> Path:
    override = os.environ.get("ALGOTUNE_EVALUATOR")
    if override:
        return Path(override)
    for parent in [HERE, *HERE.parents]:
        for candidate in (
            parent / "tasks" / "algotune" / "evaluate.py",
            parent / "benchmarking" / "tasks" / "algotune" / "evaluate.py",
        ):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "could not locate benchmarking/tasks/algotune/evaluate.py; "
        "set ALGOTUNE_EVALUATOR"
    )


def evaluate(program_path: str) -> dict:
    """Score ``program_path`` with the shared AlgoTune evaluator."""
    meta_path = HERE / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    seeds = meta.get("seeds") or [1, 2]

    env = os.environ.copy()
    env["ALGOTUNE_TASK"] = str(meta.get("task") or HERE.name)
    if meta.get("n") is not None:
        env["ALGOTUNE_N"] = str(meta["n"])
    env["ALGOTUNE_SEEDS"] = ",".join(str(seed) for seed in seeds)
    if not env.get("ALGOTUNE_ROOT") and env.get("ALGOTUNE_PATH"):
        env["ALGOTUNE_ROOT"] = env["ALGOTUNE_PATH"]

    proc = subprocess.run(
        [sys.executable, str(_find_evaluator()), "--solution", str(program_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=float(os.environ.get("ALGOTUNE_EVAL_TIMEOUT", 3600)),
    )

    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "combined_score" in result:
            return result

    return {
        "combined_score": 0.0,
        "valid": False,
        "compiled": False,
        "correctness": False,
        "error": (proc.stderr or proc.stdout or "evaluator produced no result")[-1000:],
    }
