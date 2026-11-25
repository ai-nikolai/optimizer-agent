import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The shared evaluator CLI. Every KernelBench agent (recursive, optimizer-agent,
# OpenEvolve) evaluates through this file so scores are identical.
# Generated task dirs live at .../benchmarking/agents/openevolve/kernelbench/
# tasks/level_<L>/pid_<N>/evaluator.py; five levels up is benchmarking/.
EVALUATOR = Path(
    os.environ.get(
        "KERNELBENCH_EVALUATOR",
        str(HERE.parents[5] / "tasks" / "kernelbench" / "evaluate.py"),
    )
)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def evaluate(program_path: str) -> dict:
    """Evaluate ``program_path`` with the shared KernelBench evaluator."""
    reference = HERE / "reference.py"
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.pathsep.join([str(EVALUATOR.parent), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run(
        [sys.executable, str(EVALUATOR), "--solution", str(program_path), "--reference", str(reference)],
        capture_output=True,
        text=True,
        env=env,
        timeout=float(os.environ.get("KERNELBENCH_EVAL_TIMEOUT", 3600)),
    )
    marker = "=== Evaluation Result ==="
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
