#!/usr/bin/env python3
"""AlgoTune evaluator adapter for optimizer-agent.

The candidate artifact must define ``class Solver`` with a
``solve(problem, **kwargs)`` method. The reference implementation is the
official AlgoTune task's ``solve`` method.

The driver sets ALGOTUNE_TASK, ALGOTUNE_N, ALGOTUNE_SEEDS and
ALGOTUNE_ROOT in the environment. This evaluator can also be called directly:

    ALGOTUNE_TASK=affine_transform_2d ALGOTUNE_N=256 ALGOTUNE_SEEDS=1,2 \
    ALGOTUNE_ROOT=/path/to/AlgoTune \
    python evaluate.py --solution candidate.py
"""

import argparse
import importlib.util
import json
import math
import os
import signal
import statistics
import sys
import time
import traceback
import uuid
from pathlib import Path

MARKER = "=== Evaluation Result ==="


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _parse_seeds(value):
    return [int(part.strip()) for part in str(value or "1").split(",") if part.strip()]


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class TimeoutError_(RuntimeError):
    pass


def _with_timeout(seconds, func, *args, **kwargs):
    if seconds is None or seconds <= 0:
        return func(*args, **kwargs)

    def _handler(signum, frame):
        raise TimeoutError_(f"evaluation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _load_solver(path, timeout):
    spec = importlib.util.spec_from_file_location(
        f"candidate_solver_{uuid.uuid4().hex}", path
    )
    module = importlib.util.module_from_spec(spec)

    def _load():
        spec.loader.exec_module(module)
        solver_cls = getattr(module, "Solver", None)
        if solver_cls is None:
            raise AttributeError("candidate does not define class Solver")
        return solver_cls()

    return _with_timeout(timeout, _load)


def _timed_call(func, args, runs, warmups, timeout):
    for _ in range(max(0, warmups)):
        _with_timeout(timeout, func, *args)
    _sync()
    samples = []
    for _ in range(max(1, runs)):
        _sync()
        start = time.perf_counter()
        _with_timeout(timeout, func, *args)
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()

    task_name = os.environ.get("ALGOTUNE_TASK", "")
    n = _env_int("ALGOTUNE_N", 64)
    seeds = _parse_seeds(os.environ.get("ALGOTUNE_SEEDS", "1"))
    timing_runs = _env_int("ALGOTUNE_TIMING_RUNS", 3)
    warmups = _env_int("ALGOTUNE_WARMUPS", 1)
    solve_timeout = _env_float("ALGOTUNE_SOLVE_TIMEOUT", 120.0)
    root = os.environ.get("ALGOTUNE_ROOT", "")

    result = {
        "combined_score": float("-inf"),
        "valid": False,
        "compiled": False,
        "loading_success": False,
        "correctness_success": False,
        "correctness": False,
        "runtime_custom_ms": None,
        "runtime_original_ms": None,
        "runtime": None,
        "runtime_original": None,
        "task": task_name,
        "n": n,
        "seeds": seeds,
        "error": "",
    }

    if root and root not in sys.path:
        sys.path.insert(0, root)

    try:
        if not task_name:
            raise ValueError("ALGOTUNE_TASK is not set")
        from AlgoTuneTasks.factory import TaskFactory

        task = TaskFactory(task_name)
        problems = [
            _with_timeout(
                solve_timeout,
                task.generate_problem,
                n,
                random_seed=seed,
            )
            for seed in seeds
        ]
        solver = _load_solver(args.solution, solve_timeout)
        result["compiled"] = True
        result["loading_success"] = True

        correctness = []
        ref_times = []
        cand_times = []
        for problem in problems:
            try:
                reference_solution = _with_timeout(
                    solve_timeout, task.solve, problem
                )
                candidate_solution = _with_timeout(
                    solve_timeout, solver.solve, problem
                )
                ok = bool(
                    _with_timeout(
                        solve_timeout,
                        task.is_solution,
                        problem,
                        candidate_solution,
                    )
                )
                correctness.append(ok)
                if not ok:
                    continue
                ref_times.append(
                    _timed_call(
                        task.solve, (problem,), timing_runs, warmups, solve_timeout
                    )
                )
                cand_times.append(
                    _timed_call(
                        solver.solve, (problem,), timing_runs, warmups, solve_timeout
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad instance is enough
                result["error"] = f"{type(exc).__name__}: {exc}"
                correctness.append(False)

        result["correctness_success"] = True
        result["correctness"] = bool(correctness) and all(correctness)
        if ref_times and cand_times:
            runtime_original = statistics.median(ref_times)
            runtime_custom = statistics.median(cand_times)
            result["runtime_original_ms"] = runtime_original
            result["runtime_custom_ms"] = runtime_custom
            result["runtime_original"] = runtime_original
            result["runtime"] = runtime_custom
            if result["correctness"] and runtime_custom > 0:
                result["combined_score"] = runtime_original / runtime_custom
                result["valid"] = True
        if not result["correctness"] and not result["error"]:
            result["error"] = "candidate output failed correctness check"

    except Exception as exc:  # noqa: BLE001 - always emit parseable JSON
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        if not result["compiled"]:
            result["loading_success"] = False
        result["correctness_success"] = False
        result["correctness"] = False

    print(MARKER)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
