#!/usr/bin/env python3
"""KernelBench evaluator adapter for optimizer-agent.

This is the task-specific evaluator (allowed and expected): it calls
``robust_kernelbench``'s evaluator and maps its result to the generic contract
used by optimizer-agent.

It accepts the same ``--solution`` / ``--reference`` interface as every other
evaluator. Both files contain KernelBench problem code; the reference defines
``Model``, ``get_inputs`` and ``get_init_inputs``, and the candidate should
define ``ModelNew``.

Runtime knobs (all optional, as environment variables so the generic agent does
not need evaluator-specific flags):
    KB_NUM_CORRECT_TRIALS=5
    KB_NUM_PERF_TRIALS=3
    KB_ATOL=...
    KB_RTOL=...
    KB_FLUSH_L2=1
"""

import argparse
import json
import os
import sys
from pathlib import Path

MARKER = "=== Evaluation Result ==="

# Allow the adapter to run from the optimizer-agent checkout while importing
# the sibling robust_kernelbench repository. Override if your layout differs.
ROBUST_KB_ROOT = Path(
    os.environ.get(
        "ROBUST_KERNELBENCH_ROOT",
        "/hx2-weka/home/nr1713/robust_kernelbench",
    )
)
if str(ROBUST_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBUST_KB_ROOT))


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _error_from(result, metadata):
    for key in ("main_error", "main_traceback"):
        if result.get(key):
            return str(result[key])
    for key in ("compilation_error", "runtime_error", "execution_error"):
        if metadata.get(key):
            return str(metadata[key])
    for key in ("compilation_output", "runtime_output"):
        if metadata.get(key):
            return str(metadata[key])
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    solution_path = Path(args.solution).resolve()
    reference_path = Path(args.reference).resolve()
    workdir = solution_path.parent

    result = {
        "combined_score": float("-inf"),
        "valid": False,
        "compiled": False,
        "loading_success": False,
        "correctness_success": False,
        "correctness": False,
        "timing_success": False,
        "runtime_custom_ms": None,
        "runtime_original_ms": None,
        "runtime": None,
        "runtime_original": None,
        "error": "",
    }

    try:
        import torch
        from robust_kernelbench.evaluate_single import eval_kernel_against_ref

        if not torch.cuda.is_available():
            result["error"] = "CUDA is not available in the evaluator environment"
        else:
            kernel_result = eval_kernel_against_ref(
                original_model_src=reference_path.read_text(),
                custom_model_src=solution_path.read_text(),
                seed_num=42,
                num_correct_trials=_env_int("KB_NUM_CORRECT_TRIALS", 5),
                num_perf_trials=_env_int("KB_NUM_PERF_TRIALS", 3),
                verbose=False,
                measure_performance=True,
                build_dir=str(workdir / "kb_build_ref"),
                build_dir_new=str(workdir / "kb_build_new"),
                device=torch.device("cuda:0"),
                atol=_env_float("KB_ATOL"),
                rtol=_env_float("KB_RTOL"),
                flush_l2=os.environ.get("KB_FLUSH_L2", "1") not in ("0", "false", "False"),
            )
            data = (
                kernel_result.model_dump()
                if hasattr(kernel_result, "model_dump")
                else vars(kernel_result)
            )
            metadata = data.get("metadata") or {}
            runtime = data.get("runtime")
            runtime_original = data.get("runtime_original")
            correctness = bool(data.get("correctness"))
            timing_success = bool(data.get("timing_success"))
            score = (
                float(runtime_original) / float(runtime)
                if correctness
                and timing_success
                and runtime
                and runtime_original
                and float(runtime) > 0
                else float("-inf")
            )
            result.update(
                {
                    "combined_score": score,
                    "valid": correctness,
                    "compiled": bool(data.get("compiled")),
                    "loading_success": bool(data.get("loading_success")),
                    "correctness_success": bool(data.get("correctness_success")),
                    "correctness": correctness,
                    "timing_success": timing_success,
                    "runtime_custom_ms": runtime,
                    "runtime_original_ms": runtime_original,
                    "runtime": runtime,
                    "runtime_original": runtime_original,
                    "error": _error_from(data, metadata),
                }
            )
    except Exception as exc:  # noqa: BLE001 - always emit a parseable result
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(MARKER)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
