"""OpenEvolve evaluator template for one KernelBench problem.

The task directory contains ``reference.py`` (the original KernelBench problem)
and this evaluator. The evolved program must define ``ModelNew``; it is compared
against the reference ``Model`` with robust_kernelbench's
``eval_kernel_against_ref``.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROBUST_KB_ROOT = Path(
    os.environ.get("ROBUST_KERNELBENCH_ROOT", "/hx2-weka/home/nr1713/robust_kernelbench")
)
if str(ROBUST_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBUST_KB_ROOT))

from robust_kernelbench.evaluate_single import eval_kernel_against_ref  # noqa: E402


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def evaluate(program_path, config=None):
    reference = (HERE / "reference.py").read_text()
    candidate = Path(program_path).read_text()
    build_root = Path(os.environ.get("KB_BUILD_ROOT", tempfile.gettempdir()))
    build_root = build_root / f"oe_kb_{os.getpid()}_{time.time_ns()}"
    build_root.mkdir(parents=True, exist_ok=True)

    try:
        result = eval_kernel_against_ref(
            original_model_src=reference,
            custom_model_src=candidate,
            seed_num=42,
            num_correct_trials=_env_int("KB_NUM_CORRECT_TRIALS", 5),
            num_perf_trials=_env_int("KB_NUM_PERF_TRIALS", 3),
            verbose=False,
            measure_performance=True,
            build_dir=str(build_root / "kb_build_ref"),
            build_dir_new=str(build_root / "kb_build_new"),
            device=torch.device("cuda:0"),
            atol=None,
            rtol=None,
            flush_l2=os.environ.get("KB_FLUSH_L2", "1") not in ("0", "false", "False"),
        )
        data = result.model_dump() if hasattr(result, "model_dump") else vars(result)
        runtime = data.get("runtime")
        runtime_original = data.get("runtime_original")
        correctness = bool(data.get("correctness"))
        timing_success = bool(data.get("timing_success"))
        speedup = (
            float(runtime_original) / float(runtime)
            if correctness
            and timing_success
            and runtime
            and runtime_original
            and float(runtime) > 0
            else 0.0
        )
        return {
            "combined_score": speedup,
            "speedup_score": speedup,
            "valid": correctness,
            "compiled": bool(data.get("compiled")),
            "loading_success": bool(data.get("loading_success")),
            "correctness": correctness,
            "timing_success": timing_success,
            "runtime_custom_ms": runtime,
            "runtime_original_ms": runtime_original,
            "error": (data.get("metadata") or {}).get("error", ""),
        }
    except Exception as exc:  # noqa: BLE001 - invalid candidates score 0
        return {
            "combined_score": 0.0,
            "speedup_score": 0.0,
            "valid": False,
            "compiled": False,
            "loading_success": False,
            "correctness": False,
            "timing_success": False,
            "runtime_custom_ms": None,
            "runtime_original_ms": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
