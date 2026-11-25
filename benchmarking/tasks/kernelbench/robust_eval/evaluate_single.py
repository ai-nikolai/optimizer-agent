# Example commands (evaluating an already-compiled kernel):
# python -m robust_kernelbench.evaluate_single --experiment exp_v3 --problem_id 88 --sample_id 0 \
#     --original_path experiments/exp_v3/trial_1/kernel/88__sample__ref_kernel/88_MinGPTNewGelu.py \
#     --new_path experiments/exp_v3/trial_1/code/88__sample_0.py
#
# python -m robust_kernelbench.evaluate_single --experiment exp_v3 --problem_id 98 --sample_id 0 \
#     --original_path experiments/exp_v3/trial_1/kernel/98__sample__ref_kernel/98_KLDivLoss.py \
#     --new_path experiments/exp_v3/trial_1/code/98__sample_0.py \
#     --build_dir_new experiments/exp_v3/trial_1/kernel/98__sample__solution_kernel

import gc
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import artifacts
from .cli_utils import positive_int
from .logging_utils import get_logger, setup_logging
from .utils.utils import read_file
from .utils.utils_data import KernelExecResult

from .utils.utils_evaluate_common import (
    register_and_format_exception,
    format_exception_string,
    OutputCapture
)

from .utils.utils_compile import (
    load_original_model_and_inputs,
    load_custom_model,
    # compile_and_get_output,
    # exec_and_get_output
)

logger = get_logger(__name__)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Ensure spawn is used

# BASIC DEFINITIONS
level1_representative_subset_problem_ids = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]
level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]
level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]

CURRENT_DATETIME=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def get_dataframe_results(results):
    """Dataframe Results"""
    # tmp = [y.__dict__ for y in results]

    out_df = pd.DataFrame.from_dict(results)

    return out_df


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


# ============================================================================
#  Correctness tolerances
# ============================================================================
# The upstream harness compares every dtype with atol=rtol=1e-2, which is lax for
# fp32/fp64 (it hides real numerical errors) and tight for bf16 (it produces spurious
# failures).  The policy below picks a tolerance from the *reference output dtype*.
# Explicit --atol/--rtol on the CLI always win, and `--atol 1e-02 --rtol 1e-02`
# reproduces the historical behaviour exactly.
DTYPE_TOLERANCES = {
    "torch.float64": (1e-5, 1e-5),
    "torch.float32": (1e-4, 1e-4),
    "torch.float16": (1e-2, 1e-2),
    "torch.bfloat16": (2e-2, 2e-2),
    "torch.complex64": (1e-4, 1e-4),
    "torch.complex128": (1e-5, 1e-5),
}
FALLBACK_TOLERANCE = (1e-2, 1e-2)


def dtype_key(dtype) -> str:
    """Stable, human-readable key for a torch dtype (e.g. ``"torch.bfloat16"``)."""
    return str(dtype)


def tolerance_for_dtype(dtype) -> tuple[float, float]:
    """``(atol, rtol)`` for an output dtype.

    Integer and boolean outputs are compared exactly.  Unknown dtypes fall back to the
    historical 1e-2/1e-2.
    """
    if dtype is None:
        return FALLBACK_TOLERANCE

    try:
        if not dtype.is_floating_point and not dtype.is_complex:
            return (0.0, 0.0)
    except AttributeError:  # pragma: no cover - non-torch dtype objects
        return FALLBACK_TOLERANCE

    return DTYPE_TOLERANCES.get(dtype_key(dtype), FALLBACK_TOLERANCE)


def resolve_tolerances(dtype=None, atol=None, rtol=None) -> tuple[float, float]:
    """Combine explicit overrides with the dtype policy.

    * both explicit        -> use them
    * one explicit         -> use it for that term, dtype policy for the other
    * neither explicit     -> dtype policy
    """
    dtype_atol, dtype_rtol = tolerance_for_dtype(dtype)
    final_atol = dtype_atol if atol is None else float(atol)
    final_rtol = dtype_rtol if rtol is None else float(rtol)
    return final_atol, final_rtol


# ============================================================================
#  L2 / cache flushing for timing
# ============================================================================
# Timing a tiny kernel twice in a row lets the second run read its inputs straight out of
# L2, which flatters small kernels badly.  KernelBench's own harness does not flush, so
# speedups against a cache-warm reference are optimistic.  We allocate one throw-away
# buffer and zero it before every timed trial (outside the timed region) to evict it.
L2_FLUSH_BYTES = 256 * 1024 * 1024  # comfortably larger than the L2 of any shipping GPU


def make_l2_flush_buffer(device=None, size_bytes: int = L2_FLUSH_BYTES):
    """Allocate the buffer used to evict L2.  Returns ``None`` on failure (CPU-only runs)."""
    try:
        return torch.empty(size_bytes, dtype=torch.int8, device=device)
    except Exception as e:  # pragma: no cover - out-of-memory / no CUDA
        logger.warning("could not allocate L2 flush buffer (%s); timing will not be cache-cold", e)
        return None


def flush_l2_cache(buffer) -> None:
    """Zero the flush buffer so the next kernel cannot find stale data in L2."""
    if buffer is None:
        return
    buffer.zero_()



def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    flush_l2: bool = True,
    flush_buffer=None,
    ) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device
        flush_l2: Evict L2 before each timed trial (measurements become cache-cold)
        flush_buffer: Pre-allocated flush buffer; allocated here when omitted

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    if flush_l2 and flush_buffer is None:
        flush_buffer = make_l2_flush_buffer(device=device)

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}, flush_l2={flush_l2}"
    )
    elapsed_times = []

    # drop whatever the warm-up left in L2 before the first measured trial
    flush_l2_cache(flush_buffer)

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Evict L2 *outside* the timed region, so each trial is cache-cold.
        flush_l2_cache(flush_buffer)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times

def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


def memory_cleanup(curr_context: dict|Any=None, device: torch.device=None, custom_string: str="", verbose: bool=False):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    if verbose:
        print("\n======Start")
        if custom_string:
            print(custom_string)
        print("---before")
        print(f"Allocated memory: {torch.cuda.memory_allocated()}")
        print(f"Cached memory: {torch.cuda.memory_reserved()}")
    # print(curr_context)

    if curr_context:
        del curr_context
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print("\n---after")
        print(f"Allocated memory: {torch.cuda.memory_allocated()}")
        print(f"Cached memory: {torch.cuda.memory_reserved()}")
        print("======End\n")



CUDA_ERROR_MARKERS = (
    "cuda error",
    "cudaError",
    "illegal memory access",
    "device-side assert",
    "misaligned address",
    "out of memory",
    "cublas",
    "cudnn",
)


def is_cuda_error(exc: BaseException) -> bool:
    """Whether ``exc`` came from the CUDA stack rather than the kernel's Python code.

    PyTorch raises ``torch.AcceleratorError`` for some faults but plain
    ``RuntimeError``/``OutOfMemoryError`` for others, so both the type and the
    message are checked.
    """
    if isinstance(exc, torch.AcceleratorError):
        return True
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    message = str(exc).lower()
    return any(marker.lower() in message for marker in CUDA_ERROR_MARKERS)


def _as_output_sequence(output: Any) -> list[Any]:
    """Normalize a model output (tensor, tuple or list) for comparison."""
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        return list(output)
    return [output]


def _compare_output_sequences(
    expected: Any,
    actual: Any,
    atol: float | None,
    rtol: float | None,
    metadata: dict,
) -> tuple[bool, str]:
    """Compare two model outputs, returning ``(matched, message)``.

    Handles the single-tensor case exactly as before and also tuple/list outputs
    (common in KernelBench levels 2/3), which used to raise ``AttributeError`` on
    ``output.shape`` and be misreported as a runtime error.
    """
    expected_items = _as_output_sequence(expected)
    actual_items = _as_output_sequence(actual)

    if len(expected_items) != len(actual_items):
        return False, (
            f"Output length mismatch: expected {len(expected_items)} value(s), "
            f"got {len(actual_items)}"
        )

    for index, (expected_item, actual_item) in enumerate(zip(expected_items, actual_items)):
        prefix = "" if len(expected_items) == 1 else f"[output {index}] "
        if isinstance(expected_item, torch.Tensor) and isinstance(actual_item, torch.Tensor):
            if expected_item.shape != actual_item.shape:
                return False, (
                    f"{prefix}Output shape mismatch: expected {expected_item.shape}, "
                    f"got {actual_item.shape}"
                )

            trial_atol, trial_rtol = resolve_tolerances(expected_item.dtype, atol=atol, rtol=rtol)
            metadata["atol"] = trial_atol
            metadata["rtol"] = trial_rtol
            metadata["tolerance_dtype"] = dtype_key(expected_item.dtype)

            if not torch.allclose(expected_item, actual_item, atol=trial_atol, rtol=trial_rtol):
                max_diff = torch.max(torch.abs(expected_item - actual_item)).item()
                avg_diff = torch.mean(torch.abs(expected_item - actual_item)).item()
                metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                return False, (
                    f"{prefix}Output mismatch: max difference is: {max_diff:.6f} "
                    f"(atol={trial_atol}, rtol={trial_rtol}, dtype={dtype_key(expected_item.dtype)})"
                )
            continue

        if isinstance(expected_item, torch.Tensor) or isinstance(actual_item, torch.Tensor):
            return False, f"{prefix}Output type mismatch: expected {type(expected_item)}, got {type(actual_item)}"

        try:
            equal = bool(expected_item == actual_item)
        except Exception:  # noqa: BLE001 - exotic output objects
            equal = expected_item is actual_item
        if not equal:
            return False, f"{prefix}Output mismatch: expected {expected_item!r}, got {actual_item!r}"

    return True, ""


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
    atol=None,
    rtol=None,
    ) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    atol/rtol: explicit tolerances; when None they are chosen from the output dtype
    """
    if num_correct_trials < 1:
        raise ValueError("num_correct_trials must be >= 1")

    pass_count = 0
    exception_str = ""
    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)

                matched, comparison_message = _compare_output_sequences(
                    output, output_new, atol=atol, rtol=rtol, metadata=metadata
                )
                if not matched:
                    metadata, exception_str = register_and_format_exception(
                        "correctness_issue", comparison_message, metadata
                    )
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: {comparison_message}")
                    return KernelExecResult(
                        compiled=True,
                        correctness_success=True,
                        correctness=False,
                        metadata=metadata,
                        main_error=format_exception_string(exception_str),
                    )

                pass_count += 1
                if verbose:
                    print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:  # noqa: BLE001 - kernel failures are measurements
                logger.warning("correctness check failed: %s", e)
                metadata, exception_str = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                return KernelExecResult(
                    cuda_success=not is_cuda_error(e),
                    compiled=True,
                    correctness_success=False,
                    correctness=False,
                    metadata=metadata,
                    main_error=format_exception_string(exception_str),
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(
            compiled=True,
            correctness_success=True,
            correctness=True,
            metadata=metadata,
            main_error=format_exception_string(exception_str),
        )
    else:
        return KernelExecResult(
            compiled=True,
            correctness_success=True,
            correctness=False,
            metadata=metadata,
            main_error=format_exception_string(exception_str),
        )


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 5,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    build_dir_new: os.PathLike = None,
    device: torch.device = None, # have to run on GPU
    atol: float | None = None,
    rtol: float | None = None,
    flush_l2: bool = True,
    ) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    atol/rtol: explicit correctness tolerances; None means "decide from the output dtype"
    flush_l2: evict L2 between timing trials (cache-cold measurements)
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    """
    if num_correct_trials < 1:
        raise ValueError("num_correct_trials must be >= 1")
    if num_perf_trials < 1:
        raise ValueError("num_perf_trials must be >= 1")

    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    (Model, get_init_inputs, get_inputs), _orig_status, _orig_output = load_original_model_and_inputs(
        original_model_src,
        context,
        build_dir,
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")



    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    metadata = {"device": str(device), "execution_status": None}

    # this is where compilation happens
    compilation_output = False
    try:

        custom_model_src = (
            "import os\n" f"os.environ['TORCH_USE_CUDA_DSA'] = '1'\n"
        ) + custom_model_src
        print("[Eval] running load custom model...")
        ModelNew, compilation_status, (std_compile, std_runtime), _ = load_custom_model(
            custom_model_src,
            context,
            build_dir_new,
            device
        )
        compilation_output = True
        print("[Eval] Loading Custom Model returned something")
        metadata["compilation_status"] = compilation_status

        try:
            torch.cuda.synchronize(device=device)  # This will cause CUDA errors
            print("[Eval] CUDA synchronize (Model compilation) successful...")
        except torch.AcceleratorError as e:
            print("[Eval] CUDA synchronize (Model compilation) failed.")
            metadata["compilation_error"] = e
            metadata["compilation_output"] = std_compile

            tmp_traceback = traceback.format_exc() #traceback.print_exc()
            metadata["compilation_traceback"] = tmp_traceback

            return KernelExecResult(
                cuda_success=False,
                compiled=True,
                runtime_success=True,
                metadata=metadata,
                main_output=std_compile,
                main_traceback=tmp_traceback if tmp_traceback else "",

            )  # skip further steps
    except Exception as e:
        print(
            f"[Eval] ERROR - Failed to compile custom CUDA kernel: Record as compilation or runtime failure."
        )
        if not compilation_output:
            print("[Eval] No compilation output. Therefore getting it from the exception.")
            try:
                _, compilation_status, (std_compile, std_runtime), _ = e.kernel_output_added
            except Exception:
                # hopefully this means that load_custom_model returned a useful output
                pass

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            memory_cleanup(context, device)
            raise e

        else:
            if compilation_status=="compilation_error":
                print("---")
                metadata["compilation_error"] = e
                metadata["compilation_output"] = std_compile

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["compilation_traceback"] = tmp_traceback

                memory_cleanup(context, device)

                return KernelExecResult(
                    compiled=False,
                    runtime_success=False,
                    metadata=metadata,
                    main_output=std_compile,
                    main_traceback=tmp_traceback if tmp_traceback else "",

                )  # skip further steps
            elif compilation_status=="runtime_error":
                print("---2")
                metadata["runtime_error"] = e
                metadata["runtime_output"] = std_runtime
                metadata["compilation_status"] = compilation_status

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["runtime_traceback"] = tmp_traceback


                memory_cleanup(context, device)

                return KernelExecResult(
                    compiled=True,
                    runtime_success=False,
                    metadata=metadata,
                    main_output=std_runtime,
                    main_error=format_exception_string(e),
                    main_traceback=tmp_traceback if tmp_traceback else "",
                )  # skip further steps
            else:
                memory_cleanup(context, device)

                print(f"[STRANGE] You are having success in compilation / runtime, but still an exception.\n---:\n{e}")
                raise e

    print("[EVAL] Compilation was fully successful.")
    # at this point we passed compilation
    exec_std_output=""
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            with OutputCapture() as capture:
                custom_model = ModelNew(*init_inputs)
            exec_std_output = capture.get_output()
            assert hasattr(custom_model, "forward")
            try:
                torch.cuda.synchronize(device=device)  # This will cause CUDA errors
                print("[Eval] CUDA synchronize (Model loading) successful...")
            except torch.AcceleratorError as e:
                print("[Eval] CUDA synchronize (Model loading) failed...")
                metadata["execution_error"] = e
                metadata["execution_output"] = exec_std_output
                metadata["execution_status"] = "error_model_loading"

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["execution_traceback"] = tmp_traceback

                return KernelExecResult(
                    cuda_success=False,
                    compiled=True,
                    runtime_success=True,
                    loading_success=True, #setting this as true... as the code worked...
                    metadata=metadata,
                    main_output=std_compile,
                    main_traceback=tmp_traceback if tmp_traceback else "",

                )  # skip further steps

            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"[Eval] Failed to load custom CUDA kernel; Compiled but not able to run, count as execution error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        memory_cleanup(context, device)

        metadata["execution_error"] = e
        metadata["execution_output"] = exec_std_output
        metadata["execution_status"] = "error_model_loading"

        tmp_traceback = traceback.format_exc() #traceback.print_exc()
        metadata["execution_traceback"] = tmp_traceback

        return KernelExecResult(
            cuda_success=not is_cuda_error(e),
            compiled=True,
            runtime_success=True,
            loading_success=False,
            correctness_success=False,
            timing_success=False,
            correctness=False,
            metadata=metadata,
            main_output=exec_std_output,
            main_traceback=tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps
    except torch.AcceleratorError as e:
        print(
            f"[Eval] Failed to load custom CUDA kernel; Compiled but not able to run, count as execution error. \nError: {e}"
        )
        # No clean-up as it further causes errors...

        metadata["execution_error"] = e
        metadata["execution_output"] = exec_std_output
        metadata["execution_status"] = "error_model_loading"

        tmp_traceback = traceback.format_exc() #traceback.print_exc()
        metadata["execution_traceback"] = tmp_traceback

        return KernelExecResult(
            cuda_success=False,
            compiled=True,
            runtime_success=True,
            loading_success=False,
            correctness_success=False,
            timing_success=False,
            correctness=False,
            metadata=metadata,
            main_output=exec_std_output,
            main_traceback=tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps
    except Exception as e:
        print("[STRANGE] - A strange error during ModelNew init().")
        print(init_inputs)
        print(e)

        metadata["execution_error"] = e
        metadata["execution_output"] = exec_std_output
        metadata["execution_status"] = "error_model_loading_2"
        tmp_traceback = traceback.format_exc() # traceback.print_exc()
        metadata["execution_traceback"] = tmp_traceback

        memory_cleanup(context, device)

        return KernelExecResult(
            cuda_success=not is_cuda_error(e),
            compiled=True,
            runtime_success=True,
            loading_success=False,
            correctness_success=False,
            timing_success=False,
            correctness=False,
            metadata=metadata,
            main_output=exec_std_output,
            main_traceback=tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            atol=atol,
            rtol=rtol,
        )
        kernel_exec_result.runtime_success=True
        kernel_exec_result.loading_success=True


    except torch.AcceleratorError as e:
        print(f"[Eval] Failed during run_and_check_correctness - function.... (TorchAcceleratorError){e}")
        # TODO: (maybe) add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["correctness_execution_error"] = e
        # metadata["execution_output"] = exec_std_output #TODO: this one here...
        metadata["correctness_execution_status"] = "error_correctness"
        tmp_traceback = traceback.format_exc() # traceback.print_exc()
        metadata["correctness_execution_traceback"] = tmp_traceback

        # memory_cleanup(context, device)

        return KernelExecResult(
            cuda_success=False,
            compiled=True,
            runtime_success=True,
            loading_success=True,
            correctness_success=False,
            timing_success=False,
            correctness=False,
            metadata=metadata,
            main_output=format_exception_string(e),
            main_traceback = tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps
    except Exception as e:
        print(f"[Eval] Failed during run_and_check_correctness - function.... (Other Exception){e}")
        # TODO: (maybe) add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["correctness_execution_error"] = e
        # metadata["execution_output"] = exec_std_output #TODO: this one here...
        metadata["correctness_execution_status"] = "error_correctness"
        tmp_traceback = traceback.format_exc() # traceback.print_exc()
        metadata["correctness_execution_traceback"] = tmp_traceback

        memory_cleanup(context, device)

        return KernelExecResult(
            compiled=True,
            runtime_success=True,
            loading_success=True,
            correctness_success=False,
            timing_success=False,
            correctness=False,
            metadata=metadata,
            main_output=format_exception_string(e),
            main_traceback = tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps

    custom_timing_success=False
    # NOTE: KernelExecResult is a pydantic model, which *copies* the metadata dict on
    # validation, so post-construction metadata must be written through the result object.
    metadata["flush_l2"] = bool(flush_l2)
    kernel_exec_result.metadata["flush_l2"] = bool(flush_l2)
    # The L2-eviction buffer is allocated lazily (only if we actually time something) and
    # reused for both the kernel and its reference, so both are measured cache-cold.
    flush_buffer = None
    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                if flush_l2 and flush_buffer is None:
                    flush_buffer = make_l2_flush_buffer(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                    flush_l2=flush_l2,
                    flush_buffer=flush_buffer,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats

                kernel_exec_result.timing_success = True
                custom_timing_success=True
        except torch.AcceleratorError as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance (TorchAccelerator): {e}")
            kernel_exec_result.metadata["error_during_performance"] = e
            kernel_exec_result.metadata["execution_status"] = "error_timing"
            kernel_exec_result.main_output=format_exception_string(e)
            tmp_traceback = traceback.format_exc() #traceback.print_exc()
            kernel_exec_result.metadata["execution_traceback"] = tmp_traceback
            kernel_exec_result.main_traceback = tmp_traceback if tmp_traceback else ""
            kernel_exec_result.main_error=format_exception_string(e)
            kernel_exec_result.cuda_success=False #this is the line we add...

            custom_timing_success = False

        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e
            kernel_exec_result.metadata["execution_status"] = "error_timing"
            kernel_exec_result.main_output=format_exception_string(e)
            tmp_traceback = traceback.format_exc() #traceback.print_exc()
            kernel_exec_result.metadata["execution_traceback"] = tmp_traceback
            kernel_exec_result.main_traceback = tmp_traceback if tmp_traceback else ""
            kernel_exec_result.main_error=format_exception_string(e)

            custom_timing_success = False

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness and custom_timing_success:
                if verbose:
                    print("[Eval] Measuring Performance of Original.")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_old = original_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times_original = time_execution_with_cuda_event(
                    model_old,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                    flush_l2=flush_l2,
                    flush_buffer=flush_buffer,
                )
                runtime_original_stats = get_timing_stats(elapsed_times_original, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_original_stats}")
                kernel_exec_result.runtime_original = runtime_original_stats["mean"]
                kernel_exec_result.runtime_original_stats = runtime_original_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance_of_original"] = e
            kernel_exec_result.metadata["execution_status"] = "error_timing_original"

    if custom_timing_success and kernel_exec_result.metadata.get("execution_status") is None:
        kernel_exec_result.metadata["execution_status"] = "success"

    memory_cleanup(context, device)

    return kernel_exec_result


# Create a wrapper function for multiprocessing
def eval_wrapper(
    reference_src,
    custom_src,
    build_dir,
    build_dir_new,
    device,
    num_correct_trials=5,
    num_perf_trials=3,
    atol=None,
    rtol=None,
    flush_l2=True,
):
    return eval_kernel_against_ref(
        original_model_src = reference_src,
        custom_model_src = custom_src,
        seed_num = 42,
        num_correct_trials = num_correct_trials,
        num_perf_trials = num_perf_trials,
        verbose  = True,
        measure_performance = True,
        build_dir = build_dir,
        build_dir_new = build_dir_new,
        device = device, # have to run on GPU
        atol = atol,
        rtol = rtol,
        flush_l2 = flush_l2,
    )

def evaluate_single(
    original_path,
    new_path,
    build_dir=None,
    build_dir_new=None,
    # other
    seed_num = 42,
    num_correct_trials = 5,
    num_perf_trials = 3,
    verbose  = True,
    measure_performance = True,
    device = None,  # resolved to the current CUDA device inside the function
    # Admin:
    experiment_folder = None,
    level=1,
    trial=1,
    problem_id=None,
    sample_id=0,
    atol=None,
    rtol=None,
    flush_l2=True,
    ):
    print(f"[Eval Main Func]\nbuild_dir_new:\t{build_dir_new}\nnew_path:\t{new_path}.")


    if build_dir:
        Path(build_dir).mkdir(parents=True, exist_ok=True)

    if build_dir_new:
        Path(build_dir_new).mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else None

    reference_src = read_file(original_path)
    custom_src = read_file(new_path)
    try:
        logger.info(
            "evaluating level=%s trial=%s pid=%s sid=%s", level, trial, problem_id, sample_id
        )
        start = datetime.now()
        results = eval_wrapper(
            reference_src,
            custom_src,
            build_dir,
            build_dir_new,
            device,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            atol=atol,
            rtol=rtol,
            flush_l2=flush_l2,
        )

        logger.info("evaluation finished in %s", datetime.now() - start)
        output = results.model_dump()

    except (Exception, SystemExit):  # noqa: BLE001 - generated code may call sys.exit
        logger.exception("evaluation failed entirely for trial=%s pid=%s sid=%s", trial, problem_id, sample_id)
        output = KernelExecResult().model_dump()

    output["problem_id"] = problem_id
    output["sample_id"] = sample_id

    try:
        eval_filename = artifacts.csv_path(experiment_folder, trial, artifacts.EVALUATIONS_CSV)
        artifacts.append_rows(get_dataframe_results([output]), eval_filename)
    except Exception:  # noqa: BLE001 - the caller records the missing row
        logger.exception("could not write results to evaluations.csv")
        return False
    return True


def get_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a single generated kernel against its reference model.")
    parser.add_argument("--experiment", type=str, default="exp_test_run", help="experiment names")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")

    parser.add_argument("--original_path", type=str, required=True, help="Path to original model source")
    parser.add_argument("--new_path", type=str, required=True, help="Path to new model source")
    parser.add_argument("--build_dir", type=str, default=None, help="Build directory for original model")
    parser.add_argument("--build_dir_new", type=str, default=None, help="Build directory for new model")
    parser.add_argument("--seed_num", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_correct_trials",
        type=positive_int,
        default=5,
        help="Number of independently seeded correctness trials (all must pass).",
    )
    parser.add_argument("--num_perf_trials", type=positive_int, default=3, help="Number of performance trials")
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance. Default: chosen from the output dtype (fp64 1e-5, fp32 1e-4, fp16 1e-2, bf16 2e-2).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance. Default: chosen from the output dtype (see --atol).",
    )
    parser.add_argument(
        "--flush_l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict L2 before each timed trial so measurements are cache-cold (default: on).",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    # parser.add_argument("--measure_performance", action="store_true", default=True, help="Measure performance")
    parser.add_argument("--problem_id", type=int, default=None, help="Problem ID")
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/...).")

    # TODO: add lora support

    args = parser.parse_args(argv)
    return args

def cli() -> None:
    """Console-script entry point (``robust-kernelbench-eval-single``)."""
    args = get_args()
    setup_logging(args.log_level, force=True)
    logger.info("entered evaluate_single")
    ok = evaluate_single(
        original_path=args.original_path,
        new_path=args.new_path,
        build_dir=args.build_dir,
        build_dir_new=args.build_dir_new,
        seed_num=args.seed_num,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        verbose=args.verbose,
        measure_performance=True,
        experiment_folder=args.experiment,
        level=args.level,
        trial=args.trial,
        problem_id=args.problem_id,
        sample_id=args.sample_id,
        atol=args.atol,
        rtol=args.rtol,
        flush_l2=args.flush_l2,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cli()


# Example Bash command:
# orginal_path = experiments/exp_local_v3/trial_1/kernel/9__sample__ref_kernel/9_Tall_skinny_matrix_multiplication_.py
# new_path = experiments/exp_local_v3/trial_1/code/9__sample_None0.py

# python3 -m robust_kernelbench.evaluate_single \
# --original_path "experiments/exp_local_v3/trial_1/kernel/98__sample__ref_kernel/98_KLDivLoss.py" \
# --new_path experiments/exp_local_v3/trial_1/code/98__sample_0.py \
# --experiment exp_local_v3

"""
# EXAMPLE USAGE:

python3 -m robust_kernelbench.evaluate_single \
--original_path "experiments/exp_slurm_v4_6_tts_Qwen2_5_Coder_7B_Instruct_20260107_150105/trial_2/kernel/39__sample__ref_kernel/39_L2Norm_.py" \
--new_path experiments/exp_local_v3/trial_1/code/39__sample_0.py \
--experiment exp_slurm_v4_6_tts_Qwen2_5_Coder_7B_Instruct_20260107_150105
"""
