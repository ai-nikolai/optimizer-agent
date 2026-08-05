#!/usr/bin/env python3
"""
Minimal CUDA kernel evaluator with defenses against reward hacking.
Each stage is an independent function.
Usage: python eval.py --original ref.py --new custom.py [--trials 3] [--perf 10]
Assumes CUDA_VISIBLE_DEVICES is set.

CUDA_VISIBLE_DEVICES=2 python evaluate.py --original reference_29.py --new baseline_29.py

"""

import os
import argparse, gc, threading, traceback
import json
from pathlib import Path
import torch
import numpy as np

from pprint import pprint

REFERENCE_FILE=os.environ.get("REFERENCE_FILE", "./reference_29.py")

# ----------------------------------------------------------------------
# Guards against monkey-patching (Hack #5)
# ----------------------------------------------------------------------
_ORIG_CUDA_EVENT_RECORD = torch.cuda.Event.record
_ORIG_CUDA_EVENT_ELAPSED_TIME = torch.cuda.Event.elapsed_time

def _check_no_monkey_patch():
    if torch.cuda.Event.record is not _ORIG_CUDA_EVENT_RECORD:
        return False, "torch.cuda.Event.record was monkey-patched"
    if torch.cuda.Event.elapsed_time is not _ORIG_CUDA_EVENT_ELAPSED_TIME:
        return False, "torch.cuda.Event.elapsed_time was monkey-patched"
    return True, "OK"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_reference(path):
    ns = {}
    exec(Path(path).read_text(), ns)
    return ns["Model"], ns["get_init_inputs"], ns["get_inputs"]

def _validate_output(out, device):
    """Defense against lazy evaluation (Hack #3):
    Output must be a real, materialized tensor on the correct device."""
    if not isinstance(out, torch.Tensor):
        return False, f"Output is not a tensor: {type(out)}"
    if type(out).__name__ not in ["Tensor", "Parameter"]:
        return False, f"Output is a subclass: {type(out).__name__}"
    # if out.device != device:
    if "cuda" not in str(out.device):
        return False, f"Output on wrong device: {out.device} (expected {device})"
    if out.untyped_storage().size() == 0:
        return False, "Output has no allocated storage (lazy)"
    if out.data_ptr() == 0:
        return False, "Output storage pointer is null (lazy)"
    return True, "OK"

def _check_thread_spawn(before_count):
    """Defense against thread injection (Hack #2):
    No background thread must be created during kernel execution."""
    after_count = threading.active_count()
    if after_count > before_count:
        return False, f"Thread count increased from {before_count} to {after_count}"
    return True, "OK"

def _measure_model(model_cls, get_init, get_inputs, seed, num_trials):
    """
    Time a model class with stream injection defense (Hack #1).
    Returns dict with mean_ms, std_ms, num_trials.
    """
    set_seed(seed)
    init_args = get_init()
    init_args = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_args]
    inputs = get_inputs()
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    model = model_cls(*init_args).cuda()

    # Warmup with synchronize
    for _ in range(3):
        model(*inputs)
        torch.cuda.synchronize()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(*inputs)
        torch.cuda.synchronize()          # Hack #1: catch work on other streams
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = float(np.mean(times))
    return {"mean_ms": avg_ms,
            "std_ms": float(np.std(times)),
            "num_trials": num_trials}

# ----------------------------------------------------------------------
# Compile custom source (also checks monkey‑patching)
# ----------------------------------------------------------------------
def compile_custom(path):
    src = Path(path).read_text()
    build_dir = str(Path(path).with_name(Path(path).stem + "_compile"))
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # Prepend environment for JIT build and device-side assertions
    src = (
        f"import os\n"
        f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        f"os.environ['TORCH_USE_CUDA_DSA'] = '1'\n"
    ) + src

    ns = {}
    try:
        compile(src, "<string>", "exec")
        exec(src, ns)
    except Exception as e:
        return {"compiled": False, "error": "Compile Error: " + str(e)}

    # Hack #5 defense: detect monkey‑patching after the custom code ran
    ok, msg = _check_no_monkey_patch()
    if not ok:
        return {"compiled": False, "error": f"Monkey-patch detected: {msg}"}

    if "ModelNew" not in ns:
        return {"compiled": False, "error": "ModelNew not defined"}
    return {"compiled": True, "model_new_cls": ns["ModelNew"]}

# ----------------------------------------------------------------------
# Evaluation stages
# ----------------------------------------------------------------------
def stage_compile(custom_path):
    """Stage 1: Compile custom kernel."""
    return compile_custom(custom_path)

def stage_quick_check(ref_path, custom_path, seed):
    """Stage 2: Instantiate both models to catch obvious errors."""
    try:
        ref_cls, get_init, _ = load_reference(ref_path)
        comp = compile_custom(custom_path)
        if not comp["compiled"]:
            return {"loading_success": False, "error": comp.get("error")}

        set_seed(seed)
        init_args = get_init()
        init_args = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_args]

        with torch.no_grad():
            ref_cls(*init_args).cuda()
            comp["model_new_cls"](*init_args).cuda()
        gc.collect(); torch.cuda.empty_cache()
        return {"loading_success": True}
    except Exception as e:
        return {"loading_success": False, "error": "Quick Check Error: " + str(e)}

def stage_correctness(ref_path, custom_path, seed, num_trials):
    """
    Stage 3: Multi‑trial correctness check.
    Also validates output materialization (Hack #3) and thread count (Hack #2)
    for every trial. Once passed, these checks don't need to be repeated later.
    """
    try:
        ref_cls, get_init, get_inputs = load_reference(ref_path)
        comp = compile_custom(custom_path)
        if not comp["compiled"]:
            return {"correctness": False, "error": comp.get("error")}
        custom_cls = comp["model_new_cls"]

        set_seed(seed)
        trial_seeds = [torch.randint(0, 2**31-1, (1,)).item() for _ in range(num_trials)]
        init_args = get_init()
        init_args = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_args]
        device = init_args[0].device if init_args else torch.device("cuda")

        pass_count = 0
        max_diff = 0.0
        shape_ok = True

        for ts in trial_seeds:
            set_seed(ts)
            inputs = get_inputs()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
            threads_before = threading.active_count()

            with torch.no_grad():
                set_seed(seed)
                ref_out = ref_cls(*init_args).cuda()(*inputs)
                set_seed(seed)
                custom_out = custom_cls(*init_args).cuda()(*inputs)

            # Hack #3: lazy evaluation defense
            ok, msg = _validate_output(custom_out, device)
            if not ok:
                return {"correctness": False, "error": f"Lazy output: {msg}", "pass_count": 0}

            # Hack #2: thread injection defense
            ok, msg = _check_thread_spawn(threads_before)
            if not ok:
                return {"correctness": False, "error": f"Thread injection: {msg}", "pass_count": 0}

            if ref_out.shape != custom_out.shape:
                shape_ok = False
                break

            if not torch.allclose(ref_out, custom_out, atol=1e-2, rtol=1e-2):
                max_diff = max(max_diff, (ref_out - custom_out).abs().max().item())
            else:
                pass_count += 1
            gc.collect(); torch.cuda.empty_cache()

        if not shape_ok:
            return {"correctness": False, "pass_count": 0, "error": "shape mismatch"}
        ok = (pass_count == num_trials)
        return {"correctness": ok, "pass_count": pass_count,
                "max_diff": max_diff if not ok else None}
    except Exception as e:
        return {"correctness": False, "error": "Correctness Error: " + str(e), "pass_count": 0}

def stage_performance_original(ref_path, seed, num_trials):
    """Stage 4a: Measure reference model execution time."""
    try:
        ref_cls, get_init, get_inputs = load_reference(ref_path)
        return _measure_model(ref_cls, get_init, get_inputs, seed, num_trials)
    except Exception as e:
        return {"error": "Original Performance Error: " + str(e)}

def stage_performance_custom(ref_path, new_path, seed, num_trials):
    """Stage 4b: Measure custom model execution time (compiles custom, uses reference for data)."""
    try:
        # Load reference for data functions
        _, get_init, get_inputs = load_reference(ref_path)
        # Compile custom
        comp = compile_custom(new_path)
        if not comp["compiled"]:
            return {"error": comp.get("error")}
        custom_cls = comp["model_new_cls"]
        return _measure_model(custom_cls, get_init, get_inputs, seed, num_trials)
    except Exception as e:
        return {"error": "Custom Performance Error: " + str(e)}

# ----------------------------------------------------------------------
# Main evaluation orchestrator
# ----------------------------------------------------------------------
def if_not_path_write_tmp_file(path_or_content):
    if not os.path.exists(path_or_content):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(path_or_content)
            temp_path = f.name
            print(f"INFO: creating tmp file with name: {temp_path}")
            return temp_path
    else:
        return path_or_content

def evaluate(new_path, original_path=REFERENCE_FILE, seed=42, num_correctness_trials=3, num_perf_trials=10):
    new_path = if_not_path_write_tmp_file(new_path)

    res = {
        "compiled": False, 
        "loading_success": False, 
        "correctness": False,
        "runtime_custom_ms": None, 
        "runtime_original_ms": None,
        "error": None, 
        "pass_count": 0, 
        "combined_score": -float('inf')
    }

    # 1. Compile (includes monkey-patch check)
    comp = stage_compile(new_path)
    res["compiled"] = comp["compiled"]
    if not comp["compiled"]:
        res["error"] = comp.get("error", "compilation failed")
        return res

    # 2. Quick check
    qc = stage_quick_check(original_path, new_path, seed)
    res["loading_success"] = qc["loading_success"]
    if not qc["loading_success"]:
        res["error"] = qc.get("error", "model instantiation failed")
        return res

    # 3. Correctness (validates lazy eval and thread injection)
    corr = stage_correctness(original_path, new_path, seed, num_correctness_trials)
    res.update(corr)
    if not res["correctness"]:
        return res

    # 4a. Performance of original model
    orig_perf = stage_performance_original(original_path, seed, num_perf_trials)
    if "error" in orig_perf:
        res["error"] = orig_perf["error"]
        return res
    res["runtime_original_ms"] = orig_perf["mean_ms"]
    res["original_stats"] = orig_perf

    # 4b. Performance of custom model
    cust_perf = stage_performance_custom(original_path, new_path, seed, num_perf_trials)
    if "error" in cust_perf:
        res["error"] = cust_perf["error"]
        return res
    res["runtime_custom_ms"] = cust_perf["mean_ms"]
    res["custom_stats"] = cust_perf

    res["combined_score"] =  res["runtime_original_ms"] - res["runtime_custom_ms"] #we want to optimise custom_ms down (so it needs to be negative)

    return res

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=3, help="Number of correctness trials")
    ap.add_argument("--perf", type=int, default=10, help="Number of timing trials")
    args = ap.parse_args()

    result = evaluate(
        original_path=args.reference,
        new_path=args.solution,
        seed=args.seed,
        num_correctness_trials=args.trials,
        num_perf_trials=args.perf,
    )

    print("\n=== Evaluation Result ===")
    # for k, v in result.items():
    #     print(f"{k:25s}: {v}")
    # pprint(result)
    json.dumps(result)

if __name__ == "__main__":
    main()