# SKILL: robust_kernelbench

A **robust, research-grade version of KernelBench** (inspired by
ScalingIntelligence/KernelBench) for LLM-generated **CUDA kernel optimization**.
Fixes the original's silent GPU errors, adds distributed compilation, richer
metrics, and per-stage failure tracking. Tied to the "FIL Hypothesis" paper.

Layout: `robust_kernelbench/robust_kernelbench/` (Python package) + `utils/` +
`kb_prompts/` (hand-written prompt examples: few_shot, cot, hardware).
Conda envs: `env_model` (sglang), `env_kb2` (KernelBench). GPU required (H200 seen).

## Pipeline (3 stages, orchestrated by `run_main.py`)
1. **Inference** — `run_inference_test_time_scaling.py` : LLM generates candidate
   kernel source (`ModelNew`), saved to `experiments/<exp>/<trial>/code/`.
2. **Compilation** — `compile_all.py` (parallel) / `compile_all_sequential.py`
   : `nvcc`-compile via torch extensions into build dirs; `compilations.csv`.
3. **Evaluation** — `evaluate_all.py` : run `evaluate_single.py` (subprocess per
   problem/sample) → `evaluations.csv` (one row per problem_id/sample_id).
   Slurm: `scripts/slurm_job_v2.sh`.

Each kernel is a `torch.nn.Module` subclass named `ModelNew` that must accept the
reference's `get_init_inputs()` / `get_inputs()` and produce numerically equal
output to the reference `Model` (`torch.allclose(atol=1e-2, rtol=1e-2)`).

## How evaluation is done — the heart of it
### Key files
- `evaluate_single.py` — single kernel vs reference (correctness + timing).
- `evaluate_all.py` — loop over all compiled kernels / dataset; spawns subprocess per eval.
- `utils/utils_compile.py` — `load_original_model_and_inputs`, `load_custom_model`,
  `run_compilation` (compile + exec the source string).
- `utils/utils_evaluate_common.py` — fd-level stdout/stderr capture (`OutputCapture`,
  `exec_and_get_output`) so `nvcc` output is captured; `register_and_format_exception`.
- `utils/utils_data.py` — pydantic `CompileResult`, `KernelExecResult` result schemas.

### `eval_kernel_against_ref()` — stage-by-stage (short-circuits with typed results)
1. **Load reference** (`load_original_model_and_inputs`): prepend
   `TORCH_EXTENSIONS_DIR=<build_dir>`, then `compile()` + `exec()` source.
   Returns `(Model, get_init_inputs, get_inputs)` or `"compilation_error"`/`"runtime_error"`.
2. **Compile custom** (`load_custom_model`): prepend `TORCH_EXTENSIONS_DIR` **and**
   `TORCH_USE_CUDA_DSA='1'` (device-side assertions — catches illegal memory access).
   RuntimeError "Error building extension" → `compilation_error`. Lock-file errors → retry.
3. **Load** (`ModelNew(*init_inputs)`): `torch.cuda.synchronize()` right after → surfaces
   illegal-access `AcceleratorError` as an `execution_error` / `loading_success=False`.
4. **Correctness** (`run_and_check_correctness`): for `num_correct_trials` (default 3)
   random-seed trials, run ref vs new; pass only if **all** trials match shape & allclose.
   Records `max/avg_difference` and `correctness_trials`.
5. **Performance** (`time_execution_with_cuda_event`, default 3 perf trials, `measure_performance`):
   warmup 3, then measure each model with `torch.cuda.Event`s (ms). Records
   `runtime` / `runtime_original` and full `*_stats` (mean/std/min/max/trials/hardware).
   Result is `KernelExecResult` (not a single number) — everything is captured.

### Result schema (`KernelExecResult`)
Booleans per stage: `cuda_success, compiled, runtime_success, loading_success,
correctness_success, timing_success`, then `correctness` (the actual pass/fail),
`runtime`(ms), `runtime_original`(ms), `runtime_stats`, `metadata` (errors, traces,
diffs, execution_status), `main_output`, `main_error`, `main_traceback`.

### `evaluate_all.py` loop details
- Dataset: `load_dataset("ai-nikolai/KernelBench")`, `ds[f"level_{level}"]` (levels 1–3).
- Compiles the **reference kernel** from the dataset into `build_dir` at eval time.
- Only evaluates kernels that compiled (`model_new_available=True` in `compilations.csv`);
  skips already-done `(problem_id, sample_id)` pairs (`pid_sid_set`).
- Runs each eval in a **subprocess** with a 600s timeout (avoids CUDA context bleed between kernels).

## Robustness features (vs original KernelBench)
- **Stage-level typed failures** so a silent CUDA error doesn't corrupt aggregate scores.
- **fd-level output capture** (`os.dup2`) so `nvcc`/subprocess output is retained in results.
- **Device-side assertions** (`TORCH_USE_CUDA_DSA`) and immediate `synchronize` to
  detect illegal memory access / device-side assertions.
- **Memory cleanup** (`gc.collect()` + `torch.cuda.empty_cache()`) between evals.
- **Distributed / sequential compilation** with retry on lock-file contention.

## Bugs / caveats
- `KernelExecResult.runtime` comments say "in us" but timings are actually ms.
- `evaluate_all.eval_main_loop` has extensive dead multiprocessing code + unused imports (Pool/pbar used but `out_results` never appended to in the active subprocess branch).
- `run_main.py` references `run_inference_test_time_scaling_v2.py` in a print but calls the non-v2 script.
- `metadata["device"]` is set inside an unused `except` block in `eval_kernel_against_ref`.
- `evaluate_single` docstrings say single correctness trial while defaults use 3 (minor inconsistency).
- `run_and_check_correctness` shape-mismatch branch returns correctly, but its `Metadata.setdefault` for diffs only records on the value-mismatch (not shape) branch.
- Heavy deps (torch, sglang/vllm, cuda-toolkit) live in conda envs, not pip-installed here.
