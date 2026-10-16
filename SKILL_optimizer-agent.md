# SKILL: optimizer-agent

A library for **LLM-agent based automatic CUDA kernel optimization & knowledge
discovery**. An LLM (served via an OpenAI-compatible endpoint — SGLang/vLLM)
proposes improved kernel source (`ModelNew`), an external evaluator scores them
(correctness + timing), and a genetic / map-elites search keeps the best and
iterates. This is the **refactored v2** of the original single-file optimizer
agent (`optimizer_agent_ai.py`), organized into small, testable modules and
capable of two prompt styles.

Author: Nikolai Rozanov. Python >= 3.9. Runtime deps: `requests`,
`textdistance==4.6.3` (torch/`tomllib`/numpy are imported lazily where used).

## Layout
- `optimizer_agent/` — the shipped, working package (installed via `pip install -e .`, entry point `optimizer-agent`).
- `optimizer_agent_ai.py` — gitignored, the **pre-refactor planning skeleton** (single monolithic file). Do not treat as the current code.
- `examples/test_example/` — random-scorer smoke test (no GPU needed).
- `examples/cuda_engineering/` — real CUDA kernel scorer (`runtime_original − runtime_custom`) with reward-hacking defenses + KB prompt config.
- `robust_kernelbench/` — the KB asset repo (prompt TOML + GPU specs); see that skill file.
- `opt_agent_experiment/` — default experiment output dir (gitignored).

## Core modules (`optimizer_agent/`)
- `agent.py::Agent` — **orchestrator**. Runs iterations -> islands -> generations; wires prompt builders + LLM + evaluation + feedback + logging. Knows nothing about prompt *content*, only control flow.
- `generation.py::GenerationPipeline` — runs the ordered LLM stages (`strategy -> core -> feedback`) against `PromptBuilder`s; also `generate_debug` for the debug loop. Extracts the final ```-fenced code block from a core response.
- `prompts/base.py::PromptBuilder` — Protocol: `build_messages(prompt_data) -> List[dict]`. The extension point.
- `prompts/simple.py::SimplePromptBuilder` — the original JSON-config templating language (`{system, order, prompts}` per section); `DEFAULT_PROMPT_CONFIG` (`improve`/`debug`, each with core/strategy/feedback).
- `prompts/kernel_bench.py::KernelBenchPromptBuilder` — the **rich** TOML-driven style from robust_kernelbench (problem statement, few/one-shot examples, reference architecture, precision note, optional GPU hardware blocks, instruction, + score-dependent feedback appended after iter 0).
- `evaluation.py` — `evaluate_script()` runs `evaluate.py` as a **subprocess** with timeout; `EvalResult` dataclass (`status`, `result`, `stdout`, `stderr`); `parse_evaluation` robustly extracts JSON after `=== Evaluation Result ===`.
- `benchmark.py` — timing helpers: `benchmark(model, inputs, method="auto")` / `time_event` / `time_cuda_graph`, returning `TimingStats`. **Captures a CUDA graph before timing** (method `auto`/`graph`) to remove host-launch overhead and fail-fast on ungraphable ops, then falls back to event timing. torch is imported lazily.
- `llm.py` — `BaseServer`/`LLMServer` (calls `.../v1/chat/completions`, lazy `requests`), `FakeLLMServer` for `--test` (never touches the network, returns a random string).
- `selection.py` — candidate populations: `CandidatesBestNAll`, `CandidatesBestNLast`, `MapElitesCandidates` (clusters generated code via `textdistance` Jaccard, threshold 0.2).
- `feedback.py` — result-driven `FeedbackStrategy` registry (`none`/`generic`/`error_case`/`score`); see below.
- `prompt_config.py::PromptConfig` — validates & loads either the simple `prompt_config.json` shape or the small KB dict `{backend, option, gpu_name, precision}`.
- `utils.py` — `extract_final_code_response` (pulls out ``` fenced code ```), `extract_eval_results` (parses JSON after marker), `classify_result` (score-independent failure class: compile/runtime/incorrect/ok).
- `cli.py` — argparse → `optimizer-agent ...` entry point (see Run below).
- `tests/test_units.py` — 14 GPU-free unit tests (prompt building both styles when KB assets present, all 4 feedback strategies).
- `agent_strategies.py` — **stub** (planned high-level strategy layer, commented out; the actual strategies live as FeedbackStrategy + the debug pipeline).

## Optimization loop (`Agent.run`)
```
for iteration:
  for island:
    iter 0: build candidates from baseline (num_samples or num_map_elites)
    iter>0: select best predecessors, mutate (map_elites key = Jaccard cluster of code)
  for candidate:
    1. GenerationPipeline.generate() -> new ModelNew .py (strategy/core/feedback sub-stages)
    2. subprocess: python evaluate.py --reference <ref> --solution <code>
    3. parse result -> combined_score / correctness / runtime; update population
    4. apply result-driven FeedbackStrategy(prompt_data, result)
    5. on failure + use_debug: debug loop (num_debug_iterations) planning around the error
predecessor_selection_method: bestN_last | bestN_all | map_elites
```
Key params: `num_iterations, num_islands, num_samples, num_debug_iterations,
predecessor_selection_method, num_map_elites, map_elites_method, use_strategy,
use_feedback, use_debug, use_kb_prompt, feedback_strategy`.

## Two prompt styles (shared control flow, different builders)
1. **simple** — original optimizer-agent behaviour. JSON config with `improve`/`debug`, each `core`/`strategy`/`feedback`. Built by `SimplePromptBuilder`.
2. **kernel_bench** — rich, TOML-driven. `--use-kb-prompt` (or `prompt_config_kb.json`) drives `KernelBenchPromptBuilder`; tunables: `--kb-backend` (cuda/triton/cute/tilelang), `--kb-option` (zero_shot/one_shot/few_shot), `--kb-gpu` (must exist in `robust_kernelbench/kb_prompts/hardware/gpu_specs.py`), `--kb-precision` (fp32/fp16/bf16). Assets (`kb_prompts/prompts.toml`, example `*.py`, `gpu_specs.py`) live in the checked-out `robust_kernelbench/` repo; the builder locates them robustly.

## How evaluation works
- The evaluator is **user-supplied**. Signature: `python evaluate.py --solution SOL [--reference REF] [--trials 3] [--perf 10]`.
- It must print a line `=== Evaluation Result ===` followed by `json.dumps({...})`.
- The parser (`utils.extract_eval_results`) reads the JSON object between the first `{` and last `}` after the marker. Missing marker / malformed JSON → `{'combined_score': -inf}` (a fail, not a crash).
- Scoring is domain-specific. Example CUDA scorer = `runtime_original_ms − runtime_custom_ms` (higher = faster). KernelBench style reports per-stage booleans (`compiled`, `correctness`, `runtime_success`, `loading_success`) + `runtime`/`runtime_original` + `metadata` (errors, traces, diffs); `classify_result` maps these to a coarse failure class for feedback.

### Reward-hacking defenses (`examples/cuda_engineering/evaluate.py`)
Against: Hack#1 (work on other streams → force `synchronize`), Hack#2 (thread injection → compare `threading.active_count`), Hack#3 (lazy/non-materialized output → check `data_ptr()`/storage size & device), Hack#5 (monkey-patching `torch.cuda.Event` → detect before/after running custom code).

## How feedback differs by style (the reason for `feedback.py`)
The original optimizer agent simply re-sent a free-form `feedback` string.
KernelBench gives **structured, score/error-dependent feedback**: the wording
depends on *why* a candidate failed (compile / runtime / incorrect / "make it
faster") plus the achieved speedup. To support both, `FeedbackStrategy` turns a
fresh `result` dict into a metadata dict merged back into `prompt_data`:
- `none` — `NoFeedback` (produce nothing).
- `generic` — `GenericFeedback` (original simple-style free-form `feedback` + `error_case`/`error_message`).
- `error_case` — `ErrorCaseFeedback` (structured KB-style: `error_case`, `error_message`, `previous_generation`, `score` = original/custom speedup). **Use this with `--use-kb-prompt`.**
- `score` — `ScoreFeedback` (compares candidate score against the previous one; generic scalar-optimization).
Select via `--feedback-strategy`.

## Run
```bash
# simple style (needs an OpenAI-compatible server on the model URL)
optimizer-agent --baseline b.py --evaluate e.py --reference r.py \
  --prompt-config-file cfg.json --num-iterations 2 --num-islands 2 \
  --num-map-elites 3 --use-strategy --use-feedback \
  --predecessor-selection-method map_elites

# kernel-bench style
optimizer-agent --use-kb-prompt --prompt-config-file examples/cuda_engineering/prompt_config_kb.json \
  --baseline examples/cuda_engineering/baseline.py --reference examples/cuda_engineering/reference.py \
  --evaluate examples/cuda_engineering/evaluate.py --num-map-elites 3 --feedback-strategy error_case

# CPU-only smoke test (random scorer, FakeLLMServer)
optimizer-agent --baseline examples/test_example/baseline.py \
  --evaluate examples/test_example/evaluate.py --reference examples/test_example/reference.py \
  --num-iterations 2 --use-strategy --use-feedback --use-debug --test
```
Start an OpenAI-compatible server first (SGLang or vLLM with tool-call/reasoning parsers) — see README. `--test` uses `FakeLLMServer` and never touches the network.

## benchmark.py / CUDA-graph timing (`benchmark.py`)
Captures the forward pass into a CUDA graph **before** benchmarking and times
replay, so the measurement reflects the kernel rather than host-side launch/
dispatch overhead; surfaces otherwise-silent CUDA errors up front (fail-fast),
and falls back to event-based timing when a graph can't be captured. API:
`TimingStats.mean/std/min/max/num_trials/hardware/device` and `benchmark(model,
inputs, method="auto"|"graph"|"event")`. torch/numpy imported lazily so the
dataclass is usable without CUDA. GPU required to actually time.

## Known bugs / caveats
- Map-elites clustering is simplistic (string-based `textdistance` Jaccard, hardcoded threshold 0.2, no feature vector).
- `KernelBenchPromptBuilder` requires the `robust_kernelbench/` checkout next to the package; it raises a clear `FileNotFoundError` if the `kb_prompts/` dir is absent, and silently drops example/hardware blocks whose asset files can't be resolved.
- `--kb-gpu` must name a GPU present in `robust_kernelbench/kb_prompts/hardware/gpu_specs.py`; unknown names raise `KeyError`.
- `agent_strategies.py` is a non-functional stub; the two-stage (improve + debug) logic lives across `agent.py` + `generation.py`.
- `FakeLLMServer` returns random text — smoke tests exercise control flow but not real correctness.
