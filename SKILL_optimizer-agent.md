# SKILL: optimizer-agent

A library for **LLM-agent based search over code / kernel optimization**. An
LLM (served via an OpenAI-compatible endpoint — SGLang/vLLM) proposes improved
source (`ModelNew`), an **external, user-supplied evaluator** scores the
candidate (correctness + timing, or any domain metric), and a genetic /
map-elites search keeps the best and iterates.

This is the **refactored v2**: the agent was rewritten to be **generic and
agnostic** — it knows nothing about CUDA, kernels, or prompt *content*. It only
manages search control flow. Everything domain-specific (the CUDA scorer, CUDA
graph timing, prompt wording) lives in `examples/` or the supplied evaluator
scripts. The old single-file monolith `optimizer_agent_ai.py` is the
pre-refactor planning skeleton (gitignored) and must not be treated as current.

Author: Nikolai Rozanov. Python >= 3.9. Runtime deps: `requests`,
`textdistance==4.6.3` (torch/`tomllib`/numpy are imported lazily where used).

## Layout
- `optimizer_agent/` — the shipped, working package (installed via `pip install -e .`, entry point `optimizer-agent`).
- `optimizer_agent_ai.py` — gitignored, the **pre-refactor planning skeleton** (single monolithic file). Do not treat as the current code.
- `optimizer_agent/agent_strategies.py` — gitignored, non-functional high-level strategy stub.
- `examples/test_example/` — random-scorer smoke test (no GPU needed).
- `examples/cuda_engineering/` — the real CUDA kernel scorer (`runtime_original − runtime_custom`) with reward-hacking defenses + a JSON prompt config.
- `robust_kernelbench/` — the KB asset repo; used only as a *reference* for the feedback rules. The package **never imports** it (verified).

## Core modules (`optimizer_agent/`)
- `agent.py::Agent` — the **orchestrator**. Runs iterations -> islands -> generations; wires a prompt builder + LLM + subprocess evaluation + result-driven feedback + logging. It is **agnostic to prompt content** (content lives in a swappable JSON config) and **agnostic to the problem domain** (only the evaluator contract matters). Knows the control flow: select predecessors, build prompts, call the model, evaluate, enrich prompt_data with feedback, record artifacts under `experiments_dir/`.
- `generation.py::GenerationPipeline` — runs a single **core** LLM stage built from a generic `PromptBuilder`; also `generate_debug` for the debug loop. Extracts the final ```-fenced code block from a core response.
- `prompts_v2.py` — the **single rules + generic builder module** (see below). This is where all feedback rules and the prompt rendering live; it does **not** import `robust_kernelbench`.
- `evaluation.py` — `evaluate_script()` runs a user-supplied `evaluate.py` as a **subprocess** with timeout; `EvalResult` dataclass (`status`, `result`, `stdout`, `stderr`); `parse_evaluation` robustly extracts the JSON after the `=== Evaluation Result ===` marker.
- `selection.py` — candidate populations: `CandidatesBestNAll`, `CandidatesBestNLast`, `MapElitesCandidates` (clusters generated code via `textdistance` Jaccard, threshold 0.2).
- `llm.py` — `BaseServer`/`LLMServer` (calls `.../v1/chat/completions`, lazy `requests`), `FakeLLMServer` for `--test` (never touches the network, returns a random string).
- `utils.py` — `extract_final_code_response` (pulls out ``` fenced code ```), `extract_eval_results` (parses JSON after marker), `STATUS_*` classification constants.
- `cli.py` — argparse → `optimizer-agent ...` entry point (see Run below).
- `tests/test_units.py` — 19 GPU-free unit tests (prompt building, `classify_result`, `compute_score`, all feedback strategies, evaluation parsing).

## Optimization loop (`Agent.run`)
```
for iteration:
  for island:
    iter 0: build candidates from baseline (num_samples or num_map_elites)
    iter>0: select best predecessors, mutate (map_elites key = Jaccard cluster of code)
  for candidate:
    1. GenerationPipeline.generate() -> new ModelNew code (single core stage)
    2. subprocess: python evaluate.py --solution <code> [--reference <ref>]
    3. parse result -> scores; update population
    4. apply result-driven FeedbackStrategy(prompt_data, result)  (single "v2" rules path)
    5. on failure + use_debug: debug loop (num_debug_iterations) planning around the error
predecessor_selection_method: bestN_last | bestN_all | map_elites
```
Key params: `num_iterations, num_islands, num_samples, num_debug_iterations,
predecessor_selection_method, num_map_elites, use_debug, feedback_strategy_name,
prompt_config`.

## The generic rules module (`prompts_v2.py`)
Everything here is domain-agnostic and does **not** import `robust_kernelbench`.

* `classify_result(result)` — score-independent failure class, mirroring KB's
  per-stage checks: compile -> runtime/load -> correctness -> improve. Returns
  `empty | eval_error | compile_error | runtime_error | incorrect | improve`.
  Works with either the generic result shape (`runtime`/`runtime_original`) or
  the CUDA evaluator's shape (`runtime_custom_ms`/`runtime_original_ms`,
  `loading_success`).
* `extract_error_message(result)` — best-effort error/traceback text.
* `compute_score(result)` — a scalar reward: an explicit `combined_score` field
  wins; else the speed-up factor `runtime_original / runtime` (+ inf / 0.0 edge
  cases).
* `GenericFeedbackStrategy` — the single rules path. Turns a fresh `result` dict
  into a metadata dict (`error_case`, `error_message`, `score`,
  `previous_generation`, `prev_score`, plus `metadata.*`) and a short
  `feedback` string; the agent merges it back into `prompt_data`. `NoFeedback`
  produces nothing. Registry: `none` / `v2`, selected via
  `--feedback-strategy` (default `v2`).
* `PromptBuilder` — a tiny, config-driven template renderer. A config is the
  shape `{"system": {order, prompts}, "user": {order, prompts}}`; each slot is a
  format string interpolated from `prompt_data`, and a slot whose placeholder is
  missing is **silently dropped** (so `## Feedback:` disappears before feedback
  exists). `DEFAULT_PROMPT_CONFIG` is the bundled neutral, domain-agnostic prompt.

## How evaluation works
- The evaluator is **user-supplied** — the agent only knows the contract.
  Signature: `python evaluate.py --solution SOL [--reference REF]`.
- It must print a line `=== Evaluation Result ===` followed by `json.dumps({...})`.
- `utils.extract_eval_results` reads the JSON between the first `{` and last `}`
  after the marker. Missing marker / malformed JSON → `{'combined_score': -inf}`
  (a fail, not a crash).
- `evaluate_script` always runs the subprocess with a timeout; non-zero exits are
  captured as failures (`eval_error`) rather than propagating. Raw stdout/stderr
  are carried on `EvalResult` for logging/diagnostics.
- Scoring is **domain-specific** and decided by the evaluator. Example CUDA scorer
  = `runtime_original_ms − runtime_custom_ms` (higher = faster). The agent stays
  agnostic by classifying results with `classify_result` and scoring with
  `compute_score` in `prompts_v2.py`.

### Reward-hacking defenses (`examples/cuda_engineering/evaluate.py`)
Against: Hack#1 (work on other streams → force `synchronize`), Hack#2 (thread
injection → compare `threading.active_count`), Hack#3 (lazy/non-materialized
output → check `data_ptr()`/storage size & device), Hack#5 (monkey-patching
`torch.cuda.Event` → detect before/after running custom code).

## Run
```bash
# simple style (needs an OpenAI-compatible server on the model URL)
optimizer-agent --baseline b.py --evaluate e.py --reference r.py \
  --prompt-config-file cfg.json --num-iterations 2 --num-islands 2 \
  --num-map-elites 3 --use-debug \
  --predecessor-selection-method map_elites --feedback-strategy v2

# CPU-only smoke test (random scorer, FakeLLMServer, no GPU, no network)
optimizer-agent --baseline examples/test_example/baseline.py \
  --evaluate examples/test_example/evaluate.py --reference examples/test_example/reference.py \
  --num-iterations 2 --num-map-elites 2 --use-debug --test
```
Start an OpenAI-compatible server first (SGLang or vLLM). `--test` uses
`FakeLLMServer` and never touches the network; it still needs real
`--baseline`/`--evaluate` files (it reads the baseline and calls the evaluator).
Known caveats: map-elites clustering is simplistic (string-based
`textdistance` Jaccard, hardcoded threshold 0.2, no feature vector);
`agent_strategies.py` is a non-functional stub; `FakeLLMServer` returns random
text, so smoke tests exercise control flow but not real correctness.
