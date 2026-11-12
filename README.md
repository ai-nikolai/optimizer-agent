# Optimizer Agent

A library to auto optimize and discover things with an LLM agent.

An LLM proposes an improved artifact, an external `evaluate.py` scores it, and a
small search loop keeps candidates and iterates. The agent knows nothing about
the problem domain: it only knows the evaluator contract (print
`=== Evaluation Result ===` followed by a JSON object with a scalar score;
`combined_score` by default).

## How it works

```
for iteration:
  iteration 0 -> generate from the baseline
  iteration >0 -> generate from the selected predecessor artifact
  candidate:
    1. optional strategy stage -> plan
    2. core stage -> new artifact
    3. evaluate.py --solution ... [--reference ...]
    4. optional feedback stage (uses the real result)
    5. optional debug loop -> regenerate until the evaluation succeeds
```

Selection policies: `bestN_last` (best from the previous generation),
`bestN_all` (best-ever, bounded top-K pool), and `map_elites` (taxonomy or
similarity cells, best per cell). Artifacts are written per generation under
`experiments_dir/`; a final `summary.json` records every candidate and the
best one.

Prompt content is a JSON config (see `examples/*/prompt_config.json`); built-in
defaults are used when `--prompt-config-file` is omitted. Both the historical
nested shape (`{"improve": ..., "debug": ...}`) and a flat
(`{"system": ..., "user": ...}`) config are accepted.

### Search and feedback parameters

Two independent choices control a run:

1. **Predecessor selection** — how the next parent is chosen.
2. **Feedback strategy** — what the model is told after an evaluation.

The classification table below is used by the `rules` / `rules_llm` strategies;
`prompt`, `none`, `llm` and `evaluator` can bypass it.

After every evaluation the agent classifies the attempt, in the spirit of
`robust_kernelbench`'s multi-stage repair prompts. Classification is
domain-agnostic; a fine `failure_stage` always maps to a coarse `failure_kind`:

| `failure_kind` | Fine stages | Meaning |
| --- | --- | --- |
| `empty` | `empty` | model returned no artifact |
| `eval_error` | `eval_error` | evaluator failed / returned no score |
| `invalid` | `invalid`, `incorrect`, `compile_error`, `runtime_error` | artifact produced but not accepted |
| `valid` | `improve` | artifact accepted and scored |

New, non-code evaluators should return `valid` / `success` / `passed` (bool),
optionally `error_kind` / `error_type` (`constraint`, `timeout`, ...), and the
selection score. They can even return `failure_stage` or `failure_kind`
directly; that always wins over heuristics. Kernel-style flags (`compiled`,
`loading_success`, `correctness`, ...) remain as optional compatibility
signals, not requirements.

The stage, kind, a deterministic hint, the error text, the selection score and
a metric change are merged into `prompt_data`. Default prompts use
`{failure_kind}`, `{failure_stage}` and `{feedback}`. `--use-feedback` adds an
optional LLM reviewer that sees the deterministic hint and can replace it with
a richer explanation.

#### Candidate / predecessor selection

Selection and feedback are **independent**; any selection method can be paired
with any feedback strategy.

| Method | Parent(s) for the next round | Typical use |
| --- | --- | --- |
| `bestN_last` | best candidate from the **previous generation** | robust_kernelbench repair with `num_samples=1`; sequential hill-climb with more samples |
| `bestN_all` | best candidate across **all generations** | graph-ish memory / best-ever restart |
| `map_elites` | top `num_map_elites` cells | diversity / multi-start |

`bestN_all` keeps a bounded top-`max_candidates` pool (default 20); only the
best-ever candidate is used as the parent.

Search parameters can live in the task config:

```json
{
  "num_iterations": 3,
  "num_samples": 1,
  "predecessor_selection_method": "bestN_last",
  "num_map_elites": 1
}
```

or on the CLI:

```bash
--num-iterations 3 --num-samples 1 \
--predecessor-selection-method bestN_last --num-map-elites 1
```

`num_samples` is the number of children generated per selected parent (on
round 0 it is also the number of initial candidates). There is currently one
population per run; independent runs are separate processes/experiments.

> **Islands were removed.** They had no migration/crossover, so `num_islands`
> only duplicated independent chains. Use separate experiments for independent
> runs, and `num_samples` for multiple children per round.

#### Feedback strategies

Feedback is a pluggable strategy, selected per task:

| Strategy | What it does | LLM call |
| --- | --- | --- |
| `none` | no hint; only raw `result`/`scores` are available | no |
| `prompt` | no hint; prompt config uses `{result}` / `{scores}` directly | no |
| `rules` | classify + deterministic hint (generic or kernel profile) | no |
| `rules_llm` | `rules`, then an LLM reviewer is appended | yes |
| `llm` | LLM reviewer sees the raw result; no deterministic hint | yes |
| `evaluator` | use the task evaluator's own `feedback` / `failure_stage` | no |

`--use-feedback` is a legacy alias that turns `rules` into `rules_llm`,
`evaluator` into `rules_llm`-style augmentation, and `none`/`prompt` into `llm`.

A prompt config can select the strategy:

```json
{
  "feedback_strategy": "rules",
  "feedback_profile": "generic",
  "feedback_hints": {
    "invalid": "The proposal violated a constraint ({error_kind}). Fix it.{detail}",
    "valid": "Accepted. Improve {metric_name}. {metric_summary}"
  }
}
```

For a task-specific setup, keep feedback in its own file and pass
`--feedback-config-file`:

```json
{
  "strategy": "rules_llm",
  "profile": "generic",
  "hints": {
    "invalid": "The proposal was rejected ({error_kind}). Fix it.{detail}",
    "valid": "Accepted. Improve {metric_name}. {metric_summary}"
  },
  "metric": {
    "score_key": "quality",
    "baseline_key": "baseline_score",
    "higher_is_better": true,
    "metric_name": "quality"
  }
}
```

Precedence: prompt config < feedback config file < explicit CLI flags
(`--feedback-strategy`, `--feedback-profile`). Examples live in
`examples/feedback/`. The `evaluator` strategy is the escape hatch for domains
whose failure taxonomy only the evaluator knows: return `feedback`, `valid`,
and optionally `failure_stage` / `failure_kind` / `error_kind` from
`evaluate.py`.

#### Combining selection and feedback

They are orthogonal. A single task config can set both:

```json
{
  "num_iterations": 3,
  "predecessor_selection_method": "bestN_last",
  "num_samples": 1,
  "feedback_strategy": "rules_llm",
  "feedback_profile": "kernel",
  "stage_prompts": {
    "compile_error": { "core": { "...": "whole prompt for compile failures" } },
    "improve":       { "core": { "...": "whole prompt for accepted-but-slow" } }
  }
}
```

Common combinations:

| Goal | `predecessor_selection_method` | `feedback_strategy` |
| --- | --- | --- |
| robust_kernelbench repair rounds (`num_samples=1`) | `bestN_last` | `rules` or `rules_llm` (kernel profile) |
| sequential hill-climb (multiple children) | `bestN_last` | `rules` / `rules_llm` |
| best-ever restart | `bestN_all` | `rules_llm` |
| diverse search | `map_elites` | `rules_llm` |
| evaluator owns the failure taxonomy | `bestN_last` | `evaluator` |
| pure prompt-driven repair (no classifier) | `bestN_last` | `prompt` / `none` |
| LLM critic only, no deterministic hint | `bestN_last` | `llm` |

CLI equivalent:

```bash
optimizer-agent \
  --baseline problem.py --reference problem.py --evaluate evaluate.py \
  --predecessor-selection-method bestN_last \
  --feedback-strategy rules_llm \
  --feedback-profile kernel \
  --feedback-config-file examples/feedback/kernel_rules_llm.json \
  --num-iterations 3 --num-samples 1
```

Precedence:
- **Search**: explicit CLI/constructor argument > task/prompt config > default.
- **Feedback**: CLI > feedback config file > prompt config > default `rules`.
- `--use-feedback` upgrades the selected strategy: `rules → rules_llm`,
  `evaluator → augmented`, `none/prompt → llm`.

A custom strategy can be registered programmatically:

```python
from optimizer_agent.feedback import FeedbackStrategy, register_feedback_strategy

class ReflectionStrategy(FeedbackStrategy):
    name = "reflection"
    has_hint = True
    def apply(self, prompt_data, eval_result, artifact="", score_key="combined_score"):
        ...

register_feedback_strategy("reflection", ReflectionStrategy, aliases=("reflexion",))
```

Feedback wording also lives in a profile:

- `generic` (default) -- "artifact", "accepted", `{metric_name}`/`{metric_summary}`
- `kernel` -- compile/runtime/correctness wording and runtime speed-up
- a dict via `prompt_config["feedback_profile"]` for a custom domain

### Artifact extraction and diff mode

Select the extractor with `--response-format`:

- `last_code` (default): last code-tagged fenced block, else last block
- `last`: last fenced block
- `largest`: largest fenced block
- `code`: largest code-tagged block, else largest
- `python`: last Python-tagged block, else last block
- `raw` / `text`: the whole response
- `json`: fenced block or the outermost JSON object/array

`--artifact-extension` controls the candidate file suffix.

`--diff-based` asks the model for an edit instead of the full artifact:

```bash
--diff-based --diff-mode unified-diff   # default
--diff-based --diff-mode json
```

Unified diffs are applied with `patch`; JSON mode expects
`{"edits":[{"old":"...","new":"..."}]}` and replaces the first occurrence. If a
diff cannot be applied, the agent logs the failure and falls back to treating
the response as a complete artifact.

### Map-elites cells

`--map-elites-method cluster|fixed|dynamic`:

- **cluster**: numerical similarity. `--map-elites-cluster-metric jaccard`
  (default) or `embedding`. Embedding mode requires `--embedding-url` and
  `--embedding-model`; no silent fallback.
- **fixed**: an LLM assigns each artifact to a configured taxonomy.
- **dynamic**: the LLM assigns to an existing class or creates a new class if
  none fits. Add `--map-elites-taxonomy-file` (JSON map of name → description).
  Failures after retries go to an `unclassified` cell.

```json
{
  "map_elites": {
    "method": "dynamic",
    "taxonomy": {
      "memory_bound": "limited by memory bandwidth",
      "compute_bound": "limited by arithmetic throughput"
    },
    "cluster_metric": "jaccard",
    "cluster_threshold": 0.2,
    "max_cells": 100
  }
}
```

The classifier prompt is the top-level `classify` stage (default built in).

### Run lifecycle

- `--max-candidates` bounds the `bestN_all` pool (default 20).
- `--minimize` makes the selection score lower-is-better; raw scores are still
  reported, ranking is inverted internally.
- `--resume` is on by default: a completed `summary.json` skips the run.
  Interrupted problems restart from scratch (completed-problems-only resume);
  use `--no-resume` to rerun a completed experiment.
- `--overwrite` deletes the experiment directory (with safety guards) first.
- Each run writes `config.json`, `run.log`, `events.jsonl`, `metadata.txt`,
  per-sample artifacts, and `summary.json`.
- LLM calls retry with backoff; after retries they raise a clear error rather
  than silently producing an empty artifact. Evaluator stdout/stderr are
  written to files and only a tail is used in prompts.

### FIL / robust_kernelbench mapping

`robust_kernelbench` uses the FIL label for its multi-stage repair loop:
compile error → runtime error → incorrect → improve, with the previous attempt
and its error/score. That behavior is supported through `rules` / `rules_llm`
with the `kernel` profile; there is no separate FIL class.

### Deliberately deferred

- The old `overall_strategy` layer (`sequential` / `parallel` / `tree` / `graph`)
  is not a separate switch. Use `bestN_last` / `bestN_all` / `map_elites` with
  `num_samples`; graph crossover remains a future feature.
- Independent runs are separate experiments/processes now that islands are
  removed.
- A separate FIL/TwoStage strategy class is not needed: robust_kernelbench’s
  FIL-related behavior is its multi-stage repair loop, covered by
  `rules`/`rules_llm`.

### Security

Evaluator subprocesses execute user-supplied and model-generated code. The agent
does not sandbox them; run untrusted tasks inside a container/VM with restricted
network and filesystem access.

A complete non-code example lives in `examples/generic_task/` (tagline
optimization, raw text artifacts, evaluator-supplied feedback).

For a KernelBench / `robust_kernelbench` replication there is
`examples/kernelbench/`: a task-specific evaluator around
`eval_kernel_against_ref`, a KernelBench prompt config, and a `run_level.py`
driver. It uses `--predecessor-selection-method bestN_last` with
`num_samples=1`, so each round repairs the **previous** candidate like
robust_kernelbench, and supports `stage_prompts` for whole-prompt switching per
failure stage. See
`examples/kernelbench/README.md` for the exact commands.

This is **not a new agent type**: it is a feedback / repair strategy inside the
existing generation loop. A new agent would only be warranted if it needed its
own tools or multi-step tool-use loop (for example, an agent that inspects
compiler logs and runs experiments interactively). The stable extension points
are `feedback_strategy`, `failure_kind`/`failure_stage`, `feedback_hints`, and
the metric/profile config.

### Layout

- `optimizer_agent/main.py` - `Agent` search loop
- `optimizer_agent/candidates.py` - populations and predecessor selection
- `optimizer_agent/feedback.py` - failure classification, hints, profiles, strategies
- `optimizer_agent/diffs.py` - unified-diff / JSON-edit application
- `optimizer_agent/prompts.py` - default prompts + config-driven builder
- `optimizer_agent/llm.py` - OpenAI-compatible client (+ fake server)
- `optimizer_agent/utils.py` - response/eval parsing and evaluator subprocess
- `optimizer_agent/cli.py` - the `optimizer-agent` command
- `tests/test_units.py` - GPU-free unit and integration tests

### Tests

```bash
python -m unittest discover -s tests -v
```

## Getting started:


1. Installation
```bash
python3 -m venv env_optagent
source env_optagent/bin/activate

pip3 install -e .
```

2. Running example:
```bash
# ./examples/test_example/run.sh
./examples/cuda_engineering/run.sh

# Run the model first
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
--model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
--tensor-parallel-size 1 \
--host 0.0.0.0 \
--port 30000 \
--enable-memory-saver
```

```bash
# ornith-ai/Ornith-1.5-35B-A3B
# ornith-ai/Ornith-1.5-9B
CUDA_VISIBLE_DEVICES=0 vllm serve ornith-ai/Ornith-1.5-9B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 30000 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

## Other useful commands:
```bash
salloc --partition=interactive-gpu --gres=gpu:h200:1 --time=08:00:00 --ntasks=1 --cpus-per-task=16 --gpu-bind=closest
srun --pty --overlap --jobid=1048235 bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate
conda activate env_model2 #sglang
conda activate env_kb2 #kb
conda activate env_vllm30 #vllm (model)
```

 
```bash 
# to install on cluster.
export TMPDIR=/hx2-weka/home/nr1713/tmp
mkdir -p "$TMPDIR"
pip install --no-cache-dir vllm
```

## Analysis

Experiment summaries can be collected and explored with the tools in
`analysis/`:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/collect_results.py --experiments-dir experiments --version v5_0
python analysis/interactive.py --data-dir experiments --version v5_0
```

See `analysis/README.md` for details.

## (C) Nikolai Rozanov - 2026 - Present