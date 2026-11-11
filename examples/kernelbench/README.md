# Replicating robust_kernelbench with optimizer-agent

This directory maps robust_kernelbench's multi-round KernelBench loop onto the
optimizer-agent:

| robust_kernelbench | optimizer-agent |
|---|---|
| one reference problem from the KernelBench dataset | `baseline` + `reference` file for one `Agent` run |
| `trial 1` one-shot generation | `iteration 0` |
| `trial N` repairs `trial N-1` | `--predecessor-selection-method bestN_last` + `num_samples=1` + `num_iterations` |
| `failure_stage(sample)` | `feedback.classify_result()` (`compile_error`, `runtime_error`, `incorrect`, `improve`, ...) |
| `select_prompt_spec` / `kb_multi_stage` | `feedback_strategy` + `feedback_hints`; optional `stage_prompts` for whole-prompt switching |
| `get_prompt_kb_tts_multi_stage` | `prompt_config.json` (KernelBench instructions) |
| `evaluate_single` | `evaluate.py` adapter (same `eval_kernel_against_ref`) |
| `evaluations.csv` | per-problem `summary.json` + driver `results_shardN.csv` |
| dataset-wide rounds across 100 problems | one sequential repair chain per problem; shard across GPUs for throughput |

## Files

- `evaluate.py` — task-specific evaluator. Calls
  `robust_kernelbench.evaluate_single.eval_kernel_against_ref` and maps its
  result to the generic evaluator contract.
- `prompt_config.json` — KernelBench instructions, `ModelNew`/`load_inline`
  output format, and the repair-feedback slot.
- `run_level.py` — loads a KernelBench level, writes each problem to a file, and
  runs one `Agent` per problem with `bestN_last` and `num_samples=1`.
- `../feedback/kernel_rules_llm.json` — kernel profile + `rules_llm` feedback.

## Exact command (server already running)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_kb2
cd /hx2-weka/home/nr1713/optimizer-agent-simple
export PYTHONPATH=$PWD
export ROBUST_KERNELBENCH_ROOT=/hx2-weka/home/nr1713/robust_kernelbench
export CUDA_VISIBLE_DEVICES=1          # evaluator GPU; server on another GPU

python examples/kernelbench/run_level.py \
    --level 1 \
    --rounds 3 \
    --limit 5 \
    --model-url http://localhost:30000 \
    --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --experiments-dir experiments/opt_agent_kb_level1
```

`--prompt-config` and `--feedback-config` default to the files above. The
evaluator defaults to KernelBench's 5 correctness trials and 3 performance
trials; override with environment variables:

```bash
export KB_NUM_CORRECT_TRIALS=5
export KB_NUM_PERF_TRIALS=3
export KB_FLUSH_L2=1
# export KB_ATOL=1e-2 KB_RTOL=1e-2
```

## Parallel shards

The driver is sequential per problem. Run shards in separate processes (each on
its own GPU, all pointing at the same model server):

```bash
# GPU 1
CUDA_VISIBLE_DEVICES=1 python examples/kernelbench/run_level.py \
    --level 1 --rounds 3 --num-shards 4 --shard-index 0 ...
# GPU 2
CUDA_VISIBLE_DEVICES=2 python examples/kernelbench/run_level.py \
    --level 1 --rounds 3 --num-shards 4 --shard-index 1 ...
# ...
```

Each shard writes `results_shardN.csv` and per-problem directories under
`experiments/.../level_1_pid_<id>/`. A problem that raises is recorded with an
`error` column and the shard continues. Completed problems are skipped on rerun
(`--no-resume` to force a rerun; `--overwrite` to delete the run directories).

## Control flow per problem

```
reference/problem file  ──► prompt (round 1)
        │
        ▼
   model generates ModelNew
        │
        ▼
evaluate.py: eval_kernel_against_ref
        │  compiled / loading / correctness / runtime
        ▼
feedback strategy (kernel profile, rules_llm)
        │  failure_stage + hint (+ LLM reviewer)
        ▼
round 2 prompt starts from the previous candidate
        │
        ▼  ... rounds ...
summary.json + results_shardN.csv
```

With `predecessor_selection_method="bestN_last"` and `num_samples=1`, each
generation contains exactly one candidate, so the next round repairs the
previous candidate — exactly robust_kernelbench's trial-to-trial behavior.
Final scoring still records the best candidate per problem in
`results_shardN.csv`.

### Mixing selection and feedback

The driver exposes both axes independently. The default for an exact
robust_kernelbench loop is:

```bash
--predecessor-selection-method bestN_last \
--feedback-strategy rules_llm \
--feedback-profile kernel
```

Other useful mixes:

| Goal | Selection | Feedback |
| --- | --- | --- |
| exact robust_kernelbench repair | `bestN_last` + `num_samples=1` | `rules_llm` + kernel profile |
| deterministic, no LLM reviewer | `bestN_last` + `num_samples=1` | `rules` + kernel profile |
| hill-climb on the best kernel | `bestN_last` | `rules_llm` |
| diverse kernel families | `map_elites` | `rules_llm` |
| evaluator supplies its own repair text | `bestN_last` | `evaluator` |
| prompt-only, no classifier | `bestN_last` | `prompt` / `none` |

Example with a non-default feedback strategy:

```bash
python examples/kernelbench/run_level.py \
    --level 1 --rounds 3 --limit 5 \
    --predecessor-selection-method bestN_last \
    --feedback-strategy rules \
    --model-url http://localhost:30000 \
    --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct
```

`run_level.py` exposes both axes as flags: `--predecessor-selection-method`,
`--feedback-strategy`, and `--feedback-profile`. Explicit flags win over the
task config; otherwise the keys in `prompt_config.json` / the feedback file are
used.


## Replacing the repair hint with whole stage-specific prompts

By default the KernelBench instruction is stable and the failure-specific text
is injected through `{feedback}`. For exact `kb_multi_stage`-style prompt
switching, add `stage_prompts` to the prompt config:

```json
{
  "improve": { "core": { "...": "base prompt" } },
  "stage_prompts": {
    "compile_error": {
      "system": {"order": ["system"], "prompts": {"system": "You are a CUDA compiler-error fixer."}},
      "user": {
        "order": ["task", "artifact", "error", "reference"],
        "prompts": {
          "task": "## Task: Fix the compilation error. Return complete code.",
          "artifact": "## Current implementation:\n{current_artifact}",
          "error": "## Compiler error:\n{error}",
          "reference": "## Reference PyTorch implementation:\n{reference}"
        }
      }
    },
    "runtime_error": { "...": "..." },
    "incorrect": { "...": "..." },
    "improve": { "...": "..." }
  }
}
```

`stage_prompts` values are merged over the base bundle, so strategy/feedback
sections are inherited unless overridden. The stage is the **previous round's**
`failure_stage`, exactly like robust_kernelbench's `select_prompt_spec`.

## Current differences

- Evaluation runs in-process through `eval_kernel_against_ref`, not
  robust_kernelbench's multi-process compile/evaluate scheduler. The underlying
  correctness/timing logic is the same.
- Problems are processed per-problem (rounds 1..N) rather than dataset-wide
  trial-by-trial. Use shards for throughput; per-problem results are equivalent.
- The driver records the best candidate per problem, while the repair chain uses
  the latest candidate. Both are in the output for analysis.
