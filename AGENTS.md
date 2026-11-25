# AGENTS.md — benchmarking + optimizer-agent improvement hand-off

Everything needed to continue this work on a new machine: what exists, how to
run it, what is scheduled, what is broken, and what to do next. Read this
before touching anything.

Last updated: 2026-11-25.

---

## 1. What this work is

Two goals:

1. **Benchmark optimizer-agent** against OpenEvolve and a recursive (AKD-style)
   baseline on **AlgoTune (154 tasks)** and **KernelBench level 1 (100
   problems)**, with identical evaluators, prompts, feedback and token
   accounting.
2. **Improve optimizer-agent itself** based on that evidence
   (`benchmarking/IMPROVEMENT_PLAN.md`).

All benchmark code lives in the `optimizer-agent` repository on
`feat/benchmarking`. No experiment code lives anywhere else (the only external
pieces are the upstream AlgoTune tasks, the `robust_kernelbench` evaluator and
the OpenEvolve competitor package).

## 2. New-server bootstrap

**Two repositories are required. Everything else is third-party.**
`robust_kernelbench` is not vendored and cannot be installed from this repo: a
new agent must pull it manually (step 2 below) before any KernelBench run works.

| Required | Why |
|---|---|
| `ai-nikolai/optimizer-agent` (this repo, `feat/benchmarking`) | the agent and the whole benchmark harness |
| `ai-nikolai/robust_kernelbench` | the KernelBench **evaluator** — imported, not vendored. This must be pulled separately. |

| Third-party, only if you run that benchmark | Why |
|---|---|
| `oripress/AlgoTune` | task implementations behind the AlgoTune evaluator |
| `ai-nikolai/openevolve` | the OpenEvolve competitor |

**Not required: the historical AKD checkout (`automatic_knowledge_discovery`).**
Nothing in this repository imports it. The old KernelBench results that were
produced there are recorded in `COMPARISON.md` for provenance only; the `clean_`
batch reproduces the same experiments from this repo.

```bash
# 1. our code
git clone https://github.com/ai-nikolai/optimizer-agent.git
cd optimizer-agent && git checkout feat/benchmarking

# 2. the evaluator we import (must be >= the commit that fixed structured
#    correctness failures; see ROBUST_KERNELBENCH_FEEDBACK.md)
# MANUAL STEP: pull the evaluator. It is imported, not vendored.
git clone https://github.com/ai-nikolai/robust_kernelbench.git ../robust_kernelbench
git -C ../robust_kernelbench log --oneline -1      # expect 1139ebc or newer
pip install -e ../robust_kernelbench                # editable, so the checkout is the source of truth

# 3. third-party, per benchmark
git clone https://github.com/oripress/AlgoTune.git ../AlgoTune
git clone https://github.com/ai-nikolai/openevolve.git ../openevolve

# 4. environments
source <conda>/etc/profile.d/conda.sh
conda activate env_kb2      # agent + evaluators (torch, numpy, scipy, numba)
pip install -e .
# env_vllm30 is only for the vLLM server (OpenAI-compatible API)

# 5. environment variables (launchers default to $HOME/<name>; export these if
#    your checkouts live elsewhere)
export ALGOTUNE_ROOT=$PWD/../AlgoTune
export ROBUST_KERNELBENCH_ROOT=$PWD/../robust_kernelbench
export OPENEVOLVE_ROOT=$PWD/../openevolve

# 6. sanity — this prints the remediation for anything missing
python scripts/check_deps.py
python -m pytest tests -q            # 65 tests

# 7. submit from the repository root: the launchers write logs via relative
#    paths and derive REPO from their own location
sbatch benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh
```

If conda lives somewhere unusual, export `CONDA_SH=/path/to/conda.sh`; the
launchers fall back to `$HOME/miniconda3/etc/profile.d/conda.sh`.

Datasets are **already in the repo** — nothing to download for AlgoTune:

```text
benchmarking/tasks/algotune/dataset/        8 curated tasks   (~140 KB)
benchmarking/tasks/algotune/dataset_full/   154 tasks        (~3 MB)
benchmarking/tasks/kernelbench/dataset/     P1..P100         (~400 KB)
```

The official 328 GB AlgoTune instance dataset is **not** used or needed. The
lightweight 154-task manifest is also published at
`https://huggingface.co/datasets/ai-nikolai/AlgoTuneTasks`.

## 3. Repository map

```text
benchmarking/
  README.md                  layout + run matrix
  COMPARISON.md              settings, alignment, clean batch, results
  ANALYSIS.md                what went wrong in the four full runs
  IMPROVEMENT_PLAN.md        evidence-based proposals for optimizer-agent
  ROBUST_KERNELBENCH_FEEDBACK.md   upstream bug report for the evaluator
  aggregate_results.py       builds the comparison table (--tag clean)
  tasks/
    algotune/                dataset, manifests, evaluate.py, builders
    kernelbench/             dataset, evaluate.py, evaluate_legacy.py
  agents/
    prompts/kernelbench.json shared prompts (all KernelBench agents)
    prompts/algotune.json    shared prompts (optimizer-agent)
    feedback_config.json     shared feedback policy (rules_llm, speedup metric)
    recursive/               AKD-style recursive baseline (KernelBench)
    optimizer-agent/         optimizer-agent runners (AlgoTune + KernelBench)
    openevolve/              OpenEvolve competitor (both task families)
```

Evaluator contract: every agent is scored by **the same CLI per task family**:

- KernelBench → `tasks/kernelbench/evaluate.py` → `robust_kernelbench.evaluate_single`
- AlgoTune    → `tasks/algotune/evaluate.py` (official `task.is_solution`, manifest seeds)

OpenEvolve's generated `evaluator.py` files are thin subprocess adapters around
those CLIs (`agents/openevolve/*/evaluator_template.py`), so scores are
identical across agents.

## 4. How to run

All launchers are `sbatch`-ready and assume Slurm with `--gres=scratch`; they
put caches in `$TMPDIR` and leave only results/logs in home.

### AlgoTune, 154 tasks

```bash
# sequential, 10 rounds, 3 shards x 1 GPU
RUN_PREFIX=clean_ EXPERIMENT_KIND=seq NUM_SHARDS=3 ROUNDS=10 \
  sbatch --array=0-2 --time=12:00:00 \
  benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh

# advanced (map_elites x3 + strategy), 4 rounds
RUN_PREFIX=clean_ EXPERIMENT_KIND=adv NUM_SHARDS=3 ROUNDS=4 \
  SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 \
  USE_STRATEGY=1 USE_DEBUG=1 \
  sbatch --array=0-2 --time=12:00:00 \
  benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh

# OpenEvolve, 20 iterations, checkpoint every iteration, 8 shards x 1 GPU
RUN_PREFIX=clean_ NUM_SHARDS=8 TOTAL_TASKS=154 ITERATIONS=20 \
  CHECKPOINT_INTERVAL=1 PREPARE=0 \
  sbatch --array=0-7 --time=18:00:00 \
  benchmarking/agents/openevolve/algotune/run_full_slurm.sh
```

### KernelBench, 100 problems

```bash
# recursive baseline, 2 GPUs (vLLM + evaluator), 2 jobs covering P1-50 / P51-100
RUN_PREFIX=clean_ EXPERIMENT_NAME=recursive_kb_seq_a PROBLEM_START=1  PROBLEM_END=50  ROUNDS=5 \
  sbatch --time=24:00:00 benchmarking/agents/recursive/run_slurm.sh
RUN_PREFIX=clean_ EXPERIMENT_NAME=recursive_kb_seq_b PROBLEM_START=51 PROBLEM_END=100 ROUNDS=5 \
  sbatch --time=24:00:00 benchmarking/agents/recursive/run_slurm.sh

# recursive advanced
RUN_PREFIX=clean_ EXPERIMENT_NAME=recursive_kb_adv_a PROBLEM_START=1  PROBLEM_END=50  ROUNDS=5 \
  SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 USE_STRATEGY=1 USE_DEBUG=1 \
  sbatch --time=24:00:00 benchmarking/agents/recursive/run_slurm.sh
#   ... same for _b with P51-100

# OpenEvolve, 50 iterations, 4 shards x 2 GPUs
RUN_PREFIX=clean_ NUM_SHARDS=4 TOTAL_TASKS=100 ITERATIONS=50 PREPARE=0 \
  sbatch --array=0-3 --time=24:00:00 \
  benchmarking/agents/openevolve/kernelbench/run_full_slurm.sh
```

Small smoke tests (1 hour, 2 GPUs / 2 tasks):

```bash
LIMIT=2 ROUNDS=1 sbatch --time=1:00:00 benchmarking/agents/optimizer-agent/algotune/run_slurm.sh
PROBLEM_START=1 PROBLEM_END=2 ROUNDS=1 \
  EXPERIMENT_NAME=smoke_kb sbatch --time=1:00:00 benchmarking/agents/recursive/run_slurm.sh
```

Before an OpenEvolve run, prepare the task dirs (they contain the evaluator
adapters and initial programs):

```bash
# AlgoTune (needs the OpenEvolve example task dirs as --source)
python benchmarking/agents/openevolve/algotune/prepare_tasks.py \
  --source <openevolve>/examples/algotune --manifest benchmarking/tasks/algotune/tasks_full.json \
  --output benchmarking/agents/openevolve/algotune/tasks_full --checkpoint-interval 1 --clean

# KernelBench
ROBUST_KERNELBENCH_ROOT=$ROBUST_KERNELBENCH_ROOT \
  python benchmarking/agents/openevolve/kernelbench/prepare_tasks.py --level 1 --limit 100 --clean
```

## 5. Where results go / how to read them

```text
experiments/<run>/results.csv          one row per task (AlgoTune, OE)
experiments/<run>/results_shardN.csv   one row per problem (KernelBench recursive)
experiments/<run>/<task>/summary.json  per-task history, scores, token_usage
logs/                                  Slurm + vLLM logs
```

Aggregate a batch:

```bash
python benchmarking/aggregate_results.py --tag clean
```

Rules the aggregator applies (do not "fix" these without reading the history):

- **OpenEvolve token logs are cumulative per task** (`token_usage.jsonl` is
  appended by every attempt). One attempt per task = `max(total)/attempts`.
  Summing CSV rows over-counts (a 8-attempt shard bug inflated 20M to 87M).
- **`summary.json` history has one entry per round**, not per candidate: if the
  main candidate failed and debug ran, the round's entry is the debug result.
  Filename (`core_response.py` vs `core_response_debug.py`) identifies the stage.
- `valid` means correctness passed **and** the candidate was timed
  (`speedup > 0`); tasks with `speedup` exactly 1.0 are real results, not no-ops.

## 6. Current state (2026-11-26)

Clean batch submitted, tag `clean_`:

| Run | Jobs | Layout |
|---|---|---|
| AlgoTune sequential | `1772601` | 3 shards, 10 rounds |
| AlgoTune advanced | `1772602` | 3 shards, 4 rounds, map_elites x3 |
| OpenEvolve AlgoTune | `1772603` | 8 shards, 20 iterations |
| recursive KernelBench seq | `1772604`, `1772605` | P1-50, P51-100 |
| recursive KernelBench adv | `1772606`, `1772607` | P1-50, P51-100 |
| OpenEvolve KernelBench | `1772608` | 4 shards, 50 iterations |

Plus two 1-hour smoke jobs (`1772599` AlgoTune, `1772600` KernelBench).
Everything is pending behind the account's own `settings_*` backlog; check
`squeue -u $USER`. Expected: smoke runs first, full batch after.

Historical (pre-alignment) artifacts are still in `experiments/`:
`algotune_full_1689334/1689335`, `openevolve_algotune_full_1689375` +
`..._resume_1709405`, `openevolve_kernelbench_full_1670750`. Their numbers are in
`COMPARISON.md`; they predate the shared evaluators, so use the `clean_` batch
for any publication. (The even older KernelBench recursive runs lived in the AKD
checkout; they are **archival only** — nothing needs them and their conclusions
are restated in `ANALYSIS.md` / `RECURSIVE_ADVANCED_ANALYSIS.md`.)

## 7. Known issues (read before debugging a bad score)

1. **`robust_kernelbench` evaluator bug — fixed upstream (`1139ebc`).** The
   reversed unpack that turned every mismatch into
   `TypeError: 'str' object does not support item assignment` is gone, and the
   fix also adds `metadata.error/error_kind`, `max_abs_diff`, `max_rel_diff`,
   `mismatch_index`, `trial`, `seed`, `correctness_status` and
   `metadata.trial_results`, plus a `timeout_s` and regression tests. The report
   that drove it is `benchmarking/ROBUST_KERNELBENCH_FEEDBACK.md`; keep the
   checkout at or above `1139ebc`. Historical KernelBench runs predate the fix,
   so their rejection messages are the old `TypeError`.
2. **OpenEvolve AlgoTune timeouts.** Ten tasks (`lp_box`, `qp`, `lasso`,
   `lyapunov_stability`, `channel_capacity`, `chebyshev_center`, `lp_centering`,
   `minimum_volume_ellipsoid`, `ode_lorenz96_nonchaotic`, `ode_stiff_robertson`)
   hit the 600 s evaluator stage-2 timeout and the 3 h per-task cap. They cannot
   finish 20 iterations; raise `TIMEOUT_PER_TASK` only if you accept longer
   shards.
3. **AlgoTune solve timeout** is 120 s per call (`ALGOTUNE_SOLVE_TIMEOUT`).
   Large-`n` tasks can time out during evaluation.
4. **Format failures** are the largest avoidable loss on KernelBench
   (`ModelNew` not defined) — largely addressed in the shared prompts; verify the
   `clean_` batch before claiming it is fixed.
5. **Queue/congestion.** This cluster's GPU partition is shared and often
   backfills for days. `sacct` occasionally goes down (`slurmdbd` connection
   refused); use `squeue` + the run artifacts.
6. **Home quota is 1 TB.** Runs must use `--gres=scratch` and `$TMPDIR`
   (launchers already do); only results and logs belong in home.
7. **v11 (robust_kernelbench harness) is invalid** (0/100 valid in the fair
   run due to CUDA OOM). Do not use it in comparisons.

## 8. What to do next

Priority order from `IMPROVEMENT_PLAN.md` (evidence in that file):

1. Pre-evaluation format gate + one cheap re-ask (parse/compile/contract).
2. Structured failure payload (error kind, metric delta, tolerances, shapes,
   violated constraint) so repair stops guessing.
3. Debug economy: de-duplicate by error signature (44-50% of debug calls repeat
   a known error), shrink debug context, stop when a task is valid.
4. Adaptive stopping (no improvement for 3 rounds) + abandon never-valid tasks
   (61% of KernelBench tokens are spent there).
5. Safety-net first KernelBench candidate (correct unoptimized `ModelNew`,
   optimize from round 1).
6. Task-aware budgets: `matrix_operations` (19% of tasks, ~1.0x) needs 1-2
   rounds; graph/differential equations deserve more.
7. Only then tune search (`map_elites` is a wash in a paired test: 62 wins /
   69 losses on AlgoTune at 1.27x tokens).

Validate with an A/B on 8 AlgoTune tasks / 10 KernelBench problems, comparing
valid@1, valid@end, median, fast@1, tokens per valid task, debug rescue rate and
repeat-error rate. Accept only changes that reduce tokens per valid result
without lowering the median.

## 9. Handy commands

```bash
# what is running
squeue -u $USER -o "%.10i %.28j %.10T %.11M %.20S %R"

# per-run progress
for d in experiments/clean_*; do echo -n "$d: "; wc -l < "$d/results.csv" 2>/dev/null || true; done

# dependencies (run this first on a new machine)
python scripts/check_deps.py

# table
python benchmarking/aggregate_results.py --tag clean

# tests
python -m pytest tests -q        # 65 tests

# sanity: prompt configs render
python - <<'PY'
import json
from optimizer_agent.prompts import create_prompt, resolve_prompt_config
data = dict(task_description="d", reference="r", current_artifact="a",
            current_implementation="a", new_artifact="n", new_implementation="n",
            result="r", failure_stage="s", strategy="s", feedback="f",
            error="e", feedback_hint="h")
for path in ("benchmarking/agents/prompts/kernelbench.json",
             "benchmarking/agents/prompts/algotune.json"):
    cfg = json.load(open(path))
    for stage in ("improve", "debug"):
        sc = resolve_prompt_config(cfg, stage)
        for part in ("core", "strategy", "feedback"):
            create_prompt(part, dict(data), sc)
print("ok")
PY
```

## 10. Files to read, in order

1. `benchmarking/README.md` — layout and run matrix.
2. `benchmarking/COMPARISON.md` — settings, alignment, clean batch.
3. `benchmarking/ANALYSIS.md` — what went wrong, with numbers.
4. `benchmarking/IMPROVEMENT_PLAN.md` — the proposals and their evidence.
5. `benchmarking/RECURSIVE_ADVANCED_ANALYSIS.md` — per-failure-class deep dive.
6. `benchmarking/ROBUST_KERNELBENCH_FEEDBACK.md` — evaluator bug report (fixed
   upstream in `1139ebc`; kept for the contract).

## 11. Pushing and the dependency contract

What a new machine needs is exactly:

```text
optimizer-agent (feat/benchmarking)   our code
robust_kernelbench (>= 1139ebc)        imported evaluator
```

Everything else is upstream (AlgoTune tasks, OpenEvolve) or archival (the AKD
checkout). Before publishing, verify that rule still holds:

```bash
# no code may reference the AKD checkout
git grep -n "automatic_knowledge_discovery" -- '*.py' '*.sh'   # expect no hits

# only these external paths may be required
git grep -n "robust_kernelbench\|ALGOTUNE_ROOT\|OPENEVOLVE_ROOT" -- '*.py' '*.sh' | head
```

Push state (check before publishing; counts drift):

```bash
git -C <repo> log --oneline @{u}..HEAD     # commits not yet pushed
git -C <repo> status --porcelain           # must be empty
```

Branches that matter:

| Repo | Branch | Role |
|---|---|---|
| optimizer-agent | `feat/benchmarking` | required: agent + harness + docs |
| robust_kernelbench | `v2.2-datasets` | required: evaluator |
| optimizer-agent | `feat/analysis` | optional: dashboard |
| automatic_knowledge_discovery | `main` | **not required**; archival only |

`experiments/`, `logs/`, generated `tasks_*/` and `openevolve_output/` are
git-ignored and must not be pushed. No credentials live in the repo (`HF_TOKEN`
and API keys come from the environment).

## 12. Do not

- Do not commit `experiments/`, `logs/`, `tasks_*/`, `openevolve_output/` — they
  are git-ignored; results stay on disk only.
- Do not edit `robust_kernelbench`'s uncommitted files (not ours).
- Do not compare numbers across the alignment boundary (see §6).
- Do not increase `max_tokens` expecting fewer truncated outputs: only 1-2% of
  candidates hit the cap; the failures are format/contract issues.
