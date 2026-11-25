# Recursive baseline (AKD `kernel_eng2`)

The recursive/repair baseline: every round rewrites the previous round's kernel
from its compile/run/correctness feedback, with optional strategy cells and
debug iterations. This is the `automatic_knowledge_discovery` `kernel_eng2`
loop, run from this repository with no dependency on that checkout.

## Contents

| File | Purpose |
|---|---|
| `run_slurm.sh` | 2-GPU launcher (TP=1 vLLM on GPU 0, agent + evaluator on GPU 1) |
| `prompt_config.json` | kernel improve/debug/strategy prompts |
| `feedback_config.json` | runtime-based metric and `rules_llm` feedback |

Task-side assets live in `../../tasks/kernelbench/`: the 100-problem dataset
(`dataset/`), the hardened `evaluate.py` and the legacy `evaluate_legacy.py`
used by the in-flight AKD runs.

The hardened evaluator is the default. Use the legacy one only when
reproducing the in-flight runs exactly.

Both evaluators honour `KERNEL_BUILD_ROOT` and place JIT artefacts under a
hashed subdirectory there, keeping the 1 TB home filesystem free.

## Reproduce the two in-flight experiments

Robust replication (`bestN_last`):

```bash
cd /hx2-weka/home/nr1713/optimizer-agent-benchmarking
EXPERIMENT_NAME=experiment_v5_0_robust_kb_optagent_replay \
  sbatch benchmarking/agents/recursive/run_slurm.sh
```

New settings (`map_elites`, dynamic, 3 cells, strategy):

```bash
cd /hx2-weka/home/nr1713/optimizer-agent-benchmarking
EXPERIMENT_NAME=experiment_v5_0_newsettings_optagent_replay \
SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 USE_STRATEGY=1 \
  sbatch benchmarking/agents/recursive/run_slurm.sh
```

The launcher uses the hardened evaluator by default. To reproduce the legacy
in-flight policy instead:

```bash
EVALUATE=benchmarking/tasks/kernelbench/evaluate_legacy.py \
  sbatch benchmarking/agents/recursive/run_slurm.sh
```

## Useful overrides

```bash
PROBLEM_START=1 PROBLEM_END=100
ROUNDS=5 NUM_SAMPLES=1 MAX_CANDIDATES=20
SELECTION=bestN_last            # or map_elites
MAP_ELITES_METHOD=cluster       # cluster | fixed | dynamic
NUM_MAP_ELITES=1
USE_STRATEGY=0 USE_DEBUG=1
FEEDBACK_STRATEGY=rules_llm FEEDBACK_PROFILE=kernel
RESUME=1 OVERWRITE=0
EVAL_TIMEOUT=1800
EXPERIMENT_DIR=experiments/my_run
```

## Differences from the original AKD launcher

- The AKD launcher invoked `optimizer_agent.cli` once per problem in a shell
  loop; this launcher does the same, writing per-run `summary.json`/`config.json`.
- Scratch (`TMPDIR`, torch/triton/inductor caches, `KERNEL_BUILD_ROOT`) is set
  from `--gres=scratch`, keeping the home filesystem free.
- The benchmark launcher stores results under `experiments/<name>/` rather than
  `kernel_eng2/<experiment_name>/P<id>`; the agent settings and evaluator
  contract are unchanged.
