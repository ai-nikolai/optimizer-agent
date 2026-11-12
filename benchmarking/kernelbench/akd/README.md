# AKD `kernel_eng2` replay

This directory contains a self-contained copy of the problem set and prompt
configuration used by the running experiments in
`automatic_knowledge_discovery/kernel_eng2`, so the same experiments can be run
from the `optimizer-agent` benchmarking branch.

## Contents

| File | Source in `kernel_eng2` | Purpose |
|---|---|---|
| `dataset/` | `dataset/` | 100 `P1..P100` problems, each with `<id>_baseline.py` and `<id>_reference.py` |
| `prompt_config.json` | `prompt_config_optimizer_agent.json` | AKD kernel improve/debug/strategy prompts |
| `feedback_config.json` | `feedback_config_optimizer_agent.json` | runtime-based metric and `rules_llm` feedback |
| `evaluate_legacy.py` | `evaluate.py` | the evaluator used by the in-flight experiments |
| `run_slurm.sh` | adapted from `run_optimizer_agent.sh` | 3-GPU launcher (TP=2 vLLM + evaluator) |

`../evaluate.py` is the hardened, `robust_kernelbench`-backed evaluator. Use it
for new experiments; use `evaluate_legacy.py` only when reproducing the
in-flight runs exactly.

Both evaluators honour `KERNEL_BUILD_ROOT` and place JIT artefacts under a
hashed subdirectory there, keeping the 1 TB home filesystem free.

## Reproduce the two in-flight experiments

Robust replication (`bestN_last`):

```bash
cd /hx2-weka/home/nr1713/optimizer-agent-benchmarking
EXPERIMENT_NAME=experiment_v5_0_robust_kb_optagent_replay \
  sbatch benchmarking/kernelbench/akd/run_slurm.sh
```

New settings (`map_elites`, dynamic, 3 cells, strategy):

```bash
cd /hx2-weka/home/nr1713/optimizer-agent-benchmarking
EXPERIMENT_NAME=experiment_v5_0_newsettings_optagent_replay \
SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 USE_STRATEGY=1 \
  sbatch benchmarking/kernelbench/akd/run_slurm.sh
```

Both launchers use the legacy evaluator by default. To use the hardened
evaluator instead:

```bash
EVALUATE=benchmarking/kernelbench/evaluate.py \
  sbatch benchmarking/kernelbench/akd/run_slurm.sh
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
  loop; this uses `../run_level.py` with `--dataset-dir`, which runs the same
  `Agent` per problem and writes the same per-run `summary.json`/`config.json`.
- The AKD launcher requested `gpu:h200:3,scratch:200g`; this launcher does the
  same, with `TMPDIR`, torch/triton/inductor caches and `KERNEL_BUILD_ROOT` on
  scratch.
- The benchmark launcher stores results under `experiments/<name>/` rather than
  `kernel_eng2/<experiment_name>/P<id>`; the agent settings and evaluator
  contract are unchanged.
