# Benchmark comparison matrix

Fair comparison between optimizer-agent settings and OpenEvolve-only
competitors. All runs use the same local model
(`Qwen/Qwen3-Coder-30B-A3B-Instruct`) served by vLLM.

## AlgoTune

| Setting | Search | Rounds / iterations | Samples per round | Feedback / strategy | Notes |
|---|---|---|---|---|---|
| **ours: sequential** | `bestN_last` | 5 | 1 | `rules_llm`, generic, no strategy/debug | robust-kernelbench-style repair chain |
| **ours: advanced** | `map_elites`, dynamic, 3 cells | 5 | 1 | `rules_llm`, generic, strategy + debug | quality-diversity search |
| **OpenEvolve** | OpenEvolve MAP-Elites, 4 islands, population 1000 | 100 iterations | OpenEvolve native | OpenEvolve native prompts | OpenEvolve only, 5 correctness trials |

Task set for the first comparison: the 8 curated AlgoTune tasks in
`benchmarking/algotune/tasks.json`, with the same `n` per task. The full
154-task manifests are available; scale the same commands with
`MANIFEST=... DATASET_ROOT=...`.

Commands:

```bash
# ours: sequential
ROUNDS=5 sbatch benchmarking/algotune/run_slurm.sh

# ours: advanced
ROUNDS=5 SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 \
  USE_STRATEGY=1 USE_DEBUG=1 sbatch benchmarking/algotune/run_slurm.sh

# OpenEvolve
sbatch benchmarking/openevolve/algotune/run_slurm.sh
```

## KernelBench

| Setting | Search | Rounds / iterations | Samples per round | Feedback / strategy | Notes |
|---|---|---|---|---|---|
| **ours: sequential** | `bestN_last` | 5 | 1 | kernel `rules_llm`, debug | `benchmarking/kernelbench/akd` / `run_level.py` |
| **ours: advanced** | `map_elites`, dynamic, 3 cells | 5 | 1 | kernel `rules_llm`, strategy + debug | AKD newsettings profile |
| **OpenEvolve** | OpenEvolve MAP-Elites, 4 islands, population 1000 | 50–100 iterations | OpenEvolve native | OpenEvolve native prompts | OpenEvolve only, robust evaluator |

The KernelBench sequential/advanced runs are the AKD `kernel_eng2` jobs (100
KernelBench-derived problems). The OpenEvolve KernelBench baseline uses the same
KernelBench level-1 problem family with the robust evaluator. Start with a
subset and scale:

```bash
# smoke
PROBLEM_LIMIT=1 ITERATIONS=1 sbatch benchmarking/openevolve/kernelbench/run_slurm.sh

# baseline
PROBLEM_LIMIT=10 ITERATIONS=50 sbatch benchmarking/openevolve/kernelbench/run_slurm.sh
```

## Fairness notes

- Same model and vLLM serving settings across methods.
- Same task sizes / problem IDs where possible.
- Same hardened KernelBench evaluator policy (5 correctness trials, 3 timing
  trials, L2 flush, dtype-aware tolerances).
- OpenEvolve keeps its native search/prompt configuration; it is not reduced to
  a one-shot or repair-only loop.
- OpenEvolve AlgoTune uses its native 5-trial correctness check, which is
  stronger than the optimizer-agent 2-seed AlgoTune check. This favours
  OpenEvolve, not the other way around.
