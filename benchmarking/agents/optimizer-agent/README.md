# optimizer-agent

Runner scripts that drive `optimizer_agent` over either task family. Task
assets (dataset, evaluator) are imported from `../../tasks/`; prompts and
feedback configs live here because they shape the agent's search.

```
agents/optimizer-agent/
  algotune/
    run_tasks.py        per-task driver, writes results.csv
    run_slurm.sh        2-GPU launcher (vLLM + agent), curated 8-task set
    run_full_slurm.sh   sharded launcher, full 154-task set
    prompt_config.json  agent prompts (generic, non-kernel)
    feedback_config.json
  kernelbench/
    run_level.py        per-problem driver, KernelBench levels
    run_slurm.sh        2-GPU launcher
    prompt_config.json  kernel prompts
```

## AlgoTune

The driver reads a manifest and dataset from `benchmarking/tasks/algotune/` and
calls `tasks/algotune/evaluate.py` for scoring. Official task `n` values come
from the manifest.

```bash
# server already running on port 30000
python benchmarking/agents/optimizer-agent/algotune/run_tasks.py \
  --manifest benchmarking/tasks/algotune/dataset_full/manifest.json \
  --dataset-root benchmarking/tasks/algotune/dataset_full \
  --algotune-root "$ALGOTUNE_ROOT" \
  --rounds 5 \
  --model-url http://localhost:30000 \
  --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --output experiments/algotune_seq/results.csv
```

Slurm:

```bash
# curated 8-task set
sbatch benchmarking/agents/optimizer-agent/algotune/run_slurm.sh

# full 154-task set, sharded
NUM_SHARDS=3 sbatch --array=0-2 benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh

# advanced (quality-diversity) setting
ROUNDS=4 SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 \
  USE_STRATEGY=1 USE_DEBUG=1 \
  sbatch benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh
```

`--start` / `--limit` / `--skip-completed` make sharding and resume explicit;
tasks already recorded as `ok` in the given CSVs are skipped.

## KernelBench

`run_level.py` drives one KernelBench level, one problem at a time, and
delegates scoring to `tasks/kernelbench/evaluate.py`.

```bash
python benchmarking/agents/optimizer-agent/kernelbench/run_level.py \
  --level 1 --rounds 3 --limit 10 \
  --model-url http://localhost:30000 \
  --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --experiments-dir experiments/kb_level1
```

Slurm:

```bash
LIMIT=10 ROUNDS=3 sbatch benchmarking/agents/optimizer-agent/kernelbench/run_slurm.sh
```

For the recursive/repair baseline that replays the AKD `kernel_eng2` loop over
the 100-problem KernelBench set, use `../recursive/run_slurm.sh` instead.

## Requirements

- `env_kb2` for the agent, AlgoTune evaluator and KernelBench evaluation
- `env_vllm30` only for the vLLM server
- `ALGOTUNE_ROOT` pointing at a checkout of https://github.com/oripress/AlgoTune
- `ROBUST_KERNELBENCH_ROOT` for KernelBench evaluation (set by the launcher)
