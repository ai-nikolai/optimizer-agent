# Benchmarking

> New machine? Read [`../AGENTS.md`](../AGENTS.md) first: it covers the
> two required checkouts (optimizer-agent + robust_kernelbench), the bootstrap,
> run commands and what is archival.

Two task families, three agents. Tasks own the dataset and the evaluator;
agents own the runner, prompts and Slurm launcher.

```text
benchmarking/
  tasks/
    algotune/      dataset (8-task + full 154), evaluator, dataset builders
    kernelbench/   dataset (P1..P100), hardened + legacy evaluators
  agents/
    recursive/          AKD kernel_eng2 recursive baseline (KernelBench)
    optimizer-agent/    optimizer-agent runners for AlgoTune and KernelBench
    openevolve/         OpenEvolve-only competitor for both task families
  aggregate_results.py  builds the comparison table
  COMPARISON.md         settings, token accounting and results
```

Run matrix:

| Task | recursive | optimizer-agent | openevolve |
|---|---|---|---|
| AlgoTune | — | `agents/optimizer-agent/algotune/run_slurm.sh` | `agents/openevolve/algotune/run_slurm.sh` |
| KernelBench | `agents/recursive/run_slurm.sh` | `agents/optimizer-agent/kernelbench/run_slurm.sh` | `agents/openevolve/kernelbench/run_slurm.sh` |

Every agent gets the task's evaluator from `tasks/`; prompts and feedback
configs stay with the agent because they are part of its search behaviour.

## KernelBench

```bash
sbatch benchmarking/agents/optimizer-agent/kernelbench/run_slurm.sh
```

See `tasks/kernelbench/README.md` for the dataset and evaluator, and
`agents/optimizer-agent/README.md` for levels, rounds and sharding.

The 100-problem repair-chain baseline (the historical AKD `kernel_eng2` policy)
is `benchmarking/agents/recursive/run_slurm.sh`. It needs only this repo plus
`robust_kernelbench`; prompts and feedback are shared in `agents/`. See
`agents/recursive/README.md`.

## AlgoTune

Official sources:

- GitHub: https://github.com/oripress/AlgoTune
- HF dataset: https://huggingface.co/datasets/oripress/AlgoTune

The official project already publishes the full AlgoTune dataset and task
definitions, so we do **not** duplicate the 328 GB instance dataset. The local
`dataset/` is a lightweight 8-task manifest used for quick runs; `dataset_full/`
is the 154-task manifest built from the official size parameters encoded in the
HF filenames.

Prepare the curated set:

```bash
conda activate env_kb2
python benchmarking/tasks/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune
```

Prepare the full 154-task set:

```bash
python benchmarking/tasks/algotune/prepare_full_manifest.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune
python benchmarking/tasks/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune \
  --manifest benchmarking/tasks/algotune/tasks_full.json \
  --output benchmarking/tasks/algotune/dataset_full
```

Upload the lightweight manifest to your own HF account if desired:

```bash
python benchmarking/tasks/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune \
  --manifest benchmarking/tasks/algotune/tasks_full.json \
  --output benchmarking/tasks/algotune/dataset_full \
  --push-to-hf your-name/AlgoTuneTasks
```

Run:

```bash
# smoke
LIMIT=1 ROUNDS=1 sbatch benchmarking/agents/optimizer-agent/algotune/run_slurm.sh

# curated 8-task manifest
sbatch benchmarking/agents/optimizer-agent/algotune/run_slurm.sh

# full 154-task manifest, 4 shards
sbatch benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh

# more shards
NUM_SHARDS=8 sbatch --array=0-7 benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh
```

The official repo also provides named task lists in `task_lists.json`
(`fast`, `medium`, `all`, `compute_heavy`, `graph_tasks`), but no top-K
importance ranking. See `tasks/algotune/README.md` for details.

Advanced AlgoTune setting (quality-diversity search):

```bash
ROUNDS=5 SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 \
  USE_STRATEGY=1 USE_DEBUG=1 sbatch benchmarking/agents/optimizer-agent/algotune/run_slurm.sh
```

OpenEvolve-only competitor:

```bash
sbatch benchmarking/agents/openevolve/algotune/run_slurm.sh
PROBLEM_LIMIT=10 ITERATIONS=50 sbatch benchmarking/agents/openevolve/kernelbench/run_slurm.sh
```

Environment for the jobs:
- `env_kb2` for optimizer-agent + AlgoTune evaluator
- `env_vllm30` only for the vLLM server
- `ALGOTUNE_ROOT` default: `/hx2-weka/home/nr1713/AlgoTune`
