# AlgoTune benchmark

A task-manifest benchmark around the official
[AlgoTune](https://github.com/oripress/AlgoTune) tasks.

Each task provides:

- `description.txt` — problem statement
- `task_source.py` — official reference implementation
- `baseline.py` — correct but unoptimized `Solver` that delegates to the official task
- `reference.txt` — description + reference source for the prompt

The evaluator generates problems with the official task, checks candidate output
with `task.is_solution`, then times reference vs candidate solves and reports
`speedup = runtime_original / runtime_custom`.

## Task sets

| File | Tasks | Purpose |
|---|---|---|
| `tasks.json` | 8 curated tasks | Quick benchmark / smoke runs |
| `tasks_full.json` | all 154 official tasks | Full AlgoTune benchmark |
| `task_lists.json` | official named lists | `fast`, `medium`, `all`, `compute_heavy`, `graph_tasks` |

There is **no official “top-K by importance” ranking**. The closest official
subsets are the named lists in `AlgoTuner/task_lists.py`:

```text
fast          : svm, max_common_subgraph, cyclic_independent_set
medium        : + nmf, communicability, eigenvalues_real
all           : + count_riemann_zeta_zeros, ode_stiff_robertson
compute_heavy : count_riemann_zeta_zeros, ode_stiff_robertson, nmf, eigenvalues_real
graph_tasks   : max_common_subgraph, cyclic_independent_set, communicability
```

`tasks.json` is our own curated 8-task set and is **not** the official `all`
list.

Run one of the official lists directly:

```bash
TASK_LIST=fast sbatch benchmarking/algotune/run_slurm.sh
TASK_LIST=medium LIMIT=6 sbatch benchmarking/algotune/run_slurm.sh
```

## Dataset

The official AlgoTune dataset already exists:
https://huggingface.co/datasets/oripress/AlgoTune

We use the official task definitions and keep only lightweight manifests
locally. The official HF filenames encode the size parameter for each task:

```text
data/<task>/<task>_T100ms_n<n>_size100_train.jsonl
```

`prepare_full_manifest.py` reads that metadata (not the 328 GB of arrays) and
builds `tasks_full.json`:

```bash
conda activate env_kb2
python benchmarking/algotune/prepare_full_manifest.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune
```

Build the local task manifest/dataset:

```bash
# Full 154-task set
python benchmarking/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune \
  --manifest benchmarking/algotune/tasks_full.json \
  --output benchmarking/algotune/dataset_full

# Curated 8-task set (default)
python benchmarking/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune
```

`dataset_full/` is only about 5 MB; it contains descriptions, reference source
and baseline solvers, not the official instance arrays.

Optional upload of the lightweight manifest to your own Hugging Face account:

```bash
python benchmarking/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune \
  --manifest benchmarking/algotune/tasks_full.json \
  --output benchmarking/algotune/dataset_full \
  --push-to-hf your-name/AlgoTuneTasks
```

## Readiness check

```bash
# Fast: imports every task and constructs it
python benchmarking/algotune/check_full_run.py

# Smoke: generate/solve/validate a subset at the official n
python benchmarking/algotune/check_full_run.py \
  --smoke --tasks affine_transform_2d,svm,tsp
```

The current full manifest passes the quick check for all 154 tasks.

## Run

Smoke:

```bash
LIMIT=1 ROUNDS=1 sbatch benchmarking/algotune/run_slurm.sh
```

Curated 8-task manifest:

```bash
sbatch benchmarking/algotune/run_slurm.sh
```

Full 154-task manifest, 4 shards:

```bash
sbatch benchmarking/algotune/run_full_slurm.sh
```

Override the shard count:

```bash
NUM_SHARDS=8 sbatch --array=0-7 benchmarking/algotune/run_full_slurm.sh
```

Each shard starts its own vLLM server on one GPU, runs a contiguous slice of
the manifest on CPU, and writes to `experiments/algotune_full_<job>_shard<N>/`.
The default 4-shard split is roughly 39 tasks per shard.

Useful overrides:

```bash
TASKS=affine_transform_2d,lu_factorization
TASK_LIST=fast                 # official named list
START=0 LIMIT=10
ROUNDS=5
MANIFEST=benchmarking/algotune/dataset_full/manifest.json
DATASET_ROOT=benchmarking/algotune/dataset_full
ALGOTUNE_TIMING_RUNS=3
ALGOTUNE_WARMUPS=1
ALGOTUNE_SOLVE_TIMEOUT=120
```

## Requirements

The full suite needs more than the original 8-task set. `env_kb2` currently has
the missing packages installed:

```text
cvxpy, ortools, scikit-learn, python-sat, hdbscan, pot, faiss-cpu
```

All 154 tasks import successfully. `orjson` is also required by the official
AlgoTune task base.

## Storage

All launchers request scratch and put heavy/temporary data there:

```text
TMPDIR, torch/triton/inductor caches, HF caches, numba cache, matplotlib cache
```

The full manifest itself is small, so the 1 TB home quota is only used for task
metadata, experiment results and logs. The official 328 GB instance dataset is
never downloaded.
