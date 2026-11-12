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

## Dataset

The official AlgoTune dataset already exists:
https://huggingface.co/datasets/oripress/AlgoTune

We therefore use the official task definitions and keep only a lightweight
manifest locally. `prepare_dataset.py` can rebuild it and optionally upload it to
your own Hugging Face account.

```bash
conda activate env_kb2
python benchmarking/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune

# optional upload
python benchmarking/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune \
  --push-to-hf your-name/AlgoTuneTasks
```

## Run

Smoke:

```bash
LIMIT=1 ROUNDS=1 sbatch benchmarking/algotune/run_slurm.sh
```

Full manifest:

```bash
sbatch benchmarking/algotune/run_slurm.sh
```

Useful overrides:

```bash
TASKS=affine_transform_2d,lu_factorization
ROUNDS=5
ALGOTUNE_TIMING_RUNS=3
ALGOTUNE_WARMUPS=1
ALGOTUNE_SOLVE_TIMEOUT=120
```

The selected tasks are CPU-heavy, so the evaluator runs on CPU while vLLM uses
the GPU.

## Requirements

`env_kb2` needs the AlgoTune dependencies for the tasks you select. The current
manifest uses numpy/scipy/networkx and `orjson`; install the official AlgoTune
requirements for the full 154-task suite.
