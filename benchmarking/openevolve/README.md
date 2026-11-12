# OpenEvolve baselines

OpenEvolve-only competitors for the optimizer-agent benchmarks. These use
OpenEvolve's own prompts, MAP-Elites search and evaluators; no optimizer-agent
components are involved.

## AlgoTune

The task files are adapted from OpenEvolve's official AlgoTune examples. The
native strong configuration is preserved:

- 100 iterations
- population 1000
- 4 islands
- 5 correctness trials, 3 timing runs, 1 warmup
- `diff_based_evolution: true`
- `num_top_programs: 3`, `num_diverse_programs: 2`
- official task size `n` from the optimizer-agent curated manifest

Prepare:

```bash
conda activate env_kb2
python benchmarking/openevolve/algotune/prepare_tasks.py
```

Run locally against an already-running vLLM server:

```bash
ALGOTUNE_PATH=/hx2-weka/home/nr1713/AlgoTune \
OPENAI_API_KEY=dummy \
PYTHONPATH=/hx2-weka/home/nr1713/openevolve:/hx2-weka/home/nr1713/AlgoTune:$PWD \
python benchmarking/openevolve/algotune/run_tasks.py \
  --iterations 100 \
  --output experiments/openevolve_algotune/results.csv
```

Slurm (starts its own vLLM):

```bash
sbatch benchmarking/openevolve/algotune/run_slurm.sh
TASKS=affine_transform_2d,lu_factorization sbatch benchmarking/openevolve/algotune/run_slurm.sh
ITERATIONS=30 TASKS=affine_transform_2d sbatch benchmarking/openevolve/algotune/run_slurm.sh
```

## KernelBench

The OpenEvolve KernelBench adapter builds one task per KernelBench problem:

- `reference.py` — original KernelBench problem
- `initial_program.py` — reference code plus a correct `ModelNew` wrapper
- `evaluator.py` — robust_kernelbench `eval_kernel_against_ref`
- `config.yaml` — native OpenEvolve search config + local vLLM

The evaluator uses 5 correctness trials, 3 performance trials and L2 flush, the
same hardened policy as the optimizer-agent KernelBench evaluator.

Prepare:

```bash
conda activate env_kb2
python benchmarking/openevolve/kernelbench/prepare_tasks.py --level 1 --limit 10
```

Run locally:

```bash
ROBUST_KERNELBENCH_ROOT=/hx2-weka/home/nr1713/robust_kernelbench \
OPENAI_API_KEY=dummy \
PYTHONPATH=/hx2-weka/home/nr1713/openevolve:/hx2-weka/home/nr1713/robust_kernelbench:$PWD \
CUDA_VISIBLE_DEVICES=1 \
python benchmarking/openevolve/kernelbench/run_tasks.py \
  --tasks-dir benchmarking/openevolve/kernelbench/tasks/level_1 \
  --iterations 50 \
  --output experiments/openevolve_kernelbench/results.csv
```

Slurm (starts its own vLLM):

```bash
# smoke: one problem, one iteration
PROBLEM_LIMIT=1 ITERATIONS=1 sbatch benchmarking/openevolve/kernelbench/run_slurm.sh

# initial baseline: 10 problems, 50 iterations
sbatch benchmarking/openevolve/kernelbench/run_slurm.sh

# full level: 100 problems, 50 iterations
PROBLEM_LIMIT=100 ITERATIONS=50 sbatch benchmarking/openevolve/kernelbench/run_slurm.sh
```

`PREPARE=1` (default) regenerates the selected task range. `PREPARE=0` reuses
existing task directories.

## Storage

Generated task directories and OpenEvolve outputs are gitignored. Slurm jobs put
`TMPDIR`, torch/triton/inductor caches, HF caches and KernelBench build dirs on
requested scratch. Only results and logs stay in home.
