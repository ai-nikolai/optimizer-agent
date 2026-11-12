# Benchmarking

Two benchmark harnesses for optimizer-agent:

- `kernelbench/` — KernelBench / robust_kernelbench-style CUDA kernel optimization
- `algotune/` — AlgoTune numerical-program optimization

Both use the same `optimizer-agent` loop and evaluator contract. Each benchmark
owns its task-specific evaluator, prompt config and Slurm launcher.

## KernelBench

```bash
sbatch benchmarking/kernelbench/run_slurm.sh
```

See `kernelbench/README.md` for levels, rounds, sharding and robust_kernelbench
comparison details.

To replay the 100-problem `automatic_knowledge_discovery` `kernel_eng2`
experiments, use `benchmarking/kernelbench/akd/run_slurm.sh`; the dataset,
prompts, feedback config and legacy evaluator are copied there. See
`kernelbench/akd/README.md`.

## AlgoTune

Official sources:

- GitHub: https://github.com/oripress/AlgoTune
- HF dataset: https://huggingface.co/datasets/oripress/AlgoTune

The official project already publishes the full AlgoTune dataset and task
definitions, so we do **not** duplicate the 328 GB instance dataset. The local
`dataset/` is a lightweight task manifest used by optimizer-agent; it contains
task descriptions, reference source, and an unoptimized baseline `Solver`.

Prepare it from an official AlgoTune checkout:

```bash
conda activate env_kb2
python benchmarking/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune
```

Upload the lightweight manifest to your own HF account if desired:

```bash
python benchmarking/algotune/prepare_dataset.py \
  --algotune-root /hx2-weka/home/nr1713/AlgoTune \
  --push-to-hf your-name/AlgoTuneTasks
```

Run:

```bash
# smoke
LIMIT=1 ROUNDS=1 sbatch benchmarking/algotune/run_slurm.sh

# full manifest
sbatch benchmarking/algotune/run_slurm.sh
```

Environment for the jobs:
- `env_kb2` for optimizer-agent + AlgoTune evaluator
- `env_vllm30` only for the vLLM server
- `ALGOTUNE_ROOT` default: `/hx2-weka/home/nr1713/AlgoTune`
