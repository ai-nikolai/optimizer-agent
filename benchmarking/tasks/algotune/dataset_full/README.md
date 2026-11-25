---
license: mit
pretty_name: AlgoTune Tasks (lightweight manifest)
tags:
  - algotune
  - numerical-optimization
  - code-generation
  - benchmark
---

# AlgoTune Tasks — lightweight manifest

A lightweight, self-contained version of the
[AlgoTune](https://github.com/oripress/AlgoTune) task set for use with
[optimizer-agent](https://github.com/ai-nikolai/optimizer-agent).

This dataset contains the **task definitions, descriptions, reference source,
and correct-but-unoptimized baseline solvers** for all 154 AlgoTune tasks. It
does **not** contain the official 328 GB pre-generated problem instances.

## Contents

```text
manifest.json                  # machine-readable task index
tasks.jsonl                    # one JSON record per task
task_lists.json                # official AlgoTune named task lists
tasks/<task>/
  description.txt              # problem statement
  task_source.py               # official reference implementation
  baseline.py                  # baseline Solver delegating to the reference
  reference.txt                # description + reference source for prompts
```

Each `manifest.json` record has:

```json
{
  "task": "affine_transform_2d",
  "category": "signal_processing",
  "n": 1123,
  "seeds": [1, 2],
  "baseline": "tasks/affine_transform_2d/baseline.py",
  "reference": "tasks/affine_transform_2d/reference.txt",
  "task_source": "tasks/affine_transform_2d/task_source.py"
}
```

The `n` values come from the official AlgoTune dataset filenames
(`<task>_T100ms_n<n>_size100_*.jsonl`), i.e. the official size for a 100 ms
reference runtime.

## Quick start

### 1. Download this dataset

```bash
huggingface-cli download ai-nikolai/AlgoTuneTasks \
  --repo-type dataset \
  --local-dir AlgoTuneTasks
```

Or from Python:

```python
from huggingface_hub import snapshot_download

root = snapshot_download(
    repo_id="ai-nikolai/AlgoTuneTasks",
    repo_type="dataset",
)
```

This is a file-based dataset, so use `snapshot_download` / `huggingface-cli`
rather than `datasets.load_dataset`.

### 2. Clone the official AlgoTune repository

The baseline solvers delegate to the official task implementations, so you need
the upstream repository:

```bash
git clone https://github.com/oripress/AlgoTune.git
export ALGOTUNE_ROOT="$PWD/AlgoTune"
```

### 3. Clone optimizer-agent

```bash
git clone https://github.com/ai-nikolai/optimizer-agent.git
cd optimizer-agent
git checkout feat/benchmarking
```

Install the agent and AlgoTune dependencies in your environment
(`env_kb2` in our setup):

```bash
pip install -e .
pip install orjson
```

### 4. Run the benchmark

Start an OpenAI-compatible model server, then run:

```bash
python benchmarking/agents/optimizer-agent/algotune/run_tasks.py \
  --manifest ../AlgoTuneTasks/manifest.json \
  --dataset-root ../AlgoTuneTasks \
  --algotune-root "$ALGOTUNE_ROOT" \
  --model-url http://localhost:30000 \
  --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --rounds 5 \
  --experiments-dir experiments/algotune
```

The evaluator checks `task.is_solution`, times the official reference and the
candidate solver, and reports:

```text
speedup = runtime_original / runtime_custom
```

### Sequential and advanced settings

```bash
# sequential repair chain
--selection bestN_last --samples 1

# quality-diversity search
--selection map_elites --map-elites-method dynamic --num-map-elites 3 \
  --use-strategy --use-debug
```

Official named subsets can be selected with `--task-list`:

```bash
--task-list fast
--task-list medium
--task-list all
--task-list compute_heavy
--task-list graph_tasks
```

### Slurm

```bash
# curated 8-task set
sbatch benchmarking/agents/optimizer-agent/algotune/run_slurm.sh

# full 154-task set, sharded
NUM_SHARDS=8 sbatch --array=0-7 benchmarking/agents/optimizer-agent/algotune/run_full_slurm.sh
```

## Official full dataset

If you need the official pre-generated problem instances and array data, use:

```text
https://huggingface.co/datasets/oripress/AlgoTune
```

That dataset is about 328 GB. This repository intentionally contains only the
lightweight task definitions and baselines.

## Related repositories

- [optimizer-agent](https://github.com/ai-nikolai/optimizer-agent) — the agent
  and benchmark harnesses used here.
- [automatic_knowledge_discovery](https://github.com/ai-nikolai/automatic_knowledge_discovery)
  — the AKD kernel-engineering experiments.
- [robust_kernelbench](https://github.com/ai-nikolai/robust_kernelbench) — the
  robust KernelBench evaluation and competitor baseline.

## Citation

If you use the underlying AlgoTune tasks, please cite:

```bibtex
@article{press2025algotune,
  title={AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs?},
  author={Press, Ori and Amos, Brandon and Zhao, Haoyu and Wu, Yikai and Ainsworth, Samuel K. and Krupke, Dominik and Kidger, Patrick and Sajed, Touqir and Stellato, Bartolomeo and Park, Jisun and Bosch, Nathanael and Meril, Eli and Steppi, Albert and Zharmagambetov, Arman and Zhang, Fangzhao and Perez-Pineiro, David and Mercurio, Alberto and Zhan, Ni and Abramovich, Talor and Lieret, Kilian and Zhang, Hanlin and Huang, Shirley and Bethge, Matthias and Press, Ofir},
  journal={arXiv preprint arXiv:2507.15887},
  year={2025},
  doi={10.48550/arXiv.2507.15887},
  url={https://arxiv.org/abs/2507.15887}
}
```

## Attribution and license

The task definitions and reference implementations are derived from the official
[AlgoTune](https://github.com/oripress/AlgoTune) repository, which is released
under the MIT license. The manifest packaging and baseline wrappers are released
under the MIT license as well.
