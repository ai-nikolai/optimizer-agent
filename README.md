# Optimizer Agent

A library to auto optimize and discover things with an LLM agent.

## Getting started:


1. Installation
```bash
python3 -m venv env_optagent
source env_optagent/bin/activate

pip3 install -e .
```

2. Running example:
```bash
# ./examples/test_example/run.sh
./examples/cuda_engineering/run.sh

# Run the model first
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
--model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
--tensor-parallel-size 1 \
--host 0.0.0.0 \
--port 30000 \
--enable-memory-saver
```

## Other useful commands:
```bash
salloc --partition=interactive-gpu --gres=gpu:h200:2 --time=02:00:00 --ntasks=1
srun --pty --overlap --jobid=120608 bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_model2 #sglang
conda activate env_kb2 #kb
```

## (C) Nikolai Rozanov - 2026 - Present