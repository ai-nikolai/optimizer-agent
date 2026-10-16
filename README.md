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

```bash
# ornith-ai/Ornith-1.5-35B-A3B
# ornith-ai/Ornith-1.5-9B
CUDA_VISIBLE_DEVICES=0 vllm serve ornith-ai/Ornith-1.5-9B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 30000 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

## Other useful commands:
```bash
salloc --partition=interactive-gpu --gres=gpu:h200:1 --time=08:00:00 --ntasks=1 --cpus-per-task=16 --gpu-bind=closest
srun --pty --overlap --jobid=1048235 bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate
conda activate env_model2 #sglang
conda activate env_kb2 #kb
conda activate env_vllm30 #vllm (model)
```

 
```bash 
# to install on cluster.
export TMPDIR=/hx2-weka/home/nr1713/tmp
mkdir -p "$TMPDIR"
pip install --no-cache-dir vllm
```

## (C) Nikolai Rozanov - 2026 - Present