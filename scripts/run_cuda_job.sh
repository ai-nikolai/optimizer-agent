#!/usr/bin/env bash
#SBATCH --job-name=optagent_cuda
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-simple
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-simple/logs/optagent_cuda_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-simple/logs/optagent_cuda_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Mirror the recent robust_kernelbench one-GPU local pattern:
#   vLLM on the GPU (55% memory), then run the optimizer on the same GPU.
# The agent's evaluator compiles CUDA kernels, so CUDA_HOME/PATH are set for nvcc.

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-simple"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-30000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/opt_agent_cuda_${JOB_ID}}"

cd "${REPO}"
mkdir -p logs experiments

source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm30

export CUDA_VISIBLE_DEVICES=0
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "=== starting vLLM job ${JOB_ID} at $(date +%T)"
setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --enforce-eager > "logs/vllm_optagent_${JOB_ID}.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "=== stopping vLLM (pgid ${SERVER_PID}) at $(date +%T)"
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== waiting for vLLM at $(date +%T)"
for _ in $(seq 1 1800); do
    if curl -s -f "http://localhost:${PORT}/health" > /dev/null 2>&1 \
       || curl -s -f "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "=== vLLM ready at $(date +%T)"
        break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null || {
        echo "FATAL: vLLM died during startup" >&2
        tail -40 "logs/vllm_optagent_${JOB_ID}.log" >&2
        exit 1
    }
    sleep 2
done

if ! curl -s -f "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "FATAL: vLLM did not become ready" >&2
    exit 1
fi

# The evaluator runs in the same interpreter as the agent, so use the env that
# has torch + nvcc. textdistance is only needed by map_elites.
conda deactivate
conda activate env_kb2
export PYTHONPATH="${REPO}"
export CUDA_VISIBLE_DEVICES=0

echo "=== optimizer-agent (CUDA) at $(date +%T)"
python -m optimizer_agent.cli \
    --baseline examples/cuda_engineering/baseline.py \
    --reference examples/cuda_engineering/reference.py \
    --evaluate examples/cuda_engineering/evaluate.py \
    --prompt-config-file examples/cuda_engineering/prompt_config.json \
    --num-iterations 3 \
    --num-samples 2 \
    --predecessor-selection-method bestN_last \
    --use-strategy \
    --use-feedback \
    --use-debug \
    --num-debug-iterations 2 \
    --model-url "http://localhost:${PORT}" \
    --model-name "${MODEL_NAME}" \
    --model-params '{"temperature": 0.2, "max_tokens": 8192}' \
    --eval-timeout 900 \
    --experiments-dir "${EXPERIMENT_DIR}"
RC=$?
echo "=== optimizer-agent rc=${RC} at $(date +%T)"

exit "${RC}"
