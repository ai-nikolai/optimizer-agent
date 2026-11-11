#!/usr/bin/env bash
#SBATCH --job-name=optagent_kb
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-simple
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-simple/logs/optagent_kb_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-simple/logs/optagent_kb_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# KernelBench / robust_kernelbench-style job for optimizer-agent.
#   GPU 0: vLLM server
#   GPU 1: run_level.py driver + evaluator subprocess
#
# Defaults to a 5-problem smoke run. Use LIMIT="" for the whole level:
#   LIMIT="" sbatch scripts/run_kernelbench_job.sh
#
# Environment overrides: MODEL_NAME, LEVEL, ROUNDS, LIMIT, SHARD_INDEX,
# NUM_SHARDS, GPU_MEM_UTIL, MAX_MODEL_LEN, MAX_TOKENS, KB_*_TRIALS.

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-simple"
ROBUST_KB="/hx2-weka/home/nr1713/robust_kernelbench"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-30000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
LEVEL="${LEVEL:-1}"
ROUNDS="${ROUNDS:-3}"
LIMIT="${LIMIT-5}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/opt_agent_kb_level${LEVEL}_${JOB_ID}}"

cd "${REPO}"
mkdir -p logs experiments

source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm30

export CUDA_VISIBLE_DEVICES=0
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export PYTHONUNBUFFERED=1

echo "=== starting vLLM for KernelBench job ${JOB_ID} at $(date +%T)"
setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --enforce-eager > "logs/vllm_optagent_kb_${JOB_ID}.log" 2>&1 &
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
        tail -40 "logs/vllm_optagent_kb_${JOB_ID}.log" >&2
        exit 1
    }
    sleep 2
done

if ! curl -s -f "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "FATAL: vLLM did not become ready" >&2
    exit 1
fi

conda deactivate
conda activate env_kb2
export PYTHONPATH="${REPO}"
export ROBUST_KERNELBENCH_ROOT="${ROBUST_KB}"
export CUDA_VISIBLE_DEVICES=1
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export KB_NUM_CORRECT_TRIALS="${KB_NUM_CORRECT_TRIALS:-5}"
export KB_NUM_PERF_TRIALS="${KB_NUM_PERF_TRIALS:-3}"
export KB_FLUSH_L2="${KB_FLUSH_L2:-1}"

EXTRA=()
if [ -n "${LIMIT}" ]; then
    EXTRA+=(--limit "${LIMIT}")
fi

echo "=== optimizer-agent KernelBench level ${LEVEL}, rounds ${ROUNDS}, limit '${LIMIT}' at $(date +%T)"
python examples/kernelbench/run_level.py \
    --level "${LEVEL}" \
    --rounds "${ROUNDS}" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    --shard-index "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}" \
    --model-url "http://localhost:${PORT}" \
    --model-name "${MODEL_NAME}" \
    --model-params "{\"temperature\": 0.2, \"max_tokens\": ${MAX_TOKENS}}" \
    --prompt-config examples/kernelbench/prompt_config.json \
    --feedback-config examples/feedback/kernel_rules_llm.json \
    --evaluate examples/kernelbench/evaluate.py \
    --experiments-dir "${EXPERIMENT_DIR}"
RC=$?
echo "=== optimizer-agent KernelBench rc=${RC} at $(date +%T)"

exit "${RC}"
