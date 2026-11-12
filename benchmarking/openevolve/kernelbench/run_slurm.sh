#!/usr/bin/env bash
#SBATCH --job-name=openevolve_kb
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_kb_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_kb_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2,scratch:200g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# OpenEvolve-only KernelBench baseline.
#   GPU 0: vLLM server
#   GPU 1: OpenEvolve + robust_kernelbench evaluator
#
# Defaults to 10 problems, 50 iterations each. Override with:
#   PROBLEM_START=0 PROBLEM_LIMIT=100 ITERATIONS=100 sbatch ...
#
# PREPARE=1 regenerates the task dirs for the selected problem range.

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
OPENEVOLVE_ROOT="/hx2-weka/home/nr1713/openevolve"
ROBUST_KB="/hx2-weka/home/nr1713/robust_kernelbench"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
SCRATCH="${TMPDIR:-/tmp}/openevolve_kb_${JOB_ID}"
mkdir -p "${SCRATCH}/tmp" "${SCRATCH}/torch_extensions" "${SCRATCH}/triton" \
    "${SCRATCH}/torchinductor" "${SCRATCH}/xdg" "${SCRATCH}/hf_datasets" \
    "${SCRATCH}/kb_builds"
export TMPDIR="${SCRATCH}/tmp"
export TORCH_EXTENSIONS_DIR="${SCRATCH}/torch_extensions"
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/torchinductor"
export XDG_CACHE_HOME="${SCRATCH}/xdg"
export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets"
export KB_BUILD_ROOT="${SCRATCH}/kb_builds"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-30000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
LEVEL="${LEVEL:-1}"
PROBLEM_START="${PROBLEM_START:-0}"
PROBLEM_LIMIT="${PROBLEM_LIMIT:-10}"
ITERATIONS="${ITERATIONS:-50}"
PREPARE="${PREPARE:-1}"
PROBLEM_IDS="${PROBLEM_IDS:-}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/openevolve_kernelbench_${JOB_ID}}"

cd "${REPO}"
mkdir -p logs experiments

source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm30

export CUDA_VISIBLE_DEVICES=0
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export PYTHONUNBUFFERED=1

echo "=== starting vLLM (OpenEvolve KernelBench) ${JOB_ID} at $(date +%T)"
setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --enforce-eager > "logs/vllm_openevolve_kb_${JOB_ID}.log" 2>&1 &
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
        tail -40 "logs/vllm_openevolve_kb_${JOB_ID}.log" >&2
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
export CUDA_VISIBLE_DEVICES=1
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export ROBUST_KERNELBENCH_ROOT="${ROBUST_KB}"
export PYTHONPATH="${REPO}:${OPENEVOLVE_ROOT}:${ROBUST_KB}:${PYTHONPATH:-}"

if [ "${PREPARE}" = "1" ]; then
    echo "=== preparing OpenEvolve KernelBench tasks at $(date +%T)"
    PREP_ARGS=(--level "${LEVEL}" --start "${PROBLEM_START}" --limit "${PROBLEM_LIMIT}" --clean)
    if [ -n "${PROBLEM_IDS}" ]; then
        PREP_ARGS+=(--problem-ids "${PROBLEM_IDS}")
    fi
    python benchmarking/openevolve/kernelbench/prepare_tasks.py \
        "${PREP_ARGS[@]}" \
        --model-url "http://localhost:${PORT}/v1" \
        --model-name "${MODEL_NAME}" \
        --max-tokens "${MAX_TOKENS}"
fi

EXTRA=()
if [ -n "${PROBLEM_IDS}" ]; then EXTRA+=(--problem-ids "${PROBLEM_IDS}"); fi

echo "=== OpenEvolve KernelBench level ${LEVEL}, iterations=${ITERATIONS} at $(date +%T)"
python benchmarking/openevolve/kernelbench/run_tasks.py \
    --tasks-dir "benchmarking/openevolve/kernelbench/tasks/level_${LEVEL}" \
    --iterations "${ITERATIONS}" \
    --openevolve-root "${OPENEVOLVE_ROOT}" \
    --robust-kb-root "${ROBUST_KB}" \
    --output "${EXPERIMENT_DIR}/results.csv" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
RC=$?
echo "=== OpenEvolve KernelBench rc=${RC} at $(date +%T)"

exit "${RC}"
