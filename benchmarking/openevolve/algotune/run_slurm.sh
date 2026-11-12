#!/usr/bin/env bash
#SBATCH --job-name=openevolve_algotune
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_algotune_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_algotune_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1,scratch:100g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# OpenEvolve-only AlgoTune baseline.
#   GPU 0: vLLM server
#   CPU  : OpenEvolve + AlgoTune evaluator
#
# Uses OpenEvolve's native task files/configs (100 iterations, population 1000,
# 4 islands, 5 correctness trials) from benchmarking/openevolve/algotune/tasks.

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
OPENEVOLVE_ROOT="/hx2-weka/home/nr1713/openevolve"
ALGOTUNE_ROOT="${ALGOTUNE_ROOT:-/hx2-weka/home/nr1713/AlgoTune}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
SCRATCH="${TMPDIR:-/tmp}/openevolve_algotune_${JOB_ID}"
mkdir -p "${SCRATCH}/tmp" "${SCRATCH}/hf_home" "${SCRATCH}/hf_datasets" \
    "${SCRATCH}/numba" "${SCRATCH}/mpl" "${SCRATCH}/torch_extensions" \
    "${SCRATCH}/triton" "${SCRATCH}/torchinductor"
export TMPDIR="${SCRATCH}/tmp"
export HF_HOME="${SCRATCH}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets"
export NUMBA_CACHE_DIR="${SCRATCH}/numba"
export MPLCONFIGDIR="${SCRATCH}/mpl"
export TORCH_EXTENSIONS_DIR="${SCRATCH}/torch_extensions"
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/torchinductor"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-30000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
ITERATIONS="${ITERATIONS:-100}"
TASKS="${TASKS:-}"
PREPARE="${PREPARE:-1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/openevolve_algotune_${JOB_ID}}"

cd "${REPO}"
mkdir -p logs experiments

source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm30

export CUDA_VISIBLE_DEVICES=0
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export PYTHONUNBUFFERED=1

echo "=== starting vLLM (OpenEvolve AlgoTune) ${JOB_ID} at $(date +%T)"
setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --enforce-eager > "logs/vllm_openevolve_algotune_${JOB_ID}.log" 2>&1 &
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
        tail -40 "logs/vllm_openevolve_algotune_${JOB_ID}.log" >&2
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
export CUDA_VISIBLE_DEVICES=""
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ALGOTUNE_PATH="${ALGOTUNE_ROOT}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export PYTHONPATH="${REPO}:${OPENEVOLVE_ROOT}:${ALGOTUNE_ROOT}:${PYTHONPATH:-}"

EXTRA=()
if [ -n "${TASKS}" ]; then EXTRA+=(--tasks "${TASKS}"); fi

if [ "${PREPARE}" = "1" ]; then
    echo "=== preparing OpenEvolve AlgoTune tasks at $(date +%T)"
    PREP_ARGS=(--clean --model-url "http://localhost:${PORT}/v1" --model-name "${MODEL_NAME}" --max-tokens "${MAX_TOKENS}")
    if [ -n "${TASKS}" ]; then PREP_ARGS+=(--tasks "${TASKS}"); fi
    python benchmarking/openevolve/algotune/prepare_tasks.py "${PREP_ARGS[@]}"
fi

echo "=== OpenEvolve AlgoTune: iterations=${ITERATIONS} at $(date +%T)"
python benchmarking/openevolve/algotune/run_tasks.py \
    --tasks-dir benchmarking/openevolve/algotune/tasks \
    --iterations "${ITERATIONS}" \
    --openevolve-root "${OPENEVOLVE_ROOT}" \
    --algotune-root "${ALGOTUNE_ROOT}" \
    --output "${EXPERIMENT_DIR}/results.csv" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
RC=$?
echo "=== OpenEvolve AlgoTune rc=${RC} at $(date +%T)"

exit "${RC}"
