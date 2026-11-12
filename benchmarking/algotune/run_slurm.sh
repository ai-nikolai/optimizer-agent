#!/usr/bin/env bash
#SBATCH --job-name=optagent_algotune
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_algotune_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_algotune_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1,scratch:100g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# AlgoTune benchmark for optimizer-agent.
#   GPU 0: vLLM server
#   CPU: agent + AlgoTune evaluator (selected tasks are CPU-based)
#
# Smoke test: LIMIT=1 ROUNDS=1 sbatch benchmarking/algotune/run_slurm.sh
# Full manifest: sbatch benchmarking/algotune/run_slurm.sh

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
ALGOTUNE_ROOT="${ALGOTUNE_ROOT:-/hx2-weka/home/nr1713/AlgoTune}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    JOB_ID="${JOB_ID}_a${SLURM_ARRAY_TASK_ID}"
fi
SCRATCH="${TMPDIR:-/tmp}/optagent_algotune_${JOB_ID}"
mkdir -p "${SCRATCH}/tmp" "${SCRATCH}/torch_extensions" "${SCRATCH}/triton" "${SCRATCH}/torchinductor" "${SCRATCH}/xdg" "${SCRATCH}/hf_datasets" "${SCRATCH}/hf_home" "${SCRATCH}/numba" "${SCRATCH}/mpl"
export TMPDIR="${SCRATCH}/tmp"
export TORCH_EXTENSIONS_DIR="${SCRATCH}/torch_extensions"
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/torchinductor"
export XDG_CACHE_HOME="${SCRATCH}/xdg"
export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets"
export HF_HOME="${SCRATCH}/hf_home"
export NUMBA_CACHE_DIR="${SCRATCH}/numba"
export MPLCONFIGDIR="${SCRATCH}/mpl"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-30000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
ROUNDS="${ROUNDS:-5}"
SAMPLES="${SAMPLES:-1}"
SELECTION="${SELECTION:-bestN_last}"
MAP_ELITES_METHOD="${MAP_ELITES_METHOD:-cluster}"
NUM_MAP_ELITES="${NUM_MAP_ELITES:-1}"
USE_STRATEGY="${USE_STRATEGY:-0}"
USE_DEBUG="${USE_DEBUG:-0}"
DEBUG_ITERATIONS="${DEBUG_ITERATIONS:-1}"
DIFF_BASED="${DIFF_BASED:-0}"
DIFF_MODE="${DIFF_MODE:-unified-diff}"
START="${START:-0}"
LIMIT="${LIMIT:-}"
TASKS="${TASKS:-}"
TASK_LIST="${TASK_LIST:-}"
MANIFEST="${MANIFEST:-${REPO}/benchmarking/algotune/dataset/manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-${REPO}/benchmarking/algotune/dataset}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/algotune_v1_${JOB_ID}}"

cd "${REPO}"
mkdir -p logs experiments

source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm30

export CUDA_VISIBLE_DEVICES=0
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export PYTHONUNBUFFERED=1

echo "=== starting vLLM job ${JOB_ID} at $(date +%T)"
setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --enforce-eager > "logs/vllm_algotune_${JOB_ID}.log" 2>&1 &
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
        tail -40 "logs/vllm_algotune_${JOB_ID}.log" >&2
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
export PYTHONPATH="${REPO}:${ALGOTUNE_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ALGOTUNE_ROOT
export ALGOTUNE_TIMING_RUNS="${ALGOTUNE_TIMING_RUNS:-3}"
export ALGOTUNE_WARMUPS="${ALGOTUNE_WARMUPS:-1}"
export ALGOTUNE_SOLVE_TIMEOUT="${ALGOTUNE_SOLVE_TIMEOUT:-120}"

EXTRA=()
if [ -n "${LIMIT}" ]; then EXTRA+=(--limit "${LIMIT}"); fi
if [ -n "${TASKS}" ]; then EXTRA+=(--tasks "${TASKS}"); fi
if [ -n "${TASK_LIST}" ]; then EXTRA+=(--task-list "${TASK_LIST}"); fi
EXTRA+=(--start "${START}" --samples "${SAMPLES}")
EXTRA+=(--selection "${SELECTION}" --map-elites-method "${MAP_ELITES_METHOD}")
EXTRA+=(--num-map-elites "${NUM_MAP_ELITES}")
if [ "${USE_STRATEGY}" = "1" ]; then EXTRA+=(--use-strategy); fi
if [ "${USE_DEBUG}" = "1" ]; then EXTRA+=(--use-debug --num-debug-iterations "${DEBUG_ITERATIONS}"); fi
if [ "${DIFF_BASED}" = "1" ]; then EXTRA+=(--diff-based --diff-mode "${DIFF_MODE}"); fi

echo "=== optimizer-agent AlgoTune at $(date +%T)"
python benchmarking/algotune/run_tasks.py \
    --manifest "${MANIFEST}" \
    --dataset-root "${DATASET_ROOT}" \
    --rounds "${ROUNDS}" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    --algotune-root "${ALGOTUNE_ROOT}" \
    --model-url "http://localhost:${PORT}" \
    --model-name "${MODEL_NAME}" \
    --model-params "{\"temperature\": 0.2, \"max_tokens\": ${MAX_TOKENS}}" \
    --experiments-dir "${EXPERIMENT_DIR}"
RC=$?
echo "=== optimizer-agent AlgoTune rc=${RC} at $(date +%T)"

exit "${RC}"
