#!/usr/bin/env bash
#SBATCH --job-name=optagent_akd
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_akd_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_akd_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:3,scratch:200g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G

# Replay of the kernel_eng2 / automatic_knowledge_discovery experiment inside
# the optimizer-agent benchmarking branch.
#
#   GPU 0,1: vLLM (TP=2)
#   GPU 2  : optimizer-agent + CUDA evaluator
#
# The default evaluator is the legacy in-flight one (evaluate_legacy.py) so the
# replay matches the running AKD experiments.  For a hardened run, use:
#   EVALUATE=benchmarking/kernelbench/evaluate.py \
#     sbatch benchmarking/kernelbench/akd/run_slurm.sh
#
# Examples:
#   # robust replication (bestN_last)
#   sbatch benchmarking/kernelbench/akd/run_slurm.sh
#
#   # new settings (map_elites/dynamic/3 + strategy)
#   SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 USE_STRATEGY=1 \
#     sbatch benchmarking/kernelbench/akd/run_slurm.sh

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
SCRATCH="${TMPDIR:-/tmp}/optagent_akd_${JOB_ID}"
mkdir -p "${SCRATCH}/tmp" "${SCRATCH}/torch_extensions" "${SCRATCH}/triton" \
    "${SCRATCH}/torchinductor" "${SCRATCH}/xdg" "${SCRATCH}/hf_datasets" \
    "${SCRATCH}/kernel_builds"
export TMPDIR="${SCRATCH}/tmp"
export TORCH_EXTENSIONS_DIR="${SCRATCH}/torch_extensions"
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/torchinductor"
export XDG_CACHE_HOME="${SCRATCH}/xdg"
export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets"
export KERNEL_BUILD_ROOT="${SCRATCH}/kernel_builds"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-30000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

VERSION="${VERSION:-v5_0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-experiment_${VERSION}_akd_replay_optagent}"
PROBLEM_START="${PROBLEM_START:-1}"
PROBLEM_END="${PROBLEM_END:-100}"
ROUNDS="${ROUNDS:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"
SELECTION="${SELECTION:-bestN_last}"
MAP_ELITES_METHOD="${MAP_ELITES_METHOD:-cluster}"
NUM_MAP_ELITES="${NUM_MAP_ELITES:-1}"
USE_STRATEGY="${USE_STRATEGY:-0}"
USE_DEBUG="${USE_DEBUG:-1}"
DEBUG_ITERATIONS="${DEBUG_ITERATIONS:-1}"
FEEDBACK_STRATEGY="${FEEDBACK_STRATEGY:-rules_llm}"
FEEDBACK_PROFILE="${FEEDBACK_PROFILE:-kernel}"
DIFF_BASED="${DIFF_BASED:-0}"
DIFF_MODE="${DIFF_MODE:-unified-diff}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"

DATASET_DIR="${DATASET_DIR:-${REPO}/benchmarking/kernelbench/akd/dataset}"
PROMPT_CONFIG="${PROMPT_CONFIG:-${REPO}/benchmarking/kernelbench/akd/prompt_config.json}"
FEEDBACK_CONFIG="${FEEDBACK_CONFIG:-${REPO}/benchmarking/kernelbench/akd/feedback_config.json}"
EVALUATE="${EVALUATE:-${REPO}/benchmarking/kernelbench/akd/evaluate_legacy.py}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO}/experiments/${EXPERIMENT_NAME}}"

cd "${REPO}"
mkdir -p logs experiments "${EXPERIMENT_DIR}"

source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm30

export CUDA_VISIBLE_DEVICES=0,1
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export PYTHONUNBUFFERED=1

echo "=== starting vLLM job ${JOB_ID} at $(date +%T)"
setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --enforce-eager > "logs/vllm_optagent_akd_${JOB_ID}.log" 2>&1 &
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
        tail -40 "logs/vllm_optagent_akd_${JOB_ID}.log" >&2
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
export CUDA_VISIBLE_DEVICES=2
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EXTRA=()
if [ "${RESUME}" != "1" ]; then EXTRA+=(--no-resume); fi
if [ "${OVERWRITE}" = "1" ]; then EXTRA+=(--overwrite); fi
if [ "${USE_STRATEGY}" = "1" ]; then EXTRA+=(--use-strategy); fi
if [ "${USE_DEBUG}" = "1" ]; then EXTRA+=(--use-debug --num-debug-iterations "${DEBUG_ITERATIONS}"); fi
if [ "${DIFF_BASED}" = "1" ]; then EXTRA+=(--diff-based --diff-mode "${DIFF_MODE}"); fi

echo "=== AKD replay: ${EXPERIMENT_NAME} problems ${PROBLEM_START}..${PROBLEM_END} at $(date +%T)"
python benchmarking/kernelbench/run_level.py \
    --dataset-dir "${DATASET_DIR}" \
    --problem-start "${PROBLEM_START}" \
    --problem-end "${PROBLEM_END}" \
    --rounds "${ROUNDS}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-candidates "${MAX_CANDIDATES}" \
    --predecessor-selection-method "${SELECTION}" \
    --map-elites-method "${MAP_ELITES_METHOD}" \
    --num-map-elites "${NUM_MAP_ELITES}" \
    --feedback-strategy "${FEEDBACK_STRATEGY}" \
    --feedback-profile "${FEEDBACK_PROFILE}" \
    --eval-timeout "${EVAL_TIMEOUT}" \
    --model-url "http://localhost:${PORT}" \
    --model-name "${MODEL_NAME}" \
    --model-params "{\"temperature\": 0.2, \"max_tokens\": ${MAX_TOKENS}}" \
    --prompt-config "${PROMPT_CONFIG}" \
    --feedback-config "${FEEDBACK_CONFIG}" \
    --evaluate "${EVALUATE}" \
    --experiments-dir "${EXPERIMENT_DIR}" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
RC=$?
echo "=== AKD replay rc=${RC} at $(date +%T)"

exit "${RC}"
