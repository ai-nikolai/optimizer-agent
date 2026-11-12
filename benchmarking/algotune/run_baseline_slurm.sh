#!/usr/bin/env bash
#SBATCH --job-name=optagent_algotune_baseline
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_algotune_baseline_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_algotune_baseline_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=scratch:50g

# CPU-only reference/baseline evaluation for AlgoTune.
# Runs each task's unoptimized baseline.py through the real evaluator.
# No vLLM and no GPU are used.

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
ALGOTUNE_ROOT="${ALGOTUNE_ROOT:-/hx2-weka/home/nr1713/AlgoTune}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
SCRATCH="${TMPDIR:-/tmp}/optagent_algotune_baseline_${JOB_ID}"
mkdir -p "${SCRATCH}/tmp" "${SCRATCH}/hf_home" "${SCRATCH}/hf_datasets" \
    "${SCRATCH}/numba" "${SCRATCH}/mpl"
export TMPDIR="${SCRATCH}/tmp"
export HF_HOME="${SCRATCH}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets"
export NUMBA_CACHE_DIR="${SCRATCH}/numba"
export MPLCONFIGDIR="${SCRATCH}/mpl"

cd "${REPO}"
mkdir -p logs experiments

source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_kb2
export PYTHONPATH="${REPO}:${ALGOTUNE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MANIFEST="${MANIFEST:-${REPO}/benchmarking/algotune/dataset_full/manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-${REPO}/benchmarking/algotune/dataset_full}"
OUTPUT="${OUTPUT:-${REPO}/experiments/algotune_baseline_${JOB_ID}/baseline_results.csv}"

echo "=== AlgoTune baseline ${JOB_ID} at $(date +%T)"
echo "=== manifest=${MANIFEST}"
echo "=== output=${OUTPUT}"

python benchmarking/algotune/run_baseline.py \
    --manifest "${MANIFEST}" \
    --dataset-root "${DATASET_ROOT}" \
    --algotune-root "${ALGOTUNE_ROOT}" \
    --timing-runs "${ALGOTUNE_TIMING_RUNS:-3}" \
    --warmups "${ALGOTUNE_WARMUPS:-1}" \
    --solve-timeout "${ALGOTUNE_SOLVE_TIMEOUT:-120}" \
    --output "${OUTPUT}"
RC=$?
echo "=== AlgoTune baseline rc=${RC} at $(date +%T)"
exit "${RC}"
