#!/usr/bin/env bash
#SBATCH --job-name=openevolve_algotune_full
#SBATCH --output=logs/openevolve_algotune_full_%A_%a.out
#SBATCH --error=logs/openevolve_algotune_full_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1,scratch:100g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-7

# Full 154-task OpenEvolve AlgoTune run, 20 iterations per task, with a
# checkpoint every iteration. Tasks are prepared once into tasks_full; shards
# use PREPARE=0.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ ! -d "${REPO}/benchmarking" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname "${REPO}")"
done
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${TOTAL_TASKS:-154}"

TASKS_PER_SHARD=$(( (TOTAL_TASKS + NUM_SHARDS - 1) / NUM_SHARDS ))
START=$(( SHARD_INDEX * TASKS_PER_SHARD ))

export PREPARE=0
export TASKS_DIR="${TASKS_DIR:-${REPO}/benchmarking/agents/openevolve/algotune/tasks_full}"
export ITERATIONS="${ITERATIONS:-20}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
export START
export LIMIT="${TASKS_PER_SHARD}"
export TIMEOUT_PER_TASK="${TIMEOUT_PER_TASK:-10800}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO}/experiments/${RUN_PREFIX:-}openevolve_algotune_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_shard${SHARD_INDEX}}"

echo "=== OpenEvolve AlgoTune full shard ${SHARD_INDEX}/${NUM_SHARDS}: start=${START} limit=${LIMIT} total=${TOTAL_TASKS}"
echo "=== iterations=${ITERATIONS} checkpoint_interval=${CHECKPOINT_INTERVAL}"
echo "=== experiment=${EXPERIMENT_DIR}"

exec bash "${REPO}/benchmarking/agents/openevolve/algotune/run_slurm.sh"
