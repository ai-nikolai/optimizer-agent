#!/usr/bin/env bash
#SBATCH --job-name=optagent_algotune_full
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_algotune_full_%A_%a.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/optagent_algotune_full_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1,scratch:100g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-3

# Full 154-task AlgoTune run, sharded as a Slurm array job.
#
# Each array task starts its own vLLM server on one GPU, runs a contiguous
# slice of the manifest on CPU, and writes to its own experiment directory.
# Scratch is requested per array task, so caches and temporary arrays do not
# touch the 1 TB home quota.
#
# Default: 4 shards. Override the array and shard count together:
#   NUM_SHARDS=8 sbatch --array=0-7 benchmarking/algotune/run_full_slurm.sh
#
# The inner launcher is the normal benchmarking/algotune/run_slurm.sh; this
# script only selects the manifest, dataset root and task range.

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${TOTAL_TASKS:-154}"

TASKS_PER_SHARD=$(( (TOTAL_TASKS + NUM_SHARDS - 1) / NUM_SHARDS ))
START=$(( SHARD_INDEX * TASKS_PER_SHARD ))

export MANIFEST="${MANIFEST:-${REPO}/benchmarking/algotune/dataset_full/manifest.json}"
export DATASET_ROOT="${DATASET_ROOT:-${REPO}/benchmarking/algotune/dataset_full}"
export START
export LIMIT="${TASKS_PER_SHARD}"
export ROUNDS="${ROUNDS:-5}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO}/experiments/algotune_full_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_shard${SHARD_INDEX}}"

echo "=== AlgoTune full shard ${SHARD_INDEX}/${NUM_SHARDS}: start=${START} limit=${LIMIT} total=${TOTAL_TASKS}"
echo "=== manifest=${MANIFEST}"
echo "=== experiment=${EXPERIMENT_DIR}"

exec bash "${REPO}/benchmarking/algotune/run_slurm.sh"
