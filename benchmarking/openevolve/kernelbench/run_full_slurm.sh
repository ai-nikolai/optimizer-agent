#!/usr/bin/env bash
#SBATCH --job-name=openevolve_kb_full
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_kb_full_%A_%a.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_kb_full_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2,scratch:200g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-5

# Full 100-problem OpenEvolve KernelBench baseline.
#
# Each array task handles a contiguous slice of the 100 KernelBench level-1
# problems, with its own vLLM server and robust_kernelbench evaluator.
# Tasks are prepared once before submission; shards use PREPARE=0.
#
# Override shard count together with the array:
#   NUM_SHARDS=10 sbatch --array=0-9 benchmarking/openevolve/kernelbench/run_full_slurm.sh

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
NUM_SHARDS="${NUM_SHARDS:-6}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${TOTAL_TASKS:-100}"

TASKS_PER_SHARD=$(( (TOTAL_TASKS + NUM_SHARDS - 1) / NUM_SHARDS ))
START=$(( SHARD_INDEX * TASKS_PER_SHARD ))

export PREPARE=0
export PROBLEM_START="${START}"
export PROBLEM_LIMIT="${TASKS_PER_SHARD}"
export ITERATIONS="${ITERATIONS:-50}"
export LEVEL="${LEVEL:-1}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO}/experiments/openevolve_kernelbench_full_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_shard${SHARD_INDEX}}"

echo "=== OpenEvolve KernelBench full shard ${SHARD_INDEX}/${NUM_SHARDS}: start=${START} limit=${TASKS_PER_SHARD} total=${TOTAL_TASKS}"
echo "=== experiment=${EXPERIMENT_DIR}"

exec bash "${REPO}/benchmarking/openevolve/kernelbench/run_slurm.sh"
