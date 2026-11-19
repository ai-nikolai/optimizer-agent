#!/usr/bin/env bash
#SBATCH --job-name=openevolve_algotune_resume
#SBATCH --chdir=/hx2-weka/home/nr1713/optimizer-agent-benchmarking
#SBATCH --output=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_algotune_resume_%A_%a.out
#SBATCH --error=/hx2-weka/home/nr1713/optimizer-agent-benchmarking/logs/openevolve_algotune_resume_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1,scratch:100g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-7

# Resume the full 154-task OpenEvolve AlgoTune run. The first attempt
# (1689375) had all 8 shards start from task 1 because run_tasks.py never
# received the shard slice, so only 22 tasks were completed. This launcher
# skips those and shards the remaining 132 tasks.

set -uo pipefail

REPO="/hx2-weka/home/nr1713/optimizer-agent-benchmarking"
PREVIOUS="${PREVIOUS:-${REPO}/experiments/openevolve_algotune_full_1689375_shard*/results.csv}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

# Count tasks that still need a run (all 154 minus the previous ok rows).
REMAINING="$(
    python - "${PREVIOUS}" "${REPO}/benchmarking/openevolve/algotune/tasks_full" <<'PY'
import csv, glob, sys
from pathlib import Path
pattern, tasks_dir = sys.argv[1], Path(sys.argv[2])
done = set()
for path in glob.glob(pattern):
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if row.get("status") == "ok":
            done.add(row["task"])
total = sum(1 for p in tasks_dir.iterdir() if p.is_dir())
print(total - len(done))
PY
)"

TASKS_PER_SHARD=$(( (REMAINING + NUM_SHARDS - 1) / NUM_SHARDS ))
START=$(( SHARD_INDEX * TASKS_PER_SHARD ))

export PREPARE=0
export TOTAL_TASKS="${REMAINING}"
export NUM_SHARDS
export START
export LIMIT="${TASKS_PER_SHARD}"
export ITERATIONS="${ITERATIONS:-20}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
export TIMEOUT_PER_TASK="${TIMEOUT_PER_TASK:-10800}"
export SKIP_COMPLETED="${PREVIOUS}"
export TASKS_DIR="${TASKS_DIR:-${REPO}/benchmarking/openevolve/algotune/tasks_full}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO}/experiments/openevolve_algotune_resume_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_shard${SHARD_INDEX}}"

echo "=== OpenEvolve AlgoTune resume shard ${SHARD_INDEX}/${NUM_SHARDS}: remaining=${REMAINING} start=${START} limit=${LIMIT}"
echo "=== iterations=${ITERATIONS} checkpoint_interval=${CHECKPOINT_INTERVAL} timeout_per_task=${TIMEOUT_PER_TASK}"

exec bash "${REPO}/benchmarking/openevolve/algotune/run_slurm.sh"
