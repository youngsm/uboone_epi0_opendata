#!/usr/bin/env bash
#SBATCH --job-name=pilarnet-full
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/full_%A_%a.out
#SBATCH --error=hpc/logs/full_%A_%a.out
# Array job for the multi-class full-event dataset. Submission:
#   bash hpc/build_manifest_full.sh
#   sbatch --array=1-$(wc -l < hpc/manifest_full.txt) hpc/array_full.sbatch
# Memory is higher than e/pi0 mode (full events keep all truth-labeled
# hits, so per-event point counts and instance tables are larger).

set -eo pipefail               # no `-u`: conda binutils activate hook trips it
cd "$SLURM_SUBMIT_DIR"
mkdir -p hpc/logs shards_full

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310_torch

if ! ls cpp/spsolver_cpp*.so >/dev/null 2>&1; then
    (cd cpp && ./build.sh)
fi

IN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/manifest_full.txt)
BASE=$(basename "$IN" .h5)
OUT="shards_full/shard_${SLURM_ARRAY_TASK_ID}_${BASE}.h5"

if [ -f "$OUT" ]; then
    echo "shard exists, skipping: $OUT"
    exit 0
fi

echo "task ${SLURM_ARRAY_TASK_ID}  in=${IN}"

srun python make_pilarnet_full.py \
    --files "$IN" \
    --out "$OUT" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --backend cpp

echo "done: $OUT"
