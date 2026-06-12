#!/usr/bin/env bash
#SBATCH --job-name=pilarnet
#SBATCH --partition=roma
#SBATCH --account=neutrino:ml-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/array_%A_%a.out
#SBATCH --error=hpc/logs/array_%A_%a.out
# Array range: set via --array=1-N on submission; N = wc -l manifest.txt
# Example:    sbatch --array=1-$(wc -l < hpc/manifest.txt) hpc/array.sbatch

set -eo pipefail               # NB: no `-u`; conda's binutils activate hook
                               # references $ADDR2LINE unguarded and trips it.
cd "$SLURM_SUBMIT_DIR"
mkdir -p hpc/logs shards

# --- conda env -------------------------------------------------------------
# `conda activate` needs the hook; loading it here is portable across nodes.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310_torch

# --- build cpp solver if missing (one-time, idempotent) -------------------
SO="cpp/spsolver_cpp.cpython-310-x86_64-linux-gnu.so"
if ! ls cpp/spsolver_cpp*.so >/dev/null 2>&1; then
    (cd cpp && ./build.sh)
fi

# --- pick this task's input file -----------------------------------------
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/manifest.txt)
MODE=$(echo "$LINE" | awk '{print $1}')
IN=$(echo "$LINE"   | awk '{print $2}')
BASE=$(basename "$IN" .h5)
OUT="shards/shard_${SLURM_ARRAY_TASK_ID}_${MODE}_${BASE}.h5"

if [ -f "$OUT" ]; then
    echo "shard exists, skipping: $OUT"
    exit 0
fi

echo "task ${SLURM_ARRAY_TASK_ID}  mode=${MODE}  in=${IN}"

if [ "$MODE" = "electron" ]; then
    FLAG="--electron-files"
else
    FLAG="--pi0-files"
fi

srun python make_pilarnet.py \
    $FLAG "$IN" \
    --out "$OUT" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --backend cpp

echo "done: $OUT"
