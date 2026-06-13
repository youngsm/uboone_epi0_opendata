#!/usr/bin/env bash
#SBATCH --job-name=pilarnet-full-merge
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/full_merge_%j.out
#SBATCH --error=hpc/logs/full_merge_%j.out
# Run after array_full completes:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> hpc/merge_full.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310_torch

python merge_pilarnet.py --in 'shards_full/shard_*.h5' --out pilarnet_full.h5
