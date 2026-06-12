#!/usr/bin/env bash
#SBATCH --job-name=pilarnet-merge
#SBATCH --partition=roma
#SBATCH --account=neutrino:ml-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=hpc/logs/merge_%j.out
#SBATCH --error=hpc/logs/merge_%j.out
# Run after the array completes. From the repo root:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> hpc/merge.sbatch

set -eo pipefail               # no `-u`: conda binutils activate hook trips it
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310_torch

python merge_pilarnet.py --in 'shards/shard_*.h5' --out pilarnet_epi0.h5
