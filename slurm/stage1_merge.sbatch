#!/bin/bash
#SBATCH --job-name=s1_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=medium
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# usage: sbatch slurm/stage1_merge.sbatch CONFIG CSEED
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env cpu
python -u scripts/merge_shards.py "$1" --cseed "$2"
