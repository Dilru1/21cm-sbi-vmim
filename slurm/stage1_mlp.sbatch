#!/bin/bash
#SBATCH --job-name=s1_mlp
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=medium
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# usage: sbatch slurm/stage1_mlp.sbatch CONFIG CSEED
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env cpu
export CUDA_VISIBLE_DEVICES=""
python -u scripts/stage1_compress.py "$1" --cseed "$2"
