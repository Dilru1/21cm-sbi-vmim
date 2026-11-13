#!/bin/bash
#SBATCH --job-name=s1_cnn
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# usage: sbatch slurm/stage1_cnn_train.sbatch CONFIG CSEED
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env gpu
python -u scripts/stage1_compress.py "$1" --cseed "$2" --train-only
