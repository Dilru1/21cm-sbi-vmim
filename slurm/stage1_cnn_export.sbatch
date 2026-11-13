#!/bin/bash
#SBATCH --job-name=s1_exp
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4%2
# usage: sbatch slurm/stage1_cnn_export.sbatch CONFIG CSEED   (one shard of the noise axis per task)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env gpu
N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
python -u scripts/stage1_compress.py "$1" --cseed "$2" --export-only --shard "$SLURM_ARRAY_TASK_ID" --nshard "$N"
