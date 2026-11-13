#!/bin/bash
#SBATCH --job-name=s4_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=medium
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# usage: sbatch slurm/stage4.sbatch EVAL_CONFIG
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env cpu
export CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
python -u scripts/stage4_eval.py "$1"
