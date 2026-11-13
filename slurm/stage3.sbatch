#!/bin/bash
#SBATCH --job-name=s3_mcmc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=medium
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-5
# usage: sbatch slurm/stage3.sbatch CONFIG CSEED NSEED   (targets split over the array tasks)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env cpu
export CUDA_VISIBLE_DEVICES=""
N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
python -u scripts/stage3_mcmc.py "$1" --cseed "$2" --nseed "$3" --task "$SLURM_ARRAY_TASK_ID" --ntasks "$N"
