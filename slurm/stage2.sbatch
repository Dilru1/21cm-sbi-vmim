#!/bin/bash
#SBATCH --job-name=s2_nle
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=medium
#SBATCH --time=17:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# usage: sbatch slurm/stage2.sbatch CONFIG CSEED NSEED
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env nle
export CUDA_VISIBLE_DEVICES=""
python -u scripts/stage2_nle.py "$1" --cseed "$2" --nseed "$3"
