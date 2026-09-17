#!/bin/bash
#SBATCH --job-name=s1_raw
#SBATCH --output=logs/s1_raw_%j.out
#SBATCH --error=logs/s1_raw_%j.err
#SBATCH --partition=short
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs
module purge; module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/ddehiwalage-don/conda_envs/torch_nadam
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
CONFIG=${1:-configs/arm_baseline.yaml}
python -u stage1_raw.py "$CONFIG"
