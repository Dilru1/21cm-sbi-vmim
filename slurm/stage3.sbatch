#!/bin/bash
#SBATCH --job-name=s3_mcmc
#SBATCH --output=logs/s3_%A_%a.out
#SBATCH --error=logs/s3_%A_%a.err
#SBATCH --partition=medium
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --array=0-5

set -euo pipefail


unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
unset ${!CONDA@} 2>/dev/null || true
export MPLBACKEND=Agg


mkdir -p logs

module purge; module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/ddehiwalage-don/conda_envs/torch_nadam

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

CONFIG=${1:?pass a config}; shift
SPLITS=6
python -u stage3_mcmc.py "$CONFIG" --task-num "${SLURM_ARRAY_TASK_ID}" --task-nb "$SPLITS" --device cpu "$@"
