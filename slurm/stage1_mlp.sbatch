#!/bin/bash
#SBATCH --job-name=s1_mlp
#SBATCH --output=logs/s1_mlp_%j.out
#SBATCH --error=logs/s1_mlp_%j.err
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail
mkdir -p logs
NODE=$(hostname -s)
if [[ "$NODE" == "tycho91" ]]; then
    module purge; module load cuda/12.0 anaconda/2024.10-1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch_gpu
else
    module purge; module load cuda/11.8 anaconda/2024.10-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /data/ddehiwalage-don/conda_envs/torch_nadam
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
CONFIG=${1:-configs/arm_mlp.yaml}
python -u stage1_compress.py "$CONFIG" "${@:2}"