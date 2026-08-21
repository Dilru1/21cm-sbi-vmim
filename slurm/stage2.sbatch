#!/bin/bash
#SBATCH --job-name=s2_nle
#SBATCH --output=logs/s2_%j.out
#SBATCH --error=logs/s2_%j.err
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

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
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DYNAMIC=FALSE; export OMP_DYNAMIC=FALSE

export CUDA_VISIBLE_DEVICES=""   # NLE is a tiny MLP; CPU avoids V100/CC-7.0 CUBLAS crash
CONFIG=${1:?pass a config}
shift   # everything after the config is forwarded verbatim to stage2_nle.py

python -u stage2_nle.py "$CONFIG" "$@"