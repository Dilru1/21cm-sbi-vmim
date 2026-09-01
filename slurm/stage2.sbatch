#!/bin/bash
#SBATCH --job-name=s2_nle
#SBATCH --output=logs/s2_%j.out
#SBATCH --error=logs/s2_%j.err
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --array=0-15%8        # 16 configs, 8 running at once
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
unset ${!CONDA@} 2>/dev/null || true
export MPLBACKEND=Agg


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

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CONFIG=${1:?pass a config}
shift   # everything after the config is forwarded verbatim to stage2_nle.py
python -u stage2_nle.py "$CONFIG" "$@"
