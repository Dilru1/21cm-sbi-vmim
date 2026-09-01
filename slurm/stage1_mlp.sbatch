#!/bin/bash
#SBATCH --job-name=s1_mlp
#SBATCH --output=logs/s1_mlp_%j.out
#SBATCH --error=logs/s1_mlp_%j.err
#SBATCH --partition=medium
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs
NODE=$(hostname -s)

# initialize the environment-modules system if the submitter's shell didn't
# (e.g. when submitted from a notebook kernel -> "module: command not found").
# if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    [[ -r "$f" ]] && source "$f" && break
  done
# fi

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
