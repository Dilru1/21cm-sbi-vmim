#!/bin/bash
#SBATCH --job-name=s1_cnn
#SBATCH --output=logs/s1_cnn_%j.out
#SBATCH --error=logs/s1_cnn_%j.err
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=tycho91

set -euo pipefail

# --- make this job independent of WHATEVER submitted it -----------------------
# sbatch inherits the submitter's environment (--export=ALL). When launched from
# a notebook kernel, that kernel's conda state (CONDA_PREFIX, CONDA_DEFAULT_ENV,
# PATH, ...) leaks in and makes a *bare-name* `conda activate torch_gpu` resolve
# to the wrong same-named env (the one WITHOUT wandb). Clearing these + activating
# by FULL PATH below makes resolution deterministic regardless of the launcher.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
unset ${!CONDA@} 2>/dev/null || true
export MPLBACKEND=Agg

mkdir -p logs


NODE=$(hostname -s)
# node-conditional env: tycho91 uses the CUDA-12 miniconda env; other nodes use
# the V100/CC-7.0-compatible env (avoids the CUBLAS_ARCH crash).

if [[ "$NODE" == "tycho91" ]]; then
    module purge; module load cuda/12.0 anaconda/2024.10-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /data/ddehiwalage-don/miniconda3/envs/torch_gpu
else
    module purge; module load cuda/11.8 anaconda/2024.10-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /data/ddehiwalage-don/conda_envs/torch_nadam
fi


export WANDB_DIR=/gscratch/ddehiwalage-don/sbi_master/wandb
export WANDB_MODE=offline                 # compute node has no internet
export WANDB_PROJECT=21cm-sbi-vmim
mkdir -p "$WANDB_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- self-report the resolved env so the log proves what ran ------------------
echo "node=$NODE  gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo "PYTHON: $(which python)"
python - <<'PY'
import importlib.util, sys
s = importlib.util.find_spec("wandb")
print("WANDB:", (s.origin if s else "NOT FOUND"))
print("PYEXE:", sys.executable)
PY

CONFIG=${1:-configs/arm_cnn.yaml}
python -u stage1_compress.py "$CONFIG" "${@:2}"
