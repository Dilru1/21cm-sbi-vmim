#!/bin/bash
#SBATCH --job-name=s2_nle
#SBATCH --output=logs/stage2/s2_%j.out
#SBATCH --error=logs/stage2/s2_%j.err
#SBATCH --partition=medium
#SBATCH --time=17:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# NOT --nodelist: -w requires ALL listed hosts, so --nodelist=tycho[84-91]
# makes this an 8-node job that waits for every one of them to be free.
# The feature constraint means "any one of tycho84-91" instead.
#  SBATCH --constraint="tycho2020|tycho2022|tycho2025|tycho2026"

set -euo pipefail
mkdir -p logs/stage2          # belt and braces; the dir must already exist for
                              # SLURM to open the --output file at job start

module purge; module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_gpu

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Stage 2 stays on CPU by design: the NLE is ~50k parameters and the GPUs are
# the scarce resource for stage-1 training and export. This also means stage 2
# never competes with the minerva/long GPU queues.
export CUDA_VISIBLE_DEVICES=""

CONFIG=${1:?pass a config}
shift   # everything after the config is forwarded verbatim to stage2_nle.py

python -u stage2_nle.py "$CONFIG" "$@"