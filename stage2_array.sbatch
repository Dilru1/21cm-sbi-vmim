#!/bin/bash
#SBATCH --job-name=s2_nle
#SBATCH --output=logs/s2_%A_%a.out
#SBATCH --error=logs/s2_%A_%a.err
#SBATCH --partition=medium
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # 1 core/run: extra threads REDUCE throughput here
#SBATCH --mem=8G                   # dataset is ~300 MB; 96G was pointless
#SBATCH --array=0-11%8             # 12 runs, 8 at a time -> packs a node
#SBATCH --nodelist=tycho05,tycho21,tycho22,tycho27,tycho49,tycho79

set -euo pipefail
mkdir -p logs
module purge; module load cuda/11.8 anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/ddehiwalage-don/conda_envs/torch_nadam

# ONE thread per process (measured optimum for the 4-D flow).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""      # staying on CPU as you found it stabler

# jobs.list: one "CONFIG [extra args]" per line; array index picks the line.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs.list)

python -u stage2_nle_fast.py $LINE \
    -o nle.batch_size=32 -o nle.lr_phase1=5e-4 -o nle.lr_phase2=1e-4
