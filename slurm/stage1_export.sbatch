#!/bin/bash
#SBATCH --job-name=s1_export
#SBATCH --output=logs/s1_export_%j.out
#SBATCH --error=logs/s1_export_%j.err
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

# Export-only stage 1: reload bestprobe checkpoint, write NLE summary memmaps.
#   sbatch slurm/stage1_export.sbatch configs/arm_cnn_vmim_blue_floor.yaml
# save_batch_size=256: the new CNN keeps full 32^3 resolution through the
# early blocks, so batch 8192 allocates 32 GiB per activation tensor (the OOM).

set -euo pipefail
mkdir -p logs
NODE=$(hostname -s)
# node-conditional env (same convention as stage1_cnn.sbatch)
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
echo "node=$NODE  gpus=${CUDA_VISIBLE_DEVICES:-none}"

CONFIG=${1:?usage: sbatch slurm/stage1_export.sbatch configs/arm_xxx.yaml}

python -u stage1_compress.py "$CONFIG" --export-only \
  -o compressor.save_batch_size=512 "${@:2}"
