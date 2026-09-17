#!/bin/bash
#SBATCH --job-name=s1_exp
#SBATCH --output=logs/stage1/s1_exp_%A_%a.out
#SBATCH --error=logs/stage1/s1_exp_%A_%a.err
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4%2
#SBATCH --nodelist=tycho[91,58]

# Stage 1 EXPORT, sharded over the noise axis as a SLURM array.
#
#   sbatch slurm/stage1_export.sbatch configs_seeds/vmim_/arm_cnn_vmim_jitter_n1.yaml
#   sbatch --dependency=afterok:<trainjob> slurm/stage1_export.sbatch <config>
#
# Each task exports noise reps [START, END) for EVERY simulation into
#   {arm}/summaries/part_<START>_<END>/
# and stage1_merge.sbatch concatenates them afterwards.
#
# Why this is safe: DeterministicNoisyExport derives its noise index from
#   (original_sim_id * TOTAL + noise_rep + 10007*channel) % n_noise_total
# using TOTAL (not the shard width), so a shard produces byte-identical rows to
# the corresponding slice of an unsharded export. The shards are disjoint and
# their union is the full table, just reordered -- which is fine, because
# stage 2 splits by sim ID via np.isin and shuffles.
#
# --array=0-9%5 : ten shards, at most five running at once. Raise or drop the
# %5 if your GPU allocation allows; lower it if you are queueing behind others.
# Change the array RANGE and the shard width follows automatically.
#
# TOTAL defaults to 1000 (data.total_nnoise). Override from the command line:
#   TOTAL_NNOISE=500 sbatch slurm/stage1_export.sbatch <config>
#
# mem=48G, not 96G: each shard writes ~1/N of the output, and the new
# CubeDataset holds the cubes once instead of twice.
#
# IMPORTANT: every shard constructs its own CubeDataset. Without
# data.cube_cache_dir set, that means N tasks each re-reading ~8 GB of float64
# per signal file off shared storage. Run compute_cube_stats.py with
# --cube-cache first, and point data.cube_cache_dir at the same directory.

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


#export CUDA_VISIBLE_DEVICES=""


CONFIG=${1:?usage: sbatch slurm/stage1_export.sbatch configs/arm_xxx.yaml}

TOTAL=${TOTAL_NNOISE:-1000}

# SLURM_ARRAY_TASK_COUNT is not set on every SLURM version; derive it from the
# task bounds when missing (the %N throttle does not change these).
NSHARD=${SLURM_ARRAY_TASK_COUNT:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}
IDX=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))

if (( NSHARD > TOTAL )); then
    echo "ERROR: $NSHARD shards for only $TOTAL noise reps" >&2; exit 1
fi

PER=$(( TOTAL / NSHARD ))
START=$(( IDX * PER ))
END=$(( START + PER ))
# the last shard absorbs the remainder when TOTAL is not divisible by NSHARD
if (( IDX == NSHARD - 1 )); then END=$TOTAL; fi

echo "node=$NODE  gpus=${CUDA_VISIBLE_DEVICES:-none}  cpus=${SLURM_CPUS_PER_TASK}"
echo "shard ${IDX}/${NSHARD}  noise[${START},${END})  of TOTAL=${TOTAL}"

python -u stage1_compress.py "$CONFIG" --export-only \
    --noise-start "$START" --noise-end "$END" \
    -o compressor.save_batch_size=512 "${@:2}"
