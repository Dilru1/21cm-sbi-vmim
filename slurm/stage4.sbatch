#!/bin/bash
#SBATCH --job-name=s4_eval
#SBATCH --output=logs/stage4/s4_%j.out
#SBATCH --error=logs/stage4/s4_%j.err
#SBATCH --partition=short
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/stage4

# hermetic env: stage 4 is pure numpy/matplotlib, and inheriting a notebook's
# PYTHONPATH has bitten this pipeline before
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
unset ${!CONDA@} 2>/dev/null || true
export MPLBACKEND=Agg

if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
           /usr/share/lmod/lmod/init/bash; do
    [[ -r "$f" ]] && source "$f" && break
  done
fi
module purge; module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/ddehiwalage-don/conda_envs/torch_nadam

# stage 4 reads chains and does numpy -- no GPU, so it never queues behind
# stage-1 training or export
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

LISTS=${1:?usage: sbatch slurm/stage4.sbatch <list[ list2 ...]> <outdir> [extra args]}
OUT=${2:?usage: sbatch slurm/stage4.sbatch <list[ list2 ...]> <outdir> [extra args]}
shift 2

# --- pre-flight: fail here, not 40 minutes in ---------------------------------
LIST_ARGS=()
for L in $LISTS; do
  [[ -f "$L" ]] || { echo "ERROR: no such list file: $L" >&2; exit 1; }
  LIST_ARGS+=(--list "$L")
done

# every arm's directory must exist and hold *.dat. stage4_eval resolves the 4th
# column as a SUBSTRING and exits if it matches more than one chains dir, so a
# line naming 'standard_t_nsf' when both _seed_42 and _seed_43 exist is fatal --
# catch it now rather than after the job starts.
NARM=0
for L in $LISTS; do
  while IFS= read -r raw; do
    line="${raw%%#*}"; line="$(echo "$line" | xargs)"
    [[ -z "$line" ]] && continue
    NARM=$((NARM+1))
    p="$(echo "$line" | cut -d, -f2 | xargs)"
    sub="$(echo "$line" | cut -d, -f4 | xargs)"
    if [[ ! -d "$p" ]]; then
      echo "ERROR: [$L] path does not exist: $p" >&2; exit 1
    fi
    if [[ -n "$sub" ]]; then
      n=$(find "$p" -maxdepth 2 -type d -name "*${sub}*" 2>/dev/null | wc -l)
      if   (( n == 0 )); then echo "ERROR: [$L] no chains dir matching '$sub' under $p" >&2; exit 1
      elif (( n > 1 ));  then echo "ERROR: [$L] '$sub' matches $n dirs under $p -- stage4 will refuse. Name the seed in full." >&2; exit 1
      fi
    fi
  done < "$L"
done
echo "lists: $LISTS   arms: $NARM   out: $OUT"

mkdir -p "$OUT"

python -u stage4_eval.py "${LIST_ARGS[@]}" \
    --out "$OUT" \
    --dlogp 10 \
    --n-corner 10 \
    --filtered-only \
    ${ADVANCED:+--advanced} \
    "$@"

echo "done -> $OUT/"
ls -la "$OUT" | head -20
