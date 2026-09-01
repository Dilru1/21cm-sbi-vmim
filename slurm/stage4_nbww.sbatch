#!/bin/bash
#SBATCH --job-name=s4_eval
#SBATCH --output=logs/s4_%j.out
#SBATCH --error=logs/s4_%j.err
#SBATCH --partition=short
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#
# Stage 4 -- single merged evaluator (eval.py). Filtered-only, publication output:
#   metrics.csv, metrics.tex, sbc_filtered_dlogp<X>.pdf, corner_sim<ID>.pdf
#   (+ latent/<arm>/*.pdf when --latent is passed)
#
# TWO WAYS TO CALL IT:
#
# 1) PRESET MODE  -- pick a named arm group baked in below:
#      sbatch slurm/stage4.sbatch preset mlp_vmim
#      sbatch slurm/stage4.sbatch preset cnn_mse
#
# 2) LIST MODE    -- pass yamls + family/scope on the command line (noise ladder):
#      sbatch slurm/stage4.sbatch list --name no_jitter_nsf --family nsf --scope std \
#          configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml \
#          configs_seeds/noise/arm_cnn_vmim_no_jitter_n2.yaml \
#          configs_seeds/noise/arm_cnn_vmim_no_jitter_n3.yaml \
#          configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml
#
#   list-mode options (before the yaml list):
#     --name NAME     report dir -> SBC_OUTPUT/NAME     (default: family_scope)
#     --family F      gmm|maf|nsf                       (default: nsf)
#     --scope S       std|raw                           (default: std)
#     --dlogp X       logp filter                       (default: 10)
#     --label-mode M  hours|basename                    (default: hours)
#     --no-all-seeds  one arm per yaml (default: expand every _s<seed> on disk)
#     --seeds a b c   restrict seed expansion
#     --latent        also emit latent-space diagnostics
# ------------------------------------------------------------------------------
set -euo pipefail

# --- make this job independent of WHATEVER submitted it -----------------------
# sbatch inherits the submitter's environment (--export=ALL). Launched from a
# notebook kernel, that leaks conda/PYTHON state in and can leave `module`
# undefined (non-login batch shell). Clear the leaks, ensure the module system
# is initialized, then activate the env by FULL PATH so resolution is
# deterministic regardless of who submitted the job.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
unset ${!CONDA@} 2>/dev/null || true
export MPLBACKEND=Agg
mkdir -p logs

# initialize the environment-modules system if the submitter's shell didn't
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    [[ -r "$f" ]] && source "$f" && break
  done
fi

module purge
module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/ddehiwalage-don/conda_envs/torch_nadam
export CUDA_VISIBLE_DEVICES=""

# --- self-report the resolved env so the log proves what ran ------------------
echo "node=$(hostname -s)"
echo "PYTHON: $(which python)"

DLOGP=10
COMMON_FLAGS=(--dlogp "$DLOGP" --corner-sim median)   # filtered-only + one overlay corner

# ============================== PRESET definitions ============================
# each preset is a list of packed items  CONFIG|FAMILY|SCOPE[|LABEL]
declare -a PRESET_mlp_vmim=(
  "configs/mlp_/arm_mlp_pdf.yaml|gmm|std|PDF"
  "configs/mlp_/arm_mlp_ps.yaml|gmm|std|PS"
  "configs/mlp_/arm_mlp_pdf_ps.yaml|gmm|std|PDF+PS"
)
declare -a PRESET_cnn_vmim=(
  "configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml|nsf|std|no-jit n1"
  "configs_seeds/noise/arm_cnn_vmim_no_jitter_n2.yaml|nsf|std|no-jit n2"
  "configs_seeds/noise/arm_cnn_vmim_no_jitter_n3.yaml|nsf|std|no-jit n3"
  "configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml|nsf|std|no-jit n4"
)
declare -a PRESET_cnn_mse=(
  "configs/mse_/arm_cnn_mse_1.yaml|gmm|std|MSE 1"
  "configs/mse_/arm_cnn_mse_2.yaml|gmm|std|MSE 2"
)

# ================================ dispatch ===================================
if [[ $# -lt 1 ]]; then
  echo "usage: sbatch slurm/stage4.sbatch [preset NAME | list --name ... yaml...]"; exit 1
fi
CALL="$1"; shift

ITEMS=(); OUT=""; EXTRA=()

if [[ "$CALL" == "preset" ]]; then
  MODE="${1:-}"; shift || true
  case "$MODE" in
    mlp_vmim) NAME="mlp_vmim"; arr=("${PRESET_mlp_vmim[@]}") ;;
    cnn_vmim) NAME="cnn_vmim"; arr=("${PRESET_cnn_vmim[@]}") ;;
    cnn_mse)  NAME="cnn_mse";  arr=("${PRESET_cnn_mse[@]}") ;;
    *) echo "unknown preset '$MODE' (mlp_vmim|cnn_vmim|cnn_mse)"; exit 1 ;;
  esac
  for it in "${arr[@]}"; do ITEMS+=(--item "$it"); done
  # remaining args after the preset name are passed through (e.g. --latent, --all-seeds)
  EXTRA=("$@")
  OUT="eval_reports/${NAME}"

elif [[ "$CALL" == "list" ]]; then
  NAME=""; FAMILY="nsf"; SCOPE="std"; LABEL_MODE="hours"
  ALL_SEEDS="--all-seeds"; SEEDS=(); YAMLS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)        NAME="$2"; shift 2 ;;
      --family)      FAMILY="$2"; shift 2 ;;
      --scope)       SCOPE="$2"; shift 2 ;;
      --dlogp)       DLOGP="$2"; COMMON_FLAGS=(--dlogp "$DLOGP" --corner-sim median); shift 2 ;;
      --label-mode)  LABEL_MODE="$2"; shift 2 ;;
      --no-all-seeds) ALL_SEEDS=""; shift ;;
      --seeds)       shift; while [[ $# -gt 0 && "$1" != -* ]]; do SEEDS+=("$1"); shift; done ;;
      --latent)      EXTRA+=(--latent); shift ;;
      -*) echo "unknown option $1"; exit 1 ;;
      *)  YAMLS+=("$1"); shift ;;
    esac
  done
  [[ ${#YAMLS[@]} -eq 0 ]] && { echo "ERROR: no yaml files given"; exit 1; }

  declare -A HOURS=( [1]="100 h (ns=1)" [2]="25 h (ns=2)" [3]="11.1 h (ns=3)" [4]="6.25 h (ns=4)" )
  for y in "${YAMLS[@]}"; do
    [[ -f "$y" ]] || { echo "ERROR: config not found: $y"; exit 1; }
    base="$(basename "$y" .yaml)"
    if [[ "$LABEL_MODE" == "hours" ]]; then
      nlev="$(echo "$base" | grep -oE 'n[0-9]+$' | tr -d 'n' || true)"
      label="${HOURS[${nlev:-x}]:-$base}"
    else
      label="$base"
    fi
    case "$base" in *no_jitter*) label="no-jit ${label}" ;; *jitter*) label="jit ${label}" ;; esac
    ITEMS+=(--item "${y}|${FAMILY}|${SCOPE}|${label}")
  done
  [[ -n "$ALL_SEEDS" ]] && EXTRA+=("$ALL_SEEDS")
  [[ ${#SEEDS[@]} -gt 0 ]] && EXTRA+=(--seeds "${SEEDS[@]}")
  OUT="SBC_OUTPUT/${NAME:-${FAMILY}_${SCOPE}}"
else
  echo "first arg must be 'preset' or 'list'"; exit 1
fi

mkdir -p "$OUT"
echo "report : $OUT"
echo "items  :"; printf '  %s\n' "${ITEMS[@]}"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "extra  : ${EXTRA[*]}"
echo

python -u eval.py "${ITEMS[@]}" "${COMMON_FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --out "$OUT"

echo "done -> $OUT/ (metrics.csv, metrics.tex, sbc_filtered_dlogp${DLOGP}.pdf, corner_sim*.pdf)"
