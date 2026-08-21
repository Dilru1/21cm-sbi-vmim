#!/bin/bash
# nle_grid.sh
# Submit the compressor -> NLE -> MCMC pipeline as chained SLURM jobs for one
# config (one seed -- the seed lives in the yaml / -o overrides, not here).
#
# The (family x t-scaling) grid fans stage 2 and stage 3 into independent jobs,
# one per combination. Stage 1 is shared by the whole grid (the compressor does
# not depend on family/scaling), so it runs ONCE and every stage-2 job chains
# off it with afterok.
#
# USAGE
#   bash nle_grid.sh CONFIG ["families"] ["scalings"] [STAGES]
#
#     CONFIG     required, e.g. configs/cnn.yaml
#     families   default "gmm maf nsf"
#     scalings   default "std raw"
#     STAGES     which stages to run (default "123"); see below
#
#   bash nle_grid.sh config/cnn.arm "nsf" "std"
#   bash nle_grid.sh config/cnn.arm "nsf" "std" 23      # skip stage1 (trained)
#   bash nle_grid.sh config/cnn.arm "gmm maf nsf" "std raw" 1

##EXAMPLE USE CASE

# bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml "nsf" "std" 1e23

#
# STAGE SELECTION (4th arg; contains the chars 1 / 2 / 3 in any combination)
#   1    stage 1: train + export        -> slurm/stage1_cnn.sbatch
#   1t   stage 1: train only, NO export (--no-export; probe/diagnostic runs)
#   1e   stage 1: export only         (--export-only; re-export from a ckpt)
#   2    stage 2: NLE grid
#   3    stage 3: MCMC grid
#   123  full pipeline chained 1 -> 2 -> 3   (default)
#   23   stages 2 and 3 (stage 1 assumed already done)
#   Notes:
#     * 1 / 1t / 1e are mutually exclusive (only the first match wins).
#     * with stage 1 AND stage 2 selected, every stage-2 job chains afterok on
#       the single stage-1 job; each stage-3 job chains on its stage-2 job.
#     * stage 2 needs EXPORTED summaries, so pair it with plain '1' (not 1t).
#
# env:   BS_GMM / BS_MAF / BS_NSF = per-family stage-2 batch size (-o nle.batch_size).
#        Empty -> yaml value. Defaults override NSF/MAF to 32, keep gmm at yaml.
set -euo pipefail

# ---- pull out repeatable -o/--override key=val; keep the rest positional ------
# These user overrides are forwarded to stage 1 AND to the stage-2/3 grid, so a
# seed override (e.g. -o compressor.init_seed=1) resolves the SAME seed-tagged
# arm directory in every stage. -o may appear anywhere on the command line.
USER_OV=()
POS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--override) USER_OV+=(-o "${2:?-o needs a key=value}"); shift 2 ;;
    -o*)           USER_OV+=(-o "${1#-o}"); shift ;;
    *)             POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

CONFIG=${1:?usage: nle_grid.sh config.yaml ["gmm maf nsf"] ["std raw"] [STAGES] [-o key=val ...]}
FAMILIES=(${2:-gmm maf nsf})
SCALINGS=(${3:-std raw})
STAGES=${4:-123}

BS_GMM=${BS_GMM:-}
BS_MAF=${BS_MAF:-32}
BS_NSF=${BS_NSF:-32}
bs_for() { case "$1" in gmm) echo "${BS_GMM}";; maf) echo "${BS_MAF}";; nsf) echo "${BS_NSF}";; *) echo "";; esac; }

# ---- decode STAGES -----------------------------------------------------------
DO1=0; DO1_MODE="full"; DO2=0; DO3=0
case "$STAGES" in
  *1t*) DO1=1; DO1_MODE="train"  ;;
  *1e*) DO1=1; DO1_MODE="export" ;;
  *1*)  DO1=1; DO1_MODE="full"   ;;
esac
[[ "$STAGES" == *2* ]] && DO2=1
[[ "$STAGES" == *3* ]] && DO3=1
if [[ $DO1 -eq 0 && $DO2 -eq 0 && $DO3 -eq 0 ]]; then
  echo "STAGES='$STAGES' selects nothing. Use e.g. 123, 23, 1, 1t, 1e."; exit 1
fi
if [[ $DO2 -eq 1 && $DO1 -eq 1 && "$DO1_MODE" != "full" ]]; then
  echo "WARNING: stage 2 needs exported summaries, but stage-1 mode is"
  echo "         '$DO1_MODE' (no export). Stage 2 will fail unless a prior"
  echo "         export exists. Use plain '1' (train+export) with stage 2."
fi

arm_yaml=$(grep -E "^arm_name:" "$CONFIG" | head -1 | awk '{print $2}')
echo "config=$CONFIG  arm(yaml)=$arm_yaml"
echo "stages=$STAGES  stage1=$([[ $DO1 -eq 1 ]] && echo $DO1_MODE || echo skip)"\
     " stage2=$([[ $DO2 -eq 1 ]] && echo yes || echo skip)"\
     " stage3=$([[ $DO3 -eq 1 ]] && echo yes || echo skip)"
echo "overrides: ${USER_OV[*]:-none}"
echo "grid: families=[${FAMILIES[*]}]  scalings=[${SCALINGS[*]}]"
echo "batches: gmm=${BS_GMM:-yaml} maf=${BS_MAF:-yaml} nsf=${BS_NSF:-yaml}"
echo

# ---- stage 1 (shared by the whole grid, submitted ONCE) ----------------------
J1=""
if [[ $DO1 -eq 1 ]]; then
  s1_extra=()
  case "$DO1_MODE" in
    train)  s1_extra=(--no-export) ;;
    export) s1_extra=(--export-only) ;;
  esac
  # shellcheck disable=SC2086
  J1=$(sbatch --job-name="s1_$(basename "$CONFIG" .yaml)" \
              slurm/stage1_cnn.sbatch "$CONFIG" ${USER_OV[@]+"${USER_OV[@]}"} "${s1_extra[@]}" \
       | awk '{print $NF}')
  echo "stage1 ($DO1_MODE): $J1"
  echo
fi

# ---- stage 2 x stage 3 grid --------------------------------------------------
n=0
for fam in "${FAMILIES[@]}"; do
  for sc in "${SCALINGS[@]}"; do
    [[ "$sc" == "raw" ]] && flag="--raw-t" || flag=""
    ov=(-o "nle.model=${fam}")
    bs=$(bs_for "$fam")
    [[ -n "$bs" ]] && ov+=(-o "nle.batch_size=${bs}")
    ov+=(${USER_OV[@]+"${USER_OV[@]}"})     # user overrides -> stage 2 and stage 3

    J2=""
    if [[ $DO2 -eq 1 ]]; then
      dep=(); [[ -n "$J1" ]] && dep=(--dependency=afterok:$J1)
      # shellcheck disable=SC2086
      J2=$(sbatch --job-name="s2_${fam}_${sc}" "${dep[@]}" \
                  slurm/stage2.sbatch "$CONFIG" "${ov[@]}" $flag \
           | awk '{print $NF}')
    fi

    J3=""
    if [[ $DO3 -eq 1 ]]; then
      dep=(); [[ -n "$J2" ]] && dep=(--dependency=afterok:$J2)
      # shellcheck disable=SC2086
      J3=$(sbatch --job-name="s3_${fam}_${sc}" "${dep[@]}" \
                  slurm/stage3.sbatch "$CONFIG" "${ov[@]}" $flag \
           | awk '{print $NF}')
    fi

    printf "  %-4s %-3s  stage2=%-10s stage3=%-10s batch=%s\n" \
           "$fam" "$sc" "${J2:-skip}" "${J3:-skip}" "${bs:-yaml}"
    n=$((n+1))
  done
done

echo
echo "submitted grid of $n family/scaling combinations."
echo "monitor: squeue -u \$USER"
echo
echo "leaf dirs (no collisions):"
for fam in "${FAMILIES[@]}"; do for sc in "${SCALINGS[@]}"; do
  scope=$([[ "$sc" == "raw" ]] && echo raw_t || echo standard_t)
  echo "  nle/$scope/$fam   chains/${scope}_${fam}"
done; done