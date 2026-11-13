#!/bin/bash
# Submit every (compressor seed, NLE seed) run of one config as a dependency chain.
#   bash slurm/submit.sh configs/patha/vmim_k4.yaml
# none: stage2 -> stage3        mlp: stage1 -> stage2 -> stage3
# cnn:  stage1 train -> export array -> merge -> stage2 -> stage3
set -euo pipefail
CFG=${1:?usage: bash slurm/submit.sh CONFIG}
source slurm/env.sh
mkdir -p logs
read -r TYPE CSEEDS NSEEDS < <($SBI_PYTHON - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
t = c["compressor"]["type"]
cs = [0] if t == "none" else c["seeds"]["compressor"]
print(t, ",".join(map(str, cs)), ",".join(map(str, c["seeds"]["nle"])))
PY
)
NAME=$(basename "$CFG" .yaml)
sub() { sbatch --parsable "$@"; }
exc() { [[ -n "$1" ]] && echo "--exclude=$1" || true; }

for cs in ${CSEEDS//,/ }; do
  dep=()
  case $TYPE in
    mlp)
      j=$(sub --job-name=s1_${NAME}_c$cs $(exc "$S1_EXCLUDE") slurm/stage1_mlp.sbatch "$CFG" $cs)
      dep=(--dependency=afterok:$j) ;;
    cnn)
      j=$(sub --job-name=s1_${NAME}_c$cs slurm/stage1_cnn_train.sbatch "$CFG" $cs)
      j=$(sub --job-name=s1e_${NAME}_c$cs --dependency=afterok:$j slurm/stage1_cnn_export.sbatch "$CFG" $cs)
      j=$(sub --job-name=s1m_${NAME}_c$cs --dependency=afterok:$j slurm/stage1_merge.sbatch "$CFG" $cs)
      dep=(--dependency=afterok:$j) ;;
  esac
  for ns in ${NSEEDS//,/ }; do
    j2=$(sub --job-name=s2_${NAME}_c${cs}_n$ns "${dep[@]}" $(exc "$S2_EXCLUDE") slurm/stage2.sbatch "$CFG" $cs $ns)
    j3=$(sub --job-name=s3_${NAME}_c${cs}_n$ns --dependency=afterok:$j2 $(exc "$S3_EXCLUDE") slurm/stage3.sbatch "$CFG" $cs $ns)
    echo "$NAME c$cs n$ns: s2 $j2 s3 $j3"
  done
done
