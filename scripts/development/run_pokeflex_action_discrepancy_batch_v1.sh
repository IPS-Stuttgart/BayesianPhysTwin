#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 CODE_ROOT DATA_ROOT OUTPUT_ROOT ROLE TAKE [TAKE ...]" >&2
  exit 2
fi

code_root=$1
data_root=$2
output_root=$3
role=$4
shift 4

case "$role" in
  development)
    calibration_flag=()
    ;;
  opened_calibration)
    calibration_flag=(--acknowledge-opened-calibration)
    ;;
  *)
    echo "ROLE must be development or opened_calibration" >&2
    exit 2
    ;;
esac

python_bin=${BPT_POKEFLEX_PYTHON:-/home/florianpfaff/.venvs/bpt-gpu/bin/python}
protocol=${BPT_POKEFLEX_PROTOCOL:-configs/sota/pokeflex_bayesian_registration_v1.json}
upstream=${BPT_POKEFLEX_UPSTREAM:-/mnt/corsair/florianpfaff/pokeflex-reconstruction-official-aaa8726}
checkpoint=${BPT_POKEFLEX_CHECKPOINT:-/mnt/corsair/florianpfaff/pokeflex-sota-audit/pretrained_models/kinect_pointclouds}

mkdir -p "$output_root/logs"
for take in "$@"; do
  echo "START $take"
  take_root="$data_root/$take"
  if [[ "$role" == "opened_calibration" ]]; then
    take_root="$take_root/$take"
  fi
  (
    cd "$code_root"
    env \
      CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      PYTHONPATH=src \
      nice -n 10 "$python_bin" \
      scripts/development/extract_pokeflex_action_discrepancy_v1.py \
      --take-root "$take_root" \
      --protocol "$protocol" \
      --upstream-checkout "$upstream" \
      --checkpoint-root "$checkpoint" \
      --output-json "$output_root/$take.json" \
      --output-npz "$output_root/$take.npz" \
      --device cpu \
      --acknowledge-opened-outcome \
      "${calibration_flag[@]}"
  ) >"$output_root/logs/$take.log" 2>&1
  echo "DONE $take"
done
