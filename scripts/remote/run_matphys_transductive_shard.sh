#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 GPU_INDEX SWEEP_ROOT CASE [CASE ...]" >&2
  exit 2
fi

gpu_index=$1
sweep_root=$2
shift 2

readonly python_bin=/home/florianpfaff/codex-runs/molmomotion-field-davis-timed-20260714/venv/bin/python
readonly control_root=/home/florianpfaff/matphys-transductive-bpt-v1/Bayesian-PhysTwin-control
readonly wrapper="$control_root/scripts/remote/run_matphys_transductive_reconstruction.py"
readonly expected_wrapper_sha256=7eff23cb2dc4f6ccdb5ca6a195dd2e49935716719e1c9523c6d2c7a7df9671ff
readonly matphys_root=/home/florianpfaff/matphys-transductive-bpt-v1/MatPhys
readonly data_root=/home/florianpfaff/matphys-loo-sota-v1/phystwin-data/extracted/data/different_types
readonly experiments_dir=/home/florianpfaff/matphys-loo-sota-v1/phystwin-sota-render-20260715/code/experiments
readonly experiments_optimization_dir=/home/florianpfaff/matphys-loo-sota-v1/phystwin-sota-render-20260715/code/experiments_optimization
readonly case_to_material="$matphys_root/semantic/case_to_material_different_types.json"
readonly results_dir="$matphys_root/results"
readonly sem_cache_dir="$matphys_root/semantic/cache"
readonly gate="$sweep_root/two_case_gate.json"
readonly python_path="$control_root/src:/home/florianpfaff/matphys-transductive-bpt-v1/compatdeps:/home/florianpfaff/matphys-loo-sota-v1/matphys-causal-sota-v1/pydeps"

actual_wrapper_sha256=$(sha256sum "$wrapper" | awk '{print $1}')
if [[ "$actual_wrapper_sha256" != "$expected_wrapper_sha256" ]]; then
  echo "wrapper hash mismatch: $actual_wrapper_sha256" >&2
  exit 1
fi

"$python_bin" - "$gate" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("passed") is not True or gate.get("decision") != "continue_full_22":
    raise SystemExit("the frozen two-case gate does not authorize continuation")
PY

mkdir -p "$sweep_root/logs"

for case_name in "$@"; do
  case_root="$sweep_root/cases/$case_name"
  training_dir="$case_root/training"
  export_dir="$case_root/export"
  result="$export_dir/transductive_reconstruction_result.json"
  log="$sweep_root/logs/$case_name.log"

  if [[ -f "$result" ]]; then
    echo "[$(date --iso-8601=seconds)] skip completed case $case_name"
    continue
  fi
  if [[ -e "$training_dir" || -e "$export_dir" ]]; then
    echo "refusing to overwrite partial outputs for $case_name" >&2
    exit 1
  fi

  mkdir -p "$case_root"
  echo "[$(date --iso-8601=seconds)] start $case_name on physical GPU $gpu_index" | tee "$log"

  CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH="$python_path" \
    "$python_bin" "$wrapper" train \
    --matphys-root "$matphys_root" \
    --data-root "$data_root" \
    --experiments-dir "$experiments_dir" \
    --experiments-optimization-dir "$experiments_optimization_dir" \
    --case-to-material "$case_to_material" \
    --results-dir "$results_dir" \
    --sem-cache-dir "$sem_cache_dir" \
    --case "$case_name" \
    --device cuda:0 \
    --output-dir "$training_dir" \
    --epochs 200 \
    --eval-every 10 \
    --acknowledge-future-observations \
    >>"$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH="$python_path" \
    "$python_bin" "$wrapper" export \
    --matphys-root "$matphys_root" \
    --data-root "$data_root" \
    --experiments-dir "$experiments_dir" \
    --experiments-optimization-dir "$experiments_optimization_dir" \
    --case-to-material "$case_to_material" \
    --results-dir "$results_dir" \
    --sem-cache-dir "$sem_cache_dir" \
    --case "$case_name" \
    --device cuda:0 \
    --checkpoint "$training_dir/best_checkpoint.pth" \
    --training-audit "$training_dir/transductive_training_audit.json" \
    --output-dir "$export_dir" \
    --acknowledge-future-observations \
    >>"$log" 2>&1

  echo "[$(date --iso-8601=seconds)] complete $case_name" | tee -a "$log"
done
