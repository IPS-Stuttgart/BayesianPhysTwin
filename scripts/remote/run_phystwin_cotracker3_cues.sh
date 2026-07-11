#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CASE [CASE ...]" >&2
  exit 2
fi

repo_root="${BPT_REPO_ROOT:-/home/florianpfaff/Bayesian-PhysTwin}"
data_root="${BPT_DATA_ROOT:-/home/florianpfaff/bayesian-phystwin-confirmatory-data}"
raw_root="${BPT_RAW_ROOT:-/mnt/corsair/florianpfaff/phystwin-data/extracted/data/different_types}"
output_root="${BPT_CUE_OUTPUT_ROOT:-${repo_root}/runs/phystwin_cotracker3_cues_v1}"
cotracker_root="${COTRACKER_ROOT:-/home/florianpfaff/co-tracker}"
checkpoint="${COTRACKER_CHECKPOINT:-${cotracker_root}/checkpoints/scaled_online.pth}"
python_bin="${BPT_PYTHON:-/home/florianpfaff/.venvs/bpt-gpu/bin/python}"

cd "${repo_root}"
for case_name in "$@"; do
  case_dir="${data_root}/${case_name}"
  output_dir="${output_root}/${case_name}"
  train_end="$(${python_bin} -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["train"][1])' \
    "${case_dir}/split.json")"
  if [[ -s "${output_dir}/summary.json" && -s "${output_dir}/cues.npz" ]]; then
    echo "skip ${case_name}: complete output exists"
    continue
  fi
  echo "extract ${case_name}: training frames [0, ${train_end})"
  mkdir -p "${output_dir}"
  PYTHONPATH="${repo_root}/src:${cotracker_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -m bayesian_phystwin.cli.phystwin_cotracker3_cues \
      "${case_dir}/final_data.pkl" \
      "${raw_root}/${case_name}" \
      "${checkpoint}" \
      "${cotracker_root}" \
      "${output_dir}/cues.npz" \
      --train-end-frame "${train_end}" \
      --summary-json "${output_dir}/summary.json"
done
