#!/usr/bin/env bash
set -euo pipefail

repo_root="${CAUSAL4D_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
case_name="${CAUSAL4D_CASE:-single_lift_sloth}"
data_root="${BPT_DATA_ROOT:-/home/florianpfaff/bayesian-phystwin-data}"
raw_root="${BPT_RAW_ROOT:-/mnt/corsair/florianpfaff/phystwin-data/extracted/data/different_types}"
official_repo="${PHYSTWIN_REPO:-/home/florianpfaff/PhysTwin-upstream}"
profile_root="${BPT_PROFILE_ROOT:-/home/florianpfaff/Bayesian-PhysTwin/runs}"
profile_dir="${CAUSAL4D_PROFILE_DIR:-${profile_root}/phystwin_${case_name}/profile_full_v1_cue_9x9}"
molmo_root="${MOLMO_ROOT:-/mnt/corsair/florianpfaff/molmomotion-field-20260711}"
molmo_checkout="${MOLMO_CHECKOUT:-${molmo_root}/allenai-molmo-motion}"
molmo_checkpoint="${MOLMO_CHECKPOINT:-${molmo_root}/assets/checkpoints/MolmoMotion-4B-H3-F30}"
bpt_python="${BPT_PYTHON:-/home/florianpfaff/.venvs/bpt-gpu/bin/python}"
molmo_python="${MOLMO_PYTHON:-${molmo_root}/venv/bin/python}"
output_root="${CAUSAL4D_OUTPUT_ROOT:-${repo_root}/runs/causal4d-phystwin-molmo-pilot}"

case_dir="${data_root}/${case_name}"
raw_case="${raw_root}/${case_name}"
train_end="$(${bpt_python} -c \
  "import json; print(json.load(open('${case_dir}/split.json'))['train'][1])")"
mkdir -p "${output_root}"

PYTHONPATH="${repo_root}/src:${molmo_checkout}/src" \
  CUDA_VISIBLE_DEVICES="${MOLMO_GPU:-1}" \
  "${molmo_python}" -m causal4d.cli.molmo_phystwin_forecast \
  "${case_dir}/final_data.pkl" \
  "${raw_case}" \
  "${molmo_checkpoint}" \
  "${output_root}/molmo.npz" \
  --train-end-frame "${train_end}" \
  --caption 'instruction=A person lifts the sloth upward with one hand.' \
  --caption 'shuffled=A person pushes the sloth sideways across the table with one hand.' \
  --caption 'generic=The sloth moves.'

for setting in known hidden ambiguous; do
  PYTHONPATH="${repo_root}/src" \
    CUDA_VISIBLE_DEVICES="${PHYSTWIN_GPU:-0}" \
    "${bpt_python}" -m causal4d.cli.phystwin_rollout_bank \
    "${official_repo}" \
    "${case_dir}" \
    "${profile_dir}/parameter_profile.npz" \
    "${profile_dir}/refit_checkpoint.pt" \
    "${output_root}/${setting}.npz" \
    --action-setting "${setting}" \
    --parameter-particles "${CAUSAL4D_PARAMETER_PARTICLES:-4}" \
    --maximum-contact-states "${CAUSAL4D_CONTACT_STATES:-12}"

  PYTHONPATH="${repo_root}/src" \
    "${bpt_python}" -m causal4d.cli.evaluate_phystwin_molmo \
    "${output_root}/${setting}.npz" \
    "${case_dir}/final_data.pkl" \
    "${output_root}/molmo.npz" \
    "${output_root}/${setting}_result.json"
done

