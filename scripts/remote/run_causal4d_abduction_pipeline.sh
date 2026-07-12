#!/usr/bin/env bash
set -euo pipefail

repo_root="${CAUSAL4D_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
case_name="${CAUSAL4D_CASE:-single_lift_sloth}"
data_root="${BPT_DATA_ROOT:-/home/florianpfaff/bayesian-phystwin-data}"
official_repo="${PHYSTWIN_REPO:-/home/florianpfaff/PhysTwin-upstream}"
profile_root="${BPT_PROFILE_ROOT:-/home/florianpfaff/Bayesian-PhysTwin/runs}"
profile_dir="${CAUSAL4D_PROFILE_DIR:-${profile_root}/phystwin_${case_name}/profile_full_v1_cue_9x9}"
python="${BPT_PYTHON:-/home/florianpfaff/.venvs/bpt-gpu/bin/python}"
output="${CAUSAL4D_OUTPUT_ROOT:-${repo_root}/runs/causal4d-abduction-v1/${case_name}}"
particle_count="${CAUSAL4D_PARAMETER_PARTICLES:-4}"
contact_count="${CAUSAL4D_CONTACT_STATES:-9}"
prefix_frames="${CAUSAL4D_O_PLUS_PREFIX_FRAMES:-6}"

case_dir="${data_root}/${case_name}"
profile="${profile_dir}/parameter_profile.npz"
checkpoint="${profile_dir}/refit_checkpoint.pt"
mkdir -p "${output}"

run() {
  PYTHONPATH="${repo_root}/src" CUDA_VISIBLE_DEVICES="${PHYSTWIN_GPU:-0}" \
    "${python}" "$@"
}

run -m causal4d.cli.export_bpt_belief \
  "${official_repo}" "${case_dir}" "${profile}" "${checkpoint}" \
  "${output}/known.twin_belief.npz" \
  --parameter-particles "${particle_count}"

run -m causal4d.cli.phystwin_rollout_bank \
  "${official_repo}" "${case_dir}" "${profile}" "${checkpoint}" \
  "${output}/known.bank.npz" \
  --action-setting known \
  --parameter-particles "${particle_count}" \
  --maximum-contact-states "${contact_count}" \
  --twin-belief "${output}/known.twin_belief.npz"

run -m causal4d.cli.abduct_phystwin_intervention \
  "${output}/known.bank.npz" \
  "${output}/known.twin_belief.npz" \
  "${case_dir}/final_data.pkl" \
  "${output}/factual.npz" \
  "${output}/factual_evaluation.json" \
  --o-plus-prefix-frames "${prefix_frames}"

for specification in "known_action:same_grasp" "history_reverse:new_contact"; do
  IFS=: read -r action contact_policy <<<"${specification}"
  run -m causal4d.cli.counterfactual_phystwin \
    "${official_repo}" "${case_dir}" "${profile}" "${checkpoint}" \
    "${output}/known.twin_belief.npz" \
    "${output}/factual.npz" \
    "${output}/${action}.physical.npz" \
    --counterfactual-action-id "${action}" \
    --contact-policy "${contact_policy}" \
    --maximum-contact-states "${contact_count}"
done

run -m causal4d.cli.evaluate_physical_counterfactual \
  "${output}/known_action.physical.npz" \
  "${case_dir}/final_data.pkl" \
  "${output}/known_action.beta0_evaluation.json" \
  --start-frame "$((prefix_frames + 1))"

if [[ -n "${MOLMO_FORECAST:-}" ]]; then
  run -m causal4d.cli.molmo_task_posterior \
    "${output}/history_reverse.physical.npz" \
    "${MOLMO_FORECAST}" \
    "${MOLMO_FORECAST_ID:-instruction}" \
    "${output}/history_reverse.task_beta0.npz" \
    --beta 0
fi

printf 'Causal4D artifacts: %s\n' "${output}"
