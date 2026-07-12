#!/usr/bin/env bash
set -euo pipefail

repo_root="${CAUSAL4D_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
case_name="${CAUSAL4D_CASE:-single_lift_sloth}"
data_root="${BPT_DATA_ROOT:-/home/florianpfaff/bayesian-phystwin-data}"
official_repo="${PHYSTWIN_REPO:-/home/florianpfaff/PhysTwin-upstream}"
profile_root="${BPT_PROFILE_ROOT:-/home/florianpfaff/Bayesian-PhysTwin/runs}"
profile_dir="${CAUSAL4D_PROFILE_DIR:-${profile_root}/phystwin_${case_name}/profile_full_v1_cue_9x9}"
python="${BPT_PYTHON:-/home/florianpfaff/.venvs/bpt-gpu/bin/python}"
input_root="${CAUSAL4D_AUDIT_INPUT_ROOT:-${repo_root}/runs/causal4d-abduction-v1/${case_name}}"
output="${CAUSAL4D_AUDIT_OUTPUT_ROOT:-${input_root}/oracle-audit-v1}"
particle_count="${CAUSAL4D_PARAMETER_PARTICLES:-4}"
expanded_contact_count="${CAUSAL4D_EXPANDED_CONTACT_STATES:-108}"

case_dir="${data_root}/${case_name}"
profile="${profile_dir}/parameter_profile.npz"
checkpoint="${profile_dir}/refit_checkpoint.pt"
belief="${CAUSAL4D_TWIN_BELIEF:-${input_root}/known.twin_belief.npz}"
current_bank="${CAUSAL4D_CURRENT_BANK:-${input_root}/known.bank.npz}"
physical="${CAUSAL4D_PHYSICAL_POSTERIOR:-${input_root}/known_action.physical.npz}"
expanded_bank="${output}/known.expanded108.bank.npz"
mkdir -p "${output}"

for required in "${belief}" "${current_bank}" "${physical}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Missing prerequisite artifact: %s\n' "${required}" >&2
    exit 1
  fi
done

run() {
  PYTHONPATH="${repo_root}/src" CUDA_VISIBLE_DEVICES="${PHYSTWIN_GPU:-0}" \
    "${python}" "$@"
}

run -m causal4d.cli.phystwin_rollout_bank \
  "${official_repo}" "${case_dir}" "${profile}" "${checkpoint}" \
  "${expanded_bank}" \
  --action-setting known \
  --parameter-particles "${particle_count}" \
  --maximum-contact-states "${expanded_contact_count}" \
  --twin-belief "${belief}"

run -m causal4d.cli.audit_real_oracle_gap \
  "${current_bank}" \
  "${expanded_bank}" \
  "${belief}" \
  "${physical}" \
  "${case_dir}/final_data.pkl" \
  "${case_dir}/inference.pkl" \
  "${output}/real_oracle_audit.json" \
  "${output}/real_oracle_components.csv"

printf 'Causal4D real oracle audit: %s\n' "${output}"
