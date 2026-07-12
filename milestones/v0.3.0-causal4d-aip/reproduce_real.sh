#!/usr/bin/env bash
set -euo pipefail

repo_root="${CAUSAL4D_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
archive_root="${CAUSAL4D_MILESTONE_VAULT:-/mnt/corsair/florianpfaff/research-milestones/v0.3.0-causal4d-aip}"
output="${1:-${repo_root}/runs/reproduce-v0.3.0-causal4d-aip/real}"

require_executable() {
  if [[ ! -x "$1" ]]; then
    printf 'Required executable is missing: %s\n' "$1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Required file is missing: %s\n' "$1" >&2
    exit 1
  fi
}

require_executable "${repo_root}/scripts/remote/run_causal4d_phystwin_molmo_pilot.sh"
require_executable "${repo_root}/scripts/remote/run_causal4d_abduction_pipeline.sh"
require_executable "${repo_root}/scripts/remote/run_causal4d_real_oracle_audit.sh"
require_file "${archive_root}/checkpoint/MolmoMotion-4B-H3-F30/config.yaml"
require_file "${archive_root}/checkpoint/MolmoMotion-4B-H3-F30/model.pt"

if [[ -d "${output}" ]]; then
  if [[ -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Refusing to overwrite nonempty output: %s\n' "${output}" >&2
    exit 1
  fi
fi
mkdir -p "${output}"

export CAUSAL4D_REPO_ROOT="${repo_root}"
export CAUSAL4D_CASE=single_lift_sloth
export CAUSAL4D_PARAMETER_PARTICLES=4
export CAUSAL4D_O_PLUS_PREFIX_FRAMES=6
export PHYSTWIN_GPU="${PHYSTWIN_GPU:-0}"
export MOLMO_GPU="${MOLMO_GPU:-1}"
export PYTHONHASHSEED=0

export MOLMO_CHECKPOINT="${archive_root}/checkpoint/MolmoMotion-4B-H3-F30"
export CAUSAL4D_CONTACT_STATES=12
export CAUSAL4D_OUTPUT_ROOT="${output}/molmo-pilot"
"${repo_root}/scripts/remote/run_causal4d_phystwin_molmo_pilot.sh"

export CAUSAL4D_CONTACT_STATES=9
export MOLMO_FORECAST="${output}/molmo-pilot/molmo.npz"
export CAUSAL4D_OUTPUT_ROOT="${output}/aip"
"${repo_root}/scripts/remote/run_causal4d_abduction_pipeline.sh"

export CAUSAL4D_AUDIT_INPUT_ROOT="${output}/aip"
export CAUSAL4D_AUDIT_OUTPUT_ROOT="${output}/aip/oracle-audit"
export CAUSAL4D_EXPANDED_CONTACT_STATES=108
"${repo_root}/scripts/remote/run_causal4d_real_oracle_audit.sh"

printf 'Real single_lift_sloth milestone reproduced: %s\n' "${output}"
