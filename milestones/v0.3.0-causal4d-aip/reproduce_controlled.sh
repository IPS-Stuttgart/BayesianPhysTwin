#!/usr/bin/env bash
set -euo pipefail

repo_root="${CAUSAL4D_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
milestone_root="${repo_root}/milestones/v0.3.0-causal4d-aip"
python="${BPT_PYTHON:-/home/florianpfaff/.venvs/bpt-gpu/bin/python}"
output="${1:-${repo_root}/runs/reproduce-v0.3.0-causal4d-aip/controlled}"

if [[ -d "${output}" ]]; then
  if [[ -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Refusing to overwrite nonempty output: %s\n' "${output}" >&2
    exit 1
  fi
fi
mkdir -p "${output}/counterfactual" "${output}/latent-contact"

run() {
  PYTHONHASHSEED=0 PYTHONPATH="${repo_root}/src" "${python}" "$@"
}

run -m causal4d.cli.counterfactual_benchmark \
  --output-dir "${output}/counterfactual" \
  --seeds 0:5 \
  --frames 56 \
  --training-repeats 2 \
  --parameter-grid-count 5 \
  --observation-noise-mm 1.5 \
  --inference-noise-mm 6.0 \
  --likelihood-power 0.12 \
  --world-control-rotation-deg 8.0 \
  --world-nonlinearity 0.18

run -m causal4d.cli.latent_contact_benchmark \
  --output-dir "${output}/latent-contact" \
  --seeds 0:5 \
  --frames 56 \
  --training-repeats 2 \
  --parameter-grid-count 5 \
  --contact-parameter-particles 12 \
  --observation-fraction 0.20 \
  --observation-noise-mm 1.5 \
  --require-gates

run "${repo_root}/scripts/release/verify_result_bundle.py" \
  "${milestone_root}/results/controlled-counterfactual/manifest.json" \
  "${output}/counterfactual" \
  --numeric-atol 1e-12
run "${repo_root}/scripts/release/verify_result_bundle.py" \
  "${milestone_root}/results/controlled/manifest.json" \
  "${output}/latent-contact" \
  --numeric-atol 1e-12

printf 'Controlled milestone reproduced within 1e-12 numeric tolerance: %s\n' "${output}"
