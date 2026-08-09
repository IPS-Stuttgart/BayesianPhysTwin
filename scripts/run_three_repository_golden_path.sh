#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_three_repository_golden_path.sh \
    <Bayesian-PhysTwin root> <Prob4D root> <Causal4D root>

Requires clean Git checkouts at exact commits. Builds one wheel from each
repository, installs only those wheels into a fresh virtual environment, copies
the integration tests outside every source tree, and runs them in Python
isolated mode.
EOF
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

absolute_path() {
  (
    cd "$1"
    pwd -P
  )
}

require_repository() {
  local candidate="$1"
  local label="$2"
  if [[ ! -f "${candidate}/pyproject.toml" ]]; then
    echo "${label} repository has no pyproject.toml: ${candidate}" >&2
    exit 2
  fi
  if ! git -C "${candidate}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "${label} is not a Git checkout: ${candidate}" >&2
    exit 2
  fi
}

repository_revision() {
  local candidate="$1"
  local label="$2"
  local revision
  revision="$(git -C "${candidate}" rev-parse --verify 'HEAD^{commit}')"
  if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "${label} HEAD is not an exact lowercase 40-character commit." >&2
    exit 2
  fi
  printf '%s' "${revision}"
}

require_clean_repository() {
  local candidate="$1"
  local label="$2"
  local status
  status="$(git -C "${candidate}" status --porcelain --untracked-files=normal)"
  if [[ -n "${status}" ]]; then
    echo "${label} checkout is dirty; evidence runs require clean sources." >&2
    printf '%s\n' "${status}" >&2
    exit 2
  fi
}

unique_wheel() {
  local wheelhouse="$1"
  local pattern="$2"
  local label="$3"
  local -a matches=()
  mapfile -t matches < <(
    find "${wheelhouse}" -maxdepth 1 -type f -name "${pattern}" -print
  )
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "Expected exactly one ${label} wheel, found ${#matches[@]}." >&2
    printf '%s\n' "${matches[@]}" >&2
    exit 1
  fi
  printf '%s' "${matches[0]}"
}

BPT_ROOT="$(absolute_path "$1")"
PROB4D_ROOT="$(absolute_path "$2")"
CAUSAL4D_ROOT="$(absolute_path "$3")"

require_repository "${BPT_ROOT}" "Bayesian-PhysTwin"
require_repository "${PROB4D_ROOT}" "Prob4D"
require_repository "${CAUSAL4D_ROOT}" "Causal4D"
require_clean_repository "${BPT_ROOT}" "Bayesian-PhysTwin"
require_clean_repository "${PROB4D_ROOT}" "Prob4D"
require_clean_repository "${CAUSAL4D_ROOT}" "Causal4D"

export BAYESIAN_PHYSTWIN_REVISION
export PROB4D_REVISION
export CAUSAL4D_REVISION
BAYESIAN_PHYSTWIN_REVISION="$(
  repository_revision "${BPT_ROOT}" "Bayesian-PhysTwin"
)"
PROB4D_REVISION="$(repository_revision "${PROB4D_ROOT}" "Prob4D")"
CAUSAL4D_REVISION="$(repository_revision "${CAUSAL4D_ROOT}" "Causal4D")"

WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/three-repository-golden-path.XXXXXX")"
cleanup() {
  rm -rf "${WORK_ROOT}"
}
trap cleanup EXIT

BUILD_VENV="${WORK_ROOT}/build-venv"
TEST_VENV="${WORK_ROOT}/test-venv"
WHEELHOUSE="${WORK_ROOT}/wheelhouse"
RUN_ROOT="${WORK_ROOT}/run"
SOURCE_ROOT="${WORK_ROOT}/sources"
BPT_BUILD_ROOT="${SOURCE_ROOT}/bayesian-phystwin"
PROB4D_BUILD_ROOT="${SOURCE_ROOT}/prob4d"
CAUSAL4D_BUILD_ROOT="${SOURCE_ROOT}/causal4d"
mkdir -p "${WHEELHOUSE}" "${RUN_ROOT}" "${SOURCE_ROOT}"

snapshot_repository() {
  local candidate="$1"
  local destination="$2"
  mkdir -p "${destination}"
  git -C "${candidate}" archive --format=tar HEAD | tar -xf - -C "${destination}"
}

snapshot_repository "${BPT_ROOT}" "${BPT_BUILD_ROOT}"
snapshot_repository "${PROB4D_ROOT}" "${PROB4D_BUILD_ROOT}"
snapshot_repository "${CAUSAL4D_ROOT}" "${CAUSAL4D_BUILD_ROOT}"

python -m venv "${BUILD_VENV}"
"${BUILD_VENV}/bin/python" -m pip install --disable-pip-version-check \
  --upgrade pip build

for repository in \
  "${PROB4D_BUILD_ROOT}" \
  "${BPT_BUILD_ROOT}" \
  "${CAUSAL4D_BUILD_ROOT}"; do
  "${BUILD_VENV}/bin/python" -m build \
    --wheel \
    --outdir "${WHEELHOUSE}" \
    "${repository}"
done

wheel_count="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name '*.whl' | wc -l)"
if [[ "${wheel_count}" -ne 3 ]]; then
  echo "Expected exactly three built wheels, found ${wheel_count}." >&2
  find "${WHEELHOUSE}" -maxdepth 1 -type f -print >&2
  exit 1
fi

BPT_WHEEL="$(
  unique_wheel "${WHEELHOUSE}" 'bayesian_phystwin-*.whl' 'Bayesian-PhysTwin'
)"
PROB4D_WHEEL="$(unique_wheel "${WHEELHOUSE}" 'prob4d-*.whl' 'Prob4D')"
CAUSAL4D_WHEEL="$(unique_wheel "${WHEELHOUSE}" 'causal4d-*.whl' 'Causal4D')"

sha256sum "${BPT_WHEEL}" "${PROB4D_WHEEL}" "${CAUSAL4D_WHEEL}"
export BAYESIAN_PHYSTWIN_WHEEL_SHA256
export PROB4D_WHEEL_SHA256
export CAUSAL4D_WHEEL_SHA256
BAYESIAN_PHYSTWIN_WHEEL_SHA256="$(sha256sum "${BPT_WHEEL}" | cut -d' ' -f1)"
PROB4D_WHEEL_SHA256="$(sha256sum "${PROB4D_WHEEL}" | cut -d' ' -f1)"
CAUSAL4D_WHEEL_SHA256="$(sha256sum "${CAUSAL4D_WHEEL}" | cut -d' ' -f1)"

python -m venv "${TEST_VENV}"
"${TEST_VENV}/bin/python" -m pip install --disable-pip-version-check \
  --upgrade pip
"${TEST_VENV}/bin/python" -m pip install --disable-pip-version-check \
  pytest "${PROB4D_WHEEL}" "${BPT_WHEEL}" "${CAUSAL4D_WHEEL}"
"${TEST_VENV}/bin/python" -m pip check

env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "${TEST_VENV}/bin/python" -I -c 'from importlib import import_module; expected="a62c693a14c227daa1f4c8db850e691a1d0081df0c853cf0174c33d0b8504ce9"; names=("prob4d.observation_contract_bundle","bayesian_phystwin.observation_contract_bundle","causal4d.observation_contract_bundle"); observed={name:import_module(name).observation_contract_bundle_manifest()["bundle_sha256"] for name in names}; assert set(observed.values())=={expected}, observed; print(f"verified shared observation-contract bundle {expected}")'

shopt -s nullglob
integration_tests=(
  "${BPT_BUILD_ROOT}"/integration_tests/test_three_repository_*.py
)
if (( ${#integration_tests[@]} == 0 )); then
  echo "No three-repository integration tests were found." >&2
  exit 1
fi
cp "${integration_tests[@]}" "${RUN_ROOT}/"

export THREE_REPO_SOURCE_ROOTS="$({
  printf '%s' "${BPT_ROOT}"
  printf ':'
  printf '%s' "${PROB4D_ROOT}"
  printf ':'
  printf '%s' "${CAUSAL4D_ROOT}"
})"

cd "${RUN_ROOT}"
env -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  "${TEST_VENV}/bin/python" -I -m pytest \
  -q \
  test_three_repository_*.py
