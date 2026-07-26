#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_three_repository_golden_path.sh \
    <Bayesian-PhysTwin root> <Prob4D root> <Causal4D root>

Builds one wheel from each repository, installs only those wheels into a fresh
virtual environment, copies the integration test outside every source tree,
and runs it with Python isolated mode.
EOF
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

require_repository() {
  local candidate="$1"
  local label="$2"
  if [[ ! -f "${candidate}/pyproject.toml" ]]; then
    echo "${label} repository has no pyproject.toml: ${candidate}" >&2
    exit 2
  fi
}

absolute_path() {
  (
    cd "$1"
    pwd -P
  )
}

BPT_ROOT="$(absolute_path "$1")"
PROB4D_ROOT="$(absolute_path "$2")"
CAUSAL4D_ROOT="$(absolute_path "$3")"

require_repository "${BPT_ROOT}" "Bayesian-PhysTwin"
require_repository "${PROB4D_ROOT}" "Prob4D"
require_repository "${CAUSAL4D_ROOT}" "Causal4D"

WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/three-repository-golden-path.XXXXXX")"
cleanup() {
  rm -rf "${WORK_ROOT}"
}
trap cleanup EXIT

BUILD_VENV="${WORK_ROOT}/build-venv"
TEST_VENV="${WORK_ROOT}/test-venv"
WHEELHOUSE="${WORK_ROOT}/wheelhouse"
RUN_ROOT="${WORK_ROOT}/run"
mkdir -p "${WHEELHOUSE}" "${RUN_ROOT}"

python -m venv "${BUILD_VENV}"
"${BUILD_VENV}/bin/python" -m pip install --disable-pip-version-check \
  --upgrade pip build

for repository in "${PROB4D_ROOT}" "${BPT_ROOT}" "${CAUSAL4D_ROOT}"; do
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

sha256sum "${WHEELHOUSE}"/*.whl

python -m venv "${TEST_VENV}"
"${TEST_VENV}/bin/python" -m pip install --disable-pip-version-check \
  --upgrade pip
"${TEST_VENV}/bin/python" -m pip install --disable-pip-version-check \
  pytest "${WHEELHOUSE}"/*.whl
"${TEST_VENV}/bin/python" -m pip check

cp \
  "${BPT_ROOT}/integration_tests/test_three_repository_golden_path.py" \
  "${RUN_ROOT}/test_three_repository_golden_path.py"

export THREE_REPO_SOURCE_ROOTS="$(
  "${TEST_VENV}/bin/python" - "${BPT_ROOT}" "${PROB4D_ROOT}" "${CAUSAL4D_ROOT}" <<'PY'
import os
import sys

print(os.pathsep.join(sys.argv[1:]))
PY
)"
export BAYESIAN_PHYSTWIN_REVISION="$(
  git -C "${BPT_ROOT}" rev-parse HEAD 2>/dev/null \
    || printf '%s' 'installed-wheel-golden-path'
)"
export PROB4D_REVISION="$(
  git -C "${PROB4D_ROOT}" rev-parse HEAD 2>/dev/null \
    || printf '%s' 'installed-wheel-golden-path'
)"
export CAUSAL4D_REVISION="$(
  git -C "${CAUSAL4D_ROOT}" rev-parse HEAD 2>/dev/null \
    || printf '%s' 'installed-wheel-golden-path'
)"

cd "${RUN_ROOT}"
env -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  "${TEST_VENV}/bin/python" -I -m pytest \
  -q \
  test_three_repository_golden_path.py
