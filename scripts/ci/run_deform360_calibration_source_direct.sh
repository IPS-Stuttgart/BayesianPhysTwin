#!/usr/bin/env bash
set -euo pipefail

: "${BPT_SOURCE_SHA:?}"
: "${PROCESSING_REVISION:?}"
: "${PROCESSING_REPO:?}"
: "${DATA_ROOT:?}"
: "${PROCESSED_ROOT:?}"
: "${EVIDENCE_ROOT:?}"
: "${PYTHON_SITE:?}"
: "${STAGED_ROOT:?}"

verify_confirmation_boundary() {
  local python_bin="${BASE_PYTHON:-$(command -v python3 || true)}"
  [[ -n "${python_bin}" ]] || return 1
  "${python_bin}" - <<'PY'
import json
import os
from pathlib import Path

selection = json.loads(
    Path(
        "protocols/locks/"
        "deform360_official_hub_visuotactile_v1_selection.json"
    ).read_text(encoding="utf-8")
)
confirmation = {
    row["object_id"] for row in selection["selection"]["confirmation"]
}
raw = Path(os.environ["DATA_ROOT"]) / "raw"
present = (
    {path.name for path in raw.iterdir() if path.is_dir()}
    if raw.exists()
    else set()
)
overlap = sorted(present & confirmation)
if overlap:
    raise SystemExit(f"confirmation payloads appeared: {overlap}")
evidence = Path(os.environ["EVIDENCE_ROOT"])
evidence.mkdir(parents=True, exist_ok=True)
(evidence / "confirmation-boundary.txt").write_text(
    "confirmation_payloads_opened=false\n",
    encoding="utf-8",
)
print("confirmation_payloads_opened=false")
PY
}

finalize() {
  local status=$?
  trap - EXIT
  set +e
  verify_confirmation_boundary
  local boundary_status=$?
  set -e
  if [[ ${boundary_status} -ne 0 ]]; then
    status=${boundary_status}
  fi
  exit "${status}"
}
trap finalize EXIT

mkdir -p -- "${DATA_ROOT}" "${PROCESSED_ROOT}" "${EVIDENCE_ROOT}" "${PYTHON_SITE}"
printf '/_deform360_processing/\n' >> .git/info/exclude

base_python=""
for candidate in \
  /usr/bin/python3.12 \
  /usr/local/bin/python3.12 \
  /usr/bin/python3 \
  /usr/local/bin/python3
do
  if [[ -x "${candidate}" ]]; then
    base_python="${candidate}"
    break
  fi
done
if [[ -z "${base_python}" ]]; then
  base_python="$(command -v python3 || true)"
fi
[[ -n "${base_python}" ]] || { echo "No system Python is available." >&2; exit 1; }
export BASE_PYTHON="${base_python}"
export PYTHONPATH="${GITHUB_WORKSPACE}/src:${PROCESSING_REPO}:${PYTHON_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "runner_name=${RUNNER_NAME}"
  echo "runner_os=${RUNNER_OS}"
  echo "runner_arch=${RUNNER_ARCH}"
  echo "repository=${GITHUB_REPOSITORY}"
  echo "event_name=${GITHUB_EVENT_NAME}"
  echo "event_revision=${GITHUB_SHA}"
  echo "authorized_source_revision=${BPT_SOURCE_SHA}"
  echo "checked_out_revision=$(git rev-parse HEAD)"
  echo "processing_revision=$(git -C "${PROCESSING_REPO}" rev-parse HEAD)"
  "${BASE_PYTHON}" --version
  "${BASE_PYTHON}" -m pip --version
} | tee "${EVIDENCE_ROOT}/runner.txt"
test "$(git rev-parse HEAD)" = "${BPT_SOURCE_SHA}"
test "$(git -C "${PROCESSING_REPO}" rev-parse HEAD)" = "${PROCESSING_REVISION}"
test -z "$(git -C "${PROCESSING_REPO}" status --porcelain)"
nvidia-smi -L | tee "${EVIDENCE_ROOT}/nvidia-smi-L.txt"
nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader \
  | tee "${EVIDENCE_ROOT}/nvidia-smi.csv"

"${BASE_PYTHON}" -m pip install \
  --break-system-packages \
  --no-cache-dir \
  --target "${PYTHON_SITE}" \
  ".[dev,graph]" \
  build \
  scipy \
  matplotlib \
  "huggingface_hub>=0.24" \
  "${PROCESSING_REPO}[all]"
"${BASE_PYTHON}" -m pip check
"${BASE_PYTHON}" -m pip freeze --path "${PYTHON_SITE}" \
  > "${EVIDENCE_ROOT}/pip-freeze.txt"

files=(
  scripts/science/deform360_calibration_source/__init__.py
  scripts/science/deform360_calibration_source/contracts.py
  scripts/science/deform360_calibration_source/planning.py
  scripts/science/deform360_calibration_source/download.py
  scripts/science/deform360_calibration_source/prepare.py
  scripts/science/deform360_calibration_source/cli.py
  scripts/science/run_deform360_official_hub_calibration_source.py
  tests/test_deform360_official_hub_calibration_source.py
  tests/test_deform360_calibration_source_workflow.py
)
"${BASE_PYTHON}" -m ruff check "${files[@]}"
"${BASE_PYTHON}" -m ruff format --check "${files[@]}"
bash -n scripts/ci/run_deform360_calibration_source_direct.sh
"${BASE_PYTHON}" -m pytest -q -p no:cacheprovider \
  tests/test_deform360_official_hub_calibration_source.py \
  tests/test_deform360_calibration_source_workflow.py \
  tests/test_deform360_calibration_execution.py \
  tests/test_deform360_visual_provider_freeze.py \
  tests/test_pull_request_workflow_integrity.py \
  | tee "${EVIDENCE_ROOT}/pytest.txt"
"${BASE_PYTHON}" -m compileall -q \
  scripts/science/deform360_calibration_source \
  scripts/science/run_deform360_official_hub_calibration_source.py
git diff --exit-code
test -z "$(git status --porcelain --untracked-files=all)"

set +e
"${BASE_PYTHON}" \
  scripts/science/run_deform360_official_hub_calibration_source.py plan \
  --protocol protocols/deform360_official_hub_calibration_source_v1.json \
  --selection-lock protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json \
  --output "${EVIDENCE_ROOT}/calibration-source-plan.json" \
  | tee "${EVIDENCE_ROOT}/plan-console.json"
plan_status=${PIPESTATUS[0]}
set -e
if [[ ${plan_status} -eq 3 ]]; then
  echo "The names-only calibration support gate did not pass." >&2
  exit 3
elif [[ ${plan_status} -ne 0 ]]; then
  exit "${plan_status}"
fi

"${BASE_PYTHON}" \
  scripts/science/run_deform360_official_hub_calibration_source.py download \
  --protocol protocols/deform360_official_hub_calibration_source_v1.json \
  --selection-lock protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json \
  --plan "${EVIDENCE_ROOT}/calibration-source-plan.json" \
  --data-root "${DATA_ROOT}" \
  --output "${EVIDENCE_ROOT}/calibration-download-manifest.json" \
  --workers 8 \
  | tee "${EVIDENCE_ROOT}/download-console.json"

set +e
"${BASE_PYTHON}" \
  scripts/science/run_deform360_official_hub_calibration_source.py prepare \
  --protocol protocols/deform360_official_hub_calibration_source_v1.json \
  --selection-lock protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json \
  --plan "${EVIDENCE_ROOT}/calibration-source-plan.json" \
  --download-manifest "${EVIDENCE_ROOT}/calibration-download-manifest.json" \
  --data-root "${DATA_ROOT}" \
  --staged-raw-root "${STAGED_ROOT}" \
  --processed-root "${PROCESSED_ROOT}/aligned" \
  --processing-repository "${PROCESSING_REPO}" \
  --output "${EVIDENCE_ROOT}/calibration-source-result.json" \
  | tee "${EVIDENCE_ROOT}/prepare-console.json"
prepare_status=${PIPESTATUS[0]}
set -e
if [[ ${prepare_status} -eq 3 ]]; then
  echo "The calibration-source support gate did not pass." >&2
  exit 3
elif [[ ${prepare_status} -ne 0 ]]; then
  exit "${prepare_status}"
fi
