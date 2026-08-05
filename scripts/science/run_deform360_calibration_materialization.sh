#!/usr/bin/env bash
# Execute the exact locked Deform360 calibration-prefix materialization.
set -euo pipefail

: "${BPT_SOURCE_SHA:?BPT_SOURCE_SHA is required}"
: "${DEFORM360_PROCESSING_REVISION:?DEFORM360_PROCESSING_REVISION is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"

python_bin="$(command -v python3.12 || command -v python3)"
test -x "${python_bin}"
site="${RUNNER_TEMP}/deform360-calibration-site-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
evidence_root="${RUNNER_TEMP}/deform360-calibration-evidence-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
requested_root="${INPUT_DATA_ROOT:-${VAR_DATA_ROOT:-}}"
if [[ -z "${requested_root}" ]]; then
  requested_root=/mnt/lexar4tb/datasets/deform360_official_hub_visuotactile_v1
fi
data_root="$(realpath -m "${requested_root}")"
workers="${INPUT_WORKERS:-4}"
[[ "${workers}" =~ ^[1-9][0-9]*$ ]]

rm -rf "${site}" "${evidence_root}"
mkdir -p "${site}" "${evidence_root}" "${data_root}"
export PYTHONPATH="${site}:${GITHUB_WORKSPACE}/src"
export DEFORM360_DATA_ROOT="${data_root}"
export DEFORM360_EVIDENCE_ROOT="${evidence_root}"
{
  echo "PYTHON_BIN=${python_bin}"
  echo "PYTHON_SITE=${site}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "DEFORM360_DATA_ROOT=${data_root}"
  echo "DEFORM360_EVIDENCE_ROOT=${evidence_root}"
  echo "DEFORM360_WORKERS=${workers}"
} >> "${GITHUB_ENV}"
{
  echo "requested_data_root=${requested_root}"
  echo "resolved_data_root=${data_root}"
  echo "workers=${workers}"
} > "${evidence_root}/storage-selection.txt"

finalize() {
  local exit_code=$?
  set +e
  {
    echo "repository=${GITHUB_REPOSITORY:-unknown}"
    echo "bpt_revision=${BPT_SOURCE_SHA}"
    echo "processing_revision=${DEFORM360_PROCESSING_REVISION}"
    echo "runner_name=${RUNNER_NAME:-unknown}"
    echo "runner_os=${RUNNER_OS:-unknown}"
    echo "runner_arch=${RUNNER_ARCH:-unknown}"
    "${python_bin}" --version
    uname -a
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
      --format=csv,noheader
    du -sh "${data_root}"
    echo "exit_code=${exit_code}"
  } > "${evidence_root}/environment.txt" 2>&1
  (
    cd "${evidence_root}" || exit
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      > SHA256SUMS
  )
  exit "${exit_code}"
}
trap finalize EXIT

"${python_bin}" -m pip install \
  --break-system-packages \
  --target "${site}" \
  "huggingface_hub>=0.24" \
  "numpy>=1.23,<2.3" \
  "opencv-contrib-python>=4.5"

"${python_bin}" - <<'PY'
import cv2
import huggingface_hub
import numpy
import bayesian_phystwin

print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("bayesian_phystwin", bayesian_phystwin.__file__)
PY

"${python_bin}" scripts/science/materialize_deform360_calibration_payloads.py \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --protocol protocols/deform360_official_hub_visuotactile_v1.json \
  --visual-provider-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json \
  --repository-root "${GITHUB_WORKSPACE}" \
  --processing-checkout "${GITHUB_WORKSPACE}/_deform360" \
  --dataset-root "${data_root}" \
  --output "${evidence_root}/materialization.json" \
  --implementation-revision "${BPT_SOURCE_SHA}" \
  --workers "${workers}" \
  --open-calibration-payloads \
  2>&1 | tee "${evidence_root}/console.log"

"${python_bin}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DEFORM360_EVIDENCE_ROOT"])
manifest = json.loads((root / "materialization.json").read_text(encoding="utf-8"))
boundary = manifest["information_boundary"]
assert manifest["status"] in {"complete", "complete_with_technical_failures"}
assert boundary["calibration_payloads_opened"] is True
assert boundary["camera_media_opened"] is False
assert boundary["confirmation_payloads_opened"] is False
assert boundary["target_outcomes_used"] is False
assert len(manifest["calibration_object_ids"]) == 10
assert len(manifest["confirmation_object_ids"]) == 12
confirmation = tuple(
    f"raw/{object_id}/" for object_id in manifest["confirmation_object_ids"]
)
opened = [item["path"] for item in manifest["downloaded_files"]]
forbidden = [
    path for path in opened if any(path.startswith(prefix) for prefix in confirmation)
]
if forbidden:
    raise SystemExit(f"confirmation paths were opened: {forbidden}")
if any(path.endswith(".mp4") for path in opened):
    raise SystemExit("camera media was opened during prefix materialization")
print(json.dumps(boundary, indent=2, sort_keys=True))
PY

git diff --exit-code
test -z "$(git status --porcelain=v1 --untracked-files=no)"
