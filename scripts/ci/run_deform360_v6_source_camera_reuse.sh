#!/usr/bin/env bash
set -euo pipefail

required=(
  AMENDMENT_PATH
  BASE_ARTIFACT_DIGEST_SHA256
  BASE_ARTIFACT_ID
  BASE_ARTIFACT_NAME
  BASE_ARTIFACT_ROOT
  BASE_HEAD_SHA
  BASE_RUN_ATTEMPT
  BASE_RUN_ID
  BPT_SOURCE_SHA
  EXECUTION_LOCK_PATH
  GITHUB_RUN_ATTEMPT
  GITHUB_RUN_ID
  METRIC_BATCH_RESULT_PATH
  METRIC_BATCH_ROOT
  METRIC_PREFIX_PLAN_PATH
  RESULTS_ROOT
  RUNNER_NAME
  RUN_CLAIM
  RUN_ROOT
  VISUAL_PRODUCTION_RESULT_PATH
  VISUAL_PRODUCTION_ROOT
)
for name in "${required[@]}"; do
  test -n "${!name:-}" || {
    echo "missing required environment variable: ${name}" >&2
    exit 2
  }
done

ordinary_file() {
  test -f "$1"
  test ! -L "$1"
}

copy_json() {
  local source="$1"
  local destination="$2"
  ordinary_file "${source}"
  install -m 600 "${source}" "${destination}"
}

umask 077
test "${RUNNER_NAME}" = "workstation2"
test ! -e "${RUN_ROOT}"
test ! -e "${RUN_CLAIM}"
mkdir -p "$(dirname "${RUN_ROOT}")"
python - "${RUN_CLAIM}" "${GITHUB_RUN_ID}" "${GITHUB_RUN_ATTEMPT}" \
  "${BPT_SOURCE_SHA}" <<'PY'
from __future__ import annotations

import os
import sys

path, run_id, run_attempt, revision = sys.argv[1:]
payload = (
    f"workflow_run_id={run_id}\n"
    f"workflow_run_attempt={run_attempt}\n"
    f"source_revision={revision}\n"
).encode("ascii")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
descriptor = os.open(path, flags, 0o600)
try:
    with os.fdopen(descriptor, "wb", closefd=False) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
finally:
    os.close(descriptor)
PY
mkdir "${RUN_ROOT}"
compact="${RUN_ROOT}/compact"
panel="${RUN_ROOT}/prediction-panel"
logs="${compact}/logs"
mkdir "${compact}" "${logs}"
ordinary_file "${RUN_CLAIM}"
install -m 600 "${RUN_CLAIM}" "${compact}/execution-started.txt"

base_plan="${BASE_ARTIFACT_ROOT}/source-plan.json"
base_batch="${BASE_ARTIFACT_ROOT}/source-prediction-batch.json"
base_receipt="${BASE_ARTIFACT_ROOT}/source-prediction-receipt.json"
base_execution_receipt="${BASE_ARTIFACT_ROOT}/execution-receipt.json"
for path in \
  "${base_plan}" \
  "${base_batch}" \
  "${base_receipt}" \
  "${base_execution_receipt}"; do
  ordinary_file "${path}"
done
test -z "$(find "${BASE_ARTIFACT_ROOT}" -type l -print -quit)"

copy_json "${AMENDMENT_PATH}" "${compact}/amendment.json"
copy_json "${EXECUTION_LOCK_PATH}" "${compact}/execution-lock.json"
copy_json "${base_plan}" "${compact}/base-source-plan.json"
copy_json "${base_batch}" "${compact}/base-prediction-batch.json"
copy_json "${base_receipt}" "${compact}/base-prediction-receipt.json"
copy_json \
  "${base_execution_receipt}" \
  "${compact}/base-execution-receipt.json"
copy_json "${METRIC_PREFIX_PLAN_PATH}" "${compact}/metric-prefix-plan.json"
copy_json "${METRIC_BATCH_RESULT_PATH}" "${compact}/metric-batch-result.json"
copy_json \
  "${VISUAL_PRODUCTION_RESULT_PATH}" \
  "${compact}/visual-production-result.json"

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  audit-base \
  --execution-lock "${compact}/execution-lock.json" \
  --base-source-plan "${compact}/base-source-plan.json" \
  --input-root "${RESULTS_ROOT}" \
  --output "${compact}/base-camera-audit.json" \
  > "${logs}/audit-base.json"

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  rank-reuse \
  --execution-lock "${compact}/execution-lock.json" \
  --base-source-plan "${compact}/base-source-plan.json" \
  --base-camera-audit "${compact}/base-camera-audit.json" \
  --metric-prefix-plan "${compact}/metric-prefix-plan.json" \
  --metric-batch-result "${compact}/metric-batch-result.json" \
  --metric-files-root "${METRIC_BATCH_ROOT}/metrics" \
  --output "${compact}/camera-reuse-preflight.json" \
  > "${logs}/rank-reuse.json"

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  build-combined-plan \
  --execution-lock "${compact}/execution-lock.json" \
  --base-source-plan "${compact}/base-source-plan.json" \
  --base-camera-audit "${compact}/base-camera-audit.json" \
  --preflight "${compact}/camera-reuse-preflight.json" \
  --metric-prefix-plan "${compact}/metric-prefix-plan.json" \
  --results-root "${RESULTS_ROOT}" \
  --prediction-root "${VISUAL_PRODUCTION_ROOT}" \
  --metric-files-root "${METRIC_BATCH_ROOT}/metrics" \
  --implementation-revision "${BPT_SOURCE_SHA}" \
  --output-plan "${compact}/combined-camera-audit-plan.json" \
  --output-receipt "${compact}/camera-reuse-receipt.json" \
  > "${logs}/build-combined-plan.json"

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  audit-combined \
  --execution-lock "${compact}/execution-lock.json" \
  --combined-plan "${compact}/combined-camera-audit-plan.json" \
  --input-root "${RESULTS_ROOT}" \
  --output "${compact}/final-camera-audit.json" \
  > "${logs}/audit-combined.json"

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  build-lineage \
  --execution-lock "${compact}/execution-lock.json" \
  --base-source-plan "${compact}/base-source-plan.json" \
  --amendment "${compact}/amendment.json" \
  --base-prediction-batch "${compact}/base-prediction-batch.json" \
  --base-prediction-receipt "${compact}/base-prediction-receipt.json" \
  --base-camera-audit "${compact}/base-camera-audit.json" \
  --preflight "${compact}/camera-reuse-preflight.json" \
  --reuse-receipt "${compact}/camera-reuse-receipt.json" \
  --combined-plan "${compact}/combined-camera-audit-plan.json" \
  --final-camera-audit "${compact}/final-camera-audit.json" \
  --metric-prefix-plan "${compact}/metric-prefix-plan.json" \
  --metric-batch-result "${compact}/metric-batch-result.json" \
  --output "${compact}/camera-reuse-lineage.json" \
  > "${logs}/build-lineage.json"

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  freeze-source-plan \
  --execution-lock "${compact}/execution-lock.json" \
  --combined-plan "${compact}/combined-camera-audit-plan.json" \
  --final-camera-audit "${compact}/final-camera-audit.json" \
  --lineage "${compact}/camera-reuse-lineage.json" \
  --implementation-revision "${BPT_SOURCE_SHA}" \
  --output "${compact}/source-plan.json" \
  > "${logs}/freeze-source-plan.json"

python scripts/science/run_deform360_joint_sparse_source_predictions_v5_2.py \
  --execution-lock "${compact}/execution-lock.json" \
  --source-plan "${compact}/source-plan.json" \
  --input-root "${RESULTS_ROOT}" \
  --output-root "${panel}" \
  > "${logs}/publish-source-panel.json"

copy_json \
  "${panel}/source-prediction-batch.json" \
  "${compact}/source-prediction-batch.json"
copy_json \
  "${panel}/source-prediction-receipt.json" \
  "${compact}/source-prediction-receipt.json"
mkdir "${compact}/source-seals"
mapfile -t seals < <(find "${panel}/source-seals" -maxdepth 1 -type f -name '*.json' -print | sort)
test "${#seals[@]}" = "100"
for seal in "${seals[@]}"; do
  ordinary_file "${seal}"
  install -m 600 "${seal}" "${compact}/source-seals/$(basename "${seal}")"
done

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  seal-execution \
  --amendment "${compact}/amendment.json" \
  --execution-lock "${compact}/execution-lock.json" \
  --source-revision "${BPT_SOURCE_SHA}" \
  --runner-name "${RUNNER_NAME}" \
  --workflow-run-id "${GITHUB_RUN_ID}" \
  --workflow-run-attempt "${GITHUB_RUN_ATTEMPT}" \
  --base-run-id "${BASE_RUN_ID}" \
  --base-run-attempt "${BASE_RUN_ATTEMPT}" \
  --base-head-sha "${BASE_HEAD_SHA}" \
  --base-artifact-id "${BASE_ARTIFACT_ID}" \
  --base-artifact-name "${BASE_ARTIFACT_NAME}" \
  --base-artifact-digest-sha256 "${BASE_ARTIFACT_DIGEST_SHA256}" \
  --artifact-root "${compact}" \
  --source-plan "${compact}/source-plan.json" \
  --prediction-batch "${compact}/source-prediction-batch.json" \
  --prediction-receipt "${compact}/source-prediction-receipt.json" \
  --source-seal-root "${compact}/source-seals" \
  --output "${compact}/execution-receipt.json" \
  > "${logs}/seal-execution.json"

(
  cd "${compact}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum --check SHA256SUMS >/dev/null
)

python scripts/science/materialize_deform360_v6_source_camera_reuse.py \
  validate-execution \
  --receipt "${compact}/execution-receipt.json" \
  > "${RUN_ROOT}/validated-execution.json"

test "$(git rev-parse HEAD)" = "${BPT_SOURCE_SHA}"
test -z "$(git status --porcelain=v1)"
printf '%s\n' "${compact}"
