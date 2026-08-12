#!/usr/bin/env bash
set -euo pipefail

required=(
  BPT_SOURCE_SHA
  CANDIDATE_AMENDMENT_PATH
  EXECUTION_LOCK_PATH
  GITHUB_RUN_ATTEMPT
  GITHUB_RUN_ID
  RESULTS_ROOT
  RUNNER_NAME
  RUN_CLAIM
  RUN_ROOT
  UPSTREAM_COMPACT_ROOT
  UPSTREAM_PREDICTION_PANEL_ROOT
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

copy_file() {
  ordinary_file "$1"
  install -m 600 "$1" "$2"
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
panel="${RUN_ROOT}/candidate-panel"
logs="${compact}/logs"
mkdir "${compact}" "${logs}"
ordinary_file "${RUN_CLAIM}"
install -m 600 "${RUN_CLAIM}" "${compact}/execution-started.txt"

source_plan="${UPSTREAM_COMPACT_ROOT}/source-plan.json"
source_batch="${UPSTREAM_COMPACT_ROOT}/source-prediction-batch.json"
source_receipt="${UPSTREAM_COMPACT_ROOT}/source-prediction-receipt.json"
source_execution="${UPSTREAM_COMPACT_ROOT}/execution-receipt.json"
source_seals="${UPSTREAM_COMPACT_ROOT}/source-seals"
for path in \
  "${source_plan}" \
  "${source_batch}" \
  "${source_receipt}" \
  "${source_execution}"; do
  ordinary_file "${path}"
done
test -d "${source_seals}"
test ! -L "${source_seals}"
test -z "$(find "${UPSTREAM_COMPACT_ROOT}" -type l -print -quit)"
test -z "$(find "${UPSTREAM_PREDICTION_PANEL_ROOT}" -type l -print -quit)"

copy_file "${CANDIDATE_AMENDMENT_PATH}" "${compact}/candidate-amendment.json"
copy_file "${EXECUTION_LOCK_PATH}" "${compact}/execution-lock.json"
copy_file "${source_plan}" "${compact}/upstream-source-plan.json"
copy_file "${source_batch}" "${compact}/upstream-prediction-batch.json"
copy_file "${source_receipt}" "${compact}/upstream-prediction-receipt.json"
copy_file "${source_execution}" "${compact}/upstream-execution-receipt.json"

python scripts/science/run_deform360_fresh_object_session_candidate_v6_1.py \
  publish-panel \
  --amendment "${compact}/candidate-amendment.json" \
  --execution-lock "${compact}/execution-lock.json" \
  --source-plan "${compact}/upstream-source-plan.json" \
  --upstream-prediction-batch "${compact}/upstream-prediction-batch.json" \
  --upstream-prediction-receipt "${compact}/upstream-prediction-receipt.json" \
  --upstream-execution-receipt "${compact}/upstream-execution-receipt.json" \
  --upstream-source-seal-root "${source_seals}" \
  --upstream-prediction-root "${UPSTREAM_PREDICTION_PANEL_ROOT}" \
  --input-root "${RESULTS_ROOT}" \
  --output-root "${panel}" \
  --candidate-revision "${BPT_SOURCE_SHA}" \
  > "${logs}/publish-panel.json"

python scripts/science/run_deform360_fresh_object_session_candidate_v6_1.py \
  validate-panel \
  --execution-lock "${compact}/execution-lock.json" \
  --output-root "${panel}" \
  > "${logs}/validate-panel.json"

copy_file \
  "${panel}/candidate-panel-receipt.json" \
  "${compact}/candidate-panel-receipt.json"
copy_file \
  "${panel}/raw-nested-prediction-batch.json" \
  "${compact}/candidate-raw-nested-prediction-batch.json"

python scripts/science/run_deform360_fresh_object_session_candidate_v6_1.py \
  seal-execution \
  --amendment "${compact}/candidate-amendment.json" \
  --execution-lock "${compact}/execution-lock.json" \
  --source-plan "${compact}/upstream-source-plan.json" \
  --upstream-prediction-batch "${compact}/upstream-prediction-batch.json" \
  --upstream-prediction-receipt "${compact}/upstream-prediction-receipt.json" \
  --upstream-execution-receipt "${compact}/upstream-execution-receipt.json" \
  --output-root "${panel}" \
  --candidate-revision "${BPT_SOURCE_SHA}" \
  --runner-name "${RUNNER_NAME}" \
  --workflow-run-id "${GITHUB_RUN_ID}" \
  --workflow-run-attempt "${GITHUB_RUN_ATTEMPT}" \
  --output "${compact}/execution-receipt.json" \
  > "${logs}/seal-execution.json"

(
  cd "${compact}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum --check SHA256SUMS >/dev/null
)

python scripts/science/run_deform360_fresh_object_session_candidate_v6_1.py \
  validate-execution \
  --receipt "${compact}/execution-receipt.json" \
  > "${RUN_ROOT}/validated-execution.json"

test "$(git rev-parse HEAD)" = "${BPT_SOURCE_SHA}"
test -z "$(git status --porcelain=v1)"
printf '%s\n' "${compact}"
