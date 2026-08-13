#!/usr/bin/env bash
set -euo pipefail

required=(
  ALIGNED_ROOT
  AUTHORIZED_RUNNER_NAME
  BPT_PYTHON
  BPT_SOURCE_SHA
  CANDIDATE_EXECUTION_RECEIPT
  CANDIDATE_ROOT
  CAUSAL4D_REPOSITORY
  DEFORM360_REPOSITORY
  ENDPOINT_ROOT
  EXECUTION_LOCK_PATH
  GITHUB_RUN_ATTEMPT
  GITHUB_RUN_ID
  GSPLAT_WHEEL
  RUNNER_NAME
  RUN_CLAIM
  RUN_ROOT
  SAM2_CHECKPOINT
  SAM2_REPOSITORY
  SOURCE_PLAN
  SOURCE_SCORING_AMENDMENT_PATH
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
test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"
test "$(git rev-parse HEAD)" = "${BPT_SOURCE_SHA}"
test -z "$(git status --porcelain=v1 --untracked-files=no)"
test ! -e "${RUN_ROOT}"
test ! -e "${RUN_CLAIM}"
test ! -e "${ENDPOINT_ROOT}"
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

mkdir "${RUN_ROOT}" "${ENDPOINT_ROOT}"
compact="${RUN_ROOT}/compact"
logs="${compact}/logs"
mkdir "${compact}" "${logs}"
install -m 600 "${RUN_CLAIM}" "${compact}/execution-started.txt"
copy_file "${SOURCE_SCORING_AMENDMENT_PATH}" "${compact}/source-scoring-amendment.json"
copy_file "${EXECUTION_LOCK_PATH}" "${compact}/execution-lock.json"
copy_file "${SOURCE_PLAN}" "${compact}/upstream-source-plan.json"
copy_file \
  "${CANDIDATE_ROOT}/candidate-panel-receipt.json" \
  "${compact}/candidate-panel-receipt.json"
copy_file \
  "${CANDIDATE_ROOT}/raw-nested-prediction-batch.json" \
  "${compact}/candidate-raw-nested-prediction-batch.json"
copy_file \
  "${CANDIDATE_EXECUTION_RECEIPT}" \
  "${compact}/candidate-execution-receipt.json"

"${BPT_PYTHON}" scripts/science/run_deform360_fresh_object_session_source_scorer_v6_1.py \
  authorize \
  --scoring-amendment "${compact}/source-scoring-amendment.json" \
  --execution-lock "${compact}/execution-lock.json" \
  --candidate-root "${CANDIDATE_ROOT}" \
  --candidate-execution-receipt "${compact}/candidate-execution-receipt.json" \
  --source-plan "${compact}/upstream-source-plan.json" \
  --scorer-revision "${BPT_SOURCE_SHA}" \
  --runner-name "${RUNNER_NAME}" \
  --workflow-run-id "${GITHUB_RUN_ID}" \
  --workflow-run-attempt "${GITHUB_RUN_ATTEMPT}" \
  --output "${compact}/source-suffix-opening-authorization.json" \
  > "${logs}/authorize-source-suffix.json"

printf 'true\n' > "${compact}/source-suffix-opened.txt"
printf 'endpoint-processing\n' > "${compact}/terminal-stage.txt"

mapfile -t object_ids < <(
  jq -er '.objects | map(.object_id) | sort | .[]' \
    "${compact}/upstream-source-plan.json"
)
test "${#object_ids[@]}" -eq 10

worker() {
  local gpu="$1"
  local index object_id status
  for index in "${!object_ids[@]}"; do
    if (( index % 2 != gpu )); then
      continue
    fi
    object_id="${object_ids[index]}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${BPT_PYTHON}" \
        scripts/remote/process_deform360_fresh_object_session_source_endpoint_v6_1.py \
        --repo "$(pwd -P)" \
        --scoring-amendment "${compact}/source-scoring-amendment.json" \
        --authorization "${compact}/source-suffix-opening-authorization.json" \
        --source-plan "${compact}/upstream-source-plan.json" \
        --aligned-root "${ALIGNED_ROOT}" \
        --output-root "${ENDPOINT_ROOT}" \
        --object-id "${object_id}" \
        --selector-source-root "${CAUSAL4D_REPOSITORY}" \
        --sam2-repository "${SAM2_REPOSITORY}" \
        --sam2-checkpoint "${SAM2_CHECKPOINT}" \
        --deform360-repository "${DEFORM360_REPOSITORY}" \
        --gsplat-wheel "${GSPLAT_WHEEL}" \
        --ffmpeg /usr/bin/ffmpeg \
        --device cuda:0 \
        > "${logs}/endpoint-${index}.json" 2>&1
    status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
      printf '%s\n' "${status}" > "${compact}/endpoint-worker-${gpu}-exit-code.txt"
      return "${status}"
    fi
  done
}

worker 0 &
worker0=$!
worker 1 &
worker1=$!
set +e
wait "${worker0}"
status0=$?
wait "${worker1}"
status1=$?
set -e
if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  exit 2
fi

objects=()
for object_id in "${object_ids[@]}"; do
  record="${ENDPOINT_ROOT}/objects/${object_id}/endpoint-object.json"
  ordinary_file "${record}"
  objects+=(--object "${record}")
done
"${BPT_PYTHON}" scripts/science/run_deform360_fresh_object_session_source_scorer_v6_1.py \
  build-endpoint-manifest \
  --authorization "${compact}/source-suffix-opening-authorization.json" \
  --source-plan "${compact}/upstream-source-plan.json" \
  --processor-revision "${BPT_SOURCE_SHA}" \
  "${objects[@]}" \
  --output "${compact}/source-endpoint-manifest.json" \
  > "${logs}/build-endpoint-manifest.json"

printf 'score-source-panel\n' > "${compact}/terminal-stage.txt"
set +e
"${BPT_PYTHON}" scripts/science/run_deform360_fresh_object_session_source_scorer_v6_1.py \
  score \
  --scoring-amendment "${compact}/source-scoring-amendment.json" \
  --execution-lock "${compact}/execution-lock.json" \
  --candidate-root "${CANDIDATE_ROOT}" \
  --candidate-execution-receipt "${compact}/candidate-execution-receipt.json" \
  --source-plan "${compact}/upstream-source-plan.json" \
  --scorer-revision "${BPT_SOURCE_SHA}" \
  --runner-name "${RUNNER_NAME}" \
  --workflow-run-id "${GITHUB_RUN_ID}" \
  --workflow-run-attempt "${GITHUB_RUN_ATTEMPT}" \
  --authorization "${compact}/source-suffix-opening-authorization.json" \
  --endpoint-manifest "${compact}/source-endpoint-manifest.json" \
  --endpoint-root "${ENDPOINT_ROOT}" \
  --output-root "${RUN_ROOT}" \
  > "${logs}/score-source-panel.json" 2>&1
score_status=$?
set -e
printf '%s\n' "${score_status}" > "${compact}/source-score-exit-code.txt"
if [[ "${score_status}" -ne 0 && "${score_status}" -ne 3 ]]; then
  exit "${score_status}"
fi

ordinary_file "${RUN_ROOT}/source-evidence.json"
ordinary_file "${RUN_ROOT}/source-result.json"
ordinary_file "${RUN_ROOT}/source-scoring-receipt.json"
copy_file "${RUN_ROOT}/source-evidence.json" "${compact}/source-evidence.json"
copy_file "${RUN_ROOT}/source-result.json" "${compact}/source-result.json"
copy_file \
  "${RUN_ROOT}/source-scoring-receipt.json" \
  "${compact}/source-scoring-receipt.json"
test -d "${RUN_ROOT}/source-outcomes"
test ! -L "${RUN_ROOT}/source-outcomes"
test -d "${RUN_ROOT}/support-reports"
test ! -L "${RUN_ROOT}/support-reports"
test -z "$(find "${RUN_ROOT}/source-outcomes" "${RUN_ROOT}/support-reports" -type l -print -quit)"
cp -a "${RUN_ROOT}/source-outcomes" "${compact}/source-outcomes"
cp -a "${RUN_ROOT}/support-reports" "${compact}/support-reports"
for object_id in "${object_ids[@]}"; do
  copy_file \
    "${ENDPOINT_ROOT}/objects/${object_id}/endpoint-object.json" \
    "${compact}/endpoint-object-${object_id}.json"
done

"${BPT_PYTHON}" scripts/science/run_deform360_fresh_object_session_source_scorer_v6_1.py \
  validate-receipt --receipt "${compact}/source-scoring-receipt.json" \
  > "${logs}/validate-source-scoring-receipt.json"

rm -f "${compact}/SHA256SUMS"
(
  cd "${compact}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum --check SHA256SUMS >/dev/null
)
test "$(git rev-parse HEAD)" = "${BPT_SOURCE_SHA}"
test -z "$(git status --porcelain=v1 --untracked-files=no)"
printf '%s\n' "${compact}"
