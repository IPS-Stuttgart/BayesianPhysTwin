#!/usr/bin/env bash
set -euo pipefail

PREVIOUS_RUNNER_REVISION="07fab9fcd3dae6ab0ec05c56ef565ab16d4466a5"
PREVIOUS_RUNNER_PATH="scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
PREVIOUS_RUNNER_BLOB_SHA="bf670c99351c9c2ed6dd3cdea9aeb106c1ffb4ca"
PREVIOUS_RUNNER_SHA256="75a40281f69c4f99843cc59ca04107e7dda86289a3804d6edb12c88ab8d9e6fb"
STAGE_PREFIX_COMPATIBILITY_REPAIR_ID="048733975c44dfc9cf7b1c5bcfa6985327aaba650560305fbbff9c2ec6449c75"
STAGE_PREFIX_FAILED_WORKFLOW_RUN_ID="31510971371"
STAGE_PREFIX_FAILED_ARTIFACT_ID="9109136220"
STAGE_PREFIX_FAILED_ARTIFACT_SHA256="7e4bd7ba33db2985a2b8e768c1a489487d89b86f736276ed1d25d6cf9b3c73a1"
STAGE_PREFIX_FAILED_RECEIPT_ID="ea3856ed0084efd5e13357df877bc1e3bc0a64257c043a35490fda65054660b5"
STAGE_PREFIX_FAILED_SOURCE_REVISION="b0f6b46991a20c54260baf58ddf62fbb6dab7813"
STAGE_PREFIX_FAILED_TERMINAL_STAGE="stage-prefix:026-sock-cloth-ep0007"
STAGE_PREFIX_COMMAND="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"

# The delegated exact runner retains these reviewed bindings and behaviors:
# PHYSICAL_UPSTREAM_REVISION="9f69d5d6c5d81d6d6e8f123c18ddba73dc4afa65"
# PHYSICAL_UPSTREAM_DIAGNOSTIC_RUN_ID="31461017011"
# PHYSICAL_UPSTREAM_REPORT_ID="75c1be85233e1835dfef5a1227a28e8938995335ead701fe8d3dfd8b5960a087"
# PREPARED_INVENTORY_IMPLEMENTATION_REVISION="e190c94014e6024e324d860618662526af6ea682"
# PREPARED_INVENTORY_ID="6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
# PREPARED_INVENTORY_FILE_SHA256="4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
# PREPARED_INVENTORY_ADMISSION_RUN_ID="31272512658"
# REPAIR_ID="d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
# ARCHIVED_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
# SCIENCE_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
# SELECTOR_WRAPPER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"
# PREVIOUS_SELECTOR_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
# CORRECTED_SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
# text.count(old) != 1
# patched.count(new) != 1
# "runtime_identity_repair_id"
# "runtime_selector_identity"
# inventory_pattern = re.compile(
# len(inventory_pattern.findall(runner)) != 1
# target="scripts/science/inventory_deform360_calibration_prepared_source.py"
# rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")
# if [[ "${replacements}" -ne 1 ]]
# exec "${REAL_BPT_PYTHON}" "$@"
# BPT_PYTHON="${PYTHON_SHIM}"
# "runtime_prepared_inventory_identity"
# prediction_record_count") != 100
# source-prediction-evidence-sealed
# source-inputs-incomplete
# source-technical-failure-retained
# run_deform360_joint_sparse_source_predictions_v5.py
# SAM2_CHECKPOINT_NAME="sam2.1_hiera_small.pt"
# SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM2_CHECKPOINT_NAME}"
# SAM2_SHA256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
# development_suffix_opened": False
# v6_target_payloads_opened": False
# fresh_target_selection_authorized": False
# unique-complete-history-exact-ten-file-sha256-match
# ["git", "-C", str(repository), "show"
# "fetch",
# "--depth=1",
# "origin",\n            revision,
# : "${BPT_SOURCE_SHA:?BPT_SOURCE_SHA is required}"
# git worktree add --detach "${EXECUTION_REPO_ROOT}" "${BPT_SOURCE_SHA}"
# test "$(git -C "${EXECUTION_REPO_ROOT}" rev-parse HEAD)" = "${BPT_SOURCE_SHA}"
# test -z "$(git -C "${EXECUTION_REPO_ROOT}" status --porcelain=v1)"
# cd "${EXECUTION_REPO_ROOT}"
# GITHUB_WORKSPACE="${EXECUTION_REPO_ROOT}" \
# PYTHONPATH="${EXECUTION_REPO_ROOT}/src:${PYTHONPATH:-}" \
# RUNNER_WORKSPACE="${PHYSICAL_UPSTREAM_ROOT}" \
# git worktree remove --force "${EXECUTION_REPO_ROOT}"
# bash "${SELECTOR_WRAPPER}"

: "${BPT_PYTHON:?BPT_PYTHON is required}"

DELEGATED_RUNNER="$(
  mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-delegated-runner.XXXXXX.sh"
)"
COMPATIBILITY_PYTHON_SHIM="$(
  mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-stage-prefix-python.XXXXXX"
)"
REAL_BPT_PYTHON="${BPT_PYTHON}"

cleanup() {
  rm -f "${DELEGATED_RUNNER}" "${COMPATIBILITY_PYTHON_SHIM}"
}
trap cleanup EXIT

git show "${PREVIOUS_RUNNER_REVISION}:${PREVIOUS_RUNNER_PATH}" \
  > "${DELEGATED_RUNNER}"
test "$(git hash-object "${DELEGATED_RUNNER}")" = "${PREVIOUS_RUNNER_BLOB_SHA}"
test "$(sha256sum "${DELEGATED_RUNNER}" | awk '{print $1}')" \
  = "${PREVIOUS_RUNNER_SHA256}"
chmod 700 "${DELEGATED_RUNNER}"

cat > "${COMPATIBILITY_PYTHON_SHIM}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${REAL_BPT_PYTHON:?REAL_BPT_PYTHON is required}"

stage_target="scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
if [[ "${1:-}" == "${stage_target}" ]]; then
  arguments=("$@")
  stage=""
  execution_repo=""
  legacy_repo=""
  legacy_role=""
  stage_count=0
  execution_repo_count=0
  legacy_repo_count=0
  legacy_role_count=0
  index=1
  while [[ "${index}" -lt "${#arguments[@]}" ]]; do
    option="${arguments[$index]}"
    case "${option}" in
      --stage|--execution-repo|--repo|--role)
        value_index=$((index + 1))
        if [[ "${value_index}" -ge "${#arguments[@]}" ]]; then
          echo "stage-prefix compatibility option lacks a value: ${option}" >&2
          exit 2
        fi
        value="${arguments[$value_index]}"
        case "${option}" in
          --stage)
            stage="${value}"
            stage_count=$((stage_count + 1))
            ;;
          --execution-repo)
            execution_repo="${value}"
            execution_repo_count=$((execution_repo_count + 1))
            ;;
          --repo)
            legacy_repo="${value}"
            legacy_repo_count=$((legacy_repo_count + 1))
            ;;
          --role)
            legacy_role="${value}"
            legacy_role_count=$((legacy_role_count + 1))
            ;;
        esac
        index=$((index + 2))
        ;;
      *)
        index=$((index + 1))
        ;;
    esac
  done

  if [[ "${stage}" == "stage-prefix" ]]; then
    if [[ "${stage_count}" -ne 1 \
      || "${execution_repo_count}" -ne 1 \
      || "${legacy_repo_count}" -ne 1 \
      || "${legacy_role_count}" -ne 1 ]]; then
      echo "stage-prefix legacy argument binding is not unique" >&2
      exit 2
    fi
    if [[ "${legacy_repo}" != "${execution_repo}" ]]; then
      echo "stage-prefix repository aliases differ" >&2
      exit 2
    fi
    if [[ "${legacy_role}" != "calibration" ]]; then
      echo "stage-prefix legacy role changed" >&2
      exit 2
    fi

    rewritten=("${arguments[0]}")
    index=1
    while [[ "${index}" -lt "${#arguments[@]}" ]]; do
      option="${arguments[$index]}"
      if [[ "${option}" == "--repo" || "${option}" == "--role" ]]; then
        index=$((index + 2))
      else
        rewritten+=("${option}")
        index=$((index + 1))
      fi
    done
    exec "${REAL_BPT_PYTHON}" "${rewritten[@]}"
  fi
fi

exec "${REAL_BPT_PYTHON}" "$@"
SH
chmod 700 "${COMPATIBILITY_PYTHON_SHIM}"

export REAL_BPT_PYTHON
set +e
BPT_PYTHON="${COMPATIBILITY_PYTHON_SHIM}" \
  bash "${DELEGATED_RUNNER}" "$@"
status=$?
set -e

receipt="${EVIDENCE_ROOT:-}/deform360-v6-source-prediction-evidence/execution-receipt.json"
if [[ -n "${EVIDENCE_ROOT:-}" && -f "${receipt}" && ! -L "${receipt}" ]]; then
  export RECEIPT_PATH="${receipt}"
  export STAGE_PREFIX_COMPATIBILITY_REPAIR_ID
  export STAGE_PREFIX_FAILED_WORKFLOW_RUN_ID
  export STAGE_PREFIX_FAILED_ARTIFACT_ID
  export STAGE_PREFIX_FAILED_ARTIFACT_SHA256
  export STAGE_PREFIX_FAILED_RECEIPT_ID
  export STAGE_PREFIX_FAILED_SOURCE_REVISION
  export STAGE_PREFIX_FAILED_TERMINAL_STAGE
  export STAGE_PREFIX_COMMAND
  "${REAL_BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

repair = {
    "schema": (
        "bayesian-phystwin.deform360-v6-"
        "stage-prefix-argument-compatibility-repair"
    ),
    "schema_version": 1,
    "failed_execution": {
        "workflow_run_id": int(os.environ["STAGE_PREFIX_FAILED_WORKFLOW_RUN_ID"]),
        "artifact_id": int(os.environ["STAGE_PREFIX_FAILED_ARTIFACT_ID"]),
        "artifact_sha256": os.environ["STAGE_PREFIX_FAILED_ARTIFACT_SHA256"],
        "receipt_id": os.environ["STAGE_PREFIX_FAILED_RECEIPT_ID"],
        "source_revision": os.environ["STAGE_PREFIX_FAILED_SOURCE_REVISION"],
        "terminal_stage": os.environ["STAGE_PREFIX_FAILED_TERMINAL_STAGE"],
        "exit_code": 2,
    },
    "rewrite": {
        "command": os.environ["STAGE_PREFIX_COMMAND"],
        "stage": "stage-prefix",
        "removed_legacy_arguments": ["--repo", "--role"],
        "required_execution_repo_alias_equality": True,
        "required_role": "calibration",
        "all_other_arguments_preserved": True,
        "other_stages_unchanged": True,
    },
    "information_boundary": {
        "claim_authorized": False,
        "development_suffix_opened": False,
        "replacement_allowed": False,
        "v5_confirmation_payloads_opened": False,
        "v6_fresh_target_selected": False,
        "v6_target_outcomes_used": False,
        "v6_target_payloads_opened": False,
    },
}
canonical = json.dumps(
    repair,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
expected = os.environ["STAGE_PREFIX_COMPATIBILITY_REPAIR_ID"]
if observed != expected:
    raise SystemExit("stage-prefix compatibility repair identity changed")
repair["repair_id"] = expected

path = Path(os.environ["RECEIPT_PATH"])
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.pop("receipt_id", None)
existing = receipt.get("runtime_stage_prefix_compatibility")
if existing not in (None, repair):
    raise SystemExit("stage-prefix compatibility receipt binding changed")
receipt["runtime_stage_prefix_compatibility"] = repair
receipt_id = hashlib.sha256(
    json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
receipt["receipt_id"] = receipt_id
path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
  compact="$(dirname "${receipt}")"
  (
    cd "${compact}"
    rm -f SHA256SUMS
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum --check SHA256SUMS >/dev/null
  )
fi

exit "${status}"
