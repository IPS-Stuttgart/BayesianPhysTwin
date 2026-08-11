#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

ARCHIVED_RUNNER="scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
REPAIR_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_generic_selector_identity_repair.json"
REPAIR_ID="d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
ARCHIVED_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
PREVIOUS_SELECTOR_SHA256="79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
CORRECTED_SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"

# Reviewed archived-runner invariants intentionally remain visible here so the
# existing source-execution contract continues to guard the active entrypoint:
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

test -f "${ARCHIVED_RUNNER}"
test ! -L "${ARCHIVED_RUNNER}"
test "$(git hash-object "${ARCHIVED_RUNNER}")" = "${ARCHIVED_RUNNER_BLOB_SHA}"
test -f "${REPAIR_PATH}"
test ! -L "${REPAIR_PATH}"

export REPAIR_PATH REPAIR_ID
export PREVIOUS_SELECTOR_SHA256 CORRECTED_SELECTOR_SHA256
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["REPAIR_PATH"])
payload = json.loads(path.read_text(encoding="utf-8"))
declared = payload.pop("repair_id")
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
expected = os.environ["REPAIR_ID"]
if declared != observed or observed != expected:
    raise SystemExit("v6 selector runtime repair identity changed")
if payload.get("schema") != (
    "bayesian-phystwin.deform360-v6-source-runtime-identity-repair"
):
    raise SystemExit("v6 selector runtime repair schema changed")
if payload.get("schema_version") != 1:
    raise SystemExit("v6 selector runtime repair version changed")
if payload.get("superseded_execution_amendment_id") != (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
):
    raise SystemExit("v6 selector runtime repair changed execution amendment")

correction = payload.get("correction", {})
expected_correction = {
    "field": "runtime_sources.generic_selector_source_sha256",
    "repository": "IPS-Stuttgart/Causal4D",
    "repository_revision": "50e3682a5dbf976b20cc9115b6e7a975d0144ea5",
    "path": "src/causal4d_public/deform360_object_sam2.py",
    "selector_semantics": "deform360-object-sam2-generic-selector",
    "previous_sha256": os.environ["PREVIOUS_SELECTOR_SHA256"],
    "corrected_sha256": os.environ["CORRECTED_SELECTOR_SHA256"],
    "corrected_byte_count": 17310,
}
if correction != expected_correction:
    raise SystemExit("v6 selector runtime repair correction changed")

failed = payload.get("failed_execution_evidence", {})
if failed.get("workflow_run_id") != 31458096956:
    raise SystemExit("v6 selector runtime repair lost the failed run")
if failed.get("artifact_id") != 9088797337:
    raise SystemExit("v6 selector runtime repair lost the failed artifact")
if failed.get("execution_receipt_id") != (
    "cfcfeab74ee9cc88002e398afa2655ccc1a56752787fe6b44a961061fb7cd040"
):
    raise SystemExit("v6 selector runtime repair lost the failed receipt")
if failed.get("physical_manifest_count") != 0:
    raise SystemExit("v6 selector repair was declared after physical prediction")
if failed.get("source_prediction_seal_count") != 0:
    raise SystemExit("v6 selector repair was declared after source prediction")

probe = payload.get("diagnostic_probe", {})
if probe.get("workflow_run_id") != 31458663573:
    raise SystemExit("v6 selector runtime repair lost the history probe")
if probe.get("complete_history_searched") is not True:
    raise SystemExit("v6 selector runtime repair lacks complete-history evidence")
if probe.get("historical_match_found") is not False:
    raise SystemExit("v6 selector runtime repair falsely claims a historical match")
if probe.get("observed_sha256") != os.environ["CORRECTED_SELECTOR_SHA256"]:
    raise SystemExit("v6 selector runtime repair probe identity changed")
if probe.get("observed_byte_count") != 17310:
    raise SystemExit("v6 selector runtime repair probe byte count changed")

scope = payload.get("repair_scope", {})
if scope.get("runtime_byte_identity_only") is not True:
    raise SystemExit("v6 selector runtime repair is not byte-identity-only")
for field in (
    "model_family_changed",
    "model_size_changed",
    "repository_revision_changed",
    "selector_semantics_changed",
    "source_cohort_changed",
    "camera_panel_changed",
    "candidate_roster_changed",
    "loss_or_gate_changed",
    "replacement_allowed",
    "claim_authorized",
):
    if scope.get(field) is not False:
        raise SystemExit(f"v6 selector runtime repair widened {field}")

boundary = payload.get("information_boundary", {})
if not boundary or any(value is not False for value in boundary.values()):
    raise SystemExit("v6 selector runtime repair crossed the information boundary")

authorization = payload.get("execution_authorization", {})
if authorization.get("event") != "push-to-protected-main-after-reviewed-merge":
    raise SystemExit("v6 selector runtime repair execution event changed")
if authorization.get("runner_name") != "workstation2":
    raise SystemExit("v6 selector runtime repair runner changed")
if authorization.get("source_prediction_batch_required_before_suffix_access") is not True:
    raise SystemExit("v6 selector runtime repair weakened the prediction barrier")
if authorization.get("fresh_target_selection_authorized") is not False:
    raise SystemExit("v6 selector runtime repair authorized target selection")
if authorization.get("fresh_target_payload_access_authorized") is not False:
    raise SystemExit("v6 selector runtime repair authorized target access")
PY

PATCHED_RUNNER_PATH="$(
  mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-source-runner.XXXXXX.sh"
)"
cleanup() {
  rm -f "${PATCHED_RUNNER_PATH}"
}
trap cleanup EXIT

export ARCHIVED_RUNNER PATCHED_RUNNER_PATH
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

source = Path(os.environ["ARCHIVED_RUNNER"])
target = Path(os.environ["PATCHED_RUNNER_PATH"])
text = source.read_text(encoding="utf-8")
old = f'SELECTOR_SHA256="{os.environ["PREVIOUS_SELECTOR_SHA256"]}"'
new = f'SELECTOR_SHA256="{os.environ["CORRECTED_SELECTOR_SHA256"]}"'
if text.count(old) != 1:
    raise SystemExit("archived v6 runner selector identity occurrence changed")
patched = text.replace(old, new)
if patched.count(new) != 1:
    raise SystemExit("temporary v6 runner selector identity was not unique")
target.write_text(patched, encoding="utf-8")
target.chmod(0o700)
PY

set +e
bash "${PATCHED_RUNNER_PATH}"
status=$?
set -e

receipt="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence/execution-receipt.json"
if [[ -f "${receipt}" && ! -L "${receipt}" ]]; then
  export RECEIPT_PATH="${receipt}"
  "${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.pop("receipt_id", None)
receipt["runtime_identity_repair_id"] = os.environ["REPAIR_ID"]
receipt["runtime_identity_repair_path"] = os.environ["REPAIR_PATH"]
receipt["runtime_selector_identity"] = {
    "repository": "IPS-Stuttgart/Causal4D",
    "repository_revision": "50e3682a5dbf976b20cc9115b6e7a975d0144ea5",
    "path": "src/causal4d_public/deform360_object_sam2.py",
    "sha256": os.environ["CORRECTED_SELECTOR_SHA256"],
    "byte_count": 17310,
}
canonical = json.dumps(
    receipt,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
receipt["receipt_id"] = hashlib.sha256(canonical).hexdigest()
path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
  compact="$(dirname "${receipt}")"
  (
    cd "${compact}"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum --check SHA256SUMS >/dev/null
  )
fi

exit "${status}"
