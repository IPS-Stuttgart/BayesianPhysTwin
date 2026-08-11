#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

SELECTOR_WRAPPER_V3="scripts/ci/archive/run_deform360_v6_source_prediction_evidence_selector_repair_v3.sh"
SELECTOR_WRAPPER_V3_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"
REPAIR_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_prepared_inventory_identity_repair.json"
REPAIR_ID="bd9b1b9e37529c7a7e555ff8ec7e62521bece77fe8554b33047b1d33a2de7fa4"
PREPARED_INVENTORY_REVISION="e190c94014e6024e324d860618662526af6ea682"
PREPARED_INVENTORY_ID="6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
PREPARED_INVENTORY_FILE_SHA256="4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
AUTHORITATIVE_RUN_ID="31272512658"
AUTHORITATIVE_ARTIFACT_ID="9026043628"
AUTHORITATIVE_ARTIFACT_SHA256="d0041af0ba0cfe6e5c5bd4008c47adb3ed4cf0cf0f6754eff67a238e746c7a86"
FAILED_RUN_ID="31462653379"
FAILED_ARTIFACT_ID="9090402942"
FAILED_RECEIPT_ID="a62ead70994b330d3eecf8d35afe6baa1275c428f4c90c4e18157aac55f3733f"

# The delegated wrappers preserve these reviewed scientific invariants:
# prediction_record_count") != 100
# source-prediction-evidence-sealed
# source-inputs-incomplete
# source-technical-failure-retained
# run_deform360_joint_sparse_source_predictions_v5.py
# development_suffix_opened": False
# v6_target_payloads_opened": False
# fresh_target_selection_authorized": False

test -f "${SELECTOR_WRAPPER_V3}"
test ! -L "${SELECTOR_WRAPPER_V3}"
test "$(git hash-object "${SELECTOR_WRAPPER_V3}")" = "${SELECTOR_WRAPPER_V3_BLOB_SHA}"
test -f "${REPAIR_PATH}"
test ! -L "${REPAIR_PATH}"

export REPAIR_PATH REPAIR_ID
export PREPARED_INVENTORY_REVISION PREPARED_INVENTORY_ID
export PREPARED_INVENTORY_FILE_SHA256
export AUTHORITATIVE_RUN_ID AUTHORITATIVE_ARTIFACT_ID AUTHORITATIVE_ARTIFACT_SHA256
export FAILED_RUN_ID FAILED_ARTIFACT_ID FAILED_RECEIPT_ID
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
if declared != observed or observed != os.environ["REPAIR_ID"]:
    raise SystemExit("v6 prepared-inventory runtime repair identity changed")
if payload.get("schema") != (
    "bayesian-phystwin.deform360-v6-source-runtime-inventory-identity-repair"
):
    raise SystemExit("v6 prepared-inventory runtime repair schema changed")
if payload.get("schema_version") != 1:
    raise SystemExit("v6 prepared-inventory runtime repair version changed")
if payload.get("superseded_execution_amendment_id") != (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
):
    raise SystemExit("v6 prepared-inventory repair changed execution amendment")

correction = payload.get("correction", {})
expected_correction = {
    "field": "prepared_source_inventory.implementation_revision",
    "previous_runtime_value": "current-protected-main-source-revision",
    "corrected_revision": os.environ["PREPARED_INVENTORY_REVISION"],
    "expected_inventory_id": os.environ["PREPARED_INVENTORY_ID"],
    "expected_file_sha256": os.environ["PREPARED_INVENTORY_FILE_SHA256"],
    "authoritative_workflow_run_id": int(os.environ["AUTHORITATIVE_RUN_ID"]),
    "authoritative_artifact_id": int(os.environ["AUTHORITATIVE_ARTIFACT_ID"]),
    "authoritative_artifact_sha256": os.environ["AUTHORITATIVE_ARTIFACT_SHA256"],
}
if correction != expected_correction:
    raise SystemExit("v6 prepared-inventory runtime correction changed")

failed = payload.get("failed_execution_evidence", {})
expected_failed = {
    "workflow_run_id": int(os.environ["FAILED_RUN_ID"]),
    "artifact_id": int(os.environ["FAILED_ARTIFACT_ID"]),
    "artifact_sha256": (
        "edbfe17c5c85c4e9f7375026fd15593132ee748456a60a5c97b2ea87776a3304"
    ),
    "execution_receipt_id": os.environ["FAILED_RECEIPT_ID"],
    "terminal_stage": "physical-source:026-sock-cloth-ep0007",
    "physical_manifest_count": 0,
    "source_prediction_seal_count": 0,
    "observed_inventory_id": (
        "3972737207fd684e5e31cd507c4b3a0e9e2d0ed1b9d5cd427774f019b2b704cc"
    ),
    "observed_file_sha256": (
        "7714c5d4b0aaed32358f21d133a84ab038d2f98eb1c48639d43df736ae801acf"
    ),
}
if failed != expected_failed:
    raise SystemExit("v6 prepared-inventory repair lost failed-run evidence")

scope = payload.get("repair_scope", {})
if scope.get("runtime_identity_replay_only") is not True:
    raise SystemExit("v6 prepared-inventory repair is not identity-replay-only")
for field in (
    "inventory_payload_changed",
    "prepared_source_bytes_changed",
    "source_cohort_changed",
    "camera_panel_changed",
    "candidate_roster_changed",
    "loss_or_gate_changed",
    "model_or_checkpoint_changed",
    "replacement_allowed",
    "claim_authorized",
):
    if scope.get(field) is not False:
        raise SystemExit(f"v6 prepared-inventory repair widened {field}")

boundary = payload.get("information_boundary", {})
if not boundary or any(value is not False for value in boundary.values()):
    raise SystemExit("v6 prepared-inventory repair crossed the information boundary")

authorization = payload.get("execution_authorization", {})
if authorization.get("event") != "push-to-protected-main-after-reviewed-merge":
    raise SystemExit("v6 prepared-inventory execution event changed")
if authorization.get("runner_name") != "workstation2":
    raise SystemExit("v6 prepared-inventory repair runner changed")
if authorization.get("source_prediction_batch_required_before_suffix_access") is not True:
    raise SystemExit("v6 prepared-inventory repair weakened the prediction barrier")
if authorization.get("fresh_target_selection_authorized") is not False:
    raise SystemExit("v6 prepared-inventory repair authorized target selection")
if authorization.get("fresh_target_payload_access_authorized") is not False:
    raise SystemExit("v6 prepared-inventory repair authorized target access")
PY

PATCHED_SELECTOR_WRAPPER="$(
  mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-selector-wrapper.XXXXXX.sh"
)"
cleanup() {
  rm -f "${PATCHED_SELECTOR_WRAPPER}"
}
trap cleanup EXIT

export SELECTOR_WRAPPER_V3 PATCHED_SELECTOR_WRAPPER
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

source = Path(os.environ["SELECTOR_WRAPPER_V3"])
target = Path(os.environ["PATCHED_SELECTOR_WRAPPER"])
text = source.read_text(encoding="utf-8")
anchor = "patched = text.replace(old, new)\n"
insertion = '''patched = text.replace(old, new)
inventory_old = '--implementation-revision "${BPT_SOURCE_SHA}"'
inventory_new = (
    f'--implementation-revision "{os.environ["PREPARED_INVENTORY_REVISION"]}"'
)
if patched.count(inventory_old) != 1:
    raise SystemExit("archived v6 runner inventory identity occurrence changed")
patched = patched.replace(inventory_old, inventory_new)
if patched.count(inventory_new) != 1:
    raise SystemExit("temporary v6 runner inventory identity was not unique")
'''
if text.count(anchor) != 1:
    raise SystemExit("selector wrapper patch anchor changed")
patched = text.replace(anchor, insertion, 1)
target.write_text(patched, encoding="utf-8")
target.chmod(0o700)
PY

set +e
bash "${PATCHED_SELECTOR_WRAPPER}"
status=$?
set -e

compact="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence"
inventory="${compact}/prepared-source-inventory.json"
if [[ -f "${inventory}" && ! -L "${inventory}" ]]; then
  export INVENTORY_PATH="${inventory}"
  "${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["INVENTORY_PATH"])
if hashlib.sha256(path.read_bytes()).hexdigest() != os.environ[
    "PREPARED_INVENTORY_FILE_SHA256"
]:
    raise SystemExit("replayed prepared-source inventory file identity changed")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("inventory_id") != os.environ["PREPARED_INVENTORY_ID"]:
    raise SystemExit("replayed prepared-source inventory content identity changed")
if payload.get("implementation_revision") != os.environ[
    "PREPARED_INVENTORY_REVISION"
]:
    raise SystemExit("replayed prepared-source inventory producer revision changed")
PY
fi

receipt="${compact}/execution-receipt.json"
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
receipt["runtime_prepared_source_inventory_identity_repair_id"] = os.environ[
    "REPAIR_ID"
]
receipt["runtime_prepared_source_inventory_identity_repair_path"] = os.environ[
    "REPAIR_PATH"
]
receipt["runtime_prepared_source_inventory"] = {
    "implementation_revision": os.environ["PREPARED_INVENTORY_REVISION"],
    "inventory_id": os.environ["PREPARED_INVENTORY_ID"],
    "file_sha256": os.environ["PREPARED_INVENTORY_FILE_SHA256"],
    "authoritative_workflow_run_id": int(os.environ["AUTHORITATIVE_RUN_ID"]),
    "authoritative_artifact_id": int(os.environ["AUTHORITATIVE_ARTIFACT_ID"]),
    "authoritative_artifact_sha256": os.environ["AUTHORITATIVE_ARTIFACT_SHA256"],
    "failed_workflow_run_id": int(os.environ["FAILED_RUN_ID"]),
    "failed_artifact_id": int(os.environ["FAILED_ARTIFACT_ID"]),
    "failed_execution_receipt_id": os.environ["FAILED_RECEIPT_ID"],
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
  (
    cd "${compact}"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum --check SHA256SUMS >/dev/null
  )
fi

exit "${status}"
