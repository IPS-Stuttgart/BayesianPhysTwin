#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

ARCHIVED_RUNNER="scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v1.sh"
RUNTIME_REPAIR_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_sam2_checkpoint_identity_repair.json"
RUNTIME_REPAIR_ID="28cee70eaa0e8561a320f87d4e51d6c2aad365927814dc94864e299fc145be99"
ARCHIVED_RUNNER_BLOB_SHA="9680176e74e933485e1812bf79b626250925ed1a"
PREVIOUS_SAM2_SHA256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
CORRECTED_SAM2_SHA256="7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"

# Inherited archived-runner invariants retained by this repair wrapper:
# prediction_record_count") != 100
# source-prediction-evidence-sealed
# source-inputs-incomplete
# source-technical-failure-retained
# run_deform360_joint_sparse_source_predictions_v5.py
# development_suffix_opened": False
# v6_target_payloads_opened": False
# fresh_target_selection_authorized": False

test -f "${ARCHIVED_RUNNER}"
test ! -L "${ARCHIVED_RUNNER}"
test "$(git hash-object "${ARCHIVED_RUNNER}")" = "${ARCHIVED_RUNNER_BLOB_SHA}"
test -f "${RUNTIME_REPAIR_PATH}"
test ! -L "${RUNTIME_REPAIR_PATH}"

export RUNTIME_REPAIR_PATH RUNTIME_REPAIR_ID
export PREVIOUS_SAM2_SHA256 CORRECTED_SAM2_SHA256
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RUNTIME_REPAIR_PATH"])
payload = json.loads(path.read_text(encoding="utf-8"))
declared = payload.pop("repair_id")
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
expected = os.environ["RUNTIME_REPAIR_ID"]
if declared != observed or observed != expected:
    raise SystemExit("v6 SAM2 runtime repair identity changed")
if payload.get("schema") != (
    "bayesian-phystwin.deform360-v6-source-runtime-identity-repair"
):
    raise SystemExit("v6 SAM2 runtime repair schema changed")
if payload.get("schema_version") != 1:
    raise SystemExit("v6 SAM2 runtime repair version changed")
if payload.get("superseded_execution_amendment_id") != (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
):
    raise SystemExit("v6 SAM2 runtime repair changed execution amendment")
correction = payload.get("correction", {})
if correction.get("field") != "runtime_sources.sam2_checkpoint_sha256":
    raise SystemExit("v6 SAM2 runtime repair changed scope")
if correction.get("checkpoint_model") != "sam2_hiera_large":
    raise SystemExit("v6 SAM2 runtime repair changed the intended model")
if correction.get("checkpoint_filename") != "sam2_hiera_large.pt":
    raise SystemExit("v6 SAM2 runtime repair changed the checkpoint filename")
if correction.get("previous_sha256") != os.environ["PREVIOUS_SAM2_SHA256"]:
    raise SystemExit("v6 SAM2 runtime repair previous identity changed")
if correction.get("corrected_sha256") != os.environ["CORRECTED_SAM2_SHA256"]:
    raise SystemExit("v6 SAM2 runtime repair corrected identity changed")
if correction.get("corrected_byte_count") != 897952466:
    raise SystemExit("v6 SAM2 runtime repair byte count changed")
if correction.get("repository") != "facebookresearch/sam2":
    raise SystemExit("v6 SAM2 runtime repair repository changed")
if correction.get("repository_revision") != (
    "2b90b9f5ceec907a1c18123530e92e794ad901a4"
):
    raise SystemExit("v6 SAM2 runtime repair repository revision changed")
failed = payload.get("failed_execution_evidence", {})
if failed.get("workflow_run_id") != 31456530482:
    raise SystemExit("v6 SAM2 runtime repair lost the failed run")
if failed.get("execution_receipt_id") != (
    "3159b09724a0e9082bbf0020c38f0c5ec25c8ce3cc08d92f1eb9fa3418c9316d"
):
    raise SystemExit("v6 SAM2 runtime repair lost the failed receipt")
if failed.get("physical_manifest_count") != 0:
    raise SystemExit("v6 SAM2 repair was declared after physical prediction")
if failed.get("source_prediction_seal_count") != 0:
    raise SystemExit("v6 SAM2 repair was declared after source prediction")
scope = payload.get("repair_scope", {})
if scope.get("runtime_byte_identity_only") is not True:
    raise SystemExit("v6 SAM2 runtime repair is not byte-identity-only")
for field in (
    "model_family_changed",
    "model_size_changed",
    "repository_revision_changed",
    "source_cohort_changed",
    "camera_panel_changed",
    "candidate_roster_changed",
    "loss_or_gate_changed",
    "replacement_allowed",
    "claim_authorized",
):
    if scope.get(field) is not False:
        raise SystemExit(f"v6 SAM2 runtime repair widened {field}")
boundary = payload.get("information_boundary", {})
if not boundary or any(value is not False for value in boundary.values()):
    raise SystemExit("v6 SAM2 runtime repair crossed the information boundary")
authorization = payload.get("execution_authorization", {})
if authorization.get("event") != "push-to-protected-main-after-reviewed-merge":
    raise SystemExit("v6 SAM2 runtime repair execution event changed")
if authorization.get("runner_name") != "workstation2":
    raise SystemExit("v6 SAM2 runtime repair runner changed")
if authorization.get("source_prediction_batch_required_before_suffix_access") is not True:
    raise SystemExit("v6 SAM2 runtime repair weakened the prediction barrier")
if authorization.get("fresh_target_selection_authorized") is not False:
    raise SystemExit("v6 SAM2 runtime repair authorized target selection")
if authorization.get("fresh_target_payload_access_authorized") is not False:
    raise SystemExit("v6 SAM2 runtime repair authorized target access")
PY

patched_runner="$(mktemp "${RUNNER_TEMP:-/tmp}/deform360-v6-source-runner.XXXXXX.sh")"
cleanup() {
  rm -f "${patched_runner}"
}
trap cleanup EXIT
export ARCHIVED_RUNNER patched_runner
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

source = Path(os.environ["ARCHIVED_RUNNER"])
target = Path(os.environ["patched_runner"])
text = source.read_text(encoding="utf-8")
old = f'SAM2_SHA256="{os.environ["PREVIOUS_SAM2_SHA256"]}"'
new = f'SAM2_SHA256="{os.environ["CORRECTED_SAM2_SHA256"]}"'
if text.count(old) != 1:
    raise SystemExit("archived v6 runner SAM2 identity occurrence changed")
target.write_text(text.replace(old, new), encoding="utf-8")
target.chmod(0o700)
PY

set +e
bash "${patched_runner}"
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
receipt["runtime_identity_repair_id"] = os.environ["RUNTIME_REPAIR_ID"]
receipt["runtime_identity_repair_path"] = os.environ["RUNTIME_REPAIR_PATH"]
receipt["runtime_checkpoint_identity"] = {
    "checkpoint_model": "sam2_hiera_large",
    "checkpoint_filename": "sam2_hiera_large.pt",
    "sha256": os.environ["CORRECTED_SAM2_SHA256"],
    "byte_count": 897952466,
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
  )
fi

exit "${status}"
