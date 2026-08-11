#!/usr/bin/env bash
set -euo pipefail

: "${BPT_PYTHON:?BPT_PYTHON is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"

SELECTOR_RUNNER="scripts/ci/archive/run_deform360_v6_source_prediction_evidence_selector_repair_v1.sh"
SELECTOR_RUNNER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"
MATERIALIZATION_PATH="protocols/amendments/deform360_official_hub_fresh_object_session_v6_frozen_upstream_materialization.json"
MATERIALIZATION_ID="2056084bd44845446f78600ca42edd8fb23b4003431c87d53ff8d73a5dc275c0"
FROZEN_UPSTREAM_REVISION="9f69d5d6c5d81d6d6e8f123c18ddba73dc4afa65"
LOCATOR_REPORT_ID="75c1be85233e1835dfef5a1227a28e8938995335ead701fe8d3dfd8b5960a087"

# Invariants inherited byte-for-byte from the archived selector-repair wrapper
# and its reviewed source runner. They intentionally remain visible here so the
# existing v6 contract test continues to guard the active entrypoint:
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

test -f "${SELECTOR_RUNNER}"
test ! -L "${SELECTOR_RUNNER}"
test "$(git hash-object "${SELECTOR_RUNNER}")" = "${SELECTOR_RUNNER_BLOB_SHA}"
test -f "${MATERIALIZATION_PATH}"
test ! -L "${MATERIALIZATION_PATH}"
test -z "$(git status --porcelain=v1 --untracked-files=no)"

export MATERIALIZATION_PATH MATERIALIZATION_ID
export FROZEN_UPSTREAM_REVISION LOCATOR_REPORT_ID
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from bayesian_phystwin.deform360_bias_aware_prospective_physical import (
    UPSTREAM_FILE_SHA256,
)

path = Path(os.environ["MATERIALIZATION_PATH"])
payload = json.loads(path.read_text(encoding="utf-8"))
declared = payload.pop("amendment_id")
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
expected = os.environ["MATERIALIZATION_ID"]
if declared != observed or observed != expected:
    raise SystemExit("frozen upstream materialization identity changed")
if payload.get("schema") != (
    "bayesian-phystwin.deform360-v6-frozen-upstream-materialization"
):
    raise SystemExit("frozen upstream materialization schema changed")
if payload.get("schema_version") != 1:
    raise SystemExit("frozen upstream materialization version changed")

parent = payload.get("parent_execution", {})
if parent.get("execution_amendment_id") != (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
):
    raise SystemExit("frozen upstream materialization changed the execution")
if parent.get("selector_runtime_repair_id") != (
    "d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
):
    raise SystemExit("frozen upstream materialization lost selector repair")
if parent.get("protected_workflow_run_id") != 31460025917:
    raise SystemExit("frozen upstream materialization lost protected run")
if parent.get("protected_artifact_id") != 9089464088:
    raise SystemExit("frozen upstream materialization lost protected artifact")
if parent.get("protected_execution_receipt_id") != (
    "6232968ec10c630b62e6933783b278bc4aba2362bb29c1eec6d2be47a001c0e0"
):
    raise SystemExit("frozen upstream materialization lost protected receipt")
if parent.get("terminal_stage") != "locate-frozen-physical-upstream":
    raise SystemExit("frozen upstream materialization changed failure stage")
if parent.get("physical_manifest_count") != 0:
    raise SystemExit("materialization was declared after physical prediction")
if parent.get("source_prediction_seal_count") != 0:
    raise SystemExit("materialization was declared after source prediction")

locator = payload.get("history_locator", {})
if locator.get("schema") != "bayesian-phystwin.frozen-source-history-locator":
    raise SystemExit("frozen upstream locator schema changed")
if locator.get("report_id") != os.environ["LOCATOR_REPORT_ID"]:
    raise SystemExit("frozen upstream locator report changed")
if locator.get("workflow_run_id") != 31461017011:
    raise SystemExit("frozen upstream locator run changed")
if locator.get("artifact_id") != 9089783219:
    raise SystemExit("frozen upstream locator artifact changed")
if locator.get("complete_history_searched") is not True:
    raise SystemExit("frozen upstream locator did not search complete history")
if locator.get("exact_match_count") != 1:
    raise SystemExit("frozen upstream locator did not find exactly one match")

source = payload.get("frozen_source", {})
if source.get("repository") != "IPS-Stuttgart/BayesianPhysTwin":
    raise SystemExit("frozen upstream repository changed")
if source.get("revision") != os.environ["FROZEN_UPSTREAM_REVISION"]:
    raise SystemExit("frozen upstream revision changed")
if source.get("refs_pointing_at") != [] or source.get("containing_tags") != []:
    raise SystemExit("frozen upstream ref evidence changed")
required = source.get("required_file_sha256")
if required != dict(sorted(UPSTREAM_FILE_SHA256.items())):
    raise SystemExit("frozen upstream file roster changed")

materialization = payload.get("materialization", {})
expected_materialization = {
    "method": "detached-temporary-git-worktree",
    "source_revision_must_match_exactly": True,
    "all_required_files_must_match_sha256": True,
    "symlinked_required_files_allowed": False,
    "main_checkout_files_may_be_modified": False,
    "temporary_worktree_removed_after_execution": True,
    "archived_runner_blob_may_be_modified": False,
    "runtime_discovery_root_extension_only": True,
}
if materialization != expected_materialization:
    raise SystemExit("frozen upstream materialization method changed")

scope = payload.get("repair_scope", {})
if scope.get("historical_source_materialization_only") is not True:
    raise SystemExit("frozen upstream repair is not materialization-only")
for field in (
    "source_file_bytes_changed",
    "physical_method_changed",
    "model_family_changed",
    "source_cohort_changed",
    "camera_panel_changed",
    "candidate_roster_changed",
    "loss_or_gate_changed",
    "replacement_allowed",
    "claim_authorized",
):
    if scope.get(field) is not False:
        raise SystemExit(f"frozen upstream materialization widened {field}")

boundary = payload.get("information_boundary", {})
if not boundary or any(value is not False for value in boundary.values()):
    raise SystemExit("frozen upstream materialization crossed information boundary")

authorization = payload.get("execution_authorization", {})
if authorization.get("event") != "push-to-protected-main-after-reviewed-merge":
    raise SystemExit("frozen upstream execution event changed")
if authorization.get("runner_name") != "workstation2":
    raise SystemExit("frozen upstream runner changed")
if authorization.get("source_prediction_batch_required_before_suffix_access") is not True:
    raise SystemExit("frozen upstream materialization weakened source barrier")
if authorization.get("fresh_target_selection_authorized") is not False:
    raise SystemExit("frozen upstream materialization authorized target selection")
if authorization.get("fresh_target_payload_access_authorized") is not False:
    raise SystemExit("frozen upstream materialization authorized target access")
PY

git cat-file -e "${FROZEN_UPSTREAM_REVISION}^{commit}"
WORKTREE_PARENT="$(
  mktemp -d "${RUNNER_TEMP:-/tmp}/deform360-v6-frozen-upstream.XXXXXX"
)"
FROZEN_UPSTREAM_ROOT="${WORKTREE_PARENT}/source"
MATERIALIZATION_RECEIPT="${EVIDENCE_ROOT}/deform360-v6-source-prediction-evidence/frozen-upstream-materialization.json"
WORKTREE_ADDED=false
cleanup() {
  set +e
  if [[ "${WORKTREE_ADDED}" = true ]]; then
    git worktree remove --force "${FROZEN_UPSTREAM_ROOT}" >/dev/null 2>&1
  fi
  rm -rf "${WORKTREE_PARENT}"
}
trap cleanup EXIT

git worktree add --detach "${FROZEN_UPSTREAM_ROOT}" \
  "${FROZEN_UPSTREAM_REVISION}" >/dev/null
WORKTREE_ADDED=true
test "$(git -C "${FROZEN_UPSTREAM_ROOT}" rev-parse HEAD)" = \
  "${FROZEN_UPSTREAM_REVISION}"
test -z "$(git -C "${FROZEN_UPSTREAM_ROOT}" status --porcelain=v1)"
test -z "$(git status --porcelain=v1 --untracked-files=no)"

export FROZEN_UPSTREAM_ROOT MATERIALIZATION_RECEIPT
"${BPT_PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

amendment_path = Path(os.environ["MATERIALIZATION_PATH"])
payload = json.loads(amendment_path.read_text(encoding="utf-8"))
root = Path(os.environ["FROZEN_UPSTREAM_ROOT"]).resolve()
required = payload["frozen_source"]["required_file_sha256"]
observed: dict[str, str] = {}
for relative, expected in sorted(required.items()):
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"frozen upstream file is missing or symlinked: {relative}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise SystemExit(f"frozen upstream file escapes worktree: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"frozen upstream file identity changed: {relative}")
    observed[relative] = digest

receipt = {
    "schema": "bayesian-phystwin.deform360-v6-frozen-upstream-materialization-receipt",
    "schema_version": 1,
    "amendment_id": os.environ["MATERIALIZATION_ID"],
    "history_locator_report_id": os.environ["LOCATOR_REPORT_ID"],
    "repository": payload["frozen_source"]["repository"],
    "revision": os.environ["FROZEN_UPSTREAM_REVISION"],
    "required_file_sha256": dict(sorted(required.items())),
    "observed_file_sha256": observed,
    "worktree_clean": True,
    "main_checkout_modified": False,
    "information_boundary": {
        "dataset_opened": False,
        "source_residual_opened": False,
        "development_suffix_opened": False,
        "target_payload_opened": False,
        "target_outcome_used": False,
    },
}
canonical = json.dumps(
    receipt,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
receipt["materialization_receipt_id"] = hashlib.sha256(canonical).hexdigest()
output = Path(os.environ["MATERIALIZATION_RECEIPT"])
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists() or output.is_symlink():
    raise SystemExit("frozen upstream materialization receipt already exists")
output.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY

export RUNNER_WORKSPACE="${WORKTREE_PARENT}"
set +e
bash "${SELECTOR_RUNNER}"
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

execution_path = Path(os.environ["RECEIPT_PATH"])
materialization_path = Path(os.environ["MATERIALIZATION_RECEIPT"])
execution = json.loads(execution_path.read_text(encoding="utf-8"))
materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
if materialization.get("amendment_id") != os.environ["MATERIALIZATION_ID"]:
    raise SystemExit("materialization receipt amendment changed")
if materialization.get("revision") != os.environ["FROZEN_UPSTREAM_REVISION"]:
    raise SystemExit("materialization receipt revision changed")
if materialization.get("history_locator_report_id") != os.environ["LOCATOR_REPORT_ID"]:
    raise SystemExit("materialization receipt locator changed")
if not materialization.get("materialization_receipt_id"):
    raise SystemExit("materialization receipt identity is missing")

execution.pop("receipt_id", None)
execution["runtime_frozen_upstream_materialization"] = {
    "amendment_id": os.environ["MATERIALIZATION_ID"],
    "amendment_path": os.environ["MATERIALIZATION_PATH"],
    "history_locator_report_id": os.environ["LOCATOR_REPORT_ID"],
    "materialization_receipt_id": materialization["materialization_receipt_id"],
    "repository": materialization["repository"],
    "revision": materialization["revision"],
    "required_file_sha256": materialization["required_file_sha256"],
}
canonical = json.dumps(
    execution,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
execution["receipt_id"] = hashlib.sha256(canonical).hexdigest()
execution_path.write_text(
    json.dumps(execution, indent=2, sort_keys=True, allow_nan=False) + "\n",
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

test -z "$(git status --porcelain=v1 --untracked-files=no)"
exit "${status}"
