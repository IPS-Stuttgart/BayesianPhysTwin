#!/usr/bin/env python3
"""Validate and apply the Deform360 v6 prepared-inventory runtime repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

REPAIR_PATH = Path(
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_prepared_inventory_repair.json"
)
REPAIR_ID = "a678606c2ceb84120a65326d083bae35cc25cf5dc2f092449801fa0134a63336"
AUTHORITATIVE_IMPLEMENTATION_REVISION = "e190c94014e6024e324d860618662526af6ea682"
AUTHORITATIVE_INVENTORY_ID = (
    "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
)
AUTHORITATIVE_FILE_SHA256 = (
    "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
)
AUTHORITATIVE_NORMALIZED_SHA256 = (
    "4d6e0dd4e35223e7cbd68cd7e9c4dac50bb3e46f2562f8e4879d8d4acc1f7bb6"
)
AUTHORITATIVE_RUN_ID = 31272512658
AUTHORITATIVE_ARTIFACT_ID = 9026043628
AUTHORITATIVE_ARTIFACT_DIGEST = (
    "sha256:d0041af0ba0cfe6e5c5bd4008c47adb3ed4cf0cf0f6754eff67a238e746c7a86"
)
FAILED_RUN_ID = 31462653379
FAILED_RECEIPT_ID = (
    "a62ead70994b330d3eecf8d35afe6baa1275c428f4c90c4e18157aac55f3733f"
)
SCIENCE_RUNNER_BLOB_SHA = "42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, name=name))


def validate_repair(path: Path = REPAIR_PATH) -> dict[str, Any]:
    payload = _load_json(path, name="v6 prepared-inventory repair")
    declared = payload.pop("repair_id", None)
    observed = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if declared != observed or observed != REPAIR_ID:
        raise ValueError("v6 prepared-inventory repair identity changed")
    if payload.get("schema") != (
        "bayesian-phystwin.deform360-v6-prepared-source-inventory-repair"
    ):
        raise ValueError("v6 prepared-inventory repair schema changed")
    if payload.get("schema_version") != 1:
        raise ValueError("v6 prepared-inventory repair version changed")
    if payload.get("superseded_execution_amendment_id") != (
        "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
    ):
        raise ValueError("v6 prepared-inventory execution amendment changed")

    authoritative = _mapping(
        payload.get("authoritative_inventory"), name="authoritative_inventory"
    )
    expected = {
        "workflow_run_id": AUTHORITATIVE_RUN_ID,
        "artifact_id": AUTHORITATIVE_ARTIFACT_ID,
        "artifact_digest": AUTHORITATIVE_ARTIFACT_DIGEST,
        "file_sha256": AUTHORITATIVE_FILE_SHA256,
        "inventory_id": AUTHORITATIVE_INVENTORY_ID,
        "implementation_revision": AUTHORITATIVE_IMPLEMENTATION_REVISION,
        "normalized_payload_sha256": AUTHORITATIVE_NORMALIZED_SHA256,
        "object_count": 10,
        "camera_view_count": 324,
    }
    for key, expected_value in expected.items():
        if authoritative.get(key) != expected_value:
            raise ValueError(f"v6 authoritative inventory {key} changed")

    correction = _mapping(payload.get("correction"), name="correction")
    if correction.get("candidate_allowed_difference_fields") != [
        "implementation_revision",
        "inventory_id",
    ]:
        raise ValueError("v6 prepared-inventory allowed differences changed")
    if correction.get("candidate_normalized_payload_sha256") != (
        AUTHORITATIVE_NORMALIZED_SHA256
    ):
        raise ValueError("v6 prepared-inventory normalized identity changed")
    if correction.get("restored_file_sha256") != AUTHORITATIVE_FILE_SHA256:
        raise ValueError("v6 prepared-inventory restored digest changed")
    if correction.get("restored_inventory_id") != AUTHORITATIVE_INVENTORY_ID:
        raise ValueError("v6 prepared-inventory restored ID changed")

    failed = _mapping(
        payload.get("failed_execution_evidence"), name="failed_execution_evidence"
    )
    if failed.get("workflow_run_id") != FAILED_RUN_ID:
        raise ValueError("v6 prepared-inventory repair lost failed run")
    if failed.get("execution_receipt_id") != FAILED_RECEIPT_ID:
        raise ValueError("v6 prepared-inventory repair lost failed receipt")
    if failed.get("physical_manifest_count") != 0:
        raise ValueError("v6 prepared-inventory repair was declared after prediction")
    if failed.get("source_prediction_seal_count") != 0:
        raise ValueError("v6 prepared-inventory repair was declared after source prediction")

    scope = _mapping(payload.get("repair_scope"), name="repair_scope")
    if scope.get("runtime_artifact_reconstruction_only") is not True:
        raise ValueError("v6 prepared-inventory repair scope changed")
    for field in (
        "model_family_changed",
        "model_size_changed",
        "source_payload_changed",
        "source_cohort_changed",
        "camera_panel_changed",
        "candidate_roster_changed",
        "loss_or_gate_changed",
        "replacement_allowed",
        "claim_authorized",
    ):
        if scope.get(field) is not False:
            raise ValueError(f"v6 prepared-inventory repair widened {field}")

    boundary = _mapping(payload.get("information_boundary"), name="information_boundary")
    if not boundary or any(value is not False for value in boundary.values()):
        raise ValueError("v6 prepared-inventory repair crossed information boundary")
    authorization = _mapping(
        payload.get("execution_authorization"), name="execution_authorization"
    )
    if authorization.get("event") != "push-to-protected-main-after-reviewed-merge":
        raise ValueError("v6 prepared-inventory execution event changed")
    if authorization.get("runner_name") != "workstation2":
        raise ValueError("v6 prepared-inventory runner changed")
    if authorization.get("source_prediction_batch_required_before_suffix_access") is not True:
        raise ValueError("v6 prepared-inventory repair weakened prediction barrier")
    if authorization.get("fresh_target_selection_authorized") is not False:
        raise ValueError("v6 prepared-inventory repair authorized target selection")
    if authorization.get("fresh_target_payload_access_authorized") is not False:
        raise ValueError("v6 prepared-inventory repair authorized target access")
    return payload


def restore_inventory(path: Path, *, runtime_revision: str) -> dict[str, str]:
    validate_repair()
    payload = _load_json(path, name="prepared-source inventory candidate")
    if payload.get("implementation_revision") != runtime_revision:
        raise ValueError("prepared-source inventory runtime revision changed")
    declared = payload.get("inventory_id")
    if not isinstance(declared, str):
        raise ValueError("prepared-source inventory candidate ID is invalid")
    candidate_without_id = dict(payload)
    candidate_without_id.pop("inventory_id")
    if hashlib.sha256(_canonical_bytes(candidate_without_id)).hexdigest() != declared:
        raise ValueError("prepared-source inventory candidate identity changed")

    normalized = dict(payload)
    normalized.pop("inventory_id")
    normalized.pop("implementation_revision")
    normalized_sha = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    if normalized_sha != AUTHORITATIVE_NORMALIZED_SHA256:
        raise ValueError(
            "prepared-source inventory payload changed beyond provenance fields"
        )

    restored = dict(payload)
    restored.pop("inventory_id")
    restored["implementation_revision"] = AUTHORITATIVE_IMPLEMENTATION_REVISION
    restored_id = hashlib.sha256(_canonical_bytes(restored)).hexdigest()
    if restored_id != AUTHORITATIVE_INVENTORY_ID:
        raise ValueError("restored prepared-source inventory ID changed")
    restored["inventory_id"] = restored_id
    content = (
        json.dumps(restored, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if hashlib.sha256(content).hexdigest() != AUTHORITATIVE_FILE_SHA256:
        raise ValueError("restored prepared-source inventory file changed")

    temporary = path.with_name(path.name + ".authoritative.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("prepared-source inventory temporary path is occupied")
    temporary.write_bytes(content)
    if temporary.is_symlink():
        raise ValueError("prepared-source inventory temporary path became a symlink")
    os.replace(temporary, path)
    return {
        "candidate_inventory_id": declared,
        "normalized_payload_sha256": normalized_sha,
        "restored_file_sha256": AUTHORITATIVE_FILE_SHA256,
        "restored_inventory_id": restored_id,
    }


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def patch_runners(
    *,
    science_source: Path,
    selector_source: Path,
    science_target: Path,
    selector_target: Path,
) -> dict[str, str]:
    if _git_blob_sha(science_source.read_bytes()) != SCIENCE_RUNNER_BLOB_SHA:
        raise ValueError("immutable v6 science runner changed")
    science = science_source.read_text(encoding="utf-8")
    anchor = '''  > "${LOG_ROOT}/prepared-source-inventory.log" 2>&1

set_stage "locate-frozen-sam2-checkpoint"
'''
    replacement = '''  > "${LOG_ROOT}/prepared-source-inventory.log" 2>&1

set_stage "restore-authoritative-prepared-source-inventory"
"${BPT_PYTHON}" scripts/ci/deform360_v6_prepared_inventory_repair.py \\
  restore \\
  --inventory "${RUN_ROOT}/prepared-source-inventory.json" \\
  --runtime-revision "${BPT_SOURCE_SHA}" \\
  > "${LOG_ROOT}/prepared-source-inventory-replay.log" 2>&1

set_stage "locate-frozen-sam2-checkpoint"
'''
    if science.count(anchor) != 1:
        raise ValueError("prepared-inventory replay patch anchor changed")
    patched_science = science.replace(anchor, replacement, 1)
    science_target.write_text(patched_science, encoding="utf-8")
    science_target.chmod(0o700)
    science_blob = _git_blob_sha(science_target.read_bytes())

    selector = selector_source.read_text(encoding="utf-8")
    old_path = (
        'ARCHIVED_RUNNER="scripts/ci/archive/'
        'run_deform360_v6_source_prediction_evidence_v2.sh"'
    )
    new_path = (
        'ARCHIVED_RUNNER="${PATCHED_SCIENCE_RUNNER:?'
        'PATCHED_SCIENCE_RUNNER is required}"'
    )
    old_blob = f'ARCHIVED_RUNNER_BLOB_SHA="{SCIENCE_RUNNER_BLOB_SHA}"'
    new_blob = f'ARCHIVED_RUNNER_BLOB_SHA="{science_blob}"'
    if selector.count(old_path) != 1 or selector.count(old_blob) != 1:
        raise ValueError("selector-wrapper delegation anchor changed")
    selector_target.write_text(
        selector.replace(old_path, new_path, 1).replace(old_blob, new_blob, 1),
        encoding="utf-8",
    )
    selector_target.chmod(0o700)
    return {
        "patched_science_blob_sha": science_blob,
        "patched_selector_blob_sha": _git_blob_sha(selector_target.read_bytes()),
    }


def augment_receipt(path: Path) -> dict[str, Any]:
    validate_repair()
    receipt = _load_json(path, name="v6 execution receipt")
    receipt.pop("receipt_id", None)
    receipt["runtime_prepared_source_inventory_repair_id"] = REPAIR_ID
    receipt["runtime_prepared_source_inventory_repair_path"] = REPAIR_PATH.as_posix()
    receipt["runtime_prepared_source_inventory"] = {
        "workflow_run_id": AUTHORITATIVE_RUN_ID,
        "artifact_id": AUTHORITATIVE_ARTIFACT_ID,
        "artifact_digest": AUTHORITATIVE_ARTIFACT_DIGEST,
        "implementation_revision": AUTHORITATIVE_IMPLEMENTATION_REVISION,
        "inventory_id": AUTHORITATIVE_INVENTORY_ID,
        "file_sha256": AUTHORITATIVE_FILE_SHA256,
        "normalized_payload_sha256": AUTHORITATIVE_NORMALIZED_SHA256,
        "operation": (
            "reconstruct-exact-authoritative-json-from-identical-normalized-payload"
        ),
    }
    receipt["receipt_id"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-repair")
    restore = subparsers.add_parser("restore")
    restore.add_argument("--inventory", type=Path, required=True)
    restore.add_argument("--runtime-revision", required=True)
    patch = subparsers.add_parser("patch-runners")
    patch.add_argument("--science-source", type=Path, required=True)
    patch.add_argument("--selector-source", type=Path, required=True)
    patch.add_argument("--science-target", type=Path, required=True)
    patch.add_argument("--selector-target", type=Path, required=True)
    receipt = subparsers.add_parser("augment-receipt")
    receipt.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-repair":
        result: object = {"repair_id": REPAIR_ID, "valid": bool(validate_repair())}
    elif args.command == "restore":
        result = restore_inventory(args.inventory, runtime_revision=args.runtime_revision)
    elif args.command == "patch-runners":
        result = patch_runners(
            science_source=args.science_source,
            selector_source=args.selector_source,
            science_target=args.science_target,
            selector_target=args.selector_target,
        )
    else:
        result = augment_receipt(args.receipt)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
