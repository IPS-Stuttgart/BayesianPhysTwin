from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.ci import deform360_v6_prepared_inventory_repair as repair

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "prepared_inventory_repair.json"
)
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
SCIENCE_RUNNER = ROOT / (
    "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
SELECTOR_WRAPPER = ROOT / (
    "scripts/ci/archive/"
    "run_deform360_v6_source_prediction_evidence_selector_repair_v3.sh"
)

REPAIR_ID = "a678606c2ceb84120a65326d083bae35cc25cf5dc2f092449801fa0134a63336"
AUTHORITATIVE_INVENTORY_ID = (
    "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
)
AUTHORITATIVE_FILE_SHA256 = (
    "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
)
NORMALIZED_PAYLOAD_SHA256 = (
    "4d6e0dd4e35223e7cbd68cd7e9c4dac50bb3e46f2562f8e4879d8d4acc1f7bb6"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def test_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == REPAIR_ID == hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    assert payload["schema"] == (
        "bayesian-phystwin.deform360-v6-prepared-source-inventory-repair"
    )
    assert payload["schema_version"] == 1
    assert payload["superseded_execution_amendment_id"] == (
        "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
    )

    authoritative = payload["authoritative_inventory"]
    assert authoritative["workflow_run_id"] == 31_272_512_658
    assert authoritative["artifact_id"] == 9_026_043_628
    assert authoritative["artifact_digest"] == (
        "sha256:d0041af0ba0cfe6e5c5bd4008c47adb3ed4cf0cf0f6754eff67a238e746c7a86"
    )
    assert authoritative["implementation_revision"] == (
        "e190c94014e6024e324d860618662526af6ea682"
    )
    assert authoritative["inventory_id"] == AUTHORITATIVE_INVENTORY_ID
    assert authoritative["file_sha256"] == AUTHORITATIVE_FILE_SHA256
    assert authoritative["normalized_payload_sha256"] == NORMALIZED_PAYLOAD_SHA256
    assert authoritative["object_count"] == 10
    assert authoritative["camera_view_count"] == 324

    failed = payload["failed_execution_evidence"]
    assert failed["workflow_run_id"] == 31_462_653_379
    assert failed["artifact_id"] == 9_090_402_942
    assert failed["execution_receipt_id"] == (
        "a62ead70994b330d3eecf8d35afe6baa1275c428f4c90c4e18157aac55f3733f"
    )
    assert failed["terminal_stage"] == "physical-source:026-sock-cloth-ep0007"
    assert failed["error"] == "prepared source inventory file changed"
    assert failed["physical_manifest_count"] == 0
    assert failed["source_prediction_seal_count"] == 0

    correction = payload["correction"]
    assert correction["candidate_allowed_difference_fields"] == [
        "implementation_revision",
        "inventory_id",
    ]
    assert correction["candidate_normalized_payload_sha256"] == (
        NORMALIZED_PAYLOAD_SHA256
    )
    assert correction["restored_file_sha256"] == AUTHORITATIVE_FILE_SHA256
    assert correction["restored_inventory_id"] == AUTHORITATIVE_INVENTORY_ID

    assert payload["information_boundary"]
    assert not any(payload["information_boundary"].values())
    scope = payload["repair_scope"]
    assert scope["runtime_artifact_reconstruction_only"] is True
    assert all(
        scope[field] is False
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
        )
    )


def test_repair_validator_accepts_only_registered_record() -> None:
    validated = repair.validate_repair(REPAIR)
    assert validated["semantics"] == (
        "restore-authoritative-retained-source-inventory-before-"
        "physical-materialization-v1"
    )


def test_restore_rejects_payload_drift_beyond_provenance_fields(
    tmp_path: Path,
) -> None:
    runtime_revision = "1" * 40
    candidate: dict[str, Any] = {
        "schema": "fixture",
        "schema_version": 1,
        "implementation_revision": runtime_revision,
        "object_count": 10,
        "objects": [],
    }
    candidate["inventory_id"] = hashlib.sha256(_canonical_bytes(candidate)).hexdigest()
    path = tmp_path / "prepared-source-inventory.json"
    path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="payload changed beyond provenance fields",
    ):
        repair.restore_inventory(path, runtime_revision=runtime_revision)


def test_runner_patching_preserves_immutable_sources(tmp_path: Path) -> None:
    science_target = tmp_path / "science.sh"
    selector_target = tmp_path / "selector.sh"
    original_science = SCIENCE_RUNNER.read_bytes()
    original_selector = SELECTOR_WRAPPER.read_bytes()

    result = repair.patch_runners(
        science_source=SCIENCE_RUNNER,
        selector_source=SELECTOR_WRAPPER,
        science_target=science_target,
        selector_target=selector_target,
    )

    assert SCIENCE_RUNNER.read_bytes() == original_science
    assert SELECTOR_WRAPPER.read_bytes() == original_selector
    assert _git_blob_sha(SCIENCE_RUNNER) == repair.SCIENCE_RUNNER_BLOB_SHA
    patched_science = science_target.read_text(encoding="utf-8")
    patched_selector = selector_target.read_text(encoding="utf-8")
    assert patched_science.count(
        'set_stage "restore-authoritative-prepared-source-inventory"'
    ) == 1
    assert "deform360_v6_prepared_inventory_repair.py" in patched_science
    assert 'ARCHIVED_RUNNER="${PATCHED_SCIENCE_RUNNER:?' in patched_selector
    assert result["patched_science_blob_sha"] in patched_selector
    assert result["patched_selector_blob_sha"] == _git_blob_sha(selector_target)


def test_receipt_records_authoritative_inventory_replay(tmp_path: Path) -> None:
    path = tmp_path / "execution-receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "fixture-receipt",
                "status": "source-technical-failure-retained",
                "receipt_id": "0" * 64,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = repair.augment_receipt(path)
    assert receipt["runtime_prepared_source_inventory_repair_id"] == REPAIR_ID
    inventory = cast(dict[str, Any], receipt["runtime_prepared_source_inventory"])
    assert inventory["inventory_id"] == AUTHORITATIVE_INVENTORY_ID
    assert inventory["file_sha256"] == AUTHORITATIVE_FILE_SHA256
    declared = receipt.pop("receipt_id")
    assert declared == hashlib.sha256(_canonical_bytes(receipt)).hexdigest()


def test_active_runner_applies_repair_without_mutating_registered_sources() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert f'INVENTORY_HELPER_BLOB_SHA="{_git_blob_sha(ROOT / repair.__file__)}"' in text
    assert "validate-repair" in text
    assert "patch-runners" in text
    assert "augment-receipt" in text
    assert "PATCHED_SCIENCE_RUNNER" in text
    assert "PATCHED_SELECTOR_WRAPPER" in text
    assert "--restore-prepared-inventory" in text
    assert "sed -i" not in text
    assert "git checkout" not in text
