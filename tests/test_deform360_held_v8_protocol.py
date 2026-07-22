from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any

import pytest

import bayesian_phystwin.deform360_frame_zero_assets as frame_zero_assets
import bayesian_phystwin.deform360_held_v8_protocol as protocol
import bayesian_phystwin.deform360_held_v8_query_artifacts as query_artifacts


def _write_json(path: Path, value: dict[str, Any], *, seal: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if seal:
        path.chmod(0o400)


def _bound_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _artifact(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["artifact_sha256"] = protocol.held_artifact_sha256(value)
    _write_json(path, value)
    return value


def _git(code: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(code), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _seal_tree(root: Path) -> None:
    paths: list[Path] = []
    for current, directories, files in os.walk(root, topdown=False):
        current_path = Path(current)
        paths.extend(current_path / name for name in files)
        paths.extend(current_path / name for name in directories)
    paths.append(root)
    for path in paths:
        path.chmod(0o500 if path.is_dir() else 0o400)


def _sealed_test_repository(root: Path) -> tuple[Path, dict[str, Any]]:
    stage = root / "code-stage"
    stage.mkdir()
    _git(stage, "init", "--quiet")
    _git(stage, "config", "user.email", "held-test@example.invalid")
    _git(stage, "config", "user.name", "Held Test")
    (stage / ".gitignore").write_text("ignored-runtime.py\n", encoding="utf-8")
    (stage / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(stage, "add", ".gitignore", "tracked.py")
    _git(stage, "commit", "--quiet", "-m", "test fixture")
    _git(stage, "checkout", "--quiet", "--detach", "HEAD")
    head = _git(stage, "rev-parse", "HEAD").decode("ascii").strip()
    deployed_code = root / f"code-{head}"
    stage.rename(deployed_code)
    _seal_tree(deployed_code)
    return deployed_code, protocol._attempt3_repository_binding(deployed_code)


def _attempt3_lineage_fixture(
    lineage: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    archive = lineage / "held-v8-attempt-3-withdrawn-postbarrier"
    archive.mkdir()
    deployed_code, deployed = _sealed_test_repository(archive)
    inventory_directory = archive / "sealed-evidence"
    inventory_directory.mkdir()
    inventory_file = inventory_directory / "payload.bin"
    inventory_payload = b"attempt-3-sealed-payload\n"
    inventory_file.write_bytes(inventory_payload)
    inventory_file.chmod(0o400)
    inventory_directory.chmod(0o500)
    report_path = archive / "execution-withdrawal-postbarrier-attempt3.json"
    pointer_path = lineage / "held-v8-attempt-3-withdrawal-pointer.json"
    completion_path = lineage / "held-v8-attempt-3-withdrawal-integrity-completion.json"
    inventory_rows = [
        {
            "path": "sealed-evidence",
            "type": "directory",
            "mode_octal": "0500",
        },
        {
            "path": "sealed-evidence/payload.bin",
            "type": "file",
            "mode_octal": "0400",
            "size_bytes": len(inventory_payload),
            "sha256": hashlib.sha256(inventory_payload).hexdigest(),
        },
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            {"rows": inventory_rows},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    inventory_count = len(inventory_rows)
    operator = {"path": "/tmp/attempt3-operator.py", "sha256": "b" * 64}
    report = _artifact(
        report_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV8Attempt3PostBarrierWithdrawalReport",
            "protocol_id": "deform360-held-online-belief-v8",
            "execution_attempt": 3,
            "status": "withdrawn-postbarrier-before-queried-prediction-or-score",
            "disposition": (
                "WITHDRAWN_AFTER_TARGET_AND_X0_BEFORE_ANY_QUERIED_PREDICTION_"
                "SEAL_OR_SCORE"
            ),
            "immutable_archive_path": str(archive),
            "executed_withdrawal_operator_source": operator,
            "deployed_code": deployed,
            "execution_boundary": {
                "online_prediction_seal_count": 15,
                "frozen_field_manifest_count": 15,
                "official_target_archive_count": 1,
                "official_x0_archive_count": 1,
                "queried_prediction_seal_count": 0,
                "score_evidence_count": 0,
                "gate_decision_count": 0,
                "confirmation_lock_count": 0,
            },
            "information_boundary": {
                "first_complete_cohort_barrier_crossed": True,
                "queried_prediction_created_or_read": False,
                "score_created_or_read": False,
                "gate_decision_created_or_read": False,
                "confirmation_created_or_read": False,
            },
        },
    )
    report_record = _bound_file(report_path)
    shared = {
        "archive_path": str(archive),
        "archive_root_mode_octal": "0500",
        "archive_fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": inventory_sha256,
        "postseal_noncode_entry_count": inventory_count,
        "withdrawal_report_path": str(report_path),
        "withdrawal_report_size_bytes": report_record["size_bytes"],
        "withdrawal_report_file_sha256": report_record["sha256"],
        "withdrawal_report_artifact_sha256": report["artifact_sha256"],
        "deployed_code": deployed,
        "independent_post_rename_integrity_verified": True,
    }
    completion = _artifact(
        completion_path,
        {
            "schema_version": 1,
            "artifact_kind": ("Deform360HeldV8Attempt3WithdrawalIntegrityCompletion"),
            "protocol_id": "deform360-held-online-belief-v8",
            "execution_attempt": 3,
            "status": "withdrawal-integrity-complete",
            "disposition": (
                "WITHDRAWN_AFTER_TARGET_AND_X0_BEFORE_ANY_QUERIED_PREDICTION_"
                "SEAL_OR_SCORE"
            ),
            **shared,
            "executed_withdrawal_operator_source": operator,
            "pointer_contract": {
                "path": str(pointer_path),
                "artifact_kind": "Deform360HeldV8Attempt3WithdrawalPointer",
                "pointer_must_bind_this_completion": True,
                "completion_does_not_predict_pointer_hash_to_avoid_circularity": True,
            },
        },
    )
    completion_record = _bound_file(completion_path)
    pointer = _artifact(
        pointer_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV8Attempt3WithdrawalPointer",
            "protocol_id": "deform360-held-online-belief-v8",
            "execution_attempt": 3,
            "status": "withdrawn-postbarrier-before-queried-prediction-or-score",
            "disposition": (
                "WITHDRAWN_AFTER_TARGET_AND_X0_BEFORE_ANY_QUERIED_PREDICTION_"
                "SEAL_OR_SCORE"
            ),
            **shared,
            "executed_withdrawal_operator_source": operator,
            "withdrawal_integrity_completion": {
                "path": str(completion_path),
                "mode_octal": "0400",
                "size_bytes": completion_record["size_bytes"],
                "file_sha256": completion_record["sha256"],
                "artifact_sha256": completion["artifact_sha256"],
            },
            "active_held_v8_root_absent_after_archive": True,
            "queried_prediction_seal_count": 0,
            "score_evidence_count": 0,
            "gate_decision_count": 0,
            "confirmation_accessed": False,
        },
    )
    pointer_record = _bound_file(pointer_path)
    archive.chmod(0o500)
    metadata_inventory = protocol._observed_attempt3_noncode_metadata_inventory(
        archive,
        deployed_code=deployed_code,
    )

    replacements = {
        "ATTEMPT3_ARCHIVE_PATH": archive,
        "ATTEMPT3_WITHDRAWAL_REPORT_PATH": report_path,
        "ATTEMPT3_WITHDRAWAL_POINTER_PATH": pointer_path,
        "ATTEMPT3_WITHDRAWAL_INTEGRITY_COMPLETION_PATH": completion_path,
        "ATTEMPT3_WITHDRAWAL_REPORT_FILE_SHA256": report_record["sha256"],
        "ATTEMPT3_WITHDRAWAL_REPORT_ARTIFACT_SHA256": report["artifact_sha256"],
        "ATTEMPT3_WITHDRAWAL_COMPLETION_FILE_SHA256": completion_record["sha256"],
        "ATTEMPT3_WITHDRAWAL_COMPLETION_ARTIFACT_SHA256": completion["artifact_sha256"],
        "ATTEMPT3_WITHDRAWAL_POINTER_FILE_SHA256": pointer_record["sha256"],
        "ATTEMPT3_WITHDRAWAL_POINTER_ARTIFACT_SHA256": pointer["artifact_sha256"],
        "ATTEMPT3_ARCHIVE_INVENTORY_SHA256": inventory_sha256,
        "ATTEMPT3_ARCHIVE_ENTRY_COUNT": inventory_count,
        "ATTEMPT3_ARCHIVE_METADATA_INVENTORY_SHA256": metadata_inventory[
            "metadata_inventory_sha256"
        ],
    }
    for name, value in replacements.items():
        monkeypatch.setattr(protocol, name, value)
    archive_integrity = {
        "path": str(archive),
        "root_mode_octal": "0500",
        "fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": inventory_sha256,
        "postseal_noncode_entry_count": inventory_count,
    }
    return {
        "archive": archive,
        "report_path": report_path,
        "pointer_path": pointer_path,
        "completion_path": completion_path,
        "report_record": report_record,
        "pointer_record": pointer_record,
        "completion_record": completion_record,
        "archive_integrity": archive_integrity,
    }


def _lock_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    disclosed_payloads = {
        "v7_outcome_withdrawal_report": b"withdrawn-v7\n",
        "retired_case_official_target": b"retired-target\n",
        "retired_case_online_prediction": b"retired-online\n",
        "retired_case_online_prediction_seal": b"retired-seal\n",
    }
    disclosed_paths: dict[str, Path] = {}
    disclosed_specs: dict[str, tuple[int, str]] = {}
    for name, payload in disclosed_payloads.items():
        path = lineage / f"{name}.bin"
        path.write_bytes(payload)
        path.chmod(0o400)
        disclosed_paths[name] = path
        disclosed_specs[name] = (len(payload), hashlib.sha256(payload).hexdigest())
    withdrawal = disclosed_paths["v7_outcome_withdrawal_report"]
    monkeypatch.setattr(protocol, "V7_DISCLOSED_FILE_SPECS", disclosed_specs)
    monkeypatch.setattr(
        protocol,
        "V7_WITHDRAWAL_REPORT_FILE_SHA256",
        disclosed_specs["v7_outcome_withdrawal_report"][1],
    )

    development = lineage / "open27-decision.json"
    development.write_text("frozen-open27-decision\n", encoding="utf-8")
    development.chmod(0o400)
    development_sha = hashlib.sha256(development.read_bytes()).hexdigest()
    monkeypatch.setattr(
        protocol,
        "OPEN27_DEVELOPMENT_DECISION_FILE_SHA256",
        development_sha,
    )
    monkeypatch.setitem(
        protocol.FROZEN_FIELD_CONTRACT,
        "open27_development_decision_file_sha256",
        development_sha,
    )
    attempt3 = _attempt3_lineage_fixture(lineage, monkeypatch)

    config = {"prediction_frame_count": 76, "test_fixture": True}
    root = tmp_path / "held-v8"
    fresh_root_capability = protocol.prepare_fresh_held_root(root)
    disclosure = root / "post-withdrawal-development-use-disclosure.json"
    disclosure_value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": protocol.POST_WITHDRAWAL_DISCLOSURE_KIND,
        "protocol_id": protocol.PROTOCOL_ID,
        "disclosed_v7_files": {
            name: {
                **_bound_file(disclosed_paths[name]),
                "mode_octal": "0400",
            }
            for name in sorted(disclosed_paths)
        },
        "disclosed_v8_attempt3_files": {
            name: {**attempt3[f"{short}_record"], "mode_octal": "0400"}
            for name, short in (
                ("v8_attempt3_withdrawal_report", "report"),
                ("v8_attempt3_withdrawal_pointer", "pointer"),
                (
                    "v8_attempt3_withdrawal_integrity_completion",
                    "completion",
                ),
            )
        },
        "v8_attempt3_archive_integrity": attempt3["archive_integrity"],
        "v8_attempt3_revision_basis": {
            "official_x0_geometry_used_to_diagnose_exclusion_liveness": True,
            "future_target_coordinates_masks_or_scores_used_for_revision": False,
            "queried_prediction_score_or_gate_existed": False,
            "revision": (
                "replace exact-one-per-center matching with the inclusive 15 mm "
                "x0-only radius union"
            ),
        },
        "post_withdrawal_development": {
            **protocol.POST_WITHDRAWAL_DEVELOPMENT_HASHES,
            "retired_official_target_opened_by_development_process": True,
            "retired_online_prediction_opened_by_development_process": True,
            "future_coordinates_or_masks_may_have_been_read": True,
            "derived_metrics_may_have_been_computed": True,
            "field_hypothesis_was_subsequently_reselected_on_independent_open27": True,
        },
        "retirement": {
            "exact_episode": protocol.RETIRED_V7_CASE_NAME,
            "replacement_episode": protocol.FRESH_REPLACEMENT_CASE_NAME,
            "replacement_search_excluded_entire_002_rope_silk_object": True,
            "reason": (
                "the exact held-v7 episode was exposed after formal withdrawal; "
                "the replacement was selected outside that object's episodes"
            ),
        },
        "v8_1_reuse_boundary": {
            "v7_target_or_staging_reused": False,
            "v7_physical_prediction_reused": False,
            "v7_online_prediction_reused": False,
            "v7_query_or_score_reused": False,
            "v7_execution_artifact_reused": False,
            "v7_withdrawal_report_used_only_as_immutable_lineage": True,
            "v8_attempt3_predictions_reused": False,
            "v8_attempt3_source_manifests_reused": False,
            "v8_attempt3_frozen_fields_reused": False,
            "v8_attempt3_target_artifacts_reused": False,
            "v8_attempt3_official_x0_query_artifacts_reused": False,
            "v8_attempt3_queried_prediction_artifacts_reused": False,
            "v8_attempt3_score_or_gate_artifacts_reused": False,
            "v8_attempt3_partial_artifacts_reused": False,
            "all_v8_1_attempt4_predictions_targets_queries_and_scores_fresh": True,
            "full_15_case_fresh_rerun_required": True,
        },
        "claim_boundary": (
            "This disclosure preserves prospective episode-level evaluation; it "
            "does not turn open development or v8.1 into an official Deform360 "
            "state-of-the-art comparison."
        ),
    }
    _artifact(disclosure, disclosure_value)
    lock = root / "calibration-lock.json"
    protocol.create_calibration_protocol_lock(
        lock,
        held_root=root,
        fresh_root_capability=fresh_root_capability,
        immutable_bindings={
            "frame_zero_default_config": protocol.held_contract_sha256(config),
            "frame_zero_exact_eight_subset_bounded_audit_contract": (
                frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256
            ),
            "replacement_automatic_twin_admission_contract": (
                protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256
            ),
            "center_exclusion_contract": (
                query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256
            ),
            "test_operator_source": "a" * 64,
        },
        v7_withdrawal_report_path=withdrawal,
        post_withdrawal_disclosure_path=disclosure,
        development_decision_path=development,
        attempt3_withdrawal_report_path=attempt3["report_path"],
        attempt3_withdrawal_pointer_path=attempt3["pointer_path"],
        attempt3_withdrawal_integrity_completion_path=attempt3["completion_path"],
    )
    return lock


def _first_barrier_artifacts(
    tmp_path: Path,
    lock: Path,
    *,
    role: str = "calibration",
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], Path | None]:
    cases = (
        protocol.CALIBRATION_CASE_NAMES
        if role == "calibration"
        else protocol.CONFIRMATION_CASE_NAMES
    )
    physical: dict[str, Path] = {}
    online: dict[str, Path] = {}
    fields: dict[str, Path] = {}
    held_root = lock.parent
    _ = tmp_path
    for case_name in cases:
        case_root = held_root / role / "cases" / case_name
        physical_path = case_root / "physical" / "physical-prior-seal.json"
        online_path = case_root / "online" / "online-prediction-seal.json"
        field_path = (
            case_root / "frozen-field" / "preoutcome-frozen-field-manifest.json"
        )
        physical_value = _artifact(
            physical_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "role": role,
                "lock": _bound_file(lock),
                "kind": "test-physical",
            },
        )
        online_value = _artifact(
            online_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "role": role,
                "lock": _bound_file(lock),
                "kind": "test-online",
            },
        )
        _artifact(
            field_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "lock": _bound_file(lock),
                "online_prediction_seal": _bound_file(online_path),
                "online_prediction_seal_artifact_sha256": online_value[
                    "artifact_sha256"
                ],
                "kind": "test-frozen-field",
            },
        )
        assert physical_value["artifact_sha256"]
        physical[case_name] = physical_path
        online[case_name] = online_path
        fields[case_name] = field_path
    source_path: Path | None = None
    if role == "calibration":
        source_path = (
            held_root / "replacement-source" / "manifests" / "aligned-source.json"
        )
        _artifact(
            source_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": protocol.FRESH_REPLACEMENT_CASE_NAME,
                "source_permit": protocol.replacement_source_permit_evidence(lock),
                "kind": "test-aligned-replacement-source",
            },
        )
    return physical, online, fields, source_path


def _second_barrier_artifacts(
    tmp_path: Path,
    lock: Path,
    *,
    role: str = "calibration",
) -> tuple[dict[str, Path], dict[str, Path]]:
    cases = (
        protocol.CALIBRATION_CASE_NAMES
        if role == "calibration"
        else protocol.CONFIRMATION_CASE_NAMES
    )
    queries: dict[str, Path] = {}
    queried: dict[str, Path] = {}
    held_root = lock.parent
    _ = tmp_path
    for case_name in cases:
        query_path = (
            held_root
            / role
            / "query-inputs"
            / case_name
            / "official-frame-zero-query-manifest.json"
        )
        queried_path = (
            held_root
            / role
            / "query-outputs"
            / case_name
            / "queried-prediction-seal.json"
        )
        query_value = _artifact(
            query_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "lock": _bound_file(lock),
                "kind": "test-official-x0",
            },
        )
        _artifact(
            queried_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "lock": _bound_file(lock),
                "official_query_manifest": _bound_file(query_path),
                "official_query_manifest_artifact_sha256": query_value[
                    "artifact_sha256"
                ],
                "kind": "test-queried-prediction",
            },
        )
        queries[case_name] = query_path
        queried[case_name] = queried_path
    return queries, queried


def _validator(
    path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str,
    expected_role: str,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert value["lock"] == _bound_file(Path(lock_path))
    assert value["case_name"] == expected_case_name
    if "role" in value:
        assert value["role"] == expected_role
    assert value["artifact_sha256"] == protocol.held_artifact_sha256(value)
    return value


def _field_validator(
    path: str | Path,
    *,
    lock_path: str | Path,
    expected_case_name: str,
) -> dict[str, Any]:
    return _validator(
        path,
        lock_path,
        expected_case_name=expected_case_name,
        expected_role="calibration",
    )


def _source_validator(
    path: str | Path,
    *,
    expected_source_permit: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert value["source_permit"] == expected_source_permit
    assert value["artifact_sha256"] == protocol.held_artifact_sha256(value)
    return value


def test_lock_replaces_only_retired_case_and_binds_frozen_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)
    lock = protocol.validate_protocol_lock(lock_path)

    assert len(protocol.CALIBRATION_CASE_NAMES) == 15
    assert protocol.RETIRED_V7_CASE_NAME not in protocol.CALIBRATION_CASE_NAMES
    assert (
        protocol.CALIBRATION_CASE_NAMES.count(protocol.FRESH_REPLACEMENT_CASE_NAME) == 1
    )
    assert protocol.CONFIRMATION_CASE_NAMES == (
        "002-rope-silk-ep0001",
        "081-stripe-rope-ep0005",
        "085-scarf-cloth-ep0002",
        "083-blanket-cloth-ep0007",
        "092-squirrel-ep0001",
        "170-spider-ep0006",
    )
    assert lock["frozen_field_contract"]["operator_id"] == (
        "gaussian-knn-normalized-v1"
    )
    assert lock["frozen_field_contract"]["neighbor_count"] == 4
    assert lock["frozen_field_contract"]["length_scale_fraction"] == 0.05
    assert lock["frozen_field_contract"]["support_radius_fraction"] == 0.5
    assert lock["frozen_field_contract"]["frame_indices"] == list(range(76))
    exclusion_contract = lock["frozen_field_contract"]["center_exclusion"]
    assert exclusion_contract == {
        **query_artifacts.CENTER_EXCLUSION_CONTRACT,
        "contract_sha256": query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256,
    }
    assert query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256 == (
        protocol.held_contract_sha256(query_artifacts.CENTER_EXCLUSION_CONTRACT)
    )
    assert exclusion_contract["operator_id"] == "x0-euclidean-radius-union-v1"
    assert exclusion_contract["inclusion_predicate"] == (
        "distance_m <= maximum_distance_m"
    )
    assert exclusion_contract["distance_compute_dtype"] == "<f8"
    assert exclusion_contract["union_semantics"] == (
        "set-union-over-all-assimilation-centers"
    )
    assert exclusion_contract["excluded_query_cardinality"] == (
        "variable-zero-to-official-query-count"
    )
    assert exclusion_contract["unmatched_center_policy"] == "exclude-no-query"
    assert exclusion_contract["per_center_nearest_query_tie_break"] == (
        "distance-then-query-identity-id"
    )
    assert lock["primary_method"]["center_exclusion_contract_sha256"] == (
        query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256
    )
    assert lock["protocol_id"] == "deform360-held-online-belief-v8.1"
    assert lock["execution_attempt"] == protocol.EXECUTION_ATTEMPT == 4
    assert lock["freshness_and_reuse"] == protocol.FRESHNESS_AND_REUSE_CONTRACT
    assert (
        lock["freshness_and_reuse"]["held_v8_root_absent_before_attempt4_lock"] is True
    )
    assert (
        lock["freshness_and_reuse"][
            "all_predictions_must_be_fresh_v8_1_attempt4_outputs"
        ]
        is True
    )
    assert (
        lock["freshness_and_reuse"][
            "all_targets_queries_and_scores_must_be_fresh_v8_1_attempt4_outputs"
        ]
        is True
    )
    assert all(
        value is False
        for key, value in lock["freshness_and_reuse"].items()
        if key.endswith("_reused")
    )
    assert lock["freshness_and_reuse"]["full_15_case_fresh_rerun_required"] is True

    lineage = lock["lineage"]
    assert lineage["v8_attempt3_withdrawal_report"]["sha256"] == (
        protocol.ATTEMPT3_WITHDRAWAL_REPORT_FILE_SHA256
    )
    assert lineage["v8_attempt3_withdrawal_pointer"]["sha256"] == (
        protocol.ATTEMPT3_WITHDRAWAL_POINTER_FILE_SHA256
    )
    assert lineage["v8_attempt3_withdrawal_integrity_completion"]["sha256"] == (
        protocol.ATTEMPT3_WITHDRAWAL_COMPLETION_FILE_SHA256
    )
    assert lineage["v8_attempt3_archive_integrity"] == {
        "path": str(protocol.ATTEMPT3_ARCHIVE_PATH),
        "root_mode_octal": "0500",
        "fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": (
            protocol.ATTEMPT3_ARCHIVE_INVENTORY_SHA256
        ),
        "postseal_noncode_entry_count": protocol.ATTEMPT3_ARCHIVE_ENTRY_COUNT,
    }

    source_permit = protocol.authorize_replacement_source_acquisition(lock_path)
    source_evidence = protocol.consume_replacement_source_acquisition_capability(
        source_permit,
        case_name=protocol.FRESH_REPLACEMENT_CASE_NAME,
        operation=protocol.REPLACEMENT_SOURCE_OPERATION,
    )
    assert source_evidence == protocol.replacement_source_permit_evidence(lock_path)
    with pytest.raises(ValueError, match="already consumed"):
        protocol.consume_replacement_source_acquisition_capability(
            source_permit,
            case_name=protocol.FRESH_REPLACEMENT_CASE_NAME,
            operation=protocol.REPLACEMENT_SOURCE_OPERATION,
        )

    with pytest.raises(ValueError, match="must be absent"):
        protocol.prepare_fresh_held_root(tmp_path / "held-v8")


def test_lock_rejects_attempt3_pointer_byte_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)
    pointer = protocol.ATTEMPT3_WITHDRAWAL_POINTER_PATH
    original = pointer.read_bytes()
    pointer.chmod(0o600)
    pointer.write_bytes(original + b" ")
    pointer.chmod(0o400)

    with pytest.raises(ValueError, match="binding changed|file hash changed"):
        protocol.validate_protocol_lock(lock_path)


def test_lock_rejects_writable_attempt3_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)
    protocol.ATTEMPT3_ARCHIVE_PATH.chmod(0o700)

    with pytest.raises(ValueError, match="archive root"):
        protocol.validate_protocol_lock(lock_path)


def test_lock_rejects_attempt3_deployed_tracked_file_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)
    archive = protocol.ATTEMPT3_ARCHIVE_PATH
    deployed_code = next(
        path for path in archive.iterdir() if path.name.startswith("code-")
    )
    tracked = deployed_code / "tracked.py"
    tracked.chmod(0o600)
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    tracked.chmod(0o400)

    with pytest.raises(
        ValueError, match="worktree content changed|tracked file content"
    ):
        protocol.validate_protocol_lock(lock_path)


@pytest.mark.parametrize("name", ("untracked-runtime.py", "ignored-runtime.py"))
def test_lock_rejects_attempt3_deployed_untracked_or_ignored_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)
    archive = protocol.ATTEMPT3_ARCHIVE_PATH
    deployed_code = next(
        path for path in archive.iterdir() if path.name.startswith("code-")
    )
    deployed_code.chmod(0o700)
    injected = deployed_code / name
    injected.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    injected.chmod(0o400)
    deployed_code.chmod(0o500)

    with pytest.raises(
        ValueError,
        match="worktree content changed|untracked or ignored|worktree path set changed",
    ):
        protocol.validate_protocol_lock(lock_path)


@pytest.mark.parametrize("mutation", ("added", "deleted", "modified"))
def test_lock_rejects_changed_attempt3_noncode_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)
    archive = protocol.ATTEMPT3_ARCHIVE_PATH
    payload = archive / "sealed-evidence" / "payload.bin"
    if mutation == "added":
        archive.chmod(0o700)
        added = archive / "unexpected.bin"
        added.write_bytes(b"unexpected\n")
        added.chmod(0o400)
        archive.chmod(0o500)
    elif mutation == "deleted":
        payload.parent.chmod(0o700)
        payload.unlink()
        payload.parent.chmod(0o500)
    else:
        payload.chmod(0o600)
        payload.write_bytes(b"changed-with-the-same-sealed-mode\n")
        payload.chmod(0o400)

    with pytest.raises(ValueError, match="archive .*inventory changed"):
        protocol.validate_protocol_lock(lock_path)


def test_routine_lock_validation_does_not_reread_attempt3_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)

    def _unexpected_content_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("routine lock validation reread the full archive")

    monkeypatch.setattr(
        protocol,
        "_observed_attempt3_noncode_inventory",
        _unexpected_content_read,
    )

    protocol.validate_protocol_lock(lock_path)


def test_barrier_one_is_complete_cohort_case_specific_and_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    incomplete = dict(fields)
    incomplete.pop(protocol.CALIBRATION_CASE_NAMES[-1])

    with pytest.raises(ValueError, match="every exact cohort"):
        protocol.authorize_target_reconstruction_capabilities(
            lock,
            physical_seal_paths=physical,
            online_seal_paths=online,
            frozen_field_manifest_paths=incomplete,
            replacement_aligned_source_manifest_path=source,
            role="calibration",
            physical_validator=_validator,
            online_validator=_validator,
            frozen_field_validator=_field_validator,
            replacement_source_validator=_source_validator,
        )

    capabilities = protocol.authorize_target_reconstruction_capabilities(
        lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        replacement_aligned_source_manifest_path=source,
        role="calibration",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
        replacement_source_validator=_source_validator,
    )
    case_name = protocol.CALIBRATION_CASE_NAMES[0]
    permit = capabilities[case_name]
    with pytest.raises(ValueError, match="case or operation"):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.FUTURE_SCORE_OPERATION,
        )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(permit)

    evidence = protocol.consume_case_capability(
        permit,
        case_name=case_name,
        operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
    )
    assert evidence["single_use_consumed"] is True
    assert evidence["process_local_capability"] is True
    with pytest.raises(ValueError, match="already consumed"):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )


def test_barrier_replay_spends_capability_when_a_seal_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    capabilities = protocol.authorize_target_reconstruction_capabilities(
        lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        replacement_aligned_source_manifest_path=source,
        role="calibration",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
        replacement_source_validator=_source_validator,
    )
    case_name = protocol.CALIBRATION_CASE_NAMES[3]
    permit = capabilities[case_name]
    changed = online[protocol.CALIBRATION_CASE_NAMES[-1]]
    original = changed.read_bytes()
    changed.chmod(0o600)
    changed.write_bytes(original + b" ")
    changed.chmod(0o400)

    with pytest.raises((AssertionError, json.JSONDecodeError, ValueError)):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )
    changed.chmod(0o600)
    changed.write_bytes(original)
    changed.chmod(0o400)
    with pytest.raises(ValueError, match="already consumed"):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )


def test_future_target_stays_closed_until_all_x0_queries_are_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    reconstruction = protocol.authorize_target_reconstruction_capabilities(
        lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        replacement_aligned_source_manifest_path=source,
        role="calibration",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
        replacement_source_validator=_source_validator,
    )
    case_name = protocol.CALIBRATION_CASE_NAMES[0]
    target_opened = False

    def try_open_future(permit: object) -> None:
        nonlocal target_opened
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.FUTURE_SCORE_OPERATION,
        )
        target_opened = True

    with pytest.raises(ValueError, match="case or operation"):
        try_open_future(reconstruction[case_name])
    assert target_opened is False

    queries, queried = _second_barrier_artifacts(tmp_path, lock)
    incomplete = dict(queried)
    incomplete.pop(protocol.CALIBRATION_CASE_NAMES[-1])
    with pytest.raises(ValueError, match="every exact cohort"):
        protocol.authorize_future_score_capabilities(
            lock,
            official_query_manifest_paths=queries,
            queried_prediction_seal_paths=incomplete,
            role="calibration",
            official_query_validator=_validator,
            queried_prediction_validator=_validator,
        )
    assert target_opened is False

    with pytest.raises(ValueError, match="all reconstruction capabilities"):
        protocol.authorize_future_score_capabilities(
            lock,
            official_query_manifest_paths=queries,
            queried_prediction_seal_paths=queried,
            role="calibration",
            official_query_validator=_validator,
            queried_prediction_validator=_validator,
        )
    for reconstruction_case, permit in reconstruction.items():
        protocol.consume_case_capability(
            permit,
            case_name=reconstruction_case,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )

    scoring = protocol.authorize_future_score_capabilities(
        lock,
        official_query_manifest_paths=queries,
        queried_prediction_seal_paths=queried,
        role="calibration",
        official_query_validator=_validator,
        queried_prediction_validator=_validator,
    )
    try_open_future(scoring[case_name])
    assert target_opened is True


def _gate_decision(
    path: Path,
    lock: Path,
    *,
    passed: bool,
) -> None:
    score = path.parent / ("score-go.json" if passed else "score-no.json")
    _write_json(score, {"sealed": True})
    value: dict[str, Any] = {
        "schema_version": protocol.SCHEMA_VERSION,
        "artifact_kind": protocol.CALIBRATION_DECISION_KIND,
        "protocol_id": protocol.PROTOCOL_ID,
        "role": "calibration",
        "lock": _bound_file(lock),
        "ordered_case_names": list(protocol.CALIBRATION_CASE_NAMES),
        "barrier_two_sha256": "b" * 64,
        "score_evidence": _bound_file(score),
        "gate_result": {
            "gate": "v8-calibration-go-no-go-v1",
            "passed": passed,
        },
        "decision": "GO" if passed else "NO-GO",
    }
    _artifact(path, value)


def test_confirmation_is_inaccessible_until_a_sealed_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_lock = _lock_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="until calibration GO"):
        protocol.validate_first_cohort_barrier(
            calibration_lock,
            physical_seal_paths={},
            online_seal_paths={},
            frozen_field_manifest_paths={},
            role="confirmation",
            physical_validator=_validator,
            online_validator=_validator,
            frozen_field_validator=_field_validator,
        )

    no_go = tmp_path / "held-v8" / "calibration" / "calibration-no-go.json"
    _gate_decision(no_go, calibration_lock, passed=False)
    with pytest.raises(ValueError, match="after calibration NO-GO"):
        protocol.create_confirmation_protocol_lock(
            tmp_path / "held-v8" / "confirmation-lock.json",
            calibration_lock,
            no_go,
        )

    go = tmp_path / "held-v8" / "calibration" / "calibration-go.json"
    _gate_decision(go, calibration_lock, passed=True)
    confirmation_lock = tmp_path / "held-v8" / "confirmation-lock.json"
    protocol.create_confirmation_protocol_lock(
        confirmation_lock,
        calibration_lock,
        go,
    )
    assert (
        protocol.locked_case_names(confirmation_lock, role="confirmation")
        == protocol.CONFIRMATION_CASE_NAMES
    )
    physical, online, fields, source = _first_barrier_artifacts(
        tmp_path / "post-go", confirmation_lock, role="confirmation"
    )
    assert source is None
    evidence = protocol.validate_first_cohort_barrier(
        confirmation_lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        role="confirmation",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
    )
    assert evidence.role == "confirmation"
    assert evidence.ordered_case_names == protocol.CONFIRMATION_CASE_NAMES


@pytest.mark.parametrize("outside_tree", ("held-v7", "held-v8-attempt-3"))
def test_prior_execution_paths_are_never_admitted_to_a_v8_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outside_tree: str,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    case_name = protocol.CALIBRATION_CASE_NAMES[0]
    prior_root = tmp_path / outside_tree / case_name
    prior_physical = prior_root / "physical.json"
    prior_physical.parent.mkdir(parents=True)
    prior_physical.write_bytes(physical[case_name].read_bytes())
    prior_physical.chmod(0o400)
    physical[case_name] = prior_physical

    with pytest.raises(ValueError, match="exact current held-v8 subtree"):
        protocol.validate_first_cohort_barrier(
            lock,
            physical_seal_paths=physical,
            online_seal_paths=online,
            frozen_field_manifest_paths=fields,
            replacement_aligned_source_manifest_path=source,
            role="calibration",
            physical_validator=_validator,
            online_validator=_validator,
            frozen_field_validator=_field_validator,
            replacement_source_validator=_source_validator,
        )


def test_attempt3_hardlink_is_never_admitted_as_a_fresh_v8_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    case_name = protocol.CALIBRATION_CASE_NAMES[0]
    current_physical = physical[case_name]
    current_physical.unlink()
    retired_attempt3_code = next(
        path
        for path in protocol.ATTEMPT3_ARCHIVE_PATH.iterdir()
        if path.name.startswith("code-")
    )
    retired_attempt3_file = retired_attempt3_code / "tracked.py"
    os.link(retired_attempt3_file, current_physical)
    assert current_physical.stat().st_ino == retired_attempt3_file.stat().st_ino
    assert current_physical.stat().st_nlink == 2

    def should_not_run(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("artifact validator ran before hardlink rejection")

    with pytest.raises(ValueError, match="single-link fresh current-execution"):
        protocol.validate_first_cohort_barrier(
            lock,
            physical_seal_paths=physical,
            online_seal_paths=online,
            frozen_field_manifest_paths=fields,
            replacement_aligned_source_manifest_path=source,
            role="calibration",
            physical_validator=should_not_run,
            online_validator=should_not_run,
            frozen_field_validator=should_not_run,
            replacement_source_validator=should_not_run,
        )


def test_barrier_rejects_cross_case_and_cross_role_artifacts_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    first_case, second_case = protocol.CALIBRATION_CASE_NAMES[-2:]
    physical[first_case], physical[second_case] = (
        physical[second_case],
        physical[first_case],
    )

    def should_not_run(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("artifact validator ran before path containment")

    with pytest.raises(ValueError, match="exact current held-v8 subtree"):
        protocol.validate_first_cohort_barrier(
            lock,
            physical_seal_paths=physical,
            online_seal_paths=online,
            frozen_field_manifest_paths=fields,
            replacement_aligned_source_manifest_path=source,
            role="calibration",
            physical_validator=should_not_run,
            online_validator=should_not_run,
            frozen_field_validator=should_not_run,
            replacement_source_validator=should_not_run,
        )

    physical[first_case], physical[second_case] = (
        physical[second_case],
        physical[first_case],
    )
    cross_role = (
        lock.parent
        / "confirmation"
        / "cases"
        / first_case
        / "physical"
        / "physical-prior-seal.json"
    )
    cross_role.parent.mkdir(parents=True)
    cross_role.write_bytes(physical[first_case].read_bytes())
    cross_role.chmod(0o400)
    physical[first_case] = cross_role
    with pytest.raises(ValueError, match="exact current held-v8 subtree"):
        protocol.validate_first_cohort_barrier(
            lock,
            physical_seal_paths=physical,
            online_seal_paths=online,
            frozen_field_manifest_paths=fields,
            replacement_aligned_source_manifest_path=source,
            role="calibration",
            physical_validator=should_not_run,
            online_validator=should_not_run,
            frozen_field_validator=should_not_run,
            replacement_source_validator=should_not_run,
        )


def test_second_barrier_preflights_all_paths_before_injected_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    queries, queried = _second_barrier_artifacts(tmp_path, lock)
    case_name = protocol.CALIBRATION_CASE_NAMES[-1]
    cross_role = (
        lock.parent
        / "confirmation"
        / "query-outputs"
        / case_name
        / "queried-prediction-seal.json"
    )
    cross_role.parent.mkdir(parents=True)
    cross_role.write_bytes(queried[case_name].read_bytes())
    cross_role.chmod(0o400)
    queried[case_name] = cross_role

    def should_not_run(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("artifact validator ran before path containment")

    with pytest.raises(ValueError, match="exact current held-v8 subtree"):
        protocol.validate_second_cohort_barrier(
            lock,
            official_query_manifest_paths=queries,
            queried_prediction_seal_paths=queried,
            role="calibration",
            official_query_validator=should_not_run,
            queried_prediction_validator=should_not_run,
        )


def test_v8_seal_creators_freeze_fresh_builder_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    case_name = protocol.CALIBRATION_CASE_NAMES[1]
    case_root = lock.parent / "calibration" / "cases" / case_name
    frame_zero = case_root / "frame-zero" / "frame-zero-bundle.manifest.json"
    frame_zero_value = _artifact(
        frame_zero,
        {
            "protocol_id": protocol.PROTOCOL_ID,
            "case_name": case_name,
            "role": "calibration",
            "artifact_kind": protocol.FRAME_ZERO_KIND,
        },
    )

    def validate_frame_zero(
        path: str | Path,
        lock_path: str | Path,
        *,
        expected_case_name: str | None = None,
        expected_role: str | None = None,
    ) -> dict[str, Any]:
        assert Path(path) == frame_zero
        assert Path(lock_path) == lock
        assert expected_case_name in {None, case_name}
        assert expected_role in {None, "calibration"}
        return frame_zero_value

    monkeypatch.setattr(
        protocol,
        "validate_frame_zero_bundle_manifest",
        validate_frame_zero,
    )
    physical_artifacts: dict[str, Path] = {}
    for role in protocol.PHYSICAL_ARTIFACT_ROLES:
        path = case_root / "physical" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode("ascii"))
        path.chmod(0o600)
        physical_artifacts[role] = path
    physical_seal = case_root / "physical" / "physical-prior-seal.json"
    protocol.create_physical_prior_seal(
        physical_seal,
        lock,
        frame_zero,
        physical_artifacts,
        case_name=case_name,
        role="calibration",
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o400 for path in physical_artifacts.values()
    )

    prefix = case_root / "prefix-authorization.json"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    protocol.create_prefix_stage_authorization(prefix, lock, physical_seal)
    online_artifacts: dict[str, Path] = {}
    for role in protocol.ONLINE_ARTIFACT_ROLES:
        path = case_root / "online" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode("ascii"))
        path.chmod(0o600)
        online_artifacts[role] = path
    online_seal = case_root / "online" / "online-prediction-seal.json"
    protocol.create_online_prediction_seal(
        online_seal,
        lock,
        prefix,
        online_artifacts,
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o400 for path in online_artifacts.values()
    )

    physical_value = json.loads(physical_seal.read_text(encoding="utf-8"))
    physical_value["physical_artifacts"]["prediction_only_input"] = _bound_file(
        protocol.ATTEMPT3_WITHDRAWAL_REPORT_PATH
    )
    physical_value["artifact_sha256"] = protocol.held_artifact_sha256(physical_value)
    physical_seal.chmod(0o600)
    _write_json(physical_seal, physical_value)
    with pytest.raises(ValueError, match="exact current held-v8 subtree"):
        protocol.validate_physical_prior_seal(
            physical_seal,
            lock,
            expected_case_name=case_name,
            expected_role="calibration",
        )
