from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin import deform360_held_v8_protocol as protocol


def _write_artifact(
    path: Path, value: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = dict(value)
    value["artifact_sha256"] = protocol.held_artifact_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o400)
    return value, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    archive = tmp_path / "held-v8-attempt-4-withdrawn-postbarrier"
    archive.mkdir()
    report_path = archive / "execution-withdrawal-postbarrier-attempt4.json"
    pointer_path = tmp_path / "held-v8-attempt-4-withdrawal-pointer.json"
    completion_path = tmp_path / "held-v8-attempt-4-withdrawal-completion.json"
    launcher_path = tmp_path / "launcher"
    launcher_path.mkdir()
    (launcher_path / "exit.code").write_bytes(b"2\n")
    (launcher_path / "output.log").write_bytes(b"fixed markers only\n")
    for path in launcher_path.iterdir():
        path.chmod(0o400)
    launcher_path.chmod(0o500)
    deployed = {
        "path": "code-" + "a" * 40,
        "git_head": "a" * 40,
        "head_text_sha256": "b" * 64,
        "git_tree_record_count": 1,
        "git_tree_manifest_sha256": "c" * 64,
        "every_working_file_matches_bound_git_blob": True,
        "no_ordinary_or_ignored_untracked_files": True,
    }
    launcher = {
        "path": str(launcher_path),
        "exact_file_allowlist": ["exit.code", "output.log"],
        "exit_code": {"mode_octal": "0400", "sha256": "d" * 64, "size_bytes": 2},
        "output_log": {"mode_octal": "0400", "sha256": "e" * 64, "size_bytes": 19},
        "terminal_marker_counts": {},
        "log_scanned_for_fixed_markers_only": True,
        "log_numerical_payload_parsed": False,
    }
    inventory = {
        "rows": [],
        "entry_count": 0,
        "directory_count": 0,
        "regular_file_count": 0,
        "regular_file_bytes": 0,
        "inventory_sha256": "f" * 64,
        "excluded_deployed_code_directory": deployed["path"],
        "excluded_withdrawal_report": report_path.name,
    }
    report, report_record = _write_artifact(
        report_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt4PostBarrierWithdrawalReport",
            "protocol_id": protocol.PROTOCOL_ID,
            "execution_attempt": 4,
            "status": protocol._ATTEMPT4_STATUS,
            "disposition": protocol._ATTEMPT4_DISPOSITION,
            "result_status": "NO_CALIBRATION_RESULT",
            "terminal_failure": {
                "evidence_origin": "durable-launcher-log-fixed-marker-scan",
                "outer_outcome_driver_exit_code": 2,
                "exception_type": "OSError",
                "errno": 24,
                "exception_message_class": "Too many open files",
                "failed_case": "002-rope-silk-ep0008",
                "failure_path": (
                    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v8/"
                    "calibration/private-targets/002-rope-silk-ep0008/"
                    "fresh-official-reconstruction/staged-aligned/episode_0000/"
                    "splatfacto/.scratch_000080/outputs/splat_80/splatfacto/"
                    "2026-07-22_192624"
                ),
                "failure_phase": (
                    "third target reconstruction after final-frame training and "
                    "before reconstruction audit, target seal, second barrier, or score"
                ),
            },
            "immutable_archive_path": str(archive),
            "deployed_code": deployed,
            "expected_postseal_inventory": inventory,
            "durable_launcher_evidence": launcher,
            "execution_boundary": {
                "online_prediction_seal_count": 15,
                "frozen_field_manifest_count": 15,
                "first_cohort_barrier_validated_count": 1,
                "official_target_archive_count": 2,
                "official_x0_archive_count": 2,
                "queried_prediction_seal_count": 2,
                "partial_reconstruction_count": 1,
                "second_cohort_barrier_validated_count": 0,
                "score_evidence_count": 0,
                "gate_decision_count": 0,
                "confirmation_lock_count": 0,
            },
            "information_boundary": {
                "first_complete_cohort_barrier_crossed": True,
                "second_complete_cohort_barrier_crossed": False,
                "score_created_or_read": False,
                "gate_decision_created_or_read": False,
                "confirmation_created_or_read": False,
            },
        },
    )
    shared = {
        "archive_path": str(archive),
        "archive_root_mode_octal": "0500",
        "archive_fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": inventory["inventory_sha256"],
        "postseal_noncode_entry_count": 0,
        "independent_post_rename_integrity_verified": True,
        "withdrawal_report_path": str(report_path),
        "withdrawal_report_size_bytes": report_record["size_bytes"],
        "withdrawal_report_file_sha256": report_record["sha256"],
        "withdrawal_report_artifact_sha256": report["artifact_sha256"],
        "deployed_code": deployed,
    }
    completion, completion_record = _write_artifact(
        completion_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt4WithdrawalIntegrityCompletion",
            "protocol_id": protocol.PROTOCOL_ID,
            "execution_attempt": 4,
            "status": "withdrawal-integrity-complete",
            "disposition": protocol._ATTEMPT4_DISPOSITION,
            **shared,
            "durable_launcher_evidence": {
                **launcher,
                "root_mode_octal": "0500",
                "fully_nonwritable": True,
            },
            "pointer_contract": {
                "path": str(pointer_path),
                "artifact_kind": "Deform360HeldV81Attempt4WithdrawalPointer",
                "pointer_must_bind_this_completion": True,
                "completion_does_not_predict_pointer_hash_to_avoid_circularity": True,
            },
        },
    )
    pointer, pointer_record = _write_artifact(
        pointer_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt4WithdrawalPointer",
            "protocol_id": protocol.PROTOCOL_ID,
            "execution_attempt": 4,
            "status": protocol._ATTEMPT4_STATUS,
            "disposition": protocol._ATTEMPT4_DISPOSITION,
            **shared,
            "durable_launcher_evidence": {
                **launcher,
                "root_mode_octal": "0500",
                "fully_nonwritable": True,
            },
            "withdrawal_integrity_completion": {
                **completion_record,
                "mode_octal": "0400",
                "artifact_sha256": completion["artifact_sha256"],
            },
            "active_held_v8_root_absent_after_archive": True,
            "completed_target_x0_queried_pairs": 2,
            "first_cohort_barrier_crossed": True,
            "second_cohort_barrier_crossed": False,
            "score_evidence_count": 0,
            "gate_decision_count": 0,
            "confirmation_accessed": False,
        },
    )
    archive.chmod(0o500)
    replacements = {
        "ATTEMPT4_ARCHIVE_PATH": archive,
        "ATTEMPT4_WITHDRAWAL_REPORT_PATH": report_path,
        "ATTEMPT4_WITHDRAWAL_POINTER_PATH": pointer_path,
        "ATTEMPT4_WITHDRAWAL_INTEGRITY_COMPLETION_PATH": completion_path,
        "ATTEMPT4_WITHDRAWAL_REPORT_FILE_SHA256": report_record["sha256"],
        "ATTEMPT4_WITHDRAWAL_REPORT_ARTIFACT_SHA256": report["artifact_sha256"],
        "ATTEMPT4_WITHDRAWAL_POINTER_FILE_SHA256": pointer_record["sha256"],
        "ATTEMPT4_WITHDRAWAL_POINTER_ARTIFACT_SHA256": pointer["artifact_sha256"],
        "ATTEMPT4_WITHDRAWAL_COMPLETION_FILE_SHA256": completion_record["sha256"],
        "ATTEMPT4_WITHDRAWAL_COMPLETION_ARTIFACT_SHA256": completion["artifact_sha256"],
        "ATTEMPT4_ARCHIVE_INVENTORY_SHA256": inventory["inventory_sha256"],
        "ATTEMPT4_ARCHIVE_ENTRY_COUNT": 0,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(protocol, name, value)
    calls = {"inventory": 0, "deployed_verify": []}

    def fake_deployed(_archive: Path, _report: dict[str, Any], *, verify_content: bool):
        calls["deployed_verify"].append(verify_content)
        return deployed

    def fake_inventory(_archive: Path, *, deployed_code: Path):
        del deployed_code
        calls["inventory"] += 1
        return inventory

    monkeypatch.setattr(protocol, "_attempt4_deployed_code", fake_deployed)
    monkeypatch.setattr(
        protocol, "_observed_attempt4_noncode_inventory", fake_inventory
    )
    monkeypatch.setattr(
        protocol,
        "_validate_attempt4_launcher",
        lambda value, *, verify_content: dict(value),
    )
    return {
        "archive": archive,
        "report": report_path,
        "pointer": pointer_path,
        "completion": completion_path,
        "calls": calls,
    }


def test_attempt4_routine_validation_does_not_rehash_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _chain(tmp_path, monkeypatch)

    result = protocol.validate_attempt4_withdrawal_lineage(
        archive_path=fixture["archive"],
        report_path=fixture["report"],
        pointer_path=fixture["pointer"],
        completion_path=fixture["completion"],
    )

    assert result["v8_attempt4_calibration_result"] == "NO_CALIBRATION_RESULT"
    assert fixture["calls"]["inventory"] == 0
    assert fixture["calls"]["deployed_verify"] == [False]


def test_attempt4_lock_creation_mode_rehashes_full_archive_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _chain(tmp_path, monkeypatch)

    protocol.validate_attempt4_withdrawal_lineage(
        archive_path=fixture["archive"],
        report_path=fixture["report"],
        pointer_path=fixture["pointer"],
        completion_path=fixture["completion"],
        verify_content_inventory=True,
    )

    assert fixture["calls"]["inventory"] == 1
    assert fixture["calls"]["deployed_verify"] == [True]


def test_attempt4_routine_validation_rejects_archive_metadata_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _chain(tmp_path, monkeypatch)
    fixture["archive"].chmod(0o700)
    added = fixture["archive"] / "unexpected.bin"
    added.write_bytes(b"changed\n")
    added.chmod(0o400)
    fixture["archive"].chmod(0o500)

    with pytest.raises(ValueError, match="metadata inventory changed"):
        protocol.validate_attempt4_withdrawal_lineage(
            archive_path=fixture["archive"],
            report_path=fixture["report"],
            pointer_path=fixture["pointer"],
            completion_path=fixture["completion"],
        )


def test_attempt4_launcher_routine_validation_checks_metadata_without_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "launcher"
    root.mkdir()
    log = root / "output.log"
    exit_code = root / "exit.code"
    log.write_bytes(b"fixed markers only\n")
    exit_code.write_bytes(b"2\n")
    log.chmod(0o400)
    exit_code.chmod(0o400)
    root.chmod(0o500)
    monkeypatch.setattr(protocol, "ATTEMPT4_LAUNCHER_PATH", root)
    monkeypatch.setattr(protocol, "ATTEMPT4_LAUNCHER_LOG_SHA256", "e" * 64)
    monkeypatch.setattr(protocol, "ATTEMPT4_LAUNCHER_EXIT_SHA256", "d" * 64)
    monkeypatch.setattr(
        protocol, "ATTEMPT4_LAUNCHER_LOG_SIZE_BYTES", log.stat().st_size
    )
    monkeypatch.setattr(
        protocol, "ATTEMPT4_LAUNCHER_EXIT_SIZE_BYTES", exit_code.stat().st_size
    )
    monkeypatch.setattr(
        protocol,
        "_sha256_file",
        lambda _path: pytest.fail("routine validation must not hash launcher bytes"),
    )
    launcher = {
        "path": str(root),
        "exact_file_allowlist": ["exit.code", "output.log"],
        "exit_code": {
            "mode_octal": "0400",
            "sha256": "d" * 64,
            "size_bytes": exit_code.stat().st_size,
        },
        "output_log": {
            "mode_octal": "0400",
            "sha256": "e" * 64,
            "size_bytes": log.stat().st_size,
        },
        "terminal_marker_counts": {
            "first_cohort_barrier_validated": 1,
            "official_target_and_x0_sealed": 2,
            "isolated_x0_query_sealed": 2,
            "second_cohort_barrier_validated": 0,
            "fail_closed": 1,
            "terminal_error_type": 1,
            "too_many_open_files": 1,
        },
        "log_scanned_for_fixed_markers_only": True,
        "log_numerical_payload_parsed": False,
    }

    assert (
        protocol._validate_attempt4_launcher(launcher, verify_content=False) == launcher
    )

    alias = tmp_path / "launcher-log-alias"
    os.link(log, alias)
    with pytest.raises(ValueError, match="metadata changed"):
        protocol._validate_attempt4_launcher(launcher, verify_content=False)
    alias.unlink()

    log_sha256 = hashlib.sha256(log.read_bytes()).hexdigest()
    exit_sha256 = hashlib.sha256(exit_code.read_bytes()).hexdigest()
    monkeypatch.setattr(protocol, "ATTEMPT4_LAUNCHER_LOG_SHA256", log_sha256)
    monkeypatch.setattr(protocol, "ATTEMPT4_LAUNCHER_EXIT_SHA256", exit_sha256)
    launcher["output_log"]["sha256"] = log_sha256
    launcher["exit_code"]["sha256"] = exit_sha256
    assert (
        protocol._validate_attempt4_launcher(launcher, verify_content=True) == launcher
    )


def test_attempt5_protocol_keeps_science_and_adds_only_resource_contracts() -> None:
    assert protocol.PROTOCOL_ID == "deform360-held-online-belief-v8.1"
    assert protocol.EXECUTION_ATTEMPT == 5
    assert protocol.FRAME_COUNT == 76
    assert protocol.UPDATE_FRAMES == (19, 38, 57)
    assert protocol.RESOURCE_LIFECYCLE_QUALIFICATION_KIND.endswith("EvidenceV2")
    assert protocol.RESOURCE_LIFECYCLE_QUALIFICATION_ID.endswith(
        "resource-lifecycle-qualification-v2"
    )
    assert protocol.RESOURCE_LIFECYCLE_GENERATOR_PROFILE == "same-as-analyzer"
    assert protocol.RESOURCE_LIFECYCLE_PHYSICAL_GPU_INDEX == 1
    assert protocol.RESOURCE_LIFECYCLE_ANALYZER_SOURCE_SHA256 == (
        "43056e39ff7ea5f760f18420784db0edbb75523031dba7f3a19eca0c6951c128"
    )
    assert protocol.RESOURCE_LIFECYCLE_ROOT_CONSUMPTION_POLICY == {
        "canonical_root_consumed_at_creation": True,
        "same_root_retry_permitted": False,
        "same_revision_retry_permitted": False,
        "in_place_reuse_permitted": False,
        "incomplete_root_sealable_or_replayable": False,
        "technical_fix_in_later_disclosed_revision_may_use_new_root": True,
        "replacement_requires_different_canonical_root": True,
        "replacement_may_change_frozen_analyzer_or_numerical_gate": False,
    }
    assert protocol.POST_CASE_RESOURCE_BOUNDARY_CONTRACT["maximum_growth"] == 32
    assert protocol.RESOURCE_LIFECYCLE_POLICY_CONTRACT["rlimit_nofile_changed"] is False
    assert (
        protocol.FRESHNESS_AND_REUSE_CONTRACT[
            "all_predictions_must_be_fresh_v8_1_attempt5_outputs"
        ]
        is True
    )
    assert all(
        value is False
        for key, value in protocol.FRESHNESS_AND_REUSE_CONTRACT.items()
        if key.startswith("v8_attempt4_") and key.endswith("_reused")
    )
