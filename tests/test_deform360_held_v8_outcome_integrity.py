from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin import deform360_held_v8_outcome_integrity as integrity


@pytest.fixture
def posix_tmp_path() -> Path:
    root = Path(tempfile.mkdtemp(prefix="bpt-v8-integrity-", dir="/tmp"))
    try:
        yield root
    finally:
        _restore_writable(root)
        shutil.rmtree(root, ignore_errors=True)


def _write_signed(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha256"] = integrity._artifact_sha256(result)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o400)
    return result


def _restore_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_symlink():
            continue
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def test_module_source_load_is_future_module_and_numpy_lazy() -> None:
    source = Path(integrity.__file__).resolve()
    code = """
import importlib.util
import pathlib
import sys
source = pathlib.Path(sys.argv[1])
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("standalone_v8_integrity", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
new = set(sys.modules) - before
forbidden = sorted(name for name in new if name == "numpy" or name.startswith("numpy.") or "deform360_held_v8_" in name)
print("\\n".join(forbidden))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(source)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.stdout.strip() == ""


def test_recursive_inventory_freezes_and_rejects_links_and_hardlinks(
    posix_tmp_path: Path,
) -> None:
    tmp_path = posix_tmp_path
    root = tmp_path / "safe"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a")
    (nested / "b.bin").write_bytes(b"b")
    try:
        integrity._freeze_tree(root, freeze_root=True)
        observed = integrity._tree_inventory(
            root, require_sealed=True, include_root=True
        )
        assert observed["regular_file_count"] == 2
        assert observed["directory_count"] == 2
        assert {row["mode_octal"] for row in observed["entries"]} == {
            "0400",
            "0500",
        }
    finally:
        _restore_writable(root)

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "payload").write_text("payload", encoding="utf-8")
    (linked / "alias").symlink_to(linked / "payload")
    with pytest.raises(ValueError, match="linked or nonregular"):
        integrity._freeze_tree(linked, freeze_root=True)

    hardlinked = tmp_path / "hardlinked"
    hardlinked.mkdir()
    original = hardlinked / "payload"
    original.write_text("payload", encoding="utf-8")
    os.link(original, hardlinked / "alias")
    with pytest.raises(ValueError, match="hard-linked"):
        integrity._freeze_tree(hardlinked, freeze_root=True)


def _fake_completed_role(tmp_path: Path) -> tuple[Path, Path, Path]:
    held_root = tmp_path / "held-v8"
    role_root = held_root / "calibration"
    payload_root = role_root / "payload"
    payload_root.mkdir(parents=True)
    payload = payload_root / "raw.bin"
    payload.write_bytes(b"sealed raw payload\n")
    payload.chmod(0o400)
    payload_root.chmod(0o500)

    lock_path = held_root / "calibration-lock.json"
    lock = _write_signed(
        lock_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldOnlineBeliefLock",
            "protocol_id": integrity.PROTOCOL_ID,
            "stage": "calibration",
            "held_root": str(held_root),
            "immutable_bindings": {},
        },
    )
    lock_record = integrity._bound_file(
        lock_path, label="test lock", required_mode=0o400
    )
    first_digest = "1" * 64
    second_digest = "2" * 64
    case_name = "object-ep0000"
    gate = {"gate": "test-gate", "passed": True}
    permit = {
        "protocol_id": integrity.PROTOCOL_ID,
        "role": "calibration",
        "case_name": case_name,
        "operation": integrity.FUTURE_SCORE_OPERATION,
        "lock_file_sha256": lock_record["sha256"],
        "lock_artifact_sha256": lock["artifact_sha256"],
        "cohort_barrier_sha256": second_digest,
        "single_use_consumed": True,
        "process_local_capability": True,
        "predecessor_reconstruction_barrier_sha256": first_digest,
    }
    records = {
        case_name: {
            "case_name": case_name,
            "future_score_permit_evidence": permit,
        }
    }
    evidence_path = role_root / "calibration-score-evidence.json"
    evidence = _write_signed(
        evidence_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV8CalibrationScoreEvidence",
            "protocol_id": integrity.PROTOCOL_ID,
            "role": "calibration",
            "lock": lock_record,
            "barrier_two_sha256": second_digest,
            "ordered_case_names": [case_name],
            "case_records": records,
            "gate_result": gate,
        },
    )
    decision_path = role_root / "calibration-gate-decision.json"
    decision = _write_signed(
        decision_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV8CalibrationGateDecision",
            "protocol_id": integrity.PROTOCOL_ID,
            "role": "calibration",
            "lock": lock_record,
            "barrier_two_sha256": second_digest,
            "score_evidence": integrity._bound_file(
                evidence_path, label="test evidence", required_mode=0o400
            ),
            "score_evidence_artifact_sha256": evidence["artifact_sha256"],
            "gate_result": gate,
            "decision": "GO",
        },
    )
    from bayesian_phystwin import deform360_held_v8_outcome_driver as outcome_driver

    source_manifest_path = outcome_driver.canonical_role_source_manifest_path(
        held_root, "calibration"
    )
    source_manifest_path.parent.mkdir(parents=True)
    source_manifest = _write_signed(
        source_manifest_path,
        {"protocol_id": integrity.PROTOCOL_ID, "role": "calibration"},
    )
    source_manifest_record = integrity._artifact_record(
        source_manifest_path,
        source_manifest,
        label="test role source manifest",
    )
    execution_completion_path = outcome_driver.canonical_role_execution_completion_path(
        held_root, "calibration"
    )
    execution_completion = _write_signed(
        execution_completion_path,
        {
            "schema_version": 1,
            "artifact_kind": outcome_driver.ROLE_EXECUTION_COMPLETION_KIND,
            "protocol_id": integrity.PROTOCOL_ID,
            "status": outcome_driver.ROLE_EXECUTION_COMPLETION_STATUS,
            "role": "calibration",
            "fake_metadata_only_fixture": True,
            "source_manifest": source_manifest_record,
        },
    )
    completion_path = integrity.canonical_role_completion_path(held_root, "calibration")
    first_bindings: list[object] = []
    second_bindings: list[object] = []
    inventory = integrity._tree_inventory(
        role_root,
        require_sealed=False,
        excluded_relative=completion_path.relative_to(role_root),
    )
    combined = {
        "ordered_case_names": [case_name],
        "case_records": records,
        "gate_result": gate,
        "decision": "GO",
    }
    completion = {
        "schema_version": integrity.SCHEMA_VERSION,
        "artifact_kind": integrity.ROLE_COMPLETION_KIND,
        "protocol_id": integrity.PROTOCOL_ID,
        "status": integrity.ROLE_COMPLETION_STATUS,
        "role": "calibration",
        "held_root": str(held_root),
        "role_root": str(role_root),
        "terminal_outcome": "GO",
        "lock": {**lock_record, "artifact_sha256": lock["artifact_sha256"]},
        "score_evidence": {
            **integrity._bound_file(
                evidence_path, label="test evidence", required_mode=0o400
            ),
            "artifact_sha256": evidence["artifact_sha256"],
        },
        "decision": {
            **integrity._bound_file(
                decision_path, label="test decision", required_mode=0o400
            ),
            "artifact_sha256": decision["artifact_sha256"],
        },
        "execution_completion": {
            **integrity._bound_file(
                execution_completion_path,
                label="test execution completion",
                required_mode=0o400,
            ),
            "artifact_sha256": execution_completion["artifact_sha256"],
        },
        "source_manifest": source_manifest_record,
        "barriers": {
            "first_complete_cohort": {
                "barrier_number": 1,
                "operation": integrity.TARGET_RECONSTRUCTION_OPERATION,
                "barrier_sha256": first_digest,
                "ordered_case_names": [case_name],
                "ordered_artifact_bindings": first_bindings,
                "ordered_artifact_bindings_sha256": integrity._digest(first_bindings),
            },
            "second_complete_cohort": {
                "barrier_number": 2,
                "operation": integrity.FUTURE_SCORE_OPERATION,
                "barrier_sha256": second_digest,
                "ordered_case_names": [case_name],
                "ordered_artifact_bindings": second_bindings,
                "ordered_artifact_bindings_sha256": integrity._digest(second_bindings),
            },
        },
        "recomputed_outcome": {
            "ordered_case_names": [case_name],
            "case_record_set_sha256": integrity._digest(records),
            "gate_result": gate,
            "gate_result_sha256": integrity._digest(gate),
            "score_and_gate_sha256": integrity._digest(combined),
            "sealed_score_evidence_exact_match": True,
            "sealed_gate_decision_exact_match": True,
            "canonical_raw_target_query_prediction_reloaded": True,
        },
        "deployed_source_operator_runtime": {"fake": True},
        "sealed_role_inventory": inventory,
        "terminal_root_finalization": {"required": False, "completion_path": None},
        "information_boundary": dict(integrity._ROLE_INFORMATION_BOUNDARY),
        "self_hash_contract": integrity._SELF_HASH_CONTRACT,
    }
    completion["artifact_sha256"] = integrity._artifact_sha256(completion)
    integrity._write_new_json(completion_path, completion)
    role_root.chmod(0o500)
    return lock_path, completion_path, payload


def test_metadata_only_validator_is_repeatable_and_detects_tree_change(
    posix_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = posix_tmp_path
    lock_path, completion_path, payload = _fake_completed_role(tmp_path)
    calls: list[bool] = []

    def validate_source(
        *_args: object, verify_runtime_modules: bool, **_kwargs: object
    ):
        calls.append(verify_runtime_modules)
        return {"fake": True}

    monkeypatch.setattr(integrity, "_validate_source_runtime_bindings", validate_source)
    from bayesian_phystwin import deform360_held_v8_outcome_driver as outcome_driver

    monkeypatch.setattr(
        outcome_driver,
        "validate_role_execution_completion",
        lambda path, **_kwargs: integrity._load_json(
            path, label="test execution completion", required_mode=0o400
        ),
    )
    monkeypatch.setattr(
        integrity,
        "_validate_role_artifacts",
        lambda *_args, **_kwargs: pytest.fail("future-bearing validation was loaded"),
    )
    source_manifest_path = integrity._role_source_manifest_path(
        lock_path.parent, "calibration"
    )
    source_manifest = integrity._load_json(
        source_manifest_path,
        label="test role source manifest",
        required_mode=0o400,
    )
    monkeypatch.setattr(
        integrity,
        "_validate_role_source_manifest",
        lambda **_kwargs: (source_manifest_path, source_manifest),
    )
    first = integrity.validate_role_outcome_completion(
        completion_path,
        lock_path=lock_path,
        expected_role="calibration",
        verify_content_inventory=True,
        recompute_scores=False,
    )
    second = integrity.validate_role_outcome_completion(
        completion_path,
        lock_path=lock_path,
        expected_role="calibration",
        verify_content_inventory=True,
        recompute_scores=False,
    )
    assert first == second
    assert first["terminal_outcome"] == "GO"
    assert calls == [False, False]

    role_root = completion_path.parent
    payload_root = payload.parent
    role_root.chmod(0o700)
    payload_root.chmod(0o700)
    payload.chmod(0o600)
    payload.write_bytes(b"changed raw payload\n")
    payload.chmod(0o400)
    payload_root.chmod(0o500)
    role_root.chmod(0o500)
    with pytest.raises(ValueError, match="inventory changed"):
        integrity.validate_role_outcome_completion(
            completion_path,
            lock_path=lock_path,
            expected_role="calibration",
            verify_content_inventory=True,
            recompute_scores=False,
        )
    _restore_writable(lock_path.parent)


def test_raw_derived_score_mismatch_is_rejected(
    posix_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_name = "object-ep0000"
    target_permit = {"permit": "target"}
    score_permit = {"permit": "score"}
    sealed_record = {
        "gate_score": {"primary_chamfer_m": 99.0},
        "future_score_permit_evidence": score_permit,
    }
    artifacts = integrity._RoleArtifacts(
        held_root=posix_tmp_path,
        role_root=posix_tmp_path / "calibration",
        lock_path=posix_tmp_path / "calibration-lock.json",
        lock={"artifact_sha256": "a" * 64},
        case_names=(case_name,),
        paths={
            case_name: {
                "field": posix_tmp_path / "field.json",
                "target": posix_tmp_path / "target.json",
                "query": posix_tmp_path / "query.json",
                "queried": posix_tmp_path / "queried.json",
            }
        },
        barrier_one=SimpleNamespace(barrier_sha256="1" * 64),
        barrier_two=SimpleNamespace(barrier_sha256="2" * 64),
        evidence_path=posix_tmp_path / "evidence.json",
        evidence={"case_records": {case_name: sealed_record}},
        decision_path=posix_tmp_path / "decision.json",
        decision={"role": "calibration"},
        execution_completion_path=posix_tmp_path / "execution.json",
        execution_completion={},
        source_manifest_path=posix_tmp_path / "source.json",
        source_manifest={},
    )
    from bayesian_phystwin import deform360_held_v8_outcome_artifacts as outcomes
    from bayesian_phystwin import deform360_held_v8_query_artifacts as queries
    from bayesian_phystwin import deform360_held_v8_scoring as scorer

    monkeypatch.setattr(
        queries,
        "validate_preoutcome_frozen_field_manifest",
        lambda *_args, **_kwargs: {
            "source_array_records": {"frame_zero_points_m": {"shape": [1, 3]}}
        },
    )
    monkeypatch.setattr(
        outcomes,
        "validate_official_target_artifact",
        lambda *_args, **_kwargs: {
            "target_reconstruction_permit_evidence": target_permit,
            "archive": {"kind": "target"},
        },
    )
    monkeypatch.setattr(
        queries,
        "validate_official_frame_zero_query_artifact",
        lambda *_args, **_kwargs: {"archive": {"kind": "query"}},
    )
    monkeypatch.setattr(
        queries,
        "validate_queried_prediction_artifact",
        lambda *_args, **_kwargs: {"archive": {"kind": "queried"}},
    )
    monkeypatch.setattr(
        integrity,
        "_expected_target_permit",
        lambda *_args, **_kwargs: target_permit,
    )
    monkeypatch.setattr(
        integrity,
        "_expected_score_permit",
        lambda *_args, **_kwargs: score_permit,
    )

    def arrays(record: object, *, label: str) -> dict[str, np.ndarray]:
        del label
        kind = record["kind"]  # type: ignore[index]
        identities = np.array([0], dtype=np.int64)
        positions = np.zeros((1, 3), dtype=np.float32)
        if kind == "target":
            return {
                "identity_ids": identities,
                "object_points": positions[None, :, :],
            }
        if kind == "query":
            return {"identity_ids": identities, "positions_m": positions}
        return {"identity_ids": identities, "positions_m": positions}

    monkeypatch.setattr(integrity, "_load_npz_from_record", arrays)
    inputs = SimpleNamespace(
        source_node_count=1,
        scoring_kwargs=lambda: {},
    )
    monkeypatch.setattr(outcomes, "_assemble_direct_scoring_inputs", lambda **_: inputs)
    monkeypatch.setattr(
        scorer,
        "score_direct_official_identity_case",
        lambda **_: {"gate_score": {"primary_chamfer_m": 1.0}},
    )
    with pytest.raises(ValueError, match="raw-derived score differs"):
        integrity._recompute_scores_from_raw(artifacts)


def test_wrong_runtime_rejects_before_any_chmod(
    posix_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role_root = posix_tmp_path / "calibration"
    role_root.mkdir()
    payload = role_root / "payload.bin"
    payload.write_bytes(b"still writable")
    initial_payload_mode = payload.stat().st_mode & 0o777
    initial_role_mode = role_root.stat().st_mode & 0o777
    artifacts = integrity._RoleArtifacts(
        held_root=posix_tmp_path,
        role_root=role_root,
        lock_path=posix_tmp_path / "calibration-lock.json",
        lock={},
        case_names=(),
        paths={},
        barrier_one=object(),
        barrier_two=object(),
        evidence_path=role_root / "evidence.json",
        evidence={},
        decision_path=role_root / "decision.json",
        decision={"decision": "GO"},
        execution_completion_path=role_root / "execution.json",
        execution_completion={},
        source_manifest_path=posix_tmp_path / "source.json",
        source_manifest={},
    )
    monkeypatch.setattr(
        integrity, "_validate_role_artifacts", lambda *_a, **_k: artifacts
    )
    monkeypatch.setattr(
        integrity,
        "_held_root_from_lock",
        lambda *_a, **_k: (artifacts.held_root, dict(artifacts.lock)),
    )
    monkeypatch.setattr(integrity, "_recompute_scores_from_raw", lambda *_a: {})
    monkeypatch.setattr(
        integrity,
        "_deployed_source_runtime_bindings",
        lambda **_: (_ for _ in ()).throw(ValueError("wrong pinned runtime")),
    )
    monkeypatch.setattr(
        integrity,
        "_freeze_tree",
        lambda *_a, **_k: pytest.fail("chmod boundary was crossed"),
    )
    with pytest.raises(ValueError, match="wrong pinned runtime"):
        integrity.seal_role_outcome(
            lock_path=artifacts.lock_path,
            role="calibration",
            deployed_code=posix_tmp_path / "code",
            operator_source=posix_tmp_path / "operator.py",
        )
    assert initial_payload_mode & 0o200
    assert payload.stat().st_mode & 0o777 == initial_payload_mode
    assert role_root.stat().st_mode & 0o777 == initial_role_mode


def test_missing_execution_completion_rejects_before_deep_validation_or_chmod(
    posix_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role_root = posix_tmp_path / "calibration"
    role_root.mkdir()
    payload = role_root / "payload.bin"
    payload.write_bytes(b"still writable")
    initial_payload_mode = payload.stat().st_mode & 0o777
    lock_path = posix_tmp_path / "calibration-lock.json"
    monkeypatch.setattr(
        integrity,
        "_held_root_from_lock",
        lambda *_a, **_k: (posix_tmp_path, {"stage": "calibration"}),
    )
    monkeypatch.setattr(integrity, "_deployed_source_runtime_bindings", lambda **_k: {})
    monkeypatch.setattr(
        integrity,
        "_validate_execution_completion",
        lambda **_k: (_ for _ in ()).throw(
            ValueError("canonical execution completion is absent")
        ),
    )
    monkeypatch.setattr(
        integrity,
        "_validate_role_artifacts",
        lambda *_a, **_k: pytest.fail("deep artifact validation was reached"),
    )
    monkeypatch.setattr(
        integrity,
        "_freeze_tree",
        lambda *_a, **_k: pytest.fail("chmod boundary was crossed"),
    )
    with pytest.raises(ValueError, match="execution completion is absent"):
        integrity.seal_role_outcome(
            lock_path=lock_path,
            role="calibration",
            deployed_code=posix_tmp_path / "code",
            operator_source=posix_tmp_path / "operator.py",
        )
    assert initial_payload_mode & 0o200
    assert payload.stat().st_mode & 0o777 == initial_payload_mode


def test_writable_descriptor_into_freeze_root_is_rejected(
    posix_tmp_path: Path,
) -> None:
    root = posix_tmp_path / "role"
    root.mkdir()
    payload = root / "outcome.log"
    payload.write_bytes(b"before seal\n")
    descriptor = os.open(payload, os.O_WRONLY | os.O_APPEND)
    try:
        with pytest.raises(ValueError, match="writable descriptors"):
            integrity._require_no_writable_descriptors_under_root(
                root, phase="test seal"
            )
    finally:
        os.close(descriptor)
    integrity._require_no_writable_descriptors_under_root(root, phase="test seal")


def test_role_top_level_allowlist_rejects_closed_side_artifact(
    posix_tmp_path: Path,
) -> None:
    role_root = posix_tmp_path / "calibration"
    role_root.mkdir()
    for directory in (
        ".shard-0.claim",
        ".shard-1.claim",
        ".v8-outcome-phase.claim",
        "cases",
        "logs",
        "private-targets",
        "query-inputs",
        "query-outputs",
    ):
        (role_root / directory).mkdir()
    evidence_path = role_root / "calibration-score-evidence.json"
    decision_path = role_root / "calibration-gate-decision.json"
    execution_path = role_root / "calibration-execution-completion.json"
    for path in (
        evidence_path,
        decision_path,
        execution_path,
        role_root / "shard-0.lock-verify.log",
        role_root / "shard-1.lock-verify.log",
    ):
        path.write_text("{}\n", encoding="utf-8")
    artifacts = integrity._RoleArtifacts(
        held_root=posix_tmp_path,
        role_root=role_root,
        lock_path=posix_tmp_path / "calibration-lock.json",
        lock={},
        case_names=(),
        paths={},
        barrier_one=object(),
        barrier_two=object(),
        evidence_path=evidence_path,
        evidence={},
        decision_path=decision_path,
        decision={"role": "calibration"},
        execution_completion_path=execution_path,
        execution_completion={},
        source_manifest_path=posix_tmp_path / "source.json",
        source_manifest={},
    )
    integrity._validate_role_top_level(artifacts)
    (role_root / "closed-side-log.txt").write_text("log\n", encoding="utf-8")
    with pytest.raises(ValueError, match="role top-level allowlist changed"):
        integrity._validate_role_top_level(artifacts)


def test_calibration_go_role_seal_freezes_exact_tree_and_binds_execution_marker(
    posix_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held_root = posix_tmp_path / "held-v8"
    role_root = held_root / "calibration"
    role_root.mkdir(parents=True)
    for directory in (
        ".shard-0.claim",
        ".shard-1.claim",
        ".v8-outcome-phase.claim",
        "cases",
        "logs",
        "private-targets",
        "query-inputs",
        "query-outputs",
    ):
        (role_root / directory).mkdir()
    for filename in ("shard-0.lock-verify.log", "shard-1.lock-verify.log"):
        (role_root / filename).write_text("verified\n", encoding="utf-8")

    lock_path = held_root / "calibration-lock.json"
    lock = _write_signed(lock_path, {"stage": "calibration"})
    evidence_path = role_root / "calibration-score-evidence.json"
    evidence = _write_signed(evidence_path, {"gate_result": {"passed": True}})
    decision_path = role_root / "calibration-gate-decision.json"
    decision = _write_signed(
        decision_path,
        {"role": "calibration", "decision": "GO", "gate_result": {"passed": True}},
    )
    execution_path = role_root / "calibration-execution-completion.json"
    execution = _write_signed(execution_path, {"status": "role-execution-complete"})
    source_manifest_path = (
        held_root / "replacement-source" / "manifests" / "aligned-source.json"
    )
    source_manifest_path.parent.mkdir(parents=True)
    source_manifest = _write_signed(
        source_manifest_path,
        {"protocol_id": integrity.PROTOCOL_ID, "role": "calibration"},
    )
    barrier_one = SimpleNamespace(
        barrier_number=1,
        operation=integrity.TARGET_RECONSTRUCTION_OPERATION,
        barrier_sha256="1" * 64,
        ordered_case_names=(),
        ordered_artifact_bindings=(),
    )
    barrier_two = SimpleNamespace(
        barrier_number=2,
        operation=integrity.FUTURE_SCORE_OPERATION,
        barrier_sha256="2" * 64,
        ordered_case_names=(),
        ordered_artifact_bindings=(),
    )
    artifacts = integrity._RoleArtifacts(
        held_root=held_root,
        role_root=role_root,
        lock_path=lock_path,
        lock=lock,
        case_names=(),
        paths={},
        barrier_one=barrier_one,
        barrier_two=barrier_two,
        evidence_path=evidence_path,
        evidence=evidence,
        decision_path=decision_path,
        decision=decision,
        execution_completion_path=execution_path,
        execution_completion=execution,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
    )
    recomputed = {
        "ordered_case_names": [],
        "gate_result": {"passed": True},
        "sealed_score_evidence_exact_match": True,
        "sealed_gate_decision_exact_match": True,
        "canonical_raw_target_query_prediction_reloaded": True,
    }
    monkeypatch.setattr(
        integrity,
        "_held_root_from_lock",
        lambda *_a, **_k: (held_root, lock),
    )
    monkeypatch.setattr(
        integrity, "_deployed_source_runtime_bindings", lambda **_k: {"fake": True}
    )
    monkeypatch.setattr(
        integrity,
        "_validate_execution_completion",
        lambda **_k: (execution_path, execution),
    )
    monkeypatch.setattr(
        integrity, "_validate_role_artifacts", lambda *_a, **_k: artifacts
    )
    monkeypatch.setattr(
        integrity, "_recompute_scores_from_raw", lambda _artifacts: recomputed
    )

    def validate_published(path: Path, **_kwargs: Any) -> dict[str, Any]:
        completion = integrity._load_json(
            path, label="published role completion", required_mode=0o400
        )
        assert (
            completion["execution_completion"]["artifact_sha256"]
            == execution["artifact_sha256"]
        )
        assert (
            completion["information_boundary"] == integrity._ROLE_INFORMATION_BOUNDARY
        )
        observed = integrity._tree_inventory(
            role_root,
            require_sealed=True,
            excluded_relative=path.relative_to(role_root),
            include_root=False,
        )
        assert completion["sealed_role_inventory"] == observed
        return completion

    monkeypatch.setattr(
        integrity, "validate_role_outcome_completion", validate_published
    )
    sealed = integrity.seal_role_outcome(
        lock_path=lock_path,
        role="calibration",
        deployed_code=held_root / f"code-{'a' * 40}",
        operator_source=held_root / "operator.py",
    )
    assert sealed["terminal_outcome"] == "GO"
    assert stat.S_IMODE(os.lstat(role_root).st_mode) == 0o500
    completion_path = integrity.canonical_role_completion_path(held_root, "calibration")
    assert stat.S_IMODE(os.lstat(completion_path).st_mode) == 0o400


def test_terminal_root_completion_detects_postseal_mutation(
    posix_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = posix_tmp_path / "held-v8"
    role_root = root / "calibration"
    role_root.mkdir(parents=True)
    role_completion_path = role_root / "calibration-outcome-integrity-completion.json"
    role_completion = _write_signed(
        role_completion_path,
        {"terminal_outcome": "NO-GO"},
    )
    payload = root / "payload.bin"
    payload.write_bytes(b"terminal payload")
    integrity._freeze_tree(root, freeze_root=True)
    role_record = {
        **integrity._bound_file(
            role_completion_path, label="role completion", required_mode=0o400
        ),
        "artifact_sha256": role_completion["artifact_sha256"],
    }
    inventory = integrity._tree_inventory(root, require_sealed=True, include_root=True)
    completion_path = integrity.canonical_terminal_completion_path(root)
    completion = {
        "schema_version": integrity.SCHEMA_VERSION,
        "artifact_kind": integrity.TERMINAL_COMPLETION_KIND,
        "protocol_id": integrity.PROTOCOL_ID,
        "status": integrity.TERMINAL_COMPLETION_STATUS,
        "held_root": str(root),
        "held_root_mode_octal": "0500",
        "terminal_role": "calibration",
        "terminal_outcome": "NO-GO",
        "role_outcome_completions": [role_record],
        "sealed_held_root_inventory": inventory,
        "deployed_source_operator_runtime": {
            "fake": True,
            "deployed_repository": {"path": str(root / "code-fake")},
        },
        "information_boundary": {
            "held_root_frozen_before_outside_completion": True,
            "all_root_files_mode_0400": True,
            "all_root_directories_mode_0500": True,
            "root_links_nonregular_and_hardlinks_rejected": True,
            "exact_terminal_top_level_allowlist_validated": True,
            "all_role_scores_deeply_recomputed_before_root_freeze": True,
            "all_role_source_manifests_deeply_revalidated": True,
            "single_use_role_sealer_capability_consumed": True,
            "no_inherited_writable_descriptor_into_held_root": True,
            "outside_completion_excluded_from_root_inventory": True,
        },
        "self_hash_contract": integrity._SELF_HASH_CONTRACT,
    }
    completion["artifact_sha256"] = integrity._artifact_sha256(completion)
    integrity._write_new_json(completion_path, completion)
    monkeypatch.setattr(
        integrity,
        "_terminal_role_completion_records",
        lambda *_a, **_k: [role_record],
    )
    monkeypatch.setattr(
        integrity,
        "_validate_source_runtime_bindings",
        lambda *_a, **_k: {"fake": True},
    )
    monkeypatch.setattr(
        integrity, "_validate_terminal_top_level", lambda *_a, **_k: None
    )
    from bayesian_phystwin import deform360_held_v8_protocol as protocol

    monkeypatch.setattr(protocol, "validate_protocol_lock", lambda *_a, **_k: {})
    integrity.validate_terminal_root_completion(
        completion_path,
        held_root=root,
        verify_content_inventory=True,
        recompute_scores=False,
    )
    root.chmod(0o700)
    payload.chmod(0o600)
    payload.write_bytes(b"mutated terminal payload")
    payload.chmod(0o400)
    root.chmod(0o500)
    with pytest.raises(ValueError, match="inventory changed"):
        integrity.validate_terminal_root_completion(
            completion_path,
            held_root=root,
            verify_content_inventory=True,
            recompute_scores=False,
        )


def test_terminal_sealer_requires_single_use_role_authority_and_deep_recompute(
    posix_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = posix_tmp_path / "held-v8"
    root.mkdir()
    with pytest.raises(ValueError, match="lacks a role-sealer capability"):
        integrity._seal_terminal_held_root(
            held_root=root,
            terminal_role="calibration",
            deployed_code=root / f"code-{'a' * 40}",
            operator_source=root / "operator.py",
            terminal_capability=object(),
        )

    capability = integrity._issue_terminal_seal_capability(
        held_root=root,
        terminal_role="calibration",
        terminal_outcome="NO-GO",
    )

    def require_deep_recompute(
        _root: Path, *, terminal_role: str, recompute_scores: bool
    ) -> list[dict[str, Any]]:
        assert terminal_role == "calibration"
        assert recompute_scores is True
        raise RuntimeError("stopped after proving deep recomputation")

    monkeypatch.setattr(
        integrity, "_terminal_role_completion_records", require_deep_recompute
    )
    from bayesian_phystwin import deform360_held_v8_protocol as protocol

    monkeypatch.setattr(protocol, "validate_protocol_lock", lambda *_a, **_k: {})
    monkeypatch.setattr(integrity, "_deployed_source_runtime_bindings", lambda **_k: {})
    monkeypatch.setattr(
        integrity, "_validate_terminal_top_level", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        integrity,
        "_validate_execution_completion",
        lambda **_k: (
            root / "calibration" / "calibration-execution-completion.json",
            {},
        ),
    )
    with pytest.raises(RuntimeError, match="proving deep recomputation"):
        integrity._seal_terminal_held_root(
            held_root=root,
            terminal_role="calibration",
            deployed_code=root / f"code-{'a' * 40}",
            operator_source=root / "operator.py",
            terminal_capability=capability,
        )
    with pytest.raises(ValueError, match="already used"):
        integrity._seal_terminal_held_root(
            held_root=root,
            terminal_role="calibration",
            deployed_code=root / f"code-{'a' * 40}",
            operator_source=root / "operator.py",
            terminal_capability=capability,
        )


def test_terminal_top_level_allowlist_rejects_unexpected_regular_file(
    posix_tmp_path: Path,
) -> None:
    root = posix_tmp_path / "held-v8"
    root.mkdir()
    deployed = root / f"code-{'a' * 40}"
    for directory in (deployed, root / "replacement-source", root / "calibration"):
        directory.mkdir()
    for filename in (
        "post-withdrawal-development-use-disclosure.json",
        "calibration-lock.json",
    ):
        (root / filename).write_text("{}\n", encoding="utf-8")
    integrity._validate_terminal_top_level(
        root,
        terminal_role="calibration",
        deployed_code=deployed,
    )
    (root / "unexpected-but-regular.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level allowlist changed"):
        integrity._validate_terminal_top_level(
            root,
            terminal_role="calibration",
            deployed_code=deployed,
        )


def test_terminal_public_api_defaults_to_deep_recompute() -> None:
    import inspect

    parameter = inspect.signature(
        integrity.validate_terminal_root_completion
    ).parameters["recompute_scores"]
    assert parameter.default is True
    assert "seal_terminal_held_root" not in integrity.__all__
