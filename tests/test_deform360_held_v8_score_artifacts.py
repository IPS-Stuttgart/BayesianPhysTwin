from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bayesian_phystwin import deform360_held_v8_score_artifacts as artifacts


def _install_fake_protocol(
    monkeypatch: pytest.MonkeyPatch,
    lock_path: Path,
    *,
    role: str,
    case_names: tuple[str, ...],
) -> dict[str, object]:
    lock = {
        "protocol_id": artifacts.protocol.PROTOCOL_ID,
        "artifact_sha256": "a" * 64,
        "stage": role,
    }
    lock_path.write_text("sealed fake lock\n", encoding="utf-8")
    os.chmod(lock_path, 0o400)
    monkeypatch.setattr(
        artifacts.protocol,
        "validate_protocol_lock",
        lambda path: lock,
    )
    monkeypatch.setattr(
        artifacts.protocol,
        "locked_case_names",
        lambda path, *, role: case_names,
    )
    monkeypatch.setattr(
        artifacts.protocol,
        "validate_calibration_gate_decision",
        lambda path, lock_path: {"validated": True},
    )
    return lock


def _records(
    lock_path: Path,
    lock: dict[str, object],
    *,
    role: str,
    case_names: tuple[str, ...],
    barrier: str,
) -> dict[str, dict[str, object]]:
    lock_sha = artifacts._bound_file(lock_path, role="test lock")["sha256"]
    result: dict[str, dict[str, object]] = {}
    for case_name in case_names:
        object_id = case_name.rpartition("-ep")[0]
        result[case_name] = {
            "protocol_id": artifacts.protocol.PROTOCOL_ID,
            "scorer_id": artifacts.scoring.SCORER_ID,
            "case_name": case_name,
            "object_id": object_id,
            "gate_score": {
                "primary_chamfer_m": 0.9,
                "comparator_chamfer_m": 1.0,
                "primary_identity_rmse_m": 0.8,
                "comparator_identity_rmse_m": 1.0,
            },
            "method_selection_or_tuning_performed": False,
            "future_score_permit_evidence": {
                "protocol_id": artifacts.protocol.PROTOCOL_ID,
                "role": role,
                "case_name": case_name,
                "operation": artifacts.protocol.FUTURE_SCORE_OPERATION,
                "lock_file_sha256": lock_sha,
                "lock_artifact_sha256": lock["artifact_sha256"],
                "cohort_barrier_sha256": barrier,
                "single_use_consumed": True,
                "process_local_capability": True,
            },
        }
    return result


@pytest.mark.parametrize(
    ("role", "case_names", "passed", "decision"),
    [
        (
            "calibration",
            tuple(f"object-{index:02d}-ep0000" for index in range(15)),
            True,
            "GO",
        ),
        (
            "confirmation",
            tuple(f"object-{index:02d}-ep0000" for index in range(6)),
            False,
            "NOT-CONFIRMED",
        ),
    ],
)
def test_create_and_revalidate_score_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    case_names: tuple[str, ...],
    passed: bool,
    decision: str,
) -> None:
    lock_path = tmp_path / "lock.json"
    lock = _install_fake_protocol(
        monkeypatch, lock_path, role=role, case_names=case_names
    )
    gate = {"gate": f"fake-{role}", "passed": passed, "aggregation": {}}
    gate_name = (
        "evaluate_calibration_gate"
        if role == "calibration"
        else "evaluate_confirmation_gate"
    )
    monkeypatch.setattr(artifacts.scoring, gate_name, lambda *args, **kwargs: gate)
    barrier = "b" * 64
    evidence_path = tmp_path / "score-evidence.json"
    decision_path = tmp_path / "decision.json"
    records = _records(
        lock_path,
        lock,
        role=role,
        case_names=case_names,
        barrier=barrier,
    )

    evidence, result = artifacts.create_score_evidence_and_decision(
        evidence_path,
        decision_path,
        lock_path=lock_path,
        role=role,
        barrier_two_sha256=barrier,
        case_records=records,
    )

    assert evidence["ordered_case_names"] == list(case_names)
    assert result["decision"] == decision
    assert result["gate_result"] == gate
    assert evidence_path.stat().st_mode & 0o777 == 0o400
    assert decision_path.stat().st_mode & 0o777 == 0o400
    assert (
        artifacts.validate_score_decision(
            decision_path, lock_path=lock_path, expected_role=role
        )
        == result
    )


def test_permit_barrier_mismatch_fails_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role = "confirmation"
    case_names = tuple(f"object-{index:02d}-ep0000" for index in range(6))
    lock_path = tmp_path / "lock.json"
    lock = _install_fake_protocol(
        monkeypatch, lock_path, role=role, case_names=case_names
    )
    records = _records(
        lock_path,
        lock,
        role=role,
        case_names=case_names,
        barrier="b" * 64,
    )
    records[case_names[0]]["future_score_permit_evidence"]["cohort_barrier_sha256"] = (
        "c" * 64
    )
    evidence_path = tmp_path / "score-evidence.json"
    decision_path = tmp_path / "decision.json"

    with pytest.raises(ValueError, match="score permit binding changed"):
        artifacts.create_score_evidence_and_decision(
            evidence_path,
            decision_path,
            lock_path=lock_path,
            role=role,
            barrier_two_sha256="b" * 64,
            case_records=records,
        )

    assert not evidence_path.exists()
    assert not decision_path.exists()


def test_sealed_evidence_corruption_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role = "confirmation"
    case_names = tuple(f"object-{index:02d}-ep0000" for index in range(6))
    lock_path = tmp_path / "lock.json"
    lock = _install_fake_protocol(
        monkeypatch, lock_path, role=role, case_names=case_names
    )
    gate = {"gate": "fake-confirmation", "passed": True, "aggregation": {}}
    monkeypatch.setattr(
        artifacts.scoring, "evaluate_confirmation_gate", lambda *args, **kwargs: gate
    )
    evidence_path = tmp_path / "score-evidence.json"
    decision_path = tmp_path / "decision.json"
    artifacts.create_score_evidence_and_decision(
        evidence_path,
        decision_path,
        lock_path=lock_path,
        role=role,
        barrier_two_sha256="b" * 64,
        case_records=_records(
            lock_path,
            lock,
            role=role,
            case_names=case_names,
            barrier="b" * 64,
        ),
    )
    os.chmod(evidence_path, 0o600)
    value = json.loads(evidence_path.read_text(encoding="utf-8"))
    value["barrier_two_sha256"] = "c" * 64
    evidence_path.write_text(json.dumps(value), encoding="utf-8")
    os.chmod(evidence_path, 0o400)

    with pytest.raises(ValueError, match="score evidence bytes changed"):
        artifacts.validate_score_decision(
            decision_path, lock_path=lock_path, expected_role=role
        )
