"""Write-once score evidence and decisions for Deform360 held v8.

The numerical scorer is intentionally pure.  This module is the small artifact
boundary around it: it verifies that every record was produced only after the
second complete-cohort barrier, evaluates the frozen gate, and seals both the
full evidence and its decision.  No target, query, prediction, or reconstruction
dependency is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Literal

from . import deform360_held_v8_protocol as protocol
from . import deform360_held_v8_scoring as scoring


SCHEMA_VERSION = 1
CALIBRATION_SCORE_EVIDENCE_KIND = "Deform360HeldV8CalibrationScoreEvidence"
CONFIRMATION_SCORE_EVIDENCE_KIND = "Deform360HeldV8ConfirmationScoreEvidence"
CONFIRMATION_DECISION_KIND = "Deform360HeldV8ConfirmationGateDecision"

_SEALED_MODE = 0o400
_ROLE_VALUES = frozenset({"calibration", "confirmation"})
_SCORE_FIELDS = frozenset(
    {
        "primary_chamfer_m",
        "comparator_chamfer_m",
        "primary_identity_rmse_m",
        "comparator_identity_rmse_m",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_regular_file(
    path: str | Path, *, role: str, required_mode: int | None = None
) -> tuple[Path, bytes, os.stat_result]:
    source = _canonical_path(path)
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"{role} is not a regular file",
    )
    _require(source.resolve() == source, f"{role} has a symlinked ancestor")
    if required_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == required_mode,
            f"{role} mode is not exactly {oct(required_mode)}",
        )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{role} changed while opening",
        )
        payload = b""
        while block := os.read(descriptor, 1024 * 1024):
            payload += block
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(source)
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        == (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ),
        f"{role} changed while reading",
    )
    return source, payload, after


def _bound_file(
    path: str | Path, *, role: str, required_mode: int | None = None
) -> dict[str, Any]:
    source, payload, observed = _read_regular_file(
        path, role=role, required_mode=required_mode
    )
    return {
        "path": str(source),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": observed.st_size,
    }


def _load_json(
    path: str | Path, *, role: str, required_mode: int | None = None
) -> dict[str, Any]:
    source, payload, _ = _read_regular_file(
        path, role=role, required_mode=required_mode
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not JSON: {source}") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    return value


def _write_new_json(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    destination = _canonical_path(path)
    parent = destination.parent
    observed_parent = os.lstat(parent)
    _require(
        stat.S_ISDIR(observed_parent.st_mode)
        and not stat.S_ISLNK(observed_parent.st_mode)
        and parent.resolve() == parent,
        "score artifact parent is not a canonical directory",
    )
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        _SEALED_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, _SEALED_MODE, follow_symlinks=False)
    except BaseException:
        if os.path.lexists(destination):
            observed = os.lstat(destination)
            if stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
                os.chmod(destination, 0o600, follow_symlinks=False)
                destination.unlink()
        raise
    return destination


def _expected_case_to_object(
    lock_path: str | Path, role: Literal["calibration", "confirmation"]
) -> dict[str, str]:
    names = protocol.locked_case_names(lock_path, role=role)
    result: dict[str, str] = {}
    for case_name in names:
        object_id, separator, episode = case_name.rpartition("-ep")
        _require(
            bool(object_id)
            and separator == "-ep"
            and len(episode) == 4
            and episode.isdigit(),
            f"locked case name is malformed: {case_name}",
        )
        result[case_name] = object_id
    return result


def _validate_case_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    lock_path: str | Path,
    role: Literal["calibration", "confirmation"],
    barrier_two_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    _require(_valid_sha256(barrier_two_sha256), "barrier-two digest is invalid")
    lock = protocol.validate_protocol_lock(lock_path)
    expected_stage = role
    _require(lock.get("stage") == expected_stage, "score role and lock stage differ")
    expected = _expected_case_to_object(lock_path, role)
    _require(set(records) == set(expected), "score record cohort changed")
    lock_record = _bound_file(lock_path, role="held-v8 score lock", required_mode=0o400)
    ordered: dict[str, dict[str, Any]] = {}
    for case_name, object_id in expected.items():
        record = deepcopy(dict(records[case_name]))
        _require(
            record.get("protocol_id") == protocol.PROTOCOL_ID
            and record.get("scorer_id") == scoring.SCORER_ID
            and record.get("case_name") == case_name
            and record.get("object_id") == object_id,
            f"score record identity changed for {case_name}",
        )
        _require(
            record.get("method_selection_or_tuning_performed") is False,
            f"score record reports tuning for {case_name}",
        )
        gate_score = record.get("gate_score")
        _require(
            isinstance(gate_score, Mapping) and set(gate_score) == _SCORE_FIELDS,
            f"score record gate fields changed for {case_name}",
        )
        permit = record.get("future_score_permit_evidence")
        _require(isinstance(permit, Mapping), f"score permit missing for {case_name}")
        _require(
            permit.get("protocol_id") == protocol.PROTOCOL_ID
            and permit.get("role") == role
            and permit.get("case_name") == case_name
            and permit.get("operation") == protocol.FUTURE_SCORE_OPERATION
            and permit.get("lock_file_sha256") == lock_record["sha256"]
            and permit.get("lock_artifact_sha256") == lock["artifact_sha256"]
            and permit.get("cohort_barrier_sha256") == barrier_two_sha256
            and permit.get("single_use_consumed") is True
            and permit.get("process_local_capability") is True,
            f"score permit binding changed for {case_name}",
        )
        # This is both a finite-JSON check and a defensive conversion away
        # from caller-owned nested mappings.
        try:
            record = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(f"score record is not finite JSON: {case_name}") from error
        ordered[case_name] = record
    return ordered, expected


def _score_evidence_kind(role: str) -> str:
    return (
        CALIBRATION_SCORE_EVIDENCE_KIND
        if role == "calibration"
        else CONFIRMATION_SCORE_EVIDENCE_KIND
    )


def _decision_kind(role: str) -> str:
    return (
        protocol.CALIBRATION_DECISION_KIND
        if role == "calibration"
        else CONFIRMATION_DECISION_KIND
    )


def _decision_label(role: str, passed: bool) -> str:
    if role == "calibration":
        return "GO" if passed else "NO-GO"
    return "CONFIRMED" if passed else "NOT-CONFIRMED"


def create_score_evidence_and_decision(
    evidence_path: str | Path,
    decision_path: str | Path,
    *,
    lock_path: str | Path,
    role: Literal["calibration", "confirmation"],
    barrier_two_sha256: str,
    case_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the frozen role gate and seal evidence before its decision."""

    _require(role in _ROLE_VALUES, "score role is invalid")
    ordered, expected = _validate_case_records(
        case_records,
        lock_path=lock_path,
        role=role,
        barrier_two_sha256=barrier_two_sha256,
    )
    gate = (
        scoring.evaluate_calibration_gate(ordered, expected_case_to_object=expected)
        if role == "calibration"
        else scoring.evaluate_confirmation_gate(
            ordered, expected_case_to_object=expected
        )
    )
    lock_record = _bound_file(
        lock_path, role="held-v8 score lock", required_mode=_SEALED_MODE
    )
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": _score_evidence_kind(role),
        "protocol_id": protocol.PROTOCOL_ID,
        "role": role,
        "lock": lock_record,
        "barrier_two_sha256": barrier_two_sha256,
        "ordered_case_names": list(expected),
        "scorer_id": scoring.SCORER_ID,
        "case_records": ordered,
        "gate_result": gate,
        "information_boundary": {
            "complete_query_cohort_sealed_before_future_score": True,
            "future_targets_read_only_after_case_capability": True,
            "single_shared_mask_for_both_arms": True,
            "identity_transport_or_assignment_performed": False,
            "method_selection_or_tuning_performed": False,
        },
    }
    evidence["artifact_sha256"] = protocol.held_artifact_sha256(evidence)
    _write_new_json(evidence_path, evidence)
    sealed_evidence = validate_score_evidence(
        evidence_path,
        lock_path=lock_path,
        expected_role=role,
        expected_barrier_two_sha256=barrier_two_sha256,
    )

    decision: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": _decision_kind(role),
        "protocol_id": protocol.PROTOCOL_ID,
        "role": role,
        "lock": lock_record,
        "ordered_case_names": list(expected),
        "barrier_two_sha256": barrier_two_sha256,
        "score_evidence": _bound_file(
            evidence_path,
            role="held-v8 score evidence",
            required_mode=_SEALED_MODE,
        ),
        "score_evidence_artifact_sha256": sealed_evidence["artifact_sha256"],
        "gate_result": gate,
        "decision": _decision_label(role, bool(gate["passed"])),
        "method_selection_or_tuning_performed": False,
    }
    decision["artifact_sha256"] = protocol.held_artifact_sha256(decision)
    try:
        _write_new_json(decision_path, decision)
        sealed_decision = validate_score_decision(
            decision_path, lock_path=lock_path, expected_role=role
        )
    except BaseException:
        # Evidence remains sealed as the immutable record of a decision-write
        # failure; callers must not silently reuse it in a new run.
        raise
    return sealed_evidence, sealed_decision


def validate_score_evidence(
    path: str | Path,
    *,
    lock_path: str | Path,
    expected_role: Literal["calibration", "confirmation"],
    expected_barrier_two_sha256: str | None = None,
) -> dict[str, Any]:
    _require(expected_role in _ROLE_VALUES, "score role is invalid")
    artifact = _load_json(
        path, role="held-v8 score evidence", required_mode=_SEALED_MODE
    )
    lock = protocol.validate_protocol_lock(lock_path)
    expected = _expected_case_to_object(lock_path, expected_role)
    _require(
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_kind") == _score_evidence_kind(expected_role)
        and artifact.get("protocol_id") == protocol.PROTOCOL_ID
        and artifact.get("role") == expected_role,
        "score evidence identity changed",
    )
    barrier = artifact.get("barrier_two_sha256")
    _require(_valid_sha256(barrier), "score evidence barrier digest is invalid")
    if expected_barrier_two_sha256 is not None:
        _require(
            barrier == expected_barrier_two_sha256,
            "score evidence binds another second barrier",
        )
    _require(
        artifact.get("lock")
        == _bound_file(lock_path, role="held-v8 score lock", required_mode=0o400)
        and artifact.get("ordered_case_names") == list(expected)
        and artifact.get("scorer_id") == scoring.SCORER_ID,
        "score evidence lock, cohort, or scorer changed",
    )
    records = artifact.get("case_records")
    _require(isinstance(records, Mapping), "score evidence records are invalid")
    normalized, _ = _validate_case_records(
        records,
        lock_path=lock_path,
        role=expected_role,
        barrier_two_sha256=str(barrier),
    )
    gate = (
        scoring.evaluate_calibration_gate(normalized, expected_case_to_object=expected)
        if expected_role == "calibration"
        else scoring.evaluate_confirmation_gate(
            normalized, expected_case_to_object=expected
        )
    )
    _require(artifact.get("gate_result") == gate, "score evidence gate changed")
    _require(
        lock.get("stage") == expected_role,
        "score evidence role and lock stage differ",
    )
    _require(
        artifact.get("artifact_sha256") == protocol.held_artifact_sha256(artifact),
        "score evidence checksum changed",
    )
    return artifact


def validate_score_decision(
    path: str | Path,
    *,
    lock_path: str | Path,
    expected_role: Literal["calibration", "confirmation"],
) -> dict[str, Any]:
    _require(expected_role in _ROLE_VALUES, "score role is invalid")
    decision = _load_json(
        path, role="held-v8 score decision", required_mode=_SEALED_MODE
    )
    expected = _expected_case_to_object(lock_path, expected_role)
    _require(
        decision.get("schema_version") == SCHEMA_VERSION
        and decision.get("artifact_kind") == _decision_kind(expected_role)
        and decision.get("protocol_id") == protocol.PROTOCOL_ID
        and decision.get("role") == expected_role,
        "score decision identity changed",
    )
    barrier = decision.get("barrier_two_sha256")
    _require(_valid_sha256(barrier), "score decision barrier digest is invalid")
    _require(
        decision.get("lock")
        == _bound_file(lock_path, role="held-v8 score lock", required_mode=0o400)
        and decision.get("ordered_case_names") == list(expected),
        "score decision lock or cohort changed",
    )
    evidence_record = decision.get("score_evidence")
    _require(isinstance(evidence_record, Mapping), "score evidence binding is invalid")
    evidence_path = evidence_record.get("path")
    _require(
        evidence_record
        == _bound_file(
            evidence_path,
            role="held-v8 score evidence",
            required_mode=_SEALED_MODE,
        ),
        "score evidence bytes changed",
    )
    evidence = validate_score_evidence(
        evidence_path,
        lock_path=lock_path,
        expected_role=expected_role,
        expected_barrier_two_sha256=str(barrier),
    )
    _require(
        decision.get("score_evidence_artifact_sha256") == evidence["artifact_sha256"]
        and decision.get("gate_result") == evidence["gate_result"],
        "decision and score evidence differ",
    )
    expected_label = _decision_label(
        expected_role, bool(evidence["gate_result"]["passed"])
    )
    _require(
        decision.get("decision") == expected_label
        and decision.get("method_selection_or_tuning_performed") is False,
        "score decision label or tuning boundary changed",
    )
    _require(
        decision.get("artifact_sha256") == protocol.held_artifact_sha256(decision),
        "score decision checksum changed",
    )
    if expected_role == "calibration":
        protocol.validate_calibration_gate_decision(path, lock_path)
    return decision


__all__ = [
    "CALIBRATION_SCORE_EVIDENCE_KIND",
    "CONFIRMATION_DECISION_KIND",
    "CONFIRMATION_SCORE_EVIDENCE_KIND",
    "create_score_evidence_and_decision",
    "validate_score_decision",
    "validate_score_evidence",
]
