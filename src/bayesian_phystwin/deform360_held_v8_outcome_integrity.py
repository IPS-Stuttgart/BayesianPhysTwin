"""Post-outcome integrity sealing for the prospective Deform360 held v8.1.

The outcome operator deliberately writes the scientific evidence before it
knows whether a role passes its gate.  This module is the independent,
repeatable closer for that evidence.  It replays both complete-cohort
barriers, reloads the three canonical raw artifacts used by the scorer,
recomputes every case and the frozen gate, and only then makes the role tree
immutable.

No protocol capability is issued here.  Capabilities are process-local and
single-use, so issuing them would make an integrity validator non-idempotent.
Instead, the already-sealed permit evidence is checked exactly and the raw
artifacts are opened through their ordinary semantic validators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import sys
from typing import Any, Literal


SCHEMA_VERSION = 1
PROTOCOL_ID = "deform360-held-online-belief-v8.2"
TARGET_RECONSTRUCTION_OPERATION = "create-official-target-v1"
FUTURE_SCORE_OPERATION = "read-official-target-for-score-v1"
ROLE_COMPLETION_KIND = "Deform360HeldV8RoleOutcomeIntegrityCompletion"
TERMINAL_COMPLETION_KIND = "Deform360HeldV8TerminalRootIntegrityCompletion"
ROLE_COMPLETION_STATUS = "role-outcome-integrity-complete"
TERMINAL_COMPLETION_STATUS = "terminal-root-integrity-complete"
ROLE_COMPLETION_SUFFIX = "-outcome-integrity-completion.json"
TERMINAL_COMPLETION_SUFFIX = "-terminal-integrity-completion.json"

_ROLES = frozenset({"calibration", "confirmation"})
_TERMINAL_OUTCOMES = frozenset({"NO-GO", "CONFIRMED", "NOT-CONFIRMED"})
_ROLE_OUTCOMES = {
    "calibration": frozenset({"GO", "NO-GO"}),
    "confirmation": frozenset({"CONFIRMED", "NOT-CONFIRMED"}),
}
_FILE_MODE = 0o400
_DIRECTORY_MODE = 0o500
_SHA256_LENGTH = 64
_OPERATOR_RELATIVE_PATH = Path("scripts/held/seal_deform360_v8_role_outcome.py")
_SOURCE_RELATIVE_PATHS = (
    Path("src/bayesian_phystwin/deform360_held_v8_outcome_integrity.py"),
    Path("src/bayesian_phystwin/deform360_held_v8_outcome_driver.py"),
    Path("src/bayesian_phystwin/deform360_held_v8_protocol.py"),
    Path("src/bayesian_phystwin/deform360_held_v8_outcome_artifacts.py"),
    Path("src/bayesian_phystwin/deform360_held_v8_query_artifacts.py"),
    Path("src/bayesian_phystwin/deform360_held_v8_scoring.py"),
    Path("src/bayesian_phystwin/deform360_held_v8_score_artifacts.py"),
    Path("src/bayesian_phystwin/deform360_held_v8_confirmation_source.py"),
    Path("scripts/held/run_deform360_v8_confirmation_source.py"),
)
_FILE_RECORD_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
}
_PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
_PINNED_PYTHON_RESOLVED = Path("/usr/bin/python3.12")
_PINNED_NUMPY_VERSION = "1.26.4"
_EXPECTED_HOST = "workstation2"
_NAMED_SOURCE_BINDINGS = {
    "held_v8_outcome_integrity_source": _SOURCE_RELATIVE_PATHS[0],
    "held_v8_outcome_driver_source": _SOURCE_RELATIVE_PATHS[1],
    "held_v8_role_outcome_integrity_sealer_source": _OPERATOR_RELATIVE_PATH,
    "held_v8_confirmation_source_operator_source": _SOURCE_RELATIVE_PATHS[-2],
    "held_v8_confirmation_source_materialization_launcher_source": (
        _SOURCE_RELATIVE_PATHS[-1]
    ),
}
_TERMINAL_SEAL_CAPABILITY_AUTHORITY = object()


@dataclass(frozen=True)
class _TerminalSealCapability:
    held_root: str
    terminal_role: str
    terminal_outcome: str
    _nonce: object
    _authority: object


@dataclass
class _TerminalSealCapabilityState:
    capability: _TerminalSealCapability
    consumed: bool = False


_TERMINAL_SEAL_CAPABILITIES: dict[int, _TerminalSealCapabilityState] = {}
_ROLE_INFORMATION_BOUNDARY = {
    "protocol_capabilities_issued_by_integrity_operator": False,
    "both_protocol_barriers_reconstructed": True,
    "canonical_raw_target_query_prediction_reloaded": True,
    "every_case_score_recomputed": True,
    "frozen_gate_recomputed": True,
    "sealed_score_and_decision_exact_compared": True,
    "durable_execution_completion_validated": True,
    "canonical_role_source_manifest_deeply_revalidated": True,
    "live_nofile_pair_rechecked_before_role_sealing": True,
    "pinned_runtime_verified_before_future_recomputation": True,
    "no_inherited_writable_descriptor_into_role_tree": True,
    "exact_role_top_level_allowlist_validated": True,
    "role_completion_excluded_from_self_referential_inventory": True,
    "role_tree_nonregular_links_and_hardlinks_rejected": True,
}
_SELF_HASH_CONTRACT = (
    "artifact_sha256-is-sha256-of-canonical-json-with-artifact_sha256-omitted-v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("integrity evidence is not canonical finite JSON") from error


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = deepcopy(dict(value))
    unsigned.pop("artifact_sha256", None)
    return _digest(unsigned)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file(
    path: str | os.PathLike[str],
    *,
    label: str,
    required_mode: int | None = None,
) -> tuple[Path, bytes, os.stat_result]:
    source = _canonical_path(path)
    try:
        before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} is absent or inaccessible: {source}") from error
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"{label} is not a regular non-link file",
    )
    _require(before.st_nlink == 1, f"{label} is hard-linked")
    _require(source.resolve(strict=True) == source, f"{label} has a linked ancestor")
    if required_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == required_mode,
            f"{label} mode is not {required_mode:04o}",
        )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and _stable_stat(opened) == _stable_stat(before),
            f"{label} changed while opening",
        )
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
        after = os.fstat(descriptor)
        current = os.lstat(source)
        _require(
            _stable_stat(opened) == _stable_stat(after) == _stable_stat(current),
            f"{label} changed while reading",
        )
    finally:
        os.close(descriptor)
    return source, bytes(payload), after


def _bound_file(
    path: str | os.PathLike[str],
    *,
    label: str,
    required_mode: int | None = None,
) -> dict[str, Any]:
    source, payload, observed = _read_regular_file(
        path, label=label, required_mode=required_mode
    )
    return {
        "path": str(source),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": observed.st_size,
    }


def _load_json(
    path: str | os.PathLike[str],
    *,
    label: str,
    required_mode: int | None = None,
) -> dict[str, Any]:
    source, payload, _ = _read_regular_file(
        path, label=label, required_mode=required_mode
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON: {source}") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _artifact_record(
    path: str | os.PathLike[str], artifact: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    artifact_sha256 = artifact.get("artifact_sha256")
    _require(_valid_sha256(artifact_sha256), f"{label} artifact digest is invalid")
    return {
        **_bound_file(path, label=label, required_mode=_FILE_MODE),
        "artifact_sha256": artifact_sha256,
    }


def _validate_file_record(record: object, *, label: str) -> dict[str, Any]:
    _require(
        isinstance(record, Mapping) and set(record) == _FILE_RECORD_FIELDS,
        f"{label} file record fields changed",
    )
    path = record.get("path")
    _require(isinstance(path, str) and path, f"{label} path is absent")
    observed = _bound_file(path, label=label, required_mode=_FILE_MODE)
    _require(observed == dict(record), f"{label} file binding changed")
    return observed


def canonical_role_completion_path(
    held_root: str | os.PathLike[str], role: Literal["calibration", "confirmation"]
) -> Path:
    _require(role in _ROLES, "outcome role is invalid")
    root = _canonical_path(held_root)
    return root / role / f"{role}{ROLE_COMPLETION_SUFFIX}"


def canonical_terminal_completion_path(
    held_root: str | os.PathLike[str],
) -> Path:
    root = _canonical_path(held_root)
    return Path(f"{root}{TERMINAL_COMPLETION_SUFFIX}")


def _held_root_from_lock(
    lock_path: str | os.PathLike[str], *, expected_role: str
) -> tuple[Path, dict[str, Any]]:
    from . import deform360_held_v8_protocol as protocol

    _require(expected_role in _ROLES, "outcome role is invalid")
    lock_path_canonical = _canonical_path(lock_path)
    lock = protocol.validate_protocol_lock(lock_path_canonical)
    _require(lock.get("stage") == expected_role, "lock and outcome role differ")
    held_root = _canonical_path(str(lock.get("held_root")))
    _require(
        lock_path_canonical == held_root / f"{expected_role}-lock.json",
        "role lock is not at its exact canonical path",
    )
    root_state = os.lstat(held_root)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and held_root.resolve(strict=True) == held_root,
        "held root is absent, linked, or non-canonical",
    )
    return held_root, lock


def _case_paths(held_root: Path, role: str, case_name: str) -> dict[str, Path]:
    case_root = held_root / role / "cases" / case_name
    private = held_root / role / "private-targets" / case_name
    query = held_root / role / "query-inputs" / case_name
    queried = held_root / role / "query-outputs" / case_name
    return {
        "physical": case_root / "physical" / "physical_prior_seal.json",
        "online": case_root / "online" / "online_prediction_seal.json",
        "field": (case_root / "frozen-field" / "preoutcome-frozen-field-manifest.json"),
        "target": private / "official-target-manifest.json",
        "query": query / "official-frame-zero-query-manifest.json",
        "queried": queried / "queried-prediction-seal.json",
    }


@dataclass(frozen=True)
class _RoleArtifacts:
    held_root: Path
    role_root: Path
    lock_path: Path
    lock: Mapping[str, Any]
    case_names: tuple[str, ...]
    paths: Mapping[str, Mapping[str, Path]]
    barrier_one: Any
    barrier_two: Any
    evidence_path: Path
    evidence: Mapping[str, Any]
    decision_path: Path
    decision: Mapping[str, Any]
    execution_completion_path: Path
    execution_completion: Mapping[str, Any]
    source_manifest_path: Path
    source_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _RoleMetadata:
    held_root: Path
    role_root: Path
    lock_path: Path
    lock: Mapping[str, Any]
    evidence_path: Path
    evidence: Mapping[str, Any]
    decision_path: Path
    decision: Mapping[str, Any]
    execution_completion_path: Path
    execution_completion: Mapping[str, Any]
    source_manifest_path: Path
    source_manifest: Mapping[str, Any]


def _validate_unsigned_artifact(value: Mapping[str, Any], *, label: str) -> None:
    _require(
        value.get("artifact_sha256") == _artifact_sha256(value),
        f"{label} artifact self-hash changed",
    )


def _validate_execution_completion(
    *,
    held_root: Path,
    lock_path: Path,
    expected_role: str,
    expected_ordered_case_names: Sequence[str] | None = None,
    verify_live_rlimit: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Validate the driver's durable proof that execution reached its boundary."""

    from . import deform360_held_v8_outcome_driver as outcome_driver

    _require(expected_role in _ROLES, "execution-completion role is invalid")
    path = outcome_driver.canonical_role_execution_completion_path(
        held_root, expected_role
    )
    completion = outcome_driver.validate_role_execution_completion(
        path,
        lock_path=lock_path,
        expected_role=expected_role,  # type: ignore[arg-type]
        expected_ordered_case_names=expected_ordered_case_names,
    )
    _require(
        isinstance(completion, dict),
        "execution-completion validator returned invalid data",
    )
    if verify_live_rlimit:
        import resource

        boundary = completion.get("resource_boundary")
        initial = (
            boundary.get("initial_nofile") if isinstance(boundary, Mapping) else None
        )
        _require(
            isinstance(initial, Mapping)
            and type(initial.get("rlimit_nofile_soft")) is int
            and type(initial.get("rlimit_nofile_hard")) is int,
            "execution completion lacks its initial NOFILE pair",
        )
        current = tuple(
            int(value) for value in resource.getrlimit(resource.RLIMIT_NOFILE)
        )
        expected = (
            initial["rlimit_nofile_soft"],
            initial["rlimit_nofile_hard"],
        )
        _require(
            current == expected,
            "live integrity-sealer RLIMIT_NOFILE pair differs from completed execution",
        )
    return path, completion


def _load_role_metadata(
    lock_path: str | os.PathLike[str], *, expected_role: str
) -> _RoleMetadata:
    """Load only sealed JSON metadata; never import a future-bearing module."""

    _require(expected_role in _ROLES, "outcome role is invalid")
    lock_path_canonical = _canonical_path(lock_path)
    lock = _load_json(
        lock_path_canonical, label="held-v8 role lock", required_mode=_FILE_MODE
    )
    _validate_unsigned_artifact(lock, label="held-v8 role lock")
    held_root_value = lock.get("held_root")
    _require(
        lock.get("protocol_id") == PROTOCOL_ID
        and lock.get("stage") == expected_role
        and isinstance(held_root_value, str),
        "held-v8 role lock identity changed",
    )
    held_root = _canonical_path(held_root_value)
    _require(
        lock_path_canonical == held_root / f"{expected_role}-lock.json",
        "role lock is not at its canonical path",
    )
    role_root = held_root / expected_role
    evidence_path = role_root / f"{expected_role}-score-evidence.json"
    decision_path = role_root / f"{expected_role}-gate-decision.json"
    evidence = _load_json(
        evidence_path, label="held-v8 score evidence", required_mode=_FILE_MODE
    )
    decision = _load_json(
        decision_path, label="held-v8 gate decision", required_mode=_FILE_MODE
    )
    _validate_unsigned_artifact(evidence, label="held-v8 score evidence")
    _validate_unsigned_artifact(decision, label="held-v8 gate decision")
    expected_evidence_kind = (
        "Deform360HeldV8CalibrationScoreEvidence"
        if expected_role == "calibration"
        else "Deform360HeldV8ConfirmationScoreEvidence"
    )
    expected_decision_kind = (
        "Deform360HeldV8CalibrationGateDecision"
        if expected_role == "calibration"
        else "Deform360HeldV8ConfirmationGateDecision"
    )
    _require(
        evidence.get("protocol_id") == PROTOCOL_ID
        and evidence.get("role") == expected_role
        and evidence.get("artifact_kind") == expected_evidence_kind
        and decision.get("protocol_id") == PROTOCOL_ID
        and decision.get("role") == expected_role
        and decision.get("artifact_kind") == expected_decision_kind
        and decision.get("decision") in _ROLE_OUTCOMES[expected_role],
        "score evidence or decision identity changed",
    )
    lock_record = _bound_file(
        lock_path_canonical, label="held-v8 role lock", required_mode=_FILE_MODE
    )
    evidence_record = _bound_file(
        evidence_path, label="held-v8 score evidence", required_mode=_FILE_MODE
    )
    _require(
        evidence.get("lock") == lock_record
        and decision.get("lock") == lock_record
        and decision.get("score_evidence") == evidence_record
        and decision.get("score_evidence_artifact_sha256")
        == evidence.get("artifact_sha256")
        and decision.get("barrier_two_sha256") == evidence.get("barrier_two_sha256")
        and decision.get("gate_result") == evidence.get("gate_result"),
        "score evidence/decision/lock metadata cross-link changed",
    )
    execution_completion_path, execution_completion = _validate_execution_completion(
        held_root=held_root,
        lock_path=lock_path_canonical,
        expected_role=expected_role,
    )
    source_manifest_path, source_manifest = _validate_role_source_manifest(
        held_root=held_root,
        lock_path=lock_path_canonical,
        expected_role=expected_role,
    )
    _require(
        execution_completion.get("source_manifest")
        == _artifact_record(
            source_manifest_path,
            source_manifest,
            label=f"{expected_role} role source manifest",
        ),
        "execution completion source-manifest binding changed",
    )
    return _RoleMetadata(
        held_root=held_root,
        role_root=role_root,
        lock_path=lock_path_canonical,
        lock=lock,
        evidence_path=evidence_path,
        evidence=evidence,
        decision_path=decision_path,
        decision=decision,
        execution_completion_path=execution_completion_path,
        execution_completion=execution_completion,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
    )


def _role_source_manifest_path(held_root: Path, role: str) -> Path:
    _require(role in _ROLES, "role source manifest received an invalid role")
    path = (
        held_root / "replacement-source" / "manifests" / "aligned-source.json"
        if role == "calibration"
        else (
            held_root
            / "confirmation-source"
            / "manifests"
            / "aligned-source-cohort.json"
        )
    )
    _require(
        _canonical_path(path) == path,
        "role source manifest path is not canonical",
    )
    return path


def _validate_role_source_manifest(
    *,
    held_root: Path,
    lock_path: Path,
    expected_role: str,
) -> tuple[Path, dict[str, Any]]:
    from . import deform360_held_v8_confirmation_source as confirmation_source
    from . import deform360_held_v8_protocol as protocol
    from . import deform360_held_v8_replacement_source as replacement_source

    path = _role_source_manifest_path(held_root, expected_role)
    if expected_role == "calibration":
        expected_permit = protocol.replacement_source_permit_evidence(lock_path)
        source = replacement_source.validate_aligned_source_manifest(
            path,
            expected_source_permit=expected_permit,
        )
        _require(
            isinstance(source, Mapping)
            and source.get("protocol_id") == PROTOCOL_ID
            and source.get("case_name") == protocol.FRESH_REPLACEMENT_CASE_NAME
            and source.get("source_permit") == expected_permit,
            "calibration source manifest identity changed",
        )
    else:
        expected_permit = protocol.confirmation_source_permit_evidence(lock_path)
        source = confirmation_source.validate_confirmation_source_cohort_manifest(
            path,
            expected_source_permit=expected_permit,
            verify_content=True,
        )
        _require(
            isinstance(source, Mapping)
            and source.get("protocol_id") == PROTOCOL_ID
            and source.get("role") == "confirmation"
            and source.get("ordered_case_names")
            == list(protocol.CONFIRMATION_CASE_NAMES)
            and source.get("confirmation_lock_and_capability") == expected_permit,
            "confirmation source manifest identity changed",
        )
    _validate_unsigned_artifact(source, label=f"{expected_role} source manifest")
    return path, dict(source)


def _validate_role_artifacts(
    lock_path: str | os.PathLike[str], *, expected_role: str
) -> _RoleArtifacts:
    from . import deform360_held_v8_protocol as protocol
    from . import deform360_held_v8_score_artifacts as score_artifacts

    held_root, lock = _held_root_from_lock(lock_path, expected_role=expected_role)
    lock_path_canonical = _canonical_path(lock_path)
    case_names = tuple(
        protocol.locked_case_names(lock_path_canonical, role=expected_role)
    )
    expected_count = 15 if expected_role == "calibration" else 6
    _require(
        len(case_names) == expected_count, "locked role cohort cardinality changed"
    )
    execution_completion_path, execution_completion = _validate_execution_completion(
        held_root=held_root,
        lock_path=lock_path_canonical,
        expected_role=expected_role,
        expected_ordered_case_names=case_names,
    )
    source_manifest_path, source_manifest = _validate_role_source_manifest(
        held_root=held_root,
        lock_path=lock_path_canonical,
        expected_role=expected_role,
    )
    _require(
        execution_completion.get("source_manifest")
        == _artifact_record(
            source_manifest_path,
            source_manifest,
            label=f"{expected_role} role source manifest",
        ),
        "execution completion source-manifest binding changed",
    )
    paths = {
        case_name: _case_paths(held_root, expected_role, case_name)
        for case_name in case_names
    }
    barrier_one = protocol.validate_first_cohort_barrier(
        lock_path_canonical,
        physical_seal_paths={case: paths[case]["physical"] for case in case_names},
        online_seal_paths={case: paths[case]["online"] for case in case_names},
        frozen_field_manifest_paths={case: paths[case]["field"] for case in case_names},
        replacement_aligned_source_manifest_path=(
            source_manifest_path if expected_role == "calibration" else None
        ),
        confirmation_aligned_source_manifest_path=(
            source_manifest_path if expected_role == "confirmation" else None
        ),
        role=expected_role,
    )
    barrier_two = protocol.validate_second_cohort_barrier(
        lock_path_canonical,
        official_query_manifest_paths={
            case: paths[case]["query"] for case in case_names
        },
        queried_prediction_seal_paths={
            case: paths[case]["queried"] for case in case_names
        },
        role=expected_role,
    )
    role_root = held_root / expected_role
    evidence_path = role_root / f"{expected_role}-score-evidence.json"
    decision_path = role_root / f"{expected_role}-gate-decision.json"
    evidence = score_artifacts.validate_score_evidence(
        evidence_path,
        lock_path=lock_path_canonical,
        expected_role=expected_role,
        expected_barrier_two_sha256=barrier_two.barrier_sha256,
    )
    decision = score_artifacts.validate_score_decision(
        decision_path,
        lock_path=lock_path_canonical,
        expected_role=expected_role,
    )
    _require(
        decision.get("barrier_two_sha256") == barrier_two.barrier_sha256
        and evidence.get("barrier_two_sha256") == barrier_two.barrier_sha256,
        "score artifacts do not bind the reconstructed second barrier",
    )
    terminal_outcome = decision.get("decision")
    _require(
        terminal_outcome in _ROLE_OUTCOMES[expected_role],
        "role decision label is invalid",
    )
    return _RoleArtifacts(
        held_root=held_root,
        role_root=role_root,
        lock_path=lock_path_canonical,
        lock=lock,
        case_names=case_names,
        paths=paths,
        barrier_one=barrier_one,
        barrier_two=barrier_two,
        evidence_path=evidence_path,
        evidence=evidence,
        decision_path=decision_path,
        decision=decision,
        execution_completion_path=execution_completion_path,
        execution_completion=execution_completion,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
    )


def _source_node_count(field_manifest: Mapping[str, Any]) -> int:
    records = field_manifest.get("source_array_records")
    _require(isinstance(records, Mapping), "frozen-field source records are absent")
    record = records.get("frame_zero_points_m")
    _require(isinstance(record, Mapping), "frozen-field source points are absent")
    shape = record.get("shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 2
        and type(shape[0]) is int
        and shape[0] > 0,
        "frozen-field source point count is invalid",
    )
    return shape[0]


def _load_npz_from_record(record: object, *, label: str) -> dict[str, Any]:
    from io import BytesIO

    import numpy as np

    bound = _validate_file_record(record, label=label)
    _, payload, _ = _read_regular_file(
        bound["path"], label=label, required_mode=_FILE_MODE
    )
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{label} is not a canonical non-pickle NPZ") from error
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def _expected_target_permit(
    artifacts: _RoleArtifacts, *, case_name: str
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "role": artifacts.decision["role"],
        "case_name": case_name,
        "operation": TARGET_RECONSTRUCTION_OPERATION,
        "lock_file_sha256": _bound_file(
            artifacts.lock_path, label="held-v8 role lock", required_mode=_FILE_MODE
        )["sha256"],
        "lock_artifact_sha256": artifacts.lock["artifact_sha256"],
        "cohort_barrier_sha256": artifacts.barrier_one.barrier_sha256,
        "single_use_consumed": True,
        "process_local_capability": True,
    }


def _expected_score_permit(
    artifacts: _RoleArtifacts, *, case_name: str
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "role": artifacts.decision["role"],
        "case_name": case_name,
        "operation": FUTURE_SCORE_OPERATION,
        "lock_file_sha256": _bound_file(
            artifacts.lock_path, label="held-v8 role lock", required_mode=_FILE_MODE
        )["sha256"],
        "lock_artifact_sha256": artifacts.lock["artifact_sha256"],
        "cohort_barrier_sha256": artifacts.barrier_two.barrier_sha256,
        "single_use_consumed": True,
        "process_local_capability": True,
        "predecessor_reconstruction_barrier_sha256": (
            artifacts.barrier_one.barrier_sha256
        ),
    }


def _bit_equal(left: Any, right: Any) -> bool:
    import numpy as np

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes()
        == np.ascontiguousarray(right_array).tobytes()
    )


def _recompute_scores_from_raw(artifacts: _RoleArtifacts) -> dict[str, Any]:
    from . import deform360_held_v8_outcome_artifacts as outcome_artifacts
    from . import deform360_held_v8_query_artifacts as query_artifacts
    from . import deform360_held_v8_scoring as scoring

    sealed_records = artifacts.evidence.get("case_records")
    _require(isinstance(sealed_records, Mapping), "sealed case records are absent")
    recomputed: dict[str, dict[str, Any]] = {}
    case_details: dict[str, dict[str, Any]] = {}
    for case_name in artifacts.case_names:
        paths = artifacts.paths[case_name]
        field_manifest = query_artifacts.validate_preoutcome_frozen_field_manifest(
            paths["field"],
            lock_path=artifacts.lock_path,
            expected_case_name=case_name,
        )
        target_manifest = outcome_artifacts.validate_official_target_artifact(
            paths["target"],
            lock_path=artifacts.lock_path,
            expected_case_name=case_name,
            expected_role=str(artifacts.decision["role"]),
        )
        query_manifest = query_artifacts.validate_official_frame_zero_query_artifact(
            paths["query"], artifacts.lock_path, expected_case_name=case_name
        )
        queried_seal = query_artifacts.validate_queried_prediction_artifact(
            paths["queried"],
            lock_path=artifacts.lock_path,
            expected_case_name=case_name,
        )
        _require(
            target_manifest.get("target_reconstruction_permit_evidence")
            == _expected_target_permit(artifacts, case_name=case_name),
            f"target permit evidence changed for {case_name}",
        )
        sealed_record = sealed_records.get(case_name)
        _require(
            isinstance(sealed_record, Mapping), f"sealed score missing: {case_name}"
        )
        score_permit = sealed_record.get("future_score_permit_evidence")
        _require(
            score_permit == _expected_score_permit(artifacts, case_name=case_name),
            f"score permit evidence changed for {case_name}",
        )
        target_arrays = _load_npz_from_record(
            target_manifest.get("archive"), label=f"official target {case_name}"
        )
        query_arrays = _load_npz_from_record(
            query_manifest.get("archive"), label=f"official query {case_name}"
        )
        queried_arrays = _load_npz_from_record(
            queried_seal.get("archive"), label=f"queried prediction {case_name}"
        )
        _require(
            set(query_arrays) == {"identity_ids", "positions_m"},
            f"official query raw array set changed for {case_name}",
        )
        _require(
            _bit_equal(query_arrays["identity_ids"], target_arrays["identity_ids"])
            and _bit_equal(
                query_arrays["positions_m"], target_arrays["object_points"][0]
            )
            and _bit_equal(query_arrays["identity_ids"], queried_arrays["identity_ids"])
            and _bit_equal(query_arrays["positions_m"], queried_arrays["positions_m"]),
            f"canonical target/query/prediction x0 bytes differ for {case_name}",
        )
        inputs = outcome_artifacts._assemble_direct_scoring_inputs(
            case_name=case_name,
            queried=queried_arrays,
            target=target_arrays,
            source_node_count=_source_node_count(field_manifest),
            permit_evidence=dict(score_permit),
        )
        score = scoring.score_direct_official_identity_case(**inputs.scoring_kwargs())
        score["future_score_permit_evidence"] = dict(score_permit)
        try:
            score = json.loads(json.dumps(score, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"recomputed score is not finite JSON: {case_name}"
            ) from error
        _require(
            score == dict(sealed_record),
            f"raw-derived score differs from sealed evidence for {case_name}",
        )
        recomputed[case_name] = score
        case_details[case_name] = {
            "source_node_count": inputs.source_node_count,
            "gate_score": score["gate_score"],
            "score_record_sha256": _digest(score),
            "frozen_field_manifest": _artifact_record(
                paths["field"], field_manifest, label=f"frozen field {case_name}"
            ),
            "official_target_manifest": _artifact_record(
                paths["target"], target_manifest, label=f"target manifest {case_name}"
            ),
            "official_target_archive": _validate_file_record(
                target_manifest["archive"], label=f"target archive {case_name}"
            ),
            "official_query_manifest": _artifact_record(
                paths["query"], query_manifest, label=f"query manifest {case_name}"
            ),
            "official_query_archive": _validate_file_record(
                query_manifest["archive"], label=f"query archive {case_name}"
            ),
            "queried_prediction_seal": _artifact_record(
                paths["queried"], queried_seal, label=f"queried seal {case_name}"
            ),
            "queried_prediction_archive": _validate_file_record(
                queried_seal["archive"], label=f"queried archive {case_name}"
            ),
        }
    expected_objects = {
        case_name: case_name.rpartition("-ep")[0] for case_name in artifacts.case_names
    }
    gate = (
        scoring.evaluate_calibration_gate(
            recomputed, expected_case_to_object=expected_objects
        )
        if artifacts.decision["role"] == "calibration"
        else scoring.evaluate_confirmation_gate(
            recomputed, expected_case_to_object=expected_objects
        )
    )
    _require(
        gate == artifacts.evidence.get("gate_result")
        and gate == artifacts.decision.get("gate_result"),
        "raw-derived gate differs from sealed evidence or decision",
    )
    expected_decision = (
        "GO"
        if artifacts.decision["role"] == "calibration" and gate["passed"]
        else "NO-GO"
        if artifacts.decision["role"] == "calibration"
        else "CONFIRMED"
        if gate["passed"]
        else "NOT-CONFIRMED"
    )
    _require(
        artifacts.decision.get("decision") == expected_decision,
        "raw-derived gate and sealed decision label differ",
    )
    combined = {
        "ordered_case_names": list(artifacts.case_names),
        "case_records": recomputed,
        "gate_result": gate,
        "decision": expected_decision,
    }
    return {
        "scorer_id": scoring.SCORER_ID,
        "ordered_case_names": list(artifacts.case_names),
        "case_details": case_details,
        "case_record_set_sha256": _digest(recomputed),
        "gate_result": gate,
        "gate_result_sha256": _digest(gate),
        "score_and_gate_sha256": _digest(combined),
        "sealed_score_evidence_exact_match": True,
        "sealed_gate_decision_exact_match": True,
        "canonical_raw_target_query_prediction_reloaded": True,
    }


def _barrier_record(barrier: Any) -> dict[str, Any]:
    bindings = [
        [case_name, [[name, value] for name, value in case_bindings]]
        for case_name, case_bindings in barrier.ordered_artifact_bindings
    ]
    return {
        "barrier_number": barrier.barrier_number,
        "operation": barrier.operation,
        "barrier_sha256": barrier.barrier_sha256,
        "ordered_case_names": list(barrier.ordered_case_names),
        "ordered_artifact_bindings": bindings,
        "ordered_artifact_bindings_sha256": _digest(bindings),
    }


def _entry_metadata(
    relative: str, observed: os.stat_result, *, entry_type: str
) -> dict[str, Any]:
    return {
        "path": relative,
        "type": entry_type,
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
        "nlink": observed.st_nlink,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _walk_inventory(
    descriptor: int,
    *,
    prefix: Path,
    require_sealed: bool,
    excluded_relative: Path | None,
    seen_files: set[tuple[int, int]],
    seen_directories: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(os.listdir(descriptor)):
        _require(name not in {"", ".", ".."} and "/" not in name, "unsafe entry name")
        relative_path = prefix / name
        if excluded_relative is not None and relative_path == excluded_relative:
            continue
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        relative = relative_path.as_posix()
        if stat.S_ISREG(before.st_mode):
            _require(before.st_nlink == 1, f"hard-linked file is forbidden: {relative}")
            identity = (before.st_dev, before.st_ino)
            _require(identity not in seen_files, f"duplicate file inode: {relative}")
            seen_files.add(identity)
            file_descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(file_descriptor)
                _require(
                    _stable_stat(opened) == _stable_stat(before),
                    f"file changed while opening: {relative}",
                )
                digest = hashlib.sha256()
                while block := os.read(file_descriptor, 1024 * 1024):
                    digest.update(block)
                after = os.fstat(file_descriptor)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                _require(
                    _stable_stat(opened)
                    == _stable_stat(after)
                    == _stable_stat(current),
                    f"file changed while inventorying: {relative}",
                )
            finally:
                os.close(file_descriptor)
            if require_sealed:
                _require(
                    stat.S_IMODE(after.st_mode) == _FILE_MODE,
                    f"sealed role file mode changed: {relative}",
                )
            row = _entry_metadata(relative, after, entry_type="file")
            row["sha256"] = digest.hexdigest()
            rows.append(row)
        elif stat.S_ISDIR(before.st_mode):
            identity = (before.st_dev, before.st_ino)
            _require(
                identity not in seen_directories,
                f"duplicate or recursively linked directory: {relative}",
            )
            seen_directories.add(identity)
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                _require(
                    stat.S_ISDIR(opened.st_mode)
                    and (opened.st_dev, opened.st_ino) == identity,
                    f"directory changed while opening: {relative}",
                )
                rows.extend(
                    _walk_inventory(
                        child,
                        prefix=relative_path,
                        require_sealed=require_sealed,
                        excluded_relative=excluded_relative,
                        seen_files=seen_files,
                        seen_directories=seen_directories,
                    )
                )
                after = os.fstat(child)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                _require(
                    _stable_stat(opened)
                    == _stable_stat(after)
                    == _stable_stat(current),
                    f"directory changed while inventorying: {relative}",
                )
            finally:
                os.close(child)
            if require_sealed:
                _require(
                    stat.S_IMODE(after.st_mode) == _DIRECTORY_MODE,
                    f"sealed role directory mode changed: {relative}",
                )
            rows.append(_entry_metadata(relative, after, entry_type="directory"))
        else:
            raise ValueError(
                f"linked or nonregular role entry is forbidden: {relative}"
            )
    return rows


def _tree_inventory(
    root: Path,
    *,
    require_sealed: bool,
    excluded_relative: Path | None = None,
    include_root: bool = False,
) -> dict[str, Any]:
    root = _canonical_path(root)
    before = os.lstat(root)
    _require(
        stat.S_ISDIR(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and root.resolve(strict=True) == root,
        "inventory root is absent, linked, or non-canonical",
    )
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            "inventory root changed while opening",
        )
        seen_directories = {(opened.st_dev, opened.st_ino)}
        rows = _walk_inventory(
            descriptor,
            prefix=Path(),
            require_sealed=require_sealed,
            excluded_relative=excluded_relative,
            seen_files=set(),
            seen_directories=seen_directories,
        )
        after = os.fstat(descriptor)
        current = os.lstat(root)
        _require(
            _stable_stat(opened) == _stable_stat(after) == _stable_stat(current),
            "inventory root changed while scanning",
        )
    finally:
        os.close(descriptor)
    if require_sealed:
        _require(
            stat.S_IMODE(after.st_mode) == _DIRECTORY_MODE,
            "sealed inventory root mode changed",
        )
    if include_root:
        rows.append(_entry_metadata(".", after, entry_type="directory"))
    rows.sort(key=lambda row: (str(row["path"]), str(row["type"])))
    content_rows = [
        {
            key: row[key]
            for key in (
                ("path", "type", "size_bytes", "sha256")
                if row["type"] == "file"
                else ("path", "type")
            )
        }
        for row in rows
    ]
    return {
        "entry_count": len(rows),
        "regular_file_count": sum(row["type"] == "file" for row in rows),
        "directory_count": sum(row["type"] == "directory" for row in rows),
        "regular_file_bytes": sum(
            int(row["size_bytes"]) for row in rows if row["type"] == "file"
        ),
        "content_inventory_sha256": _digest(content_rows),
        "metadata_inventory_sha256": _digest(rows),
        "entries": rows,
    }


def _freeze_directory(
    descriptor: int,
    *,
    prefix: Path,
    excluded_relative: Path | None,
    seen_files: set[tuple[int, int]],
    seen_directories: set[tuple[int, int]],
) -> None:
    for name in sorted(os.listdir(descriptor)):
        relative = prefix / name
        if excluded_relative is not None and relative == excluded_relative:
            continue
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISREG(before.st_mode):
            _require(before.st_nlink == 1, f"hard-linked file is forbidden: {relative}")
            identity = (before.st_dev, before.st_ino)
            _require(identity not in seen_files, f"duplicate file inode: {relative}")
            seen_files.add(identity)
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                _require(
                    stat.S_ISREG(opened.st_mode)
                    and (opened.st_dev, opened.st_ino) == identity
                    and opened.st_nlink == 1,
                    f"file changed while freezing: {relative}",
                )
                if stat.S_IMODE(opened.st_mode) != _FILE_MODE:
                    os.fchmod(child, _FILE_MODE)
                sealed = os.fstat(child)
                _require(
                    stat.S_IMODE(sealed.st_mode) == _FILE_MODE,
                    f"file did not freeze: {relative}",
                )
            finally:
                os.close(child)
        elif stat.S_ISDIR(before.st_mode):
            identity = (before.st_dev, before.st_ino)
            _require(
                identity not in seen_directories, f"duplicate directory: {relative}"
            )
            seen_directories.add(identity)
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                _require(
                    stat.S_ISDIR(opened.st_mode)
                    and (opened.st_dev, opened.st_ino) == identity,
                    f"directory changed while freezing: {relative}",
                )
                _freeze_directory(
                    child,
                    prefix=relative,
                    excluded_relative=excluded_relative,
                    seen_files=seen_files,
                    seen_directories=seen_directories,
                )
                current = os.fstat(child)
                if stat.S_IMODE(current.st_mode) != _DIRECTORY_MODE:
                    os.fchmod(child, _DIRECTORY_MODE)
                _require(
                    stat.S_IMODE(os.fstat(child).st_mode) == _DIRECTORY_MODE,
                    f"directory did not freeze: {relative}",
                )
            finally:
                os.close(child)
        else:
            raise ValueError(
                f"linked or nonregular role entry is forbidden: {relative}"
            )


def _freeze_tree(
    root: Path,
    *,
    freeze_root: bool,
    excluded_relative: Path | None = None,
) -> None:
    # A complete read-only scan occurs before the first chmod.  Consequently a
    # structural rejection never leaves a partly frozen tree.
    _tree_inventory(
        root,
        require_sealed=False,
        excluded_relative=excluded_relative,
        include_root=False,
    )
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _freeze_directory(
            descriptor,
            prefix=Path(),
            excluded_relative=excluded_relative,
            seen_files=set(),
            seen_directories={(opened.st_dev, opened.st_ino)},
        )
        current = os.fstat(descriptor)
        if freeze_root and stat.S_IMODE(current.st_mode) != _DIRECTORY_MODE:
            os.fchmod(descriptor, _DIRECTORY_MODE)
        if freeze_root:
            _require(
                stat.S_IMODE(os.fstat(descriptor).st_mode) == _DIRECTORY_MODE,
                "root did not freeze",
            )
    finally:
        os.close(descriptor)


def _require_no_writable_descriptors_under_root(root: Path, *, phase: str) -> None:
    """Reject inherited writable handles that could mutate a frozen artifact."""

    import fcntl

    canonical_root = _canonical_path(root)
    proc_fds = Path("/proc/self/fd")
    _require(proc_fds.is_dir(), f"{phase} cannot inspect the process descriptor table")
    offenders: list[dict[str, Any]] = []
    for name in os.listdir(proc_fds):
        if not name.isdigit():
            continue
        descriptor = int(name)
        try:
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            target_text = os.readlink(proc_fds / name)
        except OSError:
            # The descriptor used internally by listdir may disappear between
            # enumeration and inspection; persistent descriptors remain.
            continue
        if flags & os.O_ACCMODE not in {os.O_WRONLY, os.O_RDWR}:
            continue
        deleted_suffix = " (deleted)"
        target_path_text = (
            target_text[: -len(deleted_suffix)]
            if target_text.endswith(deleted_suffix)
            else target_text
        )
        if not target_path_text.startswith("/"):
            continue
        target = _canonical_path(target_path_text)
        if target == canonical_root or canonical_root in target.parents:
            offenders.append(
                {
                    "descriptor": descriptor,
                    "access_mode": flags & os.O_ACCMODE,
                    "target": target_text,
                }
            )
    _require(
        not offenders,
        f"{phase} has writable descriptors into the tree being frozen: {offenders!r}",
    )


def _validate_role_top_level(artifacts: _RoleArtifacts) -> None:
    """Reject role-local logs or side artifacts outside the frozen graph."""

    role = str(artifacts.decision["role"])
    expected = {
        artifacts.role_root / ".shard-0.claim",
        artifacts.role_root / ".shard-1.claim",
        artifacts.role_root / ".v8-outcome-phase.claim",
        artifacts.role_root / "cases",
        artifacts.role_root / "logs",
        artifacts.role_root / "private-targets",
        artifacts.role_root / "query-inputs",
        artifacts.role_root / "query-outputs",
        artifacts.role_root / "shard-0.lock-verify.log",
        artifacts.role_root / "shard-1.lock-verify.log",
        artifacts.evidence_path,
        artifacts.decision_path,
        artifacts.execution_completion_path,
    }
    _require(role in _ROLES, "role top-level allowlist received an invalid role")
    observed = set(artifacts.role_root.iterdir())
    _require(
        observed == expected,
        "role top-level allowlist changed: "
        f"missing={sorted(str(path) for path in expected - observed)!r}, "
        f"unexpected={sorted(str(path) for path in observed - expected)!r}",
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = _canonical_path(path)
    parent = destination.parent
    parent_state = os.lstat(parent)
    _require(
        stat.S_ISDIR(parent_state.st_mode)
        and not stat.S_ISLNK(parent_state.st_mode)
        and parent.resolve(strict=True) == parent,
        "completion parent is absent, linked, or non-canonical",
    )
    _require(
        not os.path.lexists(destination), f"completion already exists: {destination}"
    )
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        _FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
        os.fchmod(descriptor, _FILE_MODE)
        _require(os.fstat(descriptor).st_nlink == 1, "new completion is hard-linked")
    except BaseException:
        try:
            os.chmod(destination, 0o600, follow_symlinks=False)
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _git(code: Path, *arguments: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(code), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_GIT_ENVIRONMENT,
        timeout=120,
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _git_tree_records(raw: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, name = encoded.partition(b"\t")
        fields = header.split(b" ")
        _require(bool(separator) and len(fields) == 3, "malformed Git tree record")
        mode, kind, object_id = (field.decode("ascii") for field in fields)
        path = name.decode("utf-8")
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and len(object_id) in {40, 64}
            and all(character in "0123456789abcdef" for character in object_id)
            and path
            and not path.startswith("/")
            and ".." not in Path(path).parts,
            f"unsafe tracked source entry: {path}",
        )
        records.append(
            {"mode": mode, "type": kind, "object_id": object_id, "path": path}
        )
    _require(
        bool(records)
        and [record["path"] for record in records]
        == sorted(record["path"] for record in records),
        "Git source tree is empty or unsorted",
    )
    return records


def _deployed_source_runtime_bindings(
    *,
    deployed_code: Path,
    held_root: Path,
    lock: Mapping[str, Any],
    operator_source: Path,
) -> dict[str, Any]:
    import numpy as np

    _require(socket.gethostname() == _EXPECTED_HOST, "integrity sealer host changed")
    _require(
        _canonical_path(sys.executable) == _PINNED_PYTHON
        and Path(sys.executable).resolve(strict=True) == _PINNED_PYTHON_RESOLVED
        and sys.flags.isolated == 1
        and sys.flags.dont_write_bytecode == 1,
        "integrity sealer did not run under the exact pinned Python with -I -B",
    )
    _require(
        np.__version__ == _PINNED_NUMPY_VERSION
        and _PINNED_PYTHON.parent.parent
        in Path(np.__file__).resolve(strict=True).parents,
        "integrity sealer NumPy runtime changed",
    )
    source = _deployed_source_bindings(
        deployed_code=deployed_code,
        held_root=held_root,
        lock=lock,
        operator_source=operator_source,
    )
    executable = _canonical_path(sys.executable)
    resolved_executable = executable.resolve(strict=True)
    numpy_source = Path(np.__file__).resolve(strict=True)
    return {
        **source,
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": str(executable),
            "python_resolved_executable": _bound_file(
                resolved_executable, label="resolved Python executable"
            ),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "python_cache_tag": sys.implementation.cache_tag,
            "python_isolated": bool(sys.flags.isolated),
            "python_dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
            "numpy_version": np.__version__,
            "numpy_source": _bound_file(numpy_source, label="NumPy package source"),
        },
    }


def _deployed_source_bindings(
    *,
    deployed_code: Path,
    held_root: Path,
    lock: Mapping[str, Any],
    operator_source: Path,
) -> dict[str, Any]:
    code = _canonical_path(deployed_code)
    _require(
        code.is_dir()
        and not code.is_symlink()
        and code.resolve(strict=True) == code
        and code.parent == held_root,
        "deployed source is outside the held root",
    )
    top = _git(code, "rev-parse", "--show-toplevel").decode().strip()
    head = _git(code, "rev-parse", "HEAD").decode().strip().lower()
    _require(
        top == str(code) and code.name == f"code-{head}", "deployed HEAD path changed"
    )
    symbolic = subprocess.run(
        ["/usr/bin/git", "-C", str(code), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_GIT_ENVIRONMENT,
    )
    _require(
        symbolic.returncode == 1 and symbolic.stdout == b"", "HEAD is not detached"
    )
    records = _git_tree_records(_git(code, "ls-tree", "-r", "-z", "HEAD"))
    terminal_frozen = stat.S_IMODE(os.lstat(held_root).st_mode) == _DIRECTORY_MODE
    if not terminal_frozen:
        _require(
            _git(code, "status", "--porcelain=v1", "--untracked-files=all") == b"",
            "deployed source tree is not clean",
        )
    tree_sha256 = _digest(records)
    head_text_sha256 = hashlib.sha256(head.encode("utf-8")).hexdigest()
    immutable = lock.get("immutable_bindings")
    _require(
        isinstance(immutable, Mapping)
        and immutable.get("method_deployed_snapshot_tree") == tree_sha256
        and immutable.get("method_head_text_sha256") == head_text_sha256,
        "deployed source does not match the role lock",
    )
    operator = _canonical_path(operator_source)
    _require(
        operator == code / _OPERATOR_RELATIVE_PATH, "integrity operator escaped source"
    )
    tracked = {record["path"]: record for record in records}
    source_records: dict[str, dict[str, Any]] = {}
    for relative in (*_SOURCE_RELATIVE_PATHS, _OPERATOR_RELATIVE_PATH):
        relative_name = relative.as_posix()
        tracked_record = tracked.get(relative_name)
        _require(
            tracked_record is not None, f"deployed source is untracked: {relative}"
        )
        expected_mode = (
            _FILE_MODE
            if terminal_frozen
            else 0o555
            if tracked_record["mode"] == "100755"
            else 0o444
        )
        source_records[relative_name] = _bound_file(
            code / relative,
            label=f"deployed source {relative}",
            required_mode=expected_mode,
        )
    for binding_name, relative in _NAMED_SOURCE_BINDINGS.items():
        _require(
            immutable.get(binding_name)
            == source_records[relative.as_posix()]["sha256"],
            f"named deployed source binding changed: {binding_name}",
        )
    return {
        "deployed_repository": {
            "path": str(code),
            "head": head,
            "head_text_sha256": head_text_sha256,
            "canonical_git_tree_sha256": tree_sha256,
            "canonical_git_tree_record_count": len(records),
        },
        "source_files": source_records,
        "operator_source": source_records[_OPERATOR_RELATIVE_PATH.as_posix()],
    }


def _validate_source_runtime_bindings(
    binding: object,
    *,
    held_root: Path,
    lock: Mapping[str, Any],
    verify_runtime_modules: bool,
) -> dict[str, Any]:
    _require(isinstance(binding, Mapping), "source/runtime binding is absent")
    repository = binding.get("deployed_repository")
    operator = binding.get("operator_source")
    _require(
        isinstance(repository, Mapping)
        and isinstance(repository.get("path"), str)
        and isinstance(operator, Mapping)
        and isinstance(operator.get("path"), str),
        "source/runtime binding paths are absent",
    )
    source = _deployed_source_bindings(
        deployed_code=Path(str(repository["path"])),
        held_root=held_root,
        lock=lock,
        operator_source=Path(str(operator["path"])),
    )
    _require(
        binding.get("deployed_repository") == source["deployed_repository"]
        and binding.get("source_files") == source["source_files"]
        and binding.get("operator_source") == source["operator_source"],
        "deployed source/operator binding changed",
    )
    runtime = binding.get("runtime")
    _require(isinstance(runtime, Mapping), "runtime binding is absent")
    executable = runtime.get("python_executable")
    resolved = runtime.get("python_resolved_executable")
    numpy_source = runtime.get("numpy_source")
    _require(
        isinstance(executable, str)
        and _canonical_path(executable) == _canonical_path(sys.executable)
        and _canonical_path(executable) == _PINNED_PYTHON
        and Path(executable).resolve(strict=True) == _PINNED_PYTHON_RESOLVED
        and isinstance(resolved, Mapping)
        and isinstance(numpy_source, Mapping)
        and runtime.get("hostname") == socket.gethostname() == _EXPECTED_HOST
        and runtime.get("platform") == platform.platform()
        and runtime.get("python_version") == sys.version
        and runtime.get("python_implementation") == platform.python_implementation()
        and runtime.get("python_cache_tag") == sys.implementation.cache_tag
        and runtime.get("python_isolated") == bool(sys.flags.isolated)
        and sys.flags.isolated == 1
        and runtime.get("python_dont_write_bytecode")
        == bool(sys.flags.dont_write_bytecode)
        and sys.flags.dont_write_bytecode == 1,
        "Python runtime binding changed",
    )
    _require(
        dict(resolved)
        == _bound_file(
            Path(executable).resolve(strict=True),
            label="resolved Python executable",
        )
        and dict(numpy_source)
        == _bound_file(
            str(numpy_source.get("path")), label="sealed NumPy package source"
        )
        and runtime.get("numpy_version") == _PINNED_NUMPY_VERSION
        and _PINNED_PYTHON.parent.parent
        in Path(str(numpy_source.get("path"))).resolve(strict=True).parents,
        "runtime file binding changed",
    )
    if verify_runtime_modules:
        observed = _deployed_source_runtime_bindings(
            deployed_code=Path(str(repository["path"])),
            held_root=held_root,
            lock=lock,
            operator_source=Path(str(operator["path"])),
        )
        _require(
            observed == dict(binding),
            "deployed source/operator/runtime module binding changed",
        )
    return dict(binding)


def _role_completion_artifact(
    artifacts: _RoleArtifacts,
    *,
    recomputed: Mapping[str, Any],
    inventory: Mapping[str, Any],
    source_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    outcome = str(artifacts.decision["decision"])
    terminal_path = canonical_terminal_completion_path(artifacts.held_root)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ROLE_COMPLETION_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": ROLE_COMPLETION_STATUS,
        "role": artifacts.decision["role"],
        "held_root": str(artifacts.held_root),
        "role_root": str(artifacts.role_root),
        "terminal_outcome": outcome,
        "lock": _artifact_record(
            artifacts.lock_path, artifacts.lock, label="held-v8 role lock"
        ),
        "score_evidence": _artifact_record(
            artifacts.evidence_path,
            artifacts.evidence,
            label="held-v8 score evidence",
        ),
        "decision": _artifact_record(
            artifacts.decision_path,
            artifacts.decision,
            label="held-v8 gate decision",
        ),
        "execution_completion": _artifact_record(
            artifacts.execution_completion_path,
            artifacts.execution_completion,
            label="held-v8 role execution completion",
        ),
        "source_manifest": _artifact_record(
            artifacts.source_manifest_path,
            artifacts.source_manifest,
            label="held-v8 role source manifest",
        ),
        "barriers": {
            "first_complete_cohort": _barrier_record(artifacts.barrier_one),
            "second_complete_cohort": _barrier_record(artifacts.barrier_two),
        },
        "recomputed_outcome": dict(recomputed),
        "deployed_source_operator_runtime": dict(source_runtime),
        "sealed_role_inventory": dict(inventory),
        "terminal_root_finalization": {
            "required": outcome in _TERMINAL_OUTCOMES,
            "completion_path": (
                str(terminal_path) if outcome in _TERMINAL_OUTCOMES else None
            ),
        },
        "information_boundary": dict(_ROLE_INFORMATION_BOUNDARY),
        "self_hash_contract": _SELF_HASH_CONTRACT,
    }
    value["artifact_sha256"] = _artifact_sha256(value)
    return value


def _issue_terminal_seal_capability(
    *,
    held_root: Path,
    terminal_role: str,
    terminal_outcome: str,
) -> object:
    """Issue one process-local authority after a role was deeply sealed."""

    _require(terminal_role in _ROLES, "terminal role is invalid")
    _require(terminal_outcome in _TERMINAL_OUTCOMES, "outcome is not terminal")
    capability = _TerminalSealCapability(
        held_root=str(held_root),
        terminal_role=terminal_role,
        terminal_outcome=terminal_outcome,
        _nonce=object(),
        _authority=_TERMINAL_SEAL_CAPABILITY_AUTHORITY,
    )
    _TERMINAL_SEAL_CAPABILITIES[id(capability)] = _TerminalSealCapabilityState(
        capability=capability
    )
    return capability


def _consume_terminal_seal_capability(
    capability: object,
    *,
    held_root: Path,
    terminal_role: str,
) -> str:
    """Consume the role sealer's single-use terminal-finalization authority."""

    _require(
        isinstance(capability, _TerminalSealCapability),
        "terminal finalization lacks a role-sealer capability",
    )
    state = _TERMINAL_SEAL_CAPABILITIES.get(id(capability))
    _require(
        state is not None
        and state.capability is capability
        and capability._authority is _TERMINAL_SEAL_CAPABILITY_AUTHORITY,
        "terminal finalization capability is not live in this process",
    )
    _require(not state.consumed, "terminal finalization capability was already used")
    # Consume before touching any terminal artifact.  A failed finalization is
    # an incident, not an authority that may silently be retried.
    state.consumed = True
    _require(
        capability.held_root == str(held_root)
        and capability.terminal_role == terminal_role,
        "terminal finalization capability binds another role or root",
    )
    return capability.terminal_outcome


def seal_role_outcome(
    *,
    lock_path: str | os.PathLike[str],
    role: Literal["calibration", "confirmation"],
    deployed_code: str | os.PathLike[str],
    operator_source: str | os.PathLike[str],
) -> dict[str, Any]:
    """Recompute, seal, and terminal-finalize one exact role when required."""

    preliminary_root, preliminary_lock = _held_root_from_lock(
        lock_path, expected_role=role
    )
    source_runtime = _deployed_source_runtime_bindings(
        deployed_code=_canonical_path(deployed_code),
        held_root=preliminary_root,
        lock=preliminary_lock,
        operator_source=_canonical_path(operator_source),
    )
    _validate_execution_completion(
        held_root=preliminary_root,
        lock_path=_canonical_path(lock_path),
        expected_role=role,
        verify_live_rlimit=True,
    )
    artifacts = _validate_role_artifacts(lock_path, expected_role=role)
    _require(
        artifacts.held_root == preliminary_root
        and dict(artifacts.lock) == preliminary_lock,
        "role lock changed after runtime verification",
    )
    completion_path = canonical_role_completion_path(artifacts.held_root, role)
    _require(
        completion_path.parent == artifacts.role_root
        and not os.path.lexists(completion_path),
        "canonical role completion is not fresh",
    )
    recomputed = _recompute_scores_from_raw(artifacts)
    excluded = completion_path.relative_to(artifacts.role_root)
    _validate_role_top_level(artifacts)
    _require_no_writable_descriptors_under_root(
        artifacts.role_root, phase="role outcome sealing"
    )
    _freeze_tree(
        artifacts.role_root,
        freeze_root=False,
        excluded_relative=excluded,
    )
    _validate_role_top_level(artifacts)
    inventory = _tree_inventory(
        artifacts.role_root,
        require_sealed=False,
        excluded_relative=excluded,
        include_root=False,
    )
    # Every descendant has already been frozen.  The role root remains 0700
    # only long enough to create the excluded completion as the last role file.
    for row in inventory["entries"]:
        expected_mode = "0400" if row["type"] == "file" else "0500"
        _require(row["mode_octal"] == expected_mode, "role tree did not fully freeze")
    completion = _role_completion_artifact(
        artifacts,
        recomputed=recomputed,
        inventory=inventory,
        source_runtime=source_runtime,
    )
    _write_new_json(completion_path, completion)
    role_descriptor = os.open(
        artifacts.role_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if stat.S_IMODE(os.fstat(role_descriptor).st_mode) != _DIRECTORY_MODE:
            os.fchmod(role_descriptor, _DIRECTORY_MODE)
        os.fsync(role_descriptor)
    finally:
        os.close(role_descriptor)
    sealed = validate_role_outcome_completion(
        completion_path,
        lock_path=artifacts.lock_path,
        expected_role=role,
        verify_content_inventory=True,
        recompute_scores=True,
    )
    if sealed["terminal_outcome"] in _TERMINAL_OUTCOMES:
        terminal_capability = _issue_terminal_seal_capability(
            held_root=artifacts.held_root,
            terminal_role=role,
            terminal_outcome=str(sealed["terminal_outcome"]),
        )
        _seal_terminal_held_root(
            held_root=artifacts.held_root,
            terminal_role=role,
            deployed_code=deployed_code,
            operator_source=operator_source,
            terminal_capability=terminal_capability,
        )
    return sealed


def _validate_completion_identity(
    completion_path: Path,
    completion: Mapping[str, Any],
    *,
    lock_path: Path,
    expected_role: str,
    artifacts: _RoleArtifacts,
) -> None:
    expected_path = canonical_role_completion_path(artifacts.held_root, expected_role)
    _require(completion_path == expected_path, "role completion path is not canonical")
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "status",
        "role",
        "held_root",
        "role_root",
        "terminal_outcome",
        "lock",
        "score_evidence",
        "decision",
        "execution_completion",
        "source_manifest",
        "barriers",
        "recomputed_outcome",
        "deployed_source_operator_runtime",
        "sealed_role_inventory",
        "terminal_root_finalization",
        "information_boundary",
        "self_hash_contract",
        "artifact_sha256",
    }
    _require(set(completion) == expected_keys, "role completion fields changed")
    _require(
        completion.get("schema_version") == SCHEMA_VERSION
        and completion.get("artifact_kind") == ROLE_COMPLETION_KIND
        and completion.get("protocol_id") == PROTOCOL_ID
        and completion.get("status") == ROLE_COMPLETION_STATUS
        and completion.get("role") == expected_role
        and completion.get("held_root") == str(artifacts.held_root)
        and completion.get("role_root") == str(artifacts.role_root)
        and completion.get("terminal_outcome") == artifacts.decision["decision"],
        "role completion identity or outcome changed",
    )
    _require(
        completion.get("artifact_sha256") == _artifact_sha256(completion),
        "role completion self-hash changed",
    )
    _require(
        completion.get("information_boundary") == _ROLE_INFORMATION_BOUNDARY
        and completion.get("self_hash_contract") == _SELF_HASH_CONTRACT,
        "role completion information-boundary contract changed",
    )
    _require(
        completion.get("lock")
        == _artifact_record(lock_path, artifacts.lock, label="held-v8 role lock")
        and completion.get("score_evidence")
        == _artifact_record(
            artifacts.evidence_path,
            artifacts.evidence,
            label="held-v8 score evidence",
        )
        and completion.get("decision")
        == _artifact_record(
            artifacts.decision_path,
            artifacts.decision,
            label="held-v8 gate decision",
        )
        and completion.get("execution_completion")
        == _artifact_record(
            artifacts.execution_completion_path,
            artifacts.execution_completion,
            label="held-v8 role execution completion",
        )
        and completion.get("source_manifest")
        == _artifact_record(
            artifacts.source_manifest_path,
            artifacts.source_manifest,
            label="held-v8 role source manifest",
        ),
        "role completion lock/evidence/decision/execution/source bindings changed",
    )
    barriers = completion.get("barriers")
    _require(
        barriers
        == {
            "first_complete_cohort": _barrier_record(artifacts.barrier_one),
            "second_complete_cohort": _barrier_record(artifacts.barrier_two),
        },
        "role completion reconstructed barriers changed",
    )
    terminal = completion.get("terminal_root_finalization")
    outcome = artifacts.decision["decision"]
    _require(
        terminal
        == {
            "required": outcome in _TERMINAL_OUTCOMES,
            "completion_path": (
                str(canonical_terminal_completion_path(artifacts.held_root))
                if outcome in _TERMINAL_OUTCOMES
                else None
            ),
        },
        "role completion terminal-finalization contract changed",
    )


def _validate_completion_metadata_identity(
    completion_path: Path,
    completion: Mapping[str, Any],
    *,
    expected_role: str,
    metadata: _RoleMetadata,
) -> None:
    expected_path = canonical_role_completion_path(metadata.held_root, expected_role)
    _require(completion_path == expected_path, "role completion path is not canonical")
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "status",
        "role",
        "held_root",
        "role_root",
        "terminal_outcome",
        "lock",
        "score_evidence",
        "decision",
        "execution_completion",
        "source_manifest",
        "barriers",
        "recomputed_outcome",
        "deployed_source_operator_runtime",
        "sealed_role_inventory",
        "terminal_root_finalization",
        "information_boundary",
        "self_hash_contract",
        "artifact_sha256",
    }
    _require(set(completion) == expected_keys, "role completion fields changed")
    outcome = metadata.decision.get("decision")
    _require(
        completion.get("schema_version") == SCHEMA_VERSION
        and completion.get("artifact_kind") == ROLE_COMPLETION_KIND
        and completion.get("protocol_id") == PROTOCOL_ID
        and completion.get("status") == ROLE_COMPLETION_STATUS
        and completion.get("role") == expected_role
        and completion.get("held_root") == str(metadata.held_root)
        and completion.get("role_root") == str(metadata.role_root)
        and completion.get("terminal_outcome") == outcome
        and completion.get("artifact_sha256") == _artifact_sha256(completion),
        "role completion metadata identity or self-hash changed",
    )
    _require(
        completion.get("information_boundary") == _ROLE_INFORMATION_BOUNDARY
        and completion.get("self_hash_contract") == _SELF_HASH_CONTRACT,
        "role completion information-boundary contract changed",
    )
    _require(
        completion.get("lock")
        == _artifact_record(
            metadata.lock_path, metadata.lock, label="held-v8 role lock"
        )
        and completion.get("score_evidence")
        == _artifact_record(
            metadata.evidence_path,
            metadata.evidence,
            label="held-v8 score evidence",
        )
        and completion.get("decision")
        == _artifact_record(
            metadata.decision_path,
            metadata.decision,
            label="held-v8 gate decision",
        )
        and completion.get("execution_completion")
        == _artifact_record(
            metadata.execution_completion_path,
            metadata.execution_completion,
            label="held-v8 role execution completion",
        )
        and completion.get("source_manifest")
        == _artifact_record(
            metadata.source_manifest_path,
            metadata.source_manifest,
            label="held-v8 role source manifest",
        ),
        "role completion lock/evidence/decision/execution/source bindings changed",
    )
    barriers = completion.get("barriers")
    _require(isinstance(barriers, Mapping), "completion barriers are absent")
    first = barriers.get("first_complete_cohort")
    second = barriers.get("second_complete_cohort")
    barrier_fields = {
        "barrier_number",
        "operation",
        "barrier_sha256",
        "ordered_case_names",
        "ordered_artifact_bindings",
        "ordered_artifact_bindings_sha256",
    }
    _require(
        isinstance(first, Mapping)
        and isinstance(second, Mapping)
        and set(first) == barrier_fields
        and set(second) == barrier_fields
        and first.get("barrier_number") == 1
        and first.get("operation") == TARGET_RECONSTRUCTION_OPERATION
        and second.get("barrier_number") == 2
        and second.get("operation") == FUTURE_SCORE_OPERATION
        and _valid_sha256(first.get("barrier_sha256"))
        and _valid_sha256(second.get("barrier_sha256"))
        and first.get("ordered_case_names")
        == metadata.evidence.get("ordered_case_names")
        and second.get("ordered_case_names")
        == metadata.evidence.get("ordered_case_names")
        and first.get("ordered_artifact_bindings_sha256")
        == _digest(first.get("ordered_artifact_bindings"))
        and second.get("ordered_artifact_bindings_sha256")
        == _digest(second.get("ordered_artifact_bindings"))
        and second.get("barrier_sha256") == metadata.evidence.get("barrier_two_sha256"),
        "stored barrier metadata changed",
    )
    records = metadata.evidence.get("case_records")
    _require(isinstance(records, Mapping), "stored case records are absent")
    for case_name in metadata.evidence.get("ordered_case_names", []):
        record = records.get(case_name)
        permit = (
            record.get("future_score_permit_evidence")
            if isinstance(record, Mapping)
            else None
        )
        _require(
            isinstance(permit, Mapping)
            and permit.get("cohort_barrier_sha256") == second["barrier_sha256"]
            and permit.get("predecessor_reconstruction_barrier_sha256")
            == first["barrier_sha256"],
            f"stored case permit barrier lineage changed for {case_name}",
        )
    recomputed = completion.get("recomputed_outcome")
    _require(
        isinstance(recomputed, Mapping)
        and recomputed.get("ordered_case_names")
        == metadata.evidence.get("ordered_case_names")
        and _valid_sha256(recomputed.get("case_record_set_sha256"))
        and _valid_sha256(recomputed.get("gate_result_sha256"))
        and _valid_sha256(recomputed.get("score_and_gate_sha256"))
        and recomputed.get("gate_result") == metadata.evidence.get("gate_result")
        and recomputed.get("sealed_score_evidence_exact_match") is True
        and recomputed.get("sealed_gate_decision_exact_match") is True
        and recomputed.get("canonical_raw_target_query_prediction_reloaded") is True,
        "stored raw-recomputation metadata changed",
    )
    _require(
        completion.get("terminal_root_finalization")
        == {
            "required": outcome in _TERMINAL_OUTCOMES,
            "completion_path": (
                str(canonical_terminal_completion_path(metadata.held_root))
                if outcome in _TERMINAL_OUTCOMES
                else None
            ),
        },
        "role completion terminal-finalization contract changed",
    )


def validate_role_outcome_completion(
    completion_path: str | os.PathLike[str],
    *,
    lock_path: str | os.PathLike[str],
    expected_role: Literal["calibration", "confirmation"],
    verify_content_inventory: bool = True,
    recompute_scores: bool = True,
) -> dict[str, Any]:
    """Validate one canonical role completion; safe to call repeatedly."""

    completion_path_canonical = _canonical_path(completion_path)
    completion = _load_json(
        completion_path_canonical,
        label="role outcome completion",
        required_mode=_FILE_MODE,
    )
    metadata = _load_role_metadata(lock_path, expected_role=expected_role)
    root_state = os.lstat(metadata.role_root)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and stat.S_IMODE(root_state.st_mode) == _DIRECTORY_MODE,
        "completed role root is not mode 0500",
    )
    _validate_completion_metadata_identity(
        completion_path_canonical,
        completion,
        expected_role=expected_role,
        metadata=metadata,
    )
    _validate_source_runtime_bindings(
        completion.get("deployed_source_operator_runtime"),
        held_root=metadata.held_root,
        lock=metadata.lock,
        verify_runtime_modules=recompute_scores,
    )
    if verify_content_inventory:
        observed_inventory = _tree_inventory(
            metadata.role_root,
            require_sealed=True,
            excluded_relative=completion_path_canonical.relative_to(metadata.role_root),
            include_root=False,
        )
        _require(
            completion.get("sealed_role_inventory") == observed_inventory,
            "sealed role content or metadata inventory changed",
        )
    if recompute_scores:
        artifacts = _validate_role_artifacts(lock_path, expected_role=expected_role)
        _validate_completion_identity(
            completion_path_canonical,
            completion,
            lock_path=_canonical_path(lock_path),
            expected_role=expected_role,
            artifacts=artifacts,
        )
        observed_recomputed = _recompute_scores_from_raw(artifacts)
        _require(
            completion.get("recomputed_outcome") == observed_recomputed,
            "completion raw-derived scores or gate changed",
        )
    return completion


def _terminal_role_completion_records(
    held_root: Path,
    *,
    terminal_role: str,
    recompute_scores: bool,
) -> list[dict[str, Any]]:
    roles: Sequence[str] = (
        ("calibration",)
        if terminal_role == "calibration"
        else ("calibration", "confirmation")
    )
    records: list[dict[str, Any]] = []
    for role in roles:
        lock_path = held_root / f"{role}-lock.json"
        completion_path = canonical_role_completion_path(held_root, role)  # type: ignore[arg-type]
        completion = validate_role_outcome_completion(
            completion_path,
            lock_path=lock_path,
            expected_role=role,  # type: ignore[arg-type]
            verify_content_inventory=True,
            recompute_scores=recompute_scores,
        )
        if role == "calibration" and terminal_role == "confirmation":
            _require(
                completion.get("terminal_outcome") == "GO",
                "confirmation terminal root lacks a sealed calibration GO",
            )
        records.append(
            _artifact_record(
                completion_path,
                completion,
                label=f"{role} role outcome completion",
            )
        )
    return records


def _validate_terminal_top_level(
    held_root: Path,
    *,
    terminal_role: str,
    deployed_code: Path,
) -> None:
    """Reject every root entry outside the exact prospective run layout."""

    expected = {
        held_root / "post-withdrawal-development-use-disclosure.json",
        held_root / "calibration-lock.json",
        held_root / "replacement-source",
        held_root / "calibration",
        deployed_code,
    }
    if terminal_role == "confirmation":
        expected.update(
            {
                held_root / "confirmation-lock.json",
                held_root / "confirmation-source",
                held_root / "confirmation",
            }
        )
    observed = set(held_root.iterdir())
    _require(
        observed == expected,
        "terminal held-root top-level allowlist changed: "
        f"missing={sorted(str(path) for path in expected - observed)!r}, "
        f"unexpected={sorted(str(path) for path in observed - expected)!r}",
    )


def _seal_terminal_held_root(
    *,
    held_root: str | os.PathLike[str],
    terminal_role: Literal["calibration", "confirmation"],
    deployed_code: str | os.PathLike[str],
    operator_source: str | os.PathLike[str],
    terminal_capability: object,
) -> dict[str, Any]:
    """Freeze a terminal root using authority issued by the deep role sealer."""

    root = _canonical_path(held_root)
    authorized_outcome = _consume_terminal_seal_capability(
        terminal_capability,
        held_root=root,
        terminal_role=terminal_role,
    )
    completion_path = canonical_terminal_completion_path(root)
    _require(not os.path.lexists(completion_path), "terminal completion already exists")
    from . import deform360_held_v8_protocol as protocol

    terminal_lock = protocol.validate_protocol_lock(root / f"{terminal_role}-lock.json")
    source_runtime = _deployed_source_runtime_bindings(
        deployed_code=_canonical_path(deployed_code),
        held_root=root,
        lock=terminal_lock,
        operator_source=_canonical_path(operator_source),
    )
    _validate_terminal_top_level(
        root,
        terminal_role=terminal_role,
        deployed_code=_canonical_path(deployed_code),
    )
    _validate_execution_completion(
        held_root=root,
        lock_path=root / f"{terminal_role}-lock.json",
        expected_role=terminal_role,
        verify_live_rlimit=True,
    )
    role_records = _terminal_role_completion_records(
        root, terminal_role=terminal_role, recompute_scores=True
    )
    terminal_role_completion = _load_json(
        role_records[-1]["path"],
        label="terminal role completion",
        required_mode=_FILE_MODE,
    )
    outcome = str(terminal_role_completion["terminal_outcome"])
    _require(
        outcome == authorized_outcome and outcome in _TERMINAL_OUTCOMES,
        "held-root outcome differs from the terminal authority",
    )
    _require_no_writable_descriptors_under_root(
        root, phase="terminal held-root sealing"
    )
    _validate_terminal_top_level(
        root,
        terminal_role=terminal_role,
        deployed_code=_canonical_path(deployed_code),
    )
    _freeze_tree(root, freeze_root=True)
    _validate_terminal_top_level(
        root,
        terminal_role=terminal_role,
        deployed_code=_canonical_path(deployed_code),
    )
    inventory = _tree_inventory(root, require_sealed=True, include_root=True)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": TERMINAL_COMPLETION_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": TERMINAL_COMPLETION_STATUS,
        "held_root": str(root),
        "held_root_mode_octal": "0500",
        "terminal_role": terminal_role,
        "terminal_outcome": outcome,
        "role_outcome_completions": role_records,
        "sealed_held_root_inventory": inventory,
        "deployed_source_operator_runtime": source_runtime,
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
        "self_hash_contract": _SELF_HASH_CONTRACT,
    }
    value["artifact_sha256"] = _artifact_sha256(value)
    _write_new_json(completion_path, value)
    return validate_terminal_root_completion(
        completion_path,
        held_root=root,
        verify_content_inventory=True,
        recompute_scores=True,
    )


def validate_terminal_root_completion(
    completion_path: str | os.PathLike[str],
    *,
    held_root: str | os.PathLike[str],
    verify_content_inventory: bool = True,
    recompute_scores: bool = True,
) -> dict[str, Any]:
    """Validate the outside-root completion for a terminal held run."""

    root = _canonical_path(held_root)
    path = _canonical_path(completion_path)
    _require(path == canonical_terminal_completion_path(root), "terminal path changed")
    completion = _load_json(
        path, label="terminal root completion", required_mode=_FILE_MODE
    )
    _require(
        set(completion)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "status",
            "held_root",
            "held_root_mode_octal",
            "terminal_role",
            "terminal_outcome",
            "role_outcome_completions",
            "sealed_held_root_inventory",
            "deployed_source_operator_runtime",
            "information_boundary",
            "self_hash_contract",
            "artifact_sha256",
        },
        "terminal completion fields changed",
    )
    terminal_role = completion.get("terminal_role")
    outcome = completion.get("terminal_outcome")
    _require(
        completion.get("schema_version") == SCHEMA_VERSION
        and completion.get("artifact_kind") == TERMINAL_COMPLETION_KIND
        and completion.get("protocol_id") == PROTOCOL_ID
        and completion.get("status") == TERMINAL_COMPLETION_STATUS
        and completion.get("held_root") == str(root)
        and completion.get("held_root_mode_octal") == "0500"
        and terminal_role in _ROLES
        and outcome in _TERMINAL_OUTCOMES,
        "terminal completion identity changed",
    )
    _require(
        completion.get("artifact_sha256") == _artifact_sha256(completion)
        and completion.get("self_hash_contract") == _SELF_HASH_CONTRACT,
        "terminal completion self-hash changed",
    )
    _require(
        completion.get("information_boundary")
        == {
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
        "terminal information-boundary attestation changed",
    )
    root_state = os.lstat(root)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and stat.S_IMODE(root_state.st_mode) == _DIRECTORY_MODE,
        "terminal held root is not mode 0500",
    )
    role_records = _terminal_role_completion_records(
        root,
        terminal_role=str(terminal_role),
        recompute_scores=recompute_scores,
    )
    _require(
        completion.get("role_outcome_completions") == role_records,
        "terminal role completion bindings changed",
    )
    _require(
        _load_json(
            role_records[-1]["path"],
            label="terminal role completion",
            required_mode=_FILE_MODE,
        ).get("terminal_outcome")
        == outcome,
        "terminal root and role outcomes differ",
    )
    from . import deform360_held_v8_protocol as protocol

    terminal_lock = protocol.validate_protocol_lock(root / f"{terminal_role}-lock.json")
    _validate_source_runtime_bindings(
        completion.get("deployed_source_operator_runtime"),
        held_root=root,
        lock=terminal_lock,
        verify_runtime_modules=recompute_scores,
    )
    source_runtime = completion.get("deployed_source_operator_runtime")
    repository = (
        source_runtime.get("deployed_repository")
        if isinstance(source_runtime, Mapping)
        else None
    )
    deployed_path = repository.get("path") if isinstance(repository, Mapping) else None
    _require(isinstance(deployed_path, str), "terminal deployed path is absent")
    _validate_terminal_top_level(
        root,
        terminal_role=str(terminal_role),
        deployed_code=_canonical_path(deployed_path),
    )
    if verify_content_inventory:
        inventory = _tree_inventory(root, require_sealed=True, include_root=True)
        _require(
            completion.get("sealed_held_root_inventory") == inventory,
            "terminal held-root content or metadata inventory changed",
        )
    return completion


__all__ = [
    "ROLE_COMPLETION_KIND",
    "ROLE_COMPLETION_STATUS",
    "SCHEMA_VERSION",
    "TERMINAL_COMPLETION_KIND",
    "TERMINAL_COMPLETION_STATUS",
    "canonical_role_completion_path",
    "canonical_terminal_completion_path",
    "seal_role_outcome",
    "validate_role_outcome_completion",
    "validate_terminal_root_completion",
]
