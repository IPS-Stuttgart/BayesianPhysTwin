"""Outcome-blind admission and sealing for the dynamic TAPNext++ study.

This module deliberately has no outcome or target arguments. It validates raw
source contracts before cohort selection, seals ordinary predictions before
future identities are read, records technical failures as a distinct
disposition, and builds the completeness barrier that precedes source scoring.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .observation_belief import (
    file_sha256,
    load_observation_belief,
)
from .tapnextpp_dynamic_multiview import PROTOCOL_ID

ADMISSION_FILENAME = "dynamic_tapnextpp_source_admission.json"
PREDICTION_ARCHIVE_FILENAME = "dynamic_tapnextpp_prediction.npz"
PREDICTION_SEAL_FILENAME = "dynamic_tapnextpp_prediction_seal.json"
TECHNICAL_FAILURE_FILENAME = "dynamic_tapnextpp_technical_failure.json"
SOURCE_BARRIER_FILENAME = "dynamic_tapnextpp_source_barrier.json"

ADMISSION_ARTIFACT_KIND = "Deform360DynamicTAPNextPPSourceAdmission"
PREDICTION_ARTIFACT_KIND = "Deform360DynamicTAPNextPPPredictionSeal"
TECHNICAL_FAILURE_ARTIFACT_KIND = (
    "Deform360DynamicTAPNextPPTechnicalFailure"
)
SOURCE_BARRIER_ARTIFACT_KIND = "Deform360DynamicTAPNextPPSourceBarrier"

EXPECTED_FRAME_COUNT = 76
EXPECTED_UPDATE_FRAMES = (19, 38, 57)
MINIMUM_CAMERA_COUNT = 8
MINIMUM_PHYSICAL_NODE_COUNT = 128
_OBJECT_HASH_DOMAIN = b"deform360-fresh-object-exclusion-v1\0"
_CASE_HASH_DOMAIN = b"deform360-dynamic-tapnextpp-case-v1\0"
_ALLOWED_BIMANUAL_VALUES = frozenset({"yes", "no"})
_ALLOWED_FAILURE_STAGES = frozenset(
    {
        "source-admission",
        "physical-backbone",
        "query-schedule",
        "birth-association",
        "tapnextpp-runtime",
        "multiview-lift",
        "observation-belief",
        "state-update",
        "prediction-seal",
    }
)
_PREDICTION_ARRAY_NAMES = frozenset(
    {
        "baseline_prediction_m",
        "candidate_prediction_m",
        "persistence_prediction_m",
        "measurement_entity_ids",
        "hidden_entity_ids",
        "update_frames",
    }
)
_FORBIDDEN_KEY_TOKENS = (
    "outcome",
    "metric",
    "ground_truth",
    "future_observation",
    "target_score",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _validate_digest(value: str, *, name: str) -> str:
    digest = str(value)
    _require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return digest


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    _require(isinstance(payload, dict), "artifact root must be an object")
    return payload


def _reject_forbidden_keys(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            _require(
                not any(token in normalized for token in _FORBIDDEN_KEY_TOKENS),
                f"outcome-bearing key is forbidden at {path}.{key}",
            )
            _reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def deform360_object_hash(object_id: str) -> str:
    """Return the shared hash-only physical-object identity."""

    _require(bool(object_id), "object_id must be nonempty")
    return _sha256_bytes(_OBJECT_HASH_DOMAIN + object_id.encode("utf-8"))


def deform360_case_hash(object_id: str, episode_id: int) -> str:
    """Return a protocol-specific hash-only episode identity."""

    _require(bool(object_id), "object_id must be nonempty")
    _require(int(episode_id) >= 0, "episode_id must be nonnegative")
    identity = f"{object_id}\0{int(episode_id)}".encode()
    return _sha256_bytes(_CASE_HASH_DOMAIN + identity)


def build_source_admission(
    output_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
    category: str,
    bimanual: str,
    episode_frame_count: int,
    robot_frame_count: int,
    physical_node_count: int,
    camera_records: Sequence[Mapping[str, Any]],
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Validate one raw case without reading motion outcomes."""

    _require(bool(category), "category must be nonempty")
    required_sources = {"metadata", "robot", "physical_geometry"}
    _require(
        required_sources <= set(source_sha256),
        "source checksums are incomplete",
    )
    validated_sources = {
        str(name): _validate_digest(str(digest), name=f"source {name}")
        for name, digest in sorted(source_sha256.items())
    }
    reasons: list[str] = []
    if bimanual not in _ALLOWED_BIMANUAL_VALUES:
        reasons.append("invalid-bimanual-enum")
    if int(episode_frame_count) != EXPECTED_FRAME_COUNT:
        reasons.append("episode-frame-count-mismatch")
    if int(robot_frame_count) != EXPECTED_FRAME_COUNT:
        reasons.append("robot-frame-count-mismatch")
    if int(physical_node_count) < MINIMUM_PHYSICAL_NODE_COUNT:
        reasons.append("physical-backend-node-count")

    camera_rows: list[dict[str, Any]] = []
    camera_names: set[str] = set()
    for index, record in enumerate(camera_records):
        _require(isinstance(record, Mapping), "camera record must be an object")
        name = str(record.get("camera_name", ""))
        _require(name and name not in camera_names, "camera names must be unique")
        camera_names.add(name)
        row = {
            "camera_name_hash": _sha256_bytes(
                b"deform360-camera-name-v1\0" + name.encode("utf-8")
            ),
            "rgb_frame_count": int(record.get("rgb_frame_count", -1)),
            "depth_frame_count": int(record.get("depth_frame_count", -1)),
            "mask_frame_count": int(record.get("mask_frame_count", -1)),
            "calibration_valid": bool(record.get("calibration_valid", False)),
            "frame_zero_projected_support_count": int(
                record.get("frame_zero_projected_support_count", -1)
            ),
            "source_index": index,
        }
        row["eligible"] = bool(
            row["rgb_frame_count"] == EXPECTED_FRAME_COUNT
            and row["depth_frame_count"] == EXPECTED_FRAME_COUNT
            and row["mask_frame_count"] == EXPECTED_FRAME_COUNT
            and row["calibration_valid"]
            and row["frame_zero_projected_support_count"] > 0
        )
        camera_rows.append(row)
    eligible_camera_count = sum(bool(row["eligible"]) for row in camera_rows)
    if eligible_camera_count < MINIMUM_CAMERA_COUNT:
        reasons.append("insufficient-eligible-camera-panel")
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ADMISSION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "object_hash": deform360_object_hash(object_id),
        "case_hash": deform360_case_hash(object_id, episode_id),
        "category": category,
        "admitted": not reasons,
        "rejection_reasons": sorted(reasons),
        "contracts": {
            "bimanual_domain": sorted(_ALLOWED_BIMANUAL_VALUES),
            "bimanual_value_valid": bimanual in _ALLOWED_BIMANUAL_VALUES,
            "expected_frame_count": EXPECTED_FRAME_COUNT,
            "episode_frame_count": int(episode_frame_count),
            "robot_frame_count": int(robot_frame_count),
            "minimum_physical_node_count": MINIMUM_PHYSICAL_NODE_COUNT,
            "physical_node_count": int(physical_node_count),
            "minimum_eligible_camera_count": MINIMUM_CAMERA_COUNT,
            "eligible_camera_count": eligible_camera_count,
        },
        "camera_records": camera_rows,
        "source_sha256": validated_sources,
        "information_boundary": {
            "metadata_read": True,
            "frame_zero_geometry_read": True,
            "camera_calibration_read": True,
            "robot_schema_read": True,
            "object_motion_after_frame_zero_read": False,
            "future_identity_read": False,
            "score_read": False,
        },
    }
    _reject_forbidden_keys(artifact)
    artifact["result_sha256"] = _canonical_sha256(
        artifact,
        digest_key="result_sha256",
    )
    _write_json_atomic(Path(output_path), artifact)
    validate_source_admission(artifact)
    return artifact


def validate_source_admission(
    artifact: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate a source-admission artifact and its information boundary."""

    payload = (
        _read_json(artifact)
        if isinstance(artifact, (str, Path))
        else dict(artifact)
    )
    _require(
        payload.get("artifact_kind") == ADMISSION_ARTIFACT_KIND,
        "wrong admission artifact kind",
    )
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol changed")
    _validate_digest(str(payload.get("object_hash", "")), name="object_hash")
    _validate_digest(str(payload.get("case_hash", "")), name="case_hash")
    _validate_digest(
        str(payload.get("result_sha256", "")),
        name="result_sha256",
    )
    _require(
        payload["result_sha256"]
        == _canonical_sha256(payload, digest_key="result_sha256"),
        "admission content checksum changed",
    )
    _require(
        bool(payload.get("admitted"))
        == (len(payload.get("rejection_reasons", [])) == 0),
        "admission decision differs from rejection reasons",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("object_motion_after_frame_zero_read") is False
        and boundary.get("future_identity_read") is False
        and boundary.get("score_read") is False,
        "admission crossed the information boundary",
    )
    _reject_forbidden_keys(payload)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    _require("object_id" not in encoded, "plaintext object identity leaked")
    _require("episode_id" not in encoded, "plaintext episode identity leaked")
    return payload


def _load_prediction_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    _require(
        set(arrays) == _PREDICTION_ARRAY_NAMES,
        "prediction archive fields changed",
    )
    baseline = arrays["baseline_prediction_m"]
    candidate = arrays["candidate_prediction_m"]
    persistence = arrays["persistence_prediction_m"]
    _require(
        baseline.ndim == 3
        and baseline.shape[0] == EXPECTED_FRAME_COUNT
        and baseline.shape[2] == 3,
        "baseline prediction must have shape (76, N, 3)",
    )
    _require(
        candidate.shape == baseline.shape and persistence.shape == baseline.shape,
        "prediction trajectories differ in shape",
    )
    _require(
        np.all(np.isfinite(baseline))
        and np.all(np.isfinite(candidate))
        and np.all(np.isfinite(persistence)),
        "prediction archive contains non-finite trajectories",
    )
    measurements = np.asarray(arrays["measurement_entity_ids"], dtype=np.int64)
    hidden = np.asarray(arrays["hidden_entity_ids"], dtype=np.int64)
    _require(
        measurements.ndim == hidden.ndim == 1
        and len(measurements)
        and len(hidden),
        "measurement and hidden identities must be nonempty vectors",
    )
    _require(
        len(np.unique(measurements)) == len(measurements)
        and len(np.unique(hidden)) == len(hidden),
        "prediction identities repeat",
    )
    _require(
        not np.intersect1d(measurements, hidden).size,
        "measurement identities overlap scored hidden identities",
    )
    _require(
        np.array_equal(
            np.asarray(arrays["update_frames"], dtype=np.int64),
            np.asarray(EXPECTED_UPDATE_FRAMES),
        ),
        "update-frame contract changed",
    )
    return arrays


def build_prediction_seal(
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    admission_path: str | Path,
    query_schedule_path: str | Path,
    observation_belief_path: str | Path,
    prediction_archive_path: str | Path,
    code_revision: str,
    environment_sha256: str,
) -> dict[str, Any]:
    """Seal an ordinary prediction before any future identity is read."""

    output = Path(output_dir)
    admission = validate_source_admission(admission_path)
    _require(admission["admitted"] is True, "source case was not admitted")
    protocol = _read_json(protocol_path)
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol file changed")
    schedule = _read_json(query_schedule_path)
    _require(
        schedule.get("protocol_id") == PROTOCOL_ID,
        "query schedule belongs to another protocol",
    )
    _require(
        schedule.get("case_hash") == admission["case_hash"],
        "query schedule case differs from admission",
    )
    belief = load_observation_belief(observation_belief_path)
    _require(
        belief.case_id == admission["case_hash"],
        "observation belief case differs from admission",
    )
    arrays = _load_prediction_archive(prediction_archive_path)
    measurement_ids = np.asarray(arrays["measurement_entity_ids"], dtype=np.int64)
    _require(
        set(map(int, belief.entity_ids)) <= set(map(int, measurement_ids)),
        "observation belief contains undeclared measurement identities",
    )
    _require(
        bool(code_revision) and len(code_revision) >= 7,
        "code revision is invalid",
    )
    environment_digest = _validate_digest(
        environment_sha256,
        name="environment_sha256",
    )
    output.mkdir(parents=True, exist_ok=True)
    archive_target = output / PREDICTION_ARCHIVE_FILENAME
    archive_bytes = Path(prediction_archive_path).read_bytes()
    temporary = archive_target.with_name(archive_target.name + ".tmp")
    temporary.write_bytes(archive_bytes)
    temporary.replace(archive_target)
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "object_hash": admission["object_hash"],
        "case_hash": admission["case_hash"],
        "code_revision": code_revision,
        "environment_sha256": environment_digest,
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "admission": file_sha256(admission_path),
            "query_schedule": file_sha256(query_schedule_path),
            "observation_belief": file_sha256(observation_belief_path),
        },
        "prediction_archive": {
            "filename": PREDICTION_ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_target),
            "frame_count": int(
                arrays["baseline_prediction_m"].shape[0]
            ),
            "material_identity_count": int(
                arrays["baseline_prediction_m"].shape[1]
            ),
            "measurement_identity_count": int(len(measurement_ids)),
            "hidden_identity_count": int(
                len(arrays["hidden_entity_ids"])
            ),
        },
        "observation_belief": {
            "artifact_id": belief.artifact_id,
            "observation_count": belief.observation_count,
            "causal_frame_stop_exclusive": belief.causal_frame_stop,
        },
        "information_boundary": {
            "prediction_complete": True,
            "future_identity_read": False,
            "future_object_geometry_read": False,
            "score_read": False,
            "measurement_identities_excluded_from_scoring": True,
        },
    }
    _reject_forbidden_keys(seal)
    seal["result_sha256"] = _canonical_sha256(
        seal,
        digest_key="result_sha256",
    )
    _write_json_atomic(output / PREDICTION_SEAL_FILENAME, seal)
    validate_prediction_seal(
        seal,
        protocol_path=protocol_path,
        admission_path=admission_path,
        query_schedule_path=query_schedule_path,
        observation_belief_path=observation_belief_path,
        prediction_dir=output,
    )
    return seal


def validate_prediction_seal(
    artifact: Mapping[str, Any] | str | Path,
    *,
    protocol_path: str | Path,
    admission_path: str | Path,
    query_schedule_path: str | Path,
    observation_belief_path: str | Path,
    prediction_dir: str | Path,
) -> dict[str, Any]:
    """Validate an ordinary prediction seal and every bound file."""

    payload = (
        _read_json(artifact)
        if isinstance(artifact, (str, Path))
        else dict(artifact)
    )
    _require(
        payload.get("artifact_kind") == PREDICTION_ARTIFACT_KIND,
        "wrong prediction artifact kind",
    )
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol changed")
    _require(
        payload.get("result_sha256")
        == _canonical_sha256(payload, digest_key="result_sha256"),
        "prediction seal content checksum changed",
    )
    expected_inputs = {
        "protocol": file_sha256(protocol_path),
        "admission": file_sha256(admission_path),
        "query_schedule": file_sha256(query_schedule_path),
        "observation_belief": file_sha256(observation_belief_path),
    }
    _require(
        payload.get("inputs_sha256") == expected_inputs,
        "prediction seal input checksum changed",
    )
    archive = Path(prediction_dir) / PREDICTION_ARCHIVE_FILENAME
    _require(archive.is_file(), "sealed prediction archive is missing")
    _require(
        payload.get("prediction_archive", {}).get("file_sha256")
        == file_sha256(archive),
        "prediction archive checksum changed",
    )
    _load_prediction_archive(archive)
    admission = validate_source_admission(admission_path)
    _require(
        payload.get("case_hash") == admission["case_hash"]
        and payload.get("object_hash") == admission["object_hash"],
        "prediction seal identity changed",
    )
    belief = load_observation_belief(observation_belief_path)
    _require(
        payload.get("observation_belief", {}).get("artifact_id")
        == belief.artifact_id,
        "observation belief artifact changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("prediction_complete") is True
        and boundary.get("future_identity_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("score_read") is False
        and boundary.get("measurement_identities_excluded_from_scoring")
        is True,
        "prediction seal crossed the information boundary",
    )
    _reject_forbidden_keys(payload)
    return payload


def record_technical_failure(
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    admission_path: str | Path,
    stage: str,
    reason_code: str,
    code_revision: str,
) -> dict[str, Any]:
    """Record a retained technical failure without calling it a prediction."""

    _require(stage in _ALLOWED_FAILURE_STAGES, "failure stage is not registered")
    _require(
        reason_code
        and all(character.isalnum() or character in "-_" for character in reason_code),
        "failure reason code is invalid",
    )
    admission = validate_source_admission(admission_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": TECHNICAL_FAILURE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "object_hash": admission["object_hash"],
        "case_hash": admission["case_hash"],
        "stage": stage,
        "reason_code": reason_code,
        "code_revision": code_revision,
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "admission": file_sha256(admission_path),
        },
        "information_boundary": {
            "ordinary_prediction_created": False,
            "future_identity_read": False,
            "future_object_geometry_read": False,
            "score_read": False,
        },
    }
    _reject_forbidden_keys(payload)
    payload["result_sha256"] = _canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    output = Path(output_dir)
    _write_json_atomic(output / TECHNICAL_FAILURE_FILENAME, payload)
    validate_technical_failure(
        payload,
        protocol_path=protocol_path,
        admission_path=admission_path,
    )
    return payload


def validate_technical_failure(
    artifact: Mapping[str, Any] | str | Path,
    *,
    protocol_path: str | Path,
    admission_path: str | Path,
) -> dict[str, Any]:
    """Validate a technical disposition independently of predictions."""

    payload = (
        _read_json(artifact)
        if isinstance(artifact, (str, Path))
        else dict(artifact)
    )
    _require(
        payload.get("artifact_kind") == TECHNICAL_FAILURE_ARTIFACT_KIND,
        "wrong technical-failure artifact kind",
    )
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol changed")
    _require(
        payload.get("stage") in _ALLOWED_FAILURE_STAGES,
        "failure stage is not registered",
    )
    _require(
        payload.get("result_sha256")
        == _canonical_sha256(payload, digest_key="result_sha256"),
        "technical-failure content checksum changed",
    )
    _require(
        payload.get("inputs_sha256")
        == {
            "protocol": file_sha256(protocol_path),
            "admission": file_sha256(admission_path),
        },
        "technical-failure input checksum changed",
    )
    admission = validate_source_admission(admission_path)
    _require(
        payload.get("case_hash") == admission["case_hash"]
        and payload.get("object_hash") == admission["object_hash"],
        "technical-failure identity changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("ordinary_prediction_created") is False
        and boundary.get("future_identity_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("score_read") is False,
        "technical failure crossed the information boundary",
    )
    _reject_forbidden_keys(payload)
    return payload


def build_source_barrier(
    output_path: str | Path,
    *,
    expected_case_hashes: Sequence[str],
    prediction_seals: Sequence[str | Path],
    technical_failures: Sequence[str | Path],
) -> dict[str, Any]:
    """Bind source completeness before any future identity is scored."""

    expected = tuple(
        sorted(
            _validate_digest(value, name="expected case hash")
            for value in expected_case_hashes
        )
    )
    _require(
        len(expected) == len(set(expected)) and len(expected) > 0,
        "expected source cases must be nonempty and unique",
    )
    prediction_payloads = tuple(_read_json(path) for path in prediction_seals)
    failure_payloads = tuple(_read_json(path) for path in technical_failures)
    for payload in prediction_payloads:
        _require(
            payload.get("artifact_kind") == PREDICTION_ARTIFACT_KIND,
            "barrier received a non-prediction seal",
        )
        _require(
            payload.get("result_sha256")
            == _canonical_sha256(payload, digest_key="result_sha256"),
            "prediction seal checksum changed before barrier",
        )
    for payload in failure_payloads:
        _require(
            payload.get("artifact_kind") == TECHNICAL_FAILURE_ARTIFACT_KIND,
            "barrier received a non-technical disposition",
        )
        _require(
            payload.get("result_sha256")
            == _canonical_sha256(payload, digest_key="result_sha256"),
            "technical disposition checksum changed before barrier",
        )
    prediction_cases = tuple(
        sorted(str(payload["case_hash"]) for payload in prediction_payloads)
    )
    failure_cases = tuple(
        sorted(str(payload["case_hash"]) for payload in failure_payloads)
    )
    _require(
        not (set(prediction_cases) & set(failure_cases)),
        "case has both prediction and technical-failure dispositions",
    )
    _require(
        len(prediction_cases) == len(set(prediction_cases))
        and len(failure_cases) == len(set(failure_cases)),
        "source disposition repeats a case",
    )
    disposition_cases = set(prediction_cases) | set(failure_cases)
    unexpected = sorted(disposition_cases - set(expected))
    missing = sorted(set(expected) - disposition_cases)
    _require(not unexpected, "source barrier contains unexpected cases")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SOURCE_BARRIER_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "expected_case_hashes": list(expected),
        "ordinary_prediction_case_hashes": list(prediction_cases),
        "technical_failure_case_hashes": list(failure_cases),
        "missing_case_hashes": missing,
        "counts": {
            "expected": len(expected),
            "ordinary_predictions": len(prediction_cases),
            "retained_technical_failures": len(failure_cases),
            "missing": len(missing),
        },
        "complete": len(missing) == 0,
        "source_scoring_authorized": len(missing) == 0,
        "input_file_sha256": {
            "prediction_seals": sorted(
                file_sha256(path) for path in prediction_seals
            ),
            "technical_failures": sorted(
                file_sha256(path) for path in technical_failures
            ),
        },
        "information_boundary": {
            "future_identity_read": False,
            "score_read": False,
            "barrier_precedes_source_scoring": True,
        },
    }
    _reject_forbidden_keys(payload)
    payload["result_sha256"] = _canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    _write_json_atomic(Path(output_path), payload)
    return payload


def authorize_source_scoring(barrier_path: str | Path) -> dict[str, Any]:
    """Authorize source scoring only after every case has a disposition."""

    payload = _read_json(barrier_path)
    _require(
        payload.get("artifact_kind") == SOURCE_BARRIER_ARTIFACT_KIND,
        "wrong source barrier artifact kind",
    )
    _require(
        payload.get("result_sha256")
        == _canonical_sha256(payload, digest_key="result_sha256"),
        "source barrier content checksum changed",
    )
    _require(
        payload.get("complete") is True
        and payload.get("source_scoring_authorized") is True
        and payload.get("counts", {}).get("missing") == 0,
        "source prediction barrier is incomplete",
    )
    _reject_forbidden_keys(payload)
    return payload


__all__ = [
    "ADMISSION_FILENAME",
    "MINIMUM_PHYSICAL_NODE_COUNT",
    "PREDICTION_ARCHIVE_FILENAME",
    "PREDICTION_SEAL_FILENAME",
    "SOURCE_BARRIER_FILENAME",
    "TECHNICAL_FAILURE_FILENAME",
    "authorize_source_scoring",
    "build_prediction_seal",
    "build_source_admission",
    "build_source_barrier",
    "deform360_case_hash",
    "deform360_object_hash",
    "record_technical_failure",
    "validate_prediction_seal",
    "validate_source_admission",
    "validate_technical_failure",
]
