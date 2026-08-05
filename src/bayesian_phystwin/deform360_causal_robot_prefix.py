"""Causal robot-prefix artifacts for the official-Hub Deform360 study.

The visual provider and the robot estimator intentionally use different camera
policies.  The visual panel is selected for object-motion coverage; this module
requires the complete, preregistered calibrated panel for metric gripper pose.
Only frames before the causal cutoff may be decoded, and a recovered trajectory
is an anchor candidate only after passing independent marker-support and UMI
kinematic checks.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ._portable_contracts import (
    canonical_json_bytes,
    content_id,
    load_strict_json_object,
    write_atomic_json,
)

DEFORM360_CAUSAL_ROBOT_PREFIX_LOCK_SCHEMA = (
    "bayesian-phystwin.deform360-causal-robot-prefix-lock"
)
DEFORM360_CAUSAL_ROBOT_PREFIX_ARTIFACT_SCHEMA = (
    "bayesian-phystwin.deform360-causal-robot-prefix-artifact"
)
DEFORM360_CAUSAL_ROBOT_PREFIX_VERSION = 1
DEFORM360_PROCESSING_REPOSITORY = "lhy0807/deform360"
DEFORM360_PROCESSING_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"
DEFORM360_VISUOTACTILE_PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"

UMI_MIN_OPENING_M = 0.040
UMI_MAX_OPENING_M = 0.112
PART_NAMES = ("wrist", "left", "right")


class _Capture(Protocol):
    def read(self) -> tuple[bool, Any]: ...


@dataclass(frozen=True, slots=True)
class CausalRobotPrefixQuality:
    """Outcome-blind admission result for one causal robot prefix."""

    admitted: bool
    reason_codes: tuple[str, ...]
    summary: Mapping[str, Any]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and int(value) >= minimum, f"invalid {name}")
    return int(value)


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    _require(type(value) in {int, float}, f"invalid {name}")
    result = float(value)
    _require(np.isfinite(result), f"invalid {name}")
    if minimum is not None:
        _require(result >= minimum, f"invalid {name}")
    return result


def _string(value: object, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"invalid {name}")
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_camera_order(cameras: Sequence[str]) -> tuple[str, ...]:
    """Return the unique sorted camera order used by every estimator pass."""

    result = tuple(sorted(_string(camera, name="camera") for camera in cameras))
    _require(len(result) > 0, "camera panel is empty")
    _require(len(result) == len(set(result)), "camera panel contains duplicates")
    return result


def run_causal_capture_loop(
    captures: Mapping[str, _Capture],
    *,
    source_frame_start: int,
    causal_frame_stop: int,
    process_frame: Callable[[int, str, Any], None],
) -> int:
    """Decode sequentially through the prefix and never read a future frame.

    Frames before ``source_frame_start`` are decoded only to advance compressed
    video streams.  The callback sees exactly ``[source_frame_start,
    causal_frame_stop)`` in canonical camera order.
    """

    start = _integer(source_frame_start, name="source_frame_start")
    stop = _integer(causal_frame_stop, name="causal_frame_stop", minimum=1)
    _require(start < stop, "causal frame range is empty")
    cameras = canonical_camera_order(tuple(captures))
    callback_count = 0
    for frame_id in range(stop):
        for camera in cameras:
            ok, frame = captures[camera].read()
            if not ok:
                raise ValueError(
                    f"camera {camera} ended before causal frame {frame_id}"
                )
            if frame_id >= start:
                process_frame(frame_id, camera, frame)
                callback_count += 1
    return callback_count


def _validated_content_address(
    value: Mapping[str, Any],
    *,
    id_field: str,
    name: str,
) -> str:
    declared = value.get(id_field)
    _require(type(declared) is str, f"{name} lacks {id_field}")
    descriptor = dict(value)
    descriptor.pop(id_field)
    computed = content_id(descriptor)
    _require(declared == computed, f"{name} content identity changed")
    return computed


def validate_deform360_causal_robot_prefix_lock(
    value: Mapping[str, Any],
) -> str:
    """Validate the frozen all-camera causal robot-prefix policy."""

    lock_id = _validated_content_address(
        value,
        id_field="artifact_id",
        name="causal robot-prefix lock",
    )
    _require(
        value.get("schema") == DEFORM360_CAUSAL_ROBOT_PREFIX_LOCK_SCHEMA
        and value.get("schema_version") == DEFORM360_CAUSAL_ROBOT_PREFIX_VERSION,
        "unsupported causal robot-prefix lock",
    )
    _require(
        value.get("status") == "locked-source-only-pre-estimation",
        "causal robot-prefix lock has the wrong information boundary",
    )
    _require(
        value.get("protocol_id") == DEFORM360_VISUOTACTILE_PROTOCOL_ID,
        "causal robot-prefix protocol changed",
    )

    source = value.get("source_case")
    window = value.get("causal_window")
    estimator = value.get("estimator")
    gate = value.get("quality_gate")
    boundary = value.get("information_boundary")
    for item, name in (
        (source, "source_case"),
        (window, "causal_window"),
        (estimator, "estimator"),
        (gate, "quality_gate"),
        (boundary, "information_boundary"),
    ):
        _require(isinstance(item, Mapping), f"{name} is missing")
    assert isinstance(source, Mapping)
    assert isinstance(window, Mapping)
    assert isinstance(estimator, Mapping)
    assert isinstance(gate, Mapping)
    assert isinstance(boundary, Mapping)

    _string(source.get("object_id"), name="object_id")
    _integer(source.get("source_episode_id"), name="source_episode_id")
    _integer(
        source.get("processing_episode_index"),
        name="processing_episode_index",
    )
    _require(type(source.get("bimanual")) is bool, "bimanual must be Boolean")
    cameras = canonical_camera_order(tuple(source.get("cameras", ())))
    _require(tuple(source.get("cameras", ())) == cameras, "cameras are not canonical")
    _require(len(cameras) >= 3, "all-camera panel has fewer than three cameras")

    start = _integer(window.get("source_frame_start"), name="source_frame_start")
    stop = _integer(window.get("causal_frame_stop"), name="causal_frame_stop")
    contact = _integer(window.get("contact_start_frame"), name="contact_start_frame")
    observed = _integer(window.get("observed_frame_count"), name="observed_frame_count")
    _require(stop - start == observed > 0, "causal window length changed")
    _require(start <= contact < stop, "contact is outside the causal prefix")
    _require(
        _integer(window.get("untouched_future_frame_start"), name="future start")
        == stop,
        "untouched future no longer starts at the cutoff",
    )

    _require(
        estimator.get("repository") == DEFORM360_PROCESSING_REPOSITORY
        and estimator.get("revision") == DEFORM360_PROCESSING_REVISION,
        "upstream robot estimator changed",
    )
    _require(estimator.get("camera_policy") == "all-calibrated-cameras", "camera policy changed")
    _require(
        estimator.get("decode_policy")
        == "sequential-read-discard-before-start-stop-before-future",
        "causal decode policy changed",
    )
    _require(_integer(estimator.get("seed"), name="seed") == 0, "seed changed")

    expected_gate = {
        "minimum_inlier_cameras_per_part": 2,
        "minimum_direct_wrist_fraction": 0.75,
        "minimum_both_fingers_fraction": 0.50,
        "contact_tail_frame_count": 6,
        "minimum_contact_ready_frames": 4,
        "minimum_opening_m": UMI_MIN_OPENING_M,
        "maximum_opening_m": UMI_MAX_OPENING_M,
        "maximum_translation_step_m": 0.05,
        "maximum_rotation_step_deg": 20.0,
        "rotation_matrix_tolerance": 1e-3,
    }
    _require(dict(gate) == expected_gate, "quality gate changed")
    expected_boundary = {
        "calibration_camera_prefix_allowed": True,
        "calibration_tactile_prefix_allowed": True,
        "calibration_scores_opened": False,
        "confirmation_payloads_opened": False,
        "future_camera_frames_used": False,
        "future_tactile_frames_used": False,
        "held_v8_accessed": False,
        "target_outcomes_used": False,
    }
    _require(dict(boundary) == expected_boundary, "information boundary changed")
    return lock_id


def load_deform360_causal_robot_prefix_lock(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and validate one causal robot-prefix lock."""

    value = load_strict_json_object(path, label="causal robot-prefix lock")
    validate_deform360_causal_robot_prefix_lock(value)
    return value


def _rotation_step_degrees(rotations: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", rotations[:-1], rotations[1:])
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def evaluate_causal_robot_prefix_quality(
    *,
    transforms: np.ndarray,
    openings_m: np.ndarray,
    part_inlier_camera_counts: np.ndarray,
    source_frame_ids: np.ndarray,
    bimanual: bool,
    quality_gate: Mapping[str, Any],
) -> CausalRobotPrefixQuality:
    """Evaluate support and hardware plausibility without prediction outcomes."""

    frame_ids = np.asarray(source_frame_ids, dtype=np.int64)
    transforms_array = np.asarray(transforms, dtype=np.float64)
    openings_array = np.asarray(openings_m, dtype=np.float64)
    counts = np.asarray(part_inlier_camera_counts, dtype=np.int64)
    if not bimanual:
        if transforms_array.ndim == 3:
            transforms_array = transforms_array[:, None]
        if openings_array.ndim == 1:
            openings_array = openings_array[:, None]
    frame_count = len(frame_ids)
    gripper_count = 2 if bimanual else 1
    _require(
        frame_ids.shape == (frame_count,)
        and transforms_array.shape == (frame_count, gripper_count, 4, 4)
        and openings_array.shape == (frame_count, gripper_count)
        and counts.shape == (frame_count, gripper_count, len(PART_NAMES)),
        "robot-prefix arrays have incompatible shapes",
    )
    _require(frame_count >= 2, "robot prefix is too short")
    _require(np.all(np.diff(frame_ids) == 1), "source frame IDs are not contiguous")

    reasons: list[str] = []
    if not (
        np.all(np.isfinite(transforms_array))
        and np.all(np.isfinite(openings_array))
    ):
        reasons.append("nonfinite-state")
    rotations = transforms_array[..., :3, :3]
    identities = np.eye(3)
    orthogonality_error = np.max(
        np.abs(np.einsum("...ji,...jk->...ik", rotations, rotations) - identities)
    )
    determinants = np.linalg.det(rotations)
    tolerance = _finite_float(
        quality_gate["rotation_matrix_tolerance"],
        name="rotation_matrix_tolerance",
    )
    if (
        not np.isfinite(orthogonality_error)
        or orthogonality_error > tolerance
        or not np.all(np.isfinite(determinants))
        or np.max(np.abs(determinants - 1.0)) > tolerance
    ):
        reasons.append("invalid-se3")

    translation_steps = np.linalg.norm(
        np.diff(transforms_array[..., :3, 3], axis=0), axis=-1
    )
    rotation_steps = _rotation_step_degrees(rotations)
    max_translation = float(np.max(translation_steps))
    max_rotation = float(np.max(rotation_steps))
    if max_translation > float(quality_gate["maximum_translation_step_m"]):
        reasons.append("translation-step-too-large")
    if max_rotation > float(quality_gate["maximum_rotation_step_deg"]):
        reasons.append("rotation-step-too-large")

    minimum_opening = float(quality_gate["minimum_opening_m"])
    maximum_opening = float(quality_gate["maximum_opening_m"])
    opening_in_range = (openings_array >= minimum_opening) & (
        openings_array <= maximum_opening
    )
    if not np.all(opening_in_range):
        reasons.append("opening-outside-released-range")

    support = counts >= int(quality_gate["minimum_inlier_cameras_per_part"])
    direct_wrist = support[..., 0]
    both_fingers = support[..., 1] & support[..., 2]
    wrist_fractions = np.mean(direct_wrist, axis=0)
    finger_fractions = np.mean(both_fingers, axis=0)
    if np.any(wrist_fractions < float(quality_gate["minimum_direct_wrist_fraction"])):
        reasons.append("insufficient-direct-wrist-support")
    if np.any(finger_fractions < float(quality_gate["minimum_both_fingers_fraction"])):
        reasons.append("insufficient-finger-support")

    tail_count = int(quality_gate["contact_tail_frame_count"])
    _require(frame_count >= tail_count, "robot prefix is shorter than contact tail")
    contact_ready = direct_wrist & both_fingers & opening_in_range
    contact_ready_counts = np.sum(contact_ready[-tail_count:], axis=0)
    if np.any(
        contact_ready_counts < int(quality_gate["minimum_contact_ready_frames"])
    ):
        reasons.append("insufficient-contact-tail-support")

    summary = {
        "frame_count": frame_count,
        "gripper_count": gripper_count,
        "source_frame_start": int(frame_ids[0]),
        "causal_frame_stop": int(frame_ids[-1]) + 1,
        "direct_wrist_fraction_by_gripper": wrist_fractions.tolist(),
        "both_fingers_fraction_by_gripper": finger_fractions.tolist(),
        "contact_ready_frames_by_gripper": contact_ready_counts.tolist(),
        "minimum_opening_observed_m": float(np.min(openings_array)),
        "maximum_opening_observed_m": float(np.max(openings_array)),
        "maximum_translation_step_observed_m": max_translation,
        "maximum_rotation_step_observed_deg": max_rotation,
        "maximum_rotation_orthogonality_error": float(orthogonality_error),
        "maximum_rotation_determinant_error": float(
            np.max(np.abs(determinants - 1.0))
        ),
    }
    return CausalRobotPrefixQuality(
        admitted=not reasons,
        reason_codes=tuple(sorted(set(reasons))),
        summary=summary,
    )


def _array_record(values: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(values)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def write_causal_robot_prefix_artifact(
    *,
    output_npz: str | Path,
    output_manifest: str | Path,
    arrays: Mapping[str, np.ndarray],
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    implementation_revision: str,
    source_artifacts: Mapping[str, str],
    quality: CausalRobotPrefixQuality,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Write one content-addressed diagnostic estimate and admission record."""

    validate_deform360_causal_robot_prefix_lock(lock)
    required = {
        "source_frame_ids",
        "actions",
        "T_worlds",
        "openings",
        "part_inlier_camera_counts",
        "marker_inlier_camera_counts",
        "raw_marker_detection_counts",
    }
    _require(set(arrays) == required, "robot-prefix array set changed")
    prepared = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    for name, value in prepared.items():
        _require(value.dtype != object, f"{name} uses object dtype")
    destination = Path(output_npz)
    manifest_path = Path(output_manifest)
    if (destination.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(destination if destination.exists() else manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **prepared)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    source_records = {key: source_artifacts[key] for key in sorted(source_artifacts)}
    descriptor_value: dict[str, Any] = {
        "schema": DEFORM360_CAUSAL_ROBOT_PREFIX_ARTIFACT_SCHEMA,
        "schema_version": DEFORM360_CAUSAL_ROBOT_PREFIX_VERSION,
        "protocol_id": lock["protocol_id"],
        "lock_id": lock["artifact_id"],
        "lock_file_sha256": lock_file_sha256,
        "implementation_revision": implementation_revision,
        "array_archive": {
            "path": destination.name,
            "sha256": _sha256_file(destination),
            "arrays": {name: _array_record(prepared[name]) for name in sorted(prepared)},
        },
        "source_artifacts": source_records,
        "quality": {
            "admitted": quality.admitted,
            "reason_codes": list(quality.reason_codes),
            "summary": dict(quality.summary),
        },
        "anchor_authorized": quality.admitted,
        "fallback": "no-contact-anchor" if not quality.admitted else "not-required",
        "information_boundary": dict(lock["information_boundary"]),
    }
    descriptor_value["artifact_id"] = content_id(descriptor_value)
    write_atomic_json(descriptor_value, manifest_path, overwrite=overwrite)
    return descriptor_value


def verify_causal_robot_prefix_artifact(
    manifest_path: str | Path,
    *,
    verify_arrays: bool = True,
) -> Mapping[str, Any]:
    """Verify content identity, archive bytes, and every packed array digest."""

    path = Path(manifest_path)
    value = load_strict_json_object(path, label="causal robot-prefix artifact")
    _validated_content_address(value, id_field="artifact_id", name="robot-prefix artifact")
    _require(
        value.get("schema") == DEFORM360_CAUSAL_ROBOT_PREFIX_ARTIFACT_SCHEMA
        and value.get("schema_version") == DEFORM360_CAUSAL_ROBOT_PREFIX_VERSION,
        "unsupported robot-prefix artifact",
    )
    archive = value.get("array_archive")
    _require(isinstance(archive, Mapping), "robot-prefix archive record is missing")
    archive_path = path.parent / _string(archive.get("path"), name="archive path")
    _require(_sha256_file(archive_path) == archive.get("sha256"), "archive digest changed")
    if verify_arrays:
        records = archive.get("arrays")
        _require(isinstance(records, Mapping), "array records are missing")
        with np.load(archive_path, allow_pickle=False) as payload:
            _require(set(payload.files) == set(records), "packed array set changed")
            for name, record in records.items():
                _require(isinstance(record, Mapping), f"invalid array record {name}")
                actual = _array_record(np.asarray(payload[name]))
                _require(actual == dict(record), f"packed array changed: {name}")
    return value


def lock_descriptor_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical JSON digest used when transporting a lock."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CausalRobotPrefixQuality",
    "DEFORM360_CAUSAL_ROBOT_PREFIX_ARTIFACT_SCHEMA",
    "DEFORM360_CAUSAL_ROBOT_PREFIX_LOCK_SCHEMA",
    "DEFORM360_CAUSAL_ROBOT_PREFIX_VERSION",
    "DEFORM360_PROCESSING_REVISION",
    "PART_NAMES",
    "UMI_MAX_OPENING_M",
    "UMI_MIN_OPENING_M",
    "canonical_camera_order",
    "evaluate_causal_robot_prefix_quality",
    "load_deform360_causal_robot_prefix_lock",
    "lock_descriptor_sha256",
    "run_causal_capture_loop",
    "validate_deform360_causal_robot_prefix_lock",
    "verify_causal_robot_prefix_artifact",
    "write_causal_robot_prefix_artifact",
]
