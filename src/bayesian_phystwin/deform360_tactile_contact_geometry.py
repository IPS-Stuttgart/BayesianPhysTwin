"""Ambiguity-preserving Deform360 tactile contact geometry.

The released sensor names distinguish two tactile groups (``tactilel`` and
``tactiler``), but the public processing code does not bind those groups to the
marker-ID gripper ordering.  This module therefore carries both assignments as
an explicit mixture.  It creates metric contact locations, not displacement
observations; covariance calibration and object association remain mandatory
before the geometry can become a Bayesian contact anchor.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._portable_contracts import content_id, load_strict_json_object, write_atomic_json

DEFORM360_TACTILE_CONTACT_GEOMETRY_LOCK_SCHEMA = (
    "bayesian-phystwin.deform360-tactile-contact-geometry-lock"
)
DEFORM360_TACTILE_CONTACT_GEOMETRY_ARTIFACT_SCHEMA = (
    "bayesian-phystwin.deform360-tactile-contact-geometry-artifact"
)
DEFORM360_TACTILE_CONTACT_GEOMETRY_VERSION = 1
DEFORM360_PROCESSING_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"

TACTILE_ROWS_USED = 12
TACTILE_COLUMNS = 32
ASSIGNMENT_PROBABILITIES = np.asarray([0.5, 0.5], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TactileContactGeometryQuality:
    admitted: bool
    reason_codes: tuple[str, ...]
    summary: Mapping[str, Any]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any], *, name: str) -> str:
    declared = value.get("artifact_id")
    _require(type(declared) is str, f"{name} lacks artifact_id")
    descriptor = dict(value)
    descriptor.pop("artifact_id")
    actual = content_id(descriptor)
    _require(declared == actual, f"{name} content identity changed")
    return actual


def parse_tactile_sensor_name(sensor_name: str) -> tuple[int, int]:
    """Return released tactile-group and finger-side indices.

    Group 0 is ``tactilel`` and group 1 is ``tactiler``.  This group index is
    deliberately not a gripper index.
    """

    prefix = "brics-odroid_tactile"
    _require(sensor_name.startswith(prefix), f"unsupported tactile sensor {sensor_name}")
    suffix = sensor_name[len(prefix) :]
    for group_name, group_index in (("l", 0), ("r", 1)):
        for side_name, side_index in (("_left", 0), ("_right", 1)):
            if suffix == group_name + side_name:
                return group_index, side_index
    raise ValueError(f"unsupported tactile sensor {sensor_name}")


def validate_deform360_tactile_contact_geometry_lock(
    value: Mapping[str, Any],
) -> str:
    """Validate one source-only geometry lock."""

    lock_id = _content_address(value, name="tactile contact-geometry lock")
    _require(
        value.get("schema") == DEFORM360_TACTILE_CONTACT_GEOMETRY_LOCK_SCHEMA
        and value.get("schema_version") == DEFORM360_TACTILE_CONTACT_GEOMETRY_VERSION,
        "unsupported tactile contact-geometry lock",
    )
    _require(
        value.get("status") == "locked-source-only-pre-geometry",
        "tactile contact-geometry lock has the wrong status",
    )
    source = value.get("source")
    geometry = value.get("geometry")
    gate = value.get("quality_gate")
    boundary = value.get("information_boundary")
    for item, name in (
        (source, "source"),
        (geometry, "geometry"),
        (gate, "quality_gate"),
        (boundary, "information_boundary"),
    ):
        _require(isinstance(item, Mapping), f"missing {name}")
    assert isinstance(source, Mapping)
    assert isinstance(geometry, Mapping)
    assert isinstance(gate, Mapping)
    assert isinstance(boundary, Mapping)
    _require(type(source.get("object_id")) is str, "missing object_id")
    _require(type(source.get("bimanual")) is bool and source["bimanual"], "expected bimanual source")
    _require(
        type(source.get("robot_prefix_anchor_authorized")) is bool
        and source["robot_prefix_anchor_authorized"],
        "parent robot prefix was not admitted",
    )
    for name in (
        "robot_prefix_artifact_id",
        "robot_prefix_manifest_sha256",
        "robot_prefix_archive_sha256",
    ):
        digest = source.get(name)
        _require(
            type(digest) is str
            and len(str(digest)) == 64
            and all(character in "0123456789abcdef" for character in str(digest)),
            f"invalid {name}",
        )
    tactile_files = source.get("tactile_files")
    _require(isinstance(tactile_files, Mapping), "missing tactile file bindings")
    expected_sensors = {
        "brics-odroid_tactilel_left",
        "brics-odroid_tactilel_right",
        "brics-odroid_tactiler_left",
        "brics-odroid_tactiler_right",
    }
    _require(set(tactile_files) == expected_sensors, "tactile sensor set changed")
    for sensor_name, record in tactile_files.items():
        _require(isinstance(record, Mapping), f"invalid tactile record {sensor_name}")
        path = record.get("relative_path")
        digest = record.get("sha256")
        _require(
            type(path) is str
            and path == f"{sensor_name}/synced_tactile.npy",
            f"tactile path changed for {sensor_name}",
        )
        _require(
            type(digest) is str
            and len(str(digest)) == 64
            and all(character in "0123456789abcdef" for character in str(digest)),
            f"invalid tactile digest for {sensor_name}",
        )
    _require(
        geometry.get("processing_revision") == DEFORM360_PROCESSING_REVISION,
        "taxel geometry implementation changed",
    )
    _require(geometry.get("active_threshold") == 0.0, "active threshold changed")
    _require(geometry.get("taxel_rows_used") == TACTILE_ROWS_USED, "taxel rows changed")
    _require(geometry.get("taxel_columns") == TACTILE_COLUMNS, "taxel columns changed")
    _require(
        geometry.get("assignments")
        == [
            {"name": "direct", "tactilel_gripper": 0, "tactiler_gripper": 1},
            {"name": "swapped", "tactilel_gripper": 1, "tactiler_gripper": 0},
        ],
        "assignment mixture changed",
    )
    _require(
        geometry.get("assignment_prior_probability") == [0.5, 0.5],
        "assignment prior changed",
    )
    _require(
        dict(gate)
        == {
            "minimum_active_frames": 3,
            "minimum_active_taxels": 6,
            "minimum_assignment_separation_m": 0.05,
        },
        "contact-geometry quality gate changed",
    )
    _require(
        dict(boundary)
        == {
            "calibration_scores_opened": False,
            "confirmation_payloads_opened": False,
            "future_tactile_values_used": False,
            "held_v8_accessed": False,
            "metric_covariance_calibrated": False,
            "object_association_fitted": False,
            "target_outcomes_used": False,
        },
        "contact-geometry information boundary changed",
    )
    return lock_id


def load_deform360_tactile_contact_geometry_lock(
    path: str | Path,
) -> Mapping[str, Any]:
    value = load_strict_json_object(path, label="tactile contact-geometry lock")
    validate_deform360_tactile_contact_geometry_lock(value)
    return value


def extract_active_tactile_rows(
    tactile_by_sensor: Mapping[str, np.ndarray],
    *,
    frame_start: int,
    frame_stop: int,
    active_threshold: float = 0.0,
) -> dict[str, np.ndarray]:
    """Extract active rows only from ``[frame_start, frame_stop)``."""

    _require(type(frame_start) is int and frame_start >= 0, "invalid frame_start")
    _require(type(frame_stop) is int and frame_stop > frame_start, "invalid frame_stop")
    records: list[tuple[int, int, int, int, int, float]] = []
    sensor_names = tuple(sorted(tactile_by_sensor))
    _require(len(sensor_names) == 4, "expected exactly four tactile sensors")
    for sensor_index, sensor_name in enumerate(sensor_names):
        group_index, side_index = parse_tactile_sensor_name(sensor_name)
        values = np.asarray(tactile_by_sensor[sensor_name])
        _require(
            values.ndim == 3
            and values.shape[0] >= frame_stop
            and values.shape[1] >= TACTILE_ROWS_USED
            and values.shape[2] == TACTILE_COLUMNS,
            f"invalid tactile shape for {sensor_name}",
        )
        causal = values[frame_start:frame_stop, :TACTILE_ROWS_USED]
        _require(np.all(np.isfinite(causal)), f"non-finite tactile values in {sensor_name}")
        for local_frame, row, column in np.argwhere(causal > active_threshold):
            records.append(
                (
                    frame_start + int(local_frame),
                    sensor_index,
                    group_index,
                    side_index,
                    int(row) * TACTILE_COLUMNS + int(column),
                    float(causal[local_frame, row, column]),
                )
            )
    records.sort()
    _require(records, "no active tactile taxels in the causal contact window")
    return {
        "source_frame_ids": np.asarray([item[0] for item in records], dtype=np.int64),
        "sensor_indices": np.asarray([item[1] for item in records], dtype=np.int16),
        "tactile_group_indices": np.asarray([item[2] for item in records], dtype=np.int8),
        "finger_side_indices": np.asarray([item[3] for item in records], dtype=np.int8),
        "taxel_flat_indices": np.asarray([item[4] for item in records], dtype=np.int16),
        "tactile_values": np.asarray([item[5] for item in records], dtype=np.float32),
        "sensor_names": np.asarray(sensor_names, dtype="U40"),
    }


def build_assignment_mixture_geometry(
    active_rows: Mapping[str, np.ndarray],
    *,
    robot_source_frame_ids: np.ndarray,
    robot_transforms: np.ndarray,
    robot_openings_m: np.ndarray,
    taxel_points: Callable[[float, np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """Map every active row under direct and swapped gripper assignments."""

    frame_ids = np.asarray(robot_source_frame_ids, dtype=np.int64)
    transforms = np.asarray(robot_transforms, dtype=np.float64)
    openings = np.asarray(robot_openings_m, dtype=np.float64)
    _require(transforms.shape == (len(frame_ids), 2, 4, 4), "expected bimanual transforms")
    _require(openings.shape == (len(frame_ids), 2), "expected bimanual openings")
    frame_lookup = {int(frame): index for index, frame in enumerate(frame_ids)}
    active_frames = np.asarray(active_rows["source_frame_ids"], dtype=np.int64)
    groups = np.asarray(active_rows["tactile_group_indices"], dtype=np.int8)
    sides = np.asarray(active_rows["finger_side_indices"], dtype=np.int8)
    flat = np.asarray(active_rows["taxel_flat_indices"], dtype=np.int16)
    count = len(active_frames)
    points = np.empty((count, 2, 3), dtype=np.float64)
    grippers = np.empty((count, 2), dtype=np.int8)
    cache: dict[tuple[int, int], np.ndarray] = {}
    for row_index in range(count):
        frame_id = int(active_frames[row_index])
        _require(frame_id in frame_lookup, "active tactile frame is outside robot prefix")
        local = frame_lookup[frame_id]
        grid_index = int(flat[row_index]) * 2 + int(sides[row_index])
        for hypothesis in range(2):
            group = int(groups[row_index])
            gripper = group if hypothesis == 0 else 1 - group
            grippers[row_index, hypothesis] = gripper
            key = (local, gripper)
            if key not in cache:
                cache[key] = np.asarray(
                    taxel_points(
                        float(openings[local, gripper]),
                        transforms[local, gripper],
                    ),
                    dtype=np.float64,
                )
                _require(cache[key].shape == (768, 3), "taxel point grid changed")
            points[row_index, hypothesis] = cache[key][grid_index]
    result = {name: np.asarray(value) for name, value in active_rows.items()}
    result["world_points_hypotheses_m"] = points
    result["gripper_indices_hypotheses"] = grippers
    result["assignment_prior_probability"] = ASSIGNMENT_PROBABILITIES.copy()
    return result


def evaluate_tactile_contact_geometry_quality(
    arrays: Mapping[str, np.ndarray],
    *,
    quality_gate: Mapping[str, Any],
) -> TactileContactGeometryQuality:
    points = np.asarray(arrays["world_points_hypotheses_m"], dtype=np.float64)
    frames = np.asarray(arrays["source_frame_ids"], dtype=np.int64)
    values = np.asarray(arrays["tactile_values"], dtype=np.float64)
    _require(points.shape == (len(frames), 2, 3), "contact point shape changed")
    reasons: list[str] = []
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        reasons.append("nonfinite-contact-geometry")
    active_frame_count = len(np.unique(frames))
    if active_frame_count < int(quality_gate["minimum_active_frames"]):
        reasons.append("insufficient-active-frames")
    if len(frames) < int(quality_gate["minimum_active_taxels"]):
        reasons.append("insufficient-active-taxels")
    separation = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
    median_separation = float(np.median(separation))
    if median_separation < float(quality_gate["minimum_assignment_separation_m"]):
        reasons.append("assignment-hypotheses-not-distinct")
    summary = {
        "active_taxel_count": len(frames),
        "active_frame_count": active_frame_count,
        "first_active_frame": int(np.min(frames)),
        "last_active_frame": int(np.max(frames)),
        "median_assignment_separation_m": median_separation,
        "minimum_assignment_separation_m": float(np.min(separation)),
        "maximum_assignment_separation_m": float(np.max(separation)),
        "tactile_value_minimum": float(np.min(values)),
        "tactile_value_maximum": float(np.max(values)),
    }
    return TactileContactGeometryQuality(
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


def write_tactile_contact_geometry_artifact(
    *,
    arrays: Mapping[str, np.ndarray],
    quality: TactileContactGeometryQuality,
    lock: Mapping[str, Any],
    output_npz: str | Path,
    output_manifest: str | Path,
    implementation_revision: str,
    source_artifacts: Mapping[str, str],
    overwrite: bool = False,
) -> Mapping[str, Any]:
    validate_deform360_tactile_contact_geometry_lock(lock)
    prepared = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    for name, values in prepared.items():
        _require(values.dtype != object, f"{name} uses object dtype")
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
    value: dict[str, Any] = {
        "schema": DEFORM360_TACTILE_CONTACT_GEOMETRY_ARTIFACT_SCHEMA,
        "schema_version": DEFORM360_TACTILE_CONTACT_GEOMETRY_VERSION,
        "lock_id": lock["artifact_id"],
        "implementation_revision": implementation_revision,
        "archive": {
            "path": destination.name,
            "sha256": _sha256_file(destination),
            "arrays": {name: _array_record(prepared[name]) for name in sorted(prepared)},
        },
        "source_artifacts": {name: source_artifacts[name] for name in sorted(source_artifacts)},
        "quality": {
            "admitted": quality.admitted,
            "reason_codes": list(quality.reason_codes),
            "summary": dict(quality.summary),
        },
        "assignment_marginalized": True,
        "metric_covariance_calibrated": False,
        "object_association_fitted": False,
        "contact_anchor_authorized": False,
        "information_boundary": dict(lock["information_boundary"]),
    }
    value["artifact_id"] = content_id(value)
    write_atomic_json(value, manifest_path, overwrite=overwrite)
    return value


def verify_tactile_contact_geometry_artifact(
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    path = Path(manifest_path)
    value = load_strict_json_object(path, label="tactile contact geometry")
    _content_address(value, name="tactile contact geometry")
    _require(
        value.get("schema") == DEFORM360_TACTILE_CONTACT_GEOMETRY_ARTIFACT_SCHEMA,
        "unsupported tactile contact geometry",
    )
    archive = value.get("archive")
    _require(isinstance(archive, Mapping), "missing contact-geometry archive")
    archive_path = path.parent / str(archive["path"])
    _require(_sha256_file(archive_path) == archive.get("sha256"), "archive digest changed")
    records = archive.get("arrays")
    _require(isinstance(records, Mapping), "missing contact-geometry array records")
    with np.load(archive_path, allow_pickle=False) as payload:
        _require(set(payload.files) == set(records), "contact-geometry array set changed")
        for name, record in records.items():
            _require(
                isinstance(record, Mapping)
                and _array_record(np.asarray(payload[name])) == dict(record),
                f"contact-geometry array changed: {name}",
            )
    return value


__all__ = [
    "ASSIGNMENT_PROBABILITIES",
    "TactileContactGeometryQuality",
    "build_assignment_mixture_geometry",
    "evaluate_tactile_contact_geometry_quality",
    "extract_active_tactile_rows",
    "load_deform360_tactile_contact_geometry_lock",
    "parse_tactile_sensor_name",
    "validate_deform360_tactile_contact_geometry_lock",
    "verify_tactile_contact_geometry_artifact",
    "write_tactile_contact_geometry_artifact",
]
