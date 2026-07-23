"""Assemble and seal target-free adaptive-confirmation predictions.

The assembler reads only one metadata-only H2 lock plus caller-designated
physical, camera-measurement, and camera-uncertainty NPZ archives.  It verifies
their complete array schemas and hashes from stable regular-file snapshots,
computes the frozen fixed-four, fixed-eight, and adaptive predictions, and
hands immutable snapshots to the case sealer.  It has no evaluation-data
input and never discovers files by walking a dataset tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    load_confirmation_cohort_lock,
)
from .deform360_adaptive_covariance_confirmation_measurement import (
    ARTIFACT_KIND as MEASUREMENT_MANIFEST_KIND,
    EXTERNAL_PHYSICAL_ARRAY_ROLES,
    MANIFEST_FILENAME as MEASUREMENT_MANIFEST_FILENAME,
    MEASUREMENT_ARRAY_ROLES,
    RETAINED_FAILURE_CAMERA_ACCOUNTING,
    RETAINED_MEASUREMENT_FAILURE_CODES,
    RETAINED_MEASUREMENT_FAILURE_STATUS,
    SCHEMA_VERSION as MEASUREMENT_MANIFEST_SCHEMA_VERSION,
    UNCERTAINTY_ARRAY_ROLES,
)
from .deform360_adaptive_covariance_confirmation_seal import (
    ARRAY_ROLES,
    array_sha256,
    seal_confirmation_case,
)
from .deform360_adaptive_covariance_rbf import (
    ADAPTIVE_COVARIANCE_PROTOCOL_ID,
    FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
    normalized_covariance_dispersion,
    predict_adaptive_covariance_selected_backbone_rbf,
)
from .deform360_held_online_prefix import (
    FRAME_COUNT,
    HELD_RBF_CONFIG,
    UPDATE_FRAMES,
    predict_support_gated_selected_backbone_rbf,
)
from .deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    ALLTRACKER_SOURCE_TREE,
    RawCameraObservationConfig,
)
from .deform360_raw_camera_uncertainty import RawCameraUncertaintyConfig


PHYSICAL_ARRAY_ROLES = EXTERNAL_PHYSICAL_ARRAY_ROLES
CAMERA_BUDGETS = (4, 8)
RETAINED_FAILURE_CODES = (
    "automatic_twin_backend_failure",
    "prediction_runtime_failure",
    "resource_exhaustion",
)
_RETAINED_FAILURE_TRACKER_KEYS = frozenset(
    {
        "name",
        "molmomotion_revision",
        "source_tree",
        "runtime_source_sha256",
        "checkpoint_sha256",
        "device",
        "execution_status",
        "failure_code",
        "inference_executed",
    }
)
_RETAINED_FAILURE_SOURCE_KEYS = frozenset(
    {
        "failure_code",
        "prediction_prefix_manifest",
        "frame_zero_manifest",
        "processed_prefix_episode",
        "dynamic_point_observations_available",
    }
)
_RETAINED_FAILURE_TRACKER_RECORD_KEYS = frozenset(
    {
        "prefix_frame_range_half_open",
        "maximum_video_frame_read",
        "decoded_frame_count",
        "decoded_rgb_prefix_sha256",
        "original_image_shape",
        "camera",
        "query_ids",
        "execution_role",
        "execution_index_within_update",
        "four_view_decision_already_materialized",
        "camera_stream_attempted",
        "tracker_inference_executed",
        "dynamic_observation_available",
        "failure_code",
    }
)
_RETAINED_FAILURE_CENTER_RECORD_KEYS = frozenset(
    {
        "center_id",
        "measurement_available",
        "covariance_valid",
        "decision",
        "failure_code",
    }
)
_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")
_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "case_identity",
        "lock_binding",
        "config",
        "plan",
        "inputs",
        "tracker",
        "updates",
        "outputs",
        "camera_accounting",
        "information_boundary",
        "artifact_sha256",
    }
)
_PHYSICAL_BACKBONE_KEYS = frozenset(
    {
        "external_backbone_seal_file_sha256",
        "external_backbone_seal_result_sha256",
        "external_physical_manifest_file_sha256",
        "external_physical_manifest_result_sha256",
        "physical_archive_file_sha256",
        "physical_archive_array_sha256",
    }
)
_CAMERA_ACCOUNTING = {
    "adaptive_charge_is_causal_offline_policy_demand": True,
    "all_eight_streams_eventually_tracked_for_fixed8_shadow": True,
    "realized_acquisition_or_wall_clock_saving_claimed": False,
    "frame_zero_all_camera_planning_excluded": True,
}
_MEASUREMENT_INFORMATION_BOUNDARY = {
    "target_path_argument_accepted": False,
    "outcome_path_argument_accepted": False,
    "target_metric_or_outcome_score_computed": False,
    "future_geometry_read": False,
    "video_prefix_rule": "update u reads exactly frames [0,u]",
    "maximum_video_frame_read_by_update": list(UPDATE_FRAMES),
    "four_view_decision_precedes_shadow_extra_four": True,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_full_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and _FULL_SHA1.fullmatch(value) is not None
        and value != "0" * 40
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and _FULL_SHA256.fullmatch(value) is not None
        and value != "0" * 64
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _json_normalized(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _manifest_artifact_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _measurement_array_sha256(value: np.ndarray) -> str:
    """Match the nested-measurement builder's array checksum convention."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _external_array_sha256(value: np.ndarray) -> str:
    """Match the frozen external physical executor's checksum convention."""

    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


@dataclass(frozen=True)
class _FileSnapshot:
    """One stable regular-file identity retained for a final recheck."""

    path: Path
    file_sha256: str
    size_bytes: int
    file_identity: tuple[int, int]


@dataclass(frozen=True)
class _ArchiveSnapshot:
    """One stable, path-safe archive snapshot and its cloned arrays."""

    path: Path
    file_sha256: str
    size_bytes: int
    file_identity: tuple[int, int]
    arrays: Mapping[str, np.ndarray]
    content_record: Mapping[str, Any]


@dataclass(frozen=True)
class _ManifestSnapshot:
    """One strict-JSON nested-measurement manifest snapshot."""

    path: Path
    file_sha256: str
    size_bytes: int
    file_identity: tuple[int, int]
    value: Mapping[str, Any]


def _stable_regular_file(path: str | Path) -> tuple[Path, bytes, os.stat_result]:
    source = Path(path).absolute()
    before = os.lstat(source)
    _require(not stat.S_ISLNK(before.st_mode), f"archive is a symlink: {source}")
    _require(stat.S_ISREG(before.st_mode), f"archive is not regular: {source}")
    _require(
        source.resolve(strict=True) == source,
        f"archive has a symlinked or noncanonical ancestor: {source}",
    )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"archive changed while opening: {source}",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        _require(
            identity
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
            f"archive changed while reading: {source}",
        )
        _require(len(payload) == opened.st_size, f"archive read was short: {source}")
        return source, payload, after
    finally:
        os.close(descriptor)


def _snapshot_regular_file(path: str | Path) -> _FileSnapshot:
    source, payload, observed = _stable_regular_file(path)
    return _FileSnapshot(
        path=source,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        file_identity=(observed.st_dev, observed.st_ino),
    )


def _load_measurement_manifest(path: str | Path) -> _ManifestSnapshot:
    source, payload, observed = _stable_regular_file(path)
    _require(
        source.name == MEASUREMENT_MANIFEST_FILENAME,
        "nested measurement manifest filename changed",
    )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("nested measurement manifest is invalid JSON") from error
    _require(
        isinstance(value, dict),
        "nested measurement manifest must be a JSON object",
    )
    return _ManifestSnapshot(
        path=source,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        file_identity=(observed.st_dev, observed.st_ino),
        value=value,
    )


def _load_npz_snapshot(
    path: str | Path,
    *,
    expected_roles: Sequence[str],
    label: str,
) -> _ArchiveSnapshot:
    source, payload, observed = _stable_regular_file(path)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(expected_roles),
                f"{label} array roles changed",
            )
            arrays = {role: np.asarray(stored[role]).copy() for role in expected_roles}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{label} is not a valid non-pickle NPZ archive") from error
    for array in arrays.values():
        array.setflags(write=False)
    records = {
        role: {
            "dtype": arrays[role].dtype.str,
            "shape": list(arrays[role].shape),
            "array_sha256": array_sha256(arrays[role]),
        }
        for role in expected_roles
    }
    return _ArchiveSnapshot(
        path=source,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        file_identity=(observed.st_dev, observed.st_ino),
        arrays=arrays,
        content_record={
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "arrays": records,
        },
    )


def _recheck_snapshots(
    snapshots: Sequence[_FileSnapshot | _ArchiveSnapshot | _ManifestSnapshot],
) -> None:
    for snapshot in snapshots:
        source, payload, observed = _stable_regular_file(snapshot.path)
        _require(source == snapshot.path, "archive canonical path changed")
        _require(
            (observed.st_dev, observed.st_ino) == snapshot.file_identity
            and len(payload) == snapshot.size_bytes
            and hashlib.sha256(payload).hexdigest() == snapshot.file_sha256,
            f"archive changed after target-free loading: {snapshot.path}",
        )


def _validate_exact_locked_case(
    lock_path: str | Path,
    case_id: str,
    *,
    expected_h1: str | None,
) -> Mapping[str, Any]:
    lock = load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=expected_h1,
    )
    _require(
        isinstance(case_id, str) and case_id in lock["selected_case_ids"],
        "case is outside the exact H2-locked cohort",
    )
    return lock


def _locked_case_identity(
    lock: Mapping[str, Any],
    case_id: str,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for stratum, records in lock["cohort"].items():
        for record in records:
            for episode in record["episodes"]:
                if episode["case_id"] == case_id:
                    matches.append(
                        {
                            "case_id": case_id,
                            "stratum": stratum,
                            "object_id": record["object_id"],
                            "episode_id": episode["episode_id"],
                        }
                    )
    _require(len(matches) == 1, "case identity is not unique in the H2 lock")
    return matches[0]


def _validate_archive_root_separation(
    lock_path: str | Path,
    output_dir: str | Path,
    snapshots: Sequence[_FileSnapshot | _ArchiveSnapshot | _ManifestSnapshot],
) -> None:
    lock = Path(lock_path).absolute().resolve(strict=True)
    output = Path(output_dir).absolute()
    _require(
        not output.exists() and not output.is_symlink(),
        f"case output already exists: {output}",
    )
    _require(not _paths_overlap(output, lock), "case output overlaps the H2 lock")
    paths = tuple(snapshot.path for snapshot in snapshots)
    _require(len(set(paths)) == len(paths), "input archive paths are duplicated")
    identities = tuple(snapshot.file_identity for snapshot in snapshots)
    _require(
        len(set(identities)) == len(identities),
        "input archives alias the same file identity",
    )
    for index, left in enumerate(paths):
        _require(
            not _paths_overlap(output, left),
            "case output overlaps an input archive",
        )
        for right in paths[index + 1 :]:
            _require(not _paths_overlap(left, right), "input archive paths overlap")


def _validate_manifest_envelope(
    snapshot: _ManifestSnapshot,
    *,
    lock: Mapping[str, Any],
    lock_snapshot: _FileSnapshot,
    h2_commit: str,
    case_id: str,
) -> Mapping[str, Any]:
    manifest = snapshot.value
    _require(
        set(manifest) == _MANIFEST_TOP_LEVEL_KEYS,
        "nested measurement manifest fields changed",
    )
    _require(
        manifest.get("schema_version") == MEASUREMENT_MANIFEST_SCHEMA_VERSION
        and manifest.get("artifact_kind") == MEASUREMENT_MANIFEST_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "nested measurement manifest identity changed",
    )
    _require(
        manifest.get("artifact_sha256") == _manifest_artifact_sha256(manifest),
        "nested measurement manifest checksum changed",
    )
    _require(
        manifest.get("case_identity") == _locked_case_identity(lock, case_id),
        "nested measurement manifest binds another case",
    )
    h1 = lock["two_commit_freeze"]["implementation_commit_h1"]
    _require(h2_commit != h1, "H2 commit must differ from H1")
    _require(
        manifest.get("lock_binding")
        == {
            "implementation_commit_h1": h1,
            "cohort_lock_commit_h2": h2_commit,
            "cohort_lock_artifact_sha256": lock["artifact_sha256"],
            "cohort_lock_file_sha256": lock_snapshot.file_sha256,
        },
        "nested measurement manifest lock binding changed",
    )
    expected_config = _json_normalized(
        {
            "observation": asdict(RawCameraObservationConfig(selected_camera_count=8)),
            "uncertainty": asdict(RawCameraUncertaintyConfig()),
            "adaptive_routing": asdict(FROZEN_ADAPTIVE_COVARIANCE_CONFIG),
        }
    )
    _require(
        manifest.get("config") == expected_config,
        "nested measurement manifest configuration changed",
    )
    tracker = manifest.get("tracker")
    retained_failure = (
        isinstance(tracker, Mapping)
        and tracker.get("execution_status") == RETAINED_MEASUREMENT_FAILURE_STATUS
    )
    expected_tracker_keys = (
        _RETAINED_FAILURE_TRACKER_KEYS
        if retained_failure
        else frozenset(
            {
                "name",
                "molmomotion_revision",
                "source_tree",
                "runtime_source_sha256",
                "checkpoint_sha256",
                "device",
            }
        )
    )
    _require(
        isinstance(tracker, Mapping)
        and set(tracker) == expected_tracker_keys
        and tracker.get("name") == "AllTracker"
        and tracker.get("molmomotion_revision") == ALLTRACKER_MOLMOMOTION_REVISION
        and tracker.get("source_tree") == ALLTRACKER_SOURCE_TREE
        and tracker.get("runtime_source_sha256") == ALLTRACKER_RUNTIME_SOURCE_SHA256
        and tracker.get("checkpoint_sha256") == ALLTRACKER_CHECKPOINT_SHA256
        and isinstance(tracker.get("device"), str)
        and bool(tracker["device"]),
        "nested measurement tracker provenance changed",
    )
    if retained_failure:
        _require(
            set(RETAINED_MEASUREMENT_FAILURE_CODES) == set(RETAINED_FAILURE_CODES)
            and tracker.get("failure_code") in RETAINED_FAILURE_CODES
            and tracker.get("inference_executed") is False
            and tracker.get("device") == "not-executed",
            "retained measurement failure tracker disposition changed",
        )
    _require(
        manifest.get("camera_accounting")
        == (
            RETAINED_FAILURE_CAMERA_ACCOUNTING
            if retained_failure
            else _CAMERA_ACCOUNTING
        ),
        "nested measurement camera accounting changed",
    )
    _require(
        manifest.get("information_boundary") == _MEASUREMENT_INFORMATION_BOUNDARY,
        "nested measurement manifest crossed the target boundary",
    )
    inputs = manifest.get("inputs")
    expected_input_keys = {
        "physical_backbone",
        "physical_archive",
        "intrinsics_sha256",
        "extrinsics_sha256",
        "selected_camera_prefixes_and_frame_zero",
        "source_stage_lineage",
    }
    if retained_failure:
        expected_input_keys.add("retained_failure_source")
    _require(
        isinstance(inputs, Mapping)
        and set(inputs) == expected_input_keys
        and _is_sha256(inputs.get("intrinsics_sha256"))
        and _is_sha256(inputs.get("extrinsics_sha256")),
        "nested measurement input provenance changed",
    )
    source_lineage = inputs.get("source_stage_lineage")
    _require(
        isinstance(source_lineage, Mapping)
        and set(source_lineage)
        == {
            "prediction_prefix_manifest",
            "frame_zero_manifest",
            "source_preparation_manifest_file_sha256",
            "source_custody_seal",
        }
        and _is_sha256(source_lineage.get("source_preparation_manifest_file_sha256")),
        "nested measurement source-stage lineage changed",
    )
    for role in ("prediction_prefix_manifest", "frame_zero_manifest"):
        record = source_lineage.get(role)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "file_sha256", "result_sha256"}
            and isinstance(record.get("path"), str)
            and bool(record["path"])
            and _is_sha256(record.get("file_sha256"))
            and _is_sha256(record.get("result_sha256")),
            f"nested measurement {role} lineage changed",
        )
    custody = source_lineage.get("source_custody_seal")
    _require(
        isinstance(custody, Mapping)
        and set(custody) == {"path", "file_sha256", "artifact_sha256"}
        and isinstance(custody.get("path"), str)
        and Path(custody["path"]).is_absolute()
        and _is_sha256(custody.get("file_sha256"))
        and _is_sha256(custody.get("artifact_sha256")),
        "nested measurement source-custody lineage changed",
    )
    physical_backbone = inputs.get("physical_backbone")
    _require(
        isinstance(physical_backbone, Mapping)
        and set(physical_backbone) == _PHYSICAL_BACKBONE_KEYS,
        "nested measurement physical backbone binding changed",
    )
    for key in _PHYSICAL_BACKBONE_KEYS - {"physical_archive_array_sha256"}:
        _require(
            _is_sha256(physical_backbone.get(key)),
            f"nested measurement physical backbone {key} is invalid",
        )
    array_hashes = physical_backbone.get("physical_archive_array_sha256")
    _require(
        isinstance(array_hashes, Mapping)
        and set(array_hashes) == set(PHYSICAL_ARRAY_ROLES)
        and all(_is_sha256(value) for value in array_hashes.values()),
        "nested measurement physical array binding changed",
    )
    physical_archive = inputs.get("physical_archive")
    _require(
        isinstance(physical_archive, Mapping)
        and set(physical_archive) == {"sha256", "frame_zero_array_sha256"}
        and _is_sha256(physical_archive.get("sha256"))
        and _is_sha256(physical_archive.get("frame_zero_array_sha256")),
        "nested measurement physical archive binding changed",
    )
    if retained_failure:
        source = inputs.get("retained_failure_source")
        _require(
            isinstance(source, Mapping)
            and set(source) == _RETAINED_FAILURE_SOURCE_KEYS
            and source.get("failure_code") == tracker["failure_code"]
            and source.get("dynamic_point_observations_available") is False,
            "retained measurement failure source binding changed",
        )
        for role in ("prediction_prefix_manifest", "frame_zero_manifest"):
            record = source.get(role)
            _require(
                isinstance(record, Mapping)
                and set(record) == {"path", "file_sha256", "result_sha256"}
                and isinstance(record.get("path"), str)
                and bool(record["path"])
                and _is_sha256(record.get("file_sha256"))
                and _is_sha256(record.get("result_sha256")),
                f"retained measurement {role} binding changed",
            )
        processed = source.get("processed_prefix_episode")
        _require(
            isinstance(processed, Mapping)
            and set(processed)
            == {
                "path",
                "intrinsics_file_sha256",
                "extrinsics_file_sha256",
            }
            and isinstance(processed.get("path"), str)
            and bool(processed["path"])
            and processed.get("intrinsics_file_sha256") == inputs["intrinsics_sha256"]
            and processed.get("extrinsics_file_sha256") == inputs["extrinsics_sha256"],
            "retained measurement processed-prefix binding changed",
        )
        _require(
            source["prediction_prefix_manifest"]
            == source_lineage["prediction_prefix_manifest"]
            and source["frame_zero_manifest"] == source_lineage["frame_zero_manifest"],
            "retained source and common source-stage lineage differ",
        )
    return manifest


def _validate_output_archive_record(
    record: object,
    *,
    manifest: _ManifestSnapshot,
    snapshot: _ArchiveSnapshot,
    budget: int,
    archive_role: str,
) -> None:
    filename = (
        "measurement.npz"
        if archive_role == "measurement_archive"
        else "measurement_uncertainty.npz"
    )
    expected_relative = f"budget-{budget}/{filename}"
    expected = {
        "relative_path": expected_relative,
        "sha256": snapshot.file_sha256,
        "size_bytes": snapshot.size_bytes,
        "arrays": dict(snapshot.content_record["arrays"]),
    }
    _require(
        isinstance(record, Mapping) and dict(record) == expected,
        f"{budget}-view {archive_role} manifest binding changed",
    )
    expected_path = (manifest.path.parent / expected_relative).absolute()
    _require(
        snapshot.path == expected_path,
        f"{budget}-view {archive_role} is outside its manifest package",
    )


def _validate_camera_prefix_manifest(
    value: object,
    *,
    cameras: Sequence[str],
    updates: Sequence[Mapping[str, Any]],
) -> None:
    _require(
        isinstance(value, Mapping) and set(value) == set(cameras),
        "nested measurement camera-prefix inputs changed",
    )
    expected_frames = {str(frame) for frame in UPDATE_FRAMES}
    expected_prefixes: dict[str, dict[str, str]] = {camera: {} for camera in cameras}
    for update in updates:
        frame = str(int(update["frame"]))
        for tracker in update["tracker"]:
            camera = str(tracker["camera"])
            expected_prefixes[camera][frame] = str(tracker["decoded_rgb_prefix_sha256"])
    for camera in cameras:
        record = value[camera]
        _require(
            isinstance(record, Mapping)
            and set(record) == {"video", "frame_zero_mask", "frame_zero_depth"},
            "nested measurement camera-prefix record changed",
        )
        video = record["video"]
        _require(
            isinstance(video, Mapping)
            and set(video)
            == {
                "path",
                "decoded_prefix_sha256_by_update",
                "whole_file_hashed_or_read",
            }
            and isinstance(video.get("path"), str)
            and bool(video["path"])
            and video.get("whole_file_hashed_or_read") is False
            and isinstance(video.get("decoded_prefix_sha256_by_update"), Mapping)
            and set(video["decoded_prefix_sha256_by_update"]) == expected_frames
            and dict(video["decoded_prefix_sha256_by_update"])
            == expected_prefixes[camera],
            "nested measurement video-prefix binding changed",
        )
        for role in ("frame_zero_mask", "frame_zero_depth"):
            frame_zero = record[role]
            _require(
                isinstance(frame_zero, Mapping)
                and set(frame_zero)
                == {
                    "path",
                    "frame_zero_array_sha256",
                    "only_index_read",
                    "whole_file_hashed_or_read",
                }
                and isinstance(frame_zero.get("path"), str)
                and bool(frame_zero["path"])
                and _is_sha256(frame_zero.get("frame_zero_array_sha256"))
                and frame_zero.get("only_index_read") == 0
                and frame_zero.get("whole_file_hashed_or_read") is False,
                "nested measurement frame-zero camera binding changed",
            )


def _validate_measurement_manifest_bindings(
    manifest_snapshot: _ManifestSnapshot,
    *,
    physical_snapshot: _ArchiveSnapshot,
    measurement_snapshots: Mapping[int, _ArchiveSnapshot],
    uncertainty_snapshots: Mapping[int, _ArchiveSnapshot],
    physical: np.ndarray,
    frame_zero: np.ndarray,
    measurements: Mapping[int, Mapping[str, Any]],
    uncertainties: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    manifest = manifest_snapshot.value
    tracker_envelope = manifest["tracker"]
    retained_failure = (
        tracker_envelope.get("execution_status") == RETAINED_MEASUREMENT_FAILURE_STATUS
    )
    failure_code = str(tracker_envelope["failure_code"]) if retained_failure else None
    outputs = manifest.get("outputs")
    _require(
        isinstance(outputs, Mapping) and set(outputs) == {"4", "8"},
        "nested measurement output budgets changed",
    )
    for budget in CAMERA_BUDGETS:
        budget_record = outputs[str(budget)]
        _require(
            isinstance(budget_record, Mapping)
            and set(budget_record) == {"measurement_archive", "uncertainty_archive"},
            f"{budget}-view nested measurement output record changed",
        )
        _validate_output_archive_record(
            budget_record["measurement_archive"],
            manifest=manifest_snapshot,
            snapshot=measurement_snapshots[budget],
            budget=budget,
            archive_role="measurement_archive",
        )
        _validate_output_archive_record(
            budget_record["uncertainty_archive"],
            manifest=manifest_snapshot,
            snapshot=uncertainty_snapshots[budget],
            budget=budget,
            archive_role="uncertainty_archive",
        )

    inputs = manifest["inputs"]
    physical_record = inputs["physical_archive"]
    _require(
        physical_record["sha256"] == physical_snapshot.file_sha256
        and physical_record["frame_zero_array_sha256"]
        == _measurement_array_sha256(frame_zero),
        "nested measurement manifest binds another physical archive",
    )
    physical_backbone = inputs["physical_backbone"]
    observed_external_hashes = {
        role: _external_array_sha256(physical_snapshot.arrays[role])
        for role in PHYSICAL_ARRAY_ROLES
    }
    _require(
        physical_backbone["physical_archive_file_sha256"]
        == physical_snapshot.file_sha256
        and physical_backbone["physical_archive_array_sha256"]
        == observed_external_hashes,
        "nested measurement physical backbone archive changed",
    )

    plan = manifest.get("plan")
    _require(
        isinstance(plan, Mapping)
        and set(plan)
        == {
            "candidate_ids",
            "center_ids",
            "camera_activation_order",
            "selected_cameras_by_budget",
            "selection_score",
        },
        "nested measurement plan fields changed",
    )
    candidate_ids = plan["candidate_ids"]
    center_ids = plan["center_ids"]
    _require(
        isinstance(candidate_ids, list)
        and candidate_ids
        and all(type(value) is int for value in candidate_ids)
        and len(candidate_ids) == len(set(candidate_ids))
        and min(candidate_ids) >= 0
        and max(candidate_ids) < physical.shape[1],
        "nested measurement candidate IDs changed",
    )
    expected_centers = measurements[4]["center_ids"].tolist()
    _require(
        center_ids == expected_centers and set(center_ids) <= set(candidate_ids),
        "nested measurement plan center IDs changed",
    )
    cameras = tuple(measurements[8]["selected_cameras"])
    selected_by_budget = plan["selected_cameras_by_budget"]
    _require(
        plan["camera_activation_order"] == list(cameras)
        and isinstance(selected_by_budget, Mapping)
        and set(selected_by_budget) == {"4", "8"}
        and selected_by_budget["4"] == list(measurements[4]["selected_cameras"])
        and selected_by_budget["8"] == list(cameras),
        "nested measurement manifest camera plan changed",
    )
    selection_score = plan["selection_score"]
    _require(
        isinstance(selection_score, Mapping)
        and set(selection_score) == {"4", "8"}
        and all(
            isinstance(selection_score[str(budget)], list)
            and len(selection_score[str(budget)]) == 4
            and all(
                type(value) in (int, float) and np.isfinite(value)
                for value in selection_score[str(budget)]
            )
            for budget in CAMERA_BUDGETS
        ),
        "nested measurement selection score changed",
    )

    update_records = manifest.get("updates")
    _require(
        isinstance(update_records, list) and len(update_records) == len(UPDATE_FRAMES),
        "nested measurement update records changed",
    )
    center_set = set(int(value) for value in center_ids)
    for update_index, (frame, record) in enumerate(zip(UPDATE_FRAMES, update_records)):
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "frame",
                "four_view_decision_materialized_before_shadow_extra_four",
                "four_view_reliable_before_shadow",
                "offline_shadow_extra_four_tracked",
                "adaptive_route",
                "adaptive_charged_camera_streams",
                "budget_reliability",
                "tracker",
                "centers",
            }
            and record.get("frame") == frame
            and record.get("four_view_decision_materialized_before_shadow_extra_four")
            is True
            and record.get("offline_shadow_extra_four_tracked")
            is (not retained_failure),
            "nested measurement update ordering changed",
        )
        expected_reliability: dict[str, Any] = {}
        for budget in CAMERA_BUDGETS:
            result = normalized_covariance_dispersion(
                uncertainties[budget]["measurement_covariance_m2"],
                uncertainties[budget]["measurement_covariance_valid"],
                measurements[budget]["center_ids"],
                frame,
                frame_zero,
                quantile=FROZEN_ADAPTIVE_COVARIANCE_CONFIG.covariance_quantile,
            )
            normalized = result["normalized_covariance_dispersion"]
            reliable = (
                result["valid_covariance_center_count"]
                >= FROZEN_ADAPTIVE_COVARIANCE_CONFIG.minimum_valid_covariance_centers
                and normalized is not None
                and normalized
                <= FROZEN_ADAPTIVE_COVARIANCE_CONFIG.maximum_normalized_covariance_dispersion
            )
            expected_reliability[str(budget)] = _json_normalized(
                {**result, "reliable": bool(reliable)}
            )
        _require(
            record.get("budget_reliability") == expected_reliability,
            "nested measurement reliability disagrees with bound covariance",
        )
        four_reliable = bool(expected_reliability["4"]["reliable"])
        eight_reliable = bool(expected_reliability["8"]["reliable"])
        expected_route = (
            "4_view_rbf"
            if four_reliable
            else ("8_view_rbf" if eight_reliable else "physical_prior_fallback")
        )
        _require(
            record.get("four_view_reliable_before_shadow") is four_reliable
            and record.get("adaptive_route") == expected_route
            and record.get("adaptive_charged_camera_streams")
            == (4 if four_reliable else 8),
            "nested measurement route disagrees with bound covariance",
        )
        if retained_failure:
            _require(
                not four_reliable
                and not eight_reliable
                and expected_route == "physical_prior_fallback"
                and record.get("adaptive_charged_camera_streams") == 8,
                "retained failure produced a dynamic measurement route",
            )
        trackers = record.get("tracker")
        _require(
            isinstance(trackers, list) and len(trackers) == 8,
            "nested measurement tracker update changed",
        )
        for camera_index, tracker in enumerate(trackers):
            _require(
                isinstance(tracker, Mapping),
                "nested measurement tracker record changed",
            )
            maximum_frame = tracker.get(
                "maximum_video_frame_read",
                tracker.get("maximum_source_video_frame_read"),
            )
            expected_role = (
                "adaptive_first_four"
                if camera_index < 4
                else (
                    "fixed_eight_shadow_after_four_decision"
                    if four_reliable
                    else "adaptive_eight_escalation"
                )
            )
            query_ids = tracker.get("query_ids")
            _require(
                tracker.get("camera") == cameras[camera_index]
                and tracker.get("execution_role") == expected_role
                and tracker.get("execution_index_within_update") == camera_index
                and tracker.get("four_view_decision_already_materialized")
                is (camera_index >= 4)
                and maximum_frame == frame
                and _is_sha256(tracker.get("decoded_rgb_prefix_sha256"))
                and isinstance(query_ids, list)
                and all(
                    type(point_id) is int and point_id in center_set
                    for point_id in query_ids
                ),
                "nested measurement tracker ordering or boundary changed",
            )
            if retained_failure:
                _require(
                    set(tracker) == _RETAINED_FAILURE_TRACKER_RECORD_KEYS
                    and tracker.get("prefix_frame_range_half_open") == [0, frame + 1]
                    and tracker.get("decoded_frame_count") == frame + 1
                    and isinstance(tracker.get("original_image_shape"), list)
                    and len(tracker["original_image_shape"]) == 2
                    and all(
                        type(size) is int and size > 0
                        for size in tracker["original_image_shape"]
                    )
                    and tracker.get("camera_stream_attempted") is True
                    and tracker.get("tracker_inference_executed") is False
                    and tracker.get("dynamic_observation_available") is False
                    and tracker.get("failure_code") == failure_code,
                    "retained failure tracker record claims a dynamic observation",
                )
        center_records = record.get("centers")
        _require(
            isinstance(center_records, Mapping)
            and set(center_records) == {"4", "8"}
            and all(
                isinstance(center_records[str(budget)], list)
                and len(center_records[str(budget)]) == len(center_ids)
                and [
                    center.get("center_id")
                    for center in center_records[str(budget)]
                    if isinstance(center, Mapping)
                ]
                == center_ids
                for budget in CAMERA_BUDGETS
            ),
            "nested measurement center diagnostics changed",
        )
        if retained_failure:
            for budget in CAMERA_BUDGETS:
                _require(
                    not np.any(
                        measurements[budget]["measurement_validity"][frame, center_ids]
                    )
                    and not np.any(
                        uncertainties[budget]["measurement_covariance_valid"][
                            frame, center_ids
                        ]
                    )
                    and all(
                        set(center) == _RETAINED_FAILURE_CENTER_RECORD_KEYS
                        and center.get("measurement_available") is False
                        and center.get("covariance_valid") is False
                        and center.get("decision")
                        == "retained_technical_failure_measurement_unavailable"
                        and center.get("failure_code") == failure_code
                        for center in record["centers"][str(budget)]
                    ),
                    "retained failure contains an available dynamic observation",
                )
        _require(
            update_index == list(UPDATE_FRAMES).index(frame),
            "nested measurement update index changed",
        )

    _validate_camera_prefix_manifest(
        inputs["selected_camera_prefixes_and_frame_zero"],
        cameras=cameras,
        updates=update_records,
    )


def _validate_physical(
    snapshot: _ArchiveSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = np.asarray(snapshot.arrays["prediction_m"])
    persistence = np.asarray(snapshot.arrays["persistence_m"])
    frame_zero = np.asarray(snapshot.arrays["frame_zero_points_m"])
    driven = np.asarray(snapshot.arrays["driven_readout_m"])
    zero_action = np.asarray(snapshot.arrays["zero_action_readout_m"])
    action_support = np.asarray(snapshot.arrays["action_support"])
    _require(
        prior.ndim == 3
        and prior.shape[0] == FRAME_COUNT
        and prior.shape[2] == 3
        and prior.shape == persistence.shape == driven.shape == zero_action.shape,
        "physical trajectories must have shape (76, N, 3)",
    )
    _require(
        frame_zero.shape == prior.shape[1:],
        "frame-zero physical geometry shape changed",
    )
    _require(
        np.issubdtype(prior.dtype, np.floating)
        and persistence.dtype == prior.dtype
        and frame_zero.dtype == prior.dtype,
        "physical archive dtypes changed",
    )
    _require(
        driven.dtype == prior.dtype
        and zero_action.dtype == prior.dtype
        and action_support.shape == (prior.shape[1],)
        and np.issubdtype(action_support.dtype, np.number),
        "physical support/readout dtypes changed",
    )
    _require(
        all(
            np.all(np.isfinite(snapshot.arrays[role])) for role in PHYSICAL_ARRAY_ROLES
        ),
        "physical archive contains non-finite values",
    )
    _require(
        np.all((action_support >= 0.0) & (action_support <= 1.0)),
        "physical action support is outside [0, 1]",
    )
    frame_zero_hash = array_sha256(frame_zero)
    _require(
        array_sha256(prior[0]) == frame_zero_hash
        and array_sha256(driven[0]) == frame_zero_hash
        and array_sha256(zero_action[0]) == frame_zero_hash
        and np.array_equal(
            persistence,
            np.repeat(frame_zero[None], FRAME_COUNT, axis=0),
        ),
        "physical archive frame-zero material identity changed",
    )
    return prior, persistence, frame_zero


def _validate_measurement(
    snapshot: _ArchiveSnapshot,
    *,
    budget: int,
    trajectory_shape: tuple[int, ...],
) -> dict[str, Any]:
    measurement = np.asarray(snapshot.arrays["measurement_m"])
    validity = np.asarray(snapshot.arrays["measurement_validity"])
    centers = np.asarray(snapshot.arrays["center_ids"])
    cameras_raw = np.asarray(snapshot.arrays["selected_cameras"])
    updates = np.asarray(snapshot.arrays["update_frames"])
    _require(measurement.shape == trajectory_shape, "measurement shape changed")
    _require(
        np.issubdtype(measurement.dtype, np.floating),
        "measurement dtype must be floating",
    )
    _require(
        validity.shape == trajectory_shape[:2] and validity.dtype == np.dtype(bool),
        "measurement validity shape or dtype changed",
    )
    _require(
        centers.ndim == 1
        and np.issubdtype(centers.dtype, np.integer)
        and len(centers) > 0
        and len(np.unique(centers)) == len(centers)
        and np.all((0 <= centers) & (centers < trajectory_shape[1])),
        "measurement center IDs are invalid",
    )
    _require(
        cameras_raw.ndim == 1
        and cameras_raw.dtype.kind in {"U", "S"}
        and len(cameras_raw) == budget,
        f"{budget}-view selected-camera array changed",
    )
    cameras = tuple(str(camera) for camera in cameras_raw.tolist())
    _require(
        len(set(cameras)) == budget and all(cameras),
        f"{budget}-view selected cameras are invalid",
    )
    _require(
        updates.ndim == 1
        and np.issubdtype(updates.dtype, np.integer)
        and tuple(int(value) for value in updates) == tuple(UPDATE_FRAMES),
        "measurement update frames changed",
    )
    _require(
        np.all(np.isfinite(measurement[validity])),
        "valid measurement contains non-finite coordinates",
    )
    _require(
        np.all(np.isnan(measurement[~validity])),
        "measurement-invalid entries must be canonical NaN placeholders",
    )
    return {
        "measurement_m": measurement,
        "measurement_validity": validity,
        "center_ids": centers.astype(np.int64, copy=False),
        "selected_cameras": cameras,
    }


def _validate_uncertainty(
    snapshot: _ArchiveSnapshot,
    *,
    measurement: Mapping[str, Any],
    trajectory_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    covariance = np.asarray(snapshot.arrays["measurement_covariance_m2"])
    validity = np.asarray(snapshot.arrays["measurement_covariance_valid"])
    _require(
        covariance.shape == (*trajectory_shape[:2], 3, 3),
        "measurement covariance shape changed",
    )
    _require(
        np.issubdtype(covariance.dtype, np.floating),
        "measurement covariance dtype must be floating",
    )
    _require(
        validity.shape == trajectory_shape[:2] and validity.dtype == np.dtype(bool),
        "covariance validity shape or dtype changed",
    )
    measurement_validity = np.asarray(measurement["measurement_validity"])
    _require(
        not np.any(validity & ~measurement_validity),
        "covariance claims a measurement-invalid entry",
    )
    selected = covariance[validity]
    _require(np.all(np.isfinite(selected)), "valid covariance is non-finite")
    _require(
        np.all(np.isnan(covariance[~validity])),
        "covariance-invalid entries must be canonical NaN placeholders",
    )
    if len(selected):
        _require(
            np.allclose(
                selected,
                np.swapaxes(selected, 1, 2),
                rtol=0.0,
                atol=1e-12,
            ),
            "valid covariance is not symmetric",
        )
        eigenvalues = np.linalg.eigvalsh(selected)
        _require(
            np.all(eigenvalues >= -1e-12),
            "valid covariance is not positive semidefinite",
        )
    return {
        "measurement_covariance_m2": covariance,
        "measurement_covariance_valid": validity,
    }


def _load_and_validate_inputs(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    output_dir: str | Path,
    physical_archive: str | Path,
    measurement_manifest: str | Path,
    measurement_archives: Mapping[int, str | Path],
    uncertainty_archives: Mapping[int, str | Path],
    *,
    expected_h1: str | None,
) -> dict[str, Any]:
    lock = _validate_exact_locked_case(
        lock_path,
        case_id,
        expected_h1=expected_h1,
    )
    lock_snapshot = _snapshot_regular_file(lock_path)
    manifest_snapshot = _load_measurement_manifest(measurement_manifest)
    _validate_manifest_envelope(
        manifest_snapshot,
        lock=lock,
        lock_snapshot=lock_snapshot,
        h2_commit=h2_commit,
        case_id=case_id,
    )
    _require(
        isinstance(measurement_archives, Mapping)
        and set(measurement_archives) == set(CAMERA_BUDGETS),
        "measurement archive budgets must be exactly integer keys 4 and 8",
    )
    _require(
        isinstance(uncertainty_archives, Mapping)
        and set(uncertainty_archives) == set(CAMERA_BUDGETS),
        "uncertainty archive budgets must be exactly integer keys 4 and 8",
    )
    physical_snapshot = _load_npz_snapshot(
        physical_archive,
        expected_roles=PHYSICAL_ARRAY_ROLES,
        label="physical archive",
    )
    measurement_snapshots = {
        budget: _load_npz_snapshot(
            measurement_archives[budget],
            expected_roles=MEASUREMENT_ARRAY_ROLES,
            label=f"{budget}-view measurement archive",
        )
        for budget in CAMERA_BUDGETS
    }
    uncertainty_snapshots = {
        budget: _load_npz_snapshot(
            uncertainty_archives[budget],
            expected_roles=UNCERTAINTY_ARRAY_ROLES,
            label=f"{budget}-view uncertainty archive",
        )
        for budget in CAMERA_BUDGETS
    }
    snapshots = (
        lock_snapshot,
        manifest_snapshot,
        physical_snapshot,
        *(measurement_snapshots[budget] for budget in CAMERA_BUDGETS),
        *(uncertainty_snapshots[budget] for budget in CAMERA_BUDGETS),
    )
    _validate_archive_root_separation(
        lock_path,
        output_dir,
        snapshots,
    )
    physical, persistence, frame_zero = _validate_physical(physical_snapshot)
    measurements = {
        budget: _validate_measurement(
            measurement_snapshots[budget],
            budget=budget,
            trajectory_shape=physical.shape,
        )
        for budget in CAMERA_BUDGETS
    }
    _require(
        np.array_equal(
            measurements[4]["center_ids"],
            measurements[8]["center_ids"],
        ),
        "center IDs changed across camera budgets",
    )
    _require(
        measurements[8]["selected_cameras"][:4] == measurements[4]["selected_cameras"],
        "four-view cameras are not the strict ordered prefix of eight views",
    )
    uncertainties = {
        budget: _validate_uncertainty(
            uncertainty_snapshots[budget],
            measurement=measurements[budget],
            trajectory_shape=physical.shape,
        )
        for budget in CAMERA_BUDGETS
    }
    _validate_measurement_manifest_bindings(
        manifest_snapshot,
        physical_snapshot=physical_snapshot,
        measurement_snapshots=measurement_snapshots,
        uncertainty_snapshots=uncertainty_snapshots,
        physical=physical,
        frame_zero=frame_zero,
        measurements=measurements,
        uncertainties=uncertainties,
    )
    input_hashes = {
        "nested_measurement_manifest": {
            "file_sha256": manifest_snapshot.file_sha256,
            "artifact_sha256": manifest_snapshot.value["artifact_sha256"],
        },
        "physical_backbone": dict(
            manifest_snapshot.value["inputs"]["physical_backbone"]
        ),
        "physical_archive": dict(physical_snapshot.content_record),
        "measurement_archives": {
            str(budget): dict(measurement_snapshots[budget].content_record)
            for budget in CAMERA_BUDGETS
        },
        "uncertainty_archives": {
            str(budget): dict(uncertainty_snapshots[budget].content_record)
            for budget in CAMERA_BUDGETS
        },
    }
    return {
        "snapshots": snapshots,
        "manifest": manifest_snapshot.value,
        "physical_snapshot": physical_snapshot,
        "physical": physical,
        "persistence": persistence,
        "frame_zero": frame_zero,
        "measurements": measurements,
        "uncertainties": uncertainties,
        "input_hashes": input_hashes,
    }


def _selected_cameras(inputs: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
    return {
        budget: tuple(inputs["measurements"][budget]["selected_cameras"])
        for budget in CAMERA_BUDGETS
    }


def _adaptive_arguments(
    inputs: Mapping[str, Any],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    measurement = {
        budget: inputs["measurements"][budget]["measurement_m"]
        for budget in CAMERA_BUDGETS
    }
    measurement_validity = {
        budget: inputs["measurements"][budget]["measurement_validity"]
        for budget in CAMERA_BUDGETS
    }
    covariance = {
        budget: inputs["uncertainties"][budget]["measurement_covariance_m2"]
        for budget in CAMERA_BUDGETS
    }
    covariance_validity = {
        budget: inputs["uncertainties"][budget]["measurement_covariance_valid"]
        for budget in CAMERA_BUDGETS
    }
    return measurement, measurement_validity, covariance, covariance_validity


def _full_technical_disposition(
    *,
    status: str,
    center_ids: np.ndarray,
    input_hashes: Mapping[str, Any],
    fixed_diagnostics: Mapping[str, Any],
    failure_code: str | None = None,
    fallback_label: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "status": status,
        "case_retained": True,
        "disposition_based_on_target_or_outcome": False,
        "center_ids": np.asarray(center_ids, dtype=np.int64).tolist(),
        "causal_input_hashes": dict(input_hashes),
        "fixed_budget_predictor_diagnostics": dict(fixed_diagnostics),
    }
    if failure_code is not None:
        value["failure_code"] = failure_code
    if fallback_label is not None:
        value["fallback_label"] = fallback_label
    return value


def assemble_and_seal_confirmation_prediction(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    output_dir: str | Path,
    physical_archive: str | Path,
    measurement_archives: Mapping[int, str | Path],
    uncertainty_archives: Mapping[int, str | Path],
    *,
    measurement_manifest: str | Path,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Compute and atomically seal all six frozen target-free trajectory roles."""

    _require(_is_full_sha1(h2_commit), "H2 commit identity is invalid")
    inputs = _load_and_validate_inputs(
        lock_path,
        h2_commit,
        case_id,
        output_dir,
        physical_archive,
        measurement_manifest,
        measurement_archives,
        uncertainty_archives,
        expected_h1=expected_h1,
    )
    centers = np.asarray(inputs["measurements"][4]["center_ids"], dtype=np.int64)
    fixed_predictions: dict[int, np.ndarray] = {}
    fixed_diagnostics: dict[str, Any] = {}
    for budget in CAMERA_BUDGETS:
        prediction, _, diagnostic = predict_support_gated_selected_backbone_rbf(
            inputs["physical"],
            inputs["persistence"],
            inputs["measurements"][budget]["measurement_m"],
            inputs["measurements"][budget]["measurement_validity"],
            center_ids=centers,
            rbf_config=HELD_RBF_CONFIG,
        )
        fixed_predictions[budget] = prediction
        fixed_diagnostics[str(budget)] = diagnostic
    measurement, validity, covariance, covariance_validity = _adaptive_arguments(inputs)
    adaptive, selected_raw, routing = predict_adaptive_covariance_selected_backbone_rbf(
        inputs["physical"],
        inputs["persistence"],
        _selected_cameras(inputs),
        measurement,
        validity,
        covariance,
        covariance_validity,
        center_ids=centers,
        config=FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
        rbf_config=HELD_RBF_CONFIG,
    )
    _recheck_snapshots(inputs["snapshots"])
    arrays = {
        "physical_prior_m": inputs["physical"],
        "persistence_m": inputs["persistence"],
        "adaptive_prediction_m": adaptive,
        "fixed_4_rbf_prediction_m": fixed_predictions[4],
        "fixed_8_rbf_prediction_m": fixed_predictions[8],
        "selected_raw_prediction_m": selected_raw,
    }
    _require(set(arrays) == set(ARRAY_ROLES), "sealed prediction role set changed")
    return seal_confirmation_case(
        lock_path,
        h2_commit,
        case_id,
        output_dir,
        arrays,
        _selected_cameras(inputs),
        routing,
        _full_technical_disposition(
            status="prediction_complete",
            center_ids=centers,
            input_hashes=inputs["input_hashes"],
            fixed_diagnostics=fixed_diagnostics,
        ),
        expected_h1=expected_h1,
    )


def _retained_failure_routing(
    selected_cameras: Mapping[int, Sequence[str]],
) -> dict[str, Any]:
    cameras = list(selected_cameras[8])
    updates = []
    for update_index, frame in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else FRAME_COUNT
        )
        budget_diagnostics = {
            str(budget): {
                "frame": int(frame),
                "valid_covariance_center_count": 0,
                "valid_covariance_center_ids": [],
                "covariance_quantile": (
                    FROZEN_ADAPTIVE_COVARIANCE_CONFIG.covariance_quantile
                ),
                "radial_standard_deviation_quantile_m": None,
                "frame_zero_bbox_diagonal_m": None,
                "normalized_covariance_dispersion": None,
                "probabilistic_calibration_claimed": False,
                "reliable": False,
            }
            for budget in CAMERA_BUDGETS
        }
        updates.append(
            {
                "frame": int(frame),
                "stop_frame_exclusive": int(stop),
                "route": "physical_prior_fallback",
                "selected_camera_budget": None,
                "tracked_camera_count": 8,
                "tracked_cameras": cameras,
                "selected_backbone": "persistence",
                "rbf_correction_applied": False,
                "state_updated": False,
                "camera_streams_charged_as_attempted": True,
                "dynamic_observation_available": False,
                "tracker_inference_executed": False,
                "budget_diagnostics": budget_diagnostics,
            }
        )
    return {
        "protocol_id": ADAPTIVE_COVARIANCE_PROTOCOL_ID,
        "config": asdict(FROZEN_ADAPTIVE_COVARIANCE_CONFIG),
        "rbf_config": asdict(HELD_RBF_CONFIG),
        "fallback": {
            "trajectory": "persistence",
            "rbf_state_update": False,
            "bit_exact": True,
        },
        "camera_budget_semantics": (
            "both nested budgets charged before retained technical failure"
        ),
        "calibration_boundary": (
            "technical failure disposition is target-free and unscored"
        ),
        "updates": updates,
    }


def seal_retained_confirmation_failure(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    output_dir: str | Path,
    physical_archive: str | Path,
    measurement_archives: Mapping[int, str | Path],
    uncertainty_archives: Mapping[int, str | Path],
    failure_code: str,
    *,
    measurement_manifest: str | Path,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Retain one declared technical failure with six persistence arms."""

    _require(
        failure_code in RETAINED_FAILURE_CODES,
        "retained failure code is outside the frozen target-free vocabulary",
    )
    _require(_is_full_sha1(h2_commit), "H2 commit identity is invalid")
    inputs = _load_and_validate_inputs(
        lock_path,
        h2_commit,
        case_id,
        output_dir,
        physical_archive,
        measurement_manifest,
        measurement_archives,
        uncertainty_archives,
        expected_h1=expected_h1,
    )
    manifest = inputs["manifest"]
    tracker = manifest["tracker"]
    _require(
        tracker.get("execution_status") == RETAINED_MEASUREMENT_FAILURE_STATUS
        and tracker.get("failure_code") == failure_code
        and tracker.get("inference_executed") is False,
        "retained case requires its exact failed-measurement carrier",
    )
    retained_source = manifest["inputs"].get("retained_failure_source")
    _require(
        isinstance(retained_source, Mapping)
        and retained_source.get("failure_code") == failure_code,
        "retained case lacks materializer source provenance",
    )
    frame_zero_record = retained_source["frame_zero_manifest"]
    frame_zero_snapshot = _snapshot_regular_file(frame_zero_record["path"])
    _require(
        frame_zero_snapshot.file_sha256 == frame_zero_record["file_sha256"],
        "retained frame-zero source file changed",
    )
    from .deform360_adaptive_covariance_confirmation_failure import (
        validate_native_original_splat_frame_zero,
    )

    native_frame_zero = validate_native_original_splat_frame_zero(
        lock_path,
        h2_commit,
        frame_zero_snapshot.path.parent,
        expected_h1=expected_h1,
    )
    _require(
        native_frame_zero.get("result_sha256") == frame_zero_record["result_sha256"]
        and native_frame_zero.get("material_point_count") == inputs["physical"].shape[1]
        and native_frame_zero.get("material_identity_sha256")
        == _external_array_sha256(inputs["frame_zero"]),
        "retained physical archive differs from native original-Splat identities",
    )
    physical_snapshot = inputs["physical_snapshot"]
    persistence = inputs["persistence"]
    _require(
        inputs["physical"].shape[1] > 16
        and all(
            np.array_equal(physical_snapshot.arrays[role], persistence)
            for role in (
                "prediction_m",
                "persistence_m",
                "driven_readout_m",
                "zero_action_readout_m",
            )
        )
        and np.array_equal(
            physical_snapshot.arrays["action_support"],
            np.zeros_like(physical_snapshot.arrays["action_support"]),
        ),
        "retained physical package is not exact persistence",
    )
    for budget in CAMERA_BUDGETS:
        measurement = inputs["measurements"][budget]
        uncertainty = inputs["uncertainties"][budget]
        validity = measurement["measurement_validity"]
        _require(
            len(measurement["center_ids"]) == 16
            and not np.any(validity[1:])
            and np.array_equal(
                measurement["measurement_m"][0, validity[0]],
                inputs["frame_zero"][validity[0]],
            )
            and not np.any(uncertainty["measurement_covariance_valid"]),
            "retained carrier contains a dynamic measurement or invalid center plan",
        )
    _recheck_snapshots((*inputs["snapshots"], frame_zero_snapshot))
    arrays = {role: persistence for role in ARRAY_ROLES}
    return seal_confirmation_case(
        lock_path,
        h2_commit,
        case_id,
        output_dir,
        arrays,
        _selected_cameras(inputs),
        _retained_failure_routing(_selected_cameras(inputs)),
        _full_technical_disposition(
            status="retained_technical_failure",
            center_ids=np.asarray(
                inputs["measurements"][4]["center_ids"],
                dtype=np.int64,
            ),
            input_hashes=inputs["input_hashes"],
            fixed_diagnostics={},
            failure_code=failure_code,
            fallback_label="persistence_only",
        ),
        expected_h1=expected_h1,
    )


__all__ = [
    "CAMERA_BUDGETS",
    "MEASUREMENT_ARRAY_ROLES",
    "PHYSICAL_ARRAY_ROLES",
    "RETAINED_FAILURE_CODES",
    "UNCERTAINTY_ARRAY_ROLES",
    "assemble_and_seal_confirmation_prediction",
    "seal_retained_confirmation_failure",
]
