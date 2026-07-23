"""Outcome-only adapter for the H2 adaptive-covariance confirmation.

This module has two deliberately separate halves.

The compatibility builder is target-free.  It first replays the complete
34-case prediction barrier, then converts only already-sealed prediction and
nested-measurement artifacts into the filenames consumed by the checksum-bound
Deform360 outcome stages from commit ``29091daa``.  In particular, the copied
measurement is the exact nested eight-view archive and its camera order must
match the camera order sealed in the corresponding case diagnostic.

The execution half patches only the old protocol/case authorization aliases.
The patched authorizer replays the complete H2 barrier and all compatibility
bindings before either frozen stage can read a future frame.  The old
``calibration`` role is used solely as a target-gate compatibility role: H2 has
one target-closed 34-case panel, and its complete prediction barrier is the
authorization capability.

No scoring function is imported or called here.  The final loader returns the
native official ``target_m``, visibility, and validity arrays produced by the
frozen outcome stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

import numpy as np

from .deform360_adaptive_covariance_confirmation_external_runtime import (
    DEFORM360_EXECUTION_COMMIT,
    EXTERNAL_EXECUTION_COMMIT,
    activate_confirmation_external_runtime,
    load_confirmation_execution_protocol,
    validate_deform360_execution_repository,
    validate_external_execution_repository,
    validate_external_module_provenance,
    validate_two_commit_execution_repository,
)
from .deform360_adaptive_covariance_confirmation_failure import (
    PCD_STAGE_SOURCE_SHA256,
    validate_original_splat_identity_persistence_manifest,
)
from .deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    load_confirmation_cohort_lock,
)
from .deform360_adaptive_covariance_confirmation_measurement import (
    ARTIFACT_KIND as NESTED_MEASUREMENT_ARTIFACT_KIND,
    IDENTITY_PERSISTENCE_ADAPTER_KEY,
    IDENTITY_PERSISTENCE_ADAPTER_KIND,
    IDENTITY_PERSISTENCE_POLICY,
    EXTERNAL_PHYSICAL_ARRAY_ROLES,
    MANIFEST_FILENAME as NESTED_MEASUREMENT_MANIFEST_FILENAME,
    MEASUREMENT_ARCHIVE_FILENAME,
    MEASUREMENT_ARRAY_ROLES,
    RETAINED_FAILURE_CAMERA_ACCOUNTING,
    RETAINED_MEASUREMENT_FAILURE_STATUS,
    SCHEMA_VERSION as NESTED_MEASUREMENT_SCHEMA_VERSION,
    UNCERTAINTY_ARCHIVE_FILENAME,
    UNCERTAINTY_ARRAY_ROLES,
)
from .deform360_adaptive_covariance_confirmation_seal import (
    ARRAY_ARCHIVE_FILENAME,
    CASE_MANIFEST_FILENAME,
    DIAGNOSTIC_FILENAME,
    RETAINED_FAILURE_CODES,
    array_sha256,
    validate_confirmation_prediction_barrier,
)
from .deform360_adaptive_covariance_confirmation_source_custody import (
    PREFIX_FRAME_COUNT as SOURCE_CUSTODY_PREFIX_FRAME_COUNT,
    validate_confirmation_source_custody_envelope,
    validate_confirmation_source_custody_seal,
)
from .deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    ALLTRACKER_SOURCE_TREE,
)


SCHEMA_VERSION = 1
COMPATIBILITY_ARTIFACT_KIND = (
    "Deform360AdaptiveCovarianceConfirmationOutcomeCompatibilityV1"
)
COMPATIBILITY_CASE_REPORT_KIND = (
    "Deform360AdaptiveCovarianceConfirmationOutcomeCompatibilityReportV1"
)
COMPATIBILITY_CASE_SEAL_KIND = (
    "Deform360AdaptiveCovarianceConfirmationOutcomeCompatibilityCaseSealV1"
)
COMPATIBILITY_MANIFEST_FILENAME = "confirmation_outcome_compatibility.json"
COMPATIBILITY_PREDICTION_ROOT = "predictions"
COMPATIBILITY_MEASUREMENT_ROOT = "measurements"

# These names are read directly by the frozen 29091 outcome stages.
EXTERNAL_PREDICTION_ARCHIVE_FILENAME = "bias_aware_prediction.npz"
EXTERNAL_PREDICTION_REPORT_FILENAME = "bias_aware_prediction.json"
EXTERNAL_PREDICTION_SEAL_FILENAME = "bias_aware_prediction_seal.json"
EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME = "authorized_future_manifest.json"
EXTERNAL_TARGET_ARCHIVE_FILENAME = "target_trajectory.npz"
EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME = "authorized_outcome_manifest.json"

EXTERNAL_AUTHORIZED_FUTURE_KIND = "Deform360BiasAwareProspectiveAuthorizedFuture"
EXTERNAL_AUTHORIZED_OUTCOME_KIND = "Deform360BiasAwareProspectiveAuthorizedOutcome"
EXTERNAL_OUTCOME_STAGE_SHA256: Mapping[str, str] = {
    "scripts/remote/stage_deform360_bias_aware_authorized_future.py": (
        "e5c09b1594fe45e2dc764d03cddae715c2c64149cffba002d1d5881cd0ce0480"
    ),
    "scripts/remote/build_deform360_bias_aware_authorized_outcome.py": (
        "dde40607c4006cba8e21bc0b85ac11b06f4278863eb83a3023e3761584a1ec87"
    ),
}
EXTERNAL_OUTCOME_STAGE_SCRIPTS: Mapping[str, str] = {
    "authorized-future": "stage_deform360_bias_aware_authorized_future.py",
    "authorized-outcome": "build_deform360_bias_aware_authorized_outcome.py",
}
EXTERNAL_GENERIC_SELECTOR_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
EXTERNAL_SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)

_COMPATIBILITY_BOUNDARY = {
    "complete_prediction_barrier_validated": True,
    "case_seals_validated": True,
    "nested_eight_view_measurements_validated": True,
    "future_rgb_read": False,
    "future_geometry_read": False,
    "target_array_read": False,
    "target_metric_or_outcome_score_computed": False,
    "compatibility_artifacts_only": True,
}
_NESTED_MEASUREMENT_BOUNDARY = {
    "target_path_argument_accepted": False,
    "outcome_path_argument_accepted": False,
    "target_metric_or_outcome_score_computed": False,
    "future_geometry_read": False,
    "video_prefix_rule": "update u reads exactly frames [0,u]",
    "maximum_video_frame_read_by_update": [19, 38, 57],
    "four_view_decision_precedes_shadow_extra_four": True,
}
_NESTED_CAMERA_ACCOUNTING = {
    "adaptive_charge_is_causal_offline_policy_demand": True,
    "all_eight_streams_eventually_tracked_for_fixed8_shadow": True,
    "realized_acquisition_or_wall_clock_saving_claimed": False,
    "frame_zero_all_camera_planning_excluded": True,
}
_AUTHORIZED_FUTURE_BOUNDARY = {
    "future_rgb_read_after_cohort_authorization": True,
    "future_masks_created_after_cohort_authorization": True,
    "future_dense_reconstruction_created": False,
    "future_particle_tracks_created": False,
    "target_metric_computed": False,
    "future_tactile_read": False,
}
_AUTHORIZED_OUTCOME_BOUNDARY = {
    "prediction_cohort_verified_before_target_construction": True,
    "future_tactile_read": False,
    "prediction_metric_computed": False,
}
_NESTED_MANIFEST_KEYS = {
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
_RETAINED_FAILURE_TRACKER_KEYS = {
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
_RETAINED_FAILURE_SOURCE_KEYS = {
    "failure_code",
    "prediction_prefix_manifest",
    "frame_zero_manifest",
    "processed_prefix_episode",
    "dynamic_point_observations_available",
}
_SOURCE_STAGE_LINEAGE_KEYS = {
    "prediction_prefix_manifest",
    "frame_zero_manifest",
    "source_preparation_manifest_file_sha256",
    "source_custody_seal",
}
_SOURCE_STAGE_MANIFEST_RECORD_KEYS = {
    "path",
    "file_sha256",
    "result_sha256",
}
_SOURCE_CUSTODY_RECORD_KEYS = {
    "path",
    "file_sha256",
    "artifact_sha256",
}
_RETAINED_FAILURE_TRACKER_RECORD_KEYS = {
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
_RETAINED_FAILURE_CENTER_RECORD_KEYS = {
    "center_id",
    "measurement_available",
    "covariance_valid",
    "decision",
    "failure_code",
}
_FULL_SHA1_LENGTH = 40
_FULL_SHA256_LENGTH = 64
_AUTHORIZED_PREFIX_FRAME_COUNT = SOURCE_CUSTODY_PREFIX_FRAME_COUNT


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ConfirmationOutcomeCompatibility:
    """Validated target-free compatibility binding for one complete H2 cohort."""

    root: Path
    manifest_path: Path
    prediction_root: Path
    measurement_root: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class ConfirmationNativeOfficialTarget:
    """Validated native target arrays plus their immutable scoring evidence."""

    target_m: np.ndarray
    target_visibility: np.ndarray
    target_validity: np.ndarray
    evidence: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _FULL_SHA1_LENGTH
        and value != "0" * _FULL_SHA1_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _FULL_SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _result_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _external_array_sha256(value: np.ndarray) -> str:
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


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"JSON artifact has duplicate key: {key}")
        value[key] = item
    return value


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_file(path: str | Path, *, label: str) -> _FileSnapshot:
    source = Path(path).absolute()
    before = os.lstat(source)
    _require(not stat.S_ISLNK(before.st_mode), f"{label} is a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    _require(before.st_nlink == 1, f"{label} is hard-linked")
    _require(source.resolve(strict=True) == source, f"{label} is noncanonical")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{label} changed while opening",
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity = _stat_identity(opened)
        _require(
            identity == _stat_identity(after) == _stat_identity(current)
            and len(payload) == opened.st_size,
            f"{label} changed while reading",
        )
    finally:
        os.close(descriptor)
    return _FileSnapshot(
        path=source,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=identity,
    )


def _recheck_snapshot(snapshot: _FileSnapshot, *, label: str) -> None:
    current = _snapshot_file(snapshot.path, label=label)
    _require(
        current.identity == snapshot.identity
        and current.sha256 == snapshot.sha256
        and current.payload == snapshot.payload,
        f"{label} changed after validation",
    )


def _canonical_directory(path: str | Path, *, label: str) -> Path:
    root = Path(path).absolute()
    _require(
        root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root,
        f"{label} is invalid",
    )
    return root


def _decoded_raw_rgb_prefix_sha256(path: Path, *, frame_count: int) -> str:
    """Hash decoded RGB bytes with the frozen authorized-future convention."""

    _require(frame_count > 0, "decoded prefix frame count is invalid")
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(process.stdout is not None, "ffmpeg output pipe is unavailable")
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    stderr = b"" if process.stderr is None else process.stderr.read()
    _require(
        process.wait() == 0,
        "cannot decode authorized RGB prefix: "
        + stderr.decode("utf-8", errors="replace")[-1000:],
    )
    return digest.hexdigest()


def _validate_authorized_future_files(
    future_root: Path,
    future: Mapping[str, Any],
    cameras: Sequence[str],
) -> dict[str, _FileSnapshot]:
    """Replay every file initially published by the authorized-future stage."""

    episode = _canonical_directory(
        future_root / "episode_0000",
        label="authorized future episode",
    )
    outputs = future["outputs_sha256"]
    expected_outputs = {
        "robot": episode / "robot" / "robot.npz",
        "frame_zero_splat": episode / "splatfacto" / "splat_0.ply",
        "intrinsics": episode / "undistorted_intrinsics.npy",
        "extrinsics": episode / "extrinsics.npy",
    }
    snapshots: dict[str, _FileSnapshot] = {}
    for role, path in expected_outputs.items():
        snapshot = _snapshot_file(path, label=f"authorized future {role}")
        _require(
            snapshot.sha256 == outputs[role],
            f"authorized future {role} file changed",
        )
        snapshots[role] = snapshot

    records = future["camera_records"]
    by_camera = {str(record["camera"]): record for record in records}
    _require(
        tuple(by_camera) == tuple(cameras),
        "authorized future camera file ordering changed",
    )
    for camera in cameras:
        camera_root = _canonical_directory(
            episode / camera,
            label=f"authorized future camera {camera}",
        )
        record = by_camera[camera]
        for role, filename, digest_key in (
            ("video", "undistorted.mp4", "video_sha256"),
            ("timestamps", "aligned_timestamps.txt", "timestamps_sha256"),
            ("masks", "mask_refined.h5", "masks_sha256"),
        ):
            snapshot = _snapshot_file(
                camera_root / filename,
                label=f"authorized future {camera} {role}",
            )
            _require(
                snapshot.sha256 == record[digest_key],
                f"authorized future {camera} {role} file changed",
            )
            snapshots[f"{camera}:{role}"] = snapshot
        _require(
            _decoded_raw_rgb_prefix_sha256(
                snapshots[f"{camera}:video"].path,
                frame_count=_AUTHORIZED_PREFIX_FRAME_COUNT,
            )
            == record["decoded_sealed_prefix_sha256"],
            f"authorized future {camera} decoded prefix changed",
        )
    return snapshots


def _validate_authorized_outcome_stage_metadata(
    future_root: Path,
    stage_metadata: Mapping[str, Any],
    cameras: Sequence[str],
) -> tuple[_FileSnapshot, ...]:
    """Bind declared processing metadata to the exact files consumed by scoring."""

    episode = _canonical_directory(
        future_root / "episode_0000",
        label="authorized outcome episode",
    )
    records: list[tuple[str, Path, str]] = [
        (
            "reconstruction",
            episode / "splatfacto" / "splatfacto.meta.json",
            str(stage_metadata["reconstruction"]),
        ),
        (
            "point cloud",
            episode / "pcd_clean" / "pcd_clean.meta.json",
            str(stage_metadata["point_cloud"]),
        ),
    ]
    for camera in cameras:
        records.extend(
            (
                (
                    f"{camera} gripper masks",
                    episode / camera / "rendered_urdf.meta.json",
                    str(stage_metadata["gripper_masks"][camera]),
                ),
                (
                    f"{camera} depth",
                    episode / camera / "rendered_depth.meta.json",
                    str(stage_metadata["depth"][camera]),
                ),
                (
                    f"{camera} tracking",
                    episode / camera / "tracking" / "tracking.meta.json",
                    str(stage_metadata["tracking"][camera]),
                ),
            )
        )
    snapshots: list[_FileSnapshot] = []
    for label, path, expected_sha256 in records:
        snapshot = _snapshot_file(
            path,
            label=f"authorized outcome {label} metadata",
        )
        _require(
            snapshot.sha256 == expected_sha256,
            f"authorized outcome {label} metadata changed",
        )
        snapshots.append(snapshot)
    return tuple(snapshots)


def _custody_inventory_file_sha256(
    custody: Mapping[str, Any],
    inventory: str,
    relative_path: str,
) -> str:
    inventories = custody.get("inventories")
    tree = inventories.get(inventory) if isinstance(inventories, Mapping) else None
    records = tree.get("records") if isinstance(tree, Mapping) else None
    matches = (
        [
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("type") == "file"
            and record.get("path") == relative_path
        ]
        if isinstance(records, list)
        else []
    )
    _require(
        len(matches) == 1 and _is_sha256(matches[0].get("sha256")),
        f"source-custody inventory lacks {inventory}/{relative_path}",
    )
    return str(matches[0]["sha256"])


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _write_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _case_identity(lock: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for stratum, object_records in lock["cohort"].items():
        for object_record in object_records:
            for episode in object_record["episodes"]:
                if episode["case_id"] == case_id:
                    matches.append(
                        {
                            "case_id": case_id,
                            "stratum": stratum,
                            "object_id": object_record["object_id"],
                            "episode_id": int(episode["episode_id"]),
                        }
                    )
    _require(len(matches) == 1, "case is outside the exact H2 lock")
    return matches[0]


def _external_case_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case": identity["case_id"],
        "object_id": identity["object_id"],
        "episode_id": int(identity["episode_id"]),
        "episode_key": (f"{identity['object_id']}/{int(identity['episode_id'])}"),
        "stratum": identity["stratum"],
        "role": "calibration",
    }


def _lock_binding(
    lock: Mapping[str, Any],
    lock_snapshot: _FileSnapshot,
    h2_commit: str,
) -> dict[str, str]:
    return {
        "implementation_commit_h1": lock["two_commit_freeze"][
            "implementation_commit_h1"
        ],
        "cohort_lock_commit_h2": h2_commit,
        "cohort_lock_artifact_sha256": lock["artifact_sha256"],
        "cohort_lock_file_sha256": lock_snapshot.sha256,
    }


def _load_npz_snapshot(
    snapshot: _FileSnapshot,
    *,
    expected_roles: set[str],
    label: str,
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as stored:
            _require(set(stored.files) == expected_roles, f"{label} roles changed")
            arrays = {role: np.asarray(stored[role]).copy() for role in expected_roles}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{label} is invalid") from error
    return arrays


def _barrier_record_by_case(barrier: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = barrier["ordered_case_seals"]
    return {str(record["case_id"]): record for record in records}


def _case_seal_inputs(
    case_id: str,
    case_root: Path,
    *,
    barrier_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], tuple[str, ...], dict[str, Any]]:
    _require(case_root.name == case_id, "case seal directory name changed")
    manifest_snapshot = _snapshot_file(
        case_root / CASE_MANIFEST_FILENAME,
        label=f"{case_id} case seal",
    )
    diagnostic_snapshot = _snapshot_file(
        case_root / DIAGNOSTIC_FILENAME,
        label=f"{case_id} target-free diagnostic",
    )
    archive_snapshot = _snapshot_file(
        case_root / ARRAY_ARCHIVE_FILENAME,
        label=f"{case_id} sealed prediction archive",
    )
    _require(
        manifest_snapshot.sha256 == barrier_record["manifest_file_sha256"]
        and diagnostic_snapshot.sha256 == barrier_record["diagnostic_file_sha256"]
        and archive_snapshot.sha256 == barrier_record["prediction_archive_sha256"],
        f"{case_id} differs from the complete prediction barrier",
    )
    manifest = _load_json_bytes(manifest_snapshot.payload, label="case seal")
    diagnostic = _load_json_bytes(
        diagnostic_snapshot.payload,
        label="target-free diagnostic",
    )
    _require(
        manifest.get("artifact_sha256") == barrier_record["manifest_artifact_sha256"]
        and diagnostic.get("artifact_sha256")
        == barrier_record["diagnostic_artifact_sha256"],
        f"{case_id} barrier artifact identity changed",
    )
    arrays = _load_npz_snapshot(
        archive_snapshot,
        expected_roles={
            "physical_prior_m",
            "fixed_4_rbf_prediction_m",
            "fixed_8_rbf_prediction_m",
            "adaptive_prediction_m",
            "selected_raw_prediction_m",
            "persistence_m",
        },
        label="sealed prediction archive",
    )
    cameras = diagnostic.get("nested_selected_cameras")
    _require(
        isinstance(cameras, Mapping)
        and set(cameras) == {"4", "8"}
        and isinstance(cameras["4"], list)
        and isinstance(cameras["8"], list)
        and len(cameras["4"]) == 4
        and len(cameras["8"]) == 8
        and cameras["8"][:4] == cameras["4"]
        and len(set(cameras["8"])) == 8
        and all(isinstance(camera, str) and camera for camera in cameras["8"]),
        f"{case_id} sealed nested camera panel changed",
    )
    source_record = {
        "case_seal_root": str(case_root),
        "case_seal_file_sha256": manifest_snapshot.sha256,
        "case_seal_artifact_sha256": manifest["artifact_sha256"],
        "prediction_archive_file_sha256": archive_snapshot.sha256,
        "prediction_arrays": {
            role: array_sha256(arrays[role])
            for role in (
                "adaptive_prediction_m",
                "selected_raw_prediction_m",
            )
        },
        "diagnostic_file_sha256": diagnostic_snapshot.sha256,
        "diagnostic_artifact_sha256": diagnostic["artifact_sha256"],
    }
    return source_record, arrays, tuple(cameras["8"]), diagnostic


def _measurement_array_records(
    arrays: Mapping[str, np.ndarray],
    roles: Sequence[str] = MEASUREMENT_ARRAY_ROLES,
) -> dict[str, dict[str, Any]]:
    return {
        role: {
            "dtype": arrays[role].dtype.str,
            "shape": list(arrays[role].shape),
            "array_sha256": array_sha256(arrays[role]),
        }
        for role in roles
    }


def _nested_output_archive(
    measurement_root: Path,
    outputs: Mapping[str, Any],
    *,
    case_id: str,
    budget: int,
    archive_role: str,
    filename: str,
    expected_roles: Sequence[str],
) -> tuple[_FileSnapshot, dict[str, np.ndarray]]:
    record = outputs[str(budget)][archive_role]
    _require(
        isinstance(record, Mapping)
        and set(record) == {"relative_path", "sha256", "size_bytes", "arrays"}
        and record.get("relative_path") == f"budget-{budget}/{filename}"
        and _is_sha256(record.get("sha256"))
        and type(record.get("size_bytes")) is int
        and record["size_bytes"] > 0,
        f"{case_id} {budget}-view {archive_role} binding changed",
    )
    snapshot = _snapshot_file(
        measurement_root / f"budget-{budget}" / filename,
        label=f"{case_id} {budget}-view {archive_role}",
    )
    arrays = _load_npz_snapshot(
        snapshot,
        expected_roles=set(expected_roles),
        label=f"{budget}-view {archive_role}",
    )
    _require(
        snapshot.sha256 == record["sha256"]
        and len(snapshot.payload) == record["size_bytes"]
        and record.get("arrays") == _measurement_array_records(arrays, expected_roles),
        f"{case_id} {budget}-view {archive_role} content changed",
    )
    return snapshot, arrays


def _bound_result_manifest(
    record: object,
    *,
    label: str,
) -> tuple[_FileSnapshot, dict[str, Any]]:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"path", "file_sha256", "result_sha256"}
        and isinstance(record.get("path"), str)
        and bool(record["path"])
        and _is_sha256(record.get("file_sha256"))
        and _is_sha256(record.get("result_sha256")),
        f"{label} binding changed",
    )
    snapshot = _snapshot_file(str(record["path"]), label=label)
    value = _load_json_bytes(snapshot.payload, label=label)
    _require(
        snapshot.sha256 == record["file_sha256"]
        and value.get("result_sha256")
        == record["result_sha256"]
        == _result_sha256(value),
        f"{label} content changed",
    )
    return snapshot, value


def _identity_persistence_provenance(
    frame_zero_snapshot: _FileSnapshot,
    frame_zero_manifest: Mapping[str, Any],
    *,
    lock_path: str | Path,
    h2_commit: str,
    lock_binding: Mapping[str, Any],
    expected_h1: str | None,
) -> dict[str, Any] | None:
    marker = frame_zero_manifest.get(IDENTITY_PERSISTENCE_ADAPTER_KEY)
    if marker is None:
        _require(
            frame_zero_manifest.get("material_point_source")
            != IDENTITY_PERSISTENCE_POLICY,
            "identity-persistence policy lacks its adapter marker",
        )
        return None
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "policy",
        "implementation_commit_h1",
        "cohort_lock_commit_h2",
        "cohort_lock_artifact_sha256",
        "adapter_source_sha256",
        "deform360_revision",
        "pcd_stage_source_sha256",
        "frame_zero_splat_file_sha256",
        "seed_parameters",
        "previous_material",
        "adapted_material",
        "preserved_fallback_diagnostics_sha256",
        "physical_twin_admitted",
    }
    _require(
        isinstance(marker, Mapping)
        and set(marker) == expected_keys
        and marker.get("schema_version") == 1
        and marker.get("artifact_kind") == IDENTITY_PERSISTENCE_ADAPTER_KIND
        and marker.get("policy") == IDENTITY_PERSISTENCE_POLICY
        and marker.get("implementation_commit_h1")
        == lock_binding["implementation_commit_h1"]
        and marker.get("cohort_lock_commit_h2") == lock_binding["cohort_lock_commit_h2"]
        and marker.get("cohort_lock_artifact_sha256")
        == lock_binding["cohort_lock_artifact_sha256"]
        and marker.get("deform360_revision") == DEFORM360_EXECUTION_COMMIT
        and marker.get("pcd_stage_source_sha256") == PCD_STAGE_SOURCE_SHA256
        and marker.get("physical_twin_admitted") is False
        and frame_zero_manifest.get("material_point_source")
        == IDENTITY_PERSISTENCE_POLICY
        and frame_zero_manifest.get("physical_policy") == "persistence_only",
        "identity-persistence frame-zero marker changed",
    )
    for role in (
        "adapter_source_sha256",
        "frame_zero_splat_file_sha256",
        "preserved_fallback_diagnostics_sha256",
    ):
        _require(
            _is_sha256(marker.get(role)),
            f"identity-persistence {role} changed",
        )
    source_snapshot = _snapshot_file(
        Path(__file__).with_name(
            "deform360_adaptive_covariance_confirmation_failure.py"
        ),
        label="identity-persistence adapter source",
    )
    _require(
        marker["adapter_source_sha256"] == source_snapshot.sha256,
        "identity-persistence adapter source differs from H1",
    )
    seed_parameters = marker.get("seed_parameters")
    _require(
        isinstance(seed_parameters, Mapping)
        and set(seed_parameters) == {"seed_count", "crop_half_extent_m", "rng_seed"}
        and seed_parameters.get("seed_count") == 10000
        and seed_parameters.get("crop_half_extent_m") == 0.5
        and seed_parameters.get("rng_seed") == 0,
        "identity-persistence seed parameters changed",
    )
    for role in ("previous_material", "adapted_material"):
        material = marker.get(role)
        _require(
            isinstance(material, Mapping)
            and set(material)
            == {"source", "point_count", "array_sha256", "file_sha256"}
            and isinstance(material.get("source"), str)
            and bool(material["source"])
            and type(material.get("point_count")) is int
            and material["point_count"] > 16
            and _is_sha256(material.get("array_sha256"))
            and _is_sha256(material.get("file_sha256")),
            f"identity-persistence {role} changed",
        )
    _require(
        marker["previous_material"]["source"] == "strict-multiview-visual-hull-surface"
        and marker["adapted_material"]["source"] == IDENTITY_PERSISTENCE_POLICY,
        "identity-persistence material source changed",
    )
    adapted = marker["adapted_material"]
    outputs = frame_zero_manifest.get("outputs_sha256")
    _require(
        isinstance(outputs, Mapping)
        and adapted["point_count"] == frame_zero_manifest.get("material_point_count")
        and adapted["array_sha256"]
        == frame_zero_manifest.get("material_identity_sha256")
        and adapted["file_sha256"] == outputs.get("frame_zero_points")
        and marker["frame_zero_splat_file_sha256"] == outputs.get("frame_zero_splat")
        and marker["preserved_fallback_diagnostics_sha256"]
        == hashlib.sha256(
            _canonical_bytes(frame_zero_manifest.get("fallback_diagnostics"))
        ).hexdigest(),
        "identity-persistence material binding changed",
    )
    archive_snapshot = _snapshot_file(
        frame_zero_snapshot.path.with_name("frame_zero_points.npz"),
        label="adapted frame-zero identity archive",
    )
    _require(
        archive_snapshot.sha256 == adapted["file_sha256"],
        "adapted frame-zero identity archive changed",
    )
    splat_snapshot = _snapshot_file(
        frame_zero_snapshot.path.parent
        / "frame-zero"
        / "episode_0000"
        / "splatfacto"
        / "splat_0.ply",
        label="identity-persistence frame-zero Splat",
    )
    _require(
        splat_snapshot.sha256 == marker["frame_zero_splat_file_sha256"],
        "identity-persistence frame-zero Splat changed",
    )
    arrays = _load_npz_snapshot(
        archive_snapshot,
        expected_roles={"points_m", "colors"},
        label="adapted frame-zero identity archive",
    )
    points = arrays["points_m"]
    _require(
        points.ndim == 2
        and points.shape == (adapted["point_count"], 3)
        and np.issubdtype(points.dtype, np.floating)
        and np.all(np.isfinite(points))
        and _external_array_sha256(points) == adapted["array_sha256"],
        "adapted frame-zero material identity changed",
    )
    validated = validate_original_splat_identity_persistence_manifest(
        lock_path,
        h2_commit,
        frame_zero_snapshot.path.parent,
        expected_h1=expected_h1,
    )
    _require(
        dict(validated) == dict(frame_zero_manifest),
        "identity-persistence validator observed another frame-zero manifest",
    )
    return {
        "frame_zero_manifest_path": str(frame_zero_snapshot.path),
        "frame_zero_manifest_file_sha256": frame_zero_snapshot.sha256,
        "frame_zero_manifest_result_sha256": frame_zero_manifest["result_sha256"],
        "adapter_source_file_sha256": source_snapshot.sha256,
        "frame_zero_splat_file_sha256": marker["frame_zero_splat_file_sha256"],
        "adapted_material": dict(adapted),
    }


def _source_stage_lineage_provenance(
    value: object,
    *,
    case_id: str,
    lock_path: str | Path,
    h2_commit: str,
    expected_h1: str | None,
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value) == _SOURCE_STAGE_LINEAGE_KEYS
        and _is_sha256(value.get("source_preparation_manifest_file_sha256")),
        f"{case_id} source-stage lineage changed",
    )
    result: dict[str, Any] = {
        "source_preparation_manifest_file_sha256": value[
            "source_preparation_manifest_file_sha256"
        ],
    }
    paths: list[Path] = []
    expected_filenames = {
        "prediction_prefix_manifest": "prediction_prefix_manifest.json",
        "frame_zero_manifest": "frame_zero_reconstruction_manifest.json",
    }
    for role, filename in expected_filenames.items():
        record = value.get(role)
        _require(
            isinstance(record, Mapping)
            and set(record) == _SOURCE_STAGE_MANIFEST_RECORD_KEYS
            and isinstance(record.get("path"), str)
            and bool(record["path"])
            and _is_sha256(record.get("file_sha256"))
            and _is_sha256(record.get("result_sha256")),
            f"{case_id} {role} lineage changed",
        )
        path = Path(str(record["path"]))
        _require(
            path.is_absolute() and path.name == filename,
            f"{case_id} {role} path changed",
        )
        paths.append(path)
        result[role] = dict(record)
    _require(
        paths[0].parent == paths[1].parent and paths[0].parent.name == case_id,
        f"{case_id} source-stage case path changed",
    )
    custody_record = value.get("source_custody_seal")
    _require(
        isinstance(custody_record, Mapping)
        and set(custody_record) == _SOURCE_CUSTODY_RECORD_KEYS
        and isinstance(custody_record.get("path"), str)
        and Path(custody_record["path"]).is_absolute()
        and _is_sha256(custody_record.get("file_sha256"))
        and _is_sha256(custody_record.get("artifact_sha256")),
        f"{case_id} source-custody lineage changed",
    )
    custody_snapshot = _snapshot_file(
        custody_record["path"],
        label=f"{case_id} source-custody seal",
    )
    _require(
        custody_snapshot.sha256 == custody_record["file_sha256"],
        f"{case_id} source-custody file changed",
    )
    custody = validate_confirmation_source_custody_envelope(
        custody_snapshot.path,
        lock_path,
        h2_commit,
        case_id,
        expected_h1=expected_h1,
        expected_staged_case_dir=paths[0].parent,
    )
    _require(
        custody["artifact_sha256"] == custody_record["artifact_sha256"]
        and custody["manifests"]["prediction_prefix"]["file_sha256"]
        == result["prediction_prefix_manifest"]["file_sha256"]
        and custody["manifests"]["prediction_prefix"]["result_sha256"]
        == result["prediction_prefix_manifest"]["result_sha256"]
        and custody["manifests"]["frame_zero"]["file_sha256"]
        == result["frame_zero_manifest"]["file_sha256"]
        and custody["manifests"]["frame_zero"]["result_sha256"]
        == result["frame_zero_manifest"]["result_sha256"]
        and custody["manifests"]["source_preparation"]["file_sha256"]
        == result["source_preparation_manifest_file_sha256"],
        f"{case_id} source-custody seal differs from source-stage lineage",
    )
    result["source_custody_seal"] = dict(custody_record)
    return result


def _retained_failure_source_provenance(
    manifest: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    lock_path: str | Path,
    h2_commit: str,
    lock_binding: Mapping[str, Any],
    failure_code: str,
    source_stage_lineage: Mapping[str, Any],
    expected_h1: str | None,
) -> dict[str, Any]:
    tracker = manifest.get("tracker")
    _require(
        isinstance(tracker, Mapping)
        and set(tracker) == _RETAINED_FAILURE_TRACKER_KEYS
        and tracker.get("name") == "AllTracker"
        and tracker.get("molmomotion_revision") == ALLTRACKER_MOLMOMOTION_REVISION
        and tracker.get("source_tree") == ALLTRACKER_SOURCE_TREE
        and tracker.get("runtime_source_sha256") == ALLTRACKER_RUNTIME_SOURCE_SHA256
        and tracker.get("checkpoint_sha256") == ALLTRACKER_CHECKPOINT_SHA256
        and tracker.get("device") == "not-executed"
        and tracker.get("execution_status") == RETAINED_MEASUREMENT_FAILURE_STATUS
        and tracker.get("failure_code") == failure_code
        and tracker.get("inference_executed") is False,
        "retained measurement tracker provenance changed",
    )
    inputs = manifest.get("inputs")
    source = (
        inputs.get("retained_failure_source")
        if isinstance(
            inputs,
            Mapping,
        )
        else None
    )
    _require(
        isinstance(inputs, Mapping)
        and set(inputs)
        == {
            "physical_backbone",
            "physical_archive",
            "intrinsics_sha256",
            "extrinsics_sha256",
            "selected_camera_prefixes_and_frame_zero",
            "source_stage_lineage",
            "retained_failure_source",
        }
        and _is_sha256(inputs.get("intrinsics_sha256"))
        and _is_sha256(inputs.get("extrinsics_sha256"))
        and isinstance(source, Mapping)
        and set(source) == _RETAINED_FAILURE_SOURCE_KEYS
        and source.get("failure_code") == failure_code
        and source.get("dynamic_point_observations_available") is False,
        "retained measurement failure source changed",
    )
    prefix_snapshot, prefix_manifest = _bound_result_manifest(
        source["prediction_prefix_manifest"],
        label="retained prediction-prefix manifest",
    )
    frame_zero_snapshot, frame_zero_manifest = _bound_result_manifest(
        source["frame_zero_manifest"],
        label="retained frame-zero manifest",
    )
    external_record = _external_case_record(identity)
    _require(
        prefix_manifest.get("artifact_kind") == "Deform360BiasAwarePredictionPrefix"
        and frame_zero_manifest.get("artifact_kind")
        == "Deform360BiasAwareFrameZeroReconstruction"
        and prefix_manifest.get("protocol_id")
        == frame_zero_manifest.get("protocol_id")
        == PROTOCOL_ID
        and prefix_manifest.get("protocol_config_sha256")
        == frame_zero_manifest.get("protocol_config_sha256")
        == lock_binding["cohort_lock_artifact_sha256"]
        and all(
            prefix_manifest.get(key) == frame_zero_manifest.get(key) == value
            for key, value in external_record.items()
        ),
        "retained external prefix/frame-zero case identity changed",
    )
    prefix_inputs = prefix_manifest.get("inputs_sha256")
    frame_zero_inputs = frame_zero_manifest.get("inputs_sha256")
    _require(
        source_stage_lineage["prediction_prefix_manifest"]
        == {
            "path": str(prefix_snapshot.path),
            "file_sha256": prefix_snapshot.sha256,
            "result_sha256": prefix_manifest["result_sha256"],
        }
        and source_stage_lineage["frame_zero_manifest"]
        == {
            "path": str(frame_zero_snapshot.path),
            "file_sha256": frame_zero_snapshot.sha256,
            "result_sha256": frame_zero_manifest["result_sha256"],
        }
        and isinstance(prefix_inputs, Mapping)
        and prefix_inputs.get("source_preparation_manifest")
        == source_stage_lineage["source_preparation_manifest_file_sha256"]
        and isinstance(frame_zero_inputs, Mapping)
        and frame_zero_inputs.get("prediction_prefix_manifest")
        == prefix_snapshot.sha256,
        "retained source-stage lineage differs from its source manifests",
    )
    processed = source.get("processed_prefix_episode")
    _require(
        isinstance(processed, Mapping)
        and set(processed)
        == {"path", "intrinsics_file_sha256", "extrinsics_file_sha256"}
        and isinstance(processed.get("path"), str)
        and bool(processed["path"])
        and _is_sha256(processed.get("intrinsics_file_sha256"))
        and _is_sha256(processed.get("extrinsics_file_sha256")),
        "retained processed-prefix binding changed",
    )
    processed_root = _canonical_directory(
        str(processed["path"]),
        label="retained processed-prefix episode",
    )
    intrinsics = _snapshot_file(
        processed_root / "undistorted_intrinsics.npy",
        label="retained prefix intrinsics",
    )
    extrinsics = _snapshot_file(
        processed_root / "extrinsics.npy",
        label="retained prefix extrinsics",
    )
    _require(
        intrinsics.sha256 == processed["intrinsics_file_sha256"]
        and extrinsics.sha256 == processed["extrinsics_file_sha256"],
        "retained processed-prefix calibration changed",
    )
    _require(
        inputs["intrinsics_sha256"] == intrinsics.sha256
        and inputs["extrinsics_sha256"] == extrinsics.sha256,
        "retained measurement binds another processed calibration",
    )
    physical_backbone = inputs.get("physical_backbone")
    _require(
        isinstance(physical_backbone, Mapping)
        and set(physical_backbone)
        == {
            "external_backbone_seal_file_sha256",
            "external_backbone_seal_result_sha256",
            "external_physical_manifest_file_sha256",
            "external_physical_manifest_result_sha256",
            "physical_archive_file_sha256",
            "physical_archive_array_sha256",
        }
        and all(
            _is_sha256(physical_backbone.get(key))
            for key in (
                "external_backbone_seal_file_sha256",
                "external_backbone_seal_result_sha256",
                "external_physical_manifest_file_sha256",
                "external_physical_manifest_result_sha256",
                "physical_archive_file_sha256",
            )
        )
        and isinstance(
            physical_backbone.get("physical_archive_array_sha256"),
            Mapping,
        )
        and set(physical_backbone["physical_archive_array_sha256"])
        == set(EXTERNAL_PHYSICAL_ARRAY_ROLES)
        and all(
            _is_sha256(value)
            for value in physical_backbone["physical_archive_array_sha256"].values()
        ),
        "retained physical backbone binding changed",
    )
    physical_archive = inputs.get("physical_archive")
    _require(
        isinstance(physical_archive, Mapping)
        and set(physical_archive) == {"sha256", "frame_zero_array_sha256"}
        and _is_sha256(physical_archive.get("sha256"))
        and _is_sha256(physical_archive.get("frame_zero_array_sha256")),
        "retained physical archive binding changed",
    )
    identity_provenance = _identity_persistence_provenance(
        frame_zero_snapshot,
        frame_zero_manifest,
        lock_path=lock_path,
        h2_commit=h2_commit,
        lock_binding=lock_binding,
        expected_h1=expected_h1,
    )
    return {
        "failure_code": failure_code,
        "prediction_prefix_manifest": {
            "path": str(prefix_snapshot.path),
            "file_sha256": prefix_snapshot.sha256,
            "result_sha256": prefix_manifest["result_sha256"],
        },
        "frame_zero_manifest": {
            "path": str(frame_zero_snapshot.path),
            "file_sha256": frame_zero_snapshot.sha256,
            "result_sha256": frame_zero_manifest["result_sha256"],
        },
        "identity_persistence_adapter": identity_provenance,
    }


def _validate_retained_failure_updates(
    manifest: Mapping[str, Any],
    output_arrays: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    case_id: str,
    sealed_cameras: tuple[str, ...],
    failure_code: str,
) -> None:
    center_ids = [
        int(value)
        for value in np.asarray(
            output_arrays[4]["measurement_archive"]["center_ids"]
        ).tolist()
    ]
    _require(
        np.array_equal(
            output_arrays[8]["measurement_archive"]["center_ids"],
            np.asarray(center_ids),
        ),
        f"{case_id} retained carrier centers differ by budget",
    )
    updates = manifest.get("updates")
    _require(
        isinstance(updates, list) and len(updates) == 3,
        f"{case_id} retained measurement updates changed",
    )
    for frame, record in zip((19, 38, 57), updates, strict=True):
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
            and record.get("four_view_reliable_before_shadow") is False
            and record.get("offline_shadow_extra_four_tracked") is False
            and record.get("adaptive_route") == "physical_prior_fallback"
            and record.get("adaptive_charged_camera_streams") == 8,
            f"{case_id} retained measurement route changed",
        )
        reliability = record.get("budget_reliability")
        _require(
            isinstance(reliability, Mapping)
            and set(reliability) == {"4", "8"}
            and all(
                isinstance(reliability[str(budget)], Mapping)
                and reliability[str(budget)].get("frame") == frame
                and reliability[str(budget)].get("valid_covariance_center_count") == 0
                and reliability[str(budget)].get("valid_covariance_center_ids") == []
                and reliability[str(budget)].get("normalized_covariance_dispersion")
                is None
                and reliability[str(budget)].get("reliable") is False
                for budget in (4, 8)
            ),
            f"{case_id} retained covariance reliability changed",
        )
        trackers = record.get("tracker")
        _require(
            isinstance(trackers, list)
            and len(trackers) == 8
            and [row.get("camera") for row in trackers if isinstance(row, Mapping)]
            == list(sealed_cameras),
            f"{case_id} retained camera carrier order changed",
        )
        for index, tracker in enumerate(trackers):
            _require(
                isinstance(tracker, Mapping)
                and set(tracker) == _RETAINED_FAILURE_TRACKER_RECORD_KEYS
                and tracker.get("prefix_frame_range_half_open") == [0, frame + 1]
                and tracker.get("maximum_video_frame_read") == frame
                and tracker.get("decoded_frame_count") == frame + 1
                and _is_sha256(tracker.get("decoded_rgb_prefix_sha256"))
                and isinstance(tracker.get("original_image_shape"), list)
                and len(tracker["original_image_shape"]) == 2
                and all(
                    type(size) is int and size > 0
                    for size in tracker["original_image_shape"]
                )
                and tracker.get("camera") == sealed_cameras[index]
                and tracker.get("execution_role")
                == ("adaptive_first_four" if index < 4 else "adaptive_eight_escalation")
                and tracker.get("execution_index_within_update") == index
                and tracker.get("four_view_decision_already_materialized")
                is (index >= 4)
                and tracker.get("camera_stream_attempted") is True
                and tracker.get("tracker_inference_executed") is False
                and tracker.get("dynamic_observation_available") is False
                and tracker.get("failure_code") == failure_code
                and isinstance(tracker.get("query_ids"), list)
                and all(
                    type(point_id) is int and point_id in center_ids
                    for point_id in tracker["query_ids"]
                ),
                f"{case_id} retained tracker record changed",
            )
        centers = record.get("centers")
        _require(
            isinstance(centers, Mapping)
            and set(centers) == {"4", "8"}
            and all(
                isinstance(centers[str(budget)], list)
                and [
                    item.get("center_id")
                    for item in centers[str(budget)]
                    if isinstance(item, Mapping)
                ]
                == center_ids
                and all(
                    isinstance(item, Mapping)
                    and set(item) == _RETAINED_FAILURE_CENTER_RECORD_KEYS
                    and item.get("measurement_available") is False
                    and item.get("covariance_valid") is False
                    and item.get("decision")
                    == "retained_technical_failure_measurement_unavailable"
                    and item.get("failure_code") == failure_code
                    for item in centers[str(budget)]
                )
                for budget in (4, 8)
            ),
            f"{case_id} retained center diagnostics changed",
        )
    for budget in (4, 8):
        measurement_arrays = output_arrays[budget]["measurement_archive"]
        uncertainty_arrays = output_arrays[budget]["uncertainty_archive"]
        _require(
            not np.any(measurement_arrays["measurement_validity"][1:])
            and not np.any(uncertainty_arrays["measurement_covariance_valid"][1:]),
            f"{case_id} retained carrier claims a dynamic observation",
        )


def _nested_eight_view_inputs(
    case_id: str,
    measurement_root: Path,
    *,
    identity: Mapping[str, Any],
    lock_path: str | Path,
    h2_commit: str,
    lock_binding: Mapping[str, Any],
    sealed_cameras: tuple[str, ...],
    technical_disposition: Mapping[str, Any],
    expected_h1: str | None,
) -> tuple[dict[str, Any], _FileSnapshot, dict[str, np.ndarray]]:
    _require(
        measurement_root.name == case_id,
        "nested measurement directory name changed",
    )
    _require(
        set(path.name for path in measurement_root.iterdir())
        == {
            NESTED_MEASUREMENT_MANIFEST_FILENAME,
            "budget-4",
            "budget-8",
        },
        "nested measurement directory contents changed",
    )
    manifest_snapshot = _snapshot_file(
        measurement_root / NESTED_MEASUREMENT_MANIFEST_FILENAME,
        label=f"{case_id} nested measurement manifest",
    )
    manifest = _load_json_bytes(
        manifest_snapshot.payload,
        label="nested measurement manifest",
    )
    _require(
        set(manifest) == _NESTED_MANIFEST_KEYS
        and manifest.get("schema_version") == NESTED_MEASUREMENT_SCHEMA_VERSION
        and manifest.get("artifact_kind") == NESTED_MEASUREMENT_ARTIFACT_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("case_identity") == identity
        and manifest.get("lock_binding") == lock_binding,
        f"{case_id} nested measurement envelope changed",
    )
    _require(
        manifest.get("artifact_sha256") == _artifact_sha256(manifest),
        f"{case_id} nested measurement self-checksum changed",
    )
    causal_inputs = technical_disposition.get("causal_input_hashes")
    sealed_measurement = (
        causal_inputs.get("nested_measurement_manifest")
        if isinstance(causal_inputs, Mapping)
        else None
    )
    _require(
        isinstance(sealed_measurement, Mapping)
        and set(sealed_measurement) == {"file_sha256", "artifact_sha256"}
        and sealed_measurement.get("file_sha256") == manifest_snapshot.sha256
        and sealed_measurement.get("artifact_sha256") == manifest["artifact_sha256"],
        f"{case_id} nested measurement differs from the sealed prediction input",
    )
    camera_accounting = manifest.get("camera_accounting")
    standard_measurement = camera_accounting == _NESTED_CAMERA_ACCOUNTING
    retained_failure_measurement = (
        camera_accounting == RETAINED_FAILURE_CAMERA_ACCOUNTING
    )
    _require(
        manifest.get("information_boundary") == _NESTED_MEASUREMENT_BOUNDARY
        and (standard_measurement or retained_failure_measurement),
        f"{case_id} nested measurement crossed the target boundary",
    )
    disposition_is_retained = (
        technical_disposition.get("status") == "retained_technical_failure"
    )
    _require(
        not retained_failure_measurement or disposition_is_retained,
        f"{case_id} tracker-free measurement lacks a retained failure case seal",
    )
    if retained_failure_measurement:
        _require(
            technical_disposition.get("failure_code") in RETAINED_FAILURE_CODES,
            f"{case_id} tracker-free measurement failure code changed",
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
    if retained_failure_measurement:
        expected_input_keys.add("retained_failure_source")
    _require(
        isinstance(inputs, Mapping) and set(inputs) == expected_input_keys,
        f"{case_id} nested measurement input provenance changed",
    )
    source_stage_lineage = _source_stage_lineage_provenance(
        inputs["source_stage_lineage"],
        case_id=case_id,
        lock_path=lock_path,
        h2_commit=h2_commit,
        expected_h1=expected_h1,
    )
    plan = manifest.get("plan")
    _require(
        isinstance(plan, Mapping)
        and isinstance(plan.get("selected_cameras_by_budget"), Mapping)
        and plan["selected_cameras_by_budget"].get("8") == list(sealed_cameras)
        and plan["selected_cameras_by_budget"].get("4") == list(sealed_cameras[:4])
        and plan.get("camera_activation_order") == list(sealed_cameras),
        f"{case_id} nested measurement camera plan differs from its case seal",
    )
    outputs = manifest.get("outputs")
    _require(
        isinstance(outputs, Mapping)
        and set(outputs) == {"4", "8"}
        and all(
            isinstance(outputs[str(budget)], Mapping)
            and set(outputs[str(budget)])
            == {"measurement_archive", "uncertainty_archive"}
            for budget in (4, 8)
        ),
        f"{case_id} nested measurement outputs changed",
    )
    snapshots: dict[int, dict[str, _FileSnapshot]] = {}
    output_arrays: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for budget in (4, 8):
        budget_root = _canonical_directory(
            measurement_root / f"budget-{budget}",
            label=f"{budget}-view measurement directory",
        )
        _require(
            set(path.name for path in budget_root.iterdir())
            == {MEASUREMENT_ARCHIVE_FILENAME, UNCERTAINTY_ARCHIVE_FILENAME},
            f"{case_id} {budget}-view measurement directory changed",
        )
        measurement_snapshot, measurement_arrays = _nested_output_archive(
            measurement_root,
            outputs,
            case_id=case_id,
            budget=budget,
            archive_role="measurement_archive",
            filename=MEASUREMENT_ARCHIVE_FILENAME,
            expected_roles=MEASUREMENT_ARRAY_ROLES,
        )
        uncertainty_snapshot, uncertainty_arrays = _nested_output_archive(
            measurement_root,
            outputs,
            case_id=case_id,
            budget=budget,
            archive_role="uncertainty_archive",
            filename=UNCERTAINTY_ARCHIVE_FILENAME,
            expected_roles=UNCERTAINTY_ARRAY_ROLES,
        )
        snapshots[budget] = {
            "measurement_archive": measurement_snapshot,
            "uncertainty_archive": uncertainty_snapshot,
        }
        output_arrays[budget] = {
            "measurement_archive": measurement_arrays,
            "uncertainty_archive": uncertainty_arrays,
        }
        selected = np.asarray(measurement_arrays["selected_cameras"]).astype(str)
        _require(
            selected.ndim == 1 and tuple(selected.tolist()) == sealed_cameras[:budget],
            f"{case_id} {budget}-view archive camera order changed",
        )
        centers = np.asarray(measurement_arrays["center_ids"])
        update_frames = np.asarray(measurement_arrays["update_frames"])
        measurement_value = measurement_arrays["measurement_m"]
        measurement_validity = measurement_arrays["measurement_validity"]
        covariance = uncertainty_arrays["measurement_covariance_m2"]
        covariance_validity = uncertainty_arrays["measurement_covariance_valid"]
        _require(
            measurement_value.ndim == 3
            and measurement_value.shape[0] == 76
            and measurement_value.shape[1] > 16
            and measurement_value.shape[2] == 3
            and np.issubdtype(measurement_value.dtype, np.floating)
            and measurement_validity.shape == measurement_value.shape[:2]
            and measurement_validity.dtype == np.dtype(bool)
            and covariance.shape == (*measurement_value.shape[:2], 3, 3)
            and np.issubdtype(covariance.dtype, np.floating)
            and covariance_validity.shape == measurement_value.shape[:2]
            and covariance_validity.dtype == np.dtype(bool)
            and centers.ndim == 1
            and len(centers) == 16
            and np.issubdtype(centers.dtype, np.integer)
            and len(set(int(value) for value in centers.tolist())) == 16
            and np.all(centers >= 0)
            and np.all(centers < measurement_value.shape[1])
            and update_frames.dtype == np.dtype(np.int64)
            and np.array_equal(
                update_frames,
                np.asarray((19, 38, 57), dtype=np.int64),
            ),
            f"{case_id} {budget}-view carrier identity changed",
        )
    retained_source: dict[str, Any] | None = None
    if retained_failure_measurement:
        failure_code = str(technical_disposition["failure_code"])
        _validate_retained_failure_updates(
            manifest,
            output_arrays,
            case_id=case_id,
            sealed_cameras=sealed_cameras,
            failure_code=failure_code,
        )
        retained_source = _retained_failure_source_provenance(
            manifest,
            identity=identity,
            lock_path=lock_path,
            h2_commit=h2_commit,
            lock_binding=lock_binding,
            failure_code=failure_code,
            source_stage_lineage=source_stage_lineage,
            expected_h1=expected_h1,
        )
    archive_snapshot = snapshots[8]["measurement_archive"]
    arrays = output_arrays[8]["measurement_archive"]
    source_record = {
        "nested_measurement_root": str(measurement_root),
        "nested_measurement_manifest_file_sha256": manifest_snapshot.sha256,
        "nested_measurement_manifest_artifact_sha256": manifest["artifact_sha256"],
        "eight_view_measurement_file_sha256": archive_snapshot.sha256,
        "eight_view_measurement_arrays": _measurement_array_records(arrays),
        "measurement_execution": (
            "retained-technical-failure-tracker-not-executed"
            if retained_failure_measurement
            else "standard-nested-four-eight"
        ),
        "all_output_archives": {
            str(budget): {
                archive_role: {
                    "file_sha256": snapshot.sha256,
                    "arrays": _measurement_array_records(
                        output_arrays[budget][archive_role],
                        (
                            MEASUREMENT_ARRAY_ROLES
                            if archive_role == "measurement_archive"
                            else UNCERTAINTY_ARRAY_ROLES
                        ),
                    ),
                }
                for archive_role, snapshot in snapshots[budget].items()
            }
            for budget in (4, 8)
        },
        "source_stage_lineage": source_stage_lineage,
        "retained_failure_source": retained_source,
    }
    return source_record, archive_snapshot, arrays


def _load_authorization_sources(
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    *,
    expected_h1: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    _FileSnapshot,
    _FileSnapshot,
    dict[str, Any],
]:
    _require(_is_sha1(h2_commit), "H2 commit is invalid")
    lock_snapshot = _snapshot_file(lock_path, label="H2 cohort lock")
    lock = load_confirmation_cohort_lock(
        lock_snapshot.path,
        expected_implementation_commit_h1=expected_h1,
    )
    _require(
        _load_json_bytes(lock_snapshot.payload, label="H2 cohort lock") == lock,
        "H2 cohort lock changed while loading",
    )
    h1 = lock["two_commit_freeze"]["implementation_commit_h1"]
    _require(h2_commit != h1, "H2 must differ from implementation H1")
    validate_two_commit_execution_repository(
        adapter_repository,
        lock_snapshot.path,
        h1_commit=h1,
        h2_commit=h2_commit,
    )
    expected_cases = tuple(lock["selected_case_ids"])
    _require(
        len(expected_cases) == 34
        and isinstance(case_seal_dirs, Mapping)
        and isinstance(nested_measurement_dirs, Mapping)
        and set(case_seal_dirs) == set(expected_cases)
        and set(nested_measurement_dirs) == set(expected_cases),
        "outcome authorization requires all exact 34 case and measurement roots",
    )
    normalized_case_dirs = {
        case_id: _canonical_directory(
            case_seal_dirs[case_id],
            label=f"{case_id} case seal directory",
        )
        for case_id in expected_cases
    }
    normalized_measurement_dirs = {
        case_id: _canonical_directory(
            nested_measurement_dirs[case_id],
            label=f"{case_id} nested measurement directory",
        )
        for case_id in expected_cases
    }
    barrier = validate_confirmation_prediction_barrier(
        barrier_path,
        lock_snapshot.path,
        h2_commit,
        normalized_case_dirs,
        expected_h1=h1,
    )
    barrier_snapshot = _snapshot_file(
        barrier_path,
        label="complete prediction barrier",
    )
    _require(
        _load_json_bytes(
            barrier_snapshot.payload,
            label="complete prediction barrier",
        )
        == barrier
        and barrier.get("artifact_sha256") == _artifact_sha256(barrier),
        "complete prediction barrier self-checksum changed",
    )
    binding = _lock_binding(lock, lock_snapshot, h2_commit)
    return (
        lock,
        binding,
        lock_snapshot,
        barrier_snapshot,
        {
            "barrier": barrier,
            "case_dirs": normalized_case_dirs,
            "measurement_dirs": normalized_measurement_dirs,
        },
    )


def _build_case_compatibility(
    staging: Path,
    final_root: Path,
    *,
    case_id: str,
    identity: Mapping[str, Any],
    lock_path: str | Path,
    h2_commit: str,
    lock_binding: Mapping[str, Any],
    barrier_record: Mapping[str, Any],
    case_root: Path,
    nested_root: Path,
    expected_h1: str | None,
) -> dict[str, Any]:
    source_case, sealed_arrays, cameras, diagnostic = _case_seal_inputs(
        case_id,
        case_root,
        barrier_record=barrier_record,
    )
    source_measurement, measurement_snapshot, _ = _nested_eight_view_inputs(
        case_id,
        nested_root,
        identity=identity,
        lock_path=lock_path,
        h2_commit=h2_commit,
        lock_binding=lock_binding,
        sealed_cameras=cameras,
        technical_disposition=diagnostic["technical_disposition"],
        expected_h1=expected_h1,
    )
    prediction_dir = staging / COMPATIBILITY_PREDICTION_ROOT / case_id
    measurement_dir = staging / COMPATIBILITY_MEASUREMENT_ROOT / case_id
    prediction_dir.mkdir(parents=True)
    measurement_dir.mkdir(parents=True)
    compatibility_measurement = measurement_dir / MEASUREMENT_ARCHIVE_FILENAME
    _write_bytes(compatibility_measurement, measurement_snapshot.payload)
    measurement_file_sha256 = hashlib.sha256(measurement_snapshot.payload).hexdigest()

    prediction_archive = prediction_dir / EXTERNAL_PREDICTION_ARCHIVE_FILENAME
    with prediction_archive.open("xb") as stream:
        np.savez_compressed(
            stream,
            prediction_m=sealed_arrays["adaptive_prediction_m"],
            selected_raw_backbone=sealed_arrays["selected_raw_prediction_m"],
        )
        stream.flush()
        os.fsync(stream.fileno())
    prediction_snapshot = _snapshot_file(
        prediction_archive,
        label="compatibility prediction archive",
    )
    compatibility_arrays = _load_npz_snapshot(
        prediction_snapshot,
        expected_roles={"prediction_m", "selected_raw_backbone"},
        label="compatibility prediction archive",
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": COMPATIBILITY_CASE_REPORT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_identity": dict(identity),
        "lock_binding": dict(lock_binding),
        "inputs_sha256": {
            "measurement_archive": measurement_file_sha256,
            "case_prediction_archive": source_case["prediction_archive_file_sha256"],
            "nested_measurement_manifest": source_measurement[
                "nested_measurement_manifest_file_sha256"
            ],
        },
        "selected_cameras": list(cameras),
        "information_boundary": dict(_COMPATIBILITY_BOUNDARY),
    }
    report["result_sha256"] = _result_sha256(report)
    report_path = prediction_dir / EXTERNAL_PREDICTION_REPORT_FILENAME
    _write_json(report_path, report)
    report_snapshot = _snapshot_file(
        report_path,
        label="compatibility prediction report",
    )

    final_prediction_dir = final_root / COMPATIBILITY_PREDICTION_ROOT / case_id
    seal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": COMPATIBILITY_CASE_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": lock_binding["cohort_lock_artifact_sha256"],
        **_external_case_record(identity),
        "lock_binding": dict(lock_binding),
        "prediction_archive": {
            "path": str(final_prediction_dir / EXTERNAL_PREDICTION_ARCHIVE_FILENAME),
            "file_sha256": prediction_snapshot.sha256,
            "prediction_array_sha256": _external_array_sha256(
                compatibility_arrays["prediction_m"]
            ),
            "baseline_array_sha256": _external_array_sha256(
                compatibility_arrays["selected_raw_backbone"]
            ),
        },
        "prediction_report": {
            "path": str(final_prediction_dir / EXTERNAL_PREDICTION_REPORT_FILENAME),
            "file_sha256": report_snapshot.sha256,
            "result_sha256": report["result_sha256"],
        },
        "source_case_seal": dict(source_case),
        "source_nested_measurement": dict(source_measurement),
        "selected_cameras": list(cameras),
        "information_boundary": dict(_COMPATIBILITY_BOUNDARY),
    }
    seal["result_sha256"] = _result_sha256(seal)
    seal_path = prediction_dir / EXTERNAL_PREDICTION_SEAL_FILENAME
    _write_json(seal_path, seal)
    seal_snapshot = _snapshot_file(
        seal_path,
        label="compatibility prediction seal",
    )
    return {
        **_external_case_record(identity),
        "case_identity": dict(identity),
        "case_seal": dict(source_case),
        "nested_measurement": dict(source_measurement),
        "selected_cameras": list(cameras),
        "compatibility_prediction": {
            "archive_file_sha256": prediction_snapshot.sha256,
            "archive_arrays": {
                role: _external_array_sha256(compatibility_arrays[role])
                for role in ("prediction_m", "selected_raw_backbone")
            },
            "report_file_sha256": report_snapshot.sha256,
            "report_result_sha256": report["result_sha256"],
            "seal_file_sha256": seal_snapshot.sha256,
            "seal_result_sha256": seal["result_sha256"],
        },
        "compatibility_measurement": {
            "file_sha256": measurement_file_sha256,
            "source_file_sha256": measurement_snapshot.sha256,
        },
    }


def build_confirmation_outcome_compatibility(
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Create target-free compatibility artifacts after the complete barrier."""

    (
        lock,
        binding,
        lock_snapshot,
        barrier_snapshot,
        authorization,
    ) = _load_authorization_sources(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        expected_h1=expected_h1,
    )
    output = Path(output_dir).absolute()
    _require(
        not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink()
        and output.parent.resolve(strict=True) == output.parent,
        "compatibility output is invalid",
    )
    for source in (
        lock_snapshot.path,
        barrier_snapshot.path,
        *authorization["case_dirs"].values(),
        *authorization["measurement_dirs"].values(),
    ):
        _require(
            not _paths_overlap(output, Path(source).absolute()),
            "compatibility output overlaps a target-free input",
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        (staging / COMPATIBILITY_PREDICTION_ROOT).mkdir()
        (staging / COMPATIBILITY_MEASUREMENT_ROOT).mkdir()
        barrier_records = _barrier_record_by_case(authorization["barrier"])
        case_records = []
        for case_id in lock["selected_case_ids"]:
            case_records.append(
                _build_case_compatibility(
                    staging,
                    output,
                    case_id=case_id,
                    identity=_case_identity(lock, case_id),
                    lock_path=lock_snapshot.path,
                    h2_commit=h2_commit,
                    lock_binding=binding,
                    barrier_record=barrier_records[case_id],
                    case_root=authorization["case_dirs"][case_id],
                    nested_root=authorization["measurement_dirs"][case_id],
                    expected_h1=binding["implementation_commit_h1"],
                )
            )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": COMPATIBILITY_ARTIFACT_KIND,
            "protocol_id": PROTOCOL_ID,
            "status": "complete-target-free-outcome-compatibility",
            "role": "calibration",
            "lock_binding": dict(binding),
            "prediction_barrier": {
                "path": str(barrier_snapshot.path),
                "file_sha256": barrier_snapshot.sha256,
                "artifact_sha256": authorization["barrier"]["artifact_sha256"],
            },
            "case_count": 34,
            "exact_case_ids": list(lock["selected_case_ids"]),
            "cases": case_records,
            "prediction_root": str(output / COMPATIBILITY_PREDICTION_ROOT),
            "measurement_root": str(output / COMPATIBILITY_MEASUREMENT_ROOT),
            "information_boundary": dict(_COMPATIBILITY_BOUNDARY),
        }
        manifest["result_sha256"] = _result_sha256(manifest)
        _write_json(
            staging / COMPATIBILITY_MANIFEST_FILENAME,
            manifest,
        )
        _fsync_directory(staging / COMPATIBILITY_PREDICTION_ROOT)
        _fsync_directory(staging / COMPATIBILITY_MEASUREMENT_ROOT)
        _fsync_directory(staging)

        # Close both ordinary source mutation windows immediately before publish.
        _recheck_snapshot(lock_snapshot, label="H2 cohort lock")
        _recheck_snapshot(barrier_snapshot, label="complete prediction barrier")
        validate_confirmation_prediction_barrier(
            barrier_snapshot.path,
            lock_snapshot.path,
            h2_commit,
            authorization["case_dirs"],
            expected_h1=binding["implementation_commit_h1"],
        )
        for record in case_records:
            case_id = record["case"]
            current_case, _, current_cameras, current_diagnostic = _case_seal_inputs(
                case_id,
                authorization["case_dirs"][case_id],
                barrier_record=barrier_records[case_id],
            )
            current_measurement, current_archive, _ = _nested_eight_view_inputs(
                case_id,
                authorization["measurement_dirs"][case_id],
                identity=record["case_identity"],
                lock_path=lock_snapshot.path,
                h2_commit=h2_commit,
                lock_binding=binding,
                sealed_cameras=current_cameras,
                technical_disposition=current_diagnostic["technical_disposition"],
                expected_h1=binding["implementation_commit_h1"],
            )
            _require(
                current_case == record["case_seal"]
                and current_measurement == record["nested_measurement"]
                and current_archive.sha256
                == record["compatibility_measurement"]["source_file_sha256"],
                f"{case_id} target-free source changed before compatibility publication",
            )
        if output.exists() or output.is_symlink():
            raise ValueError("compatibility output appeared before publication")
        os.rename(staging, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_confirmation_outcome_compatibility(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        output,
        expected_h1=expected_h1,
    ).manifest


def _validate_compatibility_case(
    compatibility: Path,
    record: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    lock_binding: Mapping[str, Any],
    source_case: Mapping[str, Any],
    source_measurement: Mapping[str, Any],
    source_measurement_snapshot: _FileSnapshot,
    cameras: tuple[str, ...],
) -> dict[str, Any]:
    case_id = str(identity["case_id"])
    prediction_dir = _canonical_directory(
        compatibility / COMPATIBILITY_PREDICTION_ROOT / case_id,
        label=f"{case_id} compatibility prediction directory",
    )
    measurement_dir = _canonical_directory(
        compatibility / COMPATIBILITY_MEASUREMENT_ROOT / case_id,
        label=f"{case_id} compatibility measurement directory",
    )
    _require(
        set(path.name for path in prediction_dir.iterdir())
        == {
            EXTERNAL_PREDICTION_ARCHIVE_FILENAME,
            EXTERNAL_PREDICTION_REPORT_FILENAME,
            EXTERNAL_PREDICTION_SEAL_FILENAME,
        }
        and set(path.name for path in measurement_dir.iterdir())
        == {MEASUREMENT_ARCHIVE_FILENAME},
        f"{case_id} compatibility case contents changed",
    )
    measurement_snapshot = _snapshot_file(
        measurement_dir / MEASUREMENT_ARCHIVE_FILENAME,
        label="compatibility eight-view measurement",
    )
    _require(
        measurement_snapshot.payload == source_measurement_snapshot.payload,
        f"{case_id} compatibility measurement is not the exact eight-view archive",
    )
    prediction_snapshot = _snapshot_file(
        prediction_dir / EXTERNAL_PREDICTION_ARCHIVE_FILENAME,
        label="compatibility prediction archive",
    )
    arrays = _load_npz_snapshot(
        prediction_snapshot,
        expected_roles={"prediction_m", "selected_raw_backbone"},
        label="compatibility prediction archive",
    )
    source_archive = _snapshot_file(
        Path(str(source_case["case_seal_root"])) / ARRAY_ARCHIVE_FILENAME,
        label="source sealed prediction archive",
    )
    source_arrays = _load_npz_snapshot(
        source_archive,
        expected_roles={
            "physical_prior_m",
            "fixed_4_rbf_prediction_m",
            "fixed_8_rbf_prediction_m",
            "adaptive_prediction_m",
            "selected_raw_prediction_m",
            "persistence_m",
        },
        label="source sealed prediction archive",
    )
    _require(
        np.array_equal(arrays["prediction_m"], source_arrays["adaptive_prediction_m"])
        and np.array_equal(
            arrays["selected_raw_backbone"],
            source_arrays["selected_raw_prediction_m"],
        ),
        f"{case_id} compatibility prediction differs from its case seal",
    )
    report_snapshot = _snapshot_file(
        prediction_dir / EXTERNAL_PREDICTION_REPORT_FILENAME,
        label="compatibility prediction report",
    )
    report = _load_json_bytes(
        report_snapshot.payload,
        label="compatibility prediction report",
    )
    expected_report_inputs = {
        "measurement_archive": measurement_snapshot.sha256,
        "case_prediction_archive": source_case["prediction_archive_file_sha256"],
        "nested_measurement_manifest": source_measurement[
            "nested_measurement_manifest_file_sha256"
        ],
    }
    _require(
        set(report)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "case_identity",
            "lock_binding",
            "inputs_sha256",
            "selected_cameras",
            "information_boundary",
            "result_sha256",
        }
        and report.get("schema_version") == SCHEMA_VERSION
        and report.get("artifact_kind") == COMPATIBILITY_CASE_REPORT_KIND
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("case_identity") == identity
        and report.get("lock_binding") == lock_binding
        and report.get("result_sha256") == _result_sha256(report)
        and report.get("inputs_sha256") == expected_report_inputs
        and report.get("selected_cameras") == list(cameras)
        and report.get("information_boundary") == _COMPATIBILITY_BOUNDARY,
        f"{case_id} compatibility report changed",
    )
    seal_snapshot = _snapshot_file(
        prediction_dir / EXTERNAL_PREDICTION_SEAL_FILENAME,
        label="compatibility prediction seal",
    )
    seal = _load_json_bytes(
        seal_snapshot.payload,
        label="compatibility prediction seal",
    )
    external_record = _external_case_record(identity)
    _require(
        set(seal)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "protocol_config_sha256",
            *external_record,
            "lock_binding",
            "prediction_archive",
            "prediction_report",
            "source_case_seal",
            "source_nested_measurement",
            "selected_cameras",
            "information_boundary",
            "result_sha256",
        }
        and seal.get("schema_version") == SCHEMA_VERSION
        and seal.get("artifact_kind") == COMPATIBILITY_CASE_SEAL_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("protocol_config_sha256")
        == lock_binding["cohort_lock_artifact_sha256"]
        and all(seal.get(key) == value for key, value in external_record.items())
        and seal.get("lock_binding") == lock_binding
        and seal.get("result_sha256") == _result_sha256(seal)
        and seal.get("source_case_seal") == source_case
        and seal.get("source_nested_measurement") == source_measurement
        and seal.get("selected_cameras") == list(cameras)
        and seal.get("information_boundary") == _COMPATIBILITY_BOUNDARY,
        f"{case_id} compatibility seal changed",
    )
    archive_binding = seal.get("prediction_archive")
    report_binding = seal.get("prediction_report")
    _require(
        isinstance(archive_binding, Mapping)
        and archive_binding.get("path")
        == str(prediction_dir / EXTERNAL_PREDICTION_ARCHIVE_FILENAME)
        and archive_binding.get("file_sha256") == prediction_snapshot.sha256
        and archive_binding.get("prediction_array_sha256")
        == _external_array_sha256(arrays["prediction_m"])
        and archive_binding.get("baseline_array_sha256")
        == _external_array_sha256(arrays["selected_raw_backbone"])
        and isinstance(report_binding, Mapping)
        and report_binding.get("path")
        == str(prediction_dir / EXTERNAL_PREDICTION_REPORT_FILENAME)
        and report_binding.get("file_sha256") == report_snapshot.sha256
        and report_binding.get("result_sha256") == report["result_sha256"],
        f"{case_id} compatibility content binding changed",
    )
    expected = {
        **external_record,
        "case_identity": dict(identity),
        "case_seal": dict(source_case),
        "nested_measurement": dict(source_measurement),
        "selected_cameras": list(cameras),
        "compatibility_prediction": {
            "archive_file_sha256": prediction_snapshot.sha256,
            "archive_arrays": {
                role: _external_array_sha256(arrays[role])
                for role in ("prediction_m", "selected_raw_backbone")
            },
            "report_file_sha256": report_snapshot.sha256,
            "report_result_sha256": report["result_sha256"],
            "seal_file_sha256": seal_snapshot.sha256,
            "seal_result_sha256": seal["result_sha256"],
        },
        "compatibility_measurement": {
            "file_sha256": measurement_snapshot.sha256,
            "source_file_sha256": source_measurement_snapshot.sha256,
        },
    }
    _require(dict(record) == expected, f"{case_id} compatibility record changed")
    return seal


def validate_confirmation_outcome_compatibility(
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    compatibility_root: str | Path,
    *,
    expected_h1: str | None = None,
) -> ConfirmationOutcomeCompatibility:
    """Replay the barrier, sources, and every target-free compatibility file."""

    (
        lock,
        binding,
        _lock_snapshot,
        barrier_snapshot,
        authorization,
    ) = _load_authorization_sources(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        expected_h1=expected_h1,
    )
    root = _canonical_directory(
        compatibility_root,
        label="outcome compatibility root",
    )
    _require(
        set(path.name for path in root.iterdir())
        == {
            COMPATIBILITY_MANIFEST_FILENAME,
            COMPATIBILITY_PREDICTION_ROOT,
            COMPATIBILITY_MEASUREMENT_ROOT,
        },
        "outcome compatibility root contents changed",
    )
    manifest_snapshot = _snapshot_file(
        root / COMPATIBILITY_MANIFEST_FILENAME,
        label="outcome compatibility manifest",
    )
    manifest = _load_json_bytes(
        manifest_snapshot.payload,
        label="outcome compatibility manifest",
    )
    expected_cases = list(lock["selected_case_ids"])
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "status",
            "role",
            "lock_binding",
            "prediction_barrier",
            "case_count",
            "exact_case_ids",
            "cases",
            "prediction_root",
            "measurement_root",
            "information_boundary",
            "result_sha256",
        }
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_kind") == COMPATIBILITY_ARTIFACT_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("status") == "complete-target-free-outcome-compatibility"
        and manifest.get("role") == "calibration"
        and manifest.get("lock_binding") == binding
        and manifest.get("case_count") == 34
        and manifest.get("exact_case_ids") == expected_cases
        and manifest.get("prediction_root") == str(root / COMPATIBILITY_PREDICTION_ROOT)
        and manifest.get("measurement_root")
        == str(root / COMPATIBILITY_MEASUREMENT_ROOT)
        and manifest.get("information_boundary") == _COMPATIBILITY_BOUNDARY
        and manifest.get("result_sha256") == _result_sha256(manifest),
        "outcome compatibility manifest changed",
    )
    barrier_binding = manifest.get("prediction_barrier")
    _require(
        isinstance(barrier_binding, Mapping)
        and barrier_binding
        == {
            "path": str(barrier_snapshot.path),
            "file_sha256": barrier_snapshot.sha256,
            "artifact_sha256": authorization["barrier"]["artifact_sha256"],
        },
        "outcome compatibility binds another prediction barrier",
    )
    records = manifest.get("cases")
    _require(
        isinstance(records, list)
        and [record.get("case") for record in records] == expected_cases,
        "outcome compatibility case order changed",
    )
    barrier_records = _barrier_record_by_case(authorization["barrier"])
    for case_id, record in zip(expected_cases, records, strict=True):
        identity = _case_identity(lock, case_id)
        source_case, _, cameras, diagnostic = _case_seal_inputs(
            case_id,
            authorization["case_dirs"][case_id],
            barrier_record=barrier_records[case_id],
        )
        source_measurement, source_snapshot, _ = _nested_eight_view_inputs(
            case_id,
            authorization["measurement_dirs"][case_id],
            identity=identity,
            lock_path=lock_path,
            h2_commit=h2_commit,
            lock_binding=binding,
            sealed_cameras=cameras,
            technical_disposition=diagnostic["technical_disposition"],
            expected_h1=binding["implementation_commit_h1"],
        )
        _validate_compatibility_case(
            root,
            record,
            identity=identity,
            lock_binding=binding,
            source_case=source_case,
            source_measurement=source_measurement,
            source_measurement_snapshot=source_snapshot,
            cameras=cameras,
        )
    return ConfirmationOutcomeCompatibility(
        root=root,
        manifest_path=manifest_snapshot.path,
        prediction_root=root / COMPATIBILITY_PREDICTION_ROOT,
        measurement_root=root / COMPATIBILITY_MEASUREMENT_ROOT,
        manifest=manifest,
    )


def validate_confirmation_outcome_execution_repository(
    repository: str | Path,
) -> dict[str, Any]:
    """Validate the exact clean 29091 checkout and both frozen outcome stages."""

    result = validate_external_execution_repository(repository)
    root = Path(repository).absolute()
    observed: dict[str, str] = {}
    for relative, expected in EXTERNAL_OUTCOME_STAGE_SHA256.items():
        source = _snapshot_file(
            root / relative,
            label=f"external outcome stage {relative}",
        )
        _require(
            source.sha256 == expected,
            f"external outcome stage changed: {relative}",
        )
        observed[relative] = source.sha256
    return {**result, "outcome_stage_sha256": observed}


def _load_external_stage(
    execution_repository: Path,
    stage: str,
) -> ModuleType:
    _require(stage in EXTERNAL_OUTCOME_STAGE_SCRIPTS, "unknown outcome stage")
    script = (
        execution_repository
        / "scripts"
        / "remote"
        / EXTERNAL_OUTCOME_STAGE_SCRIPTS[stage]
    )
    name = _external_stage_module_name(stage)
    specification = importlib.util.spec_from_file_location(name, script)
    _require(
        specification is not None and specification.loader is not None,
        "cannot load frozen outcome stage",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _external_stage_module_name(stage: str) -> str:
    return f"_adaptive_confirmation_{stage.replace('-', '_')}"


def _case_record_from_object(
    lock: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    identities = [
        _case_identity(lock, case_id) for case_id in lock["selected_case_ids"]
    ]
    matches = [
        identity
        for identity in identities
        if identity["object_id"] == object_id
        and int(identity["episode_id"]) == int(episode_id)
    ]
    _require(len(matches) == 1, "authorized case is outside the exact H2 lock")
    return _external_case_record(matches[0])


def make_confirmation_outcome_authorizer(
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    compatibility_root: str | Path,
    *,
    expected_h1: str | None = None,
):
    """Return the process-local authorization hook used by both frozen stages."""

    def authorize(
        cohort_seal: Mapping[str, Any],
        *,
        protocol_path: str | Path,
        role: str,
        artifact_root: str | Path,
        object_id: str,
        episode_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        binding = validate_confirmation_outcome_compatibility(
            adapter_repository,
            lock_path,
            h2_commit,
            barrier_path,
            case_seal_dirs,
            nested_measurement_dirs,
            compatibility_root,
            expected_h1=expected_h1,
        )
        _require(
            Path(protocol_path).absolute().resolve(strict=True)
            == Path(lock_path).absolute().resolve(strict=True)
            and role == "calibration"
            and Path(artifact_root).absolute().resolve(strict=True)
            == binding.prediction_root.resolve(strict=True)
            and dict(cohort_seal) == dict(binding.manifest),
            "frozen outcome stage used another H2 authorization binding",
        )
        lock = load_confirmation_cohort_lock(
            lock_path,
            expected_implementation_commit_h1=expected_h1,
        )
        record = _case_record_from_object(
            lock,
            object_id=object_id,
            episode_id=episode_id,
        )
        prediction_dir = binding.prediction_root / str(record["case"])
        seal_snapshot = _snapshot_file(
            prediction_dir / EXTERNAL_PREDICTION_SEAL_FILENAME,
            label="authorized compatibility prediction seal",
        )
        seal = _load_json_bytes(
            seal_snapshot.payload,
            label="authorized compatibility prediction seal",
        )
        rows = [
            row for row in binding.manifest["cases"] if row["case"] == record["case"]
        ]
        _require(
            len(rows) == 1
            and rows[0]["compatibility_prediction"]["seal_file_sha256"]
            == seal_snapshot.sha256
            and rows[0]["compatibility_prediction"]["seal_result_sha256"]
            == seal["result_sha256"],
            "authorized compatibility prediction differs from the complete binding",
        )
        return record, seal

    return authorize


def patch_confirmation_outcome_stage_module(
    module: ModuleType,
    *,
    stage: str,
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    compatibility_root: str | Path,
    expected_h1: str | None = None,
) -> None:
    """Patch only protocol identity and complete-barrier authorization aliases."""

    _require(stage in EXTERNAL_OUTCOME_STAGE_SCRIPTS, "unknown outcome stage")
    required = (
        "PROTOCOL_ID",
        "load_bias_aware_prospective_protocol",
        "authorize_prospective_outcome_case",
    )
    _require(
        all(hasattr(module, name) for name in required),
        "frozen outcome stage interface changed",
    )
    module.PROTOCOL_ID = PROTOCOL_ID
    module.load_bias_aware_prospective_protocol = load_confirmation_execution_protocol
    module.authorize_prospective_outcome_case = make_confirmation_outcome_authorizer(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        compatibility_root,
        expected_h1=expected_h1,
    )


def _remove_bound_option(
    arguments: list[str],
    option: str,
    expected: str,
) -> list[str]:
    result: list[str] = []
    observed: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == option:
            _require(index + 1 < len(arguments), f"{option} has no value")
            observed.append(arguments[index + 1])
            index += 2
            continue
        if value.startswith(f"{option}="):
            observed.append(value.split("=", 1)[1])
            index += 1
            continue
        result.append(value)
        index += 1
    _require(len(observed) <= 1, f"{option} is duplicated")
    if observed:
        if option in {
            "--repo",
            "--protocol",
            "--cohort-seal",
            "--prediction-root",
            "--measurement-root",
            "--deform360-repo",
            "--staged-case-dir",
            "--source-aligned-root",
        }:
            _require(
                Path(observed[0]).absolute().resolve(strict=True)
                == Path(expected).absolute().resolve(strict=True),
                f"{option} differs from the authorization-bound path",
            )
        else:
            _require(observed[0] == expected, f"{option} changed")
    return result


def _required_option_value(arguments: Sequence[str], option: str) -> str:
    observed: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == option:
            _require(index + 1 < len(arguments), f"{option} has no value")
            observed.append(arguments[index + 1])
            index += 2
            continue
        if value.startswith(f"{option}="):
            observed.append(value.split("=", 1)[1])
        index += 1
    _require(
        len(observed) == 1 and bool(observed[0]), f"{option} must occur exactly once"
    )
    _require(
        not observed[0].startswith("--"),
        f"{option} has no value",
    )
    return observed[0]


def _reject_reserved_option_abbreviations(
    arguments: Sequence[str],
    reserved: set[str],
) -> None:
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        _require(
            not any(
                candidate.startswith(option) and candidate != option
                for candidate in reserved
            ),
            f"reserved option abbreviation is forbidden: {option}",
        )


def run_confirmation_outcome_stage(
    stage: str,
    stage_arguments: Sequence[str],
    *,
    adapter_repository: str | Path,
    execution_repository: str | Path,
    deform360_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    compatibility_root: str | Path,
    expected_h1: str | None = None,
) -> int:
    """Run one exact frozen outcome stage under the complete H2 authorization."""

    _require(stage in EXTERNAL_OUTCOME_STAGE_SCRIPTS, "unknown outcome stage")
    compatibility = validate_confirmation_outcome_compatibility(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        compatibility_root,
        expected_h1=expected_h1,
    )
    execution = _canonical_directory(
        execution_repository,
        label="external outcome execution repository",
    )
    deform360 = _canonical_directory(
        deform360_repository,
        label="Deform360 execution repository",
    )
    validate_confirmation_outcome_execution_repository(execution)
    validate_deform360_execution_repository(deform360)
    arguments = list(stage_arguments)
    bound: list[tuple[str, str]] = [
        ("--repo", str(execution)),
        ("--protocol", str(Path(lock_path).absolute())),
        ("--role", "calibration"),
        ("--cohort-seal", str(compatibility.manifest_path)),
        ("--prediction-root", str(compatibility.prediction_root)),
    ]
    custody_replay: (
        tuple[
            _FileSnapshot,
            str,
            Path,
            Path,
        ]
        | None
    ) = None
    if stage == "authorized-future":
        object_id = _required_option_value(arguments, "--object-id")
        episode_text = _required_option_value(arguments, "--episode-id")
        _require(
            episode_text.isdecimal() and str(int(episode_text)) == episode_text,
            "--episode-id must be a canonical nonnegative integer",
        )
        episode_id = int(episode_text)
        matching_cases = [
            record
            for record in compatibility.manifest["cases"]
            if record.get("object_id") == object_id
            and record.get("episode_id") == episode_id
        ]
        _require(
            len(matching_cases) == 1,
            "authorized-future case is outside the exact H2 compatibility cohort",
        )
        compatibility_case = matching_cases[0]
        case_id = str(compatibility_case["case"])
        custody_record = compatibility_case["nested_measurement"][
            "source_stage_lineage"
        ]["source_custody_seal"]
        custody_snapshot = _snapshot_file(
            custody_record["path"],
            label=f"{case_id} source-custody seal",
        )
        _require(
            custody_snapshot.sha256 == custody_record["file_sha256"],
            f"{case_id} source-custody seal changed before future authorization",
        )
        custody = validate_confirmation_source_custody_envelope(
            custody_snapshot.path,
            lock_path,
            h2_commit,
            case_id,
            expected_h1=expected_h1,
        )
        source_episode = _canonical_directory(
            custody["path_binding"]["source_episode_dir"],
            label=f"{case_id} source episode",
        )
        staged_case = _canonical_directory(
            custody["path_binding"]["staged_case_dir"],
            label=f"{case_id} staged prediction case",
        )
        source_aligned_root = _canonical_directory(
            source_episode.parent.parent,
            label="source aligned root",
        )
        _require(
            source_episode
            == source_aligned_root / object_id / f"episode_{episode_id:04d}"
            and staged_case.name == case_id,
            "source-custody path differs from the authorized-future case",
        )
        replayed_custody = validate_confirmation_source_custody_seal(
            custody_snapshot.path,
            lock_path,
            h2_commit,
            case_id,
            source_episode,
            staged_case,
            expected_h1=expected_h1,
        )
        _require(
            replayed_custody == custody
            and custody["artifact_sha256"] == custody_record["artifact_sha256"],
            f"{case_id} source-custody replay changed before future opening",
        )
        custody_replay = (
            custody_snapshot,
            case_id,
            source_episode,
            staged_case,
        )
        bound.append(("--measurement-root", str(compatibility.measurement_root)))
        bound.append(("--staged-case-dir", str(staged_case)))
        bound.append(("--source-aligned-root", str(source_aligned_root)))
    else:
        bound.append(("--deform360-repo", str(deform360)))
    _reject_reserved_option_abbreviations(
        arguments,
        {option for option, _ in bound} | {"--calibration-gate"},
    )
    for option, expected in bound:
        arguments = _remove_bound_option(arguments, option, expected)
    _require(
        not any(
            value == "--calibration-gate" or value.startswith("--calibration-gate=")
            for value in arguments
        ),
        "H2 outcome authorization must not consume the old target gate",
    )
    previous_sys_path = list(sys.path)
    stage_module_name = _external_stage_module_name(stage)
    try:
        sys.path.insert(0, str(deform360))
        with activate_confirmation_external_runtime(execution):
            module = _load_external_stage(execution, stage)
            validate_external_module_provenance(execution)
            patch_confirmation_outcome_stage_module(
                module,
                stage=stage,
                adapter_repository=adapter_repository,
                lock_path=lock_path,
                h2_commit=h2_commit,
                barrier_path=barrier_path,
                case_seal_dirs=case_seal_dirs,
                nested_measurement_dirs=nested_measurement_dirs,
                compatibility_root=compatibility.root,
                expected_h1=expected_h1,
            )
            script = (
                execution / "scripts" / "remote" / EXTERNAL_OUTCOME_STAGE_SCRIPTS[stage]
            )
            previous_argv = sys.argv
            sys.argv = [
                str(script),
                *(item for option, expected in bound for item in (option, expected)),
                *arguments,
            ]
            try:
                status = int(module.main())
            finally:
                sys.argv = previous_argv
            if custody_replay is not None:
                custody_snapshot, case_id, source_episode, staged_case = custody_replay
                _recheck_snapshot(
                    custody_snapshot,
                    label=f"{case_id} source-custody seal",
                )
                validate_confirmation_source_custody_seal(
                    custody_snapshot.path,
                    lock_path,
                    h2_commit,
                    case_id,
                    source_episode,
                    staged_case,
                    expected_h1=expected_h1,
                )
            return status
    finally:
        sys.path[:] = previous_sys_path
        sys.modules.pop(stage_module_name, None)


def _compatibility_case(
    binding: ConfirmationOutcomeCompatibility,
    case_id: str,
) -> Mapping[str, Any]:
    rows = [row for row in binding.manifest["cases"] if row["case"] == case_id]
    _require(len(rows) == 1, "case is absent from outcome compatibility")
    return rows[0]


def validate_confirmation_native_official_target(
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    compatibility_root: str | Path,
    case_id: str,
    authorized_future_case_dir: str | Path,
    authorized_outcome_case_dir: str | Path,
    *,
    expected_h1: str | None = None,
) -> ConfirmationNativeOfficialTarget:
    """Validate one frozen-stage target and return arrays plus scoring evidence."""

    binding = validate_confirmation_outcome_compatibility(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        compatibility_root,
        expected_h1=expected_h1,
    )
    compatibility_case = _compatibility_case(binding, case_id)
    lock = load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=expected_h1,
    )
    lock_snapshot = _snapshot_file(lock_path, label="H2 cohort lock")
    compatibility_manifest_snapshot = _snapshot_file(
        binding.manifest_path,
        label="outcome compatibility manifest",
    )
    _require(
        _load_json_bytes(lock_snapshot.payload, label="H2 cohort lock") == lock
        and _load_json_bytes(
            compatibility_manifest_snapshot.payload,
            label="outcome compatibility manifest",
        )
        == binding.manifest,
        "outcome compatibility changed after validation",
    )
    identity = _case_identity(lock, case_id)
    external_record = _external_case_record(identity)
    future_root = _canonical_directory(
        authorized_future_case_dir,
        label="authorized future case",
    )
    outcome_root = _canonical_directory(
        authorized_outcome_case_dir,
        label="authorized outcome case",
    )
    _require(
        future_root.name == case_id and outcome_root.name == case_id,
        "authorized outcome case directory name changed",
    )
    _require(
        not _paths_overlap(future_root, outcome_root)
        and not _paths_overlap(future_root, binding.root)
        and not _paths_overlap(outcome_root, binding.root),
        "authorized future/outcome roots overlap sealed prediction inputs",
    )
    future_snapshot = _snapshot_file(
        future_root / EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME,
        label="authorized future manifest",
    )
    future = _load_json_bytes(
        future_snapshot.payload,
        label="authorized future manifest",
    )
    future_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "protocol_config_sha256",
        *external_record,
        "code_revision",
        "raw_frame_range_half_open",
        "frame_count",
        "selected_cameras",
        "camera_records",
        "inputs_sha256",
        "outputs_sha256",
        "authorization",
        "information_boundary",
        "result_sha256",
    }
    _require(
        set(future) == future_keys
        and future.get("schema_version") == 1
        and future.get("artifact_kind") == EXTERNAL_AUTHORIZED_FUTURE_KIND
        and future.get("protocol_id") == PROTOCOL_ID
        and future.get("protocol_config_sha256") == lock["artifact_sha256"]
        and all(future.get(key) == value for key, value in external_record.items())
        and future.get("code_revision") == EXTERNAL_EXECUTION_COMMIT
        and future.get("result_sha256") == _result_sha256(future)
        and future.get("selected_cameras") == compatibility_case["selected_cameras"]
        and future.get("information_boundary") == _AUTHORIZED_FUTURE_BOUNDARY,
        "authorized future differs from the H2 compatibility binding",
    )
    raw_frame_range = future.get("raw_frame_range_half_open")
    _require(
        isinstance(raw_frame_range, list)
        and len(raw_frame_range) == 2
        and all(type(value) is int and value >= 0 for value in raw_frame_range)
        and raw_frame_range[1] - raw_frame_range[0] == 81
        and future.get("frame_count") == 81,
        "authorized future frame window changed",
    )
    future_inputs = future.get("inputs_sha256")
    _require(
        isinstance(future_inputs, Mapping)
        and set(future_inputs)
        == {
            "protocol",
            "prediction_cohort_seal",
            "prediction_seal",
            "prediction_archive",
            "prediction_prefix_manifest",
            "source_preparation_manifest",
            "frame_zero_reconstruction_manifest",
            "source_robot",
            "measurement_archive",
            "generic_selector_source",
            "sam2_checkpoint",
            "calibration_gate",
        }
        and future_inputs.get("protocol") == lock_snapshot.sha256
        and future_inputs.get("prediction_cohort_seal")
        == compatibility_manifest_snapshot.sha256
        and future_inputs.get("prediction_seal")
        == compatibility_case["compatibility_prediction"]["seal_file_sha256"]
        and future_inputs.get("prediction_archive")
        == compatibility_case["compatibility_prediction"]["archive_file_sha256"]
        and future_inputs.get("measurement_archive")
        == compatibility_case["compatibility_measurement"]["file_sha256"]
        and future_inputs.get("calibration_gate") is None
        and all(
            _is_sha256(future_inputs.get(key))
            for key in (
                "prediction_prefix_manifest",
                "source_preparation_manifest",
                "frame_zero_reconstruction_manifest",
                "source_robot",
                "generic_selector_source",
                "sam2_checkpoint",
            )
        ),
        "authorized future input binding changed",
    )
    future_outputs = future.get("outputs_sha256")
    _require(
        isinstance(future_outputs, Mapping)
        and set(future_outputs)
        == {"robot", "frame_zero_splat", "intrinsics", "extrinsics"}
        and all(_is_sha256(value) for value in future_outputs.values()),
        "authorized future output binding changed",
    )
    retained_source = compatibility_case["nested_measurement"].get(
        "retained_failure_source"
    )
    identity_persistence = (
        retained_source.get("identity_persistence_adapter")
        if isinstance(retained_source, Mapping)
        else None
    )
    source_stage_lineage = compatibility_case["nested_measurement"].get(
        "source_stage_lineage"
    )
    _require(
        isinstance(source_stage_lineage, Mapping)
        and future_inputs.get("prediction_prefix_manifest")
        == source_stage_lineage["prediction_prefix_manifest"]["file_sha256"]
        and future_inputs.get("frame_zero_reconstruction_manifest")
        == source_stage_lineage["frame_zero_manifest"]["file_sha256"]
        and future_inputs.get("source_preparation_manifest")
        == source_stage_lineage["source_preparation_manifest_file_sha256"],
        "authorized future does not bind the sealed source-stage lineage",
    )
    custody_record = source_stage_lineage.get("source_custody_seal")
    _require(
        isinstance(custody_record, Mapping)
        and set(custody_record) == _SOURCE_CUSTODY_RECORD_KEYS,
        "authorized future source-custody record changed",
    )
    custody_snapshot = _snapshot_file(
        custody_record["path"],
        label=f"{case_id} source-custody seal",
    )
    _require(
        custody_snapshot.sha256 == custody_record["file_sha256"],
        "authorized future source-custody seal changed",
    )
    custody_envelope = validate_confirmation_source_custody_envelope(
        custody_snapshot.path,
        lock_path,
        h2_commit,
        case_id,
        expected_h1=expected_h1,
    )
    source_episode = Path(
        custody_envelope["path_binding"]["source_episode_dir"],
    )
    staged_case = Path(custody_envelope["path_binding"]["staged_case_dir"])
    custody = validate_confirmation_source_custody_seal(
        custody_snapshot.path,
        lock_path,
        h2_commit,
        case_id,
        source_episode,
        staged_case,
        expected_h1=expected_h1,
    )
    selected_cameras = compatibility_case["selected_cameras"]
    _require(
        custody == custody_envelope
        and custody.get("artifact_sha256") == custody_record["artifact_sha256"]
        and set(selected_cameras) <= set(custody["camera_panel"])
        and future_inputs.get("source_preparation_manifest")
        == custody["manifests"]["source_preparation"]["file_sha256"]
        and future_inputs.get("source_robot")
        == _custody_inventory_file_sha256(
            custody,
            "aligned_source_episode",
            "robot/robot.npz",
        )
        and future_inputs.get("generic_selector_source")
        == EXTERNAL_GENERIC_SELECTOR_SHA256
        and future_inputs.get("sam2_checkpoint") == EXTERNAL_SAM2_CHECKPOINT_SHA256
        and future_outputs.get("frame_zero_splat")
        == custody["frame_zero_custody"]["splat_file_sha256"],
        "authorized future differs from sealed source custody",
    )
    if isinstance(retained_source, Mapping):
        _require(
            future_inputs.get("prediction_prefix_manifest")
            == retained_source["prediction_prefix_manifest"]["file_sha256"],
            "authorized future does not bind the retained prediction prefix",
        )
    if identity_persistence is not None:
        _require(
            future_inputs.get("frame_zero_reconstruction_manifest")
            == identity_persistence["frame_zero_manifest_file_sha256"]
            and future_outputs.get("frame_zero_splat")
            == identity_persistence["frame_zero_splat_file_sha256"],
            "authorized future does not bind the identity-persistence adapter",
        )
    camera_records = future.get("camera_records")
    _require(
        isinstance(camera_records, list)
        and len(camera_records) == 8
        and [row.get("camera") for row in camera_records if isinstance(row, Mapping)]
        == compatibility_case["selected_cameras"]
        and all(
            isinstance(row, Mapping)
            and set(row)
            == {
                "camera",
                "video_sha256",
                "decoded_sealed_prefix_sha256",
                "timestamps_sha256",
                "masks_sha256",
                "sam2_diagnostics",
            }
            and all(
                _is_sha256(row.get(key))
                for key in (
                    "video_sha256",
                    "decoded_sealed_prefix_sha256",
                    "timestamps_sha256",
                    "masks_sha256",
                )
            )
            for row in camera_records
        ),
        "authorized future camera provenance changed",
    )
    camera_records_by_name = {
        str(record["camera"]): record for record in camera_records
    }
    _require(
        all(
            custody["camera_custody"][camera]["source_prefix_frame_range_half_open"][0]
            == raw_frame_range[0]
            and custody["camera_custody"][camera][
                "source_prefix_frame_range_half_open"
            ][1]
            == raw_frame_range[0] + SOURCE_CUSTODY_PREFIX_FRAME_COUNT
            and custody["raw_rgb24_prefix"]["by_camera"][camera]
            == camera_records_by_name[camera]["decoded_sealed_prefix_sha256"]
            for camera in selected_cameras
        ),
        "authorized future RGB prefix differs from sealed source custody",
    )
    future_file_snapshots = _validate_authorized_future_files(
        future_root,
        future,
        compatibility_case["selected_cameras"],
    )
    expected_authorization = {
        "prediction_cohort_result_sha256": binding.manifest["result_sha256"],
        "prediction_result_sha256": compatibility_case["compatibility_prediction"][
            "seal_result_sha256"
        ],
        "calibration_gate_result_sha256": None,
        "prediction_cohort_verified_before_future_read": True,
        "target_access_gate_verified": False,
    }
    _require(
        future.get("authorization") == expected_authorization,
        "future was opened under another authorization",
    )
    outcome_snapshot = _snapshot_file(
        outcome_root / EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME,
        label="authorized outcome manifest",
    )
    outcome = _load_json_bytes(
        outcome_snapshot.payload,
        label="authorized outcome manifest",
    )
    outcome_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "protocol_config_sha256",
        *external_record,
        "code_revision",
        "deform360_revision",
        "cameras",
        "raw_frame_count",
        "target_frame_count",
        "material_point_count",
        "material_identity_sha256",
        "reconstruction",
        "inputs_sha256",
        "stage_metadata_sha256",
        "output",
        "authorization",
        "information_boundary",
        "result_sha256",
    }
    _require(
        set(outcome) == outcome_keys
        and outcome.get("schema_version") == 1
        and outcome.get("artifact_kind") == EXTERNAL_AUTHORIZED_OUTCOME_KIND
        and outcome.get("protocol_id") == PROTOCOL_ID
        and outcome.get("protocol_config_sha256") == lock["artifact_sha256"]
        and all(outcome.get(key) == value for key, value in external_record.items())
        and outcome.get("code_revision") == EXTERNAL_EXECUTION_COMMIT
        and outcome.get("deform360_revision") == DEFORM360_EXECUTION_COMMIT
        and outcome.get("result_sha256") == _result_sha256(outcome)
        and outcome.get("cameras") == compatibility_case["selected_cameras"]
        and outcome.get("raw_frame_count") == 81
        and outcome.get("target_frame_count") == 76
        and outcome.get("reconstruction")
        == {
            "minimum_visual_hull_points": 512,
            "voxel_resolution": 120,
            "cube_half_extent_m": 0.5,
            "first_frame_iterations": 500,
            "warm_start_iterations": 250,
            "sealed_frame_zero_splat_reused": True,
        }
        and outcome.get("information_boundary") == _AUTHORIZED_OUTCOME_BOUNDARY,
        "authorized outcome differs from the H2 compatibility binding",
    )
    _require(
        outcome.get("authorization")
        == {
            "prediction_cohort_result_sha256": binding.manifest["result_sha256"],
            "prediction_result_sha256": compatibility_case["compatibility_prediction"][
                "seal_result_sha256"
            ],
            "calibration_gate_result_sha256": None,
        }
        and outcome.get("inputs_sha256", {}).get("authorized_future_manifest")
        == future_snapshot.sha256,
        "authorized outcome used another future or prediction barrier",
    )
    outcome_inputs = outcome.get("inputs_sha256")
    _require(
        isinstance(outcome_inputs, Mapping)
        and set(outcome_inputs)
        == {
            "protocol",
            "prediction_cohort_seal",
            "prediction_seal",
            "prediction_archive",
            "authorized_future_manifest",
            "tracking_checkpoint",
            "cotracker_predictor",
            "calibration_gate",
            "reconstruct_stage",
            "urdf_render",
            "depth_stage",
            "tracking_stage",
            "pcd_stage",
        }
        and outcome_inputs.get("protocol") == lock_snapshot.sha256
        and outcome_inputs.get("prediction_cohort_seal")
        == compatibility_manifest_snapshot.sha256
        and outcome_inputs.get("prediction_seal")
        == compatibility_case["compatibility_prediction"]["seal_file_sha256"]
        and outcome_inputs.get("prediction_archive")
        == compatibility_case["compatibility_prediction"]["archive_file_sha256"]
        and outcome_inputs.get("authorized_future_manifest") == future_snapshot.sha256
        and outcome_inputs.get("calibration_gate") is None
        and all(
            _is_sha256(outcome_inputs.get(key))
            for key in (
                "tracking_checkpoint",
                "cotracker_predictor",
                "reconstruct_stage",
                "urdf_render",
                "depth_stage",
                "tracking_stage",
                "pcd_stage",
            )
        ),
        "authorized outcome input binding changed",
    )
    stage_metadata = outcome.get("stage_metadata_sha256")
    _require(
        isinstance(stage_metadata, Mapping)
        and set(stage_metadata)
        == {
            "reconstruction",
            "gripper_masks",
            "depth",
            "tracking",
            "point_cloud",
        }
        and _is_sha256(stage_metadata.get("reconstruction"))
        and _is_sha256(stage_metadata.get("point_cloud"))
        and all(
            isinstance(stage_metadata.get(role), Mapping)
            and set(stage_metadata[role]) == set(compatibility_case["selected_cameras"])
            and all(_is_sha256(value) for value in stage_metadata[role].values())
            for role in ("gripper_masks", "depth", "tracking")
        ),
        "authorized outcome stage provenance changed",
    )
    outcome_metadata_snapshots = _validate_authorized_outcome_stage_metadata(
        future_root,
        stage_metadata,
        compatibility_case["selected_cameras"],
    )
    archive_snapshot = _snapshot_file(
        outcome_root / EXTERNAL_TARGET_ARCHIVE_FILENAME,
        label="native official target archive",
    )
    output = outcome.get("output")
    _require(
        isinstance(output, Mapping)
        and set(output)
        == {
            "target_archive",
            "target_archive_sha256",
            "target_array_sha256",
            "frame_zero_bit_exact_to_sealed_baseline",
        }
        and output.get("target_archive")
        == str(outcome_root / EXTERNAL_TARGET_ARCHIVE_FILENAME)
        and output.get("target_archive_sha256") == archive_snapshot.sha256
        and type(output.get("frame_zero_bit_exact_to_sealed_baseline")) is bool,
        "native official target archive binding changed",
    )
    arrays = _load_npz_snapshot(
        archive_snapshot,
        expected_roles={"target_m", "target_visibility", "target_validity"},
        label="native official target archive",
    )
    target = arrays["target_m"]
    visibility = arrays["target_visibility"]
    validity = arrays["target_validity"]
    _require(
        target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3
        and target.dtype == np.dtype(np.float32)
        and np.all(np.isfinite(target))
        and visibility.shape == target.shape[:2]
        and visibility.dtype == np.dtype(bool)
        and validity.shape == target.shape[:2]
        and validity.dtype == np.dtype(bool),
        "native official target arrays are invalid",
    )
    _require(
        outcome.get("material_point_count") == target.shape[1]
        and outcome.get("material_identity_sha256") == _external_array_sha256(target[0])
        and output.get("target_array_sha256") == _external_array_sha256(target)
        and np.all(visibility)
        and np.all(validity),
        "native official target array checksum changed",
    )
    prediction_snapshot = _snapshot_file(
        binding.prediction_root / case_id / EXTERNAL_PREDICTION_ARCHIVE_FILENAME,
        label="compatibility baseline archive",
    )
    prediction = _load_npz_snapshot(
        prediction_snapshot,
        expected_roles={"prediction_m", "selected_raw_backbone"},
        label="compatibility baseline archive",
    )
    sealed_baseline = prediction["selected_raw_backbone"]
    frame_zero_is_exact = target.shape == sealed_baseline.shape and np.array_equal(
        target[0], sealed_baseline[0]
    )
    _require(
        target.shape[1] >= sealed_baseline.shape[1]
        and output["frame_zero_bit_exact_to_sealed_baseline"] is frame_zero_is_exact,
        "native official target frame-zero identity declaration changed",
    )
    if identity_persistence is not None:
        _require(
            frame_zero_is_exact
            and identity_persistence["adapted_material"]["point_count"]
            == target.shape[1]
            and identity_persistence["adapted_material"]["array_sha256"]
            == _external_array_sha256(target[0]),
            "official target differs from the adapted Splat material identity",
        )
    result_arrays = {
        "target_m": np.array(target, copy=True),
        "target_visibility": np.array(visibility, copy=True),
        "target_validity": np.array(validity, copy=True),
    }
    for value in result_arrays.values():
        value.setflags(write=False)
    for label, snapshot in (
        ("source-custody seal", custody_snapshot),
        *(
            (f"authorized future {role}", snapshot)
            for role, snapshot in future_file_snapshots.items()
        ),
        *(
            ("authorized outcome stage metadata", snapshot)
            for snapshot in outcome_metadata_snapshots
        ),
    ):
        _recheck_snapshot(snapshot, label=label)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "case_identity": dict(identity),
        "lock_binding": dict(binding.manifest["lock_binding"]),
        "prediction_barrier": dict(binding.manifest["prediction_barrier"]),
        "case_seal": dict(compatibility_case["case_seal"]),
        "nested_measurement": dict(compatibility_case["nested_measurement"]),
        "selected_cameras": list(compatibility_case["selected_cameras"]),
        "identity_persistence_adapter": identity_persistence,
        "compatibility_manifest": {
            "path": str(compatibility_manifest_snapshot.path),
            "file_sha256": compatibility_manifest_snapshot.sha256,
            "result_sha256": binding.manifest["result_sha256"],
        },
        "authorized_future_manifest": {
            "path": str(future_snapshot.path),
            "file_sha256": future_snapshot.sha256,
            "result_sha256": future["result_sha256"],
        },
        "authorized_outcome_manifest": {
            "path": str(outcome_snapshot.path),
            "file_sha256": outcome_snapshot.sha256,
            "result_sha256": outcome["result_sha256"],
        },
        "target_archive": {
            "path": str(archive_snapshot.path),
            "file_sha256": archive_snapshot.sha256,
            "arrays": {
                role: _external_array_sha256(array)
                for role, array in result_arrays.items()
            },
        },
        "information_boundary": {
            "native_official_arrays_returned": True,
            "metric_or_score_computed": False,
        },
    }
    return ConfirmationNativeOfficialTarget(
        target_m=result_arrays["target_m"],
        target_visibility=result_arrays["target_visibility"],
        target_validity=result_arrays["target_validity"],
        evidence=evidence,
    )


def load_confirmation_native_official_target(
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    compatibility_root: str | Path,
    case_id: str,
    authorized_future_case_dir: str | Path,
    authorized_outcome_case_dir: str | Path,
    *,
    expected_h1: str | None = None,
) -> dict[str, np.ndarray]:
    """Return native official target arrays without computing any metric."""

    target = validate_confirmation_native_official_target(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        compatibility_root,
        case_id,
        authorized_future_case_dir,
        authorized_outcome_case_dir,
        expected_h1=expected_h1,
    )
    return {
        "target_m": target.target_m,
        "target_visibility": target.target_visibility,
        "target_validity": target.target_validity,
    }


__all__ = [
    "COMPATIBILITY_ARTIFACT_KIND",
    "COMPATIBILITY_MANIFEST_FILENAME",
    "ConfirmationNativeOfficialTarget",
    "ConfirmationOutcomeCompatibility",
    "EXTERNAL_OUTCOME_STAGE_SCRIPTS",
    "EXTERNAL_OUTCOME_STAGE_SHA256",
    "build_confirmation_outcome_compatibility",
    "load_confirmation_native_official_target",
    "make_confirmation_outcome_authorizer",
    "patch_confirmation_outcome_stage_module",
    "run_confirmation_outcome_stage",
    "validate_confirmation_native_official_target",
    "validate_confirmation_outcome_compatibility",
    "validate_confirmation_outcome_execution_repository",
]
