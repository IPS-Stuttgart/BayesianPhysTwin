"""Target-free nested four/eight-view measurements for fresh confirmation.

The builder reads a sealed frame-zero physical archive, frame-zero masks/depth,
calibration, and exact causal RGB prefixes.  It never accepts a target,
outcome, score, or future-geometry path.  The four-view plan is the exact
legacy exhaustive optimum and is an ordered prefix of the eight-view plan.

All eight views are eventually tracked so the same sealed case can provide a
fixed-eight shadow comparator.  Crucially, the four-view reliability decision
is materialized before the extra four shadow streams are decoded.  Reported
adaptive camera demand is therefore a causal offline-policy charge, not the
wall-clock acquisition or compute of this shadow-evaluation run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    load_confirmation_cohort_lock,
)
from .deform360_adaptive_covariance_confirmation_seal import (
    array_sha256 as confirmation_array_sha256,
)
from .deform360_adaptive_covariance_confirmation_source_custody import (
    validate_confirmation_source_custody_envelope,
)
from .deform360_adaptive_covariance_rbf import (
    FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
    normalized_covariance_dispersion,
)
from .deform360_held_online_prefix import FRAME_COUNT, UPDATE_FRAMES
from .deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    ALLTRACKER_SOURCE_TREE,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    _array_sha256,
    _causal_selected_camera_inputs,
    _canonical_sha256,
    _load_calibration,
    _projection_matrix,
    _read_h5_frame_zero,
    frame_zero_camera_support,
    select_nested_frame_zero_observation_plans,
    triangulate_observation_ransac,
)
from .deform360_raw_camera_uncertainty import (
    RawCameraUncertaintyConfig,
    _pixel_sigma_from_median_reprojection,
    jacobian_measurement_covariance,
    leave_one_camera_out_covariance,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "Deform360AdaptiveCovarianceNestedMeasurementV1"
MEASUREMENT_ARCHIVE_FILENAME = "measurement.npz"
UNCERTAINTY_ARCHIVE_FILENAME = "measurement_uncertainty.npz"
MANIFEST_FILENAME = "nested_measurement_manifest.json"
CAMERA_BUDGETS = (4, 8)
EXTERNAL_BACKBONE_SEAL_FILENAME = "prediction_seal.json"
EXTERNAL_PHYSICAL_MANIFEST_FILENAME = "physical_prediction_manifest.json"
EXTERNAL_BACKBONE_ARTIFACT_KIND = "Deform360BiasAwareProspectiveBackboneSeal"
EXTERNAL_PHYSICAL_ARTIFACT_KIND = "Deform360BiasAwareProspectivePhysicalPrediction"
EXTERNAL_PHYSICAL_ARRAY_ROLES = (
    "action_support",
    "driven_readout_m",
    "frame_zero_points_m",
    "persistence_m",
    "prediction_m",
    "zero_action_readout_m",
)
MEASUREMENT_ARRAY_ROLES = (
    "measurement_m",
    "measurement_validity",
    "center_ids",
    "selected_cameras",
    "update_frames",
)
UNCERTAINTY_ARRAY_ROLES = (
    "measurement_covariance_m2",
    "measurement_covariance_valid",
)
_BASE_EXTERNAL_PHYSICAL_INPUT_ROLES = {
    "protocol",
    "prediction_prefix_manifest",
    "frame_zero_manifest",
    "frame_zero_geometry",
    "known_action",
    "prediction_only_input",
    "prediction_only_summary",
}
_TWIN_EXTERNAL_PHYSICAL_INPUT_ROLES = _BASE_EXTERNAL_PHYSICAL_INPUT_ROLES | {
    "episode_graph",
    "simulator_final_data",
    "state_artifact",
    "twin_summary",
    "automatic_twin_log",
}
_WARP_EXTERNAL_PHYSICAL_INPUT_ROLES = _TWIN_EXTERNAL_PHYSICAL_INPUT_ROLES | {
    "driven_result",
    "driven_trajectory",
    "zero_action_result",
    "zero_action_trajectory",
}

_EXTERNAL_BACKBONE_BOUNDARY = {
    "object_observation_frames_used": [0],
    "known_future_robot_action_read": True,
    "future_object_rgb_read": False,
    "future_object_geometry_read": False,
    "future_object_track_read": False,
    "future_tactile_read": False,
    "target_metric_read": False,
    "prediction_hashed_before_future_outcome_scoring": True,
}
_EXTERNAL_PHYSICAL_BOUNDARY = {
    "object_observation_frames_used": [0],
    "known_future_robot_action_read": True,
    "future_object_rgb_read": False,
    "future_object_geometry_read": False,
    "future_object_track_read": False,
    "future_tactile_read": False,
    "outcome_read": False,
    "prediction_hashed_before_future_outcome_scoring": True,
}
RETAINED_MEASUREMENT_FAILURE_STATUS = "retained_technical_failure"
RETAINED_MEASUREMENT_FAILURE_CODES = (
    "automatic_twin_backend_failure",
    "prediction_runtime_failure",
    "resource_exhaustion",
)
RETAINED_FAILURE_CAMERA_ACCOUNTING = {
    "adaptive_charge_is_causal_offline_policy_demand": True,
    "all_eight_streams_eventually_tracked_for_fixed8_shadow": False,
    "realized_acquisition_or_wall_clock_saving_claimed": False,
    "frame_zero_all_camera_planning_excluded": True,
}
IDENTITY_PERSISTENCE_ADAPTER_KEY = "identity_persistence_adapter"
IDENTITY_PERSISTENCE_POLICY = "original-splat-identity-persistence"
IDENTITY_PERSISTENCE_ADAPTER_KIND = (
    "Deform360AdaptiveCovarianceOriginalSplatIdentityPersistenceV1"
)


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _case_identity(lock: Mapping[str, Any], case_id: str) -> dict[str, Any]:
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
    _require(len(matches) == 1, "case is outside the exact H2-locked cohort")
    return matches[0]


def _regular_file(path: str | Path, *, label: str) -> Path:
    source = Path(path).absolute()
    _require(source.is_file() and not source.is_symlink(), f"{label} is invalid")
    _require(source.resolve(strict=True) == source, f"{label} is noncanonical")
    return source


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_regular_file(path: str | Path, *, label: str) -> _FileSnapshot:
    """Read one canonical regular file without following a final symlink."""

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


def _recheck_file_snapshot(snapshot: _FileSnapshot, *, label: str) -> None:
    current = _snapshot_regular_file(snapshot.path, label=label)
    _require(
        current.identity == snapshot.identity
        and current.sha256 == snapshot.sha256
        and current.payload == snapshot.payload,
        f"{label} changed after loading",
    )


def _load_json_snapshot(snapshot: _FileSnapshot, *, label: str) -> dict[str, Any]:
    def strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _self_sha256(value: Mapping[str, Any], *, digest_key: str) -> str:
    payload = dict(value)
    payload.pop(digest_key, None)
    return _canonical_sha256(payload)


def _require_target_free_metadata(value: Any, *, label: str) -> None:
    """Reject undeclared evaluation-bearing metadata fields recursively."""

    allowed_boundary_fields = {
        "target_metric_read": False,
        "outcome_read": False,
        "prediction_hashed_before_future_outcome_scoring": True,
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(
                token in lowered
                for token in ("target", "outcome", "metric", "score", "ground_truth")
            ):
                _require(
                    key == "automatic_twin_state_metrics"
                    or (
                        key in allowed_boundary_fields
                        and item is allowed_boundary_fields[key]
                    ),
                    f"{label} carries undeclared evaluation metadata",
                )
            if key == "automatic_twin_state_metrics":
                # Frozen frame-zero twin-admission diagnostics use "target" for
                # the observed frame-zero point set, not the future outcome.
                continue
            _require_target_free_metadata(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_target_free_metadata(item, label=label)


def _external_array_sha256(value: np.ndarray) -> str:
    """Match the checksum convention of the frozen external executor."""

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


def _physical_arrays(
    snapshot: _FileSnapshot,
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(EXTERNAL_PHYSICAL_ARRAY_ROLES),
                "physical archive roles changed",
            )
            arrays = {
                role: np.asarray(stored[role]).copy()
                for role in EXTERNAL_PHYSICAL_ARRAY_ROLES
            }
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("physical archive is invalid") from error
    prior = arrays["prediction_m"]
    persistence = arrays["persistence_m"]
    frame_zero = arrays["frame_zero_points_m"]
    driven = arrays["driven_readout_m"]
    zero_action = arrays["zero_action_readout_m"]
    action_support = arrays["action_support"]
    _require(
        prior.ndim == 3
        and prior.shape[0] == FRAME_COUNT
        and prior.shape[2] == 3
        and persistence.shape == prior.shape
        and driven.shape == prior.shape
        and zero_action.shape == prior.shape
        and frame_zero.shape == prior.shape[1:],
        "physical trajectories have invalid shapes",
    )
    _require(
        np.issubdtype(prior.dtype, np.floating)
        and persistence.dtype == prior.dtype
        and frame_zero.dtype == prior.dtype,
        "physical trajectories have inconsistent dtypes",
    )
    _require(
        action_support.shape == (prior.shape[1],)
        and np.issubdtype(action_support.dtype, np.number)
        and np.all(np.isfinite(action_support))
        and np.all((action_support >= 0.0) & (action_support <= 1.0)),
        "physical action support is invalid",
    )
    _require(
        all(
            np.all(np.isfinite(arrays[role])) for role in EXTERNAL_PHYSICAL_ARRAY_ROLES
        ),
        "physical trajectories are non-finite",
    )
    _require(
        np.array_equal(prior[0], frame_zero)
        and np.array_equal(driven[0], frame_zero)
        and np.array_equal(zero_action[0], frame_zero)
        and np.array_equal(
            persistence,
            np.repeat(frame_zero[None], FRAME_COUNT, axis=0),
        ),
        "physical frame-zero material identity changed",
    )
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def _external_case_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case": identity["case_id"],
        "object_id": identity["object_id"],
        "episode_id": identity["episode_id"],
        "episode_key": (f"{identity['object_id']}/{int(identity['episode_id'])}"),
        "stratum": identity["stratum"],
        # The adapter deliberately exposes every target-closed H2 case through
        # the old engine's calibration role.  It does not authorize evaluation.
        "role": "calibration",
    }


def _bound_external_path(
    record: Mapping[str, Any],
    expected: _FileSnapshot,
    *,
    label: str,
    hash_key: str,
) -> None:
    _require(
        isinstance(record, Mapping)
        and set(record) >= {"path", hash_key}
        and isinstance(record.get("path"), str),
        f"{label} binding is invalid",
    )
    bound = _regular_file(str(record["path"]), label=label)
    _require(bound == expected.path, f"{label} binds another file")
    _require(record.get(hash_key) == expected.sha256, f"{label} checksum changed")


def _validate_external_physical_provenance(
    *,
    lock: Mapping[str, Any],
    identity: Mapping[str, Any],
    archive_snapshot: _FileSnapshot,
    manifest_snapshot: _FileSnapshot,
    seal_snapshot: _FileSnapshot,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Validate the complete H2-adapted external physical artifact chain."""

    manifest = _load_json_snapshot(
        manifest_snapshot,
        label="external physical manifest",
    )
    seal = _load_json_snapshot(seal_snapshot, label="external backbone seal")
    _require_target_free_metadata(manifest, label="external physical manifest")
    _require_target_free_metadata(seal, label="external backbone seal")
    expected_case = _external_case_record(identity)
    expected_manifest_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "protocol_config_sha256",
        *expected_case,
        "physical_mode",
        "physical_admitted",
        "fallback_diagnostics",
        "frozen_predictor",
        "physical_prediction_archive",
        "input_files",
        "runtime_provenance",
        "information_boundary",
        "passed",
        "result_sha256",
    }
    _require(
        set(manifest) == expected_manifest_keys,
        "external physical manifest fields changed",
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_kind") == EXTERNAL_PHYSICAL_ARTIFACT_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("protocol_config_sha256") == lock["artifact_sha256"]
        and all(manifest.get(key) == value for key, value in expected_case.items())
        and manifest.get("passed") is True,
        "external physical manifest is not bound to this H2 case",
    )
    _require(
        manifest.get("result_sha256")
        == _self_sha256(manifest, digest_key="result_sha256"),
        "external physical manifest self-checksum changed",
    )
    _require(
        manifest.get("information_boundary") == _EXTERNAL_PHYSICAL_BOUNDARY,
        "external physical manifest crossed the target boundary",
    )
    mode = manifest.get("physical_mode")
    _require(
        mode in {"warp_twin", "persistence_fallback"}
        and manifest.get("physical_admitted") is (mode == "warp_twin")
        and (manifest.get("fallback_diagnostics") is not None)
        is (mode == "persistence_fallback"),
        "external physical disposition is inconsistent",
    )
    input_files = manifest.get("input_files")
    allowed_input_sets = (
        (_WARP_EXTERNAL_PHYSICAL_INPUT_ROLES,)
        if mode == "warp_twin"
        else (
            _BASE_EXTERNAL_PHYSICAL_INPUT_ROLES,
            _TWIN_EXTERNAL_PHYSICAL_INPUT_ROLES,
        )
    )
    _require(
        isinstance(input_files, Mapping)
        and any(set(input_files) == expected for expected in allowed_input_sets)
        and all(
            isinstance(record, Mapping)
            and set(record) == {"path", "sha256"}
            and isinstance(record.get("path"), str)
            and bool(record["path"])
            and isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in record["sha256"])
            for record in input_files.values()
        ),
        "external physical input boundary changed",
    )
    manifest_archive = manifest.get("physical_prediction_archive")
    _require(
        isinstance(manifest_archive, Mapping)
        and set(manifest_archive) == {"path", "file_sha256", "array_sha256"},
        "external physical manifest archive binding changed",
    )
    # The frozen outer backbone builder copies the work archive but copies the
    # physical manifest byte-for-byte.  Its path therefore still names the
    # work archive.  Bind the copied archive by exact file/array identities;
    # requiring the stale work path would make the sealed case non-portable.
    _require(
        isinstance(manifest_archive.get("path"), str)
        and bool(manifest_archive["path"])
        and manifest_archive.get("file_sha256") == archive_snapshot.sha256,
        "external physical manifest binds another archive identity",
    )
    array_hashes = manifest_archive.get("array_sha256")
    observed_hashes = {
        role: _external_array_sha256(arrays[role])
        for role in EXTERNAL_PHYSICAL_ARRAY_ROLES
    }
    _require(
        isinstance(array_hashes, Mapping) and dict(array_hashes) == observed_hashes,
        "external physical manifest array hashes changed",
    )

    expected_seal_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "protocol_config_sha256",
        *expected_case,
        "frame_count",
        "material_point_count",
        "material_identity_sha256",
        "prediction_archive",
        "physical_manifest",
        "information_boundary",
        "result_sha256",
    }
    _require(set(seal) == expected_seal_keys, "external backbone seal fields changed")
    _require(
        seal.get("schema_version") == 1
        and seal.get("artifact_kind") == EXTERNAL_BACKBONE_ARTIFACT_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("protocol_config_sha256") == lock["artifact_sha256"]
        and all(seal.get(key) == value for key, value in expected_case.items()),
        "external backbone seal is not bound to this H2 case",
    )
    _require(
        seal.get("result_sha256") == _self_sha256(seal, digest_key="result_sha256"),
        "external backbone seal self-checksum changed",
    )
    _require(
        seal.get("information_boundary") == _EXTERNAL_BACKBONE_BOUNDARY,
        "external backbone seal crossed the target boundary",
    )
    _require(
        seal.get("frame_count") == FRAME_COUNT
        and seal.get("material_point_count") == len(arrays["frame_zero_points_m"])
        and seal.get("material_identity_sha256")
        == observed_hashes["frame_zero_points_m"],
        "external backbone material identity changed",
    )
    seal_archive = seal.get("prediction_archive")
    _require(
        isinstance(seal_archive, Mapping)
        and set(seal_archive) == {"path", "file_sha256", "array_sha256"},
        "external backbone archive binding changed",
    )
    _bound_external_path(
        seal_archive,
        archive_snapshot,
        label="seal-bound physical archive",
        hash_key="file_sha256",
    )
    _require(
        seal_archive.get("array_sha256") == observed_hashes,
        "external backbone archive array hashes changed",
    )
    seal_manifest = seal.get("physical_manifest")
    _require(
        isinstance(seal_manifest, Mapping)
        and set(seal_manifest) == {"path", "file_sha256"},
        "external backbone manifest binding changed",
    )
    _bound_external_path(
        seal_manifest,
        manifest_snapshot,
        label="seal-bound physical manifest",
        hash_key="file_sha256",
    )
    return {
        "external_backbone_seal_file_sha256": seal_snapshot.sha256,
        "external_backbone_seal_result_sha256": seal["result_sha256"],
        "external_physical_manifest_file_sha256": manifest_snapshot.sha256,
        "external_physical_manifest_result_sha256": manifest["result_sha256"],
        "physical_archive_file_sha256": archive_snapshot.sha256,
        "physical_archive_array_sha256": observed_hashes,
    }


def _validated_source_stage_lineage(
    *,
    lock: Mapping[str, Any],
    lock_snapshot: _FileSnapshot,
    identity: Mapping[str, Any],
    h2_commit: str,
    physical_manifest_snapshot: _FileSnapshot,
    processed_episode_dir: Path,
    source_custody_seal: str | Path,
) -> dict[str, Any]:
    """Bind camera inputs to the exact staged case consumed by the backbone."""

    physical_manifest = _load_json_snapshot(
        physical_manifest_snapshot,
        label="external physical manifest",
    )
    inputs = physical_manifest["input_files"]

    def bound_manifest(
        role: str, filename: str
    ) -> tuple[_FileSnapshot, dict[str, Any]]:
        record = inputs.get(role)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "sha256"}
            and isinstance(record.get("path"), str),
            f"external physical {role} binding changed",
        )
        snapshot = _snapshot_regular_file(
            record["path"],
            label=f"external physical {role}",
        )
        _require(
            snapshot.path.name == filename and snapshot.sha256 == record["sha256"],
            f"external physical {role} content changed",
        )
        return snapshot, _load_json_snapshot(snapshot, label=role)

    prefix_snapshot, prefix = bound_manifest(
        "prediction_prefix_manifest",
        "prediction_prefix_manifest.json",
    )
    frame_snapshot, frame = bound_manifest(
        "frame_zero_manifest",
        "frame_zero_reconstruction_manifest.json",
    )
    staged = prefix_snapshot.path.parent
    _require(
        frame_snapshot.path.parent == staged
        and processed_episode_dir == staged / "prefix" / "episode_0000",
        "processed prefix is outside the exact physical staged case",
    )
    expected_case = _external_case_record(identity)
    _require(
        prefix.get("artifact_kind") == "Deform360BiasAwarePredictionPrefix"
        and frame.get("artifact_kind") == "Deform360BiasAwareFrameZeroReconstruction"
        and prefix.get("protocol_id") == frame.get("protocol_id") == PROTOCOL_ID
        and prefix.get("protocol_config_sha256")
        == frame.get("protocol_config_sha256")
        == lock["artifact_sha256"]
        and prefix.get("result_sha256")
        == _self_sha256(prefix, digest_key="result_sha256")
        and frame.get("result_sha256")
        == _self_sha256(frame, digest_key="result_sha256")
        and all(
            prefix.get(key) == frame.get(key) == value
            for key, value in expected_case.items()
        ),
        "physical source-stage manifests bind another H2 case",
    )
    prefix_inputs = prefix.get("inputs_sha256")
    frame_inputs = frame.get("inputs_sha256")
    source_preparation_sha256 = (
        prefix_inputs.get("source_preparation_manifest")
        if isinstance(prefix_inputs, Mapping)
        else None
    )
    _require(
        isinstance(prefix_inputs, Mapping)
        and prefix_inputs.get("protocol") == lock_snapshot.sha256
        and isinstance(frame_inputs, Mapping)
        and frame_inputs.get("prediction_prefix_manifest") == prefix_snapshot.sha256
        and isinstance(source_preparation_sha256, str)
        and len(source_preparation_sha256) == 64
        and all(
            character in "0123456789abcdef" for character in source_preparation_sha256
        ),
        "physical source-stage lineage changed",
    )
    camera_records = prefix.get("camera_records")
    _require(
        isinstance(camera_records, list)
        and prefix.get("camera_count") == len(camera_records)
        and camera_records,
        "prediction-prefix camera records changed",
    )
    by_camera: dict[str, dict[str, Any]] = {}
    for record in camera_records:
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "camera",
                "prefix_video_sha256",
                "frame_zero_video_sha256",
                "frame_zero_mask_sha256",
            }
            and isinstance(record.get("camera"), str)
            and bool(record["camera"])
            and record["camera"] not in by_camera
            and all(
                isinstance(record.get(role), str)
                and len(record[role]) == 64
                and all(character in "0123456789abcdef" for character in record[role])
                for role in (
                    "prefix_video_sha256",
                    "frame_zero_video_sha256",
                    "frame_zero_mask_sha256",
                )
            ),
            "prediction-prefix camera binding changed",
        )
        by_camera[str(record["camera"])] = dict(record)
    planning_cameras = tuple(sorted(by_camera))
    frame_cameras = frame.get("cameras")
    frame_outputs = frame.get("outputs_sha256")
    depth_by_camera = (
        frame_outputs.get("depth_by_camera")
        if isinstance(frame_outputs, Mapping)
        else None
    )
    _require(
        isinstance(frame_cameras, list)
        and frame_cameras == list(planning_cameras)
        and frame.get("camera_count") == len(planning_cameras)
        and isinstance(depth_by_camera, Mapping)
        and set(depth_by_camera) == set(planning_cameras)
        and all(
            isinstance(depth_by_camera[camera], str)
            and len(depth_by_camera[camera]) == 64
            and all(
                character in "0123456789abcdef" for character in depth_by_camera[camera]
            )
            for camera in planning_cameras
        ),
        "prefix and frame-zero planning camera panels differ",
    )
    custody_snapshot = _snapshot_regular_file(
        source_custody_seal,
        label="source-custody seal",
    )
    custody = validate_confirmation_source_custody_envelope(
        custody_snapshot.path,
        lock_snapshot.path,
        h2_commit,
        str(identity["case_id"]),
        expected_h1=lock["two_commit_freeze"]["implementation_commit_h1"],
        expected_staged_case_dir=staged,
    )
    _require(
        custody.get("camera_panel") == list(planning_cameras)
        and custody["manifests"]["prediction_prefix"]["file_sha256"]
        == prefix_snapshot.sha256
        and custody["manifests"]["prediction_prefix"]["result_sha256"]
        == prefix["result_sha256"]
        and custody["manifests"]["frame_zero"]["file_sha256"] == frame_snapshot.sha256
        and custody["manifests"]["frame_zero"]["result_sha256"]
        == frame["result_sha256"]
        and custody["manifests"]["source_preparation"]["file_sha256"]
        == source_preparation_sha256,
        "source-custody seal differs from the physical source-stage lineage",
    )
    return {
        "manifest_record": {
            "prediction_prefix_manifest": {
                "path": str(prefix_snapshot.path),
                "file_sha256": prefix_snapshot.sha256,
                "result_sha256": prefix["result_sha256"],
            },
            "frame_zero_manifest": {
                "path": str(frame_snapshot.path),
                "file_sha256": frame_snapshot.sha256,
                "result_sha256": frame["result_sha256"],
            },
            "source_preparation_manifest_file_sha256": source_preparation_sha256,
            "source_custody_seal": {
                "path": str(custody_snapshot.path),
                "file_sha256": custody_snapshot.sha256,
                "artifact_sha256": custody["artifact_sha256"],
            },
        },
        "camera_records": by_camera,
        "planning_cameras": planning_cameras,
        "depth_file_sha256_by_camera": dict(depth_by_camera),
        "snapshots": (prefix_snapshot, frame_snapshot, custody_snapshot),
    }


def _validate_planning_camera_source_bindings(
    processed_episode_dir: Path,
    selected_inputs: Mapping[str, Any],
    prefix_camera_records: Mapping[str, Mapping[str, Any]],
    planning_cameras: Sequence[str],
    depth_file_sha256_by_camera: Mapping[str, str],
) -> dict[str, Any]:
    """Replay the complete planning panel after causal route decisions."""

    selected_video_snapshots: list[_FileSnapshot] = []
    mask_snapshots: list[_FileSnapshot] = []
    depth_snapshots: list[_FileSnapshot] = []
    frame_zero_records: dict[str, dict[str, str]] = {}
    _require(
        tuple(planning_cameras) == tuple(sorted(prefix_camera_records))
        and set(depth_file_sha256_by_camera) == set(planning_cameras)
        and set(selected_inputs) <= set(planning_cameras),
        "planning camera panel differs from its source-stage manifests",
    )
    for camera in planning_cameras:
        expected_root = processed_episode_dir / camera
        prefix_record = prefix_camera_records[camera]
        video_path = expected_root / "undistorted.mp4"
        _require(
            video_path.is_file()
            and not video_path.is_symlink()
            and video_path.resolve(strict=True) == video_path,
            "planning camera video is missing or noncanonical",
        )
        mask_path = expected_root / "mask_refined.h5"
        depth_path = expected_root / "rendered_depth.h5"
        mask_snapshot = _snapshot_regular_file(
            mask_path,
            label=f"{camera} sealed frame-zero mask",
        )
        depth_snapshot = _snapshot_regular_file(
            depth_path,
            label=f"{camera} sealed frame-zero depth",
        )
        mask_zero = _read_h5_frame_zero(mask_path)
        depth_zero = _read_h5_frame_zero(depth_path)
        mask_sha256 = _array_sha256(mask_zero)
        depth_array_sha256 = _array_sha256(depth_zero)
        _recheck_file_snapshot(
            mask_snapshot,
            label=f"{camera} sealed frame-zero mask",
        )
        _recheck_file_snapshot(
            depth_snapshot,
            label=f"{camera} sealed frame-zero depth",
        )
        _require(
            mask_snapshot.sha256 == prefix_record["frame_zero_mask_sha256"]
            and depth_snapshot.sha256 == depth_file_sha256_by_camera[camera],
            "planning mask/depth differs from its source-stage seal",
        )
        frame_zero_records[camera] = {
            "mask_frame_zero_array_sha256": mask_sha256,
            "mask_file_sha256": mask_snapshot.sha256,
            "depth_frame_zero_array_sha256": depth_array_sha256,
            "depth_file_sha256": depth_snapshot.sha256,
        }
        mask_snapshots.append(mask_snapshot)
        depth_snapshots.append(depth_snapshot)
        if camera in selected_inputs:
            input_record = selected_inputs[camera]
            _require(
                isinstance(input_record, Mapping)
                and input_record["video"]["path"] == str(video_path)
                and input_record["frame_zero_mask"]["path"] == str(mask_path)
                and input_record["frame_zero_depth"]["path"] == str(depth_path)
                and input_record["frame_zero_mask"]["frame_zero_array_sha256"]
                == mask_sha256
                and input_record["frame_zero_depth"]["frame_zero_array_sha256"]
                == depth_array_sha256,
                "selected camera files differ from the planning-panel replay",
            )
            video_snapshot = _snapshot_regular_file(
                video_path,
                label=f"{camera} complete staged prefix video",
            )
            _require(
                video_snapshot.sha256 == prefix_record["prefix_video_sha256"],
                "selected camera video differs from the prediction-prefix seal",
            )
            selected_video_snapshots.append(video_snapshot)
    return {
        "frame_zero_records": frame_zero_records,
        "snapshots": (
            *mask_snapshots,
            *depth_snapshots,
            *selected_video_snapshots,
        ),
    }


def _validate_calibration_camera_panel(
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    planning_cameras: Sequence[str],
    *,
    label: str = "calibration",
) -> None:
    """Require calibration dictionaries for exactly the sealed planning panel."""

    expected = tuple(planning_cameras)
    _require(
        tuple(sorted(intrinsics)) == expected and tuple(sorted(extrinsics)) == expected,
        f"{label} camera panel differs from the sealed planning panel",
    )


def _empty_budget_arrays(
    prior: np.ndarray,
    frame_zero: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    # Match the released raw-measurement and uncertainty builders exactly.
    measurement = np.full(prior.shape, np.nan, dtype=np.float32)
    visibility = np.zeros(prior.shape[:2], dtype=bool)
    validity = np.zeros(prior.shape[:2], dtype=bool)
    measurement[0, candidates] = frame_zero[candidates]
    visibility[0, candidates] = True
    validity[0, candidates] = True
    covariance = np.full((*prior.shape[:2], 3, 3), np.nan, dtype=np.float32)
    covariance_valid = np.zeros(prior.shape[:2], dtype=bool)
    return {
        "measurement_m": measurement,
        "measurement_visibility": visibility,
        "measurement_validity": validity,
        "measurement_covariance_m2": covariance,
        "measurement_covariance_valid": covariance_valid,
    }


def _triangulate_budget_update(
    *,
    frame: int,
    centers: np.ndarray,
    selected_cameras: Sequence[str],
    tracks_by_camera: Mapping[str, Mapping[int, np.ndarray]],
    projection_matrices: Mapping[str, np.ndarray],
    camera_origins: Mapping[str, np.ndarray],
    frame_zero: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    observation_config: RawCameraObservationConfig,
    uncertainty_config: RawCameraUncertaintyConfig,
) -> list[dict[str, Any]]:
    """Fill one budget/update and return target-free center diagnostics."""

    records: list[dict[str, Any]] = []
    for center_id_value in centers:
        center_id = int(center_id_value)
        observations = {
            camera: np.asarray(tracks_by_camera[camera][center_id], dtype=float)
            for camera in selected_cameras
            if center_id in tracks_by_camera[camera]
        }
        point, triangulation = triangulate_observation_ransac(
            observations,
            projection_matrices,
            camera_origins,
            frame_zero[center_id],
            config=observation_config,
        )
        record: dict[str, Any] = {
            **triangulation,
            "center_id": center_id,
            "covariance_valid": False,
            "covariance_decision": "measurement_rejected",
        }
        if point is None:
            records.append(record)
            continue
        arrays["measurement_m"][frame, center_id] = point
        arrays["measurement_visibility"][frame, center_id] = True
        arrays["measurement_validity"][frame, center_id] = True
        inlier_cameras = tuple(str(value) for value in triangulation["inlier_cameras"])
        inlier_observations = {
            camera: observations[camera] for camera in inlier_cameras
        }
        sigma = _pixel_sigma_from_median_reprojection(
            float(triangulation["median_reprojection_error_px"]),
            uncertainty_config.pixel_noise_floor_px,
        )
        geometric, geometric_diagnostic = jacobian_measurement_covariance(
            point,
            [projection_matrices[camera] for camera in sorted(inlier_observations)],
            sigma,
            maximum_condition_number=(
                uncertainty_config.maximum_information_condition_number
            ),
        )
        record["jacobian"] = geometric_diagnostic
        record["pixel_sigma"] = sigma
        if geometric is None:
            record["covariance_decision"] = geometric_diagnostic["decision"]
            records.append(record)
            continue
        empirical, leave_one_out = leave_one_camera_out_covariance(
            inlier_observations,
            projection_matrices,
        )
        combined = 0.5 * (
            geometric + empirical + np.swapaxes(geometric + empirical, 0, 1)
        )
        eigenvalues = np.linalg.eigvalsh(combined)
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
            record["covariance_decision"] = "combined_covariance_failure"
            records.append(record)
            continue
        arrays["measurement_covariance_m2"][frame, center_id] = combined
        arrays["measurement_covariance_valid"][frame, center_id] = True
        record.update(
            {
                "covariance_valid": True,
                "covariance_decision": "accepted",
                "leave_one_out_sample_count": int(len(leave_one_out)),
                "principal_standard_deviation_m": np.sqrt(eigenvalues).tolist(),
            }
        )
        records.append(record)
    return records


def _reliability_record(
    arrays: Mapping[str, np.ndarray],
    centers: np.ndarray,
    frame: int,
    frame_zero: np.ndarray,
) -> dict[str, Any]:
    routing = FROZEN_ADAPTIVE_COVARIANCE_CONFIG
    result = normalized_covariance_dispersion(
        arrays["measurement_covariance_m2"],
        arrays["measurement_covariance_valid"],
        centers,
        frame,
        frame_zero,
        quantile=routing.covariance_quantile,
    )
    normalized = result["normalized_covariance_dispersion"]
    reliable = (
        result["valid_covariance_center_count"]
        >= routing.minimum_valid_covariance_centers
        and normalized is not None
        and normalized <= routing.maximum_normalized_covariance_dispersion
    )
    return {**result, "reliable": bool(reliable)}


def _write_budget_archives(
    root: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    centers: np.ndarray,
    selected_cameras: Sequence[str],
) -> dict[str, Any]:
    root.mkdir()
    measurement_path = root / MEASUREMENT_ARCHIVE_FILENAME
    np.savez_compressed(
        measurement_path,
        measurement_m=arrays["measurement_m"],
        measurement_validity=arrays["measurement_validity"],
        center_ids=centers,
        selected_cameras=np.asarray(tuple(selected_cameras)),
        update_frames=np.asarray(UPDATE_FRAMES, dtype=np.int64),
    )
    uncertainty_path = root / UNCERTAINTY_ARCHIVE_FILENAME
    np.savez_compressed(
        uncertainty_path,
        measurement_covariance_m2=arrays["measurement_covariance_m2"],
        measurement_covariance_valid=arrays["measurement_covariance_valid"],
    )

    def record(
        path: Path,
        *,
        relative_path: str,
        expected_roles: Sequence[str],
    ) -> dict[str, Any]:
        snapshot = _snapshot_regular_file(path, label="measurement output archive")
        try:
            with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as stored:
                _require(
                    set(stored.files) == set(expected_roles),
                    "measurement output archive roles changed",
                )
                content = {
                    role: {
                        "dtype": np.asarray(stored[role]).dtype.str,
                        "shape": list(np.asarray(stored[role]).shape),
                        "array_sha256": confirmation_array_sha256(
                            np.asarray(stored[role])
                        ),
                    }
                    for role in expected_roles
                }
        except (OSError, ValueError, KeyError) as error:
            raise ValueError("measurement output archive is invalid") from error
        return {
            "relative_path": relative_path,
            "sha256": snapshot.sha256,
            "size_bytes": len(snapshot.payload),
            "arrays": content,
        }

    return {
        "measurement_archive": record(
            measurement_path,
            relative_path=f"{root.name}/{MEASUREMENT_ARCHIVE_FILENAME}",
            expected_roles=MEASUREMENT_ARRAY_ROLES,
        ),
        "uncertainty_archive": record(
            uncertainty_path,
            relative_path=f"{root.name}/{UNCERTAINTY_ARCHIVE_FILENAME}",
            expected_roles=UNCERTAINTY_ARRAY_ROLES,
        ),
    }


def build_confirmation_nested_measurements(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    physical_archive: str | Path,
    processed_episode_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    physical_manifest: str | Path,
    physical_prediction_seal: str | Path,
    source_custody_seal: str | Path,
    expected_h1: str | None = None,
    observation_config: RawCameraObservationConfig | None = None,
    uncertainty_config: RawCameraUncertaintyConfig | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one H2-locked nested measurement package."""

    forbidden_names = {
        "target",
        "outcome",
        "metric",
        "score",
        "ground_truth",
    }
    for value in (
        str(physical_archive),
        str(physical_manifest),
        str(physical_prediction_seal),
        str(source_custody_seal),
        str(processed_episode_dir),
        str(output_dir),
    ):
        lowered = Path(value).as_posix().lower()
        _require(
            not any(token in lowered for token in forbidden_names),
            "confirmation measurement path crosses the target boundary",
        )
    lock_snapshot = _snapshot_regular_file(lock_path, label="H2 lock")
    lock_source = lock_snapshot.path
    lock = load_confirmation_cohort_lock(
        lock_source,
        expected_implementation_commit_h1=expected_h1,
    )
    _recheck_file_snapshot(lock_snapshot, label="H2 lock")
    identity = _case_identity(lock, case_id)
    _require(
        isinstance(h2_commit, str)
        and len(h2_commit) == 40
        and h2_commit != lock["two_commit_freeze"]["implementation_commit_h1"]
        and all(character in "0123456789abcdef" for character in h2_commit),
        "H2 commit is invalid",
    )
    cfg = observation_config or RawCameraObservationConfig(selected_camera_count=8)
    uncertainty_cfg = uncertainty_config or RawCameraUncertaintyConfig()
    _require(
        cfg == RawCameraObservationConfig(selected_camera_count=8),
        "confirmation observation configuration changed",
    )
    _require(
        uncertainty_cfg == RawCameraUncertaintyConfig(),
        "confirmation uncertainty configuration changed",
    )
    uncertainty_cfg.validate()
    _require(
        runtime.config == cfg
        and runtime.source_sha256 == ALLTRACKER_RUNTIME_SOURCE_SHA256
        and runtime.checkpoint_sha256 == ALLTRACKER_CHECKPOINT_SHA256,
        "AllTracker runtime configuration or source identity changed",
    )

    physical_snapshot = _snapshot_regular_file(
        physical_archive,
        label="physical archive",
    )
    physical_manifest_snapshot = _snapshot_regular_file(
        physical_manifest,
        label="external physical manifest",
    )
    physical_seal_snapshot = _snapshot_regular_file(
        physical_prediction_seal,
        label="external backbone seal",
    )
    _require(
        physical_manifest_snapshot.path.name == EXTERNAL_PHYSICAL_MANIFEST_FILENAME
        and physical_seal_snapshot.path.name == EXTERNAL_BACKBONE_SEAL_FILENAME,
        "external physical provenance filenames changed",
    )
    physical_arrays = _physical_arrays(physical_snapshot)
    physical_provenance = _validate_external_physical_provenance(
        lock=lock,
        identity=identity,
        archive_snapshot=physical_snapshot,
        manifest_snapshot=physical_manifest_snapshot,
        seal_snapshot=physical_seal_snapshot,
        arrays=physical_arrays,
    )
    prior = physical_arrays["prediction_m"]
    frame_zero = physical_arrays["frame_zero_points_m"]
    processed = Path(processed_episode_dir).absolute()
    _require(
        processed.is_dir()
        and not processed.is_symlink()
        and processed.resolve(strict=True) == processed,
        "processed prefix episode is invalid",
    )
    source_lineage = _validated_source_stage_lineage(
        lock=lock,
        lock_snapshot=lock_snapshot,
        identity=identity,
        h2_commit=h2_commit,
        physical_manifest_snapshot=physical_manifest_snapshot,
        processed_episode_dir=processed,
        source_custody_seal=source_custody_seal,
    )
    output = Path(output_dir).absolute()
    _require(
        not output.exists() and not output.is_symlink(),
        "measurement output already exists",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(
        output.parent.resolve(strict=True) == output.parent,
        "measurement output parent is noncanonical",
    )
    for source in (
        lock_source,
        physical_snapshot.path,
        physical_manifest_snapshot.path,
        physical_seal_snapshot.path,
        *(
            snapshot.path
            for snapshot in source_lineage["snapshots"]
            if snapshot.path
            not in {
                physical_manifest_snapshot.path,
                physical_seal_snapshot.path,
            }
        ),
        processed,
    ):
        _require(
            output != source
            and output not in source.parents
            and source not in output.parents,
            "measurement output overlaps an input",
        )

    intrinsics_snapshot = _snapshot_regular_file(
        processed / "undistorted_intrinsics.npy",
        label="intrinsics",
    )
    extrinsics_snapshot = _snapshot_regular_file(
        processed / "extrinsics.npy",
        label="extrinsics",
    )
    intrinsics, extrinsics = _load_calibration(processed)
    _recheck_file_snapshot(intrinsics_snapshot, label="intrinsics")
    _recheck_file_snapshot(extrinsics_snapshot, label="extrinsics")
    planning_cameras = source_lineage["planning_cameras"]
    _validate_calibration_camera_panel(
        intrinsics,
        extrinsics,
        planning_cameras,
    )
    cameras, support, projected = frame_zero_camera_support(
        frame_zero,
        processed,
        intrinsics,
        extrinsics,
        depth_tolerance_m=cfg.frame_zero_depth_tolerance_m,
    )
    _require(
        tuple(cameras) == planning_cameras,
        "frame-zero planner camera panel differs from the sealed planning panel",
    )
    nested = select_nested_frame_zero_observation_plans(
        frame_zero,
        cameras,
        support,
        projected,
        extrinsics,
        config=cfg,
    )
    centers = np.asarray(nested["center_ids"], dtype=np.int64)
    candidates = np.asarray(nested["candidate_ids"], dtype=np.int64)
    plans = nested["prefix_plans"]
    selected = {
        budget: tuple(plans[budget]["selected_cameras"]) for budget in CAMERA_BUDGETS
    }
    _require(selected[8][:4] == selected[4], "nested camera plan changed")
    for camera in selected[4]:
        _require(
            np.array_equal(
                np.asarray(plans[4]["query_ids"][camera]),
                np.asarray(plans[8]["query_ids"][camera]),
            )
            and np.array_equal(
                np.asarray(plans[4]["query_pixels"][camera]),
                np.asarray(plans[8]["query_pixels"][camera]),
            ),
            "four/eight nested query plans changed",
        )
    projection_matrices = {
        camera: _projection_matrix(intrinsics[camera], extrinsics[camera])
        for camera in selected[8]
    }
    camera_origins = {
        camera: np.asarray(extrinsics[camera], dtype=float)[:3, 3]
        for camera in selected[8]
    }
    arrays = {
        budget: _empty_budget_arrays(
            prior,
            frame_zero,
            candidates,
        )
        for budget in CAMERA_BUDGETS
    }

    update_records: list[dict[str, Any]] = []
    plan8 = plans[8]
    for frame in UPDATE_FRAMES:
        tracks_by_camera: dict[str, dict[int, np.ndarray]] = {}
        tracker_records: list[dict[str, Any]] = []

        def track(
            cameras_to_track: Sequence[str],
            *,
            role: str,
            four_view_decision_materialized: bool,
        ) -> None:
            for camera in cameras_to_track:
                query_ids = np.asarray(plan8["query_ids"][camera], dtype=np.int64)
                query_pixels = np.asarray(plan8["query_pixels"][camera], dtype=float)
                tracks, visible, record = runtime.track_prefix(
                    processed / camera / "undistorted.mp4",
                    query_pixels,
                    frame,
                )
                tracks = np.asarray(tracks)
                visible = np.asarray(visible)
                _require(
                    tracks.shape == (len(query_ids), 2)
                    and np.issubdtype(tracks.dtype, np.floating)
                    and visible.shape == (len(query_ids),)
                    and visible.dtype == np.dtype(bool)
                    and np.all(np.isfinite(tracks[visible])),
                    "AllTracker prefix result is invalid",
                )
                _require(
                    isinstance(record, Mapping)
                    and record.get("maximum_video_frame_read") == frame
                    and isinstance(record.get("decoded_rgb_prefix_sha256"), str)
                    and len(record["decoded_rgb_prefix_sha256"]) == 64
                    and all(
                        character in "0123456789abcdef"
                        for character in record["decoded_rgb_prefix_sha256"]
                    ),
                    "AllTracker prefix provenance is invalid",
                )
                _require_target_free_metadata(
                    record,
                    label="AllTracker prefix provenance",
                )
                tracks_by_camera[camera] = {
                    int(point_id): np.asarray(tracks[index], dtype=float)
                    for index, point_id in enumerate(query_ids)
                    if bool(visible[index])
                }
                tracker_records.append(
                    {
                        **record,
                        "camera": camera,
                        "query_ids": query_ids.tolist(),
                        "execution_role": role,
                        "execution_index_within_update": len(tracker_records),
                        "four_view_decision_already_materialized": (
                            four_view_decision_materialized
                        ),
                    }
                )

        track(
            selected[4],
            role="adaptive_first_four",
            four_view_decision_materialized=False,
        )
        centers4 = _triangulate_budget_update(
            frame=frame,
            centers=centers,
            selected_cameras=selected[4],
            tracks_by_camera=tracks_by_camera,
            projection_matrices=projection_matrices,
            camera_origins=camera_origins,
            frame_zero=frame_zero,
            arrays=arrays[4],
            observation_config=cfg,
            uncertainty_config=uncertainty_cfg,
        )
        reliability4 = _reliability_record(arrays[4], centers, frame, frame_zero)
        four_accepted_before_shadow = bool(reliability4["reliable"])
        extra = selected[8][4:]
        track(
            extra,
            role=(
                "fixed_eight_shadow_after_four_decision"
                if four_accepted_before_shadow
                else "adaptive_eight_escalation"
            ),
            four_view_decision_materialized=True,
        )
        centers8 = _triangulate_budget_update(
            frame=frame,
            centers=centers,
            selected_cameras=selected[8],
            tracks_by_camera=tracks_by_camera,
            projection_matrices=projection_matrices,
            camera_origins=camera_origins,
            frame_zero=frame_zero,
            arrays=arrays[8],
            observation_config=cfg,
            uncertainty_config=uncertainty_cfg,
        )
        reliability8 = _reliability_record(arrays[8], centers, frame, frame_zero)
        if four_accepted_before_shadow:
            route = "4_view_rbf"
            charged = 4
        elif reliability8["reliable"]:
            route = "8_view_rbf"
            charged = 8
        else:
            route = "physical_prior_fallback"
            charged = 8
        update_records.append(
            {
                "frame": int(frame),
                "four_view_decision_materialized_before_shadow_extra_four": True,
                "four_view_reliable_before_shadow": four_accepted_before_shadow,
                "offline_shadow_extra_four_tracked": True,
                "adaptive_route": route,
                "adaptive_charged_camera_streams": charged,
                "budget_reliability": {
                    "4": reliability4,
                    "8": reliability8,
                },
                "tracker": tracker_records,
                "centers": {
                    "4": centers4,
                    "8": centers8,
                },
            }
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        archive_records = {
            str(budget): _write_budget_archives(
                staging / f"budget-{budget}",
                arrays=arrays[budget],
                centers=centers,
                selected_cameras=selected[budget],
            )
            for budget in CAMERA_BUDGETS
        }
        selected_inputs = _causal_selected_camera_inputs(
            processed,
            selected[8],
            update_records,
        )
        planning_source_replay = _validate_planning_camera_source_bindings(
            processed,
            selected_inputs,
            source_lineage["camera_records"],
            planning_cameras,
            source_lineage["depth_file_sha256_by_camera"],
        )
        planning_source_snapshots = planning_source_replay["snapshots"]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "protocol_id": PROTOCOL_ID,
            "case_identity": identity,
            "lock_binding": {
                "implementation_commit_h1": lock["two_commit_freeze"][
                    "implementation_commit_h1"
                ],
                "cohort_lock_commit_h2": h2_commit,
                "cohort_lock_artifact_sha256": lock["artifact_sha256"],
                "cohort_lock_file_sha256": lock_snapshot.sha256,
            },
            "config": {
                "observation": asdict(cfg),
                "uncertainty": asdict(uncertainty_cfg),
                "adaptive_routing": asdict(FROZEN_ADAPTIVE_COVARIANCE_CONFIG),
            },
            "plan": {
                "candidate_ids": candidates.tolist(),
                "center_ids": centers.tolist(),
                "camera_activation_order": list(nested["camera_activation_order"]),
                "selected_cameras_by_budget": {
                    str(budget): list(selected[budget]) for budget in CAMERA_BUDGETS
                },
                "selection_score": {
                    str(budget): list(plans[budget]["selection_score"])
                    for budget in CAMERA_BUDGETS
                },
            },
            "inputs": {
                "physical_backbone": physical_provenance,
                "physical_archive": {
                    "sha256": physical_snapshot.sha256,
                    "frame_zero_array_sha256": _array_sha256(frame_zero),
                },
                "intrinsics_sha256": intrinsics_snapshot.sha256,
                "extrinsics_sha256": extrinsics_snapshot.sha256,
                "selected_camera_prefixes_and_frame_zero": selected_inputs,
                "source_stage_lineage": source_lineage["manifest_record"],
            },
            "tracker": {
                "name": "AllTracker",
                "molmomotion_revision": ALLTRACKER_MOLMOMOTION_REVISION,
                "source_tree": ALLTRACKER_SOURCE_TREE,
                "runtime_source_sha256": runtime.source_sha256,
                "checkpoint_sha256": runtime.checkpoint_sha256,
                "device": runtime.device_name,
            },
            "updates": update_records,
            "outputs": archive_records,
            "camera_accounting": {
                "adaptive_charge_is_causal_offline_policy_demand": True,
                "all_eight_streams_eventually_tracked_for_fixed8_shadow": True,
                "realized_acquisition_or_wall_clock_saving_claimed": False,
                "frame_zero_all_camera_planning_excluded": True,
            },
            "information_boundary": {
                "target_path_argument_accepted": False,
                "outcome_path_argument_accepted": False,
                "target_metric_or_outcome_score_computed": False,
                "future_geometry_read": False,
                "video_prefix_rule": "update u reads exactly frames [0,u]",
                "maximum_video_frame_read_by_update": list(UPDATE_FRAMES),
                "four_view_decision_precedes_shadow_extra_four": True,
            },
        }
        payload["artifact_sha256"] = _canonical_sha256(payload)
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        input_snapshots = (
            lock_snapshot,
            physical_snapshot,
            physical_manifest_snapshot,
            physical_seal_snapshot,
            intrinsics_snapshot,
            extrinsics_snapshot,
            *source_lineage["snapshots"],
            *planning_source_snapshots,
        )
        input_labels = (
            "H2 lock",
            "physical archive",
            "external physical manifest",
            "external backbone seal",
            "intrinsics",
            "extrinsics",
            "prediction-prefix manifest",
            "frame-zero manifest",
            "source-custody seal",
            *(
                f"{snapshot.path.parent.name} planning source file"
                for snapshot in planning_source_snapshots
            ),
        )
        _require(
            _causal_selected_camera_inputs(
                processed,
                selected[8],
                update_records,
            )
            == selected_inputs,
            "selected camera frame-zero inputs changed before publication",
        )
        replayed_planning_sources = _validate_planning_camera_source_bindings(
            processed,
            selected_inputs,
            source_lineage["camera_records"],
            planning_cameras,
            source_lineage["depth_file_sha256_by_camera"],
        )
        _require(
            replayed_planning_sources["frame_zero_records"]
            == planning_source_replay["frame_zero_records"],
            "planning camera frame-zero inputs changed before publication",
        )
        for snapshot, label in zip(input_snapshots, input_labels):
            _recheck_file_snapshot(snapshot, label=label)
        for budget in CAMERA_BUDGETS:
            for archive_role in ("measurement_archive", "uncertainty_archive"):
                record = archive_records[str(budget)][archive_role]
                path = staging / record["relative_path"]
                _require(
                    _snapshot_regular_file(
                        path,
                        label=f"{budget}-view {archive_role}",
                    ).sha256
                    == record["sha256"],
                    f"{budget}-view {archive_role} changed before publication",
                )
        if output.exists() or output.is_symlink():
            raise ValueError("measurement output appeared before publication")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))


__all__ = [
    "ARTIFACT_KIND",
    "CAMERA_BUDGETS",
    "EXTERNAL_BACKBONE_SEAL_FILENAME",
    "EXTERNAL_PHYSICAL_MANIFEST_FILENAME",
    "MANIFEST_FILENAME",
    "MEASUREMENT_ARRAY_ROLES",
    "MEASUREMENT_ARCHIVE_FILENAME",
    "SCHEMA_VERSION",
    "UNCERTAINTY_ARRAY_ROLES",
    "UNCERTAINTY_ARCHIVE_FILENAME",
    "build_confirmation_nested_measurements",
]
