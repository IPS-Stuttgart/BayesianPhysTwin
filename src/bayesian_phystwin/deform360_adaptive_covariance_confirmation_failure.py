"""Target-free materialization of retained Deform360 technical failures.

The prospective confirmation retains a locked case when either the physical
backend or AllTracker cannot complete.  Retention must not manufacture a
successful observation.  This module therefore materializes:

* an exact persistence physical/backbone package from sealed frame-zero data;
* the frozen nested camera plan with causal RGB-prefix checksums, but no
  AllTracker inference and no dynamic point/covariance observations; and
* a normal target-free case seal whose six trajectory roles are persistence.

The strict visual-hull fallback needs one additional adapter.  Its material
points are not the native Splat identities used by the official outcome
builder.  Immediately after the frozen frame-zero stage, the adapter
deterministically reruns the pinned official ``seed_points_from_splat`` and
replaces only the frame-zero identity archive and its manifest bindings.
The visual-hull diagnostic remains in the manifest as the reason that no
physical twin is admitted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any

import numpy as np

from .deform360_adaptive_covariance_confirmation_external_runtime import (
    DEFORM360_EXECUTION_COMMIT,
    EXTERNAL_EXECUTION_COMMIT,
    activate_confirmation_external_runtime,
    validate_deform360_execution_repository,
    validate_external_execution_repository,
    validate_external_module_provenance,
    validate_two_commit_execution_repository,
)
from .deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    load_confirmation_cohort_lock,
)
from . import deform360_adaptive_covariance_confirmation_measurement as measurement
from .deform360_adaptive_covariance_confirmation_measurement import (
    CAMERA_BUDGETS,
    IDENTITY_PERSISTENCE_ADAPTER_KEY,
    IDENTITY_PERSISTENCE_ADAPTER_KIND,
    IDENTITY_PERSISTENCE_POLICY,
    RETAINED_FAILURE_CAMERA_ACCOUNTING,
    RETAINED_MEASUREMENT_FAILURE_CODES,
    RETAINED_MEASUREMENT_FAILURE_STATUS,
)
from .deform360_adaptive_covariance_confirmation_prediction import (
    seal_retained_confirmation_failure,
)
from .deform360_adaptive_covariance_rbf import (
    FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
)
from .deform360_held_online_prefix import FRAME_COUNT, UPDATE_FRAMES
from .deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    ALLTRACKER_SOURCE_TREE,
    RawCameraObservationConfig,
    _array_sha256 as raw_array_sha256,
    _causal_selected_camera_inputs,
    _load_calibration,
    frame_zero_camera_support,
    select_nested_frame_zero_observation_plans,
)
from .deform360_raw_camera_uncertainty import RawCameraUncertaintyConfig


FRAME_ZERO_MANIFEST_FILENAME = "frame_zero_reconstruction_manifest.json"
PREDICTION_PREFIX_MANIFEST_FILENAME = "prediction_prefix_manifest.json"
FRAME_ZERO_ARCHIVE_FILENAME = "frame_zero_points.npz"
KNOWN_ACTION_RELATIVE_PATH = Path("known-action") / "robot.npz"
PROCESSED_PREFIX_RELATIVE_PATH = Path("prefix") / "episode_0000"
FRAME_ZERO_SPLAT_RELATIVE_PATH = (
    Path("frame-zero") / "episode_0000" / "splatfacto" / "splat_0.ply"
)
PCD_STAGE_SOURCE_SHA256 = (
    "87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d"
)
_ADAPTER_TRANSACTION_FILENAME = ".identity-persistence-adapter.incomplete"
_STRICT_HULL_SOURCE = "strict-multiview-visual-hull-surface"
_PERSISTENCE_ONLY = "persistence_only"
_EXTERNAL_CASE_KEYS = (
    "case",
    "object_id",
    "episode_id",
    "episode_key",
    "stratum",
    "role",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(
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


def _is_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value != "0" * 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_directory(path: str | Path, *, label: str) -> Path:
    root = Path(path).absolute()
    _require(
        root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root,
        f"{label} is invalid",
    )
    return root


def _absent_directory(path: str | Path, *, label: str) -> Path:
    root = Path(path).absolute()
    _require(
        not root.exists() and not root.is_symlink(),
        f"{label} already exists",
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    _require(
        root.parent.resolve(strict=True) == root.parent,
        f"{label} parent is noncanonical",
    )
    return root


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _external_case_identity(
    lock: Mapping[str, Any],
    case_id: str,
) -> dict[str, Any]:
    identity = measurement._case_identity(lock, case_id)
    return {
        "case": case_id,
        "object_id": identity["object_id"],
        "episode_id": int(identity["episode_id"]),
        "episode_key": (f"{identity['object_id']}/{int(identity['episode_id'])}"),
        "stratum": identity["stratum"],
        "role": "calibration",
    }


def _load_json_snapshot(path: Path, *, label: str) -> tuple[Any, dict[str, Any]]:
    snapshot = measurement._snapshot_regular_file(path, label=label)
    value = measurement._load_json_snapshot(snapshot, label=label)
    return snapshot, value


def _validate_external_stage_manifest(
    value: Mapping[str, Any],
    *,
    kind: str,
    lock: Mapping[str, Any],
    external_identity: Mapping[str, Any],
    label: str,
) -> None:
    _require(
        value.get("artifact_kind") == kind
        and value.get("protocol_id") == PROTOCOL_ID
        and value.get("protocol_config_sha256") == lock["artifact_sha256"]
        and value.get("result_sha256") == _result_sha256(value)
        and all(
            value.get(key) == external_identity[key] for key in _EXTERNAL_CASE_KEYS
        ),
        f"{label} is not bound to this H2 case",
    )


def _load_frame_zero_archive(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, bytes]:
    snapshot = measurement._snapshot_regular_file(
        path,
        label="frame-zero identity archive",
    )
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as stored:
            _require(
                set(stored.files) == {"points_m", "colors"},
                "frame-zero identity archive roles changed",
            )
            points = np.asarray(stored["points_m"]).copy()
            colors = np.asarray(stored["colors"]).copy()
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("frame-zero identity archive is invalid") from error
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 16
        and colors.shape == points.shape
        and np.issubdtype(points.dtype, np.floating)
        and np.issubdtype(colors.dtype, np.floating)
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(colors)),
        "frame-zero identities must have finite floating shape (N, 3), N > 16",
    )
    return points, colors, snapshot.payload


def _identity_marker_shape(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    required = {
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
    return (
        set(value) == required
        and value.get("schema_version") == 1
        and value.get("artifact_kind") == IDENTITY_PERSISTENCE_ADAPTER_KIND
        and value.get("policy") == IDENTITY_PERSISTENCE_POLICY
        and value.get("deform360_revision") == DEFORM360_EXECUTION_COMMIT
        and value.get("pcd_stage_source_sha256") == PCD_STAGE_SOURCE_SHA256
        and value.get("physical_twin_admitted") is False
    )


def confirmation_frame_zero_physical_policy(
    manifest: Mapping[str, Any],
    *,
    original_policy: Callable[[Mapping[str, Any]], str],
) -> str:
    """Return persistence for a strictly formed identity-adapter marker."""

    marker = manifest.get(IDENTITY_PERSISTENCE_ADAPTER_KEY)
    if marker is None:
        return original_policy(manifest)
    _require(
        _identity_marker_shape(marker)
        and manifest.get("material_point_source") == IDENTITY_PERSISTENCE_POLICY
        and manifest.get("physical_policy") == _PERSISTENCE_ONLY
        and marker["adapted_material"].get("point_count")
        == manifest.get("material_point_count")
        and marker["adapted_material"].get("array_sha256")
        == manifest.get("material_identity_sha256")
        and marker["adapted_material"].get("file_sha256")
        == manifest.get("outputs_sha256", {}).get("frame_zero_points"),
        "identity-persistence frame-zero marker is inconsistent",
    )
    return _PERSISTENCE_ONLY


def _import_pinned_pcd_stage(repository: Path) -> ModuleType:
    expected = repository / "deform360" / "processing" / "pcd_stage.py"
    _require(
        expected.is_file()
        and not expected.is_symlink()
        and _file_sha256(expected) == PCD_STAGE_SOURCE_SHA256,
        "official Deform360 pcd_stage changed",
    )
    inserted = str(repository) not in sys.path
    if inserted:
        sys.path.insert(0, str(repository))
    try:
        module = importlib.import_module("deform360.processing.pcd_stage")
    finally:
        if inserted:
            sys.path.remove(str(repository))
    source = Path(str(module.__file__)).resolve(strict=True)
    _require(
        source == expected.resolve(strict=True), "pcd_stage import escaped the pin"
    )
    return module


def _validated_staged_frame_zero(
    lock_path: str | Path,
    h2_commit: str,
    staged_case_dir: str | Path,
    *,
    expected_h1: str | None,
    require_identity_adapter: bool = False,
) -> dict[str, Any]:
    _require(_is_sha1(h2_commit), "H2 commit is invalid")
    lock_snapshot = measurement._snapshot_regular_file(lock_path, label="H2 lock")
    lock = load_confirmation_cohort_lock(
        lock_snapshot.path,
        expected_implementation_commit_h1=expected_h1,
    )
    h1 = lock["two_commit_freeze"]["implementation_commit_h1"]
    _require(h2_commit != h1, "H2 must differ from implementation H1")
    staged = _canonical_directory(staged_case_dir, label="staged case")
    case_id = staged.name
    external_identity = _external_case_identity(lock, case_id)
    _require(
        not (staged / _ADAPTER_TRANSACTION_FILENAME).exists(),
        "incomplete identity-adapter transaction is present",
    )
    prefix_snapshot, prefix = _load_json_snapshot(
        staged / PREDICTION_PREFIX_MANIFEST_FILENAME,
        label="prediction-prefix manifest",
    )
    frame_snapshot, frame = _load_json_snapshot(
        staged / FRAME_ZERO_MANIFEST_FILENAME,
        label="frame-zero manifest",
    )
    _validate_external_stage_manifest(
        prefix,
        kind="Deform360BiasAwarePredictionPrefix",
        lock=lock,
        external_identity=external_identity,
        label="prediction-prefix manifest",
    )
    _validate_external_stage_manifest(
        frame,
        kind="Deform360BiasAwareFrameZeroReconstruction",
        lock=lock,
        external_identity=external_identity,
        label="frame-zero manifest",
    )
    prefix_boundary = prefix.get("information_boundary", {})
    frame_boundary = frame.get("information_boundary", {})
    _require(
        prefix_boundary.get("source_object_frames_after_prefix_read") is False
        and prefix_boundary.get("future_dense_reconstruction_read") is False
        and prefix_boundary.get("future_particle_tracks_read") is False
        and prefix_boundary.get("target_metric_read") is False
        and frame_boundary.get("object_observation_frames_used") == [0]
        and frame_boundary.get("future_object_rgb_read") is False
        and frame_boundary.get("future_dense_reconstruction_read") is False
        and frame_boundary.get("future_particle_tracks_read") is False
        and frame_boundary.get("target_metric_read") is False,
        "staged case crossed the target-free prediction boundary",
    )
    _require(
        prefix.get("inputs_sha256", {}).get("protocol") == lock_snapshot.sha256
        and frame.get("inputs_sha256", {}).get("prediction_prefix_manifest")
        == prefix_snapshot.sha256,
        "staged manifests bind another lock or prediction prefix",
    )
    geometry_path = staged / FRAME_ZERO_ARCHIVE_FILENAME
    action_path = staged / KNOWN_ACTION_RELATIVE_PATH
    points, colors, _ = _load_frame_zero_archive(geometry_path)
    geometry_snapshot = measurement._snapshot_regular_file(
        geometry_path,
        label="frame-zero identity archive",
    )
    action_snapshot = measurement._snapshot_regular_file(
        action_path,
        label="known action archive",
    )
    _require(
        frame.get("outputs_sha256", {}).get("frame_zero_points")
        == geometry_snapshot.sha256
        and prefix.get("staged_robot_sha256", {}).get("known_action")
        == action_snapshot.sha256
        and frame.get("material_point_count") == len(points)
        and frame.get("material_identity_sha256")
        == measurement._external_array_sha256(points),
        "staged frame-zero material binding changed",
    )
    marker = frame.get(IDENTITY_PERSISTENCE_ADAPTER_KEY)
    if marker is not None:
        _require(
            _identity_marker_shape(marker)
            and marker.get("implementation_commit_h1") == h1
            and marker.get("cohort_lock_commit_h2") == h2_commit
            and marker.get("cohort_lock_artifact_sha256") == lock["artifact_sha256"]
            and marker.get("adapter_source_sha256") == _file_sha256(__file__)
            and marker.get("adapted_material")
            == {
                "source": IDENTITY_PERSISTENCE_POLICY,
                "point_count": len(points),
                "array_sha256": measurement._external_array_sha256(points),
                "file_sha256": geometry_snapshot.sha256,
            }
            and marker.get("preserved_fallback_diagnostics_sha256")
            == hashlib.sha256(
                _canonical_bytes(frame.get("fallback_diagnostics"))
            ).hexdigest()
            and confirmation_frame_zero_physical_policy(
                frame,
                original_policy=lambda _value: "invalid",
            )
            == _PERSISTENCE_ONLY,
            "identity-persistence adapter binding changed",
        )
    if require_identity_adapter:
        _require(marker is not None, "frame-zero identity adapter is absent")
    for snapshot, label in (
        (lock_snapshot, "H2 lock"),
        (prefix_snapshot, "prediction-prefix manifest"),
        (frame_snapshot, "frame-zero manifest"),
        (geometry_snapshot, "frame-zero identity archive"),
        (action_snapshot, "known action archive"),
    ):
        measurement._recheck_file_snapshot(snapshot, label=label)
    return {
        "lock": lock,
        "lock_snapshot": lock_snapshot,
        "staged": staged,
        "case_id": case_id,
        "external_identity": external_identity,
        "prefix": prefix,
        "prefix_snapshot": prefix_snapshot,
        "frame_zero": frame,
        "frame_zero_snapshot": frame_snapshot,
        "geometry_path": geometry_path,
        "geometry_snapshot": geometry_snapshot,
        "action_path": action_path,
        "action_snapshot": action_snapshot,
        "points": points,
        "colors": colors,
    }


def validate_original_splat_identity_persistence_manifest(
    lock_path: str | Path,
    h2_commit: str,
    staged_case_dir: str | Path,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Replay every lock, source, array, and marker checksum after adaptation."""

    return _validated_staged_frame_zero(
        lock_path,
        h2_commit,
        staged_case_dir,
        expected_h1=expected_h1,
        require_identity_adapter=True,
    )["frame_zero"]


def _validated_native_original_splat_frame_zero(
    lock_path: str | Path,
    h2_commit: str,
    staged_case_dir: str | Path,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    context = _validated_staged_frame_zero(
        lock_path,
        h2_commit,
        staged_case_dir,
        expected_h1=expected_h1,
    )
    frame = context["frame_zero"]
    source = frame.get("material_point_source", "original-splat")
    marker = frame.get(IDENTITY_PERSISTENCE_ADAPTER_KEY)
    _require(
        len(context["points"]) > 16
        and (
            (source == "original-splat" and marker is None)
            or (source == IDENTITY_PERSISTENCE_POLICY and marker is not None)
        ),
        "retained materialization requires native original-Splat identities",
    )
    return context


def validate_native_original_splat_frame_zero(
    lock_path: str | Path,
    h2_commit: str,
    staged_case_dir: str | Path,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Require native original-Splat identities before retained materialization."""

    return _validated_native_original_splat_frame_zero(
        lock_path,
        h2_commit,
        staged_case_dir,
        expected_h1=expected_h1,
    )["frame_zero"]


def _temporary_file_bytes(parent: Path, *, prefix: str, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=parent)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _restore_file(path: Path, payload: bytes) -> None:
    temporary = _temporary_file_bytes(
        path.parent,
        prefix=f".{path.name}.restore-",
        payload=payload,
    )
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def adapt_frame_zero_original_splat_identity_persistence(
    lock_path: str | Path,
    h2_commit: str,
    staged_case_dir: str | Path,
    deform360_repository: str | Path,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Replace strict-hull identities with pinned native Splat seed identities.

    Original-Splat frame-zero cases are returned unchanged.  A strict-hull
    fallback is adapted exactly once.  A durable transaction marker makes an
    interrupted two-file replacement fail closed on every downstream loader.
    """

    deform360 = Path(deform360_repository).absolute()
    validate_deform360_execution_repository(deform360)
    context = _validated_staged_frame_zero(
        lock_path,
        h2_commit,
        staged_case_dir,
        expected_h1=expected_h1,
    )
    frame = context["frame_zero"]
    marker = frame.get(IDENTITY_PERSISTENCE_ADAPTER_KEY)
    if marker is not None:
        validate_original_splat_identity_persistence_manifest(
            lock_path,
            h2_commit,
            staged_case_dir,
            expected_h1=expected_h1,
        )
        return frame
    source = frame.get("material_point_source", "original-splat")
    if source == "original-splat":
        return frame
    _require(
        source == _STRICT_HULL_SOURCE
        and frame.get("physical_policy") == _PERSISTENCE_ONLY
        and isinstance(frame.get("fallback_diagnostics"), Mapping),
        "only the frozen strict-hull persistence fallback may be adapted",
    )
    pcd_stage = _import_pinned_pcd_stage(deform360)
    splat_path = context["staged"] / FRAME_ZERO_SPLAT_RELATIVE_PATH
    splat_snapshot = measurement._snapshot_regular_file(
        splat_path,
        label="sealed frame-zero Splat",
    )
    _require(
        frame.get("outputs_sha256", {}).get("frame_zero_splat")
        == splat_snapshot.sha256,
        "frame-zero manifest binds another Splat",
    )
    crop = float(pcd_stage.CROP_HALF_EXTENT_M)
    seed_count = int(pcd_stage.SEED_POINT_COUNT)
    points, colors = pcd_stage.seed_points_from_splat(
        splat_path,
        crop_half_extent_m=crop,
        seed_count=seed_count,
        rng_seed=0,
    )
    points = np.asarray(points)
    colors = np.asarray(colors)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 16
        and colors.shape == points.shape
        and np.issubdtype(points.dtype, np.floating)
        and np.issubdtype(colors.dtype, np.floating)
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(colors)),
        "pinned Splat seed identities have invalid shape or support",
    )
    archive_buffer = io.BytesIO()
    np.savez_compressed(archive_buffer, points_m=points, colors=colors)
    archive_payload = archive_buffer.getvalue()
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    material_sha256 = measurement._external_array_sha256(points)
    h1 = context["lock"]["two_commit_freeze"]["implementation_commit_h1"]
    adapted = json.loads(_canonical_bytes(frame))
    adapted["material_point_source"] = IDENTITY_PERSISTENCE_POLICY
    adapted["physical_policy"] = _PERSISTENCE_ONLY
    adapted["material_point_count"] = len(points)
    adapted["material_identity_sha256"] = material_sha256
    adapted["outputs_sha256"]["frame_zero_points"] = archive_sha256
    adapted[IDENTITY_PERSISTENCE_ADAPTER_KEY] = {
        "schema_version": 1,
        "artifact_kind": IDENTITY_PERSISTENCE_ADAPTER_KIND,
        "policy": IDENTITY_PERSISTENCE_POLICY,
        "implementation_commit_h1": h1,
        "cohort_lock_commit_h2": h2_commit,
        "cohort_lock_artifact_sha256": context["lock"]["artifact_sha256"],
        "adapter_source_sha256": _file_sha256(__file__),
        "deform360_revision": DEFORM360_EXECUTION_COMMIT,
        "pcd_stage_source_sha256": PCD_STAGE_SOURCE_SHA256,
        "frame_zero_splat_file_sha256": splat_snapshot.sha256,
        "seed_parameters": {
            "crop_half_extent_m": crop,
            "seed_count": seed_count,
            "rng_seed": 0,
        },
        "previous_material": {
            "source": source,
            "point_count": len(context["points"]),
            "array_sha256": measurement._external_array_sha256(context["points"]),
            "file_sha256": context["geometry_snapshot"].sha256,
        },
        "adapted_material": {
            "source": IDENTITY_PERSISTENCE_POLICY,
            "point_count": len(points),
            "array_sha256": material_sha256,
            "file_sha256": archive_sha256,
        },
        "preserved_fallback_diagnostics_sha256": hashlib.sha256(
            _canonical_bytes(frame["fallback_diagnostics"])
        ).hexdigest(),
        "physical_twin_admitted": False,
    }
    adapted["result_sha256"] = _result_sha256(adapted)
    manifest_payload = (
        json.dumps(adapted, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    staged = context["staged"]
    geometry_path = context["geometry_path"]
    manifest_path = staged / FRAME_ZERO_MANIFEST_FILENAME
    transaction_path = staged / _ADAPTER_TRANSACTION_FILENAME
    _require(
        not transaction_path.exists() and not transaction_path.is_symlink(),
        "identity-adapter transaction is already present",
    )
    archive_temporary = _temporary_file_bytes(
        staged,
        prefix=f".{FRAME_ZERO_ARCHIVE_FILENAME}.adapt-",
        payload=archive_payload,
    )
    manifest_temporary = _temporary_file_bytes(
        staged,
        prefix=f".{FRAME_ZERO_MANIFEST_FILENAME}.adapt-",
        payload=manifest_payload,
    )
    _write_json_exclusive(
        transaction_path,
        {
            "artifact_kind": IDENTITY_PERSISTENCE_ADAPTER_KIND,
            "case": context["case_id"],
            "archive_file_sha256": archive_sha256,
            "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        },
    )
    _fsync_directory(staged)
    archive_replaced = False
    manifest_replaced = False
    try:
        os.replace(archive_temporary, geometry_path)
        archive_replaced = True
        os.replace(manifest_temporary, manifest_path)
        manifest_replaced = True
        _fsync_directory(staged)
        transaction_path.unlink()
        _fsync_directory(staged)
        validate_original_splat_identity_persistence_manifest(
            lock_path,
            h2_commit,
            staged,
            expected_h1=expected_h1,
        )
    except BaseException:
        archive_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
        if archive_replaced:
            _restore_file(geometry_path, context["geometry_snapshot"].payload)
        if manifest_replaced:
            _restore_file(manifest_path, context["frame_zero_snapshot"].payload)
        transaction_path.unlink(missing_ok=True)
        _fsync_directory(staged)
        raise
    return adapted


def _build_prediction_only_bundle_any_point_count(
    physical_module: ModuleType,
    artifacts_module: ModuleType,
    geometry_path: Path,
    action_path: Path,
    output_path: Path,
    *,
    external_identity: Mapping[str, Any],
) -> dict[str, Any]:
    with np.load(geometry_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == {"points_m", "colors"},
            "frame-zero geometry roles changed",
        )
        points = np.asarray(stored["points_m"], dtype=np.float32)
        colors = np.asarray(stored["colors"], dtype=np.float32)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 16
        and colors.shape == points.shape
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(colors)),
        "persistence bundle requires finite frame-zero identities, N > 16",
    )
    controllers, action = physical_module.load_controller_trajectory(action_path)
    object_points = np.repeat(points[None], FRAME_COUNT, axis=0)
    object_colors = np.repeat(colors[None], FRAME_COUNT, axis=0)
    observed = np.ones(object_points.shape[:2], dtype=bool)
    marker = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case": external_identity["case"],
        "object_id": external_identity["object_id"],
        "episode_id": int(external_identity["episode_id"]),
        "object_observation_frames_used": [0],
        "known_future_robot_trajectory_used": True,
        "future_object_observations_present": False,
        "future_tactile_used": False,
        "frame_zero_geometry_sha256": artifacts_module.file_sha256(geometry_path),
        "known_action_sha256": artifacts_module.file_sha256(action_path),
        "action_window": action,
    }
    payload = {
        "object_points": object_points,
        "object_colors": object_colors,
        "object_visibilities": observed,
        "object_motions_valid": observed.copy(),
        "controller_points": controllers,
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
        "prediction_only_input": marker,
    }
    with output_path.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "frame_count": FRAME_COUNT,
        "point_count": len(points),
        "controller_point_count": int(controllers.shape[1]),
        "frame_zero_points_sha256": artifacts_module.array_sha256(points),
        "controller_trajectory_sha256": artifacts_module.array_sha256(controllers),
        "output_sha256": artifacts_module.file_sha256(output_path),
        "action_window": action,
    }


def _persistence_backbone_arrays(points: np.ndarray) -> dict[str, np.ndarray]:
    initial = np.asarray(points, dtype=np.float32)
    _require(
        initial.ndim == 2
        and initial.shape[1] == 3
        and len(initial) > 16
        and np.all(np.isfinite(initial)),
        "persistence backbone requires finite frame-zero identities, N > 16",
    )
    persistence = np.repeat(initial[None], FRAME_COUNT, axis=0)
    return {
        "prediction_m": persistence.copy(),
        "persistence_m": persistence.copy(),
        "driven_readout_m": persistence.copy(),
        "zero_action_readout_m": persistence.copy(),
        "action_support": np.zeros(len(initial), dtype=np.float32),
        "frame_zero_points_m": initial.copy(),
    }


def build_retained_failure_physical_backbone(
    adapter_repository: str | Path,
    external_execution_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    staged_case_dir: str | Path,
    physical_work_dir: str | Path,
    backbone_dir: str | Path,
    failure_code: str,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Materialize and seal a checksum-bound persistence physical package."""

    _require(
        failure_code in RETAINED_MEASUREMENT_FAILURE_CODES,
        "retained failure code is outside the frozen vocabulary",
    )
    adapter = _canonical_directory(adapter_repository, label="adapter repository")
    execution = _canonical_directory(
        external_execution_repository,
        label="external execution repository",
    )
    validate_external_execution_repository(execution)
    staged_context = _validated_native_original_splat_frame_zero(
        lock_path,
        h2_commit,
        staged_case_dir,
        expected_h1=expected_h1,
    )
    _require(staged_context["case_id"] == case_id, "staged case ID changed")
    h1 = staged_context["lock"]["two_commit_freeze"]["implementation_commit_h1"]
    validate_two_commit_execution_repository(
        adapter,
        lock_path,
        h1_commit=h1,
        h2_commit=h2_commit,
    )
    work = _absent_directory(physical_work_dir, label="physical work directory")
    backbone = _absent_directory(backbone_dir, label="backbone directory")
    inputs = (
        Path(lock_path).absolute(),
        staged_context["staged"],
        adapter,
        execution,
    )
    _require(
        not _paths_overlap(work, backbone)
        and all(
            not _paths_overlap(output, source)
            for output in (work, backbone)
            for source in inputs
        ),
        "retained physical output overlaps an input or another output",
    )
    try:
        work.mkdir()
        prediction_input = work / "prediction_only_input.pkl"
        prediction_summary_path = work / "prediction_only_input.json"
        archive_path = work / "prediction.npz"
        physical_manifest_path = work / measurement.EXTERNAL_PHYSICAL_MANIFEST_FILENAME
        with activate_confirmation_external_runtime(execution) as modules:
            validate_external_module_provenance(execution)
            physical_module = modules["physical"]
            artifacts_module = modules["artifacts"]
            summary = _build_prediction_only_bundle_any_point_count(
                physical_module,
                artifacts_module,
                staged_context["geometry_path"],
                staged_context["action_path"],
                prediction_input,
                external_identity=staged_context["external_identity"],
            )
            _write_json_exclusive(prediction_summary_path, summary)
            arrays = _persistence_backbone_arrays(staged_context["points"])
            physical_manifest = physical_module.write_physical_artifacts(
                archive_path,
                physical_manifest_path,
                arrays,
                case_record=staged_context["external_identity"],
                protocol_config_sha256=staged_context["lock"]["artifact_sha256"],
                physical_mode="persistence_fallback",
                input_files={
                    "protocol": Path(lock_path).absolute(),
                    "prediction_prefix_manifest": staged_context[
                        "prefix_snapshot"
                    ].path,
                    "frame_zero_manifest": staged_context["frame_zero_snapshot"].path,
                    "frame_zero_geometry": staged_context["geometry_path"],
                    "known_action": staged_context["action_path"],
                    "prediction_only_input": prediction_input,
                    "prediction_only_summary": prediction_summary_path,
                },
                runtime_provenance={
                    "external_execution_commit": EXTERNAL_EXECUTION_COMMIT,
                    "adapter_implementation_commit_h1": h1,
                    "cohort_lock_commit_h2": h2_commit,
                    "materializer_source_sha256": _file_sha256(__file__),
                    "physical_runtime_required": False,
                    "runtime_seconds": {
                        "automatic_twin": 0.0,
                        "warp": 0.0,
                    },
                },
                fallback_diagnostics={
                    "reason": (
                        "retained_technical_failure_persistence_materialization"
                    ),
                    "failure_code": failure_code,
                    "failed_backend_artifact_consumed": False,
                    "warp_attempted": False,
                    "state_update_available": False,
                    "identity_policy": staged_context["frame_zero"].get(
                        "material_point_source",
                        "original-splat",
                    ),
                },
            )
            _require(
                physical_manifest.get("result_sha256")
                == _result_sha256(physical_manifest),
                "retained physical manifest self-checksum changed",
            )
            seal = artifacts_module.build_prospective_backbone_seal(
                lock_path,
                backbone,
                object_id=staged_context["external_identity"]["object_id"],
                episode_id=staged_context["external_identity"]["episode_id"],
                physical_archive=archive_path,
                physical_manifest=physical_manifest_path,
            )
        physical_archive = backbone / "physical_prediction.npz"
        copied_manifest = backbone / measurement.EXTERNAL_PHYSICAL_MANIFEST_FILENAME
        copied_seal = backbone / measurement.EXTERNAL_BACKBONE_SEAL_FILENAME
        archive_snapshot = measurement._snapshot_regular_file(
            physical_archive,
            label="retained physical archive",
        )
        arrays = measurement._physical_arrays(archive_snapshot)
        provenance = measurement._validate_external_physical_provenance(
            lock=staged_context["lock"],
            identity=measurement._case_identity(
                staged_context["lock"],
                case_id,
            ),
            archive_snapshot=archive_snapshot,
            manifest_snapshot=measurement._snapshot_regular_file(
                copied_manifest,
                label="retained physical manifest",
            ),
            seal_snapshot=measurement._snapshot_regular_file(
                copied_seal,
                label="retained backbone seal",
            ),
            arrays=arrays,
        )
        _require(
            np.array_equal(
                arrays["prediction_m"],
                arrays["persistence_m"],
            ),
            "retained physical package is not bit-exact persistence",
        )
    except BaseException:
        if backbone.exists() or backbone.is_symlink():
            shutil.rmtree(backbone, ignore_errors=True)
        if work.exists() or work.is_symlink():
            shutil.rmtree(work, ignore_errors=True)
        raise
    return {
        "case_id": case_id,
        "failure_code": failure_code,
        "physical_work_dir": str(work),
        "backbone_dir": str(backbone),
        "physical_archive": str(physical_archive),
        "physical_manifest": str(copied_manifest),
        "physical_prediction_seal": str(copied_seal),
        "physical_backbone": provenance,
        "external_backbone_seal_result_sha256": seal["result_sha256"],
    }


def _decode_rgb_prefix(
    video_path: Path,
    update_frame: int,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - remote dependency
        raise RuntimeError(
            "OpenCV is required for causal RGB-prefix binding"
        ) from error
    capture = cv2.VideoCapture(str(video_path))
    digest = hashlib.sha256()
    image_shape: tuple[int, int] | None = None
    try:
        for frame_index in range(update_frame + 1):
            okay, bgr = capture.read()
            _require(
                okay and bgr is not None,
                f"cannot read causal frame {frame_index}: {video_path}",
            )
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            current_shape = (int(rgb.shape[0]), int(rgb.shape[1]))
            if image_shape is None:
                image_shape = current_shape
            _require(
                current_shape == image_shape,
                "causal RGB prefix changed image shape",
            )
            digest.update(str(rgb.dtype).encode("ascii"))
            digest.update(np.asarray(rgb.shape, dtype=np.int64).tobytes())
            digest.update(rgb.tobytes())
    finally:
        capture.release()
    _require(image_shape is not None, "causal RGB prefix is empty")
    return {
        "prefix_frame_range_half_open": [0, update_frame + 1],
        "maximum_video_frame_read": update_frame,
        "decoded_frame_count": update_frame + 1,
        "decoded_rgb_prefix_sha256": digest.hexdigest(),
        "original_image_shape": list(image_shape),
    }


def build_retained_failure_nested_measurements(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    physical_archive: str | Path,
    processed_episode_dir: str | Path,
    output_dir: str | Path,
    failure_code: str,
    *,
    physical_manifest: str | Path,
    physical_prediction_seal: str | Path,
    staged_case_dir: str | Path,
    source_custody_seal: str | Path,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Publish exact nested plans with unavailable dynamic observations."""

    _require(
        failure_code in RETAINED_MEASUREMENT_FAILURE_CODES,
        "retained failure code is outside the frozen vocabulary",
    )
    staged_context = _validated_native_original_splat_frame_zero(
        lock_path,
        h2_commit,
        staged_case_dir,
        expected_h1=expected_h1,
    )
    _require(staged_context["case_id"] == case_id, "staged case ID changed")
    processed = _canonical_directory(
        processed_episode_dir,
        label="processed prefix episode",
    )
    _require(
        processed == staged_context["staged"] / PROCESSED_PREFIX_RELATIVE_PATH,
        "processed prefix is outside the sealed staged case",
    )
    output = _absent_directory(output_dir, label="nested measurement output")
    _require(
        output.name == case_id,
        "nested measurement directory must use the exact case ID",
    )
    physical_snapshot = measurement._snapshot_regular_file(
        physical_archive,
        label="physical archive",
    )
    physical_manifest_snapshot = measurement._snapshot_regular_file(
        physical_manifest,
        label="external physical manifest",
    )
    physical_seal_snapshot = measurement._snapshot_regular_file(
        physical_prediction_seal,
        label="external backbone seal",
    )
    _require(
        physical_manifest_snapshot.path.name
        == measurement.EXTERNAL_PHYSICAL_MANIFEST_FILENAME
        and physical_seal_snapshot.path.name
        == measurement.EXTERNAL_BACKBONE_SEAL_FILENAME,
        "external physical provenance filenames changed",
    )
    physical_arrays = measurement._physical_arrays(physical_snapshot)
    physical_provenance = measurement._validate_external_physical_provenance(
        lock=staged_context["lock"],
        identity=measurement._case_identity(
            staged_context["lock"],
            case_id,
        ),
        archive_snapshot=physical_snapshot,
        manifest_snapshot=physical_manifest_snapshot,
        seal_snapshot=physical_seal_snapshot,
        arrays=physical_arrays,
    )
    source_lineage = measurement._validated_source_stage_lineage(
        lock=staged_context["lock"],
        lock_snapshot=staged_context["lock_snapshot"],
        identity=measurement._case_identity(
            staged_context["lock"],
            case_id,
        ),
        h2_commit=h2_commit,
        physical_manifest_snapshot=physical_manifest_snapshot,
        processed_episode_dir=processed,
        source_custody_seal=source_custody_seal,
    )
    _require(
        np.array_equal(
            physical_arrays["prediction_m"],
            physical_arrays["persistence_m"],
        ),
        "retained measurement package requires a persistence physical archive",
    )
    prior = physical_arrays["prediction_m"]
    frame_zero = physical_arrays["frame_zero_points_m"]
    _require(
        measurement._external_array_sha256(frame_zero)
        == measurement._external_array_sha256(staged_context["points"]),
        "physical and staged frame-zero identities differ",
    )
    intrinsics_snapshot = measurement._snapshot_regular_file(
        processed / "undistorted_intrinsics.npy",
        label="intrinsics",
    )
    extrinsics_snapshot = measurement._snapshot_regular_file(
        processed / "extrinsics.npy",
        label="extrinsics",
    )
    intrinsics, extrinsics = _load_calibration(processed)
    measurement._recheck_file_snapshot(intrinsics_snapshot, label="intrinsics")
    measurement._recheck_file_snapshot(extrinsics_snapshot, label="extrinsics")
    planning_cameras = source_lineage["planning_cameras"]
    measurement._validate_calibration_camera_panel(
        intrinsics,
        extrinsics,
        planning_cameras,
        label="retained calibration",
    )
    observation_config = RawCameraObservationConfig(selected_camera_count=8)
    uncertainty_config = RawCameraUncertaintyConfig()
    cameras, support, projected = frame_zero_camera_support(
        frame_zero,
        processed,
        intrinsics,
        extrinsics,
        depth_tolerance_m=observation_config.frame_zero_depth_tolerance_m,
    )
    _require(
        tuple(cameras) == planning_cameras,
        "retained frame-zero planner camera panel differs from the sealed planning panel",
    )
    nested = select_nested_frame_zero_observation_plans(
        frame_zero,
        cameras,
        support,
        projected,
        extrinsics,
        config=observation_config,
    )
    centers = np.asarray(nested["center_ids"], dtype=np.int64)
    candidates = np.asarray(nested["candidate_ids"], dtype=np.int64)
    _require(
        centers.shape == (16,)
        and len(np.unique(centers)) == 16
        and np.all((0 <= centers) & (centers < prior.shape[1])),
        "retained nested plan did not produce exact 16 center IDs",
    )
    plans = nested["prefix_plans"]
    selected = {
        budget: tuple(plans[budget]["selected_cameras"]) for budget in CAMERA_BUDGETS
    }
    _require(
        selected[8][:4] == selected[4],
        "retained nested camera plan changed",
    )
    arrays = {
        budget: measurement._empty_budget_arrays(
            prior,
            frame_zero,
            candidates,
        )
        for budget in CAMERA_BUDGETS
    }
    update_records: list[dict[str, Any]] = []
    plan8 = plans[8]
    for frame in UPDATE_FRAMES:
        tracker_records: list[dict[str, Any]] = []

        def decode(
            cameras_to_decode: Sequence[str],
            *,
            role: str,
            four_view_decision_materialized: bool,
        ) -> None:
            for camera in cameras_to_decode:
                query_ids = np.asarray(
                    plan8["query_ids"][camera],
                    dtype=np.int64,
                )
                prefix = _decode_rgb_prefix(
                    processed / camera / "undistorted.mp4",
                    frame,
                )
                tracker_records.append(
                    {
                        **prefix,
                        "camera": camera,
                        "query_ids": query_ids.tolist(),
                        "execution_role": role,
                        "execution_index_within_update": len(tracker_records),
                        "four_view_decision_already_materialized": (
                            four_view_decision_materialized
                        ),
                        "camera_stream_attempted": True,
                        "tracker_inference_executed": False,
                        "dynamic_observation_available": False,
                        "failure_code": failure_code,
                    }
                )

        decode(
            selected[4],
            role="adaptive_first_four",
            four_view_decision_materialized=False,
        )
        reliability4 = measurement._reliability_record(
            arrays[4],
            centers,
            frame,
            frame_zero,
        )
        _require(reliability4["reliable"] is False, "empty four-view route accepted")
        decode(
            selected[8][4:],
            role="adaptive_eight_escalation",
            four_view_decision_materialized=True,
        )
        reliability8 = measurement._reliability_record(
            arrays[8],
            centers,
            frame,
            frame_zero,
        )
        _require(reliability8["reliable"] is False, "empty eight-view route accepted")
        center_records = {
            str(budget): [
                {
                    "center_id": int(center_id),
                    "measurement_available": False,
                    "covariance_valid": False,
                    "decision": ("retained_technical_failure_measurement_unavailable"),
                    "failure_code": failure_code,
                }
                for center_id in centers
            ]
            for budget in CAMERA_BUDGETS
        }
        update_records.append(
            {
                "frame": int(frame),
                "four_view_decision_materialized_before_shadow_extra_four": True,
                "four_view_reliable_before_shadow": False,
                "offline_shadow_extra_four_tracked": False,
                "adaptive_route": "physical_prior_fallback",
                "adaptive_charged_camera_streams": 8,
                "budget_reliability": {
                    "4": reliability4,
                    "8": reliability8,
                },
                "tracker": tracker_records,
                "centers": center_records,
            }
        )
    selected_inputs = _causal_selected_camera_inputs(
        processed,
        selected[8],
        update_records,
    )
    planning_source_replay = measurement._validate_planning_camera_source_bindings(
        processed,
        selected_inputs,
        source_lineage["camera_records"],
        planning_cameras,
        source_lineage["depth_file_sha256_by_camera"],
    )
    planning_source_snapshots = planning_source_replay["snapshots"]
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        archive_records = {
            str(budget): measurement._write_budget_archives(
                staging / f"budget-{budget}",
                arrays=arrays[budget],
                centers=centers,
                selected_cameras=selected[budget],
            )
            for budget in CAMERA_BUDGETS
        }
        payload: dict[str, Any] = {
            "schema_version": measurement.SCHEMA_VERSION,
            "artifact_kind": measurement.ARTIFACT_KIND,
            "protocol_id": PROTOCOL_ID,
            "case_identity": measurement._case_identity(
                staged_context["lock"],
                case_id,
            ),
            "lock_binding": {
                "implementation_commit_h1": staged_context["lock"]["two_commit_freeze"][
                    "implementation_commit_h1"
                ],
                "cohort_lock_commit_h2": h2_commit,
                "cohort_lock_artifact_sha256": staged_context["lock"][
                    "artifact_sha256"
                ],
                "cohort_lock_file_sha256": staged_context["lock_snapshot"].sha256,
            },
            "config": {
                "observation": asdict(observation_config),
                "uncertainty": asdict(uncertainty_config),
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
                    "frame_zero_array_sha256": raw_array_sha256(frame_zero),
                },
                "intrinsics_sha256": intrinsics_snapshot.sha256,
                "extrinsics_sha256": extrinsics_snapshot.sha256,
                "selected_camera_prefixes_and_frame_zero": selected_inputs,
                "source_stage_lineage": source_lineage["manifest_record"],
                "retained_failure_source": {
                    "failure_code": failure_code,
                    "prediction_prefix_manifest": {
                        "path": str(staged_context["prefix_snapshot"].path),
                        "file_sha256": staged_context["prefix_snapshot"].sha256,
                        "result_sha256": staged_context["prefix"]["result_sha256"],
                    },
                    "frame_zero_manifest": {
                        "path": str(staged_context["frame_zero_snapshot"].path),
                        "file_sha256": staged_context["frame_zero_snapshot"].sha256,
                        "result_sha256": staged_context["frame_zero"]["result_sha256"],
                    },
                    "processed_prefix_episode": {
                        "path": str(processed),
                        "intrinsics_file_sha256": intrinsics_snapshot.sha256,
                        "extrinsics_file_sha256": extrinsics_snapshot.sha256,
                    },
                    "dynamic_point_observations_available": False,
                },
            },
            "tracker": {
                "name": "AllTracker",
                "molmomotion_revision": ALLTRACKER_MOLMOMOTION_REVISION,
                "source_tree": ALLTRACKER_SOURCE_TREE,
                "runtime_source_sha256": ALLTRACKER_RUNTIME_SOURCE_SHA256,
                "checkpoint_sha256": ALLTRACKER_CHECKPOINT_SHA256,
                "device": "not-executed",
                "execution_status": RETAINED_MEASUREMENT_FAILURE_STATUS,
                "failure_code": failure_code,
                "inference_executed": False,
            },
            "updates": update_records,
            "outputs": archive_records,
            "camera_accounting": dict(RETAINED_FAILURE_CAMERA_ACCOUNTING),
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
        payload["artifact_sha256"] = measurement._canonical_sha256(payload)
        _write_json_exclusive(staging / measurement.MANIFEST_FILENAME, payload)
        _require(
            _causal_selected_camera_inputs(
                processed,
                selected[8],
                update_records,
            )
            == selected_inputs,
            "selected camera frame-zero inputs changed before publication",
        )
        replayed_planning_sources = (
            measurement._validate_planning_camera_source_bindings(
                processed,
                selected_inputs,
                source_lineage["camera_records"],
                planning_cameras,
                source_lineage["depth_file_sha256_by_camera"],
            )
        )
        _require(
            replayed_planning_sources["frame_zero_records"]
            == planning_source_replay["frame_zero_records"],
            "planning camera frame-zero inputs changed before publication",
        )
        for snapshot, label in (
            (staged_context["lock_snapshot"], "H2 lock"),
            (staged_context["prefix_snapshot"], "prediction-prefix manifest"),
            (staged_context["frame_zero_snapshot"], "frame-zero manifest"),
            (staged_context["geometry_snapshot"], "frame-zero identity archive"),
            (physical_snapshot, "physical archive"),
            (physical_manifest_snapshot, "external physical manifest"),
            (physical_seal_snapshot, "external backbone seal"),
            (intrinsics_snapshot, "intrinsics"),
            (extrinsics_snapshot, "extrinsics"),
            *(
                (snapshot, "source-stage manifest")
                for snapshot in source_lineage["snapshots"]
            ),
            *(
                (snapshot, f"{snapshot.path.parent.name} planning source file")
                for snapshot in planning_source_snapshots
            ),
        ):
            measurement._recheck_file_snapshot(snapshot, label=label)
        _require(
            not output.exists() and not output.is_symlink(),
            "nested measurement output appeared before publication",
        )
        os.rename(staging, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return json.loads(
        (output / measurement.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )


def materialize_and_seal_retained_confirmation_failure(
    adapter_repository: str | Path,
    external_execution_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    staged_case_dir: str | Path,
    processed_episode_dir: str | Path,
    source_custody_seal: str | Path,
    physical_work_dir: str | Path,
    backbone_dir: str | Path,
    measurement_output_dir: str | Path,
    case_output_dir: str | Path,
    failure_code: str,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Materialize every target-free compatibility artifact and seal the case."""

    backbone = Path(backbone_dir).absolute()
    measurement_root = Path(measurement_output_dir).absolute()
    case_root = Path(case_output_dir).absolute()
    _require(
        backbone.name == measurement_root.name == case_root.name == case_id,
        "backbone, measurement, and case directories must use the exact case ID",
    )
    physical = build_retained_failure_physical_backbone(
        adapter_repository,
        external_execution_repository,
        lock_path,
        h2_commit,
        case_id,
        staged_case_dir,
        physical_work_dir,
        backbone,
        failure_code,
        expected_h1=expected_h1,
    )
    nested = build_retained_failure_nested_measurements(
        lock_path,
        h2_commit,
        case_id,
        physical["physical_archive"],
        processed_episode_dir,
        measurement_root,
        failure_code,
        physical_manifest=physical["physical_manifest"],
        physical_prediction_seal=physical["physical_prediction_seal"],
        staged_case_dir=staged_case_dir,
        source_custody_seal=source_custody_seal,
        expected_h1=expected_h1,
    )
    sealed = seal_retained_confirmation_failure(
        lock_path,
        h2_commit,
        case_id,
        case_root,
        physical["physical_archive"],
        {
            budget: (
                measurement_root
                / f"budget-{budget}"
                / measurement.MEASUREMENT_ARCHIVE_FILENAME
            )
            for budget in CAMERA_BUDGETS
        },
        {
            budget: (
                measurement_root
                / f"budget-{budget}"
                / measurement.UNCERTAINTY_ARCHIVE_FILENAME
            )
            for budget in CAMERA_BUDGETS
        },
        failure_code,
        measurement_manifest=(measurement_root / measurement.MANIFEST_FILENAME),
        expected_h1=expected_h1,
    )
    return {
        "case_id": case_id,
        "failure_code": failure_code,
        "physical": physical,
        "nested_measurement_artifact_sha256": nested["artifact_sha256"],
        "case_seal": sealed,
    }


__all__ = [
    "FRAME_ZERO_MANIFEST_FILENAME",
    "IDENTITY_PERSISTENCE_ADAPTER_KEY",
    "IDENTITY_PERSISTENCE_POLICY",
    "adapt_frame_zero_original_splat_identity_persistence",
    "build_retained_failure_nested_measurements",
    "build_retained_failure_physical_backbone",
    "confirmation_frame_zero_physical_policy",
    "materialize_and_seal_retained_confirmation_failure",
    "validate_native_original_splat_frame_zero",
    "validate_original_splat_identity_persistence_manifest",
]
