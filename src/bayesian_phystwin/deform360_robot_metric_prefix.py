"""Causal metric-gauge grids from released Deform360 robot measurements.

Deform360's released rendered depth is a privileged full-sequence product and
must not enter predictive experiments.  This module instead projects the
released, synchronized gripper taxel geometry into each calibrated camera over
the registered prefix.  The resulting sparse metric rows calibrate only the
local Sim(3) gauge of a visual prediction; they are not an object-state target.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import genuine_integer, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    require_exact_fields,
    sha256_digest,
)
from .deform360_bias_aware_prospective_physical import _gripper_taxel_points
from .deform360_calibration_visual_execution_admission import (
    _load_stable_json_object,
    validate_deform360_prepared_source_inventory,
)
from .deform360_public_contact_prefix import (
    _inventory_object,
    _load_robot_prefix,
    _mapping,
    _ordinary_directory,
    _ordinary_file,
    _sequence,
    _verified_inventory_file,
)

DEFORM360_ROBOT_METRIC_PREFIX_SCHEMA: Final = (
    "bayesian-phystwin.deform360-robot-metric-prefix"
)
DEFORM360_ROBOT_METRIC_PREFIX_VERSION: Final = 1
DEFORM360_ROBOT_METRIC_PREFIX_SEMANTICS: Final = (
    "released-prefix-robot-taxel-geometry-on-motioncrafter-grid-v1"
)
DEFORM360_ROBOT_METRIC_CALIBRATION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-robot-metric-calibration"
)
DEFORM360_ROBOT_METRIC_CALIBRATION_VERSION: Final = 1
DEFORM360_ROBOT_METRIC_SOURCE_KIND: Final = "released-deform360-robot-taxel-gauge-v1"
DEFORM360_ROBOT_METRIC_PREFIX_CLAIM_BOUNDARY: Final = (
    "Calibration-source projection of released robot geometry into released "
    "camera coordinates over the registered causal prefix. The artifact is "
    "not an object-state target, visual-accuracy result, contact estimate, "
    "confirmation authorization, deployment-safety result, or state of the art."
)

METRIC_PREFIX_ARRAYS: Final = frozenset(
    {"frame_indices", "points_world_m", "valid_mask"}
)
METRIC_PREFIX_FILENAME: Final = "metric-prefix.npz"
METRIC_CALIBRATION_FILENAME: Final = "metric-calibration.json"
METRIC_MANIFEST_FILENAME: Final = "metric-prefix.json"

_INFORMATION_BOUNDARY = {
    "calibration_robot_state_opened": True,
    "calibration_camera_calibration_opened": True,
    "calibration_camera_images_opened": False,
    "calibration_tactile_payloads_opened": False,
    "rendered_depth_opened": False,
    "full_sequence_reconstruction_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
}

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "status",
        "object_id",
        "episode_id",
        "stratum",
        "camera_id",
        "prepared_source_inventory_id",
        "prepared_source_inventory_file_sha256",
        "processing_revision",
        "causal_frame_range_half_open",
        "source_image_shape",
        "target_image_shape",
        "bimanual",
        "robot_axis_count",
        "projected_point_count",
        "per_frame_projected_point_count",
        "source_kind",
        "files",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_CALIBRATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "object_id",
        "episode_id",
        "camera_id",
        "processing_revision",
        "coordinate_frame",
        "source_image_shape",
        "target_image_shape",
        "cover_resize",
        "intrinsics",
        "camera_to_world",
        "source_artifacts",
        "information_boundary",
        "calibration_id",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _load_camera_dictionary(payload: bytes, *, name: str) -> dict[str, np.ndarray]:
    """Load one trusted, inventory-hashed first-party Deform360 dictionary."""

    try:
        stored = np.load(io.BytesIO(payload), allow_pickle=True)
        value = stored.item()
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(f"cannot load {name}") from error
    _require(isinstance(value, dict), f"{name} must contain a camera dictionary")
    result: dict[str, np.ndarray] = {}
    for key, item in value.items():
        camera = _literal_string(key, name=f"{name} camera")
        _require(camera not in result, f"{name} repeats camera {camera!r}")
        result[camera] = np.asarray(item, dtype=np.float64)
    return result


def _validate_intrinsics(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (3, 3), "camera intrinsics must have shape (3, 3)")
    _require(np.all(np.isfinite(matrix)), "camera intrinsics are not finite")
    _require(matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0, "invalid focal length")
    _require(
        np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-10, rtol=0.0),
        "camera intrinsics use an unsupported projective row",
    )
    return matrix


def _validate_camera_to_world(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4), "camera extrinsics must have shape (4, 4)")
    _require(np.all(np.isfinite(matrix)), "camera extrinsics are not finite")
    _require(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0),
        "camera extrinsics are not homogeneous",
    )
    rotation = matrix[:3, :3]
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0)
        and np.linalg.det(rotation) > 0.0,
        "camera extrinsics do not contain a proper rotation",
    )
    return matrix


def _cover_resize_parameters(
    source_shape: tuple[int, int], target_shape: tuple[int, int]
) -> dict[str, int | float | str]:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    _require(min(*source_shape, *target_shape) > 0, "image dimensions must be positive")
    scale = max(target_height / source_height, target_width / source_width)
    resized_height = int(source_height * scale)
    resized_width = int(source_width * scale)
    _require(
        resized_height >= target_height and resized_width >= target_width,
        "cover resize does not cover the target grid",
    )
    return {
        "scale": float(scale),
        "resized_height": resized_height,
        "resized_width": resized_width,
        "crop_top": (resized_height - target_height) // 2,
        "crop_left": (resized_width - target_width) // 2,
        "pixel_center_convention": "(index+0.5)*scale-0.5",
        "collision_policy": "nearest-camera-depth-then-taxel-index",
    }


def _project_world_points(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world_m, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 3, "points must have shape (N,3)")
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    points_camera = (points - translation) @ rotation
    depth = points_camera[:, 2]
    pixels: np.ndarray = np.full((len(points), 2), np.nan, dtype=np.float64)
    front = depth > 1e-9
    pixels[front, 0] = (
        intrinsics[0, 0] * points_camera[front, 0] / depth[front] + intrinsics[0, 2]
    )
    pixels[front, 1] = (
        intrinsics[1, 1] * points_camera[front, 1] / depth[front] + intrinsics[1, 2]
    )
    return pixels, depth


def project_robot_taxels_to_motioncrafter_grid(
    *,
    poses: np.ndarray,
    openings: np.ndarray,
    bimanual: bool,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the released robot surface grid without reading camera pixels."""

    pose_values = np.asarray(poses, dtype=np.float64)
    opening_values = np.asarray(openings, dtype=np.float64)
    frame_count = len(pose_values)
    expected_pose_tail = (2, 4, 4) if bimanual else (4, 4)
    expected_opening_tail = (2,) if bimanual else ()
    _require(
        pose_values.shape == (frame_count, *expected_pose_tail),
        "robot prefix pose shape changed",
    )
    _require(
        opening_values.shape == (frame_count, *expected_opening_tail),
        "robot prefix opening shape changed",
    )
    target_height, target_width = target_shape
    metric: np.ndarray = np.zeros(
        (frame_count, target_height, target_width, 3), dtype=np.float64
    )
    valid: np.ndarray = np.zeros(
        (frame_count, target_height, target_width), dtype=np.bool_
    )
    counts: np.ndarray = np.zeros(frame_count, dtype=np.int64)
    cover = _cover_resize_parameters(source_shape, target_shape)
    scale = cast(float, cover["scale"])
    top = cast(int, cover["crop_top"])
    left = cast(int, cover["crop_left"])
    axis_count = 2 if bimanual else 1

    for frame in range(frame_count):
        blocks = []
        for axis in range(axis_count):
            pose = pose_values[frame, axis] if bimanual else pose_values[frame]
            opening = opening_values[frame, axis] if bimanual else opening_values[frame]
            blocks.append(_gripper_taxel_points(float(opening), pose))
        points = np.concatenate(blocks, axis=0)
        source_pixels, depth = _project_world_points(
            points, intrinsics, camera_to_world
        )
        target_x = (source_pixels[:, 0] + 0.5) * scale - 0.5 - left
        target_y = (source_pixels[:, 1] + 0.5) * scale - 0.5 - top
        finite = np.isfinite(target_x) & np.isfinite(target_y)
        columns = np.rint(np.where(finite, target_x, 0.0)).astype(np.int64)
        rows = np.rint(np.where(finite, target_y, 0.0)).astype(np.int64)
        inside = (
            finite
            & (depth > 1e-9)
            & (rows >= 0)
            & (rows < target_height)
            & (columns >= 0)
            & (columns < target_width)
        )
        source_indices = np.flatnonzero(inside)
        if not len(source_indices):
            continue
        rows = rows[source_indices]
        columns = columns[source_indices]
        selected_depth = depth[source_indices]
        flattened = rows * target_width + columns
        order = np.lexsort((source_indices, selected_depth, flattened))
        ordered_flattened = flattened[order]
        unique = np.concatenate(
            (
                np.asarray([True]),
                ordered_flattened[1:] != ordered_flattened[:-1],
            )
        )
        chosen = source_indices[order[unique]]
        chosen_rows = rows[order[unique]]
        chosen_columns = columns[order[unique]]
        metric[frame, chosen_rows, chosen_columns] = points[chosen]
        valid[frame, chosen_rows, chosen_columns] = True
        counts[frame] = len(chosen)
    return metric, valid, counts


def _camera_record(object_row: Mapping[str, Any], camera_id: str) -> Mapping[str, Any]:
    matches = [
        cast(Mapping[str, Any], item)
        for item in _sequence(object_row["cameras"], name="inventory cameras")
        if isinstance(item, Mapping) and item.get("camera") == camera_id
    ]
    _require(len(matches) == 1, f"inventory does not contain camera {camera_id!r}")
    return matches[0]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _shape_pair(value: object, *, name: str) -> tuple[int, int]:
    values = _sequence(value, name=name)
    _require(len(values) == 2, f"{name} must have two dimensions")
    return (
        genuine_integer(values[0], name=f"{name}[0]", minimum=1),
        genuine_integer(values[1], name=f"{name}[1]", minimum=1),
    )


def validate_deform360_robot_metric_prefix(
    directory: str | Path,
) -> dict[str, Any]:
    """Strictly reload one robot metric-prefix artifact."""

    root = _ordinary_directory(directory, name="robot metric-prefix directory")
    manifest_path = _ordinary_file(root / METRIC_MANIFEST_FILENAME, name="manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load robot metric-prefix manifest") from error
    _require(
        isinstance(manifest, dict), "robot metric-prefix manifest must be an object"
    )
    require_exact_fields(
        manifest, expected=_MANIFEST_FIELDS, name="robot metric prefix"
    )
    _require(
        manifest["schema"] == DEFORM360_ROBOT_METRIC_PREFIX_SCHEMA
        and manifest["schema_version"] == DEFORM360_ROBOT_METRIC_PREFIX_VERSION
        and manifest["semantics"] == DEFORM360_ROBOT_METRIC_PREFIX_SEMANTICS,
        "robot metric-prefix contract changed",
    )
    _require(manifest["status"] == "materialized", "robot metric prefix is incomplete")
    _require(
        manifest["information_boundary"] == _INFORMATION_BOUNDARY,
        "information boundary changed",
    )
    _require(
        manifest["claim_boundary"] == DEFORM360_ROBOT_METRIC_PREFIX_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    identity = dict(manifest)
    declared_id = sha256_digest(identity.pop("artifact_id"), name="artifact_id")
    _require(content_id(identity) == declared_id, "robot metric-prefix ID changed")
    exact_revision(manifest["processing_revision"], name="processing_revision")
    _literal_string(manifest["object_id"], name="object_id")
    _literal_string(manifest["camera_id"], name="camera_id")
    _literal_string(manifest["stratum"], name="stratum")
    genuine_integer(manifest["episode_id"], name="episode_id")
    _require(
        manifest["source_kind"] == DEFORM360_ROBOT_METRIC_SOURCE_KIND,
        "robot metric source kind changed",
    )
    _require(type(manifest["bimanual"]) is bool, "bimanual must be Boolean")
    axis_count = genuine_integer(
        manifest["robot_axis_count"], name="robot_axis_count", minimum=1
    )
    _require(
        axis_count == (2 if manifest["bimanual"] else 1),
        "robot axis count changed",
    )
    source_shape = _shape_pair(
        manifest["source_image_shape"], name="source image shape"
    )
    target_shape = _shape_pair(
        manifest["target_image_shape"], name="target image shape"
    )
    frame_range = _sequence(
        manifest["causal_frame_range_half_open"], name="causal frame range"
    )
    _require(len(frame_range) == 2, "causal frame range must have two bounds")
    start = genuine_integer(frame_range[0], name="causal start")
    stop = genuine_integer(frame_range[1], name="causal stop", minimum=1)
    _require(start < stop, "causal frame range is empty")
    files = _mapping(manifest["files"], name="metric-prefix files")
    _require(
        set(files) == {METRIC_PREFIX_FILENAME, METRIC_CALIBRATION_FILENAME},
        "metric-prefix file roster changed",
    )
    for name, digest in files.items():
        path = _ordinary_file(root / name, name=f"metric-prefix file {name}")
        _require(
            _sha256_file(path) == sha256_digest(digest, name=f"{name} digest"),
            f"{name} digest changed",
        )
    checksum_path = _ordinary_file(root / "SHA256SUMS", name="SHA256SUMS")
    checksum_names = (
        METRIC_PREFIX_FILENAME,
        METRIC_CALIBRATION_FILENAME,
        METRIC_MANIFEST_FILENAME,
    )
    expected_checksums = "".join(
        f"{_sha256_file(root / name)}  {name}\n" for name in sorted(checksum_names)
    )
    _require(
        checksum_path.read_text(encoding="ascii") == expected_checksums,
        "SHA256SUMS changed",
    )
    try:
        with np.load(root / METRIC_PREFIX_FILENAME, allow_pickle=False) as stored:
            _require(
                set(stored.files) == METRIC_PREFIX_ARRAYS,
                "metric-prefix arrays changed",
            )
            frames = np.asarray(stored["frame_indices"])
            points = np.asarray(stored["points_world_m"])
            valid = np.asarray(stored["valid_mask"])
    except (OSError, ValueError) as error:
        raise ValueError("cannot load metric-prefix arrays") from error
    expected_frames: np.ndarray = np.arange(start, stop, dtype=np.int64)
    _require(
        frames.dtype == np.dtype(np.int64) and np.array_equal(frames, expected_frames),
        "metric-prefix frames changed",
    )
    expected_grid = (len(expected_frames), *target_shape)
    _require(
        points.shape == (*expected_grid, 3) and points.dtype == np.dtype(np.float64),
        "metric point grid changed",
    )
    _require(
        valid.shape == expected_grid and valid.dtype == np.dtype(np.bool_),
        "metric valid grid changed",
    )
    _require(np.all(np.isfinite(points)), "metric point grid is not finite")
    counts = np.count_nonzero(valid, axis=(1, 2)).astype(np.int64)
    declared_counts = _sequence(
        manifest["per_frame_projected_point_count"],
        name="per-frame projected-point count",
    )
    _require(
        len(declared_counts) == len(expected_frames)
        and all(type(value) is int and value >= 0 for value in declared_counts),
        "per-frame projected-point counts changed",
    )
    projected_count = genuine_integer(
        manifest["projected_point_count"],
        name="projected_point_count",
        minimum=1,
    )
    _require(
        counts.tolist() == list(declared_counts)
        and int(np.sum(counts)) == projected_count,
        "projected-point accounting changed",
    )
    calibration = json.loads(
        (root / METRIC_CALIBRATION_FILENAME).read_text(encoding="utf-8")
    )
    _require(isinstance(calibration, dict), "metric calibration must be an object")
    require_exact_fields(
        calibration, expected=_CALIBRATION_FIELDS, name="metric calibration"
    )
    calibration_identity = dict(calibration)
    calibration_id = sha256_digest(
        calibration_identity.pop("calibration_id"), name="calibration_id"
    )
    _require(
        content_id(calibration_identity) == calibration_id,
        "metric calibration ID changed",
    )
    _require(
        calibration["schema"] == DEFORM360_ROBOT_METRIC_CALIBRATION_SCHEMA
        and calibration["schema_version"] == DEFORM360_ROBOT_METRIC_CALIBRATION_VERSION,
        "metric calibration contract changed",
    )
    _require(
        _shape_pair(calibration["source_image_shape"], name="calibration source shape")
        == source_shape
        and _shape_pair(
            calibration["target_image_shape"], name="calibration target shape"
        )
        == target_shape,
        "metric calibration image shape changed",
    )
    _require(
        calibration["object_id"] == manifest["object_id"]
        and calibration["episode_id"] == manifest["episode_id"]
        and calibration["camera_id"] == manifest["camera_id"]
        and calibration["processing_revision"] == manifest["processing_revision"]
        and calibration["coordinate_frame"] == "deform360-world"
        and calibration["source_artifacts"] == manifest["source_artifacts"]
        and calibration["information_boundary"] == _INFORMATION_BOUNDARY,
        "metric calibration lineage changed",
    )
    _validate_intrinsics(calibration["intrinsics"])
    _validate_camera_to_world(calibration["camera_to_world"])
    _require(
        calibration["cover_resize"]
        == _cover_resize_parameters(source_shape, target_shape),
        "metric calibration cover-resize policy changed",
    )
    return cast(dict[str, Any], manifest)


def materialize_deform360_robot_metric_prefix(
    *,
    prepared_source_inventory_path: str | Path,
    processed_root: str | Path,
    object_id: str,
    camera_id: str,
    expected_processing_revision: str,
    target_height: int,
    target_width: int,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Publish one causal robot-geometry metric grid atomically."""

    inventory_path = _ordinary_file(
        prepared_source_inventory_path, name="prepared-source inventory"
    )
    inventory_value, inventory_sha256, _inventory_bytes = _load_stable_json_object(
        inventory_path, label="prepared-source inventory"
    )
    inventory = validate_deform360_prepared_source_inventory(inventory_value)
    processing_revision = exact_revision(
        expected_processing_revision, name="expected_processing_revision"
    )
    _require(
        inventory["processing_revision"] == processing_revision,
        "prepared-source processing revision changed",
    )
    object_name = _literal_string(object_id, name="object_id")
    camera_name = _literal_string(camera_id, name="camera_id")
    object_row = _inventory_object(inventory, object_id=object_name)
    camera = _camera_record(object_row, camera_name)
    source_shape = (
        genuine_integer(camera["height"], name="camera height", minimum=1),
        genuine_integer(camera["width"], name="camera width", minimum=1),
    )
    target_shape = (
        genuine_integer(target_height, name="target_height", minimum=1),
        genuine_integer(target_width, name="target_width", minimum=1),
    )
    root = _ordinary_directory(processed_root, name="processed root")
    episode_files = _mapping(object_row["episode_files"], name="episode files")
    robot_payload, robot_relative, robot_sha256 = _verified_inventory_file(
        root, episode_files.get("robot"), name=f"{object_name} robot"
    )
    intrinsics_payload, intrinsics_relative, intrinsics_sha256 = (
        _verified_inventory_file(
            root,
            episode_files.get("undistorted_intrinsics"),
            name=f"{object_name} intrinsics",
        )
    )
    extrinsics_payload, extrinsics_relative, extrinsics_sha256 = (
        _verified_inventory_file(
            root,
            episode_files.get("extrinsics"),
            name=f"{object_name} extrinsics",
        )
    )
    action_window = _mapping(object_row["action_window"], name="action window")
    prefix = _sequence(
        action_window["prefix_raw_frame_range_half_open"], name="prefix frame range"
    )
    _require(len(prefix) == 2, "prefix frame range must have two bounds")
    prefix_start = genuine_integer(prefix[0], name="prefix start")
    prefix_stop = genuine_integer(prefix[1], name="prefix stop", minimum=1)
    _require(prefix_start < prefix_stop, "prefix frame range is empty")
    _require(
        prefix_stop
        <= genuine_integer(camera["frame_count"], name="camera frame count", minimum=1),
        "camera does not cover the causal prefix",
    )
    poses, openings, bimanual = _load_robot_prefix(
        robot_payload, prefix_start=prefix_start, prefix_stop=prefix_stop
    )
    intrinsics_by_camera = _load_camera_dictionary(
        intrinsics_payload, name="undistorted intrinsics"
    )
    extrinsics_by_camera = _load_camera_dictionary(
        extrinsics_payload, name="camera extrinsics"
    )
    _require(
        camera_name in intrinsics_by_camera and camera_name in extrinsics_by_camera,
        "camera calibration is missing",
    )
    intrinsics = _validate_intrinsics(intrinsics_by_camera[camera_name])
    camera_to_world = _validate_camera_to_world(extrinsics_by_camera[camera_name])
    points, valid, counts = project_robot_taxels_to_motioncrafter_grid(
        poses=poses,
        openings=openings,
        bimanual=bimanual,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        source_shape=source_shape,
        target_shape=target_shape,
    )
    _require(np.any(valid), "released robot geometry is outside this camera prefix")
    frame_indices: np.ndarray = np.arange(prefix_start, prefix_stop, dtype=np.int64)
    source_artifacts = dict(
        sorted(
            {
                "prepared-source-inventory.json": inventory_sha256,
                robot_relative: robot_sha256,
                intrinsics_relative: intrinsics_sha256,
                extrinsics_relative: extrinsics_sha256,
            }.items()
        )
    )
    cover = _cover_resize_parameters(source_shape, target_shape)
    calibration_identity: dict[str, Any] = {
        "schema": DEFORM360_ROBOT_METRIC_CALIBRATION_SCHEMA,
        "schema_version": DEFORM360_ROBOT_METRIC_CALIBRATION_VERSION,
        "object_id": object_name,
        "episode_id": object_row["episode_id"],
        "camera_id": camera_name,
        "processing_revision": processing_revision,
        "coordinate_frame": "deform360-world",
        "source_image_shape": list(source_shape),
        "target_image_shape": list(target_shape),
        "cover_resize": cover,
        "intrinsics": intrinsics.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "source_artifacts": source_artifacts,
        "information_boundary": dict(_INFORMATION_BOUNDARY),
    }
    calibration = {
        **calibration_identity,
        "calibration_id": content_id(calibration_identity),
    }

    target = Path(output_directory).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(target.parent, name="output parent")
    _require(not os.path.lexists(target), "output directory already exists")
    lock = target.parent / f".{target.name}.lock"
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    lock_descriptor: int | None = None
    lock_created = False
    temporary.mkdir(mode=0o700)
    try:
        np.savez_compressed(
            temporary / METRIC_PREFIX_FILENAME,
            frame_indices=frame_indices,
            points_world_m=points,
            valid_mask=valid,
        )
        _write_json(temporary / METRIC_CALIBRATION_FILENAME, calibration)
        files = {
            METRIC_PREFIX_FILENAME: _sha256_file(temporary / METRIC_PREFIX_FILENAME),
            METRIC_CALIBRATION_FILENAME: _sha256_file(
                temporary / METRIC_CALIBRATION_FILENAME
            ),
        }
        manifest_identity: dict[str, Any] = {
            "schema": DEFORM360_ROBOT_METRIC_PREFIX_SCHEMA,
            "schema_version": DEFORM360_ROBOT_METRIC_PREFIX_VERSION,
            "semantics": DEFORM360_ROBOT_METRIC_PREFIX_SEMANTICS,
            "status": "materialized",
            "object_id": object_name,
            "episode_id": object_row["episode_id"],
            "stratum": object_row["stratum"],
            "camera_id": camera_name,
            "prepared_source_inventory_id": inventory["inventory_id"],
            "prepared_source_inventory_file_sha256": inventory_sha256,
            "processing_revision": processing_revision,
            "causal_frame_range_half_open": [prefix_start, prefix_stop],
            "source_image_shape": list(source_shape),
            "target_image_shape": list(target_shape),
            "bimanual": bimanual,
            "robot_axis_count": 2 if bimanual else 1,
            "projected_point_count": int(np.sum(counts)),
            "per_frame_projected_point_count": counts.tolist(),
            "source_kind": DEFORM360_ROBOT_METRIC_SOURCE_KIND,
            "files": files,
            "source_artifacts": source_artifacts,
            "information_boundary": dict(_INFORMATION_BOUNDARY),
            "claim_boundary": DEFORM360_ROBOT_METRIC_PREFIX_CLAIM_BOUNDARY,
        }
        manifest = {
            **manifest_identity,
            "artifact_id": content_id(manifest_identity),
        }
        _write_json(temporary / METRIC_MANIFEST_FILENAME, manifest)
        checksum_names = (
            METRIC_PREFIX_FILENAME,
            METRIC_CALIBRATION_FILENAME,
            METRIC_MANIFEST_FILENAME,
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(temporary / name)}  {name}\n"
                for name in sorted(checksum_names)
            ),
            encoding="ascii",
        )
        validate_deform360_robot_metric_prefix(temporary)
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_created = True
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(lock_descriptor)
        lock_descriptor = None
        _require(not os.path.lexists(target), "output directory already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_created:
            lock.unlink(missing_ok=True)
    return validate_deform360_robot_metric_prefix(target)


__all__ = [
    "DEFORM360_ROBOT_METRIC_PREFIX_SCHEMA",
    "DEFORM360_ROBOT_METRIC_PREFIX_SEMANTICS",
    "DEFORM360_ROBOT_METRIC_PREFIX_VERSION",
    "DEFORM360_ROBOT_METRIC_SOURCE_KIND",
    "METRIC_CALIBRATION_FILENAME",
    "METRIC_MANIFEST_FILENAME",
    "METRIC_PREFIX_ARRAYS",
    "METRIC_PREFIX_FILENAME",
    "materialize_deform360_robot_metric_prefix",
    "project_robot_taxels_to_motioncrafter_grid",
    "validate_deform360_robot_metric_prefix",
]
