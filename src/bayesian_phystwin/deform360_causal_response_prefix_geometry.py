"""Hash-bound prefix geometry custody for the V14 Deform360 source study."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_direct_depth_assets import (
    PREFIX_FRAME_COUNT,
    canonical_sha256,
    validate_v14_prefix_mask_artifact,
)
from .deform360_object_exclusion import file_sha256

GEOMETRY_PROTOCOL_KIND = "Deform360CausalDirectDepthPrefixGeometryProtocolV14"
GEOMETRY_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-prefix-geometry"
)
GEOMETRY_CONTRACT = (
    "deform360-causal-response-direct-depth-prefix-geometry-v14"
)
GEOMETRY_MANIFEST_KIND = "Deform360CausalDirectDepthPrefixGeometryV14"
GEOMETRY_RESULT_KIND = "Deform360CausalDirectDepthPrefixGeometryResultV14"
MINIMUM_PHYSICAL_NODE_COUNT = 128
MAXIMUM_PHYSICAL_NODE_COUNT = 10000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {source}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def load_v14_prefix_geometry_protocol(path: str | Path) -> dict[str, Any]:
    """Validate the child lock for prefix-only geometry reconstruction."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == GEOMETRY_PROTOCOL_KIND
        and payload.get("protocol_id") == GEOMETRY_PROTOCOL_ID
        and payload.get("status")
        == "locked_after_prefix_masks_before_geometry_reconstruction",
        "V14 prefix geometry protocol identity changed",
    )
    _require(
        payload.get("config_sha256")
        == canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-"
                b"protocol-v14\0"
            ),
            digest_key="config_sha256",
        ),
        "V14 prefix geometry protocol checksum changed",
    )
    parent = payload.get("parent_prefix_assets")
    _require(
        isinstance(parent, Mapping)
        and parent.get("protocol_id")
        == "deform360-causal-response-direct-depth-v14-prefix-assets"
        and isinstance(parent.get("config_sha256"), str)
        and isinstance(parent.get("file_sha256"), str),
        "V14 prefix geometry parent binding changed",
    )
    inputs = payload.get("mask_inputs")
    _require(
        isinstance(inputs, list)
        and [row.get("queue_rank") for row in inputs] == list(range(3, 15))
        and all(
            isinstance(row.get("artifact_sha256"), str)
            and isinstance(row.get("file_sha256"), str)
            and int(row.get("successful_camera_count", 0)) >= 8
            for row in inputs
        ),
        "V14 prefix geometry mask bindings changed",
    )
    geometry = payload.get("geometry")
    _require(
        isinstance(geometry, Mapping)
        and geometry.get("prefix_frame_count") == PREFIX_FRAME_COUNT
        and geometry.get("maximum_object_observation_frame")
        == PREFIX_FRAME_COUNT - 1
        and geometry.get("minimum_visual_hull_points") == 512
        and geometry.get("voxel_resolution") == 120
        and geometry.get("cube_half_extent_m") == 0.5
        and geometry.get("first_frame_iterations") == 500
        and geometry.get("warm_start_iterations") == 250
        and geometry.get("seed_point_count") == MAXIMUM_PHYSICAL_NODE_COUNT
        and geometry.get("seed_crop_half_extent_m") == 0.5
        and geometry.get("seed_rng") == 0
        and geometry.get("minimum_physical_node_count")
        == MINIMUM_PHYSICAL_NODE_COUNT
        and geometry.get("maximum_physical_node_count")
        == MAXIMUM_PHYSICAL_NODE_COUNT,
        "V14 prefix geometry reconstruction contract changed",
    )
    runtime = payload.get("runtime")
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("deform360_revision")
        == "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
        and runtime.get("required_path_prefix") == "/usr/local/cuda/bin"
        and runtime.get("nvcc_path") == "/usr/local/cuda/bin/nvcc"
        and runtime.get("gsplat_version") == "1.4.0"
        and runtime.get("torch_version") == "2.4.0+cu121"
        and runtime.get("required_backend_probe") == "CameraModelType.PINHOLE",
        "V14 prefix geometry runtime contract changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("maximum_object_observation_frame")
        == PREFIX_FRAME_COUNT - 1
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 prefix geometry protocol crossed its information boundary",
    )
    return payload


def mask_binding_for_rank(
    protocol: Mapping[str, Any], queue_rank: int
) -> Mapping[str, Any]:
    """Return the unique frozen mask binding for a queue rank."""

    matches = [
        row
        for row in protocol["mask_inputs"]
        if int(row["queue_rank"]) == int(queue_rank)
    ]
    _require(len(matches) == 1, "V14 geometry rank lacks one mask binding")
    return matches[0]


def validate_v14_geometry_mask_input(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    asset_protocol: Mapping[str, Any],
    mask_episode: str | Path,
    queue_rank: int,
) -> dict[str, Any]:
    """Validate one exact mask carrier against the child geometry lock."""

    source = Path(path)
    binding = mask_binding_for_rank(protocol, queue_rank)
    _require(
        file_sha256(source) == binding["file_sha256"],
        "V14 prefix geometry mask file changed",
    )
    payload = validate_v14_prefix_mask_artifact(
        source,
        asset_protocol=asset_protocol,
        mask_episode=mask_episode,
    )
    _require(
        payload.get("queue_rank") == queue_rank
        and payload.get("artifact_sha256") == binding["artifact_sha256"]
        and payload.get("successful_camera_count")
        == binding["successful_camera_count"]
        and payload.get("status") == "ready_for_prefix_geometry",
        "V14 prefix geometry mask disposition changed",
    )
    return payload


def projected_seed_support(
    points_m: np.ndarray,
    *,
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    depth_mm_by_camera: Mapping[str, np.ndarray],
    depth_tolerance_m: float,
) -> tuple[int, dict[str, int]]:
    """Count seeded nodes agreeing with at least one frame-zero depth map."""

    points = np.asarray(points_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 0
        and np.all(np.isfinite(points)),
        "frame-zero seed points are invalid",
    )
    _require(depth_tolerance_m > 0.0, "depth tolerance must be positive")
    cameras = tuple(sorted(depth_mm_by_camera))
    _require(
        cameras
        and set(cameras).issubset(intrinsics_by_camera)
        and set(cameras).issubset(camera_to_world_by_camera),
        "frame-zero depth panel lacks calibration",
    )
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    supported_any = np.zeros(len(points), dtype=bool)
    per_camera: dict[str, int] = {}
    for camera in cameras:
        intrinsics = np.asarray(intrinsics_by_camera[camera], dtype=np.float64)
        camera_to_world = np.asarray(
            camera_to_world_by_camera[camera], dtype=np.float64
        )
        depth_mm = np.asarray(depth_mm_by_camera[camera])
        _require(
            intrinsics.shape == (3, 3)
            and camera_to_world.shape == (4, 4)
            and depth_mm.ndim == 2,
            f"invalid frame-zero support input: {camera}",
        )
        camera_points = (np.linalg.inv(camera_to_world) @ homogeneous.T).T
        z_m = camera_points[:, 2]
        positive = z_m > 0.0
        projected = (intrinsics @ camera_points[:, :3].T).T
        u = np.rint(projected[:, 0] / np.maximum(projected[:, 2], 1e-12)).astype(
            np.int64
        )
        v = np.rint(projected[:, 1] / np.maximum(projected[:, 2], 1e-12)).astype(
            np.int64
        )
        height, width = depth_mm.shape
        inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        observed_m = np.zeros(len(points), dtype=np.float64)
        observed_m[inside] = depth_mm[v[inside], u[inside]].astype(np.float64) / 1000.0
        supported = (
            inside
            & (observed_m > 0.0)
            & (np.abs(observed_m - z_m) <= depth_tolerance_m)
        )
        per_camera[camera] = int(np.count_nonzero(supported))
        supported_any |= supported
    return int(np.count_nonzero(supported_any)), per_camera


def validate_v14_prefix_geometry_manifest(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    geometry_episode: str | Path,
    forbidden_plaintext: str | None = None,
) -> dict[str, Any]:
    """Validate one successful hash-only V14 prefix geometry manifest."""

    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if forbidden_plaintext is not None:
        _require(
            forbidden_plaintext not in raw,
            "V14 geometry manifest leaked plaintext object identity",
        )
    payload = _read_json(source)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == GEOMETRY_MANIFEST_KIND
        and payload.get("contract") == GEOMETRY_CONTRACT
        and payload.get("protocol_id") == GEOMETRY_PROTOCOL_ID
        and payload.get("geometry_protocol_config_sha256")
        == protocol["config_sha256"]
        and payload.get("status") == "ready_for_physical_preflight"
        and payload.get("artifact_sha256")
        == canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-v14\0"
            ),
            digest_key="artifact_sha256",
        ),
        "V14 prefix geometry manifest binding changed",
    )
    node_count = int(payload.get("physical_node_count", 0))
    _require(
        MINIMUM_PHYSICAL_NODE_COUNT
        <= node_count
        <= MAXIMUM_PHYSICAL_NODE_COUNT,
        "V14 prefix geometry node count is inadmissible",
    )
    episode = Path(geometry_episode)
    outputs = payload.get("outputs_sha256")
    _require(isinstance(outputs, Mapping), "V14 geometry outputs are missing")
    fixed = {
        "intrinsics": episode / "undistorted_intrinsics.npy",
        "extrinsics": episode / "extrinsics.npy",
        "robot": episode / "robot" / "robot.npz",
        "frame_zero_splat": episode / "splatfacto" / "splat_000000.ply",
        "frame_zero_points": episode / "start_obj_pcd.ply",
    }
    _require(
        all(
            file.is_file() and file_sha256(file) == outputs.get(role)
            for role, file in fixed.items()
        ),
        "V14 prefix geometry fixed outputs changed",
    )
    cameras = payload.get("cameras")
    camera_records = payload.get("camera_records")
    depth_outputs = outputs.get("depth_by_camera")
    _require(
        isinstance(cameras, list)
        and isinstance(camera_records, list)
        and [row.get("camera") for row in camera_records] == cameras
        and all(
            row.get("rgb_frame_count")
            == row.get("mask_frame_count")
            == row.get("depth_frame_count")
            == row.get("gripper_mask_frame_count")
            == PREFIX_FRAME_COUNT
            for row in camera_records
        )
        and payload.get("calibration_valid") is True
        and isinstance(depth_outputs, Mapping)
        and set(cameras) == set(depth_outputs)
        and all(
            file_sha256(episode / camera / "rendered_depth.h5")
            == depth_outputs[camera]
            for camera in cameras
        ),
        "V14 prefix geometry depth panel changed",
    )
    return payload


__all__ = [
    "GEOMETRY_CONTRACT",
    "GEOMETRY_MANIFEST_KIND",
    "GEOMETRY_PROTOCOL_ID",
    "GEOMETRY_PROTOCOL_KIND",
    "GEOMETRY_RESULT_KIND",
    "MAXIMUM_PHYSICAL_NODE_COUNT",
    "MINIMUM_PHYSICAL_NODE_COUNT",
    "load_v14_prefix_geometry_protocol",
    "mask_binding_for_rank",
    "projected_seed_support",
    "validate_v14_geometry_mask_input",
    "validate_v14_prefix_geometry_manifest",
]
