"""Prospective reserve-batch custody for the V14 source panel."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_causal_response_direct_depth_assets import (
    PREFIX_FRAME_COUNT,
    canonical_sha256,
    validate_v14_prefix_mask_artifact,
)
from .deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from .deform360_causal_response_prefix_geometry import (
    GEOMETRY_CONTRACT,
    GEOMETRY_MANIFEST_KIND,
    GEOMETRY_PROTOCOL_ID,
    GEOMETRY_RESULT_KIND,
    MAXIMUM_PHYSICAL_NODE_COUNT,
    MINIMUM_PHYSICAL_NODE_COUNT,
)
from .deform360_object_exclusion import file_sha256

RESERVE_BATCH_KIND = (
    "Deform360CausalResponseDirectDepthReserveBatchProtocolV14V1"
)
RESERVE_BATCH_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-batch-v1"
)
RESERVE_BATCH_RANKS = tuple(range(15, 23))
RESERVE_ADMISSIONS_REQUIRED = 4
RESERVE_GEOMETRY_KIND = (
    "Deform360CausalResponseDirectDepthReserveGeometryProtocolV14V1"
)
RESERVE_GEOMETRY_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-geometry-v1"
)
RESERVE_GEOMETRY_CONTRACT = (
    "deform360-causal-response-direct-depth-reserve-geometry-v14-v1"
)
RESERVE_GEOMETRY_RANKS = (15, 16, 17, 18, 19, 21, 22)
RESERVE_GEOMETRY_APPLICATION_KIND = (
    "Deform360CausalResponseDirectDepthReserveGeometryApplicationV14V1"
)
RESERVE_GEOMETRY_APPLICATION_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-geometry-application-v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    namespace: bytes = (
        b"deform360-causal-response-direct-depth-reserve-batch-v14-v1\0"
    ),
    digest_key: str = "config_sha256",
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read V14 reserve JSON: {source}") from error
    _require(isinstance(payload, dict), "V14 reserve JSON is not an object")
    return payload


def load_v14_reserve_geometry_protocol(
    path: str | Path,
    *,
    reserve_batch_path: str | Path | None = None,
    asset_protocol_path: str | Path | None = None,
    queue_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the exact reserve mask ledger and geometry runtime."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RESERVE_GEOMETRY_KIND
        and payload.get("contract") == RESERVE_GEOMETRY_CONTRACT
        and payload.get("protocol_id") == RESERVE_GEOMETRY_ID
        and payload.get("status")
        == "locked_after_reserve_prefix_masks_before_geometry",
        "V14 reserve geometry identity changed",
    )
    _require(
        payload.get("config_sha256")
        == _canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-reserve-geometry-"
                b"v14-v1\0"
            ),
        ),
        "V14 reserve geometry checksum changed",
    )
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents)
        == {
            "parent_geometry_protocol",
            "prefix_assets",
            "reserve_batch",
            "staging_queue",
        }
        and all(
            isinstance(record, Mapping)
            and all(
                _valid_digest(value)
                for key, value in record.items()
                if key.endswith("sha256")
            )
            for record in parents.values()
        ),
        "V14 reserve geometry parent bindings changed",
    )
    inputs = payload.get("mask_inputs")
    _require(
        isinstance(inputs, list)
        and tuple(int(row.get("queue_rank", 0)) for row in inputs)
        == RESERVE_GEOMETRY_RANKS
        and all(
            _valid_digest(row.get("artifact_sha256"))
            and _valid_digest(row.get("file_sha256"))
            and 8 <= int(row.get("successful_camera_count", 0)) <= 12
            for row in inputs
        ),
        "V14 reserve geometry mask ledger changed",
    )
    failures = payload.get("technical_dispositions")
    _require(
        isinstance(failures, list)
        and len(failures) == 1
        and failures[0].get("queue_rank") == 20
        and failures[0].get("status") == "technical_preflight_failure"
        and failures[0].get("stage") == "source_preparation"
        and _valid_digest(failures[0].get("artifact_sha256"))
        and _valid_digest(failures[0].get("file_sha256")),
        "V14 reserve geometry technical disposition changed",
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
        "V14 reserve geometry numerical contract changed",
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
        and runtime.get("gsplat_extension_sha256")
        == "c9ef20c1ac070cd3d1b4b1dc58ceb58b2293968a672958a1e360a7bd0e075b65"
        and runtime.get("required_backend_probe") == "CameraModelType.PINHOLE",
        "V14 reserve geometry runtime changed",
    )
    implementation = payload.get("implementation_file_sha256")
    _require(
        isinstance(implementation, Mapping)
        and set(implementation)
        == {
            "geometry_builder",
            "geometry_module",
            "reserve_module",
            "reserve_wrapper",
        }
        and all(_valid_digest(value) for value in implementation.values()),
        "V14 reserve geometry implementation binding changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("maximum_object_observation_frame")
        == PREFIX_FRAME_COUNT - 1
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve geometry crossed its information boundary",
    )

    optional_parents = (
        (reserve_batch_path, parents["reserve_batch"], "config_sha256"),
        (asset_protocol_path, parents["prefix_assets"], "config_sha256"),
    )
    for parent_path, binding, semantic_key in optional_parents:
        if parent_path is None:
            continue
        parent = _read_json(parent_path)
        _require(
            parent.get(semantic_key) == binding[semantic_key]
            and file_sha256(parent_path) == binding["file_sha256"],
            "V14 reserve geometry uses another parent protocol",
        )
    if queue_path is not None:
        queue = validate_v14_staging_queue(queue_path)
        binding = parents["staging_queue"]
        _require(
            queue["queue_sha256"] == binding["queue_sha256"]
            and file_sha256(queue_path) == binding["file_sha256"],
            "V14 reserve geometry uses another staging queue",
        )
    return payload


def reserve_geometry_binding_for_rank(
    protocol: Mapping[str, Any], queue_rank: int
) -> Mapping[str, Any]:
    """Return one exact reserve mask binding."""

    matches = [
        row
        for row in protocol["mask_inputs"]
        if int(row["queue_rank"]) == int(queue_rank)
    ]
    _require(len(matches) == 1, "V14 reserve rank lacks one mask binding")
    return matches[0]


def validate_v14_reserve_geometry_mask_input(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    asset_protocol: Mapping[str, Any],
    mask_episode: str | Path,
    queue_rank: int,
) -> dict[str, Any]:
    """Validate one reserve mask against its immutable child ledger."""

    binding = reserve_geometry_binding_for_rank(protocol, queue_rank)
    _require(
        file_sha256(path) == binding["file_sha256"],
        "V14 reserve geometry mask file changed",
    )
    payload = validate_v14_prefix_mask_artifact(
        path,
        asset_protocol=asset_protocol,
        mask_episode=mask_episode,
    )
    _require(
        payload.get("queue_rank") == queue_rank
        and payload.get("artifact_sha256") == binding["artifact_sha256"]
        and payload.get("successful_camera_count")
        == binding["successful_camera_count"]
        and payload.get("status") == "ready_for_prefix_geometry",
        "V14 reserve geometry mask disposition changed",
    )
    return payload


def validate_v14_reserve_geometry_bundle(
    *,
    manifest_path: str | Path,
    result_path: str | Path,
    application_path: str | Path,
    protocol: Mapping[str, Any],
    geometry_episode: str | Path,
    forbidden_plaintext: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate one direct-runtime reserve geometry bundle without mutation."""

    paths = tuple(
        Path(path) for path in (manifest_path, result_path, application_path)
    )
    if forbidden_plaintext is not None:
        _require(
            all(
                forbidden_plaintext not in path.read_text(encoding="utf-8")
                for path in paths
            ),
            "V14 reserve geometry bundle leaked plaintext identity",
        )
    manifest = _read_json(paths[0])
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_kind") == GEOMETRY_MANIFEST_KIND
        and manifest.get("contract") == GEOMETRY_CONTRACT
        and manifest.get("protocol_id") == GEOMETRY_PROTOCOL_ID
        and manifest.get("geometry_protocol_config_sha256")
        == protocol["config_sha256"]
        and manifest.get("status") == "ready_for_physical_preflight"
        and manifest.get("artifact_sha256")
        == canonical_sha256(
            manifest,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-v14\0"
            ),
            digest_key="artifact_sha256",
        ),
        "V14 reserve geometry manifest binding changed",
    )
    rank = int(manifest.get("queue_rank", 0))
    node_count = int(manifest.get("physical_node_count", 0))
    _require(
        rank in RESERVE_GEOMETRY_RANKS
        and MINIMUM_PHYSICAL_NODE_COUNT
        <= node_count
        <= MAXIMUM_PHYSICAL_NODE_COUNT,
        "V14 reserve geometry rank or node count is inadmissible",
    )
    episode = Path(geometry_episode)
    outputs = manifest.get("outputs_sha256")
    _require(isinstance(outputs, Mapping), "V14 reserve geometry outputs are missing")
    fixed = {
        "intrinsics": episode / "undistorted_intrinsics.npy",
        "extrinsics": episode / "extrinsics.npy",
        "robot": episode / "robot" / "robot.npz",
        "frame_zero_splat": episode / "splatfacto" / "splat_0.ply",
        "frame_zero_points": episode / "start_obj_pcd.ply",
    }
    _require(
        all(
            file.is_file() and file_sha256(file) == outputs.get(role)
            for role, file in fixed.items()
        ),
        "V14 reserve geometry fixed outputs changed",
    )
    cameras = manifest.get("cameras")
    camera_records = manifest.get("camera_records")
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
        and manifest.get("calibration_valid") is True
        and isinstance(depth_outputs, Mapping)
        and set(cameras) == set(depth_outputs)
        and all(
            file_sha256(episode / camera / "rendered_depth.h5")
            == depth_outputs[camera]
            for camera in cameras
        ),
        "V14 reserve geometry camera panel changed",
    )
    runtime = manifest.get("runtime")
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("gsplat_extension_sha256")
        == protocol["runtime"]["gsplat_extension_sha256"]
        and runtime.get("python_version") == protocol["runtime"]["python_version"]
        and runtime.get("torch_version") == protocol["runtime"]["torch_version"]
        and runtime.get("torch_cuda_version")
        == protocol["runtime"]["torch_cuda_version"]
        and runtime.get("gsplat_version") == protocol["runtime"]["gsplat_version"]
        and runtime.get("backend_probe")
        == protocol["runtime"]["required_backend_probe"],
        "V14 reserve geometry runtime provenance changed",
    )
    result = _read_json(paths[1])
    _require(
        result.get("artifact_kind") == GEOMETRY_RESULT_KIND
        and result.get("contract") == GEOMETRY_CONTRACT
        and result.get("protocol_id") == GEOMETRY_PROTOCOL_ID
        and result.get("geometry_protocol_config_sha256")
        == protocol["config_sha256"]
        and result.get("status") == "ready_for_source_lock"
        and result.get("artifact_sha256")
        == canonical_sha256(
            result,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-"
                b"result-v14\0"
            ),
            digest_key="artifact_sha256",
        )
        and result.get("geometry_manifest_artifact_sha256")
        == manifest["artifact_sha256"]
        and result.get("geometry_manifest_file_sha256") == file_sha256(paths[0])
        and result.get("physical_node_count") == node_count
        and result.get("queue_rank") == rank
        and result.get("object_hash") == manifest.get("object_hash")
        and result.get("case_hash") == manifest.get("case_hash"),
        "V14 reserve geometry result binding changed",
    )
    application = _read_json(paths[2])
    _require(
        application.get("artifact_kind") == RESERVE_GEOMETRY_APPLICATION_KIND
        and application.get("protocol_id") == RESERVE_GEOMETRY_APPLICATION_ID
        and application.get("status") == "reserve_geometry_child_applied"
        and application.get("queue_rank") == rank
        and application.get("object_hash") == manifest.get("object_hash")
        and application.get("case_hash") == manifest.get("case_hash")
        and application.get("reserve_geometry_config_sha256")
        == protocol["config_sha256"]
        and application.get("geometry_result_artifact_sha256")
        == result["artifact_sha256"]
        and application.get("geometry_result_file_sha256")
        == file_sha256(paths[1])
        and application.get("artifact_sha256")
        == canonical_sha256(
            application,
            namespace=(
                b"deform360-causal-response-direct-depth-reserve-geometry-"
                b"application-v14-v1\0"
            ),
            digest_key="artifact_sha256",
        ),
        "V14 reserve geometry application binding changed",
    )
    boundary = application.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("maximum_object_observation_frame") == 57
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve geometry application crossed its information boundary",
    )
    return manifest, result, application


def load_v14_reserve_batch_protocol(
    path: str | Path,
    *,
    method_protocol_path: str | Path | None = None,
    asset_protocol_path: str | Path | None = None,
    queue_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the fixed, outcome-blind reserve batch."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RESERVE_BATCH_KIND
        and payload.get("protocol_id") == RESERVE_BATCH_ID
        and payload.get("status")
        == "locked_before_reserve_prefix_mask_generation",
        "V14 reserve batch identity changed",
    )
    _require(
        payload.get("config_sha256") == _canonical_sha256(payload),
        "V14 reserve batch checksum changed",
    )
    contract = payload.get("batch_contract")
    _require(
        isinstance(contract, Mapping)
        and tuple(contract.get("batch_queue_ranks", ())) == RESERVE_BATCH_RANKS
        and contract.get("batch_size") == len(RESERVE_BATCH_RANKS)
        and contract.get("admissions_required") == RESERVE_ADMISSIONS_REQUIRED
        and contract.get("fixed_before_prefix_mask_generation") is True
        and contract.get("selection_rule")
        == "take the first four admitted candidates in immutable queue order",
        "V14 reserve selection contract changed",
    )
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents) == {"method_protocol", "prefix_assets", "staging_queue"}
        and all(
            isinstance(record, Mapping)
            and all(
                _valid_digest(value)
                for key, value in record.items()
                if key.endswith("sha256")
            )
            for record in parents.values()
        ),
        "V14 reserve parent bindings changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("prefix_mask_or_geometry_read_before_lock") is False
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve batch crossed its information boundary",
    )

    path_bindings = (
        (method_protocol_path, parents["method_protocol"], "config_sha256"),
        (asset_protocol_path, parents["prefix_assets"], "config_sha256"),
    )
    for parent_path, binding, semantic_key in path_bindings:
        if parent_path is None:
            continue
        parent = _read_json(parent_path)
        _require(
            parent.get(semantic_key) == binding[semantic_key]
            and file_sha256(parent_path) == binding["file_sha256"],
            "V14 reserve batch uses another parent protocol",
        )
    if queue_path is not None:
        queue = validate_v14_staging_queue(queue_path)
        binding = parents["staging_queue"]
        _require(
            queue["queue_sha256"] == binding["queue_sha256"]
            and file_sha256(queue_path) == binding["file_sha256"],
            "V14 reserve batch uses another staging queue",
        )
        _require(
            RESERVE_BATCH_RANKS[-1] <= len(queue["candidates"]),
            "V14 reserve ranks exceed the frozen queue",
        )
    return payload


__all__ = [
    "RESERVE_ADMISSIONS_REQUIRED",
    "RESERVE_BATCH_ID",
    "RESERVE_BATCH_KIND",
    "RESERVE_BATCH_RANKS",
    "RESERVE_GEOMETRY_APPLICATION_ID",
    "RESERVE_GEOMETRY_APPLICATION_KIND",
    "RESERVE_GEOMETRY_CONTRACT",
    "RESERVE_GEOMETRY_ID",
    "RESERVE_GEOMETRY_KIND",
    "RESERVE_GEOMETRY_RANKS",
    "load_v14_reserve_batch_protocol",
    "load_v14_reserve_geometry_protocol",
    "reserve_geometry_binding_for_rank",
    "validate_v14_reserve_geometry_bundle",
    "validate_v14_reserve_geometry_mask_input",
]
