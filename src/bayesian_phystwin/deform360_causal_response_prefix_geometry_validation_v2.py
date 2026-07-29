"""Validated bundle custody for interpreter-bound V14 prefix geometry."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_causal_response_direct_depth_assets import (
    PREFIX_FRAME_COUNT,
    canonical_sha256,
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

VALIDATION_KIND = "Deform360CausalDirectDepthPrefixGeometryValidationV14V2"
VALIDATION_ID = (
    "deform360-causal-response-direct-depth-v14-prefix-geometry-validation-v2"
)
RUNTIME_APPLICATION_KIND = (
    "Deform360CausalDirectDepthPrefixGeometryRuntimeApplicationV14V2"
)
RUNTIME_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-prefix-geometry-runtime-v2"
)


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


def load_v14_prefix_geometry_validation_v2(path: str | Path) -> dict[str, Any]:
    """Validate the runtime-v2 bundle-custody amendment."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == VALIDATION_KIND
        and payload.get("protocol_id") == VALIDATION_ID
        and payload.get("status")
        == "locked_after_runtime_v2_smoke_before_source_lock",
        "V14 prefix geometry validation-v2 identity changed",
    )
    _require(
        payload.get("config_sha256")
        == canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-"
                b"validation-v14-v2\0"
            ),
            digest_key="config_sha256",
        ),
        "V14 prefix geometry validation-v2 checksum changed",
    )
    trigger = payload.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("queue_rank") == 4
        and trigger.get("status") == "ready_for_source_lock"
        and trigger.get("manifest_or_output_bytes_changed") is False
        and trigger.get("future_identity_or_metric_read") is False
        and trigger.get("target_object_or_outcome_read") is False
        and trigger.get("held_v8_access") is False,
        "V14 prefix geometry validation-v2 trigger changed",
    )
    policy = payload.get("validation_policy")
    _require(
        isinstance(policy, Mapping)
        and policy.get("canonical_frame_zero_splat_filename") == "splat_0.ply"
        and policy.get("applies_to_queue_ranks") == list(range(4, 15))
        and policy.get("existing_artifacts_are_not_rewritten") is True
        and policy.get("method_or_gate_changed") is False,
        "V14 prefix geometry validation-v2 policy changed",
    )
    for key in (
        "parent_geometry_protocol",
        "parent_runtime_v2_amendment",
        "parent_validation_amendment",
    ):
        parent = payload.get(key)
        _require(
            isinstance(parent, Mapping)
            and isinstance(parent.get("config_sha256"), str)
            and isinstance(parent.get("file_sha256"), str),
            "V14 prefix geometry validation-v2 parent binding changed",
        )
    implementation = payload.get("implementation_file_sha256")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("validation_module_v2"), str),
        "V14 prefix geometry validation-v2 implementation binding changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 prefix geometry validation-v2 crossed its information boundary",
    )
    return payload


def validate_v14_prefix_geometry_bundle_v2(
    *,
    manifest_path: str | Path,
    result_path: str | Path,
    runtime_application_path: str | Path,
    geometry_protocol: Mapping[str, Any],
    runtime_amendment_v2: Mapping[str, Any],
    validation_amendment_v2: Mapping[str, Any],
    geometry_episode: str | Path,
    forbidden_plaintext: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate one runtime-v2 geometry bundle without mutation."""

    _require(
        validation_amendment_v2.get("artifact_kind") == VALIDATION_KIND
        and validation_amendment_v2.get("protocol_id") == VALIDATION_ID
        and geometry_protocol.get("config_sha256")
        == validation_amendment_v2["parent_geometry_protocol"]["config_sha256"]
        and runtime_amendment_v2.get("config_sha256")
        == validation_amendment_v2["parent_runtime_v2_amendment"][
            "config_sha256"
        ],
        "V14 prefix geometry validation-v2 uses another parent lock",
    )
    paths = tuple(
        Path(path)
        for path in (manifest_path, result_path, runtime_application_path)
    )
    if forbidden_plaintext is not None:
        _require(
            all(
                forbidden_plaintext not in path.read_text(encoding="utf-8")
                for path in paths
            ),
            "V14 prefix geometry runtime-v2 bundle leaked plaintext identity",
        )
    manifest = _read_json(paths[0])
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_kind") == GEOMETRY_MANIFEST_KIND
        and manifest.get("contract") == GEOMETRY_CONTRACT
        and manifest.get("protocol_id") == GEOMETRY_PROTOCOL_ID
        and manifest.get("geometry_protocol_config_sha256")
        == geometry_protocol["config_sha256"]
        and manifest.get("status") == "ready_for_physical_preflight"
        and manifest.get("artifact_sha256")
        == canonical_sha256(
            manifest,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-v14\0"
            ),
            digest_key="artifact_sha256",
        ),
        "V14 prefix geometry runtime-v2 manifest binding changed",
    )
    rank = int(manifest.get("queue_rank", 0))
    node_count = int(manifest.get("physical_node_count", 0))
    _require(
        rank in validation_amendment_v2["validation_policy"][
            "applies_to_queue_ranks"
        ]
        and MINIMUM_PHYSICAL_NODE_COUNT
        <= node_count
        <= MAXIMUM_PHYSICAL_NODE_COUNT,
        "V14 prefix geometry runtime-v2 rank or node count is inadmissible",
    )
    episode = Path(geometry_episode)
    outputs = manifest.get("outputs_sha256")
    _require(isinstance(outputs, Mapping), "V14 geometry outputs are missing")
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
        "V14 prefix geometry runtime-v2 fixed outputs changed",
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
        "V14 prefix geometry runtime-v2 camera panel changed",
    )
    runtime = manifest.get("runtime")
    runtime_parent = runtime_amendment_v2["parent_runtime_amendment"]
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("runtime_amendment_config_sha256")
        == runtime_amendment_v2["config_sha256"]
        and runtime.get("runtime_amendment_file_sha256")
        == validation_amendment_v2["parent_runtime_v2_amendment"]["file_sha256"]
        and runtime.get("parent_runtime_amendment_config_sha256")
        == runtime_parent["config_sha256"]
        and runtime.get("parent_runtime_amendment_file_sha256")
        == runtime_parent["file_sha256"]
        and runtime.get("gsplat_extension_sha256")
        == runtime_amendment_v2["runtime_amendment"][
            "interpreter_relinked_extension_sha256"
        ],
        "V14 prefix geometry runtime-v2 provenance changed",
    )
    result = _read_json(paths[1])
    _require(
        result.get("artifact_kind") == GEOMETRY_RESULT_KIND
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
        "V14 prefix geometry runtime-v2 result binding changed",
    )
    application = _read_json(paths[2])
    _require(
        application.get("artifact_kind") == RUNTIME_APPLICATION_KIND
        and application.get("protocol_id") == RUNTIME_PROTOCOL_ID
        and application.get("status") == "runtime_v2_amendment_applied"
        and application.get("artifact_sha256")
        == canonical_sha256(
            application,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-"
                b"runtime-application-v14-v2\0"
            ),
            digest_key="artifact_sha256",
        )
        and application.get("runtime_amendment_config_sha256")
        == runtime_amendment_v2["config_sha256"]
        and application.get("runtime_amendment_file_sha256")
        == validation_amendment_v2["parent_runtime_v2_amendment"]["file_sha256"]
        and application.get("parent_runtime_amendment_config_sha256")
        == runtime_parent["config_sha256"]
        and application.get("parent_runtime_amendment_file_sha256")
        == runtime_parent["file_sha256"]
        and application.get("geometry_result_artifact_sha256")
        == result["artifact_sha256"]
        and application.get("geometry_result_file_sha256")
        == file_sha256(paths[1]),
        "V14 prefix geometry runtime-v2 application changed",
    )
    return manifest, result, application


__all__ = [
    "RUNTIME_APPLICATION_KIND",
    "RUNTIME_PROTOCOL_ID",
    "VALIDATION_ID",
    "VALIDATION_KIND",
    "load_v14_prefix_geometry_validation_v2",
    "validate_v14_prefix_geometry_bundle_v2",
]
