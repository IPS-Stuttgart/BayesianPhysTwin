"""Physical and action custody for the V14 reserve source batch."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_causal_response_direct_depth_assets import (
    PREFIX_FRAME_COUNT,
    canonical_sha256,
)
from .deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from .deform360_causal_response_direct_depth_physical import (
    GRAPH_BASIS_RANK,
    METHOD_PROTOCOL_ID,
    PHYSICAL_FRAME_COUNT,
)
from .deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
)
from .deform360_causal_response_direct_depth_reserve_v14 import (
    RESERVE_GEOMETRY_RANKS,
    load_v14_reserve_geometry_protocol,
)
from .deform360_causal_response_prefix_geometry import (
    GEOMETRY_CONTRACT,
    GEOMETRY_MANIFEST_KIND,
    GEOMETRY_PROTOCOL_ID,
    GEOMETRY_RESULT_KIND,
)
from .deform360_causal_response_preflight import deform360_object_hash
from .deform360_fresh_pairwise_physical import CANONICAL_NODE_COUNT
from .deform360_object_exclusion import file_sha256

RESERVE_PHYSICAL_PRELOCK_KIND = (
    "Deform360CausalResponseDirectDepthReservePhysicalPrelockProtocolV14V1"
)
RESERVE_PHYSICAL_PRELOCK_CONTRACT = (
    "deform360-causal-response-direct-depth-reserve-physical-prelock-v14-v1"
)
RESERVE_PHYSICAL_PRELOCK_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-physical-prelock-v1"
)
RESERVE_PHYSICAL_RUNTIME_KIND = (
    "Deform360CausalResponseDirectDepthReservePhysicalRuntimeV14V1"
)
RESERVE_PHYSICAL_RUNTIME_CONTRACT = (
    "deform360-causal-response-direct-depth-reserve-physical-runtime-v14-v1"
)
RESERVE_PHYSICAL_RUNTIME_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-physical-runtime-v1"
)
RESERVE_GEOMETRY_RUNTIME_V2_KIND = (
    "Deform360CausalResponseDirectDepthReserveGeometryRuntimeV14V2"
)
RESERVE_GEOMETRY_RUNTIME_V2_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-geometry-runtime-v2"
)
RESERVE_GEOMETRY_APPLICATION_V2_KIND = (
    "Deform360CausalResponseDirectDepthReserveGeometryApplicationV14V2"
)
RESERVE_GEOMETRY_APPLICATION_V2_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-geometry-application-v2"
)

_CASE_DIGEST_FIELDS = (
    "case_hash",
    "object_hash",
    "metadata_sha256",
    "geometry_manifest_artifact_sha256",
    "geometry_manifest_file_sha256",
    "geometry_result_artifact_sha256",
    "geometry_result_file_sha256",
    "runtime_application_artifact_sha256",
    "runtime_application_file_sha256",
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


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read V14 reserve physical JSON: {source}") from error
    _require(
        isinstance(payload, dict),
        "V14 reserve physical JSON is not an object",
    )
    return payload


def _canonical_sha256(payload: Mapping[str, Any], *, namespace: bytes) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def reserve_geometry_ledger_sha256(records: list[dict[str, Any]]) -> str:
    """Hash the ordered reserve geometry ledger."""

    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-reserve-physical-geometry-"
        b"v14-v1\0"
        + json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_v14_reserve_physical_prelock(
    path: str | Path,
) -> dict[str, Any]:
    """Validate the reserve geometry ledger before physical execution."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RESERVE_PHYSICAL_PRELOCK_KIND
        and payload.get("contract") == RESERVE_PHYSICAL_PRELOCK_CONTRACT
        and payload.get("protocol_id") == RESERVE_PHYSICAL_PRELOCK_ID
        and payload.get("method_protocol_id") == METHOD_PROTOCOL_ID
        and payload.get("status")
        == "locked_after_reserve_geometry_before_physical_execution",
        "V14 reserve physical prelock identity changed",
    )
    _require(
        payload.get("config_sha256")
        == _canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-reserve-physical-"
                b"prelock-v14-v1\0"
            ),
        ),
        "V14 reserve physical prelock checksum changed",
    )
    parents = payload.get("parent_artifacts")
    required_flat = {
        "geometry_protocol_file_sha256",
        "runtime_v1_file_sha256",
        "runtime_v2_file_sha256",
        "staging_queue_file_sha256",
        "staging_queue_sha256",
        "validation_v1_file_sha256",
        "validation_v2_file_sha256",
    }
    _require(
        isinstance(parents, Mapping)
        and required_flat.issubset(parents)
        and all(_valid_digest(parents[key]) for key in required_flat)
        and all(
            isinstance(parents.get(key), Mapping)
            and _valid_digest(parents[key].get("config_sha256"))
            and _valid_digest(parents[key].get("file_sha256"))
            for key in (
                "parent_physical_prelock",
                "reserve_batch",
                "reserve_geometry",
                "reserve_geometry_runtime_v2",
            )
        ),
        "V14 reserve physical parent bindings changed",
    )
    implementation = payload.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and isinstance(implementation.get("file_sha256"), Mapping)
        and set(implementation["file_sha256"])
        == {
            "artifact_module",
            "automatic_twin",
            "parent_physical_runner",
            "reserve_physical_module",
            "reserve_physical_runner",
        }
        and all(
            _valid_digest(value) for value in implementation["file_sha256"].values()
        ),
        "V14 reserve physical implementation binding changed",
    )
    numerical = payload.get("numerical_contract")
    _require(
        isinstance(numerical, Mapping)
        and numerical.get("canonical_node_count") == CANONICAL_NODE_COUNT
        and numerical.get("graph_basis_rank") == GRAPH_BASIS_RANK
        and numerical.get("prediction_frame_count") == PHYSICAL_FRAME_COUNT
        and numerical.get("automatic_twin_source") == "frame_zero_geometry_only"
        and numerical.get("future_robot_action_known") is True
        and numerical.get("automatic_twin_inadmissible_fallback")
        == "bit_exact_persistence",
        "V14 reserve physical numerical contract changed",
    )
    cases = payload.get("geometry_cases")
    _require(
        isinstance(cases, list)
        and tuple(int(record.get("queue_rank", 0)) for record in cases)
        == RESERVE_GEOMETRY_RANKS,
        "V14 reserve physical geometry ranks changed",
    )
    for record in cases:
        _require(
            record.get("runtime_contract_version") == "reserve-v2"
            and 128 <= int(record.get("physical_node_count", 0)) <= 10_000
            and 8 <= int(record.get("successful_camera_count", 0)) <= 12
            and all(_valid_digest(record.get(key)) for key in _CASE_DIGEST_FIELDS),
            "V14 reserve physical geometry record changed",
        )
    _require(
        payload.get("geometry_ledger_sha256")
        == reserve_geometry_ledger_sha256(cases),
        "V14 reserve physical geometry ledger checksum changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("object_observation_frames_used") == [0]
        and boundary.get("known_robot_action_frames_used")
        == list(range(PHYSICAL_FRAME_COUNT))
        and boundary.get("future_object_observation_read") is False
        and boundary.get("prefix_tactile_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("source_lock_required_before_execution") is False
        and boundary.get("source_lock_construction_uses_output_hashes_only") is True
        and boundary.get("plaintext_identity_retained_in_sealed_output") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve physical prelock crossed its information boundary",
    )
    return payload


def v14_reserve_physical_case_record(
    protocol: Mapping[str, Any],
    queue: Mapping[str, Any] | str | Path,
    *,
    queue_rank: int,
) -> dict[str, Any]:
    """Return one hash-only reserve physical case binding."""

    normalized_queue = validate_v14_staging_queue(queue)
    _require(
        normalized_queue["queue_sha256"]
        == protocol["parent_artifacts"]["staging_queue_sha256"],
        "V14 reserve physical queue semantic checksum changed",
    )
    geometry = next(
        (
            record
            for record in protocol["geometry_cases"]
            if int(record["queue_rank"]) == int(queue_rank)
        ),
        None,
    )
    _require(geometry is not None, "V14 reserve rank is not geometry-bound")
    candidate = normalized_queue["candidates"][queue_rank - 1]
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    object_hash = deform360_object_hash(object_id)
    case_hash = deform360_v14_case_hash(object_id, episode_id)
    _require(
        geometry["object_hash"] == object_hash
        and geometry["case_hash"] == case_hash
        and geometry["metadata_sha256"] == candidate["metadata_sha256"],
        "V14 reserve geometry differs from the frozen queue",
    )
    return {
        "queue_rank": int(queue_rank),
        "object_hash": object_hash,
        "case_hash": case_hash,
        "category": str(candidate["category"]),
        "bimanual_value": str(candidate["bimanual"]),
        "metadata_sha256": str(candidate["metadata_sha256"]),
        "physical_node_count": int(geometry["physical_node_count"]),
        "successful_camera_count": int(geometry["successful_camera_count"]),
        "runtime_contract_version": geometry["runtime_contract_version"],
        **{
            key: str(geometry[key])
            for key in _CASE_DIGEST_FIELDS
            if key not in {"case_hash", "object_hash", "metadata_sha256"}
        },
    }


def load_v14_reserve_physical_runtime(
    path: str | Path,
    *,
    parent_prelock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the reserve known-action ledger."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RESERVE_PHYSICAL_RUNTIME_KIND
        and payload.get("contract") == RESERVE_PHYSICAL_RUNTIME_CONTRACT
        and payload.get("protocol_id") == RESERVE_PHYSICAL_RUNTIME_ID
        and payload.get("status")
        == "locked_before_reserve_physical_execution",
        "V14 reserve physical runtime identity changed",
    )
    _require(
        payload.get("config_sha256")
        == _canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-reserve-physical-"
                b"runtime-v14-v1\0"
            ),
        ),
        "V14 reserve physical runtime checksum changed",
    )
    parent = payload.get("parent_physical_prelock")
    _require(
        isinstance(parent, Mapping)
        and _valid_digest(parent.get("config_sha256"))
        and _valid_digest(parent.get("file_sha256")),
        "V14 reserve physical runtime parent binding changed",
    )
    if parent_prelock_path is not None:
        prelock = _read_json(parent_prelock_path)
        _require(
            prelock.get("config_sha256") == parent["config_sha256"]
            and file_sha256(parent_prelock_path) == parent["file_sha256"],
            "V14 reserve physical runtime uses another prelock",
        )
    action = payload.get("action_contract")
    _require(
        isinstance(action, Mapping)
        and action.get("known_action_source")
        == "exact_action_only_staged_window_robot"
        and action.get("accepted_staged_frame_counts") == [76, 81]
        and action.get("physical_frame_count") == PHYSICAL_FRAME_COUNT
        and action.get("object_observation_source") == "frame_zero_only",
        "V14 reserve physical action contract changed",
    )
    cases = payload.get("action_cases")
    _require(
        isinstance(cases, list)
        and tuple(int(record.get("queue_rank", 0)) for record in cases)
        == RESERVE_GEOMETRY_RANKS,
        "V14 reserve physical action ranks changed",
    )
    for record in cases:
        _require(
            all(
                _valid_digest(record.get(key))
                for key in (
                    "object_hash",
                    "case_hash",
                    "window_stage_artifact_sha256",
                    "window_stage_file_sha256",
                    "known_action_file_sha256",
                )
            )
            and record.get("staged_frame_count") in {76, 81},
            "V14 reserve physical action record changed",
        )
    implementation = payload.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and set(implementation.get("file_sha256", {}))
        == {"physical_runner", "reserve_runner", "runtime_module"}
        and all(
            _valid_digest(value)
            for value in implementation["file_sha256"].values()
        ),
        "V14 reserve physical runtime implementation changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("known_future_robot_action_read") is True
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve physical runtime crossed its information boundary",
    )
    return payload


def validate_v14_reserve_physical_action(
    protocol: Mapping[str, Any],
    *,
    queue_rank: int,
    object_hash: str,
    case_hash: str,
    window_stage_result_path: str | Path,
    known_action_path: str | Path,
    staged_frame_count: int,
) -> dict[str, Any]:
    """Validate one reserve action against its exact staged ledger."""

    record = next(
        (
            item
            for item in protocol["action_cases"]
            if int(item["queue_rank"]) == int(queue_rank)
        ),
        None,
    )
    _require(record is not None, "V14 reserve action rank is outside the ledger")
    _require(
        record["object_hash"] == object_hash
        and record["case_hash"] == case_hash
        and record["window_stage_file_sha256"]
        == file_sha256(window_stage_result_path)
        and record["known_action_file_sha256"] == file_sha256(known_action_path)
        and record["staged_frame_count"] == int(staged_frame_count),
        "V14 reserve physical action differs from its ledger",
    )
    stage = _read_json(window_stage_result_path)
    _require(
        stage.get("artifact_sha256") == record["window_stage_artifact_sha256"]
        and stage.get("status") == "staged"
        and stage.get("queue_rank") == int(queue_rank)
        and stage.get("object_hash") == object_hash
        and stage.get("case_hash") == case_hash,
        "V14 reserve physical action uses another staged window",
    )
    return dict(record)


def validate_v14_reserve_geometry_bundle_v2(
    *,
    manifest_path: str | Path,
    result_path: str | Path,
    application_path: str | Path,
    geometry_protocol_path: str | Path,
    reserve_batch_path: str | Path,
    runtime_v2_path: str | Path,
    geometry_episode: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate one reserve geometry bundle used by the physical child."""

    geometry_protocol = load_v14_reserve_geometry_protocol(
        geometry_protocol_path,
        reserve_batch_path=reserve_batch_path,
    )
    runtime_v2 = _read_json(runtime_v2_path)
    _require(
        runtime_v2.get("artifact_kind") == RESERVE_GEOMETRY_RUNTIME_V2_KIND
        and runtime_v2.get("protocol_id") == RESERVE_GEOMETRY_RUNTIME_V2_ID
        and runtime_v2.get("config_sha256")
        == _canonical_sha256(
            runtime_v2,
            namespace=(
                b"deform360-causal-response-direct-depth-reserve-geometry-"
                b"runtime-v14-v2\0"
            ),
        )
        and runtime_v2.get("parent_reserve_geometry", {}).get("config_sha256")
        == geometry_protocol["config_sha256"]
        and runtime_v2.get("parent_reserve_geometry", {}).get("file_sha256")
        == file_sha256(geometry_protocol_path),
        "V14 reserve geometry runtime-v2 parent changed",
    )
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("artifact_kind") == GEOMETRY_MANIFEST_KIND
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
        "V14 reserve geometry manifest changed",
    )
    rank = int(manifest.get("queue_rank", 0))
    _require(
        rank in RESERVE_GEOMETRY_RANKS
        and 128 <= int(manifest.get("physical_node_count", 0)) <= 10_000,
        "V14 reserve geometry manifest rank or node count changed",
    )
    episode = Path(geometry_episode)
    outputs = manifest.get("outputs_sha256")
    fixed = {
        "intrinsics": episode / "undistorted_intrinsics.npy",
        "extrinsics": episode / "extrinsics.npy",
        "robot": episode / "robot" / "robot.npz",
        "frame_zero_splat": episode / "splatfacto" / "splat_0.ply",
        "frame_zero_points": episode / "start_obj_pcd.ply",
    }
    _require(
        isinstance(outputs, Mapping)
        and all(
            path.is_file() and file_sha256(path) == outputs.get(role)
            for role, path in fixed.items()
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
        and isinstance(depth_outputs, Mapping)
        and set(cameras) == set(depth_outputs)
        and all(
            file_sha256(episode / camera / "rendered_depth.h5")
            == depth_outputs[camera]
            for camera in cameras
        ),
        "V14 reserve geometry camera outputs changed",
    )
    runtime = manifest.get("runtime")
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("gsplat_extension_sha256")
        == geometry_protocol["runtime"]["gsplat_extension_sha256"]
        and runtime.get("python_version")
        == geometry_protocol["runtime"]["python_version"]
        and runtime.get("torch_version")
        == geometry_protocol["runtime"]["torch_version"],
        "V14 reserve geometry runtime provenance changed",
    )
    result = _read_json(result_path)
    _require(
        result.get("artifact_kind") == GEOMETRY_RESULT_KIND
        and result.get("contract") == GEOMETRY_CONTRACT
        and result.get("protocol_id") == GEOMETRY_PROTOCOL_ID
        and result.get("geometry_protocol_config_sha256")
        == geometry_protocol["config_sha256"]
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
        and result.get("geometry_manifest_file_sha256")
        == file_sha256(manifest_path)
        and result.get("queue_rank") == rank
        and result.get("object_hash") == manifest.get("object_hash")
        and result.get("case_hash") == manifest.get("case_hash"),
        "V14 reserve geometry result changed",
    )
    application = _read_json(application_path)
    _require(
        application.get("artifact_kind") == RESERVE_GEOMETRY_APPLICATION_V2_KIND
        and application.get("protocol_id") == RESERVE_GEOMETRY_APPLICATION_V2_ID
        and application.get("status") == "reserve_geometry_runtime_v2_applied"
        and application.get("runtime_v2_config_sha256")
        == runtime_v2["config_sha256"]
        and application.get("runtime_v2_file_sha256")
        == file_sha256(runtime_v2_path)
        and application.get("reserve_geometry_config_sha256")
        == geometry_protocol["config_sha256"]
        and application.get("reserve_geometry_file_sha256")
        == file_sha256(geometry_protocol_path)
        and application.get("geometry_result_artifact_sha256")
        == result["artifact_sha256"]
        and application.get("geometry_result_file_sha256")
        == file_sha256(result_path)
        and application.get("queue_rank") == rank
        and application.get("object_hash") == manifest.get("object_hash")
        and application.get("case_hash") == manifest.get("case_hash")
        and application.get("artifact_sha256")
        == canonical_sha256(
            application,
            namespace=(
                b"deform360-causal-response-direct-depth-reserve-geometry-"
                b"application-v14-v2\0"
            ),
            digest_key="artifact_sha256",
        ),
        "V14 reserve geometry runtime-v2 application changed",
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


__all__ = [
    "RESERVE_PHYSICAL_PRELOCK_CONTRACT",
    "RESERVE_PHYSICAL_PRELOCK_ID",
    "RESERVE_PHYSICAL_PRELOCK_KIND",
    "RESERVE_PHYSICAL_RUNTIME_CONTRACT",
    "RESERVE_PHYSICAL_RUNTIME_ID",
    "RESERVE_PHYSICAL_RUNTIME_KIND",
    "load_v14_reserve_physical_prelock",
    "load_v14_reserve_physical_runtime",
    "reserve_geometry_ledger_sha256",
    "v14_reserve_physical_case_record",
    "validate_v14_reserve_geometry_bundle_v2",
    "validate_v14_reserve_physical_action",
]
