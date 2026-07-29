"""Causal prefix-asset custody for the V14 Deform360 source study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
)
from .deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    deform360_object_hash,
)
from .deform360_object_exclusion import file_sha256

ASSET_PROTOCOL_KIND = "Deform360CausalDirectDepthAssetProtocolV14"
ASSET_PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-prefix-assets"
MASK_ARTIFACT_KIND = "Deform360CausalDirectDepthPrefixMasksV14"
MASK_CONTRACT = "deform360-causal-response-direct-depth-prefix-masks-v14"
STAGE_ARTIFACT_KIND = "Deform360CausalDirectDepthWindowStageV14"
STAGE_CONTRACT = "deform360-causal-response-direct-depth-window-v14"
METHOD_PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-source"
PREFIX_FRAME_COUNT = 58
RAW_FRAME_COUNT = 81
PREDICTION_FRAME_COUNT = 76


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


def canonical_sha256(
    payload: Mapping[str, Any],
    *,
    namespace: bytes,
    digest_key: str,
) -> str:
    """Hash one canonical JSON artifact after removing its self digest."""

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


def load_v14_asset_protocol(path: str | Path) -> dict[str, Any]:
    """Validate the prefix-only V14 asset amendment."""

    source = Path(path)
    payload = _read_json(source)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == ASSET_PROTOCOL_KIND
        and payload.get("protocol_id") == ASSET_PROTOCOL_ID
        and payload.get("status")
        == "locked_after_source_window_staging_before_object_prefix_decode",
        "V14 asset protocol identity changed",
    )
    _require(
        payload.get("config_sha256")
        == canonical_sha256(
            payload,
            namespace=b"deform360-causal-response-direct-depth-assets-v14\0",
            digest_key="config_sha256",
        ),
        "V14 asset protocol checksum changed",
    )
    amendment = payload.get("causal_prefix_amendment")
    _require(
        isinstance(amendment, Mapping)
        and amendment.get("camera_rgb_mask_depth_frame_count")
        == PREFIX_FRAME_COUNT
        and amendment.get("maximum_object_observation_frame")
        == PREFIX_FRAME_COUNT - 1
        and amendment.get("robot_tactile_and_prediction_frame_count")
        == PREDICTION_FRAME_COUNT
        and amendment.get("future_camera_assets_created_before_prediction_seal")
        is False
        and amendment.get("method_threshold_or_gate_changed") is False,
        "V14 causal prefix amendment changed",
    )
    staging = payload.get("staging")
    _require(
        isinstance(staging, Mapping)
        and staging.get("raw_frame_count") == RAW_FRAME_COUNT
        and staging.get("prediction_frame_count") == PREDICTION_FRAME_COUNT
        and staging.get("technical_failure_ranks_preserved") == [1, 2]
        and staging.get("first_unfailed_rank") == 3,
        "V14 asset staging boundary changed",
    )
    mask = payload.get("mask")
    _require(
        isinstance(mask, Mapping)
        and mask.get("camera_count") == len(REGISTERED_CAMERA_IDS)
        and mask.get("minimum_successful_camera_count") == 8
        and mask.get("manual_prompting") is False,
        "V14 mask contract changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("object_rgb_mask_or_depth_after_frame_57_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 asset protocol crossed its information boundary",
    )
    return payload


def validate_v14_staged_window(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    asset_protocol: Mapping[str, Any],
    queue: Mapping[str, Any],
    queue_rank: int,
    stage_episode: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one successful staged source window and its exact files."""

    _require(
        1 <= queue_rank <= len(queue["candidates"]),
        "V14 staged rank is outside the queue",
    )
    candidate = queue["candidates"][queue_rank - 1]
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    payload = _read_json(path)
    _require(
        payload.get("artifact_kind") == STAGE_ARTIFACT_KIND
        and payload.get("contract") == STAGE_CONTRACT
        and payload.get("status") == "staged"
        and payload.get("protocol_id") == METHOD_PROTOCOL_ID
        and payload.get("protocol_config_sha256") == protocol["config_sha256"]
        and payload.get("queue_sha256") == queue["queue_sha256"]
        and payload.get("queue_rank") == queue_rank
        and payload.get("object_hash") == deform360_object_hash(object_id)
        and payload.get("case_hash")
        == deform360_v14_case_hash(object_id, episode_id)
        and payload.get("repository_revision")
        == asset_protocol["staging"]["successful_window_implementation_commit"]
        and payload.get("raw_frame_count") == RAW_FRAME_COUNT
        and payload.get("prediction_frame_count") == PREDICTION_FRAME_COUNT
        and payload.get("artifact_sha256")
        == canonical_sha256(
            payload,
            namespace=b"deform360-causal-response-direct-depth-window-v14\0",
            digest_key="artifact_sha256",
        ),
        "V14 staged window binding changed",
    )
    records = payload.get("camera_records")
    _require(
        isinstance(records, list)
        and tuple(row.get("camera") for row in records)
        == REGISTERED_CAMERA_IDS,
        "V14 staged camera panel changed",
    )
    root = Path(stage_episode)
    for row in records:
        camera = str(row["camera"])
        video = root / camera / "undistorted.mp4"
        _require(
            row.get("decoded_frame_count") == RAW_FRAME_COUNT
            and video.is_file()
            and file_sha256(video) == row.get("video_sha256"),
            f"V14 staged camera changed: {camera}",
        )
    return payload, candidate


def validate_v14_prefix_mask_artifact(
    path: str | Path,
    *,
    asset_protocol: Mapping[str, Any],
    mask_episode: str | Path,
) -> dict[str, Any]:
    """Validate one prefix-only V14 mask artifact and its successful masks."""

    payload = _read_json(path)
    _require(
        payload.get("artifact_kind") == MASK_ARTIFACT_KIND
        and payload.get("contract") == MASK_CONTRACT
        and payload.get("protocol_id") == ASSET_PROTOCOL_ID
        and payload.get("asset_protocol_config_sha256")
        == asset_protocol["config_sha256"]
        and payload.get("artifact_sha256")
        == canonical_sha256(
            payload,
            namespace=b"deform360-causal-response-direct-depth-prefix-masks-v14\0",
            digest_key="artifact_sha256",
        ),
        "V14 prefix mask artifact binding changed",
    )
    records = payload.get("camera_records")
    _require(
        isinstance(records, list)
        and tuple(row.get("camera") for row in records)
        == REGISTERED_CAMERA_IDS
        and payload.get("input_camera_count") == len(REGISTERED_CAMERA_IDS),
        "V14 prefix mask camera panel changed",
    )
    successful = [row for row in records if row.get("status") == "success"]
    expected_status = (
        "ready_for_prefix_geometry"
        if len(successful)
        >= int(asset_protocol["mask"]["minimum_successful_camera_count"])
        else "technical_preflight_failure"
    )
    _require(
        payload.get("status") == expected_status
        and payload.get("successful_camera_count") == len(successful),
        "V14 prefix mask disposition changed",
    )
    root = Path(mask_episode)
    for row in successful:
        camera = str(row["camera"])
        mask = root / camera / "mask_refined.h5"
        video = root / camera / "prefix.mp4"
        _require(
            row.get("frame_count") == PREFIX_FRAME_COUNT
            and mask.is_file()
            and video.is_file()
            and file_sha256(mask) == row.get("mask_sha256")
            and file_sha256(video) == row.get("prefix_video_sha256"),
            f"V14 prefix mask output changed: {camera}",
        )
    return payload


__all__ = [
    "ASSET_PROTOCOL_ID",
    "ASSET_PROTOCOL_KIND",
    "MASK_ARTIFACT_KIND",
    "MASK_CONTRACT",
    "PREFIX_FRAME_COUNT",
    "PREDICTION_FRAME_COUNT",
    "RAW_FRAME_COUNT",
    "canonical_sha256",
    "load_v14_asset_protocol",
    "validate_v14_prefix_mask_artifact",
    "validate_v14_staged_window",
]
