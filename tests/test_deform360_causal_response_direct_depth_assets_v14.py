from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    MASK_ARTIFACT_KIND,
    MASK_CONTRACT,
    PREFIX_FRAME_COUNT,
    canonical_sha256,
    load_v14_asset_protocol,
    validate_v14_prefix_mask_artifact,
    validate_v14_staged_window,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    deform360_object_hash,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14.json"
)
ASSETS = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_assets.json"
)
ASSET_MODULE = ROOT / "src" / "bayesian_phystwin" / (
    "deform360_causal_response_direct_depth_assets.py"
)
MASK_BUILDER = ROOT / "scripts" / "remote" / (
    "build_deform360_causal_response_direct_depth_v14_masks.py"
)
QUEUE = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_staging_queue.json"
)


def test_v14_asset_protocol_locks_prefix_only_camera_assets() -> None:
    protocol = load_v14_asset_protocol(ASSETS)

    assert protocol["parent_method"]["file_sha256"] == file_sha256(METHOD)
    assert (
        protocol["causal_prefix_amendment"][
            "camera_rgb_mask_depth_frame_count"
        ]
        == PREFIX_FRAME_COUNT
    )
    assert (
        protocol["causal_prefix_amendment"][
            "future_camera_assets_created_before_prediction_seal"
        ]
        is False
    )
    assert protocol["staging"]["technical_failure_ranks_preserved"] == [1, 2]
    assert protocol["mask"]["asset_module_source_sha256"] == file_sha256(
        ASSET_MODULE
    )
    assert protocol["mask"]["mask_builder_source_sha256"] == file_sha256(
        MASK_BUILDER
    )


def test_v14_stage_and_prefix_masks_preserve_hash_custody(
    tmp_path: Path,
) -> None:
    method = json.loads(METHOD.read_text(encoding="utf-8"))
    assets = load_v14_asset_protocol(ASSETS)
    queue = validate_v14_staging_queue(QUEUE)
    rank = 3
    candidate = queue["candidates"][rank - 1]
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    episode = tmp_path / "stage"
    camera_records = []
    for camera in REGISTERED_CAMERA_IDS:
        video = episode / camera / "undistorted.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(camera.encode("utf-8"))
        camera_records.append(
            {
                "camera": camera,
                "decoded_frame_count": 81,
                "video_sha256": file_sha256(video),
            }
        )
    stage = {
        "artifact_kind": "Deform360CausalDirectDepthWindowStageV14",
        "contract": "deform360-causal-response-direct-depth-window-v14",
        "status": "staged",
        "protocol_id": method["protocol_id"],
        "protocol_config_sha256": method["config_sha256"],
        "queue_sha256": queue["queue_sha256"],
        "queue_rank": rank,
        "object_hash": deform360_object_hash(object_id),
        "case_hash": deform360_v14_case_hash(object_id, episode_id),
        "repository_revision": assets["staging"][
            "successful_window_implementation_commit"
        ],
        "raw_frame_count": 81,
        "prediction_frame_count": 76,
        "camera_records": camera_records,
    }
    stage["artifact_sha256"] = canonical_sha256(
        stage,
        namespace=b"deform360-causal-response-direct-depth-window-v14\0",
        digest_key="artifact_sha256",
    )
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")

    validated, _ = validate_v14_staged_window(
        stage_path,
        protocol=method,
        asset_protocol=assets,
        queue=queue,
        queue_rank=rank,
        stage_episode=episode,
    )
    assert validated["artifact_sha256"] == stage["artifact_sha256"]

    mask_episode = tmp_path / "masks"
    mask_records = []
    for index, camera in enumerate(REGISTERED_CAMERA_IDS):
        if index < 8:
            camera_root = mask_episode / camera
            camera_root.mkdir(parents=True)
            video = camera_root / "prefix.mp4"
            mask = camera_root / "mask_refined.h5"
            video.write_bytes(f"video-{camera}".encode())
            mask.write_bytes(f"mask-{camera}".encode())
            mask_records.append(
                {
                    "camera": camera,
                    "status": "success",
                    "frame_count": PREFIX_FRAME_COUNT,
                    "prefix_video_sha256": file_sha256(video),
                    "mask_sha256": file_sha256(mask),
                }
            )
        else:
            mask_records.append(
                {"camera": camera, "status": "technical_failure"}
            )
    mask_artifact = {
        "artifact_kind": MASK_ARTIFACT_KIND,
        "contract": MASK_CONTRACT,
        "protocol_id": assets["protocol_id"],
        "asset_protocol_config_sha256": assets["config_sha256"],
        "status": "ready_for_prefix_geometry",
        "input_camera_count": len(mask_records),
        "successful_camera_count": 8,
        "camera_records": mask_records,
    }
    mask_artifact["artifact_sha256"] = canonical_sha256(
        mask_artifact,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-masks-v14\0"
        ),
        digest_key="artifact_sha256",
    )
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask_artifact), encoding="utf-8")

    validated_mask = validate_v14_prefix_mask_artifact(
        mask_path,
        asset_protocol=assets,
        mask_episode=mask_episode,
    )
    assert validated_mask["successful_camera_count"] == 8
