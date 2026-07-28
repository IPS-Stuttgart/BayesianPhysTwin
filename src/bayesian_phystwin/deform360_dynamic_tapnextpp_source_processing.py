"""Frozen processing contract for dynamic TAPNext++ Deform360 sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_dynamic_tapnextpp_source_window import (
    FROZEN_CAMERA_PANEL,
    MASK_ARTIFACT_KIND,
    MASK_PROTOCOL_ID,
    PREDICTION_FRAME_COUNT,
    RAW_FRAME_COUNT,
    canonical_sha256,
    file_sha256,
)

PROCESSING_PROTOCOL_KIND = "Deform360DynamicTapNextppSourceProcessingProtocol"
PROCESSING_PROTOCOL_ID = "deform360-dynamic-tapnextpp-source-processing-v1"
PROCESSING_ARTIFACT_KIND = "Deform360DynamicTapNextppSourceProcessing"
RUNTIME_AMENDMENT_KIND = (
    "Deform360DynamicTapNextppSourceProcessingRuntimeAmendment"
)
RUNTIME_AMENDMENT_ID = (
    "deform360-dynamic-tapnextpp-source-processing-runtime-amendment-v1"
)
TERMINAL_FAILURE_RESULT_SHA256 = (
    "8bb42a03c7adf8f2d1a4a038df896eee7c0dcc54e98099f5fa34c9a1d0eacead",
    "2ba0483c6e44d1d987ffe9c84eb9ddfab02769757c5f9c0202f72eadc6a305b0",
    "e959a81e93a2c4ce4ccf948122ded6dfa3f8d733be7ca7b4d41a6dd0d3ac2",
    "3b60edf2df1da81478cbc02b3a9b34c5a66e8447310cad6a9b6bfdc8152ce2b8",
    "355a86d343afd2dbdded1865f335ae4dbea5f0c3802fe3946bcbdfe8b85b918d",
    "6ecf1a03bdf667f53600658bc844f079b914fefd7a55cec5ed33a75048f173ca",
    "11ba0fbf6240156ce67df86504139d1f25f9315c92bf0ed7a354907d6dcbf34c",
    "12f08e7854c117af3523f4f9fd5a645f8c34cef16101539777a2013c0bf3cb36",
)
TERMINAL_FAILURE_FILE_SHA256 = (
    "4976aa99f5d173f06b7a5372b2dfb19c2fc8172f500063be4fc3cc61a64cb52f",
    "e2093a2d45c8a0412fcf4e85e5dd42896cd29dea34113d587dbbb3975ab4ea5c",
    "f85609b1e50ecf10e8c922412601cbcb78421e36d5081203352f76ee9a419c68",
    "24a603e02f2a31aa471bf3ab56f20de9432fa3506e9952d18461c823e34f03b1",
    "4cb2be4e926a2f3f683aa799c4296a0c504fbbbc84b21a64200417524f6c2443",
    "d08b1f61f6ea89ba5ba06a7f4130d5d2c4cd9896a37d17bb9f113bd1e3a573d9",
    "4c8db3c6603e877b48550ee663541c0b2154329ca67c58e5c321264101385293",
    "9fe1bf8bcbe83ddb73e9749310060f9c8462847e906953dd50e722a727bd98df",
)
DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
DEFORM360_SOURCE_SHA256 = {
    "reconstruct_stage": (
        "53a1e8b73e56a1c68a0c4344b279c2817ed4b3ed93e8f5ea792def26d5099c7c"
    ),
    "urdf_render": ("c4d6a10e980ed4952f974d2e8a991c6fb819a3e6fdc6c121d3ce6925c94c2467"),
    "depth_stage": ("34befb732107b805f1e1924699f1e26fc2ca5d3041561b920d8c23d8e85feef0"),
    "tracking_stage": (
        "04533cd9cd900ae2f5bd139568ed1a2442661f14ceda009dd7bb85e4fbd83ec2"
    ),
    "pcd_stage": ("87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d"),
    "control_points_stage": (
        "9ff82c86c22e38c56dd2ce5d872850afb6ffeb502da7338baf0b55108afb7373"
    ),
}
COTRACKER_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_TREE = "f0296ad047b50c1530063b67e575908257478cab"
COTRACKER_PREDICTOR_SHA256 = (
    "783536d6c77790fa6c8e005f2df1d6bd4f0d8955c2c67d464bfb4e64d366375f"
)
COTRACKER_CHECKPOINT_SHA256 = (
    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {source}") from exc
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def load_dynamic_source_processing_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the source-processing protocol."""

    protocol = _load_json(path)
    _require(protocol.get("schema_version") == 1, "processing protocol schema changed")
    _require(
        protocol.get("artifact_kind") == PROCESSING_PROTOCOL_KIND,
        "wrong processing protocol kind",
    )
    _require(
        protocol.get("protocol_id") == PROCESSING_PROTOCOL_ID,
        "processing protocol ID changed",
    )
    _require(
        protocol.get("status") == "locked_before_fresh_source_geometry_reconstruction",
        "processing protocol is not locked",
    )
    _require(
        protocol.get("config_sha256")
        == canonical_sha256(protocol, digest_key="config_sha256"),
        "processing protocol checksum changed",
    )
    parent = protocol.get("parent_mask_protocol")
    _require(
        isinstance(parent, Mapping)
        and parent.get("protocol_id") == MASK_PROTOCOL_ID
        and isinstance(parent.get("config_sha256"), str)
        and len(parent["config_sha256"]) == 64
        and isinstance(parent.get("file_sha256"), str)
        and len(parent["file_sha256"]) == 64
        and isinstance(parent.get("implementation_commit"), str)
        and len(parent["implementation_commit"]) == 40,
        "parent mask protocol changed",
    )
    deform360 = protocol.get("deform360")
    _require(
        deform360
        == {
            "repository": "https://github.com/lhy0807/deform360",
            "revision": DEFORM360_REVISION,
            "source_sha256": DEFORM360_SOURCE_SHA256,
        },
        "Deform360 processing dependency changed",
    )
    camera = protocol.get("camera_policy")
    _require(
        camera
        == {
            "selection": "all successful frozen-panel cameras in lexical order",
            "minimum_camera_count": 8,
            "outcome_dependent_selection": False,
            "repair_or_replacement_after_failure": False,
        },
        "source-processing camera policy changed",
    )
    reconstruction = protocol.get("reconstruction")
    _require(
        reconstruction
        == {
            "raw_frame_count": RAW_FRAME_COUNT,
            "minimum_visual_hull_points": 512,
            "voxel_resolution": 120,
            "cube_half_extent_m": 0.5,
            "first_frame_iterations": 500,
            "warm_start_iterations": 250,
            "warm_start_from_previous_frame": True,
        },
        "source reconstruction contract changed",
    )
    _require(
        protocol.get("depth")
        == {
            "expected_depth": True,
            "object_mask_applied": True,
            "gripper_urdf_mask_applied": True,
            "preview_video": False,
        },
        "source depth contract changed",
    )
    tracking = protocol.get("tracking")
    _require(
        tracking
        == {
            "model": "facebook/cotracker3-scaled-offline",
            "repository_revision": COTRACKER_REVISION,
            "repository_tree": COTRACKER_TREE,
            "predictor_source_sha256": COTRACKER_PREDICTOR_SHA256,
            "checkpoint_sha256": COTRACKER_CHECKPOINT_SHA256,
        },
        "CoTracker dependency changed",
    )
    point_cloud = protocol.get("point_cloud")
    _require(
        point_cloud
        == {
            "seed_point_count": 10000,
            "crop_half_extent_m": 0.5,
            "radius_neighbors": 30,
            "radius_m": 0.02,
            "statistical_neighbors": 30,
            "statistical_std_ratio": 3.5,
            "fusion_ransac_threshold_m_per_s": 0.01,
            "fusion_minimum_inliers": 4,
            "rng_seed": 0,
            "output_frame_count": PREDICTION_FRAME_COUNT,
            "tracking_tail_frames_skipped": 5,
            "frame_rate_hz": 30.0,
        },
        "point-cloud contract changed",
    )
    _require(
        protocol.get("source_admission")
        == {
            "minimum_camera_count": 8,
            "minimum_point_count": 128,
            "maximum_point_count": 10000,
            "required_frame_count": PREDICTION_FRAME_COUNT,
            "update_frames": [19, 38, 57],
            "minimum_test_frame_count": 8,
            "future_geometry_deserialized_for_admission": False,
        },
        "source-admission contract changed",
    )
    failure = protocol.get("failure_accounting")
    _require(
        failure
        == {
            "mask_technical_failure_is_terminal_for_queue_entry": True,
            "processing_failure_is_preserved_as_technical_failure": True,
            "source_rejection_is_not_a_model_prediction": True,
            "implicit_replacement": False,
            "minimum_final_admissions": 20,
            "fewer_than_minimum_requires_a_new_locked_reserve_queue": True,
        },
        "source-processing failure accounting changed",
    )
    boundary = protocol.get("information_boundary")
    _require(
        boundary
        == {
            "all_81_source_rgb_and_masks_used_for_source_processing": True,
            "known_action_used_as_a_conditioning_input": True,
            "tactile_read": False,
            "target_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
            "sealed_window_and_mask_artifacts_mutated": False,
            "derived_processing_workspace_only": True,
        },
        "source-processing information boundary changed",
    )
    return protocol


def load_dynamic_source_processing_runtime_amendment(
    path: str | Path,
    *,
    parent_protocol_path: str | Path,
) -> dict[str, Any]:
    """Load the runtime-only amendment for untouched source entries."""

    amendment = _load_json(path)
    _require(
        amendment.get("schema_version") == 1,
        "runtime amendment schema changed",
    )
    _require(
        amendment.get("artifact_kind") == RUNTIME_AMENDMENT_KIND
        and amendment.get("amendment_id") == RUNTIME_AMENDMENT_ID,
        "wrong runtime amendment identity",
    )
    _require(
        amendment.get("status")
        == (
            "locked_after_eight_environment_failures_before_any_"
            "successful_reconstruction"
        ),
        "runtime amendment is not locked",
    )
    _require(
        amendment.get("config_sha256")
        == canonical_sha256(amendment, digest_key="config_sha256"),
        "runtime amendment checksum changed",
    )
    parent_protocol = load_dynamic_source_processing_protocol(
        parent_protocol_path
    )
    _require(
        amendment.get("parent_processing_protocol")
        == {
            "protocol_id": PROCESSING_PROTOCOL_ID,
            "config_sha256": parent_protocol["config_sha256"],
            "file_sha256": file_sha256(parent_protocol_path),
            "implementation_commit": (
                "474b58fd0f12af640cbbe8ed6cfba394c299ad2a"
            ),
        },
        "runtime amendment binds another processing protocol",
    )
    trigger = amendment.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("attempted_queue_entries") == 8
        and trigger.get("technical_failure_count") == 8
        and trigger.get("successful_reconstruction_count") == 0
        and trigger.get("successful_tracker_count") == 0
        and trigger.get("source_admission_count") == 0
        and trigger.get("error_type") == "AttributeError"
        and trigger.get("error_message")
        == "'NoneType' object has no attribute 'CameraModelType'"
        and tuple(trigger.get("failure_result_sha256", ()))
        == TERMINAL_FAILURE_RESULT_SHA256
        and tuple(trigger.get("failure_file_sha256", ()))
        == TERMINAL_FAILURE_FILE_SHA256
        and trigger.get("derived_splat_artifact_count") == 0
        and trigger.get("derived_point_cloud_artifact_count") == 0
        and trigger.get("target_metric_read") is False
        and trigger.get("provider_outcome_read") is False,
        "runtime amendment trigger changed",
    )
    runtime = amendment.get("runtime_contract")
    _require(
        runtime
        == {
            "required_path_prefix": "/usr/local/cuda/bin",
            "nvcc_path": "/usr/local/cuda/bin/nvcc",
            "nvcc_version_line": (
                "Cuda compilation tools, release 12.6, V12.6.85"
            ),
            "gsplat_version": "1.4.0",
            "torch_version": "2.4.0+cu121",
            "gsplat_extension_path": (
                "/home/florianpfaff/.cache/torch_extensions/"
                "py310_cu121/gsplat_cuda/gsplat_cuda.so"
            ),
            "gsplat_extension_sha256": (
                "58c95816cdf011dbbd13a71f1d98312c9e661ef34c95592cc00ff93c72cab89b"
            ),
            "required_backend_probe": "CameraModelType.PINHOLE",
        },
        "runtime contract changed",
    )
    _require(
        amendment.get("application_policy")
        == {
            "applies_only_to_unattempted_queue_entries": True,
            "unattempted_queue_entry_count": 26,
            "failed_entries_remain_terminal": True,
            "retry_failed_entries": False,
            "replace_failed_entries": False,
            "camera_method_reconstruction_tracking_and_admission_unchanged": True,
            "minimum_final_admissions_unchanged": 20,
        },
        "runtime amendment application policy changed",
    )
    _require(
        amendment.get("information_boundary")
        == {
            "source_rgb_depth_mask_or_geometry_read_by_amendment": False,
            "target_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
        },
        "runtime amendment crossed the information boundary",
    )
    return amendment


def validate_dynamic_source_mask_artifact(
    path: str | Path,
    *,
    mask_protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    mask_episode_dir: str | Path,
    expected_code_revision: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Validate one completed mask artifact and return its frozen camera panel."""

    manifest = _load_json(path)
    _require(
        manifest.get("artifact_kind") == MASK_ARTIFACT_KIND
        and manifest.get("protocol_id") == MASK_PROTOCOL_ID
        and manifest.get("protocol_config_sha256") == mask_protocol["config_sha256"]
        and manifest.get("code_revision") == expected_code_revision
        and manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256")
        and all(manifest.get(key) == value for key, value in case.items()),
        "fresh source mask artifact changed",
    )
    _require(
        manifest.get("status") == "ready_for_source_processing",
        "fresh source mask artifact is not processing-ready",
    )
    records = manifest.get("camera_records")
    _require(
        isinstance(records, list)
        and len(records) == len(FROZEN_CAMERA_PANEL)
        and tuple(row.get("camera") for row in records) == FROZEN_CAMERA_PANEL,
        "fresh source mask camera records changed",
    )
    successful = tuple(
        sorted(str(row["camera"]) for row in records if row.get("status") == "success")
    )
    _require(
        len(successful) == manifest.get("successful_camera_count")
        and len(successful) >= 8,
        "fresh source successful-camera count changed",
    )
    root = Path(mask_episode_dir)
    by_camera = {str(row["camera"]): row for row in records}
    for camera in successful:
        row = by_camera[camera]
        mask_path = root / camera / "mask_refined.h5"
        _require(
            mask_path.is_file()
            and file_sha256(mask_path) == row.get("mask_sha256")
            and row.get("frame_count") == RAW_FRAME_COUNT,
            f"frozen source mask changed: {camera}",
        )
    return manifest, successful


__all__ = [
    "COTRACKER_CHECKPOINT_SHA256",
    "COTRACKER_PREDICTOR_SHA256",
    "COTRACKER_REVISION",
    "COTRACKER_TREE",
    "DEFORM360_REVISION",
    "DEFORM360_SOURCE_SHA256",
    "PROCESSING_ARTIFACT_KIND",
    "PROCESSING_PROTOCOL_ID",
    "load_dynamic_source_processing_protocol",
    "load_dynamic_source_processing_runtime_amendment",
    "validate_dynamic_source_mask_artifact",
]
