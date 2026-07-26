"""Frozen action-window selection for the fresh Deform360 source queue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_fresh_source_download import (
    DOWNLOAD_KIND,
    QUEUE_KIND,
)
from .deform360_selective_virtual_sensing_staging import (
    closure_confidence,
    end_effector_origins,
)


PROTOCOL_KIND = "Deform360FreshSourceWindowProtocol"
PROTOCOL_ID = "deform360-fresh-object-pairwise-belief-window-v1"
SELECTION_KIND = "Deform360FreshSourceWindowSelection"
MASK_PROTOCOL_KIND = "Deform360FreshSourceMaskProtocol"
MASK_PROTOCOL_ID = "deform360-fresh-object-pairwise-belief-masks-v1"
MASK_ARTIFACT_KIND = "Deform360FreshSourceMasks"
FROZEN_METHOD_COMMIT = "e2f8d827bfd60df79eeffee511a5df7e2d53ea21"
FROZEN_METHOD_ARM = "raw_selected_backbone_full_blend_rbf_pairwise_clique"
FROZEN_WINDOW_IMPLEMENTATION_COMMIT = (
    "9e0622efbf9244ce9aa6d75a87505c99ec457f6f"
)
FROZEN_WINDOW_PROTOCOL_FILE_SHA256 = (
    "a92595fedb85bc17723743914db38bb085ff7179126d47f83b2ac328d9db0ba1"
)
FROZEN_OBJECT_SAM2_SOURCE_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
FROZEN_BASE_SAM2_SOURCE_SHA256 = (
    "419be2e98ab2b01627ea188c8658b43b39d8b3d4e34e8b33559f32ccdcd04184"
)
FROZEN_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
FROZEN_SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
FROZEN_QUEUE_SHA256 = (
    "f80fed80ca2b9f1857539834bd92c6acb1b45a88eefbcae16e35cddaf9185d0e"
)
FROZEN_QUEUE_FILE_SHA256 = (
    "4f31de5f2a4300916bff0b78c29796adffb60b84aad94b0d4895c1615831c372"
)
FROZEN_DOWNLOAD_SHA256 = (
    "a7774030848e2df5d4f33de37d8b6292b79665914053d690eff37b0f56f958ff"
)
FROZEN_DOWNLOAD_FILE_SHA256 = (
    "04651369f533398bac976aa50ef96a6b513c4d4fc7db80545e30b2ecddbb3bde"
)
FROZEN_CAMERA_PANEL = (
    "brics-odroid-001_cam0",
    "brics-odroid-006_cam0",
    "brics-odroid-007_cam0",
    "brics-odroid-008_cam0",
    "brics-odroid-010_cam0",
    "brics-odroid-013_cam0",
    "brics-odroid-014_cam1",
    "brics-odroid-015_cam1",
    "brics-odroid-019_cam1",
    "brics-odroid-021_cam1",
    "brics-odroid-024_cam1",
    "brics-odroid-027_cam0",
)
RAW_FRAME_COUNT = 81
PREDICTION_FRAME_COUNT = 76
FIRST_UPDATE_FRAME = 19
CANDIDATE_FIRST_FRAME = 8
CANDIDATE_STRIDE_FRAMES = 6
SCORE_STEP_RANGE = (FIRST_UPDATE_FRAME, PREDICTION_FRAME_COUNT - 1)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Return the streaming SHA-256 of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(
    payload: Mapping[str, Any], *, digest_key: str
) -> str:
    """Hash a JSON artifact after removing its self-digest."""

    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {source}") from exc
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def load_fresh_source_window_protocol(path: str | Path) -> dict[str, Any]:
    """Load the immutable source-window protocol and reject recomputed changes."""

    protocol = _load_json(path)
    _require(protocol.get("schema_version") == 1, "window protocol schema changed")
    _require(protocol.get("artifact_kind") == PROTOCOL_KIND, "wrong protocol kind")
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "window protocol ID changed")
    _require(
        protocol.get("status") == "locked_before_source_rgb_decode",
        "window protocol is not locked",
    )
    _require(
        protocol.get("config_sha256")
        == canonical_sha256(protocol, digest_key="config_sha256"),
        "window protocol checksum changed",
    )
    method = protocol.get("frozen_method")
    _require(
        isinstance(method, Mapping)
        and method.get("commit") == FROZEN_METHOD_COMMIT
        and method.get("arm") == FROZEN_METHOD_ARM,
        "frozen prediction method changed",
    )
    bindings = protocol.get("source_bindings")
    _require(isinstance(bindings, Mapping), "source bindings are missing")
    queue = bindings.get("queue")
    download = bindings.get("download")
    _require(
        isinstance(queue, Mapping)
        and queue.get("queue_sha256") == FROZEN_QUEUE_SHA256
        and queue.get("file_sha256") == FROZEN_QUEUE_FILE_SHA256
        and queue.get("candidate_count") == 18,
        "frozen source queue binding changed",
    )
    _require(
        isinstance(download, Mapping)
        and download.get("manifest_sha256") == FROZEN_DOWNLOAD_SHA256
        and download.get("file_sha256") == FROZEN_DOWNLOAD_FILE_SHA256
        and download.get("object_count") == 18
        and download.get("file_count") == 1452
        and download.get("total_bytes") == 1_834_930_956,
        "frozen source download binding changed",
    )
    selection = protocol.get("window_selection")
    _require(
        isinstance(selection, Mapping)
        and selection.get("raw_window_frame_count") == RAW_FRAME_COUNT
        and selection.get("prediction_frame_count") == PREDICTION_FRAME_COUNT
        and selection.get("first_update_frame") == FIRST_UPDATE_FRAME
        and selection.get("candidate_first_frame") == CANDIDATE_FIRST_FRAME
        and selection.get("candidate_stride_frames") == CANDIDATE_STRIDE_FRAMES
        and selection.get("action_position_field") == "robot.actions[...,0,:]"
        and selection.get("score_step_range_half_open")
        == list(SCORE_STEP_RANGE)
        and selection.get("tie_break") == "earliest candidate start",
        "frozen window-selection rule changed",
    )
    _require(
        tuple(protocol.get("camera_panel", ())) == FROZEN_CAMERA_PANEL,
        "frozen camera panel changed",
    )
    cohort = protocol.get("cohort_boundary")
    _require(
        isinstance(cohort, Mapping)
        and cohort.get("window_selection_is_not_source_admission") is True
        and cohort.get("motion_or_response_threshold_used_for_cohort_membership")
        is False,
        "window selection crossed the source-admission boundary",
    )
    boundary = protocol.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    required_true = (
        "known_future_action_used_for_window_selection",
        "action_is_a_conditioning_input",
        "selection_sealed_before_object_processing_and_prediction",
    )
    required_false = (
        "object_geometry_used_for_window_selection",
        "object_tracks_used_for_window_selection",
        "object_response_used_for_window_selection",
        "tactile_used_for_window_selection",
        "target_metric_used_for_window_selection",
        "held_v8_target_query_score_barrier_or_outcome_access",
    )
    _require(
        all(boundary.get(key) is True for key in required_true)
        and all(boundary.get(key) is False for key in required_false),
        "window protocol crossed its information boundary",
    )
    return protocol


def load_fresh_source_mask_protocol(path: str | Path) -> dict[str, Any]:
    """Load the frozen generic-SAM2 source-mask protocol."""

    protocol = _load_json(path)
    _require(protocol.get("schema_version") == 1, "mask protocol schema changed")
    _require(
        protocol.get("artifact_kind") == MASK_PROTOCOL_KIND,
        "wrong mask protocol kind",
    )
    _require(
        protocol.get("protocol_id") == MASK_PROTOCOL_ID,
        "mask protocol ID changed",
    )
    _require(
        protocol.get("status") == "locked_before_source_mask_generation",
        "mask protocol is not locked",
    )
    _require(
        protocol.get("config_sha256")
        == canonical_sha256(protocol, digest_key="config_sha256"),
        "mask protocol checksum changed",
    )
    parent = protocol.get("parent_window_protocol")
    _require(
        isinstance(parent, Mapping)
        and parent.get("protocol_id") == PROTOCOL_ID
        and parent.get("config_sha256")
        == "015305926274bda59ae0b03390a86ac321e615b598001961fc70f13ee9f69511"
        and parent.get("file_sha256") == FROZEN_WINDOW_PROTOCOL_FILE_SHA256
        and parent.get("implementation_commit")
        == FROZEN_WINDOW_IMPLEMENTATION_COMMIT,
        "parent window protocol changed",
    )
    selector = protocol.get("generic_selector")
    _require(
        isinstance(selector, Mapping)
        and selector.get("object_source_sha256")
        == FROZEN_OBJECT_SAM2_SOURCE_SHA256
        and selector.get("base_source_sha256")
        == FROZEN_BASE_SAM2_SOURCE_SHA256
        and selector.get("manual_prompting") is False,
        "generic SAM2 selector changed",
    )
    sam2 = protocol.get("sam2")
    _require(
        isinstance(sam2, Mapping)
        and sam2.get("commit") == FROZEN_SAM2_COMMIT
        and sam2.get("checkpoint_sha256")
        == FROZEN_SAM2_CHECKPOINT_SHA256,
        "SAM2 dependency changed",
    )
    contract = protocol.get("mask_contract")
    _require(
        isinstance(contract, Mapping)
        and contract.get("input_camera_count") == len(FROZEN_CAMERA_PANEL)
        and contract.get("frame_count") == RAW_FRAME_COUNT
        and contract.get("minimum_successful_cameras") == 8
        and contract.get("implicit_replacement") is False,
        "source-mask contract changed",
    )
    boundary = protocol.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("source_rgb_read") is True
        and boundary.get("all_81_frames_used_to_create_observation_assets") is True
        and boundary.get("manual_mask_selection") is False
        and boundary.get("object_geometry_read") is False
        and boundary.get("particle_tracks_read") is False
        and boundary.get("target_metric_read") is False
        and boundary.get("held_v8_target_query_score_barrier_or_outcome_access")
        is False
        and boundary.get(
            "prediction_method_may_receive_only_its_separately_authorized_prefix"
        )
        is True,
        "source-mask protocol crossed its information boundary",
    )
    return protocol


def validate_window_sources(
    protocol_path: str | Path,
    queue_path: str | Path,
    download_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the protocol against the exact queue and downloaded inventory."""

    protocol = load_fresh_source_window_protocol(protocol_path)
    queue = _load_json(queue_path)
    download = _load_json(download_manifest_path)
    _require(queue.get("artifact_kind") == QUEUE_KIND, "wrong source queue kind")
    _require(
        queue.get("queue_sha256") == FROZEN_QUEUE_SHA256
        and file_sha256(queue_path) == FROZEN_QUEUE_FILE_SHA256,
        "source queue bytes changed",
    )
    _require(
        queue.get("queue_sha256")
        == canonical_sha256(queue, digest_key="queue_sha256"),
        "source queue checksum changed",
    )
    _require(download.get("artifact_kind") == DOWNLOAD_KIND, "wrong download kind")
    _require(
        download.get("manifest_sha256") == FROZEN_DOWNLOAD_SHA256
        and file_sha256(download_manifest_path) == FROZEN_DOWNLOAD_FILE_SHA256,
        "source download bytes changed",
    )
    _require(
        download.get("manifest_sha256")
        == canonical_sha256(download, digest_key="manifest_sha256"),
        "source download checksum changed",
    )
    _require(
        download.get("queue_sha256") == FROZEN_QUEUE_SHA256
        and download.get("download_scope") == "queued_episode_camera_source_only"
        and download.get("audio_included") is False
        and download.get("tactile_included") is False,
        "source download scope changed",
    )
    rows = download.get("objects")
    _require(isinstance(rows, list) and len(rows) == 18, "download object panel changed")
    _require(
        sum(int(row["file_count"]) for row in rows) == 1452
        and sum(int(row["total_bytes"]) for row in rows) == 1_834_930_956,
        "download inventory changed",
    )
    return protocol, queue, download


def fresh_source_case(
    queue: Mapping[str, Any], object_id: str, episode_id: int
) -> dict[str, Any]:
    """Return one unique source-queue record."""

    matches = [
        row
        for row in queue.get("candidates", ())
        if isinstance(row, Mapping)
        and row.get("object_id") == object_id
        and row.get("episode_id") == episode_id
    ]
    _require(len(matches) == 1, "case is outside the frozen source queue")
    return dict(matches[0])


def select_fresh_source_window(
    actions: np.ndarray,
    openings: np.ndarray,
) -> dict[str, Any]:
    """Select the frozen window from measured end-effector translations only."""

    origins = end_effector_origins(actions)
    closed = closure_confidence(openings)
    _require(
        closed.shape == origins.shape[:2],
        "gripper openings do not match action groups",
    )
    candidates = np.arange(
        CANDIDATE_FIRST_FRAME,
        len(origins) - RAW_FRAME_COUNT + 1,
        CANDIDATE_STRIDE_FRAMES,
        dtype=np.int64,
    )
    _require(len(candidates) > 0, "episode has no complete frozen window candidate")
    score_start, score_stop = SCORE_STEP_RANGE
    rows: list[dict[str, Any]] = []
    for start_value in candidates:
        start = int(start_value)
        selected_origins = origins[start : start + RAW_FRAME_COUNT]
        selected_closed = closed[start : start + RAW_FRAME_COUNT]
        step = np.linalg.norm(np.diff(selected_origins, axis=0), axis=-1)
        adjacent_closure = np.minimum(selected_closed[:-1], selected_closed[1:])
        weighted = step * adjacent_closure
        score = float(np.mean(np.sum(weighted[score_start:score_stop], axis=0)))
        rows.append(
            {
                "start_frame": start,
                "score_m": score,
                "unweighted_future_path_m": float(
                    np.mean(np.sum(step[score_start:score_stop], axis=0))
                ),
                "mean_future_closure_confidence": float(
                    np.mean(adjacent_closure[score_start:score_stop])
                ),
            }
        )
    selected_index = int(np.argmax([row["score_m"] for row in rows]))
    selected = rows[selected_index]
    start = int(selected["start_frame"])
    return {
        "selection_rule": (
            "maximum-mean-closure-weighted-end-effector-translation-path-"
            "after-first-update"
        ),
        "selected_raw_frame_range_half_open": [start, start + RAW_FRAME_COUNT],
        "prediction_raw_frame_range_half_open": [
            start,
            start + PREDICTION_FRAME_COUNT,
        ],
        "score_step_range_in_staged_window_half_open": list(SCORE_STEP_RANGE),
        "selected_score_m": selected["score_m"],
        "selected_unweighted_future_path_m": selected[
            "unweighted_future_path_m"
        ],
        "selected_mean_future_closure_confidence": selected[
            "mean_future_closure_confidence"
        ],
        "candidate_first_frame": CANDIDATE_FIRST_FRAME,
        "candidate_stride_frames": CANDIDATE_STRIDE_FRAMES,
        "candidate_count": len(rows),
        "tie_break": "earliest candidate start",
        "candidate_scores": rows,
        "input_fields": ["robot.actions[...,0,:]", "robot.openings"],
        "known_future_action_is_conditioning_input": True,
        "object_geometry_read": False,
        "object_tracks_read": False,
        "object_response_read": False,
        "tactile_read": False,
        "target_metric_read": False,
    }


def seal_fresh_source_window_selection(
    *,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_robot_sha256: str,
    source_preparation_sha256: str,
    code_revision: str,
) -> dict[str, Any]:
    """Build a checksummed per-case selection artifact."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SELECTION_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "object_id": case["object_id"],
        "episode_id": case["episode_id"],
        "queue_rank": case["queue_rank"],
        "category": case["category"],
        "frozen_method": dict(protocol["frozen_method"]),
        "window_selection": dict(selection),
        "camera_panel": list(FROZEN_CAMERA_PANEL),
        "source_robot_sha256": source_robot_sha256,
        "source_preparation_sha256": source_preparation_sha256,
        "code_revision": code_revision,
        "information_boundary": {
            "known_future_action_read": True,
            "object_geometry_read": False,
            "object_tracks_read": False,
            "object_response_read": False,
            "tactile_read": False,
            "target_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
            "selection_sealed_before_object_processing_and_prediction": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    return payload


def validate_fresh_source_window_stage(
    path: str | Path,
    *,
    window_protocol: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a staged source window before derived observations are created."""

    stage = _load_json(path)
    _require(
        stage.get("artifact_kind") == "Deform360FreshSourceWindowStage"
        and stage.get("protocol_id") == PROTOCOL_ID
        and stage.get("protocol_config_sha256")
        == window_protocol["config_sha256"]
        and stage.get("code_revision") == FROZEN_WINDOW_IMPLEMENTATION_COMMIT
        and stage.get("result_sha256")
        == canonical_sha256(stage, digest_key="result_sha256")
        and all(stage.get(key) == value for key, value in case.items()),
        "fresh source window stage changed",
    )
    _require(
        stage.get("staged_frame_count") == RAW_FRAME_COUNT
        and stage.get("camera_count") == len(FROZEN_CAMERA_PANEL),
        "fresh source window dimensions changed",
    )
    camera_rows = stage.get("camera_records")
    _require(
        isinstance(camera_rows, list)
        and tuple(row.get("camera") for row in camera_rows)
        == FROZEN_CAMERA_PANEL
        and all(row.get("decoded_frame_count") == RAW_FRAME_COUNT for row in camera_rows),
        "fresh source camera records changed",
    )
    boundary = stage.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("known_future_action_read") is True
        and boundary.get("object_rgb_materialized_after_selection_seal_built")
        is True
        and boundary.get("object_geometry_read") is False
        and boundary.get("object_tracks_read") is False
        and boundary.get("object_response_used_for_window_selection") is False
        and boundary.get("tactile_read") is False
        and boundary.get("target_metric_read") is False,
        "fresh source stage crossed its information boundary",
    )
    return stage


__all__ = [
    "CANDIDATE_FIRST_FRAME",
    "CANDIDATE_STRIDE_FRAMES",
    "FIRST_UPDATE_FRAME",
    "FROZEN_BASE_SAM2_SOURCE_SHA256",
    "FROZEN_CAMERA_PANEL",
    "FROZEN_OBJECT_SAM2_SOURCE_SHA256",
    "FROZEN_SAM2_CHECKPOINT_SHA256",
    "FROZEN_SAM2_COMMIT",
    "FROZEN_WINDOW_IMPLEMENTATION_COMMIT",
    "MASK_ARTIFACT_KIND",
    "MASK_PROTOCOL_ID",
    "PREDICTION_FRAME_COUNT",
    "PROTOCOL_ID",
    "RAW_FRAME_COUNT",
    "SCORE_STEP_RANGE",
    "SELECTION_KIND",
    "canonical_sha256",
    "file_sha256",
    "fresh_source_case",
    "load_fresh_source_mask_protocol",
    "load_fresh_source_window_protocol",
    "seal_fresh_source_window_selection",
    "select_fresh_source_window",
    "validate_fresh_source_window_stage",
    "validate_window_sources",
]
