"""Frozen source assets for the dynamic TAPNext++ Deform360 study."""

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


PROTOCOL_KIND = "Deform360DynamicTapNextppSourceWindowProtocol"
PROTOCOL_ID = "deform360-dynamic-tapnextpp-source-window-v1"
SELECTION_KIND = "Deform360DynamicTapNextppSourceWindowSelection"
PREPARATION_KIND = "Deform360DynamicTapNextppSourcePreparation"
STAGE_KIND = "Deform360DynamicTapNextppSourceWindowStage"
MASK_PROTOCOL_KIND = "Deform360DynamicTapNextppSourceMaskProtocol"
MASK_PROTOCOL_ID = "deform360-dynamic-tapnextpp-source-masks-v1"
MASK_ARTIFACT_KIND = "Deform360DynamicTapNextppSourceMasks"
FROZEN_PROVIDER_COMMIT = "31e55bf18f26363969411ebf2741be8fc3a8e7d8"
FROZEN_PROVIDER_PROTOCOL_ID = "deform360-dynamic-tapnextpp-provider-v1"
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
FROZEN_QUEUE_SHA256 = "8afcfe64fe62af36a303e376a8c2f1fb78fc855446eadcd6687c13e12a650bdc"
FROZEN_QUEUE_FILE_SHA256 = (
    "8dbe094990b9e4bd23c614f50ca6a8b62eef4178e6895d30275f52328d3a1203"
)
FROZEN_DOWNLOAD_SHA256 = (
    "0160df4204c484ae5906ff1258ec623b7745e574cc0330af74e4ecd3db224d15"
)
FROZEN_DOWNLOAD_FILE_SHA256 = (
    "0567f05c6e68d532de699c0434ea9ffba5851dd01eb120fe5f4cd60431f6d7fe"
)
FROZEN_DOWNLOAD_OBJECT_COUNT = 36
FROZEN_DOWNLOAD_FILE_COUNT = 2820
FROZEN_DOWNLOAD_TOTAL_BYTES = 3_192_349_000
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


def canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
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


def load_dynamic_source_window_protocol(path: str | Path) -> dict[str, Any]:
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
    method = protocol.get("frozen_provider")
    _require(
        isinstance(method, Mapping)
        and method.get("commit") == FROZEN_PROVIDER_COMMIT
        and method.get("protocol_id") == FROZEN_PROVIDER_PROTOCOL_ID,
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
        and queue.get("candidate_count") == FROZEN_DOWNLOAD_OBJECT_COUNT,
        "frozen source queue binding changed",
    )
    _require(
        isinstance(download, Mapping)
        and download.get("manifest_sha256") == FROZEN_DOWNLOAD_SHA256
        and download.get("file_sha256") == FROZEN_DOWNLOAD_FILE_SHA256
        and download.get("object_count") == FROZEN_DOWNLOAD_OBJECT_COUNT
        and download.get("file_count") == FROZEN_DOWNLOAD_FILE_COUNT
        and download.get("total_bytes") == FROZEN_DOWNLOAD_TOTAL_BYTES,
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
        and selection.get("score_step_range_half_open") == list(SCORE_STEP_RANGE)
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


def load_dynamic_source_mask_protocol(path: str | Path) -> dict[str, Any]:
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
        and isinstance(parent.get("config_sha256"), str)
        and len(parent["config_sha256"]) == 64
        and isinstance(parent.get("file_sha256"), str)
        and len(parent["file_sha256"]) == 64
        and isinstance(parent.get("implementation_commit"), str)
        and len(parent["implementation_commit"]) == 40,
        "parent window protocol changed",
    )
    selector = protocol.get("generic_selector")
    _require(
        isinstance(selector, Mapping)
        and selector.get("object_source_sha256") == FROZEN_OBJECT_SAM2_SOURCE_SHA256
        and selector.get("base_source_sha256") == FROZEN_BASE_SAM2_SOURCE_SHA256
        and selector.get("manual_prompting") is False,
        "generic SAM2 selector changed",
    )
    sam2 = protocol.get("sam2")
    _require(
        isinstance(sam2, Mapping)
        and sam2.get("commit") == FROZEN_SAM2_COMMIT
        and sam2.get("checkpoint_sha256") == FROZEN_SAM2_CHECKPOINT_SHA256,
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


def validate_dynamic_window_sources(
    protocol_path: str | Path,
    queue_path: str | Path,
    download_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the protocol against the exact queue and downloaded inventory."""

    protocol = load_dynamic_source_window_protocol(protocol_path)
    queue = _load_json(queue_path)
    download = _load_json(download_manifest_path)
    _require(queue.get("artifact_kind") == QUEUE_KIND, "wrong source queue kind")
    _require(
        queue.get("queue_sha256") == FROZEN_QUEUE_SHA256
        and file_sha256(queue_path) == FROZEN_QUEUE_FILE_SHA256,
        "source queue bytes changed",
    )
    _require(
        queue.get("queue_sha256") == canonical_sha256(queue, digest_key="queue_sha256"),
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
    _require(
        isinstance(rows, list) and len(rows) == FROZEN_DOWNLOAD_OBJECT_COUNT,
        "download object panel changed",
    )
    _require(
        sum(int(row["file_count"]) for row in rows) == FROZEN_DOWNLOAD_FILE_COUNT
        and sum(int(row["total_bytes"]) for row in rows) == FROZEN_DOWNLOAD_TOTAL_BYTES,
        "download inventory changed",
    )
    return protocol, queue, download


def dynamic_source_case(
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


def validate_dynamic_source_preparation(
    manifest: Mapping[str, Any],
    *,
    window_protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    expected_code_revision: str,
) -> None:
    """Validate alignment provenance despite the legacy bimanual key collision."""

    identity = {key: value for key, value in case.items() if key != "bimanual"}
    _require(
        manifest.get("artifact_kind") == PREPARATION_KIND
        and manifest.get("protocol_config_sha256") == window_protocol["config_sha256"]
        and manifest.get("code_revision") == expected_code_revision
        and manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256")
        and all(manifest.get(key) == value for key, value in identity.items()),
        "dynamic TAPNext++ source preparation is incompatible",
    )
    queued_bimanual = case.get("bimanual")
    if queued_bimanual is not None:
        _require(
            queued_bimanual in {"yes", "no"}
            and manifest.get("bimanual") == (queued_bimanual == "yes"),
            "dynamic TAPNext++ bimanual preparation differs from the queue",
        )


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
        "selected_unweighted_future_path_m": selected["unweighted_future_path_m"],
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


def seal_dynamic_source_window_selection(
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
        "frozen_provider": dict(protocol["frozen_provider"]),
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
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    return payload


def validate_dynamic_source_window_stage(
    path: str | Path,
    *,
    window_protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    expected_code_revision: str,
) -> dict[str, Any]:
    """Validate a staged source window before derived observations are created."""

    stage = _load_json(path)
    _require(
        stage.get("artifact_kind") == STAGE_KIND
        and stage.get("protocol_id") == PROTOCOL_ID
        and stage.get("protocol_config_sha256") == window_protocol["config_sha256"]
        and stage.get("code_revision") == expected_code_revision
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
        and tuple(row.get("camera") for row in camera_rows) == FROZEN_CAMERA_PANEL
        and all(
            row.get("decoded_frame_count") == RAW_FRAME_COUNT for row in camera_rows
        ),
        "fresh source camera records changed",
    )
    boundary = stage.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("known_future_action_read") is True
        and boundary.get("object_rgb_materialized_after_selection_seal_built") is True
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
    "FROZEN_DOWNLOAD_FILE_COUNT",
    "FROZEN_DOWNLOAD_OBJECT_COUNT",
    "FROZEN_DOWNLOAD_TOTAL_BYTES",
    "MASK_ARTIFACT_KIND",
    "MASK_PROTOCOL_ID",
    "PREDICTION_FRAME_COUNT",
    "PREPARATION_KIND",
    "PROTOCOL_ID",
    "RAW_FRAME_COUNT",
    "SCORE_STEP_RANGE",
    "SELECTION_KIND",
    "STAGE_KIND",
    "canonical_sha256",
    "file_sha256",
    "dynamic_source_case",
    "load_dynamic_source_mask_protocol",
    "load_dynamic_source_window_protocol",
    "seal_dynamic_source_window_selection",
    "select_fresh_source_window",
    "validate_dynamic_source_window_stage",
    "validate_dynamic_source_preparation",
    "validate_dynamic_window_sources",
]
