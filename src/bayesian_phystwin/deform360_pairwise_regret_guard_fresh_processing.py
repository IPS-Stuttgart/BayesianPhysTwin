"""Frozen preprocessing contract for the fresh pairwise-regret-guard study.

The public catalog leaves one untouched Deform360 object.  This module binds
its nine metadata-valid episodes to the same target-free action-window,
generic-mask, official reconstruction, and source-admission rules used by the
earlier fresh-source study.  It deliberately contains no target reader or
metric implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_pairwise_regret_guard_fresh_protocol import (
    FROZEN_CAMERA_PANEL,
    file_sha256,
    validate_fresh_download_manifest,
    validate_fresh_source_plan,
    validate_fresh_technical_lock,
)
from .deform360_selective_virtual_sensing_staging import (
    closure_confidence,
    end_effector_origins,
)

PROTOCOL_KIND = "Deform360PairwiseRegretGuardFreshProcessingProtocol"
PROTOCOL_ID = "deform360-pairwise-regret-guard-fresh-processing-v1"
PREPARATION_KIND = "Deform360PairwiseRegretGuardFreshPreparation"
WINDOW_SELECTION_KIND = "Deform360PairwiseRegretGuardFreshWindowSelection"
WINDOW_STAGE_KIND = "Deform360PairwiseRegretGuardFreshWindowStage"
MASK_KIND = "Deform360PairwiseRegretGuardFreshMasks"
PROCESSING_KIND = "Deform360PairwiseRegretGuardFreshProcessing"
ADMISSION_KIND = "Deform360PairwiseRegretGuardFreshAdmission"

RAW_FRAME_COUNT = 81
PREDICTION_FRAME_COUNT = 76
FIRST_UPDATE_FRAME = 19
UPDATE_FRAMES = (19, 38, 57)
CANDIDATE_FIRST_FRAME = 8
CANDIDATE_STRIDE_FRAMES = 6
SCORE_STEP_RANGE = (FIRST_UPDATE_FRAME, PREDICTION_FRAME_COUNT - 1)

DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
DEFORM360_SOURCE_SHA256 = {
    "undistort": ("06a500ab2ced8cc960d649d9e200d6d479804ef542ba5aac8fedc5733e74aba9"),
    "robot_stage": ("5944301cc781f179bea96470af50273836a13fdbb367af9a89a59ce1911c11e0"),
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
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
SAM2_OBJECT_SOURCE_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
SAM2_BASE_SOURCE_SHA256 = (
    "419be2e98ab2b01627ea188c8658b43b39d8b3d4e34e8b33559f32ccdcd04184"
)
COTRACKER_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_TREE = "f0296ad047b50c1530063b67e575908257478cab"
COTRACKER_PREDICTOR_SHA256 = (
    "783536d6c77790fa6c8e005f2df1d6bd4f0d8955c2c67d464bfb4e64d366375f"
)
COTRACKER_CHECKPOINT_SHA256 = (
    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(payload: Mapping[str, Any], *, digest_key: str) -> dict[str, Any]:
    result = json.loads(json.dumps(payload, allow_nan=False))
    result[digest_key] = canonical_sha256(result, digest_key=digest_key)
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {source}") from exc
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def _validate_download_against_plan(
    plan: Mapping[str, Any], download: Mapping[str, Any]
) -> None:
    _require(
        download.get("source_plan_sha256") == plan.get("source_plan_sha256")
        and download.get("repository") == plan.get("repository")
        and download.get("revision") == plan.get("revision")
        and download.get("object_id") == plan.get("object_id"),
        "download identity differs from the source plan",
    )
    expected_rows = plan.get("download", {}).get("files")
    actual_rows = download.get("files")
    _require(
        isinstance(expected_rows, list)
        and isinstance(actual_rows, list)
        and len(expected_rows) == len(actual_rows)
        and download.get("file_count") == plan.get("download", {}).get("file_count")
        and download.get("total_bytes") == plan.get("download", {}).get("total_bytes"),
        "download inventory dimensions differ from the source plan",
    )
    expected = {str(row["path"]): row for row in expected_rows}
    actual = {str(row["path"]): row for row in actual_rows}
    _require(set(actual) == set(expected), "download paths differ from the source plan")
    for path, expected_row in expected.items():
        actual_row = actual[path]
        _require(
            actual_row.get("size") == expected_row.get("size"),
            f"download size differs from the source plan: {path}",
        )
        lfs_sha256 = expected_row.get("lfs_sha256")
        if lfs_sha256 is not None:
            _require(
                actual_row.get("sha256") == lfs_sha256,
                f"download digest differs from the source plan: {path}",
            )


def build_fresh_processing_protocol(
    technical_lock_path: str | Path,
    source_plan_path: str | Path,
    download_manifest_path: str | Path,
    *,
    implementation_commit: str,
) -> dict[str, Any]:
    """Bind source bytes and preprocessing dependencies before RGB decoding."""

    _require(_HEX40.fullmatch(implementation_commit) is not None, "bad commit")
    lock = _load_json(technical_lock_path)
    plan = _load_json(source_plan_path)
    download = _load_json(download_manifest_path)
    validate_fresh_technical_lock(lock)
    validate_fresh_source_plan(plan)
    validate_fresh_download_manifest(download)
    _require(
        plan["technical_lock_sha256"] == lock["lock_sha256"],
        "source plan binds another technical lock",
    )
    _require(
        download["source_plan_sha256"] == plan["source_plan_sha256"],
        "download binds another source plan",
    )
    _validate_download_against_plan(plan, download)
    artifact = {
        "schema_version": 1,
        "artifact_kind": PROTOCOL_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "locked_before_source_rgb_decode_or_processed_geometry",
        "implementation_commit": implementation_commit,
        "bindings": {
            "technical_lock_sha256": lock["lock_sha256"],
            "technical_lock_file_sha256": file_sha256(technical_lock_path),
            "source_plan_sha256": plan["source_plan_sha256"],
            "source_plan_file_sha256": file_sha256(source_plan_path),
            "download_sha256": download["download_sha256"],
            "download_file_sha256": file_sha256(download_manifest_path),
            "source_tree_sha256": download["source_tree_sha256"],
            "file_count": download["file_count"],
            "total_bytes": download["total_bytes"],
        },
        "dataset": {
            "object_id": lock["selected_physical_object"]["object_id"],
            "episode_ids": [
                row["episode_id"]
                for row in lock["selected_physical_object"]["valid_episodes"]
            ],
            "camera_panel": list(FROZEN_CAMERA_PANEL),
        },
        "window": {
            "raw_frame_count": RAW_FRAME_COUNT,
            "prediction_frame_count": PREDICTION_FRAME_COUNT,
            "first_update_frame": FIRST_UPDATE_FRAME,
            "update_frames": list(UPDATE_FRAMES),
            "candidate_first_frame": CANDIDATE_FIRST_FRAME,
            "candidate_stride_frames": CANDIDATE_STRIDE_FRAMES,
            "score_step_range_half_open": list(SCORE_STEP_RANGE),
            "action_position_field": "robot.actions[...,0,:]",
            "tie_break": "earliest candidate start",
        },
        "mask": {
            "input_camera_count": len(FROZEN_CAMERA_PANEL),
            "minimum_successful_cameras": 8,
            "frame_count": RAW_FRAME_COUNT,
            "manual_prompting": False,
            "sam2_commit": SAM2_COMMIT,
            "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
            "object_selector_source_sha256": SAM2_OBJECT_SOURCE_SHA256,
            "base_selector_source_sha256": SAM2_BASE_SOURCE_SHA256,
        },
        "processing": {
            "deform360_revision": DEFORM360_REVISION,
            "deform360_source_sha256": DEFORM360_SOURCE_SHA256,
            "minimum_processing_cameras": 8,
            "minimum_visual_hull_points": 512,
            "voxel_resolution": 120,
            "cube_half_extent_m": 0.5,
            "first_frame_iterations": 500,
            "warm_start_iterations": 250,
            "cotracker_revision": COTRACKER_REVISION,
            "cotracker_tree": COTRACKER_TREE,
            "cotracker_predictor_sha256": COTRACKER_PREDICTOR_SHA256,
            "cotracker_checkpoint_sha256": COTRACKER_CHECKPOINT_SHA256,
        },
        "admission": {
            "minimum_camera_count": 3,
            "minimum_point_count": 128,
            "maximum_point_count": 10000,
            "required_frame_count": PREDICTION_FRAME_COUNT,
            "update_frames": list(UPDATE_FRAMES),
            "minimum_test_frame_count": 8,
            "future_geometry_deserialized_for_admission": False,
        },
        "failure_accounting": {
            "all_nine_valid_episodes_are_attempted": True,
            "technical_failures_are_retained": True,
            "source_rejections_are_not_predictions": True,
            "implicit_replacement": False,
            "malformed_episode_six_remains_unsealable": True,
        },
        "information_boundary": {
            "known_future_action_used_only_as_conditioning_input": True,
            "window_sealed_before_object_processing": True,
            "all_81_frames_may_create_observation_assets": True,
            "future_object_positions_deserialized_for_admission": False,
            "target_metric_read": False,
            "outcomes_may_not_change_processing_or_admission": True,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    return _seal(artifact, digest_key="protocol_sha256")


def validate_fresh_processing_protocol(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "wrong processing schema")
    _require(payload.get("artifact_kind") == PROTOCOL_KIND, "wrong processing kind")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "processing ID changed")
    _require(
        payload.get("status")
        == "locked_before_source_rgb_decode_or_processed_geometry",
        "processing protocol is not locked",
    )
    _require(
        payload.get("protocol_sha256")
        == canonical_sha256(payload, digest_key="protocol_sha256"),
        "processing checksum changed",
    )
    _require(
        _HEX40.fullmatch(str(payload.get("implementation_commit"))) is not None,
        "processing implementation commit is malformed",
    )
    dataset = payload.get("dataset", {})
    _require(
        dataset.get("object_id") == "197-hand-sanitizer"
        and tuple(dataset.get("camera_panel", ())) == FROZEN_CAMERA_PANEL
        and dataset.get("episode_ids") == [0, 1, 2, 3, 4, 5, 7, 8, 9],
        "processing dataset panel changed",
    )
    window = payload.get("window", {})
    _require(
        window
        == {
            "raw_frame_count": RAW_FRAME_COUNT,
            "prediction_frame_count": PREDICTION_FRAME_COUNT,
            "first_update_frame": FIRST_UPDATE_FRAME,
            "update_frames": list(UPDATE_FRAMES),
            "candidate_first_frame": CANDIDATE_FIRST_FRAME,
            "candidate_stride_frames": CANDIDATE_STRIDE_FRAMES,
            "score_step_range_half_open": list(SCORE_STEP_RANGE),
            "action_position_field": "robot.actions[...,0,:]",
            "tie_break": "earliest candidate start",
        },
        "processing window contract changed",
    )
    _require(
        payload.get("mask")
        == {
            "input_camera_count": len(FROZEN_CAMERA_PANEL),
            "minimum_successful_cameras": 8,
            "frame_count": RAW_FRAME_COUNT,
            "manual_prompting": False,
            "sam2_commit": SAM2_COMMIT,
            "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
            "object_selector_source_sha256": SAM2_OBJECT_SOURCE_SHA256,
            "base_selector_source_sha256": SAM2_BASE_SOURCE_SHA256,
        },
        "mask dependency contract changed",
    )
    _require(
        payload.get("processing")
        == {
            "deform360_revision": DEFORM360_REVISION,
            "deform360_source_sha256": DEFORM360_SOURCE_SHA256,
            "minimum_processing_cameras": 8,
            "minimum_visual_hull_points": 512,
            "voxel_resolution": 120,
            "cube_half_extent_m": 0.5,
            "first_frame_iterations": 500,
            "warm_start_iterations": 250,
            "cotracker_revision": COTRACKER_REVISION,
            "cotracker_tree": COTRACKER_TREE,
            "cotracker_predictor_sha256": COTRACKER_PREDICTOR_SHA256,
            "cotracker_checkpoint_sha256": COTRACKER_CHECKPOINT_SHA256,
        },
        "processing dependency contract changed",
    )
    _require(
        payload.get("admission")
        == {
            "minimum_camera_count": 3,
            "minimum_point_count": 128,
            "maximum_point_count": 10000,
            "required_frame_count": PREDICTION_FRAME_COUNT,
            "update_frames": list(UPDATE_FRAMES),
            "minimum_test_frame_count": 8,
            "future_geometry_deserialized_for_admission": False,
        },
        "source admission contract changed",
    )
    _require(
        payload.get("failure_accounting")
        == {
            "all_nine_valid_episodes_are_attempted": True,
            "technical_failures_are_retained": True,
            "source_rejections_are_not_predictions": True,
            "implicit_replacement": False,
            "malformed_episode_six_remains_unsealable": True,
        },
        "failure accounting changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("future_object_positions_deserialized_for_admission") is False
        and boundary.get("target_metric_read") is False
        and boundary.get("outcomes_may_not_change_processing_or_admission") is True
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "processing protocol crossed its boundary",
    )


def validate_fresh_processing_sources(
    protocol_path: str | Path,
    technical_lock_path: str | Path,
    source_plan_path: str | Path,
    download_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(protocol_path)
    lock = _load_json(technical_lock_path)
    plan = _load_json(source_plan_path)
    download = _load_json(download_manifest_path)
    validate_fresh_processing_protocol(protocol)
    validate_fresh_technical_lock(lock)
    validate_fresh_source_plan(plan)
    validate_fresh_download_manifest(download)
    _validate_download_against_plan(plan, download)
    _require(
        plan.get("technical_lock_sha256") == lock.get("lock_sha256"),
        "source plan binds another technical lock",
    )
    bindings = protocol["bindings"]
    _require(
        bindings["technical_lock_sha256"] == lock["lock_sha256"]
        and bindings["technical_lock_file_sha256"] == file_sha256(technical_lock_path)
        and bindings["source_plan_sha256"] == plan["source_plan_sha256"]
        and bindings["source_plan_file_sha256"] == file_sha256(source_plan_path)
        and bindings["download_sha256"] == download["download_sha256"]
        and bindings["download_file_sha256"] == file_sha256(download_manifest_path)
        and bindings["source_tree_sha256"] == download["source_tree_sha256"],
        "processing protocol source bindings changed",
    )
    _require(
        bindings["file_count"] == download["file_count"]
        and bindings["total_bytes"] == download["total_bytes"],
        "processing protocol source dimensions changed",
    )
    return protocol, lock, plan, download


def fresh_processing_case(
    lock: Mapping[str, Any], object_id: str, episode_id: int
) -> dict[str, Any]:
    selected = lock["selected_physical_object"]
    _require(selected.get("object_id") == object_id, "object is outside the lock")
    matches = [
        row
        for row in selected.get("valid_episodes", ())
        if isinstance(row, Mapping) and row.get("episode_id") == episode_id
    ]
    _require(len(matches) == 1, "episode is outside the valid lock")
    row = dict(matches[0])
    return {
        "case": f"{object_id}-ep{episode_id:04d}",
        "object_id": object_id,
        "episode_id": episode_id,
        "action": row["action"],
        "bimanual": row["bimanual"],
        "nonprehensile": row["nonprehensile"],
    }


def select_fresh_source_window(
    actions: np.ndarray, openings: np.ndarray
) -> dict[str, Any]:
    """Select the locked action-only 81-frame window."""

    origins = end_effector_origins(actions)
    closed = closure_confidence(openings)
    _require(closed.shape == origins.shape[:2], "action/opening shape mismatch")
    candidates = np.arange(
        CANDIDATE_FIRST_FRAME,
        len(origins) - RAW_FRAME_COUNT + 1,
        CANDIDATE_STRIDE_FRAMES,
        dtype=np.int64,
    )
    _require(len(candidates) > 0, "episode has no complete source window")
    score_start, score_stop = SCORE_STEP_RANGE
    rows: list[dict[str, Any]] = []
    for value in candidates:
        start = int(value)
        selected_origins = origins[start : start + RAW_FRAME_COUNT]
        selected_closed = closed[start : start + RAW_FRAME_COUNT]
        step = np.linalg.norm(np.diff(selected_origins, axis=0), axis=-1)
        adjacent_closure = np.minimum(selected_closed[:-1], selected_closed[1:])
        weighted = step * adjacent_closure
        rows.append(
            {
                "start_frame": start,
                "score_m": float(
                    np.mean(np.sum(weighted[score_start:score_stop], axis=0))
                ),
                "unweighted_future_path_m": float(
                    np.mean(np.sum(step[score_start:score_stop], axis=0))
                ),
                "mean_future_closure_confidence": float(
                    np.mean(adjacent_closure[score_start:score_stop])
                ),
            }
        )
    selected = rows[int(np.argmax([row["score_m"] for row in rows]))]
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
        "candidate_count": len(rows),
        "candidate_scores": rows,
        "input_fields": ["robot.actions[...,0,:]", "robot.openings"],
        "known_future_action_is_conditioning_input": True,
        "object_geometry_read": False,
        "object_tracks_read": False,
        "object_response_read": False,
        "tactile_read": False,
        "target_metric_read": False,
    }


def seal_case_artifact(
    artifact_kind: str,
    *,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = {
        "schema_version": 1,
        "artifact_kind": artifact_kind,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        **case,
        **payload,
    }
    return _seal(artifact, digest_key="result_sha256")


def validate_case_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_kind: str,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
) -> None:
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("artifact_kind") == artifact_kind
        and artifact.get("protocol_id") == PROTOCOL_ID
        and artifact.get("protocol_sha256") == protocol["protocol_sha256"]
        and artifact.get("result_sha256")
        == canonical_sha256(artifact, digest_key="result_sha256")
        and all(artifact.get(key) == value for key, value in case.items()),
        f"{artifact_kind} changed",
    )


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(1024 * 1024)
    _require(b"end_header" in header, "frame-zero PLY header is incomplete")
    text = header[: header.index(b"end_header")].decode("ascii", errors="strict")
    counts: list[int] = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) == 3 and tokens[:2] == ["element", "vertex"]:
            counts.append(int(tokens[2]))
    _require(len(counts) == 1 and counts[0] >= 0, "PLY count is ambiguous")
    return counts[0]


def _valid_stage_inputs(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if _HEX64.fullmatch(str(value.get("robot_sha256"))) is None:
        return False
    if _HEX64.fullmatch(str(value.get("pcd_sha256"))) is None:
        return False
    tactile = value.get("tactile_sha256")
    return isinstance(tactile, Mapping) and all(
        isinstance(name, str)
        and bool(name)
        and _HEX64.fullmatch(str(digest)) is not None
        for name, digest in tactile.items()
    )


def build_fresh_source_admission(
    episode_dir: str | Path,
    metadata_path: str | Path,
    *,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit a processed source without deserializing future object positions."""

    episode = Path(episode_dir)
    metadata_source = Path(metadata_path)
    paths = {
        "metadata": metadata_source,
        "control_meta": episode / "control_points.meta.json",
        "split": episode / "split.json",
        "calibrate": episode / "calibrate.pkl",
        "frame_zero": episode / "start_obj_pcd.ply",
        "future_payload": episode / "final_data.pkl",
    }
    for name, path in paths.items():
        _require(path.is_file(), f"required {name} file is missing: {path}")
    metadata = _load_json(metadata_source)
    control = _load_json(paths["control_meta"])
    split = _load_json(paths["split"])
    reasons: list[str] = []
    sequence = metadata.get("sequences", {}).get(str(case["episode_id"]), {})
    if metadata.get("object") != case["object_id"]:
        reasons.append("metadata object differs from locked object")
    if sequence.get("bimanual") != case["bimanual"]:
        reasons.append("metadata bimanual value differs from lock")
    if control.get("schema") != "deform360.processing/control-points/v1":
        reasons.append("control-point schema differs from upstream v1")
    inputs = control.get("inputs")
    if not _valid_stage_inputs(inputs):
        reasons.append("control-point input provenance is missing or malformed")
    outputs = (
        control.get("outputs") if isinstance(control.get("outputs"), Mapping) else {}
    )
    parameters = (
        control.get("parameters")
        if isinstance(control.get("parameters"), Mapping)
        else {}
    )
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    for name, field in {
        "calibrate": "calibrate_sha256",
        "frame_zero": "start_ply_sha256",
        "split": "split_sha256",
        "future_payload": "final_data_sha256",
    }.items():
        expected = outputs.get(field)
        if not isinstance(expected, str) or expected != hashes[name]:
            reasons.append(f"{name} checksum differs from control-point provenance")
    cameras = parameters.get("cameras", [])
    if not isinstance(cameras, list) or len(set(cameras)) != len(cameras):
        cameras = []
        reasons.append("camera panel is malformed")
    admission = protocol["admission"]
    if len(cameras) < int(admission["minimum_camera_count"]):
        reasons.append("camera panel is below the locked minimum")
    try:
        point_count = _ply_vertex_count(paths["frame_zero"])
    except (OSError, UnicodeError, ValueError) as exc:
        point_count = None
        reasons.append(str(exc))
    if point_count is not None and not (
        int(admission["minimum_point_count"])
        <= point_count
        <= int(admission["maximum_point_count"])
    ):
        reasons.append("frame-zero point count is outside backend admission")
    frame_len = split.get("frame_len")
    active_count = outputs.get("num_active_frames")
    if frame_len != PREDICTION_FRAME_COUNT or active_count != frame_len:
        reasons.append("processed trajectory row count differs from the lock")
    train = split.get("train")
    test = split.get("test")
    expected_train_end = int(0.8 * PREDICTION_FRAME_COUNT)
    if train != [0, expected_train_end] or test != [
        expected_train_end,
        PREDICTION_FRAME_COUNT,
    ]:
        reasons.append("split differs from the released contiguous 80/20 rule")
    if parameters.get("train_fraction") != 0.8:
        reasons.append("train fraction differs from the released rule")
    contact_start = outputs.get("contact_start_frame")
    contact_end = outputs.get("contact_end_frame")
    if not (
        isinstance(contact_start, int)
        and isinstance(contact_end, int)
        and contact_end - contact_start + 1 == PREDICTION_FRAME_COUNT
    ):
        reasons.append("contact window differs from processed row count")
    artifact = seal_case_artifact(
        ADMISSION_KIND,
        protocol=protocol,
        case=case,
        payload={
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "observed_source_contract": {
                "camera_count": len(cameras),
                "cameras": cameras,
                "frame_zero_point_count": point_count,
                "split_frame_count": frame_len,
                "active_frame_count": active_count,
                "train": train,
                "test": test,
                "contact_start_frame": contact_start,
                "contact_end_frame": contact_end,
                "stage_inputs_valid": _valid_stage_inputs(inputs),
            },
            "source_files": {
                name: {"basename": path.name, "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "information_boundary": {
                "future_object_positions_deserialized": False,
                "future_payload_bytes_hashed": True,
                "future_metrics_read": False,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        },
    )
    return artifact


def validate_fresh_source_admission(
    artifact: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
) -> None:
    """Validate one source-admission decision and its information boundary."""

    validate_case_artifact(
        artifact,
        artifact_kind=ADMISSION_KIND,
        protocol=protocol,
        case=case,
    )
    accepted = artifact.get("accepted")
    reasons = artifact.get("rejection_reasons")
    _require(isinstance(accepted, bool), "admission decision is malformed")
    _require(
        isinstance(reasons, list)
        and all(isinstance(reason, str) and bool(reason) for reason in reasons)
        and len(reasons) == len(set(reasons))
        and accepted == (len(reasons) == 0),
        "admission reasons are malformed or inconsistent",
    )
    boundary = artifact.get("information_boundary", {})
    _require(
        boundary.get("future_object_positions_deserialized") is False
        and boundary.get("future_payload_bytes_hashed") is True
        and boundary.get("future_metrics_read") is False
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "admission crossed its future-outcome boundary",
    )
    observed = artifact.get("observed_source_contract", {})
    sources = artifact.get("source_files")
    _require(
        isinstance(sources, Mapping)
        and set(sources)
        == {
            "metadata",
            "control_meta",
            "split",
            "calibrate",
            "frame_zero",
            "future_payload",
        }
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("basename"), str)
            and _HEX64.fullmatch(str(row.get("sha256"))) is not None
            for row in sources.values()
        ),
        "admission source bindings are malformed",
    )
    if accepted:
        admission = protocol["admission"]
        point_count = observed.get("frame_zero_point_count")
        _require(
            isinstance(point_count, int)
            and int(admission["minimum_point_count"])
            <= point_count
            <= int(admission["maximum_point_count"]),
            "accepted admission violates the backend point-count contract",
        )
        _require(
            observed.get("camera_count", 0) >= int(admission["minimum_camera_count"])
            and observed.get("split_frame_count") == PREDICTION_FRAME_COUNT
            and observed.get("active_frame_count") == PREDICTION_FRAME_COUNT
            and observed.get("train") == [0, 60]
            and observed.get("test") == [60, 76]
            and observed.get("stage_inputs_valid") is True,
            "accepted admission violates the frozen source contract",
        )


def write_json_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ADMISSION_KIND",
    "CANDIDATE_FIRST_FRAME",
    "CANDIDATE_STRIDE_FRAMES",
    "COTRACKER_CHECKPOINT_SHA256",
    "COTRACKER_PREDICTOR_SHA256",
    "COTRACKER_REVISION",
    "COTRACKER_TREE",
    "DEFORM360_REVISION",
    "DEFORM360_SOURCE_SHA256",
    "FIRST_UPDATE_FRAME",
    "MASK_KIND",
    "PREDICTION_FRAME_COUNT",
    "PREPARATION_KIND",
    "PROCESSING_KIND",
    "PROTOCOL_ID",
    "RAW_FRAME_COUNT",
    "SAM2_BASE_SOURCE_SHA256",
    "SAM2_CHECKPOINT_SHA256",
    "SAM2_COMMIT",
    "SAM2_OBJECT_SOURCE_SHA256",
    "SCORE_STEP_RANGE",
    "UPDATE_FRAMES",
    "WINDOW_SELECTION_KIND",
    "WINDOW_STAGE_KIND",
    "build_fresh_processing_protocol",
    "build_fresh_source_admission",
    "canonical_sha256",
    "fresh_processing_case",
    "seal_case_artifact",
    "select_fresh_source_window",
    "validate_case_artifact",
    "validate_fresh_source_admission",
    "validate_fresh_processing_protocol",
    "validate_fresh_processing_sources",
    "write_json_artifact",
]
