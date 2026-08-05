from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_causal_robot_prefix import (
    DEFORM360_PROCESSING_REVISION,
    evaluate_causal_robot_prefix_quality,
    load_deform360_causal_robot_prefix_lock,
    run_causal_capture_loop,
    validate_deform360_causal_robot_prefix_lock,
    verify_causal_robot_prefix_artifact,
    write_causal_robot_prefix_artifact,
)


def _quality_gate() -> dict[str, object]:
    return {
        "minimum_inlier_cameras_per_part": 2,
        "minimum_direct_wrist_fraction": 0.75,
        "minimum_both_fingers_fraction": 0.50,
        "contact_tail_frame_count": 6,
        "minimum_contact_ready_frames": 4,
        "minimum_opening_m": 0.04,
        "maximum_opening_m": 0.112,
        "maximum_translation_step_m": 0.05,
        "maximum_rotation_step_deg": 20.0,
        "rotation_matrix_tolerance": 1e-3,
    }


def _lock() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform360-causal-robot-prefix-lock",
        "schema_version": 1,
        "status": "locked-source-only-pre-estimation",
        "protocol_id": "deform360-official-hub-visuotactile-v1",
        "source_case": {
            "object_id": "026-sock-cloth",
            "source_episode_id": 7,
            "processing_episode_index": 0,
            "bimanual": True,
            "cameras": ["camera-a", "camera-b", "camera-c"],
        },
        "causal_window": {
            "source_frame_start": 108,
            "causal_frame_stop": 150,
            "contact_start_frame": 144,
            "observed_frame_count": 42,
            "untouched_future_frame_start": 150,
        },
        "estimator": {
            "repository": "lhy0807/deform360",
            "revision": DEFORM360_PROCESSING_REVISION,
            "implementation_revision": "1" * 40,
            "camera_policy": "all-calibrated-cameras",
            "decode_policy": "sequential-read-discard-before-start-stop-before-future",
            "seed": 0,
        },
        "quality_gate": _quality_gate(),
        "information_boundary": {
            "calibration_camera_prefix_allowed": True,
            "calibration_tactile_prefix_allowed": True,
            "calibration_scores_opened": False,
            "confirmation_payloads_opened": False,
            "future_camera_frames_used": False,
            "future_tactile_frames_used": False,
            "held_v8_accessed": False,
            "target_outcomes_used": False,
        },
    }
    value["artifact_id"] = content_id(value)
    return value


class _Capture:
    def __init__(self, frames: list[int]) -> None:
        self.frames = frames
        self.index = 0

    def read(self) -> tuple[bool, int | None]:
        if self.index >= len(self.frames):
            return False, None
        value = self.frames[self.index]
        self.index += 1
        return True, value


def _quality_inputs() -> dict[str, Any]:
    frame_count = 8
    transforms = np.tile(np.eye(4), (frame_count, 2, 1, 1))
    transforms[:, 0, 0, 3] = np.linspace(0.0, 0.007, frame_count)
    transforms[:, 1, 1, 3] = np.linspace(0.0, 0.007, frame_count)
    return {
        "transforms": transforms,
        "openings_m": np.full((frame_count, 2), 0.08),
        "part_inlier_camera_counts": np.full((frame_count, 2, 3), 2),
        "source_frame_ids": np.arange(108, 116),
        "bimanual": True,
        "quality_gate": _quality_gate(),
    }


def _arrays(admitted: bool = True) -> tuple[dict[str, np.ndarray], object]:
    inputs = _quality_inputs()
    if not admitted:
        inputs["openings_m"][0, 0] = 0.2
    quality = evaluate_causal_robot_prefix_quality(**inputs)
    transforms = inputs["transforms"]
    openings = inputs["openings_m"]
    actions = np.zeros((8, 2, 5, 3))
    actions[:, :, 0] = transforms[:, :, :3, 3]
    actions[:, :, 1:4] = transforms[:, :, :3, :3]
    actions[:, :, 4, 0] = openings
    arrays = {
        "source_frame_ids": inputs["source_frame_ids"],
        "actions": actions,
        "T_worlds": transforms,
        "openings": openings,
        "part_inlier_camera_counts": inputs["part_inlier_camera_counts"],
        "marker_inlier_camera_counts": np.full((8, 2, 8), 2, dtype=np.int16),
        "raw_marker_detection_counts": np.ones((8, 3, 16), dtype=np.int16),
    }
    return arrays, quality


def test_lock_is_content_addressed_and_fail_closed() -> None:
    lock = _lock()
    assert validate_deform360_causal_robot_prefix_lock(lock) == lock["artifact_id"]

    changed = copy.deepcopy(lock)
    changed["quality_gate"]["minimum_contact_ready_frames"] = 3
    descriptor = dict(changed)
    descriptor.pop("artifact_id")
    changed["artifact_id"] = content_id(descriptor)
    with pytest.raises(ValueError, match="quality gate changed"):
        validate_deform360_causal_robot_prefix_lock(changed)


def test_lock_roundtrip_rejects_future_use(tmp_path: Path) -> None:
    lock = _lock()
    path = tmp_path / "lock.json"
    write_atomic_json(lock, path, overwrite=False)
    assert load_deform360_causal_robot_prefix_lock(path)["artifact_id"] == lock[
        "artifact_id"
    ]

    changed = copy.deepcopy(lock)
    changed["information_boundary"]["future_camera_frames_used"] = True
    descriptor = dict(changed)
    descriptor.pop("artifact_id")
    changed["artifact_id"] = content_id(descriptor)
    with pytest.raises(ValueError, match="information boundary changed"):
        validate_deform360_causal_robot_prefix_lock(changed)


def test_committed_source_only_smoke_lock_remains_valid() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "protocols/locks/deform360_official_hub_causal_robot_prefix_smoke_v1.json"
    )

    lock = load_deform360_causal_robot_prefix_lock(path)

    assert lock["artifact_id"] == (
        "7e4f7a30d9ad00da9f47d2c0debd42fea704c0985e65f78f8cd4f584dc52bc34"
    )
    assert len(lock["source_case"]["cameras"]) == 36
    assert lock["information_boundary"]["calibration_scores_opened"] is False


def test_causal_decoder_is_camera_order_invariant_and_never_reads_future() -> None:
    first = {
        "camera-b": _Capture([0, 1, 2, 3, 900, 901]),
        "camera-a": _Capture([10, 11, 12, 13, 910, 911]),
    }
    second = {
        "camera-a": _Capture([10, 11, 12, 13, -10, -11]),
        "camera-b": _Capture([0, 1, 2, 3, -20, -21]),
    }
    observed_first: list[tuple[int, str, int]] = []
    observed_second: list[tuple[int, str, int]] = []

    count_first = run_causal_capture_loop(
        first,
        source_frame_start=2,
        causal_frame_stop=4,
        process_frame=lambda frame, camera, value: observed_first.append(
            (frame, camera, value)
        ),
    )
    count_second = run_causal_capture_loop(
        second,
        source_frame_start=2,
        causal_frame_stop=4,
        process_frame=lambda frame, camera, value: observed_second.append(
            (frame, camera, value)
        ),
    )

    assert count_first == count_second == 4
    assert observed_first == observed_second
    assert [camera for _, camera, _ in observed_first] == [
        "camera-a",
        "camera-b",
        "camera-a",
        "camera-b",
    ]
    assert all(capture.index == 4 for capture in (*first.values(), *second.values()))


def test_quality_admits_supported_plausible_prefix() -> None:
    result = evaluate_causal_robot_prefix_quality(**_quality_inputs())

    assert result.admitted
    assert result.reason_codes == ()
    assert result.summary["contact_ready_frames_by_gripper"] == [6, 6]


def test_quality_rejects_impossible_opening_and_missing_wrist_support() -> None:
    inputs = _quality_inputs()
    inputs["openings_m"][3, 0] = 0.417
    inputs["part_inlier_camera_counts"][:, 1, 0] = 1

    result = evaluate_causal_robot_prefix_quality(**inputs)

    assert not result.admitted
    assert "opening-outside-released-range" in result.reason_codes
    assert "insufficient-direct-wrist-support" in result.reason_codes
    assert "insufficient-contact-tail-support" in result.reason_codes


def test_diagnostic_artifact_authorizes_anchor_only_after_gate(tmp_path: Path) -> None:
    lock = _lock()
    arrays, quality = _arrays(admitted=False)
    manifest_path = tmp_path / "prefix.json"

    manifest = write_causal_robot_prefix_artifact(
        output_npz=tmp_path / "prefix.npz",
        output_manifest=manifest_path,
        arrays=arrays,
        lock=lock,
        lock_file_sha256="2" * 64,
        implementation_revision="3" * 40,
        source_artifacts={"camera-a/undistorted.mp4": "4" * 64},
        quality=quality,  # type: ignore[arg-type]
    )

    assert manifest["anchor_authorized"] is False
    assert manifest["fallback"] == "no-contact-anchor"
    verified = verify_causal_robot_prefix_artifact(manifest_path)
    assert verified["artifact_id"] == manifest["artifact_id"]
    assert verified["quality"]["reason_codes"] == [
        "opening-outside-released-range"
    ]


def test_archive_verification_detects_mutation(tmp_path: Path) -> None:
    lock = _lock()
    arrays, quality = _arrays()
    manifest_path = tmp_path / "prefix.json"
    archive_path = tmp_path / "prefix.npz"
    write_causal_robot_prefix_artifact(
        output_npz=archive_path,
        output_manifest=manifest_path,
        arrays=arrays,
        lock=lock,
        lock_file_sha256="2" * 64,
        implementation_revision="3" * 40,
        source_artifacts={"camera-a/undistorted.mp4": "4" * 64},
        quality=quality,  # type: ignore[arg-type]
    )
    archive_path.write_bytes(archive_path.read_bytes() + b"mutation")

    with pytest.raises(ValueError, match="archive digest changed"):
        verify_causal_robot_prefix_artifact(manifest_path)
