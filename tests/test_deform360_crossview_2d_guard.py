from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
)
from bayesian_phystwin.deform360_crossview_2d_guard import (
    camera_balanced_pair_weights,
    predict_direct_crossview_guarded_candidate_arrays,
)
from bayesian_phystwin.deform360_crossview_2d_guard_artifact import (
    ARCHIVE_FILENAME,
    ARTIFACT_KIND,
    PROTOCOL_ID,
    REPORT_FILENAME,
    build_direct_crossview_guard_prediction,
    load_direct_crossview_guard_prediction,
)
from bayesian_phystwin.deform360_crossview_guard import CrossViewGuardConfig
from bayesian_phystwin.deform360_raw_camera_observation import project_world_points


def _camera_to_world(x: float, y: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = (x, y, 0.0)
    return result


def _supplement(frame_zero: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    camera_count = 8
    intrinsics = np.repeat(
        np.asarray(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
        )[None],
        camera_count,
        axis=0,
    )
    angles = np.linspace(0.0, 2.0 * np.pi, camera_count, endpoint=False)
    camera_to_world = np.stack(
        [_camera_to_world(0.4 * np.cos(angle), 0.4 * np.sin(angle)) for angle in angles]
    )
    initial = np.empty((camera_count, len(frame_zero), 2), dtype=np.float32)
    tracks = np.empty((1, camera_count, len(frame_zero), 2), dtype=np.float32)
    for camera in range(camera_count):
        initial[camera] = project_world_points(
            frame_zero, intrinsics[camera], camera_to_world[camera]
        )[0]
        tracks[0, camera] = project_world_points(
            observed, intrinsics[camera], camera_to_world[camera]
        )[0]
    return {
        "track_pixels_xy": tracks,
        "track_visibility": np.ones(tracks.shape[:-1], dtype=bool),
        "frame_zero_pixels_xy": initial,
        "frame_zero_support": np.ones(initial.shape[:-1], dtype=bool),
        "center_ids": np.arange(len(frame_zero), dtype=np.int64),
        "selected_cameras": np.asarray(
            [f"camera-{index}" for index in range(camera_count)]
        ),
        "update_frames": np.asarray([2], dtype=np.int64),
        "center_frame_zero_points_m": frame_zero.astype(np.float32),
        "intrinsics": intrinsics,
        "camera_to_world": camera_to_world,
    }


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_count = 12
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.15 * np.cos(angle), 0.10 * np.sin(angle), np.full(point_count, 3.0))
    )
    mode = np.sin(2.0 * angle)
    correction = np.zeros_like(frame_zero)
    correction[:, 2] = 0.008 * mode
    baseline = np.repeat(frame_zero[None], 5, axis=0).astype(np.float32)
    response = np.zeros_like(baseline)
    response[:, :, 2] = np.linspace(0.0, 0.008, 5)[:, None] * mode
    return frame_zero, correction, baseline, response


def _development() -> Deform360BiasAwareDevelopmentConfig:
    return Deform360BiasAwareDevelopmentConfig(
        update_frames=(2,),
        physical_response_rank=1,
        minimum_physical_response_m=0.0005,
    )


def test_camera_balancing_does_not_accumulate_duplicated_points() -> None:
    original = camera_balanced_pair_weights(
        np.asarray([0, 0, 1, 1]), effective_observations_per_camera=8.0
    )
    duplicated = camera_balanced_pair_weights(
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        effective_observations_per_camera=8.0,
    )

    np.testing.assert_allclose(np.sum(original[:2]), np.sum(duplicated[:4]))
    np.testing.assert_allclose(np.sum(original[2:]), np.sum(duplicated[4:]))


def test_direct_2d_guard_accepts_crossview_supported_local_mode() -> None:
    frame_zero, correction, baseline, response = _inputs()
    supplement = _supplement(frame_zero, frame_zero + correction)

    report, guarded = predict_direct_crossview_guarded_candidate_arrays(
        baseline,
        response,
        frame_zero,
        np.ones(len(frame_zero)),
        supplement,
        development_config=_development(),
        guard_config=CrossViewGuardConfig(
            minimum_heldout_improvement_fraction=0.01,
            minimum_heldout_mean_error_reduction_px=0.01,
        ),
    )

    assert report["accepted_count"] == 1
    assert report["updates"][0]["fit_a_validation_passed"] is True
    assert report["updates"][0]["fit_b_validation_passed"] is True
    assert not np.array_equal(guarded[3:], baseline[3:])


def test_direct_2d_guard_removes_per_camera_constant_offsets() -> None:
    frame_zero, _, baseline, response = _inputs()
    supplement = _supplement(frame_zero, frame_zero)
    offsets = np.column_stack(
        (np.linspace(-4.0, 4.0, 8), np.linspace(3.0, -3.0, 8))
    )
    supplement["track_pixels_xy"][0] += offsets[:, None]

    report, guarded = predict_direct_crossview_guarded_candidate_arrays(
        baseline,
        response,
        frame_zero,
        np.ones(len(frame_zero)),
        supplement,
        development_config=_development(),
    )

    assert report["accepted_count"] == 0
    assert guarded.tobytes() == baseline.tobytes()


def _write_direct_artifact(root: Path) -> None:
    root.mkdir()
    baseline = np.zeros((4, 6, 3), dtype=np.float32)
    archive = root / ARCHIVE_FILENAME
    np.savez_compressed(
        archive,
        baseline_m=baseline,
        direct_2d_crossview_guarded_m=baseline.copy(),
        center_ids=np.asarray([0, 3, 5]),
        update_frames=np.asarray([1, 2]),
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "output": {"archive_file_sha256": file_sha256(archive)},
    }
    report["result_sha256"] = canonical_sha256(
        report, digest_key="result_sha256"
    )
    (root / REPORT_FILENAME).write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )


def test_direct_prediction_builder_accepts_no_target_or_outcome() -> None:
    parameters = inspect.signature(
        build_direct_crossview_guard_prediction
    ).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters


def test_direct_prediction_rejects_mutated_archive(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_direct_artifact(artifact)
    load_direct_crossview_guard_prediction(artifact)
    with (artifact / ARCHIVE_FILENAME).open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="archive checksum"):
        load_direct_crossview_guard_prediction(artifact)
