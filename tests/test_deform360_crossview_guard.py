from __future__ import annotations

import numpy as np

from bayesian_phystwin.bias_aware_belief import BiasAwareStateUpdateConfig
from bayesian_phystwin.deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
)
from bayesian_phystwin.deform360_crossview_guard import (
    CrossViewGuardConfig,
    conservative_triangulation_variance_m2,
    deterministic_camera_halves,
    heldout_reprojection_diagnostic,
    predict_crossview_guarded_candidate_arrays,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    project_world_points,
)


def _camera_to_world(x: float, y: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = (x, y, 0.0)
    return result


def _calibration(camera_count: int) -> tuple[np.ndarray, np.ndarray]:
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
    return intrinsics, camera_to_world


def _projection_matrix(intrinsic: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    return intrinsic @ np.linalg.inv(camera_to_world)[:3]


def _synthetic_supplement(
    frame_zero: np.ndarray,
    observed: np.ndarray,
    *,
    update_frames: tuple[int, ...] = (2,),
) -> dict[str, np.ndarray]:
    camera_count = 8
    intrinsics, camera_to_world = _calibration(camera_count)
    tracks = np.empty(
        (len(update_frames), camera_count, len(frame_zero), 2), dtype=np.float32
    )
    initial_pixels = np.empty((camera_count, len(frame_zero), 2), dtype=np.float32)
    for camera_index in range(camera_count):
        initial_pixels[camera_index] = project_world_points(
            frame_zero, intrinsics[camera_index], camera_to_world[camera_index]
        )[0]
        for update_index in range(len(update_frames)):
            tracks[update_index, camera_index] = project_world_points(
                observed, intrinsics[camera_index], camera_to_world[camera_index]
            )[0]
    return {
        "track_pixels_xy": tracks,
        "track_visibility": np.ones(tracks.shape[:-1], dtype=bool),
        "frame_zero_pixels_xy": initial_pixels,
        "frame_zero_support": np.ones(initial_pixels.shape[:-1], dtype=bool),
        "center_ids": np.arange(len(frame_zero), dtype=np.int64),
        "selected_cameras": np.asarray(
            [f"camera-{index}" for index in range(camera_count)]
        ),
        "update_frames": np.asarray(update_frames, dtype=np.int64),
        "center_frame_zero_points_m": frame_zero.astype(np.float32),
        "intrinsics": intrinsics,
        "camera_to_world": camera_to_world,
    }


def test_camera_halves_are_disjoint_and_identity_deterministic() -> None:
    cameras = ["camera-3", "camera-0", "camera-6", "camera-1", "camera-5", "camera-2"]

    first, second = deterministic_camera_halves(cameras)

    assert {cameras[index] for index in first} == {"camera-0", "camera-2", "camera-5"}
    assert {cameras[index] for index in second} == {"camera-1", "camera-3", "camera-6"}
    assert not set(first).intersection(second)


def test_duplicate_camera_block_does_not_increase_precision() -> None:
    intrinsics, camera_to_world = _calibration(2)
    matrices = [
        _projection_matrix(intrinsics[index], camera_to_world[index])
        for index in range(2)
    ]
    point = np.asarray([0.02, -0.01, 3.0])

    original = conservative_triangulation_variance_m2(
        point, matrices, pixel_std=2.0
    )
    duplicated = conservative_triangulation_variance_m2(
        point, matrices + matrices, pixel_std=2.0
    )

    np.testing.assert_allclose(duplicated, original, rtol=1e-12, atol=0.0)


def test_heldout_reprojection_distinguishes_helpful_and_harmful_updates() -> None:
    point_count = 8
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    truth = np.column_stack(
        (0.15 * np.cos(angle), 0.10 * np.sin(angle), np.full(point_count, 3.0))
    )
    local = 0.008 * np.sin(2.0 * angle)
    baseline = truth.copy()
    baseline[:, 2] -= local
    correction = np.zeros_like(baseline)
    correction[:, 2] = local
    supplement = _synthetic_supplement(baseline, truth)

    helpful = heldout_reprojection_diagnostic(
        baseline,
        correction,
        supplement,
        0,
        np.asarray([1, 3, 5, 7]),
        huber_delta_px=3.0,
    )
    harmful = heldout_reprojection_diagnostic(
        baseline,
        -correction,
        supplement,
        0,
        np.asarray([1, 3, 5, 7]),
        huber_delta_px=3.0,
    )

    assert helpful["huber_improvement_fraction"] > 0.99
    assert helpful["mean_error_reduction_px"] > 0.1
    assert harmful["huber_improvement_fraction"] < 0.0


def _development_config() -> Deform360BiasAwareDevelopmentConfig:
    return Deform360BiasAwareDevelopmentConfig(
        update_frames=(2,),
        minimum_available_center_count=8,
        minimum_motion_center_count=3,
        physical_response_rank=1,
        minimum_physical_response_m=0.0005,
        minimum_observed_motion_m=0.0005,
        minimum_physical_agreement_gain=0.4,
        state_update=BiasAwareStateUpdateConfig(
            observation_std_m=0.003,
            state_prior_std_m=0.05,
            shared_bias_prior_std_m=0.02,
            camera_bias_prior_std_m=0.01,
        ),
    )


def test_crossview_guard_accepts_bidirectionally_supported_local_mode() -> None:
    point_count = 12
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.15 * np.cos(angle), 0.10 * np.sin(angle), np.full(point_count, 3.0))
    )
    mode = np.sin(2.0 * angle)
    correction = np.zeros_like(frame_zero)
    correction[:, 2] = 0.008 * mode
    frame_count = 5
    baseline = np.repeat(frame_zero[None], frame_count, axis=0).astype(np.float32)
    response = np.zeros_like(baseline)
    response[:, :, 2] = np.linspace(0.0, 0.008, frame_count)[:, None] * mode
    supplement = _synthetic_supplement(frame_zero, frame_zero + correction)

    report, guarded = predict_crossview_guarded_candidate_arrays(
        baseline,
        response,
        frame_zero,
        np.ones(point_count),
        supplement,
        development_config=_development_config(),
        raw_config=RawCameraObservationConfig(
            selected_camera_count=8, update_frames=(2,)
        ),
        guard_config=CrossViewGuardConfig(
            minimum_heldout_improvement_fraction=0.01,
            minimum_heldout_mean_error_reduction_px=0.01,
        ),
    )

    assert report["accepted_count"] == 1
    assert report["updates"][0]["fit_a_validation_passed"] is True
    assert report["updates"][0]["fit_b_validation_passed"] is True
    assert not np.array_equal(guarded[3:], baseline[3:])


def test_crossview_guard_is_bit_exact_when_view_support_is_insufficient() -> None:
    point_count = 12
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.15 * np.cos(angle), 0.10 * np.sin(angle), np.full(point_count, 3.0))
    )
    frame_count = 5
    baseline = np.repeat(frame_zero[None], frame_count, axis=0).astype(np.float32)
    response = np.zeros_like(baseline)
    response[:, :, 2] = np.linspace(0.0, 0.008, frame_count)[:, None]
    supplement = _synthetic_supplement(frame_zero, frame_zero)
    supplement["track_visibility"][:, 2:] = False

    report, guarded = predict_crossview_guarded_candidate_arrays(
        baseline,
        response,
        frame_zero,
        np.ones(point_count),
        supplement,
        development_config=_development_config(),
        raw_config=RawCameraObservationConfig(
            selected_camera_count=8, update_frames=(2,)
        ),
    )

    assert report["accepted_count"] == 0
    assert report["exact_fallback_count"] == 1
    assert guarded.tobytes() == baseline.tobytes()
