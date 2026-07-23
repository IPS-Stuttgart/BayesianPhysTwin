from __future__ import annotations

import inspect

import numpy as np
import pytest

from bayesian_phystwin.deform360_raw_camera_observation import (
    _projection_matrix,
    project_world_points,
)
from bayesian_phystwin.deform360_raw_camera_uncertainty import (
    build_raw_camera_uncertainty_case,
    jacobian_measurement_covariance,
    leave_one_camera_out_covariance,
    normalized_leave_one_camera_out_triangulation_dispersion,
    projection_jacobian,
)


def _camera_to_world(x: float, y: float = 0.0) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = (x, y, 0.0)
    return result


def test_projection_jacobian_matches_finite_difference() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 790.0, 240.0], [0.0, 0.0, 1.0]])
    projection = _projection_matrix(intrinsics, _camera_to_world(-0.4, 0.1))
    point = np.array([0.1, -0.05, 2.5])
    analytic = projection_jacobian(point, projection)
    numerical = np.empty((2, 3))
    epsilon = 1.0e-6
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = epsilon
        plus = project_world_points(
            (point + offset)[None], intrinsics, _camera_to_world(-0.4, 0.1)
        )[0][0]
        minus = project_world_points(
            (point - offset)[None], intrinsics, _camera_to_world(-0.4, 0.1)
        )[0][0]
        numerical[:, axis] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-6)


def test_wider_baseline_reduces_maximum_covariance_eigenvalue() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    point = np.array([0.0, 0.0, 3.0])

    def covariance(baseline: float) -> np.ndarray:
        matrices = [
            _projection_matrix(intrinsics, _camera_to_world(-baseline)),
            _projection_matrix(intrinsics, _camera_to_world(baseline)),
        ]
        result, diagnostic = jacobian_measurement_covariance(
            point,
            matrices,
            1.0,
            maximum_condition_number=1.0e12,
        )
        assert diagnostic["decision"] == "accepted"
        assert result is not None
        return result

    assert np.max(np.linalg.eigvalsh(covariance(0.5))) < np.max(
        np.linalg.eigvalsh(covariance(0.05))
    )


def test_leave_one_camera_covariance_is_positive_semidefinite() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    extrinsics = {
        "left": _camera_to_world(-0.5),
        "middle": _camera_to_world(0.0, 0.4),
        "right": _camera_to_world(0.5),
    }
    point = np.array([0.05, -0.02, 3.0])
    matrices = {
        camera: _projection_matrix(intrinsics, transform)
        for camera, transform in extrinsics.items()
    }
    observations = {
        camera: project_world_points(point[None], intrinsics, transform)[0][0]
        for camera, transform in extrinsics.items()
    }
    observations["middle"] += np.array([0.3, -0.2])

    covariance, samples = leave_one_camera_out_covariance(observations, matrices)

    assert samples.shape == (3, 3)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-15
    assert np.trace(covariance) > 0.0


def test_normalized_loo_dispersion_is_per_update_center_and_q90() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    extrinsics = {
        "left": _camera_to_world(-0.6),
        "middle": _camera_to_world(0.0, 0.4),
        "right": _camera_to_world(0.6),
        "upper": _camera_to_world(0.0, -0.5),
    }
    matrices = {
        camera: _projection_matrix(intrinsics, transform)
        for camera, transform in extrinsics.items()
    }
    frame_zero = np.array(
        [
            [-1.0, 0.0, 3.0],
            [0.0, 0.1, 3.0],
            [1.0, 0.0, 3.0],
        ]
    )

    def observations(point: np.ndarray, offset_scale: float) -> dict[str, np.ndarray]:
        result = {
            camera: project_world_points(
                point[None],
                intrinsics,
                transform,
            )[0][0]
            for camera, transform in extrinsics.items()
        }
        result["middle"] = result["middle"] + offset_scale * np.array([0.3, -0.2])
        result["upper"] = result["upper"] + offset_scale * np.array([-0.1, 0.25])
        return result

    first = observations(frame_zero[0], 1.0)
    second = observations(frame_zero[1], 2.0)
    third = observations(frame_zero[2], 3.0)
    nested_observations = ((first, second), (third, None))
    frame_zero_before = frame_zero.copy()
    observation_before = tuple(
        {camera: pixel.copy() for camera, pixel in observation.items()}
        for observation in (first, second, third)
    )

    result = normalized_leave_one_camera_out_triangulation_dispersion(
        nested_observations,
        matrices,
        frame_zero,
    )

    expected = []
    for update_index, center_index, source in (
        (0, 0, first),
        (0, 1, second),
        (1, 0, third),
    ):
        covariance, samples = leave_one_camera_out_covariance(source, matrices)
        assert len(samples) == 4
        value = float(np.sqrt(np.trace(covariance)) / 2.0)
        expected.append(value)
        assert result["normalized_dispersion_by_update_center"][
            update_index, center_index
        ] == pytest.approx(value)
    assert np.isnan(result["normalized_dispersion_by_update_center"][1, 1])
    assert result["object_diameter_m"] == pytest.approx(2.0)
    assert result["valid_update_center_count"] == 3
    assert result["q90_normalized_dispersion_by_update"] == pytest.approx(
        (
            np.quantile(expected[:2], 0.90),
            expected[2],
        )
    )
    assert result["pooled_q90_normalized_dispersion"] == pytest.approx(
        np.quantile(expected, 0.90)
    )
    assert result["pooled_summary_causal_for_online_routing"] is False
    np.testing.assert_array_equal(frame_zero, frame_zero_before)
    for source, before in zip(
        (first, second, third),
        observation_before,
        strict=True,
    ):
        for camera, pixel in source.items():
            np.testing.assert_array_equal(pixel, before[camera])


def test_normalized_loo_dispersion_fails_closed_without_three_views() -> None:
    matrices = {
        "left": _projection_matrix(np.eye(3), _camera_to_world(-0.5)),
        "right": _projection_matrix(np.eye(3), _camera_to_world(0.5)),
    }
    point = np.array([0.0, 0.0, 3.0])
    observations = {
        camera: project_world_points(
            point[None],
            np.eye(3),
            transform,
        )[0][0]
        for camera, transform in {
            "left": _camera_to_world(-0.5),
            "right": _camera_to_world(0.5),
        }.items()
    }

    result = normalized_leave_one_camera_out_triangulation_dispersion(
        ((observations,),),
        matrices,
        np.array([[-1.0, 0.0, 3.0], [1.0, 0.0, 3.0]]),
    )

    assert result["valid_update_center_count"] == 0
    assert result["q90_normalized_dispersion_by_update"] == (None,)
    assert result["pooled_q90_normalized_dispersion"] is None
    assert result["pooled_summary_causal_for_online_routing"] is False
    assert np.isnan(result["normalized_dispersion_by_update_center"][0, 0])
    assert result["leave_one_camera_out_sample_count_by_update_center"][0, 0] == 0


def test_uncertainty_builder_has_no_target_or_outcome_argument() -> None:
    parameters = inspect.signature(build_raw_camera_uncertainty_case).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters
    assert set(parameters) == {
        "panel_case_dir",
        "processed_episode_dir",
        "measurement_dir",
        "output_dir",
        "runtime",
        "config",
    }
