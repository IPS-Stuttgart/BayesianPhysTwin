from __future__ import annotations

import inspect

import numpy as np

from bayesian_phystwin.deform360_raw_camera_observation import (
    _projection_matrix,
    project_world_points,
)
from bayesian_phystwin.deform360_raw_camera_uncertainty import (
    build_raw_camera_uncertainty_case,
    jacobian_measurement_covariance,
    leave_one_camera_out_covariance,
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
