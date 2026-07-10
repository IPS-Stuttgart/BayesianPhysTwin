import numpy as np
import pytest

from bayesian_phystwin.phystwin_state_injection import (
    _released_self_collision_for_case,
    _trajectory_error,
    estimate_endpoint_velocity_delta,
)


def test_velocity_delta_recovers_linear_correction_motion() -> None:
    frame_dt = 0.05
    velocity = np.array([[0.2, -0.1, 0.05], [-0.3, 0.0, 0.4]])
    offset = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    time = frame_dt * np.arange(4)
    history = offset[None] + time[:, None, None] * velocity[None]

    estimated = estimate_endpoint_velocity_delta(history, frame_dt=frame_dt)

    np.testing.assert_allclose(estimated, velocity, atol=1e-12)


def test_velocity_delta_rejects_invalid_history() -> None:
    with pytest.raises(ValueError, match="T>=2"):
        estimate_endpoint_velocity_delta(np.zeros((1, 3, 3)), frame_dt=0.1)
    with pytest.raises(ValueError, match="frame_dt"):
        estimate_endpoint_velocity_delta(np.zeros((2, 3, 3)), frame_dt=0.0)


def test_trajectory_error_uses_vector_and_coordinate_units() -> None:
    reference = np.zeros((2, 1, 3))
    candidate = np.ones((2, 1, 3))

    result = _trajectory_error(reference, candidate)

    assert result["coordinate_rmse_m"] == pytest.approx(1.0)
    assert result["vector_rmse_m"] == pytest.approx(np.sqrt(3.0))
    assert result["maximum_norm_m"] == pytest.approx(np.sqrt(3.0))


@pytest.mark.parametrize(
    ("case_name", "expected"),
    (
        ("double_lift_cloth_1", True),
        ("cloth_blue_fold", True),
        ("double_push_package", True),
        ("single_lift_sloth", False),
        ("single_push_rope", False),
    ),
)
def test_released_self_collision_matches_phystwin_case_rule(
    case_name: str, expected: bool
) -> None:
    assert _released_self_collision_for_case(case_name) is expected
