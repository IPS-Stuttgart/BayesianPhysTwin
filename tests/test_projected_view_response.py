from __future__ import annotations

import numpy as np

from bayesian_phystwin.action_response_admission import (
    ActionResponseAdmissionConfig,
    evaluate_action_response_admission,
)
from bayesian_phystwin.projected_view_response import (
    build_projected_view_response,
)


def _fixture() -> dict[str, np.ndarray]:
    sensor_count = 3
    frame_count = 4
    point_count = 6
    initial = np.zeros((sensor_count, point_count, 2), dtype=np.float64)
    initial[..., 0] = np.linspace(100.0, 200.0, point_count)
    initial[..., 1] = 120.0
    physical = np.repeat(initial[:, None], frame_count, axis=1)
    shape = np.linspace(-1.0, 1.0, point_count)
    for frame, progress in enumerate(np.linspace(0.0, 1.0, frame_count)):
        physical[:, frame, :, 1] += 80.0 * progress * shape
    return {
        "physical_pixels_px": physical,
        "observed_pixels_px": physical.copy(),
        "observation_validity": np.ones(
            (sensor_count, frame_count, point_count),
            dtype=bool,
        ),
        "initial_depth_m": np.ones((sensor_count, point_count)),
        "focal_lengths_px": np.full((sensor_count, 2), 500.0),
        "cycle_error_px": np.zeros(
            (sensor_count, frame_count, point_count)
        ),
        "source_confidence": np.full(
            (sensor_count, frame_count, point_count),
            0.9,
        ),
    }


def _admission(projected: object):
    action = np.zeros((4, 1, 3), dtype=np.float64)
    action[:, 0, 0] = np.linspace(0.0, 0.01, 4)
    return evaluate_action_response_admission(
        projected.physical_positions_m,
        projected.observed_positions_m,
        projected.observation_validity,
        projected.observation_covariance_m2,
        projected.prior_reliability,
        projected.association_probability,
        action,
        ("camera-0", "camera-1", "camera-2"),
        tuple(f"node-{index}" for index in range(6)),
        np.ones(6),
        physical_prefix_id="physical",
        observation_prefix_id="observation",
        action_prefix_id="action",
        config=ActionResponseAdmissionConfig(
            minimum_identifiable_physical_rms_m=1e-4,
            minimum_observed_response_rms_m=1e-4,
            minimum_response_gain=0.05,
            minimum_direction_cosine=0.8,
        ),
    )


def test_projected_physical_response_is_admitted() -> None:
    projected = build_projected_view_response(**_fixture())

    result = _admission(projected)

    assert result.admitted
    assert result.passing_group_count == 3
    assert all(group.direction_cosine > 0.99 for group in result.groups)


def test_camera_translation_without_shape_response_is_rejected() -> None:
    inputs = _fixture()
    observed = np.asarray(inputs["observed_pixels_px"]).copy()
    observed[:] = observed[:, :1]
    for frame, progress in enumerate(np.linspace(0.0, 1.0, 4)):
        observed[:, frame, :, 0] += 12.0 * progress
        observed[:, frame, :, 1] -= 5.0 * progress
    inputs["observed_pixels_px"] = observed
    projected = build_projected_view_response(**inputs)

    result = _admission(projected)

    assert not result.admitted
    assert result.passing_group_count == 0


def test_cycle_error_changes_association_not_prior_reliability() -> None:
    first = build_projected_view_response(**_fixture())
    inputs = _fixture()
    inputs["cycle_error_px"][:, -1] = 20.0
    second = build_projected_view_response(**inputs)

    np.testing.assert_array_equal(
        second.prior_reliability,
        first.prior_reliability,
    )
    assert np.all(
        second.association_probability[:, -1]
        < first.association_probability[:, -1]
    )


def test_state_residual_does_not_change_prior_reliability() -> None:
    first = build_projected_view_response(**_fixture())
    inputs = _fixture()
    inputs["observed_pixels_px"][:, -1, :, 1] += 200.0
    second = build_projected_view_response(**inputs)

    np.testing.assert_array_equal(
        second.prior_reliability,
        first.prior_reliability,
    )
    np.testing.assert_array_equal(
        second.association_probability,
        first.association_probability,
    )


def test_metric_covariance_scales_quadratically_with_depth() -> None:
    first = build_projected_view_response(**_fixture())
    inputs = _fixture()
    inputs["initial_depth_m"] *= 2.0
    second = build_projected_view_response(**inputs)

    first_variance = first.observation_covariance_m2[0, 0, 0, 0, 0]
    second_variance = second.observation_covariance_m2[0, 0, 0, 0, 0]
    assert second_variance > 3.9 * first_variance
