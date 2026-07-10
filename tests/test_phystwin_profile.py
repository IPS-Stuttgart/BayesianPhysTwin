import numpy as np
import pytest

from bayesian_phystwin.phystwin_profile import (
    causal_model_discrepancy_variance,
    clustered_track_log_likelihood,
    grid_parameter_posterior,
    predictive_observation_calibration,
    truncate_profile_prediction_weights,
    weighted_trajectory_moments,
)
from bayesian_phystwin.phystwin_refit import build_phystwin_track_objective


def _objective(variant="hard"):
    visible = np.ones((3, 2), dtype=bool)
    valid = np.ones((3, 2), dtype=bool)
    cues = {"flow_inconsistency": np.zeros((2, 2))}
    return build_phystwin_track_objective(
        visible,
        valid,
        cues=cues,
        variant=variant,
    )


@pytest.mark.parametrize("variant", ["hard", "cue", "mixture"])
def test_clustered_likelihood_prefers_lower_residual(variant):
    observed = np.zeros((3, 2, 3))
    exact = np.zeros_like(observed)
    offset = np.full_like(observed, 0.01)
    objective = _objective(variant)

    exact_log_likelihood = clustered_track_log_likelihood(
        observed,
        exact,
        objective,
        start_frame=1,
        end_frame=3,
        variance=2.5e-5,
    )
    offset_log_likelihood = clustered_track_log_likelihood(
        observed,
        offset,
        objective,
        start_frame=1,
        end_frame=3,
        variance=2.5e-5,
    )

    assert exact_log_likelihood == pytest.approx(0.0)
    assert exact_log_likelihood > offset_log_likelihood


def test_clustered_likelihood_accepts_trajectory_truncated_at_end_frame():
    observed = np.zeros((4, 2, 3))
    trajectory = np.zeros_like(observed)
    visible = np.ones((4, 2), dtype=bool)
    objective = build_phystwin_track_objective(
        visible,
        visible,
        variant="hard",
    )

    full = clustered_track_log_likelihood(
        observed,
        trajectory,
        objective,
        start_frame=1,
        end_frame=3,
        variance=1e-4,
    )
    truncated = clustered_track_log_likelihood(
        observed,
        trajectory[:3],
        objective,
        start_frame=1,
        end_frame=3,
        variance=1e-4,
    )

    assert truncated == pytest.approx(full)


def test_grid_posterior_normalizes_and_summarizes_correlation():
    object_scales = np.array([-0.1, 0.0, 0.1])
    controller_scales = np.array([-0.2, 0.0, 0.2])
    object_grid, controller_grid = np.meshgrid(
        object_scales,
        controller_scales,
        indexing="ij",
    )
    log_likelihood = -100.0 * np.square(object_grid - 0.5 * controller_grid)

    posterior = grid_parameter_posterior(
        object_scales,
        controller_scales,
        log_likelihood,
        object_prior_std=0.2,
        controller_prior_std=0.4,
    )

    assert np.sum(posterior.weights) == pytest.approx(1.0)
    assert posterior.summary["correlation"] > 0.0
    assert posterior.summary["effective_grid_points"] > 1.0


def test_weighted_moments_and_predictive_calibration():
    trajectories = np.stack(
        [np.zeros((2, 1, 3)), np.full((2, 1, 3), 0.02)]
    )
    mean, variance = weighted_trajectory_moments(
        trajectories,
        np.array([0.75, 0.25]),
    )
    calibration = predictive_observation_calibration(
        np.zeros_like(mean),
        mean,
        variance,
        np.ones((2, 1), dtype=bool),
        observation_variance=2.5e-5,
    )

    np.testing.assert_allclose(mean, 0.005)
    np.testing.assert_allclose(variance, 7.5e-5)
    assert calibration["count"] == 2
    assert calibration["coordinate_coverage_90"] == 1.0


def test_profile_prediction_truncation_keeps_minimum_highest_mass_set():
    weights = np.array([[0.50, 0.30], [0.15, 0.05]])

    truncated, retained, count = truncate_profile_prediction_weights(
        weights,
        retained_mass=0.80,
    )

    assert retained == pytest.approx(0.80)
    assert count == 2
    assert np.sum(truncated) == pytest.approx(1.0)
    np.testing.assert_allclose(truncated, [[0.625, 0.375], [0.0, 0.0]])


@pytest.mark.parametrize("retained_mass", [0.0, 1.01])
def test_profile_prediction_truncation_rejects_invalid_mass(retained_mass):
    with pytest.raises(ValueError):
        truncate_profile_prediction_weights(
            np.ones((2, 2)),
            retained_mass=retained_mass,
        )


def test_causal_discrepancy_does_not_use_current_frame_residual():
    observed = np.zeros((4, 1, 3))
    mean = np.zeros_like(observed)
    mean[2, 0] = 0.02
    mask = np.ones((4, 1), dtype=bool)

    discrepancy = causal_model_discrepancy_variance(
        observed,
        mean,
        np.zeros_like(observed),
        mask,
        observation_variance=2.5e-5,
        decay=0.0,
    )

    assert discrepancy[2] == 0.0
    assert discrepancy[3] == pytest.approx(0.000375)


def test_predictive_calibration_accepts_framewise_discrepancy():
    observed = np.zeros((2, 1, 3))
    mean = np.full_like(observed, 0.01)
    frame_variance = np.array([0.0, 1e-3])

    calibration = predictive_observation_calibration(
        observed,
        mean,
        np.zeros_like(observed),
        np.ones((2, 1), dtype=bool),
        observation_variance=2.5e-5,
        model_discrepancy_variance=frame_variance,
    )

    assert calibration["count"] == 2
    assert calibration["coordinate_coverage_90"] == 0.5
