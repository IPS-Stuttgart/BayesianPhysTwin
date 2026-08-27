from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.coupled_action_regret import (
    RegretCalibration,
    action_regret_upper,
    bias_marginalized_weights,
    calibrate_simultaneous_regret,
    guarded_action,
    selected_commands,
    weighted_quantile,
)


def test_shared_loss_uncertainty_cancels_only_in_paired_regret():
    baseline = np.array([1.0, 2.0, 3.0])
    losses = np.stack([baseline, baseline - 0.2], axis=1)
    weights = np.full(3, 1 / 3)
    joint = action_regret_upper(losses, weights)
    independent = action_regret_upper(losses, weights, coupling="independent")
    assert joint[1] == pytest.approx(-0.2)
    assert independent[1] == pytest.approx(1.8)
    calibration = RegretCalibration(0.9, 39, 36, 0.0)
    assert guarded_action(weights @ losses, joint, calibration) == 1
    assert guarded_action(weights @ losses, independent, calibration) == 0


def test_placebo_never_passes_strict_improvement_and_fallback_is_original_object():
    losses = np.tile(np.array([1.0, 2.0, 3.0])[:, None], (1, 4))
    means = np.mean(losses, axis=0)
    calibration = RegretCalibration(0.9, 39, 36, 0.0)
    for coupling in ("joint", "independent"):
        bound = action_regret_upper(losses, np.ones(3), coupling=coupling)
        assert guarded_action(means, bound, calibration) == 0
    actions = tuple(np.full((5, 2, 3), x, dtype=np.float32) for x in range(4))
    result = selected_commands(actions, 0)
    assert result is actions[0]
    assert result.dtype == np.float32


def test_simultaneous_calibration_uses_maximum_not_chosen_action():
    upper = np.zeros((9, 3))
    losses = np.zeros((9, 3))
    losses[:, 1] = -1.0
    losses[:, 2] = np.arange(9) / 10
    calibration = calibrate_simultaneous_regret(upper, losses)
    assert calibration.rank == 9
    assert calibration.offset == pytest.approx(0.8)


def test_calibration_order_statistic_and_no_negative_shrinkage():
    upper = np.zeros((39, 3))
    losses = np.tile(np.arange(39)[:, None], (1, 3)).astype(float)
    losses[:, 1:] = losses[:, :1] - 10.0
    calibration = calibrate_simultaneous_regret(upper, losses)
    assert calibration.rank == 36
    assert calibration.offset == 0


def test_too_little_calibration_forces_fallback():
    calibration = calibrate_simultaneous_regret(np.zeros((8, 2)), np.zeros((8, 2)))
    assert calibration.offset is None
    assert guarded_action(np.array([1.0, 0.0]), np.array([0.0, -1.0]), calibration) == 0


def test_correlated_sensor_likelihood_matches_dense_gaussian():
    rng = np.random.default_rng(13)
    predictions = rng.normal(0, 0.01, (5, 2, 3, 3))
    observed = rng.normal(0, 0.01, (2, 3, 3))
    prior = np.arange(1, 6, dtype=float)
    result = bias_marginalized_weights(
        observed,
        predictions,
        noise_std_m=0.003,
        shared_bias_std_m=0.012,
        prior_weights=prior,
    )
    covariance = 0.003**2 * np.eye(6) + 0.012**2 * np.ones((6, 6))
    residual = (observed[None] - predictions).reshape(5, 6, 3)
    log_weight = np.log(prior) - 0.5 * np.einsum(
        "kni,nm,kmi->k",
        residual,
        np.linalg.inv(covariance),
        residual,
    )
    expected = np.exp(log_weight - log_weight.max())
    expected /= expected.sum()
    np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-13)


def test_common_sensor_bias_not_counted_as_independent_point_errors():
    observed = np.full((2, 4, 3), 0.01)
    predictions = np.stack([np.zeros_like(observed), observed])
    joint = bias_marginalized_weights(
        observed,
        predictions,
        noise_std_m=0.003,
        shared_bias_std_m=0.02,
    )
    iid = bias_marginalized_weights(
        observed,
        predictions,
        noise_std_m=0.003,
        shared_bias_std_m=0.0,
    )
    assert joint[0] > 0.3
    assert iid[0] < 1e-6


@pytest.mark.parametrize("kind", ["missing", "baseline", "shape"])
def test_invalid_calibration_is_not_dropped_or_imputed(kind):
    upper, losses = np.zeros((39, 3)), np.zeros((39, 3))
    if kind == "missing":
        losses[3, 1] = np.nan
    elif kind == "baseline":
        upper[0, 0] = 1.0
    else:
        losses = losses[:2]
    with pytest.raises(ValueError):
        calibrate_simultaneous_regret(upper, losses)


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 2.0], [1.0, np.nan]])
def test_invalid_particle_weights_rejected(weights):
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array(weights), 0.9)


def test_invalid_quantile_and_coupling_rejected():
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0]), np.array([1.0]), 0)
    with pytest.raises(ValueError):
        action_regret_upper(np.zeros((2, 3)), np.ones(2), coupling="oracle")


@pytest.mark.parametrize(
    "arguments",
    [
        (0.9, -1, 1, None),
        (0.9, 39, 35, 0.0),
        (0.9, 8, 9, 0.0),
        (0.9, 39, 36, None),
        (0.9, 39, 36, -0.1),
        (1.1, 39, 36, 0.0),
    ],
)
def test_calibration_metadata_cannot_bypass_fallback(arguments):
    with pytest.raises(ValueError):
        RegretCalibration(*arguments)
