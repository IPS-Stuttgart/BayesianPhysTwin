from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_predictive_coupling import score
from scripts.verify_deform_predictive_coupling import _frame, _metrics, _posterior


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_observation_space_audit_agrees_with_latent_precision_solve(count: int) -> None:
    rng = np.random.default_rng(260828)
    observation = rng.normal(size=(5, 3, 7)) * 0.02
    future = rng.normal(size=(4, 2, 3, 7)) * 0.03
    noise_factor = rng.normal(size=(5, 3, 3)) * 0.001
    noise = noise_factor @ noise_factor.transpose(0, 2, 1) + 1e-6 * np.eye(3)
    floor = np.broadcast_to(1e-4 * np.eye(3), (4, 2, 3, 3))
    selected = np.arange(count)
    innovation = rng.normal(size=(5, 3)) * 0.01
    correction, covariance = _posterior(
        observation, future, noise, floor, selected, innovation
    )
    precision = np.eye(7)
    information = np.zeros(7)
    for index in selected:
        precision += observation[index].T @ np.linalg.solve(
            noise[index], observation[index]
        )
        information += observation[index].T @ np.linalg.solve(
            noise[index], innovation[index]
        )
    expected_mean = future @ np.linalg.solve(precision, information)
    expected_covariance = (
        future @ np.linalg.inv(precision) @ future.swapaxes(-1, -2) + floor
    )
    np.testing.assert_allclose(correction, expected_mean, atol=1e-12)
    np.testing.assert_allclose(covariance, expected_covariance, atol=1e-12)


def test_cholesky_audit_matches_registered_marginal_metrics() -> None:
    rng = np.random.default_rng(260828)
    truth = rng.normal(size=(9, 4, 3)) * 0.01
    means = rng.normal(size=(6, 9, 4, 3)) * 0.01
    factor = rng.normal(size=(6, 9, 4, 3, 3)) * 0.01
    covariance = factor @ factor.swapaxes(-1, -2)
    batched = _metrics(means, covariance, truth, 1e-6)
    for index, (mean, cov) in enumerate(zip(means, covariance, strict=True)):
        registered = score(mean, cov, truth, 0.001)
        assert batched.keys() == registered.keys()
        for key, value in registered.items():
            assert value == pytest.approx(batched[key][index], rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("vertical", [False, True])
def test_action_frame_fallback_axes_are_orthonormal(vertical: bool) -> None:
    reference = np.zeros((2, 12, 3))
    reference[:, :, 2 if vertical else 0] = np.arange(12)
    frame = _frame(reference)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1)
