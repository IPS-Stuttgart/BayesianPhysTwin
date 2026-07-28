from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.propagated_state_belief import (
    PropagatedStateBeliefConfig,
    infer_propagated_state_belief,
)


def _one_step_outlier_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    innovation = np.zeros((1, 3, 3), dtype=np.float64)
    innovation[0, :, 0] = np.asarray([1.0, 1.0, 10.0])
    available = np.ones((1, 3), dtype=bool)
    response = np.zeros((1, 3, 3, 1), dtype=np.float64)
    response[0, :, 0, 0] = 1.0
    bias_basis = np.zeros((3, 0), dtype=np.float64)
    return innovation, available, response, bias_basis


def test_final_system_uses_returned_robust_weights() -> None:
    innovation, available, response, bias_basis = _one_step_outlier_problem()
    config = PropagatedStateBeliefConfig(
        observation_std_m=1.0,
        state_weight_prior_std=10.0,
        effective_samples_per_frame=3.0,
        effective_frame_count=1.0,
        maximum_iterations=1,
        reject_unidentifiable_state=False,
    )

    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        config=config,
    )

    assert result.accepted
    robust = result.robust_weights[0]
    expected_precision = 1.0 / config.state_weight_prior_std**2 + np.sum(robust)
    expected_right = float(robust @ innovation[0, :, 0])
    assert result.state_weights[0] == pytest.approx(
        expected_right / expected_precision,
        rel=1e-12,
        abs=1e-12,
    )
    assert result.posterior_covariance[0, 0] == pytest.approx(
        1.0 / expected_precision,
        rel=1e-12,
        abs=1e-12,
    )
    assert result.diagnostics["final_system_uses_returned_robust_weights"] is True
    assert result.diagnostics["posterior_solver"] == "cholesky"


def test_positive_definite_paths_do_not_use_numpy_inverse(monkeypatch) -> None:
    innovation, available, response, bias_basis = _one_step_outlier_problem()

    def fail_inverse(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("np.linalg.inv must not be used for SPD systems")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)
    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        state_prior_covariance=np.asarray([[4.0]]),
        config=PropagatedStateBeliefConfig(
            maximum_iterations=2,
            reject_unidentifiable_state=False,
        ),
    )

    assert result.accepted
    np.testing.assert_allclose(
        result.posterior_covariance,
        result.posterior_covariance.T,
        atol=0.0,
        rtol=0.0,
    )
