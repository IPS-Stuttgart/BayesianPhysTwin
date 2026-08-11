from __future__ import annotations

import numpy as np

from bayesian_phystwin.bias_aware_belief_v2 import (
    BiasAwareStateUpdateConfigV2,
    update_bias_aware_state_v2,
)


def _anchor_only_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    innovation = np.zeros((1, 1, 3), dtype=np.float64)
    available = np.zeros((1, 1), dtype=bool)
    anchor_innovation = np.asarray(
        [[0.02, 0.0, 0.0], [-0.01, 0.0, 0.0]],
        dtype=np.float64,
    )
    anchor_state_basis = np.asarray([[1.0], [2.0]], dtype=np.float64)
    return innovation, available, anchor_innovation, anchor_state_basis


def test_v2_fails_closed_when_irls_iteration_budget_is_exhausted() -> None:
    innovation, available, anchor_innovation, anchor_state_basis = (
        _anchor_only_problem()
    )

    result = update_bias_aware_state_v2(
        innovation,
        available,
        np.ones((1, 1), dtype=np.float64),
        np.zeros((1, 0), dtype=np.float64),
        anchor_innovation_m=anchor_innovation,
        anchor_state_basis=anchor_state_basis,
        anchor_variance_m2=np.full(2, 1e-6, dtype=np.float64),
        config=BiasAwareStateUpdateConfigV2(
            state_prior_std_m=0.1,
            maximum_iterations=1,
            convergence_tolerance=1e-12,
        ),
    )

    assert not result.accepted
    assert result.reason == "irls-fixed-point-not-converged"
    assert result.diagnostics["irls_fixed_point_converged"] is False
    assert result.diagnostics["iterations"] == 1
    assert result.diagnostics["irls_robust_weight_delta"] > 1e-12
    assert np.count_nonzero(result.state_coefficients_m) == 0
    assert np.count_nonzero(result.posterior_covariance_m2) == 0
    assert np.count_nonzero(result.anchor_robust_weights) == 0


def test_v2_still_accepts_after_reaching_the_irls_fixed_point() -> None:
    innovation, available, anchor_innovation, anchor_state_basis = (
        _anchor_only_problem()
    )

    result = update_bias_aware_state_v2(
        innovation,
        available,
        np.ones((1, 1), dtype=np.float64),
        np.zeros((1, 0), dtype=np.float64),
        anchor_innovation_m=anchor_innovation,
        anchor_state_basis=anchor_state_basis,
        anchor_variance_m2=np.full(2, 1e-6, dtype=np.float64),
        config=BiasAwareStateUpdateConfigV2(
            state_prior_std_m=0.1,
            maximum_iterations=8,
            convergence_tolerance=1e-12,
        ),
    )

    assert result.accepted
    assert result.reason == "accepted"
    assert result.diagnostics["irls_fixed_point_converged"] is True
    assert result.diagnostics["iterations"] == 5
    np.testing.assert_allclose(
        result.anchor_robust_weights,
        np.asarray([0.02, 1.0]),
        atol=1e-12,
        rtol=0.0,
    )
