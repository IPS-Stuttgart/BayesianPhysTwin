from __future__ import annotations

import numpy as np

from bayesian_phystwin.propagated_state_belief import (
    PropagatedStateBeliefConfig,
    infer_propagated_state_belief,
    propagated_state_readout,
)


def _problem(seed: int = 7):
    rng = np.random.default_rng(seed)
    frame_count = 7
    point_count = 18
    state_count = 4
    response = rng.normal(scale=0.002, size=(frame_count, point_count, 3, state_count))
    # Temporal variation makes the physical response identifiable from a
    # persistent spatial bias, even when they overlap at the first frame.
    response *= np.linspace(0.2, 1.4, frame_count)[:, None, None, None]
    bias_basis = np.column_stack(
        (np.ones(point_count), np.linspace(-1.0, 1.0, point_count))
    )
    state_weights = np.asarray((0.5, -0.25, 0.1, 0.35))
    bias_coefficients = np.asarray(((0.003, -0.002, 0.001), (0.001, 0.0, -0.001)))
    innovation = propagated_state_readout(
        response, state_weights, bias_basis, bias_coefficients
    )
    available = np.ones((frame_count, point_count), dtype=bool)
    return (
        innovation,
        available,
        response,
        bias_basis,
        state_weights,
        bias_coefficients,
    )


def test_recovers_action_propagated_state_separately_from_persistent_bias() -> None:
    innovation, available, response, bias_basis, state, bias = _problem()
    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.full(available.shape, 1e-10),
        config=PropagatedStateBeliefConfig(
            state_weight_prior_std=100.0,
            shared_bias_prior_std_m=100.0,
        ),
    )

    assert result.accepted
    np.testing.assert_allclose(result.state_weights, state, atol=2e-5)
    np.testing.assert_allclose(result.shared_bias_coefficients_m, bias, atol=2e-7)
    assert result.diagnostics["prior_reliability_uses_innovation"] is False
    assert result.diagnostics["innovation_likelihood_count"] == 1


def test_gross_outlier_is_robust_without_changing_prior_reliability() -> None:
    innovation, available, response, bias_basis, state, _ = _problem()
    corrupted = innovation.copy()
    corrupted[3, 4] += 0.10
    reliability = np.ones(available.shape)
    result = infer_propagated_state_belief(
        corrupted,
        available,
        response,
        bias_basis,
        prior_reliability=reliability,
        observation_variance_m2=np.full(available.shape, 25e-6),
        config=PropagatedStateBeliefConfig(
            state_weight_prior_std=100.0,
            shared_bias_prior_std_m=100.0,
        ),
    )

    assert result.accepted
    np.testing.assert_array_equal(result.prior_reliability, reliability)
    assert result.robust_weights[3, 4] == 0.02
    np.testing.assert_allclose(result.state_weights, state, atol=0.02)


def test_duplicate_dense_nodes_do_not_create_unbounded_precision() -> None:
    innovation, available, response, bias_basis, _, _ = _problem()
    config = PropagatedStateBeliefConfig(
        effective_samples_per_frame=8.0,
        effective_frame_count=3.0,
        state_weight_prior_std=2.0,
    )
    original = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        config=config,
    )
    copies = 20
    duplicated = infer_propagated_state_belief(
        np.repeat(innovation, copies, axis=1),
        np.repeat(available, copies, axis=1),
        np.repeat(response, copies, axis=1),
        np.repeat(bias_basis, copies, axis=0),
        config=config,
    )

    np.testing.assert_allclose(
        duplicated.state_weights, original.state_weights, atol=1e-10
    )
    np.testing.assert_allclose(
        duplicated.posterior_covariance,
        original.posterior_covariance,
        atol=1e-10,
    )


def test_static_state_response_confounded_with_bias_falls_back_exactly() -> None:
    frame_count = 5
    point_count = 12
    basis = np.linspace(-1.0, 1.0, point_count)[:, None]
    response = np.zeros((frame_count, point_count, 3, 1))
    response[:, :, 0, 0] = basis[:, 0]
    innovation = np.zeros((frame_count, point_count, 3))
    innovation[:, :, 0] = 0.01 * basis[:, 0]
    result = infer_propagated_state_belief(
        innovation,
        np.ones((frame_count, point_count), dtype=bool),
        response,
        basis,
    )

    assert not result.accepted
    assert result.reason == "state-response-confounded-with-persistent-bias"
    np.testing.assert_array_equal(result.state_weights, np.zeros(1))


def test_innovation_magnitude_does_not_change_prior_reliability() -> None:
    innovation, available, response, bias_basis, _, _ = _problem()
    first = infer_propagated_state_belief(innovation, available, response, bias_basis)
    second = infer_propagated_state_belief(
        100.0 * innovation, available, response, bias_basis
    )

    np.testing.assert_array_equal(first.prior_reliability, second.prior_reliability)


def test_propagated_irls_waits_for_robust_weight_fixed_point() -> None:
    innovation = np.zeros((1, 2, 3), dtype=np.float64)
    innovation[0, :, 0] = np.asarray([0.02, -0.01])
    response = np.zeros((1, 2, 3, 1), dtype=np.float64)
    response[0, :, 0, 0] = np.asarray([1.0, 2.0])

    result = infer_propagated_state_belief(
        innovation,
        np.ones((1, 2), dtype=bool),
        response,
        np.zeros((2, 0), dtype=np.float64),
        observation_variance_m2=np.full((1, 2), 1e-6, dtype=np.float64),
        config=PropagatedStateBeliefConfig(
            state_weight_prior_std=0.1,
            maximum_iterations=8,
            convergence_tolerance=1e-12,
        ),
    )

    assert result.accepted
    assert result.diagnostics["iterations"] >= 4
    np.testing.assert_allclose(
        result.state_weights,
        np.asarray([-0.004875500609437576]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.robust_weights[0],
        np.asarray([0.02, 1.0]),
        atol=1e-12,
        rtol=0.0,
    )
