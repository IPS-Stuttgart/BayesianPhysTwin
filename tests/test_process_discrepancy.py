from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.process_discrepancy import (
    ProcessDiscrepancyDynamicsV1,
    ProcessDiscrepancyFitBoundaryV1,
    ProcessDiscrepancyModelV1,
    apply_process_discrepancy_force,
    build_process_discrepancy_basis,
    initial_process_discrepancy_state,
    predict_process_discrepancy,
    process_discrepancy_force_moments,
    process_discrepancy_step,
    update_process_discrepancy,
)


def _graph_basis(node_count: int = 12, rank: int = 5) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, node_count)
    columns = (
        np.ones(node_count),
        grid,
        grid**2,
        np.sin(np.pi * grid),
        np.cos(np.pi * grid),
        np.sin(2.0 * np.pi * grid),
    )
    return np.linalg.qr(np.column_stack(columns[:rank]), mode="reduced")[0]


def _positions(node_count: int = 12) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)
    return np.column_stack(
        (
            0.2 * np.cos(angle),
            0.1 * np.sin(angle),
            np.linspace(-0.04, 0.04, node_count),
        )
    )


def _model(
    *,
    local_power_prior_std_w: float | None = None,
) -> ProcessDiscrepancyModelV1:
    node_count = 12
    basis = build_process_discrepancy_basis(
        _graph_basis(node_count),
        _positions(node_count),
        graph_eigenvalues=np.asarray((0.0, 0.2, 0.5, 1.0, 1.7)),
        support_weights=np.linspace(0.4, 1.0, node_count),
        externally_supported=np.asarray(
            (
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            )
        ),
    )
    return ProcessDiscrepancyModelV1(
        basis=basis,
        dynamics=ProcessDiscrepancyDynamicsV1(
            autoregressive_coefficient=0.92,
            stationary_coefficient_std_n=0.4,
            graph_roughness_strength=0.3,
            observation_noise_floor_n=1e-7,
            local_power_prior_std_w=local_power_prior_std_w,
        ),
        fit_boundary=ProcessDiscrepancyFitBoundaryV1(
            method_freeze_id="process-discrepancy-source-v1",
            split_id="object-source-development-target-v1",
            baseline_id="released-phystwin-v1",
            readout_comparator_id="readout-only-discrepancy-v1",
        ),
        metadata={"claim_status": "diagnostic-only"},
    )


def test_basis_enforces_internal_momentum_constraints() -> None:
    model = _model()
    rng = np.random.default_rng(20260727)
    coefficients = rng.normal(size=model.basis.latent_dimension)
    force = model.basis.force_from_coefficients(coefficients)

    np.testing.assert_allclose(
        model.basis.constraint_residual(force),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        model.basis.force_operator.T @ model.basis.force_operator,
        np.eye(model.basis.latent_dimension),
        atol=1e-10,
    )
    graph_coefficients = model.basis.graph_coefficients_from_latent(coefficients)
    reconstructed = (
        model.basis.support_weights[:, None]
        * model.basis.graph_basis
        @ graph_coefficients
    )
    np.testing.assert_allclose(reconstructed, force, atol=1e-10)


def test_contact_nodes_are_excluded_from_internal_balance() -> None:
    node_count = 8
    basis = build_process_discrepancy_basis(
        _graph_basis(node_count, rank=4),
        _positions(node_count),
        externally_supported=np.asarray(
            (True, False, False, False, False, False, False, False)
        ),
    )
    rng = np.random.default_rng(3)
    force = basis.force_from_coefficients(rng.normal(size=basis.latent_dimension))
    internal = basis.balanced_nodes

    np.testing.assert_allclose(np.sum(force[internal], axis=0), 0.0, atol=1e-10)
    center = np.mean(basis.node_positions_m[internal], axis=0)
    torque = np.sum(
        np.cross(basis.node_positions_m[internal] - center, force[internal]),
        axis=0,
    )
    np.testing.assert_allclose(torque, 0.0, atol=1e-10)


def test_stationary_covariance_is_invariant_under_prediction() -> None:
    model = _model()
    state = initial_process_discrepancy_state(model)
    predicted = predict_process_discrepancy(model, state)

    np.testing.assert_allclose(predicted.covariance_n2, state.covariance_n2)
    np.testing.assert_allclose(predicted.mean_coefficients_n, 0.0)
    assert predicted.step_index == 1


def test_force_observations_recover_latent_process_and_reduce_uncertainty() -> None:
    model = _model()
    prior = initial_process_discrepancy_state(model)
    rng = np.random.default_rng(12)
    true_coefficients = rng.normal(
        scale=0.12,
        size=model.basis.latent_dimension,
    )
    observed_force = model.basis.force_from_coefficients(true_coefficients)
    result = update_process_discrepancy(
        model,
        prior,
        observed_force,
        1e-8,
    )

    prior_error = np.linalg.norm(true_coefficients - prior.mean_coefficients_n)
    posterior_error = np.linalg.norm(
        true_coefficients - result.posterior.mean_coefficients_n
    )
    assert posterior_error < 1e-3 * prior_error
    assert np.trace(result.posterior.covariance_n2) < np.trace(prior.covariance_n2)
    assert result.observed_coordinate_count == 3 * model.basis.node_count
    assert result.information_gain_nats > 0.0
    assert result.constraint_residual_l2_n < 1e-10



def test_force_moments_expose_posterior_uncertainty() -> None:
    model = _model()
    state = initial_process_discrepancy_state(model)

    mean, standard_deviation = process_discrepancy_force_moments(model, state)

    np.testing.assert_allclose(mean, 0.0)
    assert mean.shape == (model.basis.node_count, 3)
    assert standard_deviation.shape == mean.shape
    assert np.all(standard_deviation >= 0.0)


def test_observed_invalid_values_fail_closed() -> None:
    model = _model()
    prior = initial_process_discrepancy_state(model)
    force = np.zeros((model.basis.node_count, 3))
    force[0, 0] = np.nan

    with pytest.raises(ValueError, match="observed force and variance"):
        update_process_discrepancy(model, prior, force, 0.1)
    with pytest.raises(ValueError, match="observed force and variance"):
        update_process_discrepancy(model, prior, np.zeros_like(force), -0.1)


def test_explicitly_unobserved_nan_is_ignored() -> None:
    model = _model()
    prior = initial_process_discrepancy_state(model)
    force = np.zeros((model.basis.node_count, 3))
    observed = np.ones_like(force, dtype=bool)
    force[0, 0] = np.nan
    observed[0, 0] = False

    result = update_process_discrepancy(
        model,
        prior,
        force,
        0.1,
        observed=observed,
    )

    assert result.observed_coordinate_count == 3 * model.basis.node_count - 1

def test_reliability_zero_ignores_coordinates() -> None:
    model = _model()
    prior = initial_process_discrepancy_state(model)
    observed_force = np.ones((model.basis.node_count, 3))
    result = update_process_discrepancy(
        model,
        prior,
        observed_force,
        0.01,
        reliability=0.0,
    )

    assert result.posterior is prior
    assert result.observed_coordinate_count == 0
    assert result.information_gain_nats == pytest.approx(0.0)


def test_local_power_prior_reduces_injected_mechanical_power() -> None:
    unregularized_model = _model(local_power_prior_std_w=None)
    regularized_model = _model(local_power_prior_std_w=1e-3)
    rng = np.random.default_rng(18)
    true_coefficients = rng.normal(
        scale=0.2,
        size=unregularized_model.basis.latent_dimension,
    )
    observed_force = unregularized_model.basis.force_from_coefficients(
        true_coefficients
    )
    velocity = observed_force.copy()

    unregularized = update_process_discrepancy(
        unregularized_model,
        initial_process_discrepancy_state(unregularized_model),
        observed_force,
        1e-3,
        node_velocity_mps=velocity,
    )
    regularized = update_process_discrepancy(
        regularized_model,
        initial_process_discrepancy_state(regularized_model),
        observed_force,
        1e-3,
        node_velocity_mps=velocity,
    )

    assert regularized.power_pseudo_observation_count > 0
    assert unregularized.total_mechanical_power_mean_w is not None
    assert regularized.total_mechanical_power_mean_w is not None
    assert abs(regularized.total_mechanical_power_mean_w) < abs(
        unregularized.total_mechanical_power_mean_w
    )


def test_process_step_predicts_then_updates() -> None:
    model = _model()
    state = initial_process_discrepancy_state(model)
    force = np.zeros((model.basis.node_count, 3))
    result = process_discrepancy_step(model, state, force, 0.1)

    assert result.prior.step_index == 1
    assert result.posterior.step_index == 1


def test_zero_force_path_preserves_exact_simulator_parity() -> None:
    nominal = np.arange(18, dtype=np.float32).reshape(6, 3)
    before = nominal.tobytes()
    zero = np.zeros_like(nominal, dtype=np.float64)

    result = apply_process_discrepancy_force(nominal, zero)

    assert result is nominal
    assert result.dtype == np.float32
    assert result.tobytes() == before
    disabled = apply_process_discrepancy_force(
        nominal,
        np.ones_like(nominal),
        enabled=False,
    )
    assert disabled is nominal


def test_nonzero_force_preserves_nominal_dtype() -> None:
    nominal = np.zeros((4, 3), dtype=np.float32)
    discrepancy = np.full((4, 3), 0.25, dtype=np.float64)
    result = apply_process_discrepancy_force(nominal, discrepancy)

    assert result.dtype == nominal.dtype
    np.testing.assert_allclose(result, 0.25)


def test_fit_boundary_rejects_outcome_leakage() -> None:
    boundary = _model().fit_boundary
    with pytest.raises(ValueError, match="future outcomes"):
        replace(boundary, future_outcomes_used_for_fit_or_selection=True)
    with pytest.raises(ValueError, match="target outcomes"):
        replace(boundary, target_outcomes_used_for_fit_or_selection=True)


def test_model_id_binds_dynamics_and_basis() -> None:
    model = _model()
    altered = replace(
        model,
        dynamics=replace(
            model.dynamics,
            autoregressive_coefficient=0.85,
        ),
    )

    assert len(model.model_id) == 64
    assert model.model_id != altered.model_id
