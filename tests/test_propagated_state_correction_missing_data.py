from __future__ import annotations

import numpy as np

from bayesian_phystwin.propagated_state_belief import PropagatedStateBeliefConfig
from bayesian_phystwin.propagated_state_correction import (
    PropagatedStateSelectionConfig,
    _weighted_rmse,
    modal_state_parameter_fields,
    select_propagated_state_update,
)


def _orthonormal_basis(point_count: int = 12, rank: int = 4) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, point_count)
    values = np.column_stack([coordinate**degree for degree in range(rank)])
    return np.linalg.qr(values)[0]


def _selection_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(4)
    frame_count = 7
    point_count = 12
    basis = _orthonormal_basis(point_count)
    position_fields, velocity_fields, position_steps, velocity_steps = (
        modal_state_parameter_fields(
            basis,
            position_step_m=0.005,
            velocity_step_mps=0.05,
        )
    )
    parameter_count = position_fields.shape[2] + velocity_fields.shape[2]
    response = rng.normal(
        scale=0.001,
        size=(frame_count, point_count, 3, parameter_count),
    )
    response *= np.linspace(0.0, 2.0, frame_count)[:, None, None, None]
    weights = np.zeros(parameter_count)
    weights[1] = 1.5
    persistent_coefficients = np.zeros((basis.shape[1], 3))
    persistent_coefficients[0, 0] = 0.003
    persistent_coefficients[1, 1] = 0.001
    innovation = (
        np.einsum("tnck,k->tnc", response, weights)
        + (basis @ persistent_coefficients)[None]
    )
    return (
        innovation,
        np.ones((frame_count, point_count), dtype=bool),
        response,
        basis,
        position_steps,
        velocity_steps,
    )


def _select(
    innovation: np.ndarray,
    available: np.ndarray,
    response: np.ndarray,
    basis: np.ndarray,
    position_steps: np.ndarray,
    velocity_steps: np.ndarray,
):
    return select_propagated_state_update(
        innovation,
        available,
        response,
        basis,
        basis,
        position_steps,
        velocity_steps,
        observation_variance_m2=np.full(available.shape, 1e-8),
        belief_config=PropagatedStateBeliefConfig(
            state_weight_prior_std=100.0,
            shared_bias_prior_std_m=100.0,
        ),
        selection_config=PropagatedStateSelectionConfig(
            minimum_validation_improvement_fraction=0.01,
        ),
    )


def test_weighted_rmse_ignores_unusable_nonfinite_residuals() -> None:
    residual = np.asarray(
        [[[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]],
        dtype=np.float64,
    )
    available = np.asarray([[True, False]], dtype=bool)
    reliability = np.ones((1, 2), dtype=np.float64)

    result = _weighted_rmse(residual, available, reliability)

    np.testing.assert_allclose(result, np.sqrt(14.0 / 3.0), atol=0.0, rtol=1e-15)


def test_prefix_guard_ignores_masked_nonfinite_validation_observation() -> None:
    innovation, available, response, basis, position_steps, velocity_steps = (
        _selection_problem()
    )
    available = available.copy()
    available[-1, 0] = False
    control_innovation = innovation.copy()
    masked_innovation = innovation.copy()
    masked_innovation[-1, 0] = np.nan

    control = _select(
        control_innovation,
        available,
        response,
        basis,
        position_steps,
        velocity_steps,
    )
    masked = _select(
        masked_innovation,
        available,
        response,
        basis,
        position_steps,
        velocity_steps,
    )

    assert masked.accepted == control.accepted
    assert masked.reason == control.reason
    np.testing.assert_allclose(masked.state_weights, control.state_weights)
    np.testing.assert_allclose(masked.position_update_m, control.position_update_m)
    np.testing.assert_allclose(masked.velocity_update_mps, control.velocity_update_mps)
    np.testing.assert_allclose(
        [
            masked.diagnostics["persistence_validation_rmse_m"],
            masked.diagnostics["joint_validation_rmse_m"],
        ],
        [
            control.diagnostics["persistence_validation_rmse_m"],
            control.diagnostics["joint_validation_rmse_m"],
        ],
    )
    assert np.isfinite(masked.diagnostics["validation_improvement_fraction"])
    assert np.isfinite(masked.diagnostics["validation_improvement_m"])
