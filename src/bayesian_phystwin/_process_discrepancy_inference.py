"""Bayesian filtering and simulator application for process discrepancy."""

from __future__ import annotations

import numpy as np

from ._process_discrepancy_common import (
    expanded_coordinate_mask,
    expanded_coordinate_values,
    logdet_spd,
    solve_spd,
)
from ._process_discrepancy_model import (
    ProcessDiscrepancyModelV1,
    ProcessDiscrepancyStateV1,
    ProcessDiscrepancyUpdateV1,
    predict_process_discrepancy,
)


def _condition_gaussian(
    prior: ProcessDiscrepancyStateV1,
    design: np.ndarray,
    observation: np.ndarray,
    variance: np.ndarray,
) -> ProcessDiscrepancyStateV1:
    dimension = len(prior.mean_coefficients_n)
    identity = np.eye(dimension)
    prior_precision = solve_spd(prior.covariance_n2, identity)
    inverse_variance = 1.0 / variance
    posterior_precision = prior_precision + design.T @ (
        inverse_variance[:, None] * design
    )
    posterior_information = (
        prior_precision @ prior.mean_coefficients_n
        + design.T @ (inverse_variance * observation)
    )
    posterior_covariance = solve_spd(posterior_precision, identity)
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.T
    )
    posterior_mean = posterior_covariance @ posterior_information
    return ProcessDiscrepancyStateV1(
        mean_coefficients_n=posterior_mean,
        covariance_n2=posterior_covariance,
        step_index=prior.step_index,
    )


def _gaussian_information_gain_nats(
    prior: ProcessDiscrepancyStateV1,
    posterior: ProcessDiscrepancyStateV1,
) -> float:
    dimension = len(prior.mean_coefficients_n)
    prior_precision = solve_spd(prior.covariance_n2, np.eye(dimension))
    mean_delta = posterior.mean_coefficients_n - prior.mean_coefficients_n
    value = 0.5 * (
        float(np.trace(prior_precision @ posterior.covariance_n2))
        + float(mean_delta @ prior_precision @ mean_delta)
        - dimension
        + logdet_spd(prior.covariance_n2)
        - logdet_spd(posterior.covariance_n2)
    )
    return max(value, 0.0)


def update_process_discrepancy(
    model: ProcessDiscrepancyModelV1,
    prior: ProcessDiscrepancyStateV1,
    observed_force_n: np.ndarray,
    observed_variance_n2: float | np.ndarray,
    *,
    observed: np.ndarray | None = None,
    reliability: float | np.ndarray = 1.0,
    node_velocity_mps: np.ndarray | None = None,
) -> ProcessDiscrepancyUpdateV1:
    """Condition the latent force process on inverse-dynamics force evidence.

    Observation variances are divided by ``reliability`` so values in ``(0, 1]``
    downweight uncertain evidence. When configured, finite node velocities add
    zero-mean local-power pseudo observations ``v_i.T @ f_i = 0``.
    """

    basis = model.basis
    if len(prior.mean_coefficients_n) != basis.latent_dimension:
        raise ValueError("prior does not match process-discrepancy basis")
    force = np.asarray(observed_force_n, dtype=float)
    if force.shape != (basis.node_count, 3):
        raise ValueError("observed_force_n must have shape (node, 3)")
    variance = expanded_coordinate_values(
        observed_variance_n2,
        node_count=basis.node_count,
        name="observed_variance_n2",
    )
    reliability_values = expanded_coordinate_values(
        reliability,
        node_count=basis.node_count,
        name="reliability",
    )
    mask = expanded_coordinate_mask(observed, node_count=basis.node_count)
    if not np.all(np.isfinite(reliability_values)) or np.any(
        reliability_values < 0.0
    ) or np.any(reliability_values > 1.0):
        raise ValueError("reliability must contain finite values in [0, 1]")
    invalid_observed = mask & (
        ~np.isfinite(force) | ~np.isfinite(variance) | (variance <= 0.0)
    )
    if np.any(invalid_observed):
        raise ValueError(
            "observed force and variance must be finite with positive variance"
        )
    selected = mask & (reliability_values > 0.0)
    flattened_selected = selected.reshape(-1)
    data_design = basis.force_operator[flattened_selected]
    data_observation = force.reshape(-1)[flattened_selected]
    data_variance = variance.reshape(-1)[flattened_selected]
    data_reliability = reliability_values.reshape(-1)[flattened_selected]
    floor_variance = model.dynamics.observation_noise_floor_n**2
    effective_data_variance = np.maximum(data_variance, floor_variance) / (
        data_reliability
    )

    power_design_rows: list[np.ndarray] = []
    velocity = None
    if node_velocity_mps is not None:
        velocity = np.asarray(node_velocity_mps, dtype=float)
        if velocity.shape != (basis.node_count, 3):
            raise ValueError("node_velocity_mps must have shape (node, 3)")
        if not np.all(np.isfinite(velocity)):
            raise ValueError("node_velocity_mps must be finite")
    power_std = model.dynamics.local_power_prior_std_w
    if velocity is not None and power_std is not None:
        for node_index in range(basis.node_count):
            if basis.support_weights[node_index] <= 0.0:
                continue
            node_operator = basis.force_operator[
                3 * node_index : 3 * node_index + 3
            ]
            row = velocity[node_index] @ node_operator
            if np.linalg.norm(row) > 1e-14:
                power_design_rows.append(row)

    if power_design_rows:
        if power_std is None:
            raise RuntimeError("power prior rows require a configured power scale")
        power_design = np.stack(power_design_rows)
        power_observation = np.zeros(len(power_design_rows), dtype=float)
        power_variance = np.full(len(power_design_rows), power_std**2, dtype=float)
    else:
        power_design = np.empty((0, basis.latent_dimension), dtype=float)
        power_observation = np.empty(0, dtype=float)
        power_variance = np.empty(0, dtype=float)

    if len(data_observation) or len(power_observation):
        design = np.vstack((data_design, power_design))
        observation = np.concatenate((data_observation, power_observation))
        combined_variance = np.concatenate(
            (effective_data_variance, power_variance)
        )
        posterior = _condition_gaussian(
            prior,
            design,
            observation,
            combined_variance,
        )
    else:
        posterior = prior

    standardized_residual_rms = None
    if len(data_observation):
        residual = data_observation - data_design @ posterior.mean_coefficients_n
        standardized_residual_rms = float(
            np.sqrt(np.mean(np.square(residual) / effective_data_variance))
        )

    power_mean = None
    power_std_total = None
    if velocity is not None:
        total_power_row = velocity.reshape(-1) @ basis.force_operator
        power_mean = float(total_power_row @ posterior.mean_coefficients_n)
        power_variance_total = float(
            total_power_row @ posterior.covariance_n2 @ total_power_row
        )
        power_std_total = float(np.sqrt(max(power_variance_total, 0.0)))

    posterior_force = basis.force_from_coefficients(
        posterior.mean_coefficients_n
    )
    constraint_residual = basis.constraint_residual(posterior_force)
    return ProcessDiscrepancyUpdateV1(
        prior=prior,
        posterior=posterior,
        observed_coordinate_count=int(len(data_observation)),
        power_pseudo_observation_count=int(len(power_observation)),
        standardized_residual_rms=standardized_residual_rms,
        information_gain_nats=_gaussian_information_gain_nats(prior, posterior),
        total_mechanical_power_mean_w=power_mean,
        total_mechanical_power_std_w=power_std_total,
        constraint_residual_l2_n=float(np.linalg.norm(constraint_residual)),
    )


def process_discrepancy_step(
    model: ProcessDiscrepancyModelV1,
    state: ProcessDiscrepancyStateV1,
    observed_force_n: np.ndarray,
    observed_variance_n2: float | np.ndarray,
    *,
    observed: np.ndarray | None = None,
    reliability: float | np.ndarray = 1.0,
    node_velocity_mps: np.ndarray | None = None,
) -> ProcessDiscrepancyUpdateV1:
    """Run one AR(1) prediction followed by Bayesian force conditioning."""

    predicted = predict_process_discrepancy(model, state)
    return update_process_discrepancy(
        model,
        predicted,
        observed_force_n,
        observed_variance_n2,
        observed=observed,
        reliability=reliability,
        node_velocity_mps=node_velocity_mps,
    )


def apply_process_discrepancy_force(
    nominal_force_n: np.ndarray,
    discrepancy_force_n: np.ndarray,
    *,
    enabled: bool = True,
) -> np.ndarray:
    """Add a process force while preserving exact zero-correction parity."""

    nominal = np.asarray(nominal_force_n)
    discrepancy = np.asarray(discrepancy_force_n)
    if nominal.shape != discrepancy.shape:
        raise ValueError("nominal_force_n and discrepancy_force_n must match")
    if not np.issubdtype(nominal.dtype, np.floating):
        raise ValueError("nominal_force_n must use a floating dtype")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(discrepancy)):
        raise ValueError("force arrays must be finite")
    if not enabled or not np.any(discrepancy != 0):
        return nominal
    return nominal + discrepancy.astype(nominal.dtype, copy=False)
