"""Horizon propagation for dynamic endpoint model averages."""

from __future__ import annotations

import numpy as np

from ._dynamic_endpoint_components import _component_matrices
from ._dynamic_endpoint_contract import (
    DynamicEndpointPosteriorV2,
    DynamicEndpointPredictionV2,
)
from ._dynamic_endpoint_filter import _admit_psd_2x2, _mixture_moments


def _compose_transition(
    later_transition: np.ndarray,
    later_process: np.ndarray,
    earlier_transition: np.ndarray,
    earlier_process: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transition = later_transition @ earlier_transition
    process = later_transition @ earlier_process @ later_transition.T + later_process
    return transition, process


def _transition_power(
    transition: np.ndarray,
    process: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    result_transition = np.eye(2)
    result_process = np.zeros((2, 2))
    base_transition = np.array(transition, copy=True)
    base_process = np.array(process, copy=True)
    remaining = int(horizon_steps)
    while remaining:
        if remaining & 1:
            result_transition, result_process = _compose_transition(
                base_transition,
                base_process,
                result_transition,
                result_process,
            )
        base_transition, base_process = _compose_transition(
            base_transition,
            base_process,
            base_transition,
            base_process,
        )
        remaining >>= 1
    return result_transition, result_process


def predict_dynamic_endpoint_model_average(
    posterior: DynamicEndpointPosteriorV2,
    *,
    horizon_steps: int,
) -> DynamicEndpointPredictionV2:
    """Propagate every component and recompute model-averaged moments."""

    if not isinstance(posterior, DynamicEndpointPosteriorV2):
        raise TypeError("posterior must be a DynamicEndpointPosteriorV2")
    if isinstance(horizon_steps, (bool, np.bool_)) or not isinstance(
        horizon_steps,
        (int, np.integer),
    ):
        raise ValueError("horizon_steps must be a nonnegative integer")
    horizon = int(horizon_steps)
    if horizon < 0:
        raise ValueError("horizon_steps must be a nonnegative integer")
    component_count = len(posterior.config.components)
    track_count = len(posterior.mean_m)
    component_mean = np.empty((component_count, track_count, 3))
    component_velocity = np.empty((component_count, track_count, 3))
    component_variance = np.empty((component_count, track_count))
    for index, component in enumerate(posterior.config.components):
        transition, process, _, _, _, _, _ = _component_matrices(component)
        propagated_transition, propagated_process = _transition_power(
            transition,
            process,
            horizon,
        )
        propagated_mean = np.einsum(
            "ab,nbc->nac",
            propagated_transition,
            posterior.component_state_mean[index],
        )
        propagated_covariance = np.matmul(
            np.matmul(
                propagated_transition[None, :, :],
                posterior.component_state_covariance[index],
            ),
            propagated_transition.T[None, :, :],
        )
        propagated_covariance += propagated_process[None, :, :]
        propagated_covariance = _admit_psd_2x2(
            propagated_covariance,
            name=f"component {index} forecast covariance",
        )
        component_mean[index] = propagated_mean[:, 0, :]
        component_velocity[index] = propagated_mean[:, 1, :]
        component_variance[index] = propagated_covariance[:, 0, 0]
    mean, covariance = _mixture_moments(
        posterior.component_weights,
        component_mean,
        component_variance,
    )
    return DynamicEndpointPredictionV2(
        mean_m=mean,
        covariance_m2=covariance,
        component_weights=posterior.component_weights,
        component_mean_m=component_mean,
        component_variance_m2=component_variance,
        component_velocity_mean_m_per_step=component_velocity,
        horizon_steps=horizon,
    )


__all__ = ["predict_dynamic_endpoint_model_average"]
