"""Causal-prefix filtering and model averaging for dynamic endpoints."""

from __future__ import annotations

import numpy as np

from ._dynamic_endpoint_components import (
    DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2,
    DynamicEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    DynamicEndpointNumericalError,
    _component_matrices,
)
from ._dynamic_endpoint_contract import DynamicEndpointPosteriorV2


def _validated_inputs(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    residual = np.asarray(residual_m, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if residual.ndim != 3 or residual.shape[2:] != (3,) or residual.shape[1] < 1:
        raise ValueError("residual_m must have shape (T, N>=1, 3)")
    if validity.shape != residual.shape[:2]:
        raise ValueError("valid must match the residual frame and track dimensions")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_m must contain only finite values")
    frame_stop = int(end_frame)
    if isinstance(end_frame, (bool, np.bool_)) or frame_stop != end_frame:
        raise ValueError("end_frame must be an integer")
    if not 0 < frame_stop <= len(residual):
        raise ValueError("end_frame must lie inside the residual sequence")
    return residual, validity, frame_stop


def _admit_psd_2x2(value: np.ndarray, *, name: str) -> np.ndarray:
    symmetric = 0.5 * (value + value.swapaxes(-1, -2))
    if not np.all(np.isfinite(symmetric)):
        raise DynamicEndpointNumericalError(f"{name} is non-finite")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.min(eigenvalues, initial=0.0) < -1e-12:
        raise DynamicEndpointNumericalError(f"{name} is not positive semidefinite")
    return symmetric


def _filter_component(
    residual: np.ndarray,
    validity: np.ndarray,
    *,
    end_frame: int,
    component: DynamicEndpointComponentV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        transition,
        process,
        initial,
        observation_variance,
        inlier_prior,
        outlier_multiplier,
        exact_persistence,
    ) = _component_matrices(component)
    track_count = residual.shape[1]
    state_mean = np.zeros((track_count, 2, 3), dtype=np.float64)
    state_covariance = np.repeat(initial[None, :, :], track_count, axis=0)
    final_probability = np.zeros(track_count, dtype=np.float64)
    update_count = np.zeros(track_count, dtype=np.int64)
    log_evidence = np.zeros(track_count, dtype=np.float64)
    log_prior = np.log(inlier_prior)
    log_outlier_prior = np.log1p(-inlier_prior)
    outlier_observation_variance = observation_variance * outlier_multiplier

    for frame in range(end_frame):
        state_mean = np.einsum("ab,nbc->nac", transition, state_mean)
        state_covariance = np.matmul(
            np.matmul(transition[None, :, :], state_covariance),
            transition.T[None, :, :],
        )
        state_covariance += process[None, :, :]
        mask = validity[frame]
        if not np.any(mask):
            continue
        predicted_mean = state_mean[mask]
        predicted_covariance = state_covariance[mask]
        innovation = residual[frame, mask] - predicted_mean[:, 0, :]
        inlier_innovation_variance = (
            predicted_covariance[:, 0, 0] + observation_variance
        )
        outlier_innovation_variance = (
            predicted_covariance[:, 0, 0] + outlier_observation_variance
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            squared_norm = np.sum(np.square(innovation), axis=1)
            log_inlier = log_prior - 0.5 * (
                3.0 * np.log(2.0 * np.pi * inlier_innovation_variance)
                + squared_norm / inlier_innovation_variance
            )
            log_outlier = log_outlier_prior - 0.5 * (
                3.0 * np.log(2.0 * np.pi * outlier_innovation_variance)
                + squared_norm / outlier_innovation_variance
            )
            log_mixture = np.logaddexp(log_inlier, log_outlier)
            probability = np.exp(log_inlier - log_mixture)
        if not all(np.all(np.isfinite(value)) for value in (log_mixture, probability)):
            raise ValueError("dynamic endpoint filtering produced non-finite numerics")
        log_evidence[mask] += log_mixture

        if exact_persistence:
            updated_mean = predicted_mean.copy()
            updated_mean[:, 0, :] = residual[frame, mask]
            updated_mean[:, 1, :] = 0.0
            effective_variance = (
                probability * observation_variance
                + (1.0 - probability) * outlier_observation_variance
            )
            updated_covariance = np.zeros_like(predicted_covariance)
            updated_covariance[:, 0, 0] = effective_variance
        else:
            inlier_gain = (
                predicted_covariance[:, :, 0] / inlier_innovation_variance[:, None]
            )
            outlier_gain = (
                predicted_covariance[:, :, 0] / outlier_innovation_variance[:, None]
            )
            inlier_mean = (
                predicted_mean + inlier_gain[:, :, None] * innovation[:, None, :]
            )
            outlier_mean = (
                predicted_mean + outlier_gain[:, :, None] * innovation[:, None, :]
            )
            updated_mean = (
                probability[:, None, None] * inlier_mean
                + (1.0 - probability)[:, None, None] * outlier_mean
            )
            identity_2 = np.eye(2, dtype=np.float64)
            observation_row = np.asarray([1.0, 0.0], dtype=np.float64)
            inlier_residual_transition = identity_2[None, :, :] - (
                inlier_gain[:, :, None] * observation_row[None, None, :]
            )
            outlier_residual_transition = identity_2[None, :, :] - (
                outlier_gain[:, :, None] * observation_row[None, None, :]
            )
            inlier_covariance = np.matmul(
                np.matmul(
                    inlier_residual_transition,
                    predicted_covariance,
                ),
                inlier_residual_transition.swapaxes(1, 2),
            ) + observation_variance * (
                inlier_gain[:, :, None] * inlier_gain[:, None, :]
            )
            outlier_covariance = np.matmul(
                np.matmul(
                    outlier_residual_transition,
                    predicted_covariance,
                ),
                outlier_residual_transition.swapaxes(1, 2),
            ) + outlier_observation_variance * (
                outlier_gain[:, :, None] * outlier_gain[:, None, :]
            )
            inlier_delta = inlier_mean - updated_mean
            outlier_delta = outlier_mean - updated_mean
            inlier_spread = (
                np.einsum(
                    "nai,nbi->nab",
                    inlier_delta,
                    inlier_delta,
                )
                / 3.0
            )
            outlier_spread = (
                np.einsum(
                    "nai,nbi->nab",
                    outlier_delta,
                    outlier_delta,
                )
                / 3.0
            )
            updated_covariance = probability[:, None, None] * (
                inlier_covariance + inlier_spread
            ) + (1.0 - probability)[:, None, None] * (
                outlier_covariance + outlier_spread
            )
            updated_covariance = _admit_psd_2x2(
                updated_covariance,
                name=f"frame {frame} component posterior covariance",
            )
        state_mean[mask] = updated_mean
        state_covariance[mask] = updated_covariance
        final_probability[mask] = probability
        update_count[mask] += 1
    return (
        state_mean,
        state_covariance,
        final_probability,
        update_count,
        log_evidence,
    )


def _normalized_component_weights(
    evidence: np.ndarray,
    update_count: np.ndarray,
    config: DynamicEndpointModelAverageConfigV2,
) -> np.ndarray:
    log_prior = np.log(np.asarray(config.component_prior_probability, dtype=np.float64))
    if config.evidence_pooling == "per_track":
        unnormalized = evidence + log_prior[None, :]
    else:
        updated = update_count > 0
        pooled = (
            np.sum(evidence[updated], axis=0)
            if np.any(updated)
            else np.zeros_like(log_prior)
        )
        unnormalized = np.repeat((pooled + log_prior)[None, :], len(evidence), axis=0)
    normalizer = np.logaddexp.reduce(unnormalized, axis=1)
    weights = np.exp(unnormalized - normalizer[:, None])
    if not np.all(np.isfinite(weights)):
        raise ValueError("component evidence normalization produced non-finite weights")
    return weights


def _mixture_moments(
    weights: np.ndarray,
    component_mean: np.ndarray,
    component_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.einsum("nk,knc->nc", weights, component_mean)
    centered = component_mean - mean[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    within = component_variance[:, :, None, None] * np.eye(3)
    covariance = np.einsum("nk,knij->nij", weights, within + outer)
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return mean, covariance


def infer_dynamic_endpoint_model_average(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: DynamicEndpointModelAverageConfigV2 | None = None,
) -> DynamicEndpointPosteriorV2:
    """Infer a robust endpoint across constant, persistent, and trend models."""

    residual, validity, frame_stop = _validated_inputs(
        residual_m,
        valid,
        end_frame=end_frame,
    )
    settings = (
        DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2 if config is None else config
    )
    if not isinstance(settings, DynamicEndpointModelAverageConfigV2):
        raise TypeError("config must be a DynamicEndpointModelAverageConfigV2")
    component_count = len(settings.components)
    track_count = residual.shape[1]
    state_mean = np.empty((component_count, track_count, 2, 3))
    state_covariance = np.empty((component_count, track_count, 2, 2))
    component_probability = np.empty((component_count, track_count))
    evidence = np.empty((track_count, component_count))
    common_update_count: np.ndarray | None = None
    for index, component in enumerate(settings.components):
        (
            state_mean[index],
            state_covariance[index],
            component_probability[index],
            update_count,
            evidence[:, index],
        ) = _filter_component(
            residual,
            validity,
            end_frame=frame_stop,
            component=component,
        )
        if common_update_count is None:
            common_update_count = update_count
        elif not np.array_equal(common_update_count, update_count):
            raise AssertionError(
                "dynamic endpoint components used different observations"
            )
    assert common_update_count is not None
    weights = _normalized_component_weights(
        evidence,
        common_update_count,
        settings,
    )
    component_mean = state_mean[:, :, 0, :]
    component_variance = state_covariance[:, :, 0, 0]
    mean, covariance = _mixture_moments(
        weights,
        component_mean,
        component_variance,
    )
    final_probability = np.einsum(
        "nk,kn->n",
        weights,
        component_probability,
    )
    return DynamicEndpointPosteriorV2(
        mean_m=mean,
        covariance_m2=covariance,
        final_nominal_probability=final_probability,
        update_count=common_update_count,
        component_weights=weights,
        component_log_evidence=evidence,
        component_state_mean=state_mean,
        component_state_covariance=state_covariance,
        config=settings,
        end_frame=frame_stop,
    )


__all__ = ["infer_dynamic_endpoint_model_average"]
