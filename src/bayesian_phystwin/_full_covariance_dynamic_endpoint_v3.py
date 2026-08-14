"""Full-covariance filtering and prediction for dynamic endpoint mixtures.

This prospective v3 path reuses the source-frozen v2 component family while
retaining complete three-axis observation and state covariance. The historical
scalar-variance v2 implementation remains unchanged.
"""

from __future__ import annotations

import numpy as np

from ._dynamic_endpoint_components import (
    DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2,
    DynamicEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    DynamicEndpointNumericalError,
    _expanded_component_matrices,
)
from ._dynamic_endpoint_contract import (
    FullCovarianceDynamicEndpointPosteriorV3,
    FullCovarianceDynamicEndpointPredictionV3,
)

_SYMMETRY_TOLERANCE = 1e-12
_OBSERVATION_MATRIX = np.concatenate(
    (np.eye(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64)),
    axis=1,
)
_IDENTITY_3 = np.eye(3, dtype=np.float64)
_IDENTITY_6 = np.eye(6, dtype=np.float64)


def _numeric_array(value: object, *, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(raw, dtype=np.float64)


def _admit_psd(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim < 2 or raw.shape[-1] != raw.shape[-2]:
        raise DynamicEndpointNumericalError(f"{name} has an invalid matrix shape")
    if not np.all(np.isfinite(raw)):
        raise DynamicEndpointNumericalError(f"{name} is non-finite")
    scale = max(1.0, float(np.max(np.abs(raw), initial=0.0)))
    asymmetry = float(np.max(np.abs(raw - raw.swapaxes(-1, -2)), initial=0.0))
    if asymmetry > _SYMMETRY_TOLERANCE * scale:
        raise DynamicEndpointNumericalError(f"{name} is not symmetric")
    symmetric = 0.5 * (raw + raw.swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(np.min(eigenvalues, initial=0.0)) < -_SYMMETRY_TOLERANCE * scale:
        raise DynamicEndpointNumericalError(f"{name} is not positive semidefinite")
    return symmetric


def _cholesky(value: np.ndarray, *, name: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(covariance)):
        raise DynamicEndpointNumericalError(f"{name} is non-finite")
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise DynamicEndpointNumericalError(
            f"{name} is not positive definite"
        ) from error
    if not np.all(np.isfinite(factor)):
        raise DynamicEndpointNumericalError(
            f"{name} produced a non-finite Cholesky factor"
        )
    return factor


def _solve_spd(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    factor = _cholesky(matrix, name=name)
    try:
        intermediate = np.linalg.solve(factor, right_hand_side)
        result = np.linalg.solve(factor.T, intermediate)
    except np.linalg.LinAlgError as error:
        raise DynamicEndpointNumericalError(f"{name} solve failed") from error
    if not np.all(np.isfinite(result)):
        raise DynamicEndpointNumericalError(f"{name} solve produced non-finite values")
    return result


def _log_gaussian_density(
    innovation: np.ndarray,
    covariance: np.ndarray,
    *,
    name: str,
) -> float:
    factor = _cholesky(covariance, name=name)
    try:
        whitened = np.linalg.solve(factor, innovation)
    except np.linalg.LinAlgError as error:
        raise DynamicEndpointNumericalError(f"{name} solve failed") from error
    diagonal = np.diag(factor)
    log_determinant = 2.0 * float(np.sum(np.log(diagonal)))
    mahalanobis = float(whitened @ whitened)
    value = -0.5 * (3.0 * np.log(2.0 * np.pi) + log_determinant + mahalanobis)
    if not np.isfinite(value):
        raise DynamicEndpointNumericalError(f"{name} log density is non-finite")
    return value


def _validated_inputs(
    residual_m: object,
    valid: object,
    observation_covariance_m2: object,
    *,
    end_frame: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    residual = _numeric_array(residual_m, name="residual_m")
    raw_validity = np.asarray(valid)
    covariance = _numeric_array(
        observation_covariance_m2,
        name="observation_covariance_m2",
    )
    if residual.ndim != 3 or residual.shape[2:] != (3,) or residual.shape[1] < 1:
        raise ValueError("residual_m must have shape (T, N>=1, 3)")
    if raw_validity.shape != residual.shape[:2]:
        raise ValueError("valid must match the residual frame and track dimensions")
    if raw_validity.dtype != np.dtype(np.bool_):
        raise ValueError("valid must contain only booleans")
    if covariance.shape != (*residual.shape[:2], 3, 3):
        raise ValueError("observation_covariance_m2 must have shape (T, N, 3, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_m must contain only finite values")
    covariance = _admit_psd(
        covariance,
        name="observation_covariance_m2",
    )
    if isinstance(end_frame, (bool, np.bool_)) or not isinstance(
        end_frame,
        (int, np.integer),
    ):
        raise ValueError("end_frame must be an integer")
    frame_stop = int(end_frame)
    if not 0 < frame_stop <= len(residual):
        raise ValueError("end_frame must lie inside the residual sequence")
    return residual, raw_validity, covariance, frame_stop


def _kalman_branch(
    predicted_mean: np.ndarray,
    predicted_covariance: np.ndarray,
    innovation: np.ndarray,
    observation_covariance: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    innovation_covariance = (
        _OBSERVATION_MATRIX @ predicted_covariance @ _OBSERVATION_MATRIX.T
        + observation_covariance
    )
    log_density = _log_gaussian_density(
        innovation,
        innovation_covariance,
        name=f"{name} innovation covariance",
    )
    cross_covariance = predicted_covariance @ _OBSERVATION_MATRIX.T
    gain = _solve_spd(
        innovation_covariance,
        cross_covariance.T,
        name=f"{name} innovation covariance",
    ).T
    updated_mean = predicted_mean + gain @ innovation
    residual_transition = _IDENTITY_6 - gain @ _OBSERVATION_MATRIX
    updated_covariance = (
        residual_transition @ predicted_covariance @ residual_transition.T
        + gain @ observation_covariance @ gain.T
    )
    return (
        updated_mean,
        _admit_psd(updated_covariance, name=f"{name} posterior covariance"),
        log_density,
    )


def _robust_update(
    predicted_mean: np.ndarray,
    predicted_covariance: np.ndarray,
    observation: np.ndarray,
    metric_covariance: np.ndarray,
    *,
    observation_variance: float,
    inlier_prior: float,
    outlier_multiplier: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    innovation = observation - predicted_mean[:3]
    inlier_covariance = metric_covariance + observation_variance * _IDENTITY_3
    outlier_covariance = (
        metric_covariance + observation_variance * outlier_multiplier * _IDENTITY_3
    )
    inlier_mean, inlier_state_covariance, inlier_density = _kalman_branch(
        predicted_mean,
        predicted_covariance,
        innovation,
        inlier_covariance,
        name=f"{name} inlier",
    )
    outlier_mean, outlier_state_covariance, outlier_density = _kalman_branch(
        predicted_mean,
        predicted_covariance,
        innovation,
        outlier_covariance,
        name=f"{name} outlier",
    )
    log_inlier = np.log(inlier_prior) + inlier_density
    log_outlier = np.log1p(-inlier_prior) + outlier_density
    log_mixture = float(np.logaddexp(log_inlier, log_outlier))
    nominal_probability = float(np.exp(log_inlier - log_mixture))
    if not np.isfinite(log_mixture) or not np.isfinite(nominal_probability):
        raise DynamicEndpointNumericalError(f"{name} mixture probability is non-finite")
    updated_mean = (
        nominal_probability * inlier_mean + (1.0 - nominal_probability) * outlier_mean
    )
    inlier_delta = inlier_mean - updated_mean
    outlier_delta = outlier_mean - updated_mean
    updated_covariance = nominal_probability * (
        inlier_state_covariance + np.outer(inlier_delta, inlier_delta)
    ) + (1.0 - nominal_probability) * (
        outlier_state_covariance + np.outer(outlier_delta, outlier_delta)
    )
    return (
        updated_mean,
        _admit_psd(updated_covariance, name=f"{name} mixture covariance"),
        nominal_probability,
        log_mixture,
    )


def _persistence_update(
    predicted_mean: np.ndarray,
    predicted_covariance: np.ndarray,
    observation: np.ndarray,
    metric_covariance: np.ndarray,
    *,
    observation_variance: float,
    inlier_prior: float,
    outlier_multiplier: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    innovation = observation - predicted_mean[:3]
    inlier_covariance = metric_covariance + observation_variance * _IDENTITY_3
    outlier_covariance = (
        metric_covariance + observation_variance * outlier_multiplier * _IDENTITY_3
    )
    predicted_level_covariance = predicted_covariance[:3, :3]
    log_inlier = np.log(inlier_prior) + _log_gaussian_density(
        innovation,
        predicted_level_covariance + inlier_covariance,
        name=f"{name} inlier innovation covariance",
    )
    log_outlier = np.log1p(-inlier_prior) + _log_gaussian_density(
        innovation,
        predicted_level_covariance + outlier_covariance,
        name=f"{name} outlier innovation covariance",
    )
    log_mixture = float(np.logaddexp(log_inlier, log_outlier))
    nominal_probability = float(np.exp(log_inlier - log_mixture))
    if not np.isfinite(log_mixture) or not np.isfinite(nominal_probability):
        raise DynamicEndpointNumericalError(f"{name} mixture probability is non-finite")
    updated_mean: np.ndarray = np.zeros(6, dtype=np.float64)
    updated_mean[:3] = observation
    updated_covariance: np.ndarray = np.zeros((6, 6), dtype=np.float64)
    updated_covariance[:3, :3] = (
        nominal_probability * inlier_covariance
        + (1.0 - nominal_probability) * outlier_covariance
    )
    return (
        updated_mean,
        _admit_psd(updated_covariance, name=f"{name} posterior covariance"),
        nominal_probability,
        log_mixture,
    )


def _filter_component(
    residual: np.ndarray,
    validity: np.ndarray,
    metric_observation_covariance: np.ndarray,
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
    ) = _expanded_component_matrices(component)
    track_count = residual.shape[1]
    state_mean = np.zeros((track_count, 6), dtype=np.float64)
    state_covariance = np.repeat(initial[None, :, :], track_count, axis=0)
    final_probability = np.zeros(track_count, dtype=np.float64)
    update_count = np.zeros(track_count, dtype=np.int64)
    log_evidence = np.zeros(track_count, dtype=np.float64)

    for frame in range(end_frame):
        state_mean = np.einsum("ab,nb->na", transition, state_mean)
        state_covariance = np.einsum(
            "ab,nbc,dc->nad",
            transition,
            state_covariance,
            transition,
        )
        state_covariance += process[None, :, :]
        state_covariance = _admit_psd(
            state_covariance,
            name=f"frame {frame} predicted state covariance",
        )
        for track in np.flatnonzero(validity[frame]):
            name = f"frame {frame} track {track}"
            if exact_persistence:
                values = _persistence_update(
                    state_mean[track],
                    state_covariance[track],
                    residual[frame, track],
                    metric_observation_covariance[frame, track],
                    observation_variance=observation_variance,
                    inlier_prior=inlier_prior,
                    outlier_multiplier=outlier_multiplier,
                    name=name,
                )
            else:
                values = _robust_update(
                    state_mean[track],
                    state_covariance[track],
                    residual[frame, track],
                    metric_observation_covariance[frame, track],
                    observation_variance=observation_variance,
                    inlier_prior=inlier_prior,
                    outlier_multiplier=outlier_multiplier,
                    name=name,
                )
            (
                state_mean[track],
                state_covariance[track],
                final_probability[track],
                log_mixture,
            ) = values
            log_evidence[track] += log_mixture
            update_count[track] += 1
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
        raise DynamicEndpointNumericalError(
            "component evidence normalization produced non-finite weights"
        )
    return weights


def _mixture_moments(
    weights: np.ndarray,
    component_mean: np.ndarray,
    component_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.einsum("nk,knc->nc", weights, component_mean)
    centered = component_mean - mean[None, :, :]
    covariance = np.einsum(
        "nk,knij->nij",
        weights,
        component_covariance + centered[:, :, :, None] * centered[:, :, None, :],
    )
    return mean, _admit_psd(covariance, name="model-average endpoint covariance")


def infer_full_covariance_dynamic_endpoint_model_average(
    residual_m: object,
    valid: object,
    observation_covariance_m2: object,
    *,
    end_frame: object,
    config: DynamicEndpointModelAverageConfigV2 | None = None,
) -> FullCovarianceDynamicEndpointPosteriorV3:
    """Infer a causal endpoint using complete metric observation covariance."""

    residual, validity, metric_covariance, frame_stop = _validated_inputs(
        residual_m,
        valid,
        observation_covariance_m2,
        end_frame=end_frame,
    )
    settings = (
        DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2 if config is None else config
    )
    if not isinstance(settings, DynamicEndpointModelAverageConfigV2):
        raise TypeError("config must be a DynamicEndpointModelAverageConfigV2")
    component_count = len(settings.components)
    track_count = residual.shape[1]
    state_mean = np.empty((component_count, track_count, 6), dtype=np.float64)
    state_covariance = np.empty(
        (component_count, track_count, 6, 6),
        dtype=np.float64,
    )
    component_probability = np.empty((component_count, track_count), dtype=np.float64)
    evidence = np.empty((track_count, component_count), dtype=np.float64)
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
            metric_covariance,
            end_frame=frame_stop,
            component=component,
        )
        if common_update_count is None:
            common_update_count = update_count
        elif not np.array_equal(common_update_count, update_count):
            raise AssertionError(
                "full-covariance endpoint components used different observations"
            )
    assert common_update_count is not None
    weights = _normalized_component_weights(
        evidence,
        common_update_count,
        settings,
    )
    component_mean = state_mean[:, :, :3]
    component_covariance = state_covariance[:, :, :3, :3]
    mean, covariance = _mixture_moments(
        weights,
        component_mean,
        component_covariance,
    )
    final_probability = np.einsum(
        "nk,kn->n",
        weights,
        component_probability,
    )
    return FullCovarianceDynamicEndpointPosteriorV3(
        mean_m=mean,
        covariance_m2=covariance,
        final_nominal_probability=final_probability,
        update_count=common_update_count,
        component_weights=weights,
        component_log_evidence=evidence,
        component_state_mean=state_mean.reshape(component_count, track_count, 2, 3),
        component_state_covariance_m2=state_covariance,
        config=settings,
        end_frame=frame_stop,
    )


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
    result_transition = np.eye(6, dtype=np.float64)
    result_process: np.ndarray = np.zeros((6, 6), dtype=np.float64)
    base_transition = np.array(transition, dtype=np.float64, copy=True)
    base_process = np.array(process, dtype=np.float64, copy=True)
    remaining = horizon_steps
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
    return result_transition, _admit_psd(
        result_process,
        name="accumulated process covariance",
    )


def predict_full_covariance_dynamic_endpoint_model_average(
    posterior: FullCovarianceDynamicEndpointPosteriorV3,
    *,
    horizon_steps: object,
) -> FullCovarianceDynamicEndpointPredictionV3:
    """Propagate every full-covariance component and recompute mixture moments."""

    if not isinstance(posterior, FullCovarianceDynamicEndpointPosteriorV3):
        raise TypeError("posterior must be a FullCovarianceDynamicEndpointPosteriorV3")
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
    component_mean: np.ndarray = np.empty(
        (component_count, track_count, 3), dtype=np.float64
    )
    component_velocity: np.ndarray = np.empty(
        (component_count, track_count, 3),
        dtype=np.float64,
    )
    component_covariance: np.ndarray = np.empty(
        (component_count, track_count, 3, 3),
        dtype=np.float64,
    )
    state_mean = posterior.component_state_mean.reshape(
        component_count,
        track_count,
        6,
    )
    for index, component in enumerate(posterior.config.components):
        transition, process, _, _, _, _, _ = _expanded_component_matrices(component)
        propagated_transition, propagated_process = _transition_power(
            transition,
            process,
            horizon,
        )
        propagated_mean = np.einsum(
            "ab,nb->na",
            propagated_transition,
            state_mean[index],
        )
        propagated_covariance = np.einsum(
            "ab,nbc,dc->nad",
            propagated_transition,
            posterior.component_state_covariance_m2[index],
            propagated_transition,
        )
        propagated_covariance += propagated_process[None, :, :]
        propagated_covariance = _admit_psd(
            propagated_covariance,
            name=f"component {index} forecast covariance",
        )
        component_mean[index] = propagated_mean[:, :3]
        component_velocity[index] = propagated_mean[:, 3:]
        component_covariance[index] = propagated_covariance[:, :3, :3]
    mean, covariance = _mixture_moments(
        posterior.component_weights,
        component_mean,
        component_covariance,
    )
    return FullCovarianceDynamicEndpointPredictionV3(
        mean_m=mean,
        covariance_m2=covariance,
        component_weights=posterior.component_weights,
        component_mean_m=component_mean,
        component_covariance_m2=component_covariance,
        component_velocity_mean_m_per_step=component_velocity,
        horizon_steps=horizon,
    )


__all__ = [
    "infer_full_covariance_dynamic_endpoint_model_average",
    "predict_full_covariance_dynamic_endpoint_model_average",
]
