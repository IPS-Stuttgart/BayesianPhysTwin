"""Robust basis-space inference for structured discrepancy beliefs."""

from __future__ import annotations

import numpy as np

from ._structured_discrepancy_contracts import (
    StructuredDiscrepancyConfigV1,
    StructuredDiscrepancyPosteriorV1,
    StructuredDiscrepancyPredictionV1,
    _validated_basis,
)
from .endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
)


def _validated_inputs(
    residual_m: np.ndarray,
    valid: np.ndarray,
    prior_reliability: np.ndarray | None,
    *,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    raw_residual = np.asarray(residual_m)
    if raw_residual.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("residual_m must be real numeric")
    residual = np.asarray(raw_residual, dtype=np.float64)
    if residual.ndim != 3 or residual.shape[2:] != (3,) or residual.shape[1] < 1:
        raise ValueError("residual_m must have shape (T, N>=1, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_m must contain only finite values")

    raw_validity = np.asarray(valid)
    if raw_validity.dtype.kind != "b":
        raise TypeError("valid must be a boolean array")
    validity = np.asarray(raw_validity, dtype=bool)
    if validity.shape != residual.shape[:2]:
        raise ValueError("valid must match the residual frame and track dimensions")

    if prior_reliability is None:
        reliability = np.ones(validity.shape, dtype=np.float64)
    else:
        raw_reliability = np.asarray(prior_reliability)
        if raw_reliability.dtype.kind in {"b", "O", "U", "S"}:
            raise TypeError("prior_reliability must be numeric")
        reliability = np.asarray(raw_reliability, dtype=np.float64)
        if reliability.shape != validity.shape:
            raise ValueError(
                "prior_reliability must match the residual frame and track dimensions"
            )
        if not np.all(np.isfinite(reliability)) or np.any(
            (reliability < 0.0) | (reliability > 1.0)
        ):
            raise ValueError("prior_reliability must lie in [0, 1]")

    if isinstance(end_frame, (bool, np.bool_)):
        raise TypeError("end_frame must be an integer")
    frame_stop = int(end_frame)
    if frame_stop != end_frame:
        raise ValueError("end_frame must be an integer")
    if not 0 < frame_stop <= len(residual):
        raise ValueError("end_frame must lie inside the residual sequence")
    return residual, validity, reliability, frame_stop


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _solve_spd(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    symmetric = _symmetric(np.asarray(matrix, dtype=np.float64))
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "structured discrepancy precision is not positive definite"
        ) from error
    intermediate = np.linalg.solve(factor, right_hand_side)
    return np.linalg.solve(factor.T, intermediate)


def _inverse_spd(matrix: np.ndarray) -> np.ndarray:
    return _solve_spd(matrix, np.eye(len(matrix), dtype=np.float64))


def _log_mixture_and_probability(
    innovation: np.ndarray,
    nominal_variance: np.ndarray,
    outlier_variance: np.ndarray,
    *,
    inlier_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    squared_norm = np.sum(np.square(innovation), axis=1)
    log_nominal = np.log(inlier_prior) - 0.5 * (
        3.0 * np.log(2.0 * np.pi * nominal_variance)
        + squared_norm / nominal_variance
    )
    log_outlier = np.log1p(-inlier_prior) - 0.5 * (
        3.0 * np.log(2.0 * np.pi * outlier_variance)
        + squared_norm / outlier_variance
    )
    log_mixture = np.logaddexp(log_nominal, log_outlier)
    probability = np.exp(log_nominal - log_mixture)
    return log_mixture, probability


def _is_binary_reliability(reliability: np.ndarray) -> bool:
    return bool(np.all((reliability == 0.0) | (reliability == 1.0)))


def _global_component_score(
    cumulative_track_score: np.ndarray,
    update_count: np.ndarray,
) -> float:
    supported = update_count > 0
    if not np.any(supported):
        return 0.0
    return float(np.mean(cumulative_track_score[supported]))


def _filter_full_rank_component(
    residual: np.ndarray,
    validity: np.ndarray,
    reliability: np.ndarray,
    *,
    end_frame: int,
    component_index: int,
    config: StructuredDiscrepancyConfigV1,
    spatial_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    component = config.endpoint_config.components[component_index]
    single_config = ModelAveragedEndpointConfigV1(
        components=(component,),
        component_prior_probability=(1.0,),
    )
    effective_validity = validity & (reliability > 0.0)
    posterior = infer_model_averaged_endpoint(
        residual,
        effective_validity,
        end_frame=end_frame,
        config=single_config,
    )
    field_variance = posterior.component_variance_m2[0]
    coefficient_mean = spatial_basis.T @ posterior.component_mean_m[0]
    coefficient_covariance = (
        spatial_basis.T @ (field_variance[:, None] * spatial_basis)
    )
    local_variance = np.zeros(residual.shape[1], dtype=np.float64)
    cumulative_score = posterior.component_log_evidence[:, 0]
    score = _global_component_score(cumulative_score, posterior.update_count)
    return (
        coefficient_mean,
        coefficient_covariance,
        local_variance,
        posterior.final_nominal_probability,
        posterior.update_count,
        score,
    )


def _filter_structured_component(
    residual: np.ndarray,
    validity: np.ndarray,
    reliability: np.ndarray,
    basis: np.ndarray,
    *,
    end_frame: int,
    component_index: int,
    config: StructuredDiscrepancyConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    component = config.endpoint_config.components[component_index]
    process_variance = component.process_std_m**2
    observation_variance = component.observation_std_m**2
    initial_variance = component.initial_std_m**2
    outlier_multiplier = component.outlier_variance_multiplier

    track_count, rank = basis.shape
    leverage = np.sum(np.square(basis), axis=1)
    complement = np.maximum(1.0 - leverage, 0.0)
    coefficient_mean = np.zeros((rank, 3), dtype=np.float64)
    coefficient_covariance = np.eye(rank, dtype=np.float64) * initial_variance
    local_variance = complement * initial_variance
    final_probability = np.zeros(track_count, dtype=np.float64)
    update_count = np.zeros(track_count, dtype=np.int64)
    cumulative_score = np.zeros(track_count, dtype=np.float64)

    for frame in range(end_frame):
        coefficient_covariance = coefficient_covariance + (
            np.eye(rank, dtype=np.float64) * process_variance
        )
        local_variance = local_variance + complement * process_variance

        mask = validity[frame] & (reliability[frame] > 0.0)
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        design = basis[indices]
        observation = residual[frame, indices]
        row_reliability = reliability[frame, indices]
        innovation = observation - design @ coefficient_mean
        represented_variance = np.einsum(
            "ir,rs,is->i",
            design,
            coefficient_covariance,
            design,
        )
        unresolved_variance = local_variance[indices]
        nominal_predictive_variance = (
            represented_variance + unresolved_variance + observation_variance
        )
        outlier_predictive_variance = (
            represented_variance
            + unresolved_variance
            + observation_variance * outlier_multiplier
        )
        log_mixture, probability = _log_mixture_and_probability(
            innovation,
            nominal_predictive_variance,
            outlier_predictive_variance,
            inlier_prior=component.inlier_prior,
        )
        cumulative_score[indices] += row_reliability * log_mixture
        final_probability[indices] = probability
        update_count[indices] += 1

        nominal_noise = unresolved_variance + observation_variance
        outlier_noise = (
            unresolved_variance + observation_variance * outlier_multiplier
        )
        effective_precision = row_reliability * (
            probability / nominal_noise
            + (1.0 - probability) / outlier_noise
        )
        prior_precision = _inverse_spd(coefficient_covariance)
        posterior_precision = prior_precision + design.T @ (
            effective_precision[:, None] * design
        )
        information_mean = (
            prior_precision @ coefficient_mean
            + design.T @ (effective_precision[:, None] * observation)
        )
        coefficient_mean = _solve_spd(posterior_precision, information_mean)
        coefficient_covariance = _inverse_spd(posterior_precision)
        coefficient_covariance = _symmetric(coefficient_covariance)

    score = _global_component_score(cumulative_score, update_count)
    return (
        coefficient_mean,
        coefficient_covariance,
        local_variance,
        final_probability,
        update_count,
        score,
    )


def infer_structured_discrepancy(
    residual_m: np.ndarray,
    valid: np.ndarray,
    spatial_basis: np.ndarray,
    *,
    end_frame: int,
    prior_reliability: np.ndarray | None = None,
    config: StructuredDiscrepancyConfigV1 | None = None,
) -> StructuredDiscrepancyPosteriorV1:
    """Infer a robust endpoint belief in a shared spatial basis.

    The basis columns must be orthonormal. Omitted spatial directions retain a
    marginal-preserving diagonal initial/process remainder derived from the
    projector leverage rather than being reported with zero uncertainty.
    """

    settings = StructuredDiscrepancyConfigV1() if config is None else config
    if not isinstance(settings, StructuredDiscrepancyConfigV1):
        raise TypeError("config must be a StructuredDiscrepancyConfigV1")
    residual, validity, reliability, frame_stop = _validated_inputs(
        residual_m,
        valid,
        prior_reliability,
        end_frame=end_frame,
    )
    basis = _validated_basis(
        spatial_basis,
        track_count=residual.shape[1],
        tolerance=settings.basis_orthonormal_atol,
    )
    component_count = len(settings.endpoint_config.components)
    rank = basis.shape[1]
    track_count = basis.shape[0]
    coefficient_mean = np.empty((component_count, rank, 3), dtype=np.float64)
    coefficient_covariance = np.empty(
        (component_count, rank, rank),
        dtype=np.float64,
    )
    local_variance = np.empty((component_count, track_count), dtype=np.float64)
    component_probability = np.empty(
        (component_count, track_count),
        dtype=np.float64,
    )
    scores = np.empty(component_count, dtype=np.float64)
    process_variance = np.empty(component_count, dtype=np.float64)
    common_update_count: np.ndarray | None = None
    exact_full_rank = basis.shape[0] == basis.shape[1] and _is_binary_reliability(
        reliability[:frame_stop]
    )
    for index, component in enumerate(settings.endpoint_config.components):
        if exact_full_rank:
            result = _filter_full_rank_component(
                residual,
                validity,
                reliability,
                end_frame=frame_stop,
                component_index=index,
                config=settings,
                spatial_basis=basis,
            )
        else:
            result = _filter_structured_component(
                residual,
                validity,
                reliability,
                basis,
                end_frame=frame_stop,
                component_index=index,
                config=settings,
            )
        (
            coefficient_mean[index],
            coefficient_covariance[index],
            local_variance[index],
            component_probability[index],
            update_count,
            scores[index],
        ) = result
        process_variance[index] = component.process_std_m**2
        if common_update_count is None:
            common_update_count = update_count
        elif not np.array_equal(common_update_count, update_count):
            raise AssertionError("structured components used different observations")
    assert common_update_count is not None
    prior = np.asarray(
        settings.endpoint_config.component_prior_probability,
        dtype=np.float64,
    )
    log_weight = np.log(prior) + scores
    log_weight = log_weight - np.max(log_weight)
    weights = np.exp(log_weight)
    weights = weights / np.sum(weights)
    return StructuredDiscrepancyPosteriorV1(
        spatial_basis=basis,
        component_coefficient_mean_m=coefficient_mean,
        component_coefficient_covariance_m2=coefficient_covariance,
        component_local_variance_m2=local_variance,
        component_weights=weights,
        component_log_score=scores,
        component_final_nominal_probability=component_probability,
        update_count=common_update_count,
        component_process_variance_m2=process_variance,
        config=settings,
        end_frame=frame_stop,
    )


def predict_structured_discrepancy(
    posterior: StructuredDiscrepancyPosteriorV1,
    *,
    horizon_steps: int,
) -> StructuredDiscrepancyPredictionV1:
    """Propagate factorized covariance without future observations."""

    if not isinstance(posterior, StructuredDiscrepancyPosteriorV1):
        raise TypeError("posterior must be a StructuredDiscrepancyPosteriorV1")
    if isinstance(horizon_steps, (bool, np.bool_)):
        raise TypeError("horizon_steps must be an integer")
    horizon = int(horizon_steps)
    if horizon != horizon_steps or horizon < 0:
        raise ValueError("horizon_steps must be a nonnegative integer")
    rank = posterior.spatial_basis.shape[1]
    identity = np.eye(rank, dtype=np.float64)
    propagated_covariance = posterior.component_coefficient_covariance_m2 + (
        horizon
        * posterior.component_process_variance_m2[:, None, None]
        * identity[None, :, :]
    )
    leverage = np.sum(np.square(posterior.spatial_basis), axis=1)
    complement = np.maximum(1.0 - leverage, 0.0)
    propagated_local = posterior.component_local_variance_m2 + (
        horizon
        * posterior.component_process_variance_m2[:, None]
        * complement[None, :]
    )
    return StructuredDiscrepancyPredictionV1(
        spatial_basis=posterior.spatial_basis,
        component_coefficient_mean_m=posterior.component_coefficient_mean_m,
        component_coefficient_covariance_m2=propagated_covariance,
        component_local_variance_m2=propagated_local,
        component_weights=posterior.component_weights,
        component_process_variance_m2=posterior.component_process_variance_m2,
        config=posterior.config,
        source_end_frame=posterior.end_frame,
        horizon_steps=horizon,
    )


__all__ = [
    "infer_structured_discrepancy",
    "predict_structured_discrepancy",
]
