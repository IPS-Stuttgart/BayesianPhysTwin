"""Low-dimensional profile posterior utilities for official PhysTwin trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phystwin_refit import PhysTwinTrackObjective


@dataclass(frozen=True)
class PhysTwinGridPosterior:
    """Normalized two-parameter grid posterior and compact diagnostics."""

    weights: np.ndarray
    log_posterior: np.ndarray
    summary: dict[str, object]


def clustered_track_log_likelihood(
    observed: np.ndarray,
    trajectory: np.ndarray,
    objective: PhysTwinTrackObjective,
    *,
    start_frame: int,
    end_frame: int,
    variance: float,
    outlier_variance_multiplier: float = 100.0,
    temperature: float = 1.0,
) -> float:
    """Return a frame-clustered composite log likelihood.

    Tracks within one frame are averaged before frame contributions are summed.
    This avoids treating thousands of spatially correlated tracks as independent
    replicates. ``temperature`` remains explicit for calibration experiments.
    """

    observed_array = np.asarray(observed, dtype=float)
    trajectory_array = np.asarray(trajectory, dtype=float)
    if observed_array.ndim != 3 or observed_array.shape[2] != 3:
        raise ValueError("observed must have shape (T, N, 3)")
    if trajectory_array.ndim != 3 or trajectory_array.shape[2] != 3:
        raise ValueError("trajectory must have shape (T, M, 3)")
    if trajectory_array.shape[0] < end_frame:
        raise ValueError("trajectory does not cover the requested frames")
    if trajectory_array.shape[1] < observed_array.shape[1]:
        raise ValueError("trajectory has fewer vertices than observed tracks")
    if not 0 <= start_frame < end_frame <= len(observed_array):
        raise ValueError("invalid frame interval")
    if variance <= 0.0:
        raise ValueError("variance must be positive")
    if outlier_variance_multiplier <= 1.0:
        raise ValueError("outlier_variance_multiplier must be greater than one")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    residual = observed_array - trajectory_array[
        : len(observed_array), : observed_array.shape[1]
    ]
    squared_norm = np.sum(np.square(residual), axis=2)
    total_negative_log_likelihood = 0.0
    for frame in range(start_frame, end_frame):
        if objective.variant == "mixture":
            support = objective.support[frame].astype(bool)
            if not np.any(support):
                continue
            prior = objective.prior_inlier_probability[frame, support].astype(float)
            frame_q = squared_norm[frame, support]
            log_inlier = np.log(prior) - 0.5 * frame_q / variance
            log_outlier = (
                np.log1p(-prior)
                - 1.5 * np.log(outlier_variance_multiplier)
                - 0.5 * frame_q / (variance * outlier_variance_multiplier)
            )
            zero_log_mixture = np.logaddexp(
                np.log(prior),
                np.log1p(-prior) - 1.5 * np.log(outlier_variance_multiplier),
            )
            track_nll = zero_log_mixture - np.logaddexp(log_inlier, log_outlier)
            total_negative_log_likelihood += float(np.mean(track_nll))
        else:
            weights = objective.weights[frame].astype(float)
            weight_sum = float(np.sum(weights))
            if weight_sum == 0.0:
                continue
            total_negative_log_likelihood += float(
                np.sum(0.5 * weights * squared_norm[frame] / variance) / weight_sum
            )
    return -total_negative_log_likelihood / temperature


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, quantile, side="left")])


def grid_parameter_posterior(
    object_log_scales: np.ndarray,
    controller_log_scales: np.ndarray,
    log_likelihood: np.ndarray,
    *,
    object_prior_std: float,
    controller_prior_std: float,
) -> PhysTwinGridPosterior:
    """Combine a clustered likelihood grid with independent zero-mean priors."""

    object_values = np.asarray(object_log_scales, dtype=float)
    controller_values = np.asarray(controller_log_scales, dtype=float)
    likelihood = np.asarray(log_likelihood, dtype=float)
    if object_values.ndim != 1 or controller_values.ndim != 1:
        raise ValueError("scale grids must be one-dimensional")
    if likelihood.shape != (len(object_values), len(controller_values)):
        raise ValueError("log_likelihood shape must match both scale grids")
    if not np.all(np.isfinite(likelihood)):
        raise ValueError("log_likelihood must contain finite values")
    if object_prior_std <= 0.0 or controller_prior_std <= 0.0:
        raise ValueError("prior standard deviations must be positive")

    object_grid, controller_grid = np.meshgrid(
        object_values,
        controller_values,
        indexing="ij",
    )
    log_prior = -0.5 * np.square(object_grid / object_prior_std)
    log_prior -= 0.5 * np.square(controller_grid / controller_prior_std)
    log_posterior = likelihood + log_prior
    shifted = log_posterior - np.max(log_posterior)
    weights = np.exp(shifted)
    weights /= np.sum(weights)

    flat_weights = weights.reshape(-1)
    flat_object = object_grid.reshape(-1)
    flat_controller = controller_grid.reshape(-1)
    object_mean = float(np.sum(flat_weights * flat_object))
    controller_mean = float(np.sum(flat_weights * flat_controller))
    object_centered = flat_object - object_mean
    controller_centered = flat_controller - controller_mean
    object_variance = float(np.sum(flat_weights * np.square(object_centered)))
    controller_variance = float(
        np.sum(flat_weights * np.square(controller_centered))
    )
    covariance = float(
        np.sum(flat_weights * object_centered * controller_centered)
    )
    denominator = np.sqrt(object_variance * controller_variance)
    correlation = covariance / denominator if denominator > 0.0 else 0.0
    map_index = np.unravel_index(np.argmax(weights), weights.shape)
    boundary_mask = np.zeros_like(weights, dtype=bool)
    boundary_mask[[0, -1], :] = True
    boundary_mask[:, [0, -1]] = True
    summary: dict[str, object] = {
        "map": {
            "object_log_scale": float(object_values[map_index[0]]),
            "controller_log_scale": float(controller_values[map_index[1]]),
        },
        "object_log_scale": {
            "mean": object_mean,
            "std": float(np.sqrt(object_variance)),
            "q05": _weighted_quantile(flat_object, flat_weights, 0.05),
            "q95": _weighted_quantile(flat_object, flat_weights, 0.95),
            "mean_multiplier": float(
                np.sum(flat_weights * np.exp(flat_object))
            ),
        },
        "controller_log_scale": {
            "mean": controller_mean,
            "std": float(np.sqrt(controller_variance)),
            "q05": _weighted_quantile(flat_controller, flat_weights, 0.05),
            "q95": _weighted_quantile(flat_controller, flat_weights, 0.95),
            "mean_multiplier": float(
                np.sum(flat_weights * np.exp(flat_controller))
            ),
        },
        "correlation": float(correlation),
        "effective_grid_points": float(1.0 / np.sum(np.square(weights))),
        "boundary_probability": float(np.sum(weights[boundary_mask])),
    }
    return PhysTwinGridPosterior(
        weights=weights,
        log_posterior=log_posterior,
        summary=summary,
    )


def weighted_trajectory_moments(
    trajectories: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and epistemic variance for trajectory particles."""

    values = np.asarray(trajectories, dtype=float)
    probabilities = np.asarray(weights, dtype=float).reshape(-1)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("trajectories must have shape (P, T, N, 3)")
    if len(probabilities) != len(values):
        raise ValueError("weights must match trajectory particle count")
    if np.any(probabilities < 0.0) or not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    mean = np.tensordot(probabilities, values, axes=(0, 0))
    second_moment = np.tensordot(probabilities, np.square(values), axes=(0, 0))
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    return mean, variance


def predictive_observation_calibration(
    observed: np.ndarray,
    mean: np.ndarray,
    epistemic_variance: np.ndarray,
    mask: np.ndarray,
    *,
    observation_variance: float,
    model_discrepancy_variance: float | np.ndarray = 0.0,
) -> dict[str, float | int]:
    """Evaluate 90% Gaussian observation intervals and normalized residuals."""

    observed_array = np.asarray(observed, dtype=float)
    mean_array = np.asarray(mean, dtype=float)
    epistemic = np.asarray(epistemic_variance, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if observed_array.shape != mean_array.shape or observed_array.shape != epistemic.shape:
        raise ValueError("observed, mean, and variance must have the same shape")
    if mask_array.shape != observed_array.shape[:2]:
        raise ValueError("mask must match the trajectory's first two axes")
    if observation_variance <= 0.0:
        raise ValueError("observation_variance must be positive")
    discrepancy = np.asarray(model_discrepancy_variance, dtype=float)
    if discrepancy.shape == ():
        discrepancy_array = np.full_like(observed_array, discrepancy.item())
    elif discrepancy.shape == (len(observed_array),):
        discrepancy_array = np.broadcast_to(
            discrepancy[:, None, None],
            observed_array.shape,
        )
    elif discrepancy.shape == observed_array.shape:
        discrepancy_array = discrepancy
    else:
        raise ValueError(
            "model_discrepancy_variance must be scalar, shape (T,), or trajectory-shaped"
        )
    if not np.all(np.isfinite(discrepancy_array)) or np.any(discrepancy_array < 0.0):
        raise ValueError("model_discrepancy_variance must be finite and nonnegative")
    selected_residual = (observed_array - mean_array)[mask_array]
    selected_epistemic = epistemic[mask_array]
    selected_discrepancy = discrepancy_array[mask_array]
    if len(selected_residual) == 0:
        return {"count": 0}
    total_variance = (
        selected_epistemic + observation_variance + selected_discrepancy
    )
    normalized_sq = np.square(selected_residual) / total_variance
    covered = np.abs(selected_residual) <= 1.6448536269514722 * np.sqrt(
        total_variance
    )
    return {
        "count": int(len(selected_residual)),
        "coordinate_coverage_90": float(np.mean(covered)),
        "mean_nees_per_coordinate": float(np.mean(normalized_sq)),
        "mean_epistemic_std_m": float(np.mean(np.sqrt(selected_epistemic))),
        "p95_epistemic_std_m": float(
            np.quantile(np.sqrt(selected_epistemic), 0.95)
        ),
    }


def causal_model_discrepancy_variance(
    observed: np.ndarray,
    mean: np.ndarray,
    epistemic_variance: np.ndarray,
    mask: np.ndarray,
    *,
    observation_variance: float,
    decay: float,
    initial_variance: float = 0.0,
) -> np.ndarray:
    """Estimate one-step model discrepancy using only previous-frame residuals."""

    observed_array = np.asarray(observed, dtype=float)
    mean_array = np.asarray(mean, dtype=float)
    epistemic = np.asarray(epistemic_variance, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if observed_array.shape != mean_array.shape or observed_array.shape != epistemic.shape:
        raise ValueError("observed, mean, and epistemic_variance must have equal shape")
    if mask_array.shape != observed_array.shape[:2]:
        raise ValueError("mask must match the trajectory's first two axes")
    if observation_variance <= 0.0 or initial_variance < 0.0:
        raise ValueError("variance values are invalid")
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")

    residual = observed_array - mean_array
    predicted = np.empty(len(observed_array), dtype=float)
    current = float(initial_variance)
    for frame in range(len(observed_array)):
        predicted[frame] = current
        selected = mask_array[frame]
        if not np.any(selected):
            continue
        residual_variance = float(
            np.mean(
                np.square(residual[frame, selected])
                - epistemic[frame, selected]
            )
        )
        target = max(residual_variance - observation_variance, 0.0)
        current = decay * current + (1.0 - decay) * target
    return predicted
