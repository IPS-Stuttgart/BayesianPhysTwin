"""Robust random-walk bias filtering for persistent track drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RandomWalkBiasConfig:
    """Variance and robustness settings for a per-track nuisance bias."""

    process_variance: float = 2.5e-6
    base_process_variance: float = 1e-10
    initial_variance: float = 4.0e-6
    outlier_variance_multiplier: float = 100.0
    probability_floor: float = 1e-6


@dataclass(frozen=True)
class RandomWalkBiasResult:
    """Filtered bias state, gross-inlier responsibility, and sequence evidence."""

    bias_mean: np.ndarray
    bias_variance: np.ndarray
    inlier_probability: np.ndarray
    sequence_log_evidence: dict[str, float]

    @property
    def total_log_evidence(self) -> float:
        return float(sum(self.sequence_log_evidence.values()))


def _validate_config(config: RandomWalkBiasConfig) -> None:
    if config.process_variance < 0.0:
        raise ValueError("process_variance must be nonnegative")
    if config.base_process_variance < 0.0:
        raise ValueError("base_process_variance must be nonnegative")
    if config.initial_variance < 0.0:
        raise ValueError("initial_variance must be nonnegative")
    if config.outlier_variance_multiplier <= 1.0:
        raise ValueError("outlier_variance_multiplier must be greater than 1")
    if not 0.0 < config.probability_floor < 0.5:
        raise ValueError("probability_floor must be in (0, 0.5)")


def _ordered_groups(
    sequence_ids: np.ndarray,
    time_values: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups: dict[str, list[int]] = {}
    for index, sequence_id in enumerate(sequence_ids):
        groups.setdefault(str(sequence_id), []).append(index)

    ordered: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sequence_id, indexes in groups.items():
        index_array = np.asarray(indexes, dtype=int)
        raw_times = time_values[index_array]
        try:
            sortable_times = raw_times.astype(float)
        except (TypeError, ValueError):
            sortable_times = np.arange(index_array.size, dtype=float)
        order = np.argsort(sortable_times, kind="mergesort")
        ordered.append(
            (
                sequence_id,
                index_array[order],
                np.asarray(sortable_times[order], dtype=float),
            )
        )
    return ordered


def _validate_inputs(
    prior_reliability: np.ndarray,
    residual: np.ndarray,
    observation_variance: np.ndarray | float,
    sequence_ids: Sequence[str | int],
    time_values: Sequence[str | int | float],
    config: RandomWalkBiasConfig,
    bias_probability: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_config(config)
    prior = np.asarray(prior_reliability, dtype=float)
    residual_array = np.asarray(residual, dtype=float)
    if residual_array.ndim != 2:
        raise ValueError("residual must have shape (p, n)")
    _, measurement_count = residual_array.shape
    ids = np.asarray(sequence_ids)
    times = np.asarray(time_values)
    for name, values in (
        ("prior_reliability", prior),
        ("sequence_ids", ids),
        ("time_values", times),
    ):
        if values.shape != (measurement_count,):
            raise ValueError(
                f"{name} must have shape ({measurement_count},), got {values.shape}"
            )
    variance = np.asarray(observation_variance, dtype=float)
    if variance.shape == ():
        variance = np.full(measurement_count, variance.item(), dtype=float)
    if variance.shape != (measurement_count,):
        raise ValueError(
            "observation_variance must be scalar or shape "
            f"({measurement_count},), got {variance.shape}"
        )
    if not np.all(np.isfinite(residual_array)):
        raise ValueError("residual must contain finite values")
    if not np.all(np.isfinite(prior)):
        raise ValueError("prior_reliability must contain finite values")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("observation_variance must be finite and positive")
    if bias_probability is None:
        bias_probability_array = np.ones(measurement_count, dtype=float)
    else:
        bias_probability_array = np.asarray(bias_probability, dtype=float)
        if bias_probability_array.shape != (measurement_count,):
            raise ValueError(
                f"bias_probability must have shape ({measurement_count},), "
                f"got {bias_probability_array.shape}"
            )
        if not np.all(np.isfinite(bias_probability_array)):
            raise ValueError("bias_probability must contain finite values")
        bias_probability_array = np.clip(bias_probability_array, 0.0, 1.0)
    prior = np.clip(prior, config.probability_floor, 1.0 - config.probability_floor)
    return prior, residual_array, variance, ids, times, bias_probability_array


def _filter_batch(
    prior: np.ndarray,
    residual: np.ndarray,
    variance: np.ndarray,
    ids: np.ndarray,
    times: np.ndarray,
    bias_probability: np.ndarray,
    config: RandomWalkBiasConfig,
    *,
    store_history: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    particle_count, measurement_count = residual.shape
    total_evidence = np.zeros(particle_count, dtype=float)
    bias_history = np.empty_like(residual) if store_history else None
    variance_history = np.empty_like(residual) if store_history else None
    inlier_history = np.empty_like(residual) if store_history else None
    sequence_evidence: dict[str, np.ndarray] = {}

    for sequence_id, indexes, sequence_times in _ordered_groups(ids, times):
        mean = np.zeros(particle_count, dtype=float)
        state_variance = np.full(particle_count, config.initial_variance, dtype=float)
        evidence = np.zeros(particle_count, dtype=float)
        previous_time = sequence_times[0]
        for offset, index in enumerate(indexes):
            delta = 1.0 if offset == 0 else max(sequence_times[offset] - previous_time, 1.0)
            previous_time = sequence_times[offset]
            process_variance = (
                config.base_process_variance
                + config.process_variance * bias_probability[index]
            )
            predicted_variance = state_variance + process_variance * delta
            innovation = residual[:, index] - mean

            inlier_innovation_variance = predicted_variance + variance[index]
            outlier_innovation_variance = (
                predicted_variance
                + variance[index] * config.outlier_variance_multiplier
            )
            log_inlier = -0.5 * (
                np.log(2.0 * np.pi * inlier_innovation_variance)
                + np.square(innovation) / inlier_innovation_variance
            )
            log_outlier = -0.5 * (
                np.log(2.0 * np.pi * outlier_innovation_variance)
                + np.square(innovation) / outlier_innovation_variance
            )
            log_inlier_component = np.log(prior[index]) + log_inlier
            log_outlier_component = np.log1p(-prior[index]) + log_outlier
            log_mixture = np.logaddexp(log_inlier_component, log_outlier_component)
            inlier_probability = np.exp(log_inlier_component - log_mixture)
            evidence += log_mixture

            inlier_gain = predicted_variance / inlier_innovation_variance
            outlier_gain = predicted_variance / outlier_innovation_variance
            inlier_mean = mean + inlier_gain * innovation
            outlier_mean = mean + outlier_gain * innovation
            inlier_variance = (1.0 - inlier_gain) * predicted_variance
            outlier_variance = (1.0 - outlier_gain) * predicted_variance
            updated_mean = (
                inlier_probability * inlier_mean
                + (1.0 - inlier_probability) * outlier_mean
            )
            updated_variance = (
                inlier_probability
                * (inlier_variance + np.square(inlier_mean - updated_mean))
                + (1.0 - inlier_probability)
                * (outlier_variance + np.square(outlier_mean - updated_mean))
            )
            mean = updated_mean
            state_variance = np.maximum(updated_variance, 0.0)

            if store_history:
                assert bias_history is not None
                assert variance_history is not None
                assert inlier_history is not None
                bias_history[:, index] = mean
                variance_history[:, index] = state_variance
                inlier_history[:, index] = inlier_probability

        sequence_evidence[sequence_id] = evidence
        total_evidence += evidence

    return (
        total_evidence,
        bias_history,
        variance_history,
        inlier_history,
        sequence_evidence,
    )


def robust_random_walk_log_evidence_batch(
    prior_reliability: np.ndarray,
    residual: np.ndarray,
    observation_variance: np.ndarray | float,
    sequence_ids: Sequence[str | int],
    time_values: Sequence[str | int | float],
    *,
    config: RandomWalkBiasConfig | None = None,
    bias_probability: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-particle evidence after marginalizing a drifting track bias."""

    cfg = config or RandomWalkBiasConfig()
    prior, residual_array, variance, ids, times, bias_probability_array = _validate_inputs(
        prior_reliability,
        residual,
        observation_variance,
        sequence_ids,
        time_values,
        cfg,
        bias_probability,
    )
    evidence, _, _, _, _ = _filter_batch(
        prior,
        residual_array,
        variance,
        ids,
        times,
        bias_probability_array,
        cfg,
        store_history=False,
    )
    return evidence


def filter_random_walk_bias(
    prior_reliability: np.ndarray,
    residual: np.ndarray,
    observation_variance: np.ndarray | float,
    sequence_ids: Sequence[str | int],
    time_values: Sequence[str | int | float],
    *,
    config: RandomWalkBiasConfig | None = None,
    bias_probability: np.ndarray | None = None,
) -> RandomWalkBiasResult:
    """Filter one residual hypothesis and expose the nuisance-bias trajectory."""

    residual_array = np.asarray(residual, dtype=float)
    if residual_array.ndim != 1:
        raise ValueError("residual must have shape (n,) for bias filtering")
    cfg = config or RandomWalkBiasConfig()
    prior, batched_residual, variance, ids, times, bias_probability_array = _validate_inputs(
        prior_reliability,
        residual_array[None, :],
        observation_variance,
        sequence_ids,
        time_values,
        cfg,
        bias_probability,
    )
    _, bias, bias_variance, inlier, sequence_evidence = _filter_batch(
        prior,
        batched_residual,
        variance,
        ids,
        times,
        bias_probability_array,
        cfg,
        store_history=True,
    )
    assert bias is not None
    assert bias_variance is not None
    assert inlier is not None
    return RandomWalkBiasResult(
        bias_mean=bias[0],
        bias_variance=bias_variance[0],
        inlier_probability=inlier[0],
        sequence_log_evidence={key: float(value[0]) for key, value in sequence_evidence.items()},
    )
