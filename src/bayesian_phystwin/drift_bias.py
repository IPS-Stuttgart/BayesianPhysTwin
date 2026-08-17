"""Robust random-walk bias filtering for persistent track drift."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

_SequenceIdentity = tuple[str, str | int]


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
    sequence_log_evidence: Mapping[str, float]

    def __post_init__(self) -> None:
        mean = np.array(self.bias_mean, dtype=np.float64, copy=True, order="C")
        variance = np.array(self.bias_variance, dtype=np.float64, copy=True, order="C")
        probability = np.array(
            self.inlier_probability,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if (
            mean.ndim != 1
            or len(mean) == 0
            or variance.shape != mean.shape
            or probability.shape != mean.shape
        ):
            raise ValueError(
                "bias result arrays must have one nonempty equal vector shape"
            )
        if not np.all(np.isfinite(mean)):
            raise ValueError("bias_mean must contain finite values")
        if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("bias_variance must be finite and nonnegative")
        if not np.all(np.isfinite(probability)) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError("inlier_probability must lie in [0, 1]")
        if not isinstance(self.sequence_log_evidence, Mapping):
            raise TypeError("sequence_log_evidence must be a mapping")
        evidence: dict[str, float] = {}
        for sequence_id, raw_value in self.sequence_log_evidence.items():
            if not isinstance(sequence_id, str) or not sequence_id:
                raise ValueError("sequence_log_evidence keys must be nonempty strings")
            if isinstance(raw_value, (bool, np.bool_)):
                raise ValueError("sequence_log_evidence values must be finite numbers")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "sequence_log_evidence values must be finite numbers"
                ) from error
            if not np.isfinite(value):
                raise ValueError("sequence_log_evidence values must be finite numbers")
            evidence[sequence_id] = value
        for array in (mean, variance, probability):
            array.setflags(write=False)
        object.__setattr__(self, "bias_mean", mean)
        object.__setattr__(self, "bias_variance", variance)
        object.__setattr__(self, "inlier_probability", probability)
        object.__setattr__(
            self,
            "sequence_log_evidence",
            MappingProxyType(dict(evidence)),
        )

    @property
    def total_log_evidence(self) -> float:
        return float(sum(self.sequence_log_evidence.values()))


def _validate_config(config: RandomWalkBiasConfig) -> None:
    for name, value in (
        ("process_variance", config.process_variance),
        ("base_process_variance", config.base_process_variance),
        ("initial_variance", config.initial_variance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (
        not np.isfinite(config.outlier_variance_multiplier)
        or config.outlier_variance_multiplier <= 1.0
    ):
        raise ValueError(
            "outlier_variance_multiplier must be finite and greater than 1"
        )
    if not np.isfinite(config.probability_floor) or not (
        0.0 < config.probability_floor < 0.5
    ):
        raise ValueError("probability_floor must be finite and in (0, 0.5)")


def _validated_config(config: RandomWalkBiasConfig | None) -> RandomWalkBiasConfig:
    if config is None:
        result = RandomWalkBiasConfig()
    elif isinstance(config, RandomWalkBiasConfig):
        result = config
    else:
        raise TypeError("config must be a RandomWalkBiasConfig")
    _validate_config(result)
    return result


def _validated_sequence_identity(value: object) -> tuple[_SequenceIdentity, str]:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("sequence_ids must contain strings or integers, not booleans")
    if isinstance(value, str):
        if not value:
            raise ValueError("sequence_ids must contain nonempty strings or integers")
        return ("str", value), value
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        return ("int", integer), str(integer)
    raise TypeError("sequence_ids must contain strings or integers")


def _ordered_groups(
    sequence_ids: np.ndarray,
    time_values: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups: dict[_SequenceIdentity, list[int]] = {}
    output_keys: dict[_SequenceIdentity, str] = {}
    output_key_owners: dict[str, _SequenceIdentity] = {}
    for index, sequence_id in enumerate(sequence_ids):
        identity, output_key = _validated_sequence_identity(sequence_id)
        owner = output_key_owners.get(output_key)
        if owner is not None and owner != identity:
            raise ValueError(
                "sequence_ids contain distinct typed identities that collide "
                f"after serialization: {output_key!r}"
            )
        output_key_owners[output_key] = identity
        output_keys[identity] = output_key
        groups.setdefault(identity, []).append(index)

    ordered: list[tuple[str, np.ndarray, np.ndarray]] = []
    for identity, indexes in groups.items():
        index_array = np.asarray(indexes, dtype=int)
        raw_times = time_values[index_array]
        try:
            sortable_times = raw_times.astype(float)
        except (TypeError, ValueError):
            sortable_times = np.arange(index_array.size, dtype=float)
        else:
            if not np.all(np.isfinite(sortable_times)):
                raise ValueError("numeric time_values must contain finite values")
        order = np.argsort(sortable_times, kind="mergesort")
        ordered.append(
            (
                output_keys[identity],
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
    particle_count, measurement_count = residual_array.shape
    if particle_count == 0 or measurement_count == 0:
        raise ValueError("at least one particle and measurement are required")
    ids = np.asarray(sequence_ids, dtype=object)
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
    if np.any((prior < 0.0) | (prior > 1.0)):
        raise ValueError("prior_reliability must lie in [0, 1]")
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
        if np.any((bias_probability_array < 0.0) | (bias_probability_array > 1.0)):
            raise ValueError("bias_probability must lie in [0, 1]")
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
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, np.ndarray],
]:
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
            delta = (
                1.0 if offset == 0 else max(sequence_times[offset] - previous_time, 1.0)
            )
            previous_time = sequence_times[offset]
            try:
                with np.errstate(
                    divide="raise",
                    invalid="raise",
                    over="raise",
                    under="ignore",
                ):
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
                    log_mixture = np.logaddexp(
                        log_inlier_component,
                        log_outlier_component,
                    )
                    inlier_probability = np.exp(log_inlier_component - log_mixture)
            except FloatingPointError as error:
                raise FloatingPointError(
                    "random-walk bias likelihood produced non-finite values"
                ) from error
            if not np.all(np.isfinite(log_mixture)) or not np.all(
                np.isfinite(inlier_probability)
            ):
                raise FloatingPointError(
                    "random-walk bias likelihood produced non-finite values"
                )
            evidence += log_mixture

            try:
                with np.errstate(
                    divide="raise",
                    invalid="raise",
                    over="raise",
                    under="ignore",
                ):
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
                    updated_variance = inlier_probability * (
                        inlier_variance + np.square(inlier_mean - updated_mean)
                    ) + (1.0 - inlier_probability) * (
                        outlier_variance + np.square(outlier_mean - updated_mean)
                    )
            except FloatingPointError as error:
                raise FloatingPointError(
                    "random-walk bias state update produced non-finite values"
                ) from error
            mean = updated_mean
            state_variance = np.maximum(updated_variance, 0.0)
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(state_variance)):
                raise FloatingPointError(
                    "random-walk bias state update produced non-finite values"
                )

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

    cfg = _validated_config(config)
    prior, residual_array, variance, ids, times, bias_probability_array = (
        _validate_inputs(
            prior_reliability,
            residual,
            observation_variance,
            sequence_ids,
            time_values,
            cfg,
            bias_probability,
        )
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
    evidence.setflags(write=False)
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
    cfg = _validated_config(config)
    prior, batched_residual, variance, ids, times, bias_probability_array = (
        _validate_inputs(
            prior_reliability,
            residual_array[None, :],
            observation_variance,
            sequence_ids,
            time_values,
            cfg,
            bias_probability,
        )
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
        sequence_log_evidence={
            key: float(value[0]) for key, value in sequence_evidence.items()
        },
    )
