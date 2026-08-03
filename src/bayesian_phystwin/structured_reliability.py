"""Temporally structured reliability for tracked pseudo-measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MARKOV_TIME_MODE_ORDER_ONLY = "order-only"
MARKOV_TIME_MODE_INTEGER_STEPS = "integer-steps"
_MARKOV_TIME_MODES = frozenset(
    {MARKOV_TIME_MODE_ORDER_ONLY, MARKOV_TIME_MODE_INTEGER_STEPS}
)


@dataclass(frozen=True)
class MarkovReliabilityConfig:
    """Persistence parameters for binary inlier/outlier track states.

    ``time_delta_mode="order-only"`` preserves the historical behavior: time
    values determine ordering but every neighboring observation receives one
    transition. ``time_delta_mode="integer-steps"`` instead raises the
    transition matrix to the number of elapsed ``time_step`` intervals. The
    latter is useful for dropped frames and irregularly sampled tracks while
    remaining exactly equivalent on unit-spaced inputs.
    """

    inlier_persistence: float = 0.98
    outlier_persistence: float = 0.90
    probability_floor: float = 1e-6
    time_delta_mode: str = MARKOV_TIME_MODE_ORDER_ONLY
    time_step: float = 1.0


@dataclass(frozen=True)
class MarkovReliabilityResult:
    """Smoothed inlier probabilities and normalized sequence evidence."""

    posterior_inlier_probability: np.ndarray
    sequence_log_evidence: dict[str, float]

    @property
    def total_log_evidence(self) -> float:
        return float(sum(self.sequence_log_evidence.values()))

    @property
    def sequence_count(self) -> int:
        return len(self.sequence_log_evidence)


def _validate_config(config: MarkovReliabilityConfig) -> None:
    if not 0.0 < config.inlier_persistence < 1.0:
        raise ValueError("inlier_persistence must be in (0, 1)")
    if not 0.0 < config.outlier_persistence < 1.0:
        raise ValueError("outlier_persistence must be in (0, 1)")
    if not 0.0 < config.probability_floor < 0.5:
        raise ValueError("probability_floor must be in (0, 0.5)")
    if config.time_delta_mode not in _MARKOV_TIME_MODES:
        raise ValueError(
            "time_delta_mode must be one of "
            f"{sorted(_MARKOV_TIME_MODES)}"
        )
    if not np.isfinite(config.time_step) or config.time_step <= 0.0:
        raise ValueError("time_step must be finite and positive")


def _transition_at(log_transition: np.ndarray, offset: int) -> np.ndarray:
    if log_transition.ndim == 2:
        return log_transition
    return log_transition[offset]


def _forward(
    unary_log_potential: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[np.ndarray, float]:
    length = unary_log_potential.shape[0]
    alpha = np.empty((length, 2), dtype=float)
    alpha[0] = log_initial + unary_log_potential[0]
    for time in range(1, length):
        transition = _transition_at(log_transition, time - 1)
        alpha[time, 0] = unary_log_potential[time, 0] + np.logaddexp(
            alpha[time - 1, 0] + transition[0, 0],
            alpha[time - 1, 1] + transition[1, 0],
        )
        alpha[time, 1] = unary_log_potential[time, 1] + np.logaddexp(
            alpha[time - 1, 0] + transition[0, 1],
            alpha[time - 1, 1] + transition[1, 1],
        )
    return alpha, float(np.logaddexp(alpha[-1, 0], alpha[-1, 1]))


def _backward(
    unary_log_potential: np.ndarray,
    log_transition: np.ndarray,
) -> np.ndarray:
    length = unary_log_potential.shape[0]
    beta = np.zeros((length, 2), dtype=float)
    for time in range(length - 2, -1, -1):
        transition = _transition_at(log_transition, time)
        beta[time, 0] = np.logaddexp(
            transition[0, 0]
            + unary_log_potential[time + 1, 0]
            + beta[time + 1, 0],
            transition[0, 1]
            + unary_log_potential[time + 1, 1]
            + beta[time + 1, 1],
        )
        beta[time, 1] = np.logaddexp(
            transition[1, 0]
            + unary_log_potential[time + 1, 0]
            + beta[time + 1, 0],
            transition[1, 1]
            + unary_log_potential[time + 1, 1]
            + beta[time + 1, 1],
        )
    return beta


def _ordered_group_indices(
    sequence_ids: np.ndarray,
    time_values: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups: dict[str, list[int]] = {}
    for index, sequence_id in enumerate(sequence_ids):
        groups.setdefault(str(sequence_id), []).append(index)

    ordered: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sequence_id, indexes in groups.items():
        index_array = np.asarray(indexes, dtype=int)
        times = time_values[index_array]
        try:
            sort_values = times.astype(float)
        except (TypeError, ValueError):
            sort_values = times.astype(str)
        order = np.argsort(sort_values, kind="mergesort")
        ordered.append((sequence_id, index_array[order], times[order]))
    return ordered


def _transition_logs_for_times(
    ordered_times: np.ndarray,
    transition: np.ndarray,
    config: MarkovReliabilityConfig,
) -> np.ndarray:
    transition_count = max(len(ordered_times) - 1, 0)
    if config.time_delta_mode == MARKOV_TIME_MODE_ORDER_ONLY:
        return np.broadcast_to(
            np.log(transition),
            (transition_count, 2, 2),
        ).copy()
    try:
        numeric_times = np.asarray(ordered_times, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "integer-step time deltas require numeric time_values"
        ) from error
    if not np.all(np.isfinite(numeric_times)):
        raise ValueError("integer-step time_values must be finite")
    deltas = np.diff(numeric_times)
    if np.any(deltas <= 0.0):
        raise ValueError(
            "integer-step time_values must be strictly increasing per sequence"
        )
    scaled = deltas / config.time_step
    steps = np.rint(scaled).astype(np.int64)
    if np.any(steps < 1) or not np.allclose(
        scaled,
        steps,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "time gaps must be positive integer multiples of time_step"
        )
    logs = np.empty((transition_count, 2, 2), dtype=np.float64)
    for index, step_count in enumerate(steps):
        powered = np.linalg.matrix_power(transition, int(step_count))
        powered /= np.sum(powered, axis=1, keepdims=True)
        logs[index] = np.log(powered)
    return logs


def _markov_parameters(
    config: MarkovReliabilityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    transition = np.array(
        [
            [config.outlier_persistence, 1.0 - config.outlier_persistence],
            [1.0 - config.inlier_persistence, config.inlier_persistence],
        ],
        dtype=float,
    )
    stationary_inlier = (1.0 - config.outlier_persistence) / (
        2.0 - config.inlier_persistence - config.outlier_persistence
    )
    log_initial = np.log([1.0 - stationary_inlier, stationary_inlier])
    return transition, log_initial


def smooth_markov_reliability(
    prior_reliability: np.ndarray,
    log_inlier_density: np.ndarray,
    log_outlier_density: np.ndarray,
    sequence_ids: Sequence[str | int],
    time_values: Sequence[str | int | float],
    *,
    config: MarkovReliabilityConfig | None = None,
) -> MarkovReliabilityResult:
    """Infer persistent inlier states independently for each tracked sequence.

    Cue-derived reliability contributes a unary Bernoulli potential and the
    Markov transition contributes temporal persistence. Sequence evidence is
    normalized by the cue-only partition function, yielding ``p(y | cues)``
    rather than an unnormalized factor-graph score.
    """

    cfg = config or MarkovReliabilityConfig()
    _validate_config(cfg)
    prior = np.asarray(prior_reliability, dtype=float)
    log_inlier = np.asarray(log_inlier_density, dtype=float)
    log_outlier = np.asarray(log_outlier_density, dtype=float)
    ids = np.asarray(sequence_ids)
    times = np.asarray(time_values)
    n = prior.size
    for name, values in (
        ("log_inlier_density", log_inlier),
        ("log_outlier_density", log_outlier),
        ("sequence_ids", ids),
        ("time_values", times),
    ):
        if values.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {values.shape}")
    if n == 0:
        raise ValueError("at least one measurement is required")
    if not np.all(np.isfinite(prior)):
        raise ValueError("prior_reliability must contain finite values")
    if not np.all(np.isfinite(log_inlier)) or not np.all(np.isfinite(log_outlier)):
        raise ValueError("log densities must contain finite values")

    prior = np.clip(prior, cfg.probability_floor, 1.0 - cfg.probability_floor)
    transition, log_initial = _markov_parameters(cfg)
    cue_unary = np.column_stack([np.log1p(-prior), np.log(prior)])
    density_unary = np.column_stack([log_outlier, log_inlier])
    joint_unary = cue_unary + density_unary
    posterior = np.empty(n, dtype=float)
    sequence_log_evidence: dict[str, float] = {}

    for sequence_id, indexes, ordered_times in _ordered_group_indices(ids, times):
        transition_logs = _transition_logs_for_times(
            ordered_times,
            transition,
            cfg,
        )
        sequence_joint = joint_unary[indexes]
        sequence_cues = cue_unary[indexes]
        alpha, joint_log_partition = _forward(
            sequence_joint,
            log_initial,
            transition_logs,
        )
        beta = _backward(sequence_joint, transition_logs)
        _, cue_log_partition = _forward(
            sequence_cues,
            log_initial,
            transition_logs,
        )
        log_marginal = alpha + beta - joint_log_partition
        posterior[indexes] = np.exp(log_marginal[:, 1])
        sequence_log_evidence[sequence_id] = joint_log_partition - cue_log_partition

    result_posterior = np.clip(posterior, 0.0, 1.0)
    result_posterior.setflags(write=False)
    return MarkovReliabilityResult(
        posterior_inlier_probability=result_posterior,
        sequence_log_evidence=sequence_log_evidence,
    )


def markov_log_evidence_batch(
    prior_reliability: np.ndarray,
    log_inlier_density: np.ndarray,
    log_outlier_density: np.ndarray,
    sequence_ids: Sequence[str | int],
    time_values: Sequence[str | int | float],
    *,
    config: MarkovReliabilityConfig | None = None,
) -> np.ndarray:
    """Compute normalized Markov log evidence for many predictions at once.

    The density arrays have shape ``(p, n)`` for ``p`` candidate physical
    parameter settings and ``n`` pseudo-measurements. This vectorized path is
    used by grid/particle parameter inference.
    """

    cfg = config or MarkovReliabilityConfig()
    _validate_config(cfg)
    prior = np.asarray(prior_reliability, dtype=float)
    log_inlier = np.asarray(log_inlier_density, dtype=float)
    log_outlier = np.asarray(log_outlier_density, dtype=float)
    ids = np.asarray(sequence_ids)
    times = np.asarray(time_values)
    if log_inlier.ndim != 2 or log_outlier.shape != log_inlier.shape:
        raise ValueError("log density arrays must have equal shape (p, n)")
    particle_count, measurement_count = log_inlier.shape
    for name, values in (
        ("prior_reliability", prior),
        ("sequence_ids", ids),
        ("time_values", times),
    ):
        if values.shape != (measurement_count,):
            raise ValueError(
                f"{name} must have shape ({measurement_count},), got {values.shape}"
            )
    if particle_count == 0 or measurement_count == 0:
        raise ValueError("at least one particle and measurement are required")
    if not np.all(np.isfinite(prior)):
        raise ValueError("prior_reliability must contain finite values")
    if not np.all(np.isfinite(log_inlier)) or not np.all(np.isfinite(log_outlier)):
        raise ValueError("log densities must contain finite values")

    prior = np.clip(prior, cfg.probability_floor, 1.0 - cfg.probability_floor)
    transition, log_initial = _markov_parameters(cfg)
    cue_unary = np.column_stack([np.log1p(-prior), np.log(prior)])
    evidence = np.zeros(particle_count, dtype=float)

    for _, indexes, ordered_times in _ordered_group_indices(ids, times):
        transition_logs = _transition_logs_for_times(
            ordered_times,
            transition,
            cfg,
        )
        cue_sequence = cue_unary[indexes]
        _, cue_log_partition = _forward(
            cue_sequence,
            log_initial,
            transition_logs,
        )

        alpha = np.empty((particle_count, 2), dtype=float)
        alpha[:, 0] = (
            log_initial[0] + cue_sequence[0, 0] + log_outlier[:, indexes[0]]
        )
        alpha[:, 1] = log_initial[1] + cue_sequence[0, 1] + log_inlier[:, indexes[0]]
        for offset in range(1, indexes.size):
            index = indexes[offset]
            current_transition = transition_logs[offset - 1]
            next_alpha = np.empty_like(alpha)
            next_alpha[:, 0] = (
                cue_sequence[offset, 0]
                + log_outlier[:, index]
                + np.logaddexp(
                    alpha[:, 0] + current_transition[0, 0],
                    alpha[:, 1] + current_transition[1, 0],
                )
            )
            next_alpha[:, 1] = (
                cue_sequence[offset, 1]
                + log_inlier[:, index]
                + np.logaddexp(
                    alpha[:, 0] + current_transition[0, 1],
                    alpha[:, 1] + current_transition[1, 1],
                )
            )
            alpha = next_alpha
        evidence += np.logaddexp(alpha[:, 0], alpha[:, 1]) - cue_log_partition

    evidence.setflags(write=False)
    return evidence
