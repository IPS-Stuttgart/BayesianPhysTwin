"""Temporally structured reliability for tracked pseudo-measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MarkovReliabilityConfig:
    """Persistence parameters for binary inlier/outlier track states."""

    inlier_persistence: float = 0.98
    outlier_persistence: float = 0.90
    probability_floor: float = 1e-6


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


def _forward(
    unary_log_potential: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[np.ndarray, float]:
    length = unary_log_potential.shape[0]
    alpha = np.empty((length, 2), dtype=float)
    alpha[0] = log_initial + unary_log_potential[0]
    for time in range(1, length):
        alpha[time, 0] = unary_log_potential[time, 0] + np.logaddexp(
            alpha[time - 1, 0] + log_transition[0, 0],
            alpha[time - 1, 1] + log_transition[1, 0],
        )
        alpha[time, 1] = unary_log_potential[time, 1] + np.logaddexp(
            alpha[time - 1, 0] + log_transition[0, 1],
            alpha[time - 1, 1] + log_transition[1, 1],
        )
    return alpha, float(np.logaddexp(alpha[-1, 0], alpha[-1, 1]))


def _backward(
    unary_log_potential: np.ndarray,
    log_transition: np.ndarray,
) -> np.ndarray:
    length = unary_log_potential.shape[0]
    beta = np.zeros((length, 2), dtype=float)
    for time in range(length - 2, -1, -1):
        beta[time, 0] = np.logaddexp(
            log_transition[0, 0]
            + unary_log_potential[time + 1, 0]
            + beta[time + 1, 0],
            log_transition[0, 1]
            + unary_log_potential[time + 1, 1]
            + beta[time + 1, 1],
        )
        beta[time, 1] = np.logaddexp(
            log_transition[1, 0]
            + unary_log_potential[time + 1, 0]
            + beta[time + 1, 0],
            log_transition[1, 1]
            + unary_log_potential[time + 1, 1]
            + beta[time + 1, 1],
        )
    return beta


def _ordered_group_indices(
    sequence_ids: np.ndarray,
    time_values: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    groups: dict[str, list[int]] = {}
    for index, sequence_id in enumerate(sequence_ids):
        groups.setdefault(str(sequence_id), []).append(index)

    ordered: list[tuple[str, np.ndarray]] = []
    for sequence_id, indexes in groups.items():
        index_array = np.asarray(indexes, dtype=int)
        times = time_values[index_array]
        try:
            sort_values = times.astype(float)
        except (TypeError, ValueError):
            sort_values = times.astype(str)
        order = np.argsort(sort_values, kind="mergesort")
        ordered.append((sequence_id, index_array[order]))
    return ordered


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
    transition = np.array(
        [
            [cfg.outlier_persistence, 1.0 - cfg.outlier_persistence],
            [1.0 - cfg.inlier_persistence, cfg.inlier_persistence],
        ],
        dtype=float,
    )
    log_transition = np.log(transition)
    stationary_inlier = (1.0 - cfg.outlier_persistence) / (
        2.0 - cfg.inlier_persistence - cfg.outlier_persistence
    )
    log_initial = np.log([1.0 - stationary_inlier, stationary_inlier])

    cue_unary = np.column_stack([np.log1p(-prior), np.log(prior)])
    density_unary = np.column_stack([log_outlier, log_inlier])
    joint_unary = cue_unary + density_unary
    posterior = np.empty(n, dtype=float)
    sequence_log_evidence: dict[str, float] = {}

    for sequence_id, indexes in _ordered_group_indices(ids, times):
        sequence_joint = joint_unary[indexes]
        sequence_cues = cue_unary[indexes]
        alpha, joint_log_partition = _forward(
            sequence_joint,
            log_initial,
            log_transition,
        )
        beta = _backward(sequence_joint, log_transition)
        _, cue_log_partition = _forward(sequence_cues, log_initial, log_transition)
        log_marginal = alpha + beta - joint_log_partition
        posterior[indexes] = np.exp(log_marginal[:, 1])
        sequence_log_evidence[sequence_id] = joint_log_partition - cue_log_partition

    return MarkovReliabilityResult(
        posterior_inlier_probability=np.clip(posterior, 0.0, 1.0),
        sequence_log_evidence=sequence_log_evidence,
    )
