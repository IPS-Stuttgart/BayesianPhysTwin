"""Matched joint/marginal action-regret bounds with held-out calibration.

These are standard weighted quantiles and split-conformal order statistics,
not a new coverage theorem. Simultaneous action coverage requires exchangeable
complete calibration episodes and a frozen predictor/action bank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def normalized_weights(weights: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    total = float(values.sum())
    if (
        values.shape != (count,)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError("invalid particle weights")
    return values / total


def bias_marginalized_weights(
    observations: np.ndarray,
    predictions: np.ndarray,
    *,
    noise_std_m: float,
    shared_bias_std_m: float,
    prior_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Marginalize one shared xyz sensor offset, not one offset per point-time."""
    observed = np.asarray(observations, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if (
        observed.ndim != 3
        or observed.shape[-1] != 3
        or observed.size == 0
        or predicted.ndim != 4
        or predicted.shape[1:] != observed.shape
        or predicted.shape[0] < 1
    ):
        raise ValueError("expected prefix observations and matching particle prefixes")
    if not all(np.isfinite(x).all() for x in (observed, predicted)):
        raise ValueError("prefix observations must be finite")
    if (
        not np.isfinite(noise_std_m)
        or noise_std_m <= 0
        or not np.isfinite(shared_bias_std_m)
        or shared_bias_std_m < 0
    ):
        raise ValueError("invalid metric sensor noise")
    count = len(predicted)
    prior = normalized_weights(
        np.ones(count) if prior_weights is None else prior_weights,
        count,
    )
    residual = (observed[None] - predicted).reshape(count, -1, 3)
    mean = residual.mean(axis=1)
    centered = residual - mean[:, None]
    noise_var = noise_std_m**2
    shared_var = shared_bias_std_m**2
    n = residual.shape[1]
    # Orthogonal decomposition of sigma^2 I + tau^2 11^T avoids cancellation.
    distance = np.sum(centered**2, axis=(1, 2)) / noise_var
    distance += n * np.sum(mean**2, axis=1) / (noise_var + n * shared_var)
    with np.errstate(divide="ignore"):
        log_weight = np.log(prior) - 0.5 * distance
    peak = np.max(log_weight)
    if not np.isfinite(peak):
        raise ValueError("invalid likelihood weights")
    return normalized_weights(np.exp(log_weight - peak), count)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("quantile values must be a nonempty finite vector")
    if not np.isfinite(level) or not 0 < level <= 1:
        raise ValueError("invalid quantile level")
    weight = normalized_weights(weights, len(value))
    order = np.argsort(value, kind="stable")
    index = min(int(np.searchsorted(np.cumsum(weight[order]), level)), len(value) - 1)
    return float(value[order[index]])


def action_regret_upper(
    particle_losses: np.ndarray,
    weights: np.ndarray,
    *,
    level: float = 0.90,
    coupling: str = "joint",
) -> np.ndarray:
    """Use the same action marginals; change only their coupling to action zero."""
    losses = np.asarray(particle_losses, dtype=np.float64)
    if losses.ndim != 2 or min(losses.shape) < 1 or not np.isfinite(losses).all():
        raise ValueError("losses must be finite particle-by-action values")
    weight = normalized_weights(weights, len(losses))
    if coupling not in ("joint", "independent"):
        raise ValueError("unknown action coupling")
    result = np.zeros(losses.shape[1], dtype=np.float64)
    for action in range(1, losses.shape[1]):
        if coupling == "joint":
            delta = losses[:, action] - losses[:, 0]
            delta_weights = weight
        else:
            delta = (losses[:, action, None] - losses[None, :, 0]).ravel()
            delta_weights = (weight[:, None] * weight[None]).ravel()
        result[action] = weighted_quantile(delta, delta_weights, level)
    return result


@dataclass(frozen=True)
class RegretCalibration:
    coverage: float
    count: int
    rank: int
    offset: float | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.coverage) or not 0 < self.coverage < 1:
            raise ValueError("invalid calibration coverage")
        if type(self.count) is not int or self.count < 0:
            raise ValueError("invalid calibration count")
        if type(self.rank) is not int or self.rank != math.ceil(
            (self.count + 1) * self.coverage
        ):
            raise ValueError("invalid calibration rank")
        if self.rank > self.count:
            if self.offset is not None:
                raise ValueError("insufficient support must force fallback")
        elif self.offset is None or not np.isfinite(self.offset) or self.offset < 0:
            raise ValueError("invalid calibration offset")


def calibrate_simultaneous_regret(
    predicted_upper: np.ndarray,
    realized_losses: np.ndarray,
    *,
    coverage: float = 0.90,
) -> RegretCalibration:
    """One maximum-over-actions nonconformity score per complete episode."""
    upper = np.asarray(predicted_upper, dtype=np.float64)
    losses = np.asarray(realized_losses, dtype=np.float64)
    if upper.ndim != 2 or upper.shape != losses.shape or upper.shape[1] < 2:
        raise ValueError("calibration requires matching episode-by-action matrices")
    if not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("invalid calibration coverage")
    if not np.isfinite(upper).all() or not np.isfinite(losses).all():
        raise ValueError("incomplete episodes cannot be silently dropped")
    if np.any(upper[:, 0] != 0):
        raise ValueError("baseline regret must be exactly zero")
    count = len(upper)
    rank = math.ceil((count + 1) * coverage)
    if rank > count:
        return RegretCalibration(coverage, count, rank, None)
    regret = losses[:, 1:] - losses[:, :1]
    score = np.max(regret - upper[:, 1:], axis=1)
    offset = max(0.0, float(np.sort(score)[rank - 1]))
    return RegretCalibration(coverage, count, rank, offset)


def guarded_action(
    expected_losses: np.ndarray,
    raw_upper: np.ndarray,
    calibration: RegretCalibration,
) -> int:
    """Minimize predicted cost only among strictly nonworsening admitted actions."""
    means = np.asarray(expected_losses, dtype=np.float64)
    upper = np.asarray(raw_upper, dtype=np.float64)
    if means.ndim != 1 or upper.shape != means.shape or len(means) < 2:
        raise ValueError("invalid action scores")
    if not np.isfinite(means).all() or not np.isfinite(upper).all() or upper[0] != 0:
        raise ValueError("action scores must be finite with exact baseline regret")
    if calibration.offset is None:
        return 0
    if not np.isfinite(calibration.offset) or calibration.offset < 0:
        raise ValueError("invalid calibration offset")
    allowed = upper + calibration.offset < 0
    allowed[0] = True
    return int(np.argmin(np.where(allowed, means, np.inf)))


def selected_commands(actions: tuple[np.ndarray, ...], index: int) -> np.ndarray:
    """Return the original command carrier, including exact baseline fallback."""
    if type(index) is not int or not 0 <= index < len(actions):
        raise ValueError("invalid action index")
    return actions[index]
