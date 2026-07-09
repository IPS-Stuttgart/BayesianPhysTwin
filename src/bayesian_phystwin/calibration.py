"""Calibration metrics for reliability and inferred inlier probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryCalibrationMetrics:
    count: int
    positive_rate: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    roc_auc: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _roc_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    positives = int(np.sum(target))
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(probability.size, dtype=float)
    start = 0
    while start < probability.size:
        end = start + 1
        while end < probability.size and sorted_probability[end] == sorted_probability[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    positive_rank_sum = float(np.sum(ranks[target]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def binary_calibration_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    n_bins: int = 10,
) -> BinaryCalibrationMetrics:
    """Compute proper scores, ECE, and AUROC for binary inlier labels."""

    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=bool)
    if probability.ndim != 1 or target.shape != probability.shape:
        raise ValueError("probability and target must be one-dimensional with equal shape")
    if probability.size == 0:
        raise ValueError("at least one probability is required")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability must contain finite values")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    probability = np.clip(probability, 0.0, 1.0)
    target_float = target.astype(float)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    log_loss = -np.mean(
        target_float * np.log(clipped) + (1.0 - target_float) * np.log1p(-clipped)
    )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(np.digitize(probability, edges[1:-1]), n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        selected = bin_index == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected])) - float(np.mean(target_float[selected]))
            )

    return BinaryCalibrationMetrics(
        count=int(probability.size),
        positive_rate=float(np.mean(target_float)),
        brier_score=float(np.mean(np.square(probability - target_float))),
        log_loss=float(log_loss),
        expected_calibration_error=float(ece),
        roc_auc=_roc_auc(probability, target),
    )
