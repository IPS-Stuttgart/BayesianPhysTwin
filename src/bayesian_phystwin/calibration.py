"""Calibration metrics for reliability and inferred inlier probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _finite_unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    scalar = float(raw.item())
    if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return scalar


@dataclass(frozen=True)
class BinaryCalibrationMetrics:
    count: int
    positive_rate: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    roc_auc: float | None

    def __post_init__(self) -> None:
        raw_count = self.count
        if (
            isinstance(raw_count, (bool, np.bool_))
            or not isinstance(raw_count, (int, np.integer))
            or raw_count < 1
        ):
            raise ValueError("count must be a positive integer")
        positive_rate = _finite_unit_interval(
            self.positive_rate,
            name="positive_rate",
        )
        brier_score = _finite_unit_interval(self.brier_score, name="brier_score")
        expected_calibration_error = _finite_unit_interval(
            self.expected_calibration_error,
            name="expected_calibration_error",
        )
        raw_log_loss = np.asarray(self.log_loss)
        if (
            raw_log_loss.shape != ()
            or raw_log_loss.dtype.kind not in "iuf"
            or not np.isfinite(float(raw_log_loss.item()))
            or float(raw_log_loss.item()) < 0.0
        ):
            raise ValueError("log_loss must be finite and nonnegative")
        roc_auc = (
            None
            if self.roc_auc is None
            else _finite_unit_interval(self.roc_auc, name="roc_auc")
        )

        object.__setattr__(self, "count", int(raw_count))
        object.__setattr__(self, "positive_rate", positive_rate)
        object.__setattr__(self, "brier_score", brier_score)
        object.__setattr__(self, "log_loss", float(raw_log_loss.item()))
        object.__setattr__(
            self,
            "expected_calibration_error",
            expected_calibration_error,
        )
        object.__setattr__(self, "roc_auc", roc_auc)

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _binary_target(value: np.ndarray, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != expected_shape:
        raise ValueError("probability and target must have equal shape")
    if raw.dtype.kind == "b":
        return np.array(raw, dtype=bool, copy=True, order="C")
    if raw.dtype.kind not in "iuf":
        raise ValueError("target must contain booleans or exact 0/1 values")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError("target must contain booleans or exact 0/1 values")
    return numeric.astype(bool)


def _roc_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    positives = int(np.sum(target))
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks: np.ndarray = np.empty(probability.size, dtype=float)
    start = 0
    while start < probability.size:
        end = start + 1
        while (
            end < probability.size
            and sorted_probability[end] == sorted_probability[start]
        ):
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

    raw_probability = np.asarray(probability)
    if raw_probability.dtype.kind not in "iuf":
        raise ValueError("probability must contain real numeric values")
    probability_array = np.asarray(raw_probability, dtype=np.float64)
    if probability_array.ndim != 1:
        raise ValueError("probability and target must be one-dimensional")
    target_array = _binary_target(target, expected_shape=probability_array.shape)
    if probability_array.size == 0:
        raise ValueError("at least one probability is required")
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probability must contain finite values")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probability must lie in [0, 1]")
    if (
        isinstance(n_bins, (bool, np.bool_))
        or not isinstance(n_bins, (int, np.integer))
        or n_bins <= 0
    ):
        raise ValueError("n_bins must be a positive integer")
    bin_count = int(n_bins)

    target_float: np.ndarray = target_array.astype(float)
    clipped = np.clip(probability_array, 1e-12, 1.0 - 1e-12)
    log_loss = -np.mean(
        target_float * np.log(clipped) + (1.0 - target_float) * np.log1p(-clipped)
    )

    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_index = np.minimum(
        np.digitize(probability_array, edges[1:-1]),
        bin_count - 1,
    )
    ece = 0.0
    for index in range(bin_count):
        selected = bin_index == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(probability_array[selected]))
                - float(np.mean(target_float[selected]))
            )

    return BinaryCalibrationMetrics(
        count=int(probability_array.size),
        positive_rate=float(np.mean(target_float)),
        brier_score=float(np.mean(np.square(probability_array - target_float))),
        log_loss=float(log_loss),
        expected_calibration_error=float(ece),
        roc_auc=_roc_auc(probability_array, target_array),
    )
