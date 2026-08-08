"""Group-balanced split-conformal bounds for independent experimental units.

The calibration unit is one physical object or independent acquisition session.
Each unit contributes exactly one maximum nonconformity score, so long sequences
cannot dominate the calibration quantile merely because they contain more frames.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .calibration import finite_group_conformal_rank

ConformalScore = Literal["scaled", "additive"]


def _score_name(value: str) -> ConformalScore:
    if value == "scaled":
        return "scaled"
    if value == "additive":
        return "additive"
    raise ValueError("score must be 'scaled' or 'additive'")


def _point_nonconformity(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    score: ConformalScore,
    name: str,
) -> np.ndarray:
    observed = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError(f"{name} target and prediction must have equal shape")
    if observed.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
        raise ValueError(f"{name} target must be finite and nonnegative")
    if not np.all(np.isfinite(predicted)):
        raise ValueError(f"{name} prediction must be finite")
    if score == "scaled":
        if np.any(predicted <= 0.0):
            raise ValueError(f"{name} scaled prediction must be positive")
        return observed / predicted
    return observed - predicted


def group_max_nonconformity_scores(
    calibration_targets: Sequence[np.ndarray],
    calibration_predictions: Sequence[np.ndarray],
    *,
    score: ConformalScore,
) -> np.ndarray:
    """Return one maximum nonconformity score per independent calibration group.

    A group can contain any number or shape of registered endpoints. The maximum
    reduction targets simultaneous coverage of all endpoints in one future group.
    """

    method = _score_name(score)
    targets = tuple(calibration_targets)
    predictions = tuple(calibration_predictions)
    if len(targets) != len(predictions):
        raise ValueError(
            "calibration_targets and calibration_predictions must contain "
            "the same number of groups"
        )
    if not targets:
        raise ValueError("at least one calibration group is required")

    group_scores: np.ndarray = np.empty(len(targets), dtype=np.float64)
    pairs = zip(targets, predictions, strict=True)
    for index, (target, prediction) in enumerate(pairs):
        point_scores = _point_nonconformity(
            target,
            prediction,
            score=method,
            name=f"calibration group {index}",
        )
        group_scores[index] = float(np.max(point_scores))
    group_scores.setflags(write=False)
    return group_scores


def finite_group_conformal_quantile(
    group_scores: np.ndarray,
    coverage: float,
) -> tuple[float, int]:
    """Return the conservative split-conformal quantile over independent groups."""

    scores = np.asarray(group_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("at least one calibration group score is required")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration group scores must be finite")
    nominal = float(coverage)
    if not 0.0 < nominal < 1.0:
        raise ValueError("coverage must lie in (0, 1)")

    rank = finite_group_conformal_rank(len(scores), nominal)
    if rank > len(scores):
        return math.inf, rank
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


@dataclass(frozen=True, slots=True)
class GroupedConformalResult:
    """Simultaneous upper bounds calibrated over independent groups."""

    upper_bound: np.ndarray
    calibration_group_scores: np.ndarray
    quantile: float
    finite_sample_rank: int
    calibration_group_count: int
    nominal_coverage: float
    score: ConformalScore

    def __post_init__(self) -> None:
        upper = np.array(self.upper_bound, dtype=np.float64, copy=True, order="C")
        scores = np.array(
            self.calibration_group_scores,
            dtype=np.float64,
            copy=True,
            order="C",
        ).reshape(-1)
        if upper.size == 0:
            raise ValueError("upper_bound cannot be empty")
        if np.any(np.isnan(upper)) or np.any(upper < 0.0):
            raise ValueError("upper_bound must be nonnegative and cannot contain NaN")
        raw_count = self.calibration_group_count
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, (int, np.integer))
            or raw_count < 1
        ):
            raise ValueError("calibration_group_count must be a positive integer")
        count = int(raw_count)
        if scores.size != count:
            raise ValueError(
                "calibration_group_count must match the nonempty group-score vector"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("calibration_group_scores must be finite")
        raw_rank = self.finite_sample_rank
        if (
            isinstance(raw_rank, bool)
            or not isinstance(raw_rank, (int, np.integer))
            or raw_rank < 1
        ):
            raise ValueError("finite_sample_rank must be a positive integer")
        rank = int(raw_rank)
        nominal = float(self.nominal_coverage)
        if not 0.0 < nominal < 1.0:
            raise ValueError("nominal_coverage must lie in (0, 1)")
        method = _score_name(self.score)
        quantile = float(self.quantile)
        if rank > count:
            if quantile != math.inf:
                raise ValueError(
                    "an impossible finite-sample rank requires infinite quantile"
                )
        elif not math.isfinite(quantile):
            raise ValueError("a feasible finite-sample rank requires finite quantile")

        upper.setflags(write=False)
        scores.setflags(write=False)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "calibration_group_scores", scores)
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "finite_sample_rank", rank)
        object.__setattr__(self, "calibration_group_count", count)
        object.__setattr__(self, "nominal_coverage", nominal)
        object.__setattr__(self, "score", method)


def grouped_conformal_upper_bounds(
    calibration_targets: Sequence[np.ndarray],
    calibration_predictions: Sequence[np.ndarray],
    future_prediction: np.ndarray,
    *,
    coverage: float,
    score: ConformalScore,
) -> GroupedConformalResult:
    """Calibrate simultaneous upper bounds from one score per independent group.

    Under exchangeability of the calibration groups and the future group, the
    maximum within-group nonconformity construction gives marginal simultaneous
    coverage for every registered endpoint in the future group. The predictor,
    score, grouping rule, and endpoint set must be frozen before target outcomes
    are opened.
    """

    method = _score_name(score)
    group_scores = group_max_nonconformity_scores(
        calibration_targets,
        calibration_predictions,
        score=method,
    )
    quantile, rank = finite_group_conformal_quantile(group_scores, coverage)

    future = np.asarray(future_prediction, dtype=np.float64)
    if future.size == 0:
        raise ValueError("future_prediction cannot be empty")
    if not np.all(np.isfinite(future)):
        raise ValueError("future_prediction must be finite")
    if method == "scaled" and np.any(future <= 0.0):
        raise ValueError("scaled future_prediction must be positive")

    if math.isinf(quantile):
        upper = np.full(future.shape, math.inf, dtype=np.float64)
    elif method == "scaled":
        upper = quantile * future
    else:
        upper = future + quantile

    return GroupedConformalResult(
        upper_bound=np.maximum(upper, 0.0),
        calibration_group_scores=group_scores,
        quantile=quantile,
        finite_sample_rank=rank,
        calibration_group_count=len(group_scores),
        nominal_coverage=coverage,
        score=method,
    )


__all__ = [
    "ConformalScore",
    "GroupedConformalResult",
    "finite_group_conformal_quantile",
    "group_max_nonconformity_scores",
    "grouped_conformal_upper_bounds",
]
