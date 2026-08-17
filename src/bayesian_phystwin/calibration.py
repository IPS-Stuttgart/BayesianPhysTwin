"""Calibration metrics and finite-group calibration design utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

CalibrationPooling = Literal["pooled", "stratum"]


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


def _finite_open_unit_interval(value: object, *, name: str) -> float:
    scalar = _finite_unit_interval(value, name=name)
    if scalar <= 0.0 or scalar >= 1.0:
        raise ValueError(f"{name} must be a finite number in (0, 1)")
    return scalar


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _coverage_fraction(coverage: object) -> tuple[float, Fraction]:
    nominal = _finite_open_unit_interval(coverage, name="coverage")
    return nominal, Fraction(str(nominal))


def finite_group_conformal_rank(
    calibration_group_count: int,
    coverage: float,
) -> int:
    """Return ``ceil((n + 1) * coverage)`` without float-boundary drift.

    The count is the number of independent calibration units, not the number of
    frames, views, tracks, points, or taxels. Decimal-exact arithmetic preserves
    ordinary nominal values such as ``0.9``. A coverage value that is exactly the
    binary-float representation of ``k / (n + 1)`` is also treated as that rank
    boundary, so values returned by :func:`maximum_finite_group_coverage` round
    trip without being spuriously promoted by one rank.
    """

    count = _positive_integer(
        calibration_group_count,
        name="calibration_group_count",
    )
    nominal, exact_nominal = _coverage_fraction(coverage)
    boundary_rank = int(round((count + 1) * nominal))
    if 1 <= boundary_rank <= count + 1 and nominal == boundary_rank / (count + 1):
        return boundary_rank
    numerator = (count + 1) * exact_nominal.numerator
    return (numerator + exact_nominal.denominator - 1) // exact_nominal.denominator


def maximum_finite_group_coverage(calibration_group_count: int) -> float:
    """Return the largest nominal coverage with a finite split-conformal rank."""

    count = _positive_integer(
        calibration_group_count,
        name="calibration_group_count",
    )
    return count / (count + 1)


def minimum_groups_for_finite_conformal(coverage: float) -> int:
    """Return the minimum independent-group count permitting a finite quantile."""

    nominal, exact_nominal = _coverage_fraction(coverage)
    remaining = exact_nominal.denominator - exact_nominal.numerator
    upper = (exact_nominal.numerator + remaining - 1) // remaining

    # The decimal-exact result is an upper bound. Search against the canonical
    # rank function so recurring rational boundaries represented as floats (for
    # example 10 / 11) remain consistent with finite_group_conformal_rank().
    lower = 1
    while lower < upper:
        midpoint = (lower + upper) // 2
        if finite_group_conformal_rank(midpoint, nominal) <= midpoint:
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


@dataclass(frozen=True)
class FiniteGroupCalibrationDesign:
    """Fail-closed design record for a split-conformal calibration stage.

    The predictor, score, grouping rule, endpoint set, and any acceptance guard
    must be frozen before the interval-calibration outcomes are inspected.
    Calibration outcomes may not also select the deployed predictor under this
    split-conformal contract. More elaborate CV+/jackknife+ procedures require
    a separately versioned design and are intentionally not represented here.
    """

    calibration_group_count: int
    nominal_coverage: float
    finite_sample_rank: int
    maximum_finite_coverage: float
    pooling: CalibrationPooling
    predictor_frozen_before_scores: bool
    calibration_outcomes_used_for_selection: bool

    def __post_init__(self) -> None:
        count = _positive_integer(
            self.calibration_group_count,
            name="calibration_group_count",
        )
        nominal, _ = _coverage_fraction(self.nominal_coverage)
        expected_rank = finite_group_conformal_rank(count, nominal)
        rank = _positive_integer(
            self.finite_sample_rank,
            name="finite_sample_rank",
        )
        if rank != expected_rank:
            raise ValueError(
                "finite_sample_rank must equal ceil((group_count + 1) * coverage)"
            )
        expected_maximum = maximum_finite_group_coverage(count)
        maximum = _finite_open_unit_interval(
            self.maximum_finite_coverage,
            name="maximum_finite_coverage",
        )
        if maximum != expected_maximum:
            raise ValueError(
                "maximum_finite_coverage must equal group_count / (group_count + 1)"
            )
        if self.pooling not in {"pooled", "stratum"}:
            raise ValueError("pooling must be 'pooled' or 'stratum'")
        if type(self.predictor_frozen_before_scores) is not bool:
            raise ValueError("predictor_frozen_before_scores must be a boolean")
        if type(self.calibration_outcomes_used_for_selection) is not bool:
            raise ValueError(
                "calibration_outcomes_used_for_selection must be a boolean"
            )
        if not self.predictor_frozen_before_scores:
            raise ValueError(
                "split-conformal calibration requires the deployed predictor, "
                "score, guard, grouping rule, and endpoint set to be frozen "
                "before calibration scores are inspected"
            )
        if self.calibration_outcomes_used_for_selection:
            raise ValueError(
                "split-conformal calibration outcomes cannot also select the "
                "deployed predictor or guard"
            )
        if rank > count:
            minimum = minimum_groups_for_finite_conformal(nominal)
            raise ValueError(
                f"{nominal:.12g} coverage requires an infinite quantile with "
                f"{count} independent groups; use at least {minimum} groups or "
                f"coverage <= {expected_maximum:.12g}"
            )

        object.__setattr__(self, "calibration_group_count", count)
        object.__setattr__(self, "nominal_coverage", nominal)
        object.__setattr__(self, "finite_sample_rank", rank)
        object.__setattr__(self, "maximum_finite_coverage", maximum)

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


def plan_finite_group_calibration(
    calibration_group_count: int,
    coverage: float,
    *,
    pooling: CalibrationPooling = "pooled",
    predictor_frozen_before_scores: bool,
    calibration_outcomes_used_for_selection: bool,
) -> FiniteGroupCalibrationDesign:
    """Build a valid finite split-conformal design or fail before target access."""

    count = _positive_integer(
        calibration_group_count,
        name="calibration_group_count",
    )
    nominal, _ = _coverage_fraction(coverage)
    return FiniteGroupCalibrationDesign(
        calibration_group_count=count,
        nominal_coverage=nominal,
        finite_sample_rank=finite_group_conformal_rank(count, nominal),
        maximum_finite_coverage=maximum_finite_group_coverage(count),
        pooling=pooling,
        predictor_frozen_before_scores=predictor_frozen_before_scores,
        calibration_outcomes_used_for_selection=(
            calibration_outcomes_used_for_selection
        ),
    )


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
