"""Decision-quality diagnostics for guarded Bayesian-PhysTwin updates.

The functions in this module compare candidate losses against the exact physical
fallback used on rejection. They also support matched method comparisons,
conditional analyses, prediction-interval calibration by horizon, and
cluster-bootstrap uncertainty without treating correlated rows as independent.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class GuardEvaluation:
    """Point diagnostics for one fixed accept/fallback decision."""

    observation_count: int
    accepted_count: int
    coverage: float
    fallback_rate: float
    baseline_mean_loss: float
    candidate_mean_loss: float
    selected_mean_loss: float
    selected_mean_excess_loss: float
    high_quantile_level: float
    accepted_mean_excess_loss: float | None
    accepted_high_quantile_excess_loss: float | None
    harmful_accepted_rate: float | None
    worst_accepted_excess_loss: float | None


@dataclass(frozen=True)
class SelectiveRiskPoint:
    """Guard diagnostics at one score threshold.

    ``threshold=None`` denotes the exact zero-coverage fallback endpoint.
    """

    threshold: float | None
    higher_is_safer: bool
    evaluation: GuardEvaluation


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile interval with explicit finite-replicate accounting."""

    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    finite_replicates: int
    bootstrap_repeats: int


@dataclass(frozen=True)
class GuardBootstrapSummary:
    """Cluster-bootstrap uncertainty for one fixed guard decision."""

    point_estimate: GuardEvaluation
    group_count: int
    confidence_level: float
    bootstrap_repeats: int
    seed: int
    coverage: BootstrapInterval
    fallback_rate: BootstrapInterval
    selected_mean_excess_loss: BootstrapInterval
    accepted_mean_excess_loss: BootstrapInterval
    accepted_high_quantile_excess_loss: BootstrapInterval
    harmful_accepted_rate: BootstrapInterval
    worst_accepted_excess_loss: BootstrapInterval


@dataclass(frozen=True)
class GuardStratumEvaluation:
    """Guard diagnostics conditioned on one declared stratum."""

    stratum: Hashable
    evaluation: GuardEvaluation


@dataclass(frozen=True)
class GuardMethodEvaluation:
    """One method evaluated with the shared fallback and loss vector."""

    method: str
    evaluation: GuardEvaluation
    selected_mean_loss_difference_from_reference: float


@dataclass(frozen=True)
class MatchedGuardEvaluation:
    """Matched evaluation of multiple guarded methods."""

    reference_method: str
    methods: tuple[GuardMethodEvaluation, ...]


@dataclass(frozen=True)
class PredictionIntervalEvaluation:
    """Calibration and sharpness diagnostics for scalar prediction intervals."""

    observation_count: int
    nominal_coverage: float
    empirical_coverage: float
    coverage_error: float
    mean_interval_width: float
    median_interval_width: float
    p90_interval_width: float
    below_interval_rate: float
    above_interval_rate: float


@dataclass(frozen=True)
class HorizonIntervalEvaluation:
    """Prediction-interval diagnostics at one declared horizon."""

    horizon: float
    evaluation: PredictionIntervalEvaluation


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array: np.ndarray = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _boolean_vector(
    values: Sequence[bool] | np.ndarray,
    *,
    name: str,
    length: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (length,) or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a Boolean vector with shape ({length},)")
    result: np.ndarray = np.asarray(array, dtype=bool).copy()
    result.setflags(write=False)
    return result


def _validated_losses(
    baseline_loss: Sequence[float] | np.ndarray,
    candidate_loss: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = _finite_vector(baseline_loss, name="baseline_loss")
    candidate = _finite_vector(candidate_loss, name="candidate_loss")
    if candidate.shape != baseline.shape:
        raise ValueError("candidate_loss must match baseline_loss")
    return baseline, candidate


def _validated_probability(value: float, *, name: str) -> float:
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return probability


def _validated_high_quantile(value: float) -> float:
    quantile = float(value)
    if not np.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("high_quantile must lie in (0, 1]")
    return quantile


def evaluate_guard(
    baseline_loss: Sequence[float] | np.ndarray,
    candidate_loss: Sequence[float] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    *,
    harmful_tolerance: float = 0.0,
    high_quantile: float = 0.95,
) -> GuardEvaluation:
    """Evaluate one fixed guard against its exact physical fallback.

    ``harmful_tolerance`` defines the smallest positive candidate-minus-baseline
    excess counted as harmful. ``high_quantile`` controls the reported upper-tail
    accepted regression. Both choices must be frozen independently of target
    outcomes when the result is used confirmatorily.
    """

    baseline, candidate = _validated_losses(baseline_loss, candidate_loss)
    decision = _boolean_vector(accepted, name="accepted", length=len(baseline))
    tolerance = float(harmful_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("harmful_tolerance must be finite and nonnegative")
    quantile = _validated_high_quantile(high_quantile)

    excess = candidate - baseline
    selected = np.where(decision, candidate, baseline)
    accepted_count = int(np.sum(decision))
    observation_count = len(baseline)
    coverage = accepted_count / observation_count

    if accepted_count:
        accepted_excess = excess[decision]
        accepted_mean_excess: float | None = float(np.mean(accepted_excess))
        accepted_high_quantile: float | None = float(
            np.quantile(accepted_excess, quantile)
        )
        harmful_rate: float | None = float(np.mean(accepted_excess > tolerance))
        worst_excess: float | None = float(np.max(accepted_excess))
    else:
        accepted_mean_excess = None
        accepted_high_quantile = None
        harmful_rate = None
        worst_excess = None

    return GuardEvaluation(
        observation_count=observation_count,
        accepted_count=accepted_count,
        coverage=float(coverage),
        fallback_rate=float(1.0 - coverage),
        baseline_mean_loss=float(np.mean(baseline)),
        candidate_mean_loss=float(np.mean(candidate)),
        selected_mean_loss=float(np.mean(selected)),
        selected_mean_excess_loss=float(np.mean(selected - baseline)),
        high_quantile_level=quantile,
        accepted_mean_excess_loss=accepted_mean_excess,
        accepted_high_quantile_excess_loss=accepted_high_quantile,
        harmful_accepted_rate=harmful_rate,
        worst_accepted_excess_loss=worst_excess,
    )


def selective_risk_curve(
    baseline_loss: Sequence[float] | np.ndarray,
    candidate_loss: Sequence[float] | np.ndarray,
    acceptance_score: Sequence[float] | np.ndarray,
    *,
    higher_is_safer: bool = True,
    harmful_tolerance: float = 0.0,
    high_quantile: float = 0.95,
    include_zero_coverage: bool = True,
) -> tuple[SelectiveRiskPoint, ...]:
    """Evaluate every distinct score threshold without splitting score ties."""

    baseline, candidate = _validated_losses(baseline_loss, candidate_loss)
    scores = _finite_vector(acceptance_score, name="acceptance_score")
    if scores.shape != baseline.shape:
        raise ValueError("acceptance_score must match baseline_loss")
    if not isinstance(higher_is_safer, bool):
        raise ValueError("higher_is_safer must be Boolean")
    if not isinstance(include_zero_coverage, bool):
        raise ValueError("include_zero_coverage must be Boolean")

    thresholds = np.unique(scores)
    if higher_is_safer:
        thresholds = thresholds[::-1]

    points: list[SelectiveRiskPoint] = []
    if include_zero_coverage:
        points.append(
            SelectiveRiskPoint(
                threshold=None,
                higher_is_safer=higher_is_safer,
                evaluation=evaluate_guard(
                    baseline,
                    candidate,
                    np.zeros(len(baseline), dtype=bool),
                    harmful_tolerance=harmful_tolerance,
                    high_quantile=high_quantile,
                ),
            )
        )
    for threshold in thresholds:
        accepted = scores >= threshold if higher_is_safer else scores <= threshold
        points.append(
            SelectiveRiskPoint(
                threshold=float(threshold),
                higher_is_safer=higher_is_safer,
                evaluation=evaluate_guard(
                    baseline,
                    candidate,
                    accepted,
                    harmful_tolerance=harmful_tolerance,
                    high_quantile=high_quantile,
                ),
            )
        )
    return tuple(points)


def _labeled_index_lists(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    length: int,
) -> tuple[tuple[Hashable, np.ndarray], ...]:
    array = np.asarray(values, dtype=object)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")

    grouped: dict[Hashable, list[int]] = {}
    for index, raw_value in enumerate(array.tolist()):
        try:
            hash(raw_value)
        except TypeError as error:
            raise ValueError(f"{name} must contain hashable values") from error
        value = cast(Hashable, raw_value)
        grouped.setdefault(value, []).append(index)
    if not grouped:
        raise ValueError(f"{name} must identify at least one group")
    return tuple(
        (label, np.asarray(indices, dtype=np.int64))
        for label, indices in grouped.items()
    )


def evaluate_guard_by_stratum(
    baseline_loss: Sequence[float] | np.ndarray,
    candidate_loss: Sequence[float] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    stratum_ids: Sequence[Any] | np.ndarray,
    *,
    harmful_tolerance: float = 0.0,
    high_quantile: float = 0.95,
) -> tuple[GuardStratumEvaluation, ...]:
    """Condition guard diagnostics on frozen horizon, reliability, or rank strata."""

    baseline, candidate = _validated_losses(baseline_loss, candidate_loss)
    decision = _boolean_vector(accepted, name="accepted", length=len(baseline))
    strata = _labeled_index_lists(
        stratum_ids,
        name="stratum_ids",
        length=len(baseline),
    )
    return tuple(
        GuardStratumEvaluation(
            stratum=label,
            evaluation=evaluate_guard(
                baseline[indices],
                candidate[indices],
                decision[indices],
                harmful_tolerance=harmful_tolerance,
                high_quantile=high_quantile,
            ),
        )
        for label, indices in strata
    )


def evaluate_matched_guards(
    baseline_loss: Sequence[float] | np.ndarray,
    candidate_losses: Mapping[str, Sequence[float] | np.ndarray],
    accepted_by_method: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    reference_method: str,
    harmful_tolerance: float = 0.0,
    high_quantile: float = 0.95,
) -> MatchedGuardEvaluation:
    """Evaluate methods with the same loss vector and exact fallback.

    Methods may use different predeclared guards, but every rejected row receives
    the same ``baseline_loss``. This prevents an unguarded comparator from being
    compared against a guarded Bayesian method under different fallback rules.
    """

    baseline = _finite_vector(baseline_loss, name="baseline_loss")
    method_names = tuple(candidate_losses)
    if not method_names:
        raise ValueError("candidate_losses must contain at least one method")
    if any(not isinstance(name, str) or not name.strip() for name in method_names):
        raise ValueError("method names must be nonempty strings")
    if set(method_names) != set(accepted_by_method):
        raise ValueError("candidate_losses and accepted_by_method must have same keys")
    if reference_method not in candidate_losses:
        raise ValueError("reference_method must name one candidate method")

    point_evaluations = {
        method: evaluate_guard(
            baseline,
            candidate_losses[method],
            accepted_by_method[method],
            harmful_tolerance=harmful_tolerance,
            high_quantile=high_quantile,
        )
        for method in method_names
    }
    reference_loss = point_evaluations[reference_method].selected_mean_loss
    methods = tuple(
        GuardMethodEvaluation(
            method=method,
            evaluation=point_evaluations[method],
            selected_mean_loss_difference_from_reference=float(
                point_evaluations[method].selected_mean_loss - reference_loss
            ),
        )
        for method in method_names
    )
    return MatchedGuardEvaluation(reference_method=reference_method, methods=methods)


def evaluate_prediction_intervals(
    target: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    nominal_coverage: float = 0.95,
) -> PredictionIntervalEvaluation:
    """Evaluate scalar prediction-interval calibration and sharpness."""

    target_values = _finite_vector(target, name="target")
    lower_values = _finite_vector(lower, name="lower")
    upper_values = _finite_vector(upper, name="upper")
    if (
        lower_values.shape != target_values.shape
        or upper_values.shape != target_values.shape
    ):
        raise ValueError("lower and upper must match target")
    if np.any(lower_values > upper_values):
        raise ValueError("lower must not exceed upper")
    nominal = _validated_probability(nominal_coverage, name="nominal_coverage")

    covered = (target_values >= lower_values) & (target_values <= upper_values)
    below = target_values < lower_values
    above = target_values > upper_values
    width = upper_values - lower_values
    empirical = float(np.mean(covered))
    return PredictionIntervalEvaluation(
        observation_count=len(target_values),
        nominal_coverage=nominal,
        empirical_coverage=empirical,
        coverage_error=float(empirical - nominal),
        mean_interval_width=float(np.mean(width)),
        median_interval_width=float(np.median(width)),
        p90_interval_width=float(np.quantile(width, 0.9)),
        below_interval_rate=float(np.mean(below)),
        above_interval_rate=float(np.mean(above)),
    )


def evaluate_prediction_intervals_by_horizon(
    target: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    horizon: Sequence[float] | np.ndarray,
    *,
    nominal_coverage: float = 0.95,
) -> tuple[HorizonIntervalEvaluation, ...]:
    """Report prediction-interval coverage and width at each declared horizon."""

    target_values = _finite_vector(target, name="target")
    lower_values = _finite_vector(lower, name="lower")
    upper_values = _finite_vector(upper, name="upper")
    horizon_values = _finite_vector(horizon, name="horizon")
    if (
        lower_values.shape != target_values.shape
        or upper_values.shape != target_values.shape
        or horizon_values.shape != target_values.shape
    ):
        raise ValueError("lower, upper, and horizon must match target")
    if np.any(horizon_values < 0.0):
        raise ValueError("horizon must be nonnegative")

    return tuple(
        HorizonIntervalEvaluation(
            horizon=float(value),
            evaluation=evaluate_prediction_intervals(
                target_values[horizon_values == value],
                lower_values[horizon_values == value],
                upper_values[horizon_values == value],
                nominal_coverage=nominal_coverage,
            ),
        )
        for value in np.unique(horizon_values)
    )


def _optional_metric(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _bootstrap_interval(
    estimate: float | None,
    replicates: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_repeats: int,
) -> BootstrapInterval:
    finite = replicates[np.isfinite(replicates)]
    if estimate is None or len(finite) == 0:
        lower = None
        upper = None
    else:
        alpha = 0.5 * (1.0 - confidence_level)
        lower, upper = map(float, np.quantile(finite, (alpha, 1.0 - alpha)))
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        finite_replicates=len(finite),
        bootstrap_repeats=bootstrap_repeats,
    )


def bootstrap_guard_evaluation(
    baseline_loss: Sequence[float] | np.ndarray,
    candidate_loss: Sequence[float] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    group_ids: Sequence[Any] | np.ndarray,
    *,
    bootstrap_repeats: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
    harmful_tolerance: float = 0.0,
    high_quantile: float = 0.95,
) -> GuardBootstrapSummary:
    """Resample declared groups and summarize fixed-guard uncertainty.

    Every sampled group contributes all of its rows, preserving within-group
    dependence. Replicates with no accepted row remain valid for coverage,
    fallback, and selected-system metrics. Their accepted-only metrics are
    missing and excluded from the corresponding percentile interval.
    """

    baseline, candidate = _validated_losses(baseline_loss, candidate_loss)
    decision = _boolean_vector(accepted, name="accepted", length=len(baseline))
    groups = _labeled_index_lists(
        group_ids,
        name="group_ids",
        length=len(baseline),
    )
    group_indices = tuple(indices for _, indices in groups)
    repeats = int(bootstrap_repeats)
    if repeats < 1 or repeats != bootstrap_repeats:
        raise ValueError("bootstrap_repeats must be a positive integer")
    confidence = _validated_probability(confidence_level, name="confidence_level")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    seed_value = int(seed)

    point = evaluate_guard(
        baseline,
        candidate,
        decision,
        harmful_tolerance=harmful_tolerance,
        high_quantile=high_quantile,
    )
    metric_names = (
        "coverage",
        "fallback_rate",
        "selected_mean_excess_loss",
        "accepted_mean_excess_loss",
        "accepted_high_quantile_excess_loss",
        "harmful_accepted_rate",
        "worst_accepted_excess_loss",
    )
    replicate_values: dict[str, np.ndarray] = {
        name: np.empty(repeats, dtype=float) for name in metric_names
    }

    generator = np.random.default_rng(seed_value)
    group_count = len(group_indices)
    for replicate_index in range(repeats):
        sampled_groups = generator.integers(0, group_count, size=group_count)
        row_indices = np.concatenate(
            tuple(group_indices[int(group_index)] for group_index in sampled_groups)
        )
        evaluation = evaluate_guard(
            baseline[row_indices],
            candidate[row_indices],
            decision[row_indices],
            harmful_tolerance=harmful_tolerance,
            high_quantile=high_quantile,
        )
        replicate_values["coverage"][replicate_index] = evaluation.coverage
        replicate_values["fallback_rate"][replicate_index] = evaluation.fallback_rate
        replicate_values["selected_mean_excess_loss"][replicate_index] = (
            evaluation.selected_mean_excess_loss
        )
        replicate_values["accepted_mean_excess_loss"][replicate_index] = (
            _optional_metric(evaluation.accepted_mean_excess_loss)
        )
        replicate_values["accepted_high_quantile_excess_loss"][replicate_index] = (
            _optional_metric(evaluation.accepted_high_quantile_excess_loss)
        )
        replicate_values["harmful_accepted_rate"][replicate_index] = _optional_metric(
            evaluation.harmful_accepted_rate
        )
        replicate_values["worst_accepted_excess_loss"][replicate_index] = (
            _optional_metric(evaluation.worst_accepted_excess_loss)
        )

    def interval(name: str, estimate: float | None) -> BootstrapInterval:
        return _bootstrap_interval(
            estimate,
            replicate_values[name],
            confidence_level=confidence,
            bootstrap_repeats=repeats,
        )

    return GuardBootstrapSummary(
        point_estimate=point,
        group_count=group_count,
        confidence_level=confidence,
        bootstrap_repeats=repeats,
        seed=seed_value,
        coverage=interval("coverage", point.coverage),
        fallback_rate=interval("fallback_rate", point.fallback_rate),
        selected_mean_excess_loss=interval(
            "selected_mean_excess_loss",
            point.selected_mean_excess_loss,
        ),
        accepted_mean_excess_loss=interval(
            "accepted_mean_excess_loss",
            point.accepted_mean_excess_loss,
        ),
        accepted_high_quantile_excess_loss=interval(
            "accepted_high_quantile_excess_loss",
            point.accepted_high_quantile_excess_loss,
        ),
        harmful_accepted_rate=interval(
            "harmful_accepted_rate",
            point.harmful_accepted_rate,
        ),
        worst_accepted_excess_loss=interval(
            "worst_accepted_excess_loss",
            point.worst_accepted_excess_loss,
        ),
    )


__all__ = [
    "BootstrapInterval",
    "GuardBootstrapSummary",
    "GuardEvaluation",
    "GuardMethodEvaluation",
    "GuardStratumEvaluation",
    "HorizonIntervalEvaluation",
    "MatchedGuardEvaluation",
    "PredictionIntervalEvaluation",
    "SelectiveRiskPoint",
    "bootstrap_guard_evaluation",
    "evaluate_guard",
    "evaluate_guard_by_stratum",
    "evaluate_matched_guards",
    "evaluate_prediction_intervals",
    "evaluate_prediction_intervals_by_horizon",
    "selective_risk_curve",
]
