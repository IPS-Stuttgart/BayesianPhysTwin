"""Selective-prediction diagnostics for guarded Bayesian-PhysTwin updates.

The functions in this module compare a candidate loss against the exact baseline
that would be returned on rejection.  Acceptance scores are evaluated without
splitting ties, and uncertainty can be estimated by resampling declared
interaction/object/session groups rather than treating correlated rows as
independent observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    accepted_mean_excess_loss: float | None
    harmful_accepted_rate: float | None
    worst_accepted_excess_loss: float | None


@dataclass(frozen=True)
class SelectiveRiskPoint:
    """Guard diagnostics at one score threshold."""

    threshold: float
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
    harmful_accepted_rate: BootstrapInterval
    worst_accepted_excess_loss: BootstrapInterval


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
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
    result = np.asarray(array, dtype=bool).copy()
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


def evaluate_guard(
    baseline_loss: Sequence[float] | np.ndarray,
    candidate_loss: Sequence[float] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    *,
    harmful_tolerance: float = 0.0,
) -> GuardEvaluation:
    """Evaluate one fixed guard against its exact fallback baseline.

    ``harmful_tolerance`` defines the smallest positive candidate-minus-baseline
    excess counted as harmful.  It must be selected independently of target
    outcomes when the resulting statistic is used confirmatorily.
    """

    baseline, candidate = _validated_losses(baseline_loss, candidate_loss)
    decision = _boolean_vector(accepted, name="accepted", length=len(baseline))
    tolerance = float(harmful_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("harmful_tolerance must be finite and nonnegative")

    excess = candidate - baseline
    selected = np.where(decision, candidate, baseline)
    accepted_count = int(np.sum(decision))
    observation_count = len(baseline)
    coverage = accepted_count / observation_count

    if accepted_count:
        accepted_excess = excess[decision]
        accepted_mean_excess: float | None = float(np.mean(accepted_excess))
        harmful_rate: float | None = float(
            np.mean(accepted_excess > tolerance)
        )
        worst_excess: float | None = float(np.max(accepted_excess))
    else:
        accepted_mean_excess = None
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
        accepted_mean_excess_loss=accepted_mean_excess,
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
) -> tuple[SelectiveRiskPoint, ...]:
    """Evaluate every distinct score threshold while preserving score ties."""

    baseline, candidate = _validated_losses(baseline_loss, candidate_loss)
    scores = _finite_vector(acceptance_score, name="acceptance_score")
    if scores.shape != baseline.shape:
        raise ValueError("acceptance_score must match baseline_loss")
    if not isinstance(higher_is_safer, bool):
        raise ValueError("higher_is_safer must be Boolean")

    thresholds = np.unique(scores)
    if higher_is_safer:
        thresholds = thresholds[::-1]

    points: list[SelectiveRiskPoint] = []
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
                ),
            )
        )
    return tuple(points)


def _group_index_lists(
    group_ids: Sequence[Any] | np.ndarray,
    *,
    length: int,
) -> tuple[np.ndarray, ...]:
    values = np.asarray(group_ids, dtype=object)
    if values.shape != (length,):
        raise ValueError(f"group_ids must have shape ({length},)")

    grouped: dict[Any, list[int]] = {}
    for index, group in enumerate(values.tolist()):
        try:
            hash(group)
        except TypeError as error:
            raise ValueError("group_ids must contain hashable values") from error
        grouped.setdefault(group, []).append(index)
    if not grouped:
        raise ValueError("group_ids must identify at least one group")
    return tuple(np.asarray(indices, dtype=np.int64) for indices in grouped.values())


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
) -> GuardBootstrapSummary:
    """Resample declared groups and summarize guard uncertainty.

    Every sampled group contributes all of its rows, preserving within-group
    dependence.  Repeatedly sampled groups are duplicated as complete clusters.
    Replicates with no accepted row remain valid for coverage/fallback and
    selected-system metrics; their accepted-only metrics are reported as missing
    and excluded from the corresponding percentile interval.
    """

    baseline, candidate = _validated_losses(baseline_loss, candidate_loss)
    decision = _boolean_vector(accepted, name="accepted", length=len(baseline))
    groups = _group_index_lists(group_ids, length=len(baseline))
    repeats = int(bootstrap_repeats)
    if repeats < 1 or repeats != bootstrap_repeats:
        raise ValueError("bootstrap_repeats must be a positive integer")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    seed_value = int(seed)

    point = evaluate_guard(
        baseline,
        candidate,
        decision,
        harmful_tolerance=harmful_tolerance,
    )
    metric_names = (
        "coverage",
        "fallback_rate",
        "selected_mean_excess_loss",
        "accepted_mean_excess_loss",
        "harmful_accepted_rate",
        "worst_accepted_excess_loss",
    )
    replicate_values = {
        name: np.empty(repeats, dtype=float) for name in metric_names
    }

    generator = np.random.default_rng(seed_value)
    group_count = len(groups)
    for replicate_index in range(repeats):
        sampled_groups = generator.integers(0, group_count, size=group_count)
        row_indices = np.concatenate(
            tuple(groups[int(group_index)] for group_index in sampled_groups)
        )
        evaluation = evaluate_guard(
            baseline[row_indices],
            candidate[row_indices],
            decision[row_indices],
            harmful_tolerance=harmful_tolerance,
        )
        replicate_values["coverage"][replicate_index] = evaluation.coverage
        replicate_values["fallback_rate"][replicate_index] = (
            evaluation.fallback_rate
        )
        replicate_values["selected_mean_excess_loss"][replicate_index] = (
            evaluation.selected_mean_excess_loss
        )
        replicate_values["accepted_mean_excess_loss"][replicate_index] = (
            _optional_metric(evaluation.accepted_mean_excess_loss)
        )
        replicate_values["harmful_accepted_rate"][replicate_index] = (
            _optional_metric(evaluation.harmful_accepted_rate)
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
    "SelectiveRiskPoint",
    "bootstrap_guard_evaluation",
    "evaluate_guard",
    "selective_risk_curve",
]
