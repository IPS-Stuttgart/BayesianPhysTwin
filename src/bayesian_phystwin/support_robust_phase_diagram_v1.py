"""Exact finite support-miss phase diagrams for act--sense--fallback plans.

For every complete direct or probe-contingent plan ``p`` and benchmark plan
``b``, let ``D0[p, b]`` be the exact represented-support loss gap returned by
the quotient decision certificate.  Suppose at most ``epsilon`` probability
mass may lie outside the represented physical support and the unknown-domain
complete plan-loss vector belongs to the declared axis-aligned box
``[lower, upper]``.  The exact pairwise contaminated-support gap is

    D_epsilon[p, b]
      = D0[p, b]
        + epsilon * max(0, upper[p] - lower[b] - D0[p, b]).

The diagonal is identically zero because a plan is compared with itself.  Each
plan's robust regret is therefore the upper envelope of finitely many affine,
nondecreasing functions of ``epsilon``.  The globally selected minimax plan is
the lower envelope of those plan-regret curves, followed by the exact
caller-owned fallback whenever the minimum regret exceeds the registered
tolerance.

This module computes every breakpoint exactly up to floating-point arithmetic:
within-plan upper-envelope changes, between-plan envelope intersections, and
regret-tolerance crossings.  It reports the decision at every breakpoint and on
every open interval between adjacent breakpoints.  No epsilon grid is used.

The result is exact for the supplied finite represented hypotheses, quotient,
losses, deterministic probe maps, costs, and axis-aligned unknown-plan loss box.
It does not estimate the support-miss probability or validate the box, probes,
loss model, quotient, target transport, deployment, or safety.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    ActSenseFallbackCertificateV1,
    act_sense_fallback_certificate,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]

SUPPORT_ROBUST_PHASE_DIAGRAM_VERSION: Final = 1
SUPPORT_ROBUST_PHASE_DIAGRAM_SEMANTICS: Final = (
    "exact-finite-act-sense-fallback-support-miss-phase-diagram-v1"
)
SUPPORT_ROBUST_PHASE_DIAGRAM_CLAIM_BOUNDARY: Final = (
    "The phase diagram is exact only for the supplied finite hypotheses, prior "
    "support, quotient masses, terminal losses, deterministic probe outcomes, "
    "probe costs, regret tolerance, and axis-aligned unknown complete-plan loss "
    "box. Terminal-action bounds are converted to complete-plan bounds by "
    "allowing unknown physics to realize any registered probe outcome. The "
    "result does not estimate support-miss probability, validate the loss box, "
    "probe physics, reset semantics, quotient, provider, target transport, "
    "online execution, calibration, deployment, or safety."
)

_NUMERICAL_ATOL: Final = 1e-12


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _finite_nonnegative_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite_nonnegative_real(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_vector(value: object, *, size: int, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    vector = np.ascontiguousarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size != size:
        raise ValueError(f"{name} must contain exactly {size} entries")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return _immutable_float64(vector)


def _loss_box(
    lower: object,
    upper: object,
    *,
    size: int,
    prefix: str,
) -> tuple[FloatArray, FloatArray]:
    lower_array = _nonnegative_vector(
        lower,
        size=size,
        name=f"{prefix}_loss_lower",
    )
    upper_array = _nonnegative_vector(
        upper,
        size=size,
        name=f"{prefix}_loss_upper",
    )
    if np.any(lower_array > upper_array + _NUMERICAL_ATOL):
        raise ValueError(f"{prefix} loss lower bounds must not exceed upper bounds")
    return lower_array, upper_array


class SupportRobustPhaseDecisionV1(NamedTuple):
    """One exact decision at a support-miss probability or inside one phase cell."""

    support_miss_probability: float
    minimax_plan_index: int
    minimax_worst_case_regret: float
    active_benchmark_plan_index: int
    has_admissible_plan: bool
    output_plan_index: int
    output_mode: str
    used_fallback: bool
    selected_probe_index: int | None
    selected_probe_name: str | None

    def summary(self) -> dict[str, object]:
        return {
            "support_miss_probability": self.support_miss_probability,
            "minimax_plan_index": self.minimax_plan_index,
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "active_benchmark_plan_index": self.active_benchmark_plan_index,
            "has_admissible_plan": self.has_admissible_plan,
            "output_plan_index": self.output_plan_index,
            "output_mode": self.output_mode,
            "used_fallback": self.used_fallback,
            "selected_probe_index": self.selected_probe_index,
            "selected_probe_name": self.selected_probe_name,
        }


class SupportRobustPhaseIntervalV1(NamedTuple):
    """Decision on the open interval ``(support_miss_left, support_miss_right)``."""

    support_miss_left: float
    support_miss_right: float
    decision: SupportRobustPhaseDecisionV1

    def summary(self) -> dict[str, object]:
        return {
            "support_miss_left": self.support_miss_left,
            "support_miss_right": self.support_miss_right,
            "open_interval": True,
            "decision": self.decision.summary(),
        }


class _EnvelopeSegment(NamedTuple):
    left: float
    right: float
    intercept: float
    slope: float
    benchmark_plan_index: int


class SupportRobustPhaseDiagramV1(NamedTuple):
    """Complete exact phase diagram over one closed support-miss interval."""

    base_certificate: ActSenseFallbackCertificateV1
    represented_pairwise_worst_case_loss_gap: FloatArray
    support_miss_pairwise_slope: FloatArray
    unknown_plan_loss_lower: FloatArray
    unknown_plan_loss_upper: FloatArray
    regret_tolerance: float
    maximum_support_miss_probability: float
    breakpoints: FloatArray
    point_decisions: tuple[SupportRobustPhaseDecisionV1, ...]
    interval_decisions: tuple[SupportRobustPhaseIntervalV1, ...]
    plan_maximum_admissible_support_miss: FloatArray
    maximum_any_plan_admissible_support_miss: float

    @property
    def plan_count(self) -> int:
        return self.base_certificate.plan_count

    def decision_at(
        self, support_miss_probability: float
    ) -> SupportRobustPhaseDecisionV1:
        """Evaluate the exact robust decision at one admissible miss probability."""

        epsilon = _probability(
            support_miss_probability,
            name="support_miss_probability",
        )
        if epsilon > self.maximum_support_miss_probability + _NUMERICAL_ATOL:
            raise ValueError(
                "support_miss_probability exceeds the phase-diagram maximum"
            )
        return _decision_at(
            self.base_certificate,
            self.represented_pairwise_worst_case_loss_gap,
            self.support_miss_pairwise_slope,
            self.regret_tolerance,
            min(epsilon, self.maximum_support_miss_probability),
        )

    def summary(self) -> dict[str, object]:
        return {
            "version": SUPPORT_ROBUST_PHASE_DIAGRAM_VERSION,
            "semantics": SUPPORT_ROBUST_PHASE_DIAGRAM_SEMANTICS,
            "plan_count": self.plan_count,
            "regret_tolerance": self.regret_tolerance,
            "maximum_support_miss_probability": (self.maximum_support_miss_probability),
            "breakpoint_count": int(self.breakpoints.size),
            "interval_count": len(self.interval_decisions),
            "maximum_any_plan_admissible_support_miss": (
                self.maximum_any_plan_admissible_support_miss
            ),
            "point_decisions": [item.summary() for item in self.point_decisions],
            "interval_decisions": [item.summary() for item in self.interval_decisions],
            "claim_boundary": SUPPORT_ROBUST_PHASE_DIAGRAM_CLAIM_BOUNDARY,
        }


def _derive_plan_loss_box(
    certificate: ActSenseFallbackCertificateV1,
    probe_costs: FloatArray,
    terminal_lower: FloatArray,
    terminal_upper: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    lower = np.empty(certificate.plan_count, dtype=np.float64)
    upper = np.empty(certificate.plan_count, dtype=np.float64)
    for plan_index, plan in enumerate(certificate.plans):
        if plan.mode == "act":
            if plan.direct_action_index is None:
                raise RuntimeError("direct plan is missing its action")
            action = plan.direct_action_index
            lower[plan_index] = terminal_lower[action]
            upper[plan_index] = terminal_upper[action]
            continue
        if plan.mode != "sense" or plan.probe_index is None:
            raise RuntimeError("unsupported contingent plan representation")
        actions = plan.terminal_action_by_outcome
        if actions.size == 0:
            raise RuntimeError("sensing plan has no registered outcome actions")
        cost = probe_costs[plan.probe_index]
        lower[plan_index] = cost + float(np.min(terminal_lower[actions]))
        upper[plan_index] = cost + float(np.max(terminal_upper[actions]))
    return _immutable_float64(lower), _immutable_float64(upper)


def _decision_at(
    certificate: ActSenseFallbackCertificateV1,
    represented_gap: FloatArray,
    slope: FloatArray,
    tolerance: float,
    epsilon: float,
) -> SupportRobustPhaseDecisionV1:
    pairwise = represented_gap + epsilon * slope
    np.fill_diagonal(pairwise, 0.0)
    regret = np.maximum(np.max(pairwise, axis=1), 0.0)
    minimum = float(np.min(regret))
    candidates = np.flatnonzero(
        np.isclose(regret, minimum, rtol=0.0, atol=_NUMERICAL_ATOL)
    )
    minimax = int(candidates[0])
    benchmark_values = pairwise[minimax]
    benchmark_maximum = float(np.max(benchmark_values))
    active_benchmarks = np.flatnonzero(
        np.isclose(
            benchmark_values,
            benchmark_maximum,
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        )
    )
    active_benchmark = int(active_benchmarks[0])
    admissible = minimum <= tolerance + _NUMERICAL_ATOL
    if not admissible:
        output_plan = certificate.fallback_plan_index
        output_mode = "fallback"
        used_fallback = True
    else:
        output_plan = minimax
        plan = certificate.plans[output_plan]
        if plan.mode == "sense":
            output_mode = "sense"
            used_fallback = False
        elif plan.direct_action_index == certificate.fallback_action_index:
            output_mode = "fallback"
            used_fallback = True
        else:
            output_mode = "act"
            used_fallback = False
    selected = certificate.plans[output_plan]
    selected_probe_index = selected.probe_index if output_mode == "sense" else None
    selected_probe_name = selected.probe_name if output_mode == "sense" else None
    return SupportRobustPhaseDecisionV1(
        support_miss_probability=float(epsilon),
        minimax_plan_index=minimax,
        minimax_worst_case_regret=minimum,
        active_benchmark_plan_index=active_benchmark,
        has_admissible_plan=admissible,
        output_plan_index=output_plan,
        output_mode=output_mode,
        used_fallback=used_fallback,
        selected_probe_index=selected_probe_index,
        selected_probe_name=selected_probe_name,
    )


def _upper_envelope_segments(
    intercepts: FloatArray,
    slopes: FloatArray,
    maximum_epsilon: float,
) -> tuple[_EnvelopeSegment, ...]:
    """Return the affine upper envelope on ``[0, maximum_epsilon]``."""

    lines = sorted(
        (
            (float(slopes[index]), float(intercepts[index]), int(index))
            for index in range(intercepts.size)
        ),
        key=lambda item: (item[0], item[1], -item[2]),
    )
    unique: list[tuple[float, float, int]] = []
    for slope, intercept, benchmark in lines:
        if unique and np.isclose(
            slope,
            unique[-1][0],
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        ):
            previous = unique[-1]
            if intercept > previous[1] + _NUMERICAL_ATOL or (
                np.isclose(
                    intercept,
                    previous[1],
                    rtol=0.0,
                    atol=_NUMERICAL_ATOL,
                )
                and benchmark < previous[2]
            ):
                unique[-1] = (slope, intercept, benchmark)
            continue
        unique.append((slope, intercept, benchmark))

    hull: list[tuple[float, float, int]] = []
    starts: list[float] = []
    for slope, intercept, benchmark in unique:
        start = -np.inf
        while hull:
            previous_slope, previous_intercept, _ = hull[-1]
            denominator = slope - previous_slope
            if denominator <= _NUMERICAL_ATOL:
                raise RuntimeError("upper-envelope slopes are not strictly increasing")
            crossing = (previous_intercept - intercept) / denominator
            if crossing <= starts[-1] + _NUMERICAL_ATOL:
                hull.pop()
                starts.pop()
                continue
            start = crossing
            break
        if not hull:
            start = -np.inf
        hull.append((slope, intercept, benchmark))
        starts.append(float(start))

    segments: list[_EnvelopeSegment] = []
    for index, (slope, intercept, benchmark) in enumerate(hull):
        left = max(0.0, starts[index])
        right = (
            maximum_epsilon
            if index + 1 == len(hull)
            else min(maximum_epsilon, starts[index + 1])
        )
        if right < -_NUMERICAL_ATOL or left > maximum_epsilon + _NUMERICAL_ATOL:
            continue
        left = min(max(left, 0.0), maximum_epsilon)
        right = min(max(right, 0.0), maximum_epsilon)
        if right + _NUMERICAL_ATOL < left:
            continue
        segments.append(
            _EnvelopeSegment(
                left=float(left),
                right=float(right),
                intercept=intercept,
                slope=slope,
                benchmark_plan_index=benchmark,
            )
        )
    if not segments:
        benchmark = int(np.argmax(intercepts))
        segments.append(
            _EnvelopeSegment(
                left=0.0,
                right=maximum_epsilon,
                intercept=float(intercepts[benchmark]),
                slope=float(slopes[benchmark]),
                benchmark_plan_index=benchmark,
            )
        )
    return tuple(segments)


def _deduplicate_breakpoints(
    values: list[float],
    *,
    maximum_epsilon: float,
) -> FloatArray:
    clipped = sorted(min(max(float(value), 0.0), maximum_epsilon) for value in values)
    unique: list[float] = []
    for value in clipped:
        if not unique or value - unique[-1] > _NUMERICAL_ATOL:
            unique.append(value)
        elif value in (0.0, maximum_epsilon):
            unique[-1] = value
    if not unique or unique[0] > _NUMERICAL_ATOL:
        unique.insert(0, 0.0)
    else:
        unique[0] = 0.0
    if maximum_epsilon - unique[-1] > _NUMERICAL_ATOL:
        unique.append(maximum_epsilon)
    else:
        unique[-1] = maximum_epsilon
    return _immutable_float64(unique)


def _phase_breakpoints(
    envelopes: tuple[tuple[_EnvelopeSegment, ...], ...],
    *,
    tolerance: float,
    maximum_epsilon: float,
    maximum_breakpoint_count: int,
) -> FloatArray:
    candidates = [0.0, maximum_epsilon]
    for segments in envelopes:
        for segment in segments:
            candidates.extend((segment.left, segment.right))
            if segment.slope > _NUMERICAL_ATOL:
                crossing = (tolerance - segment.intercept) / segment.slope
                if (
                    segment.left - _NUMERICAL_ATOL
                    <= crossing
                    <= segment.right + _NUMERICAL_ATOL
                ):
                    candidates.append(crossing)

    for left_plan in range(len(envelopes)):
        for right_plan in range(left_plan + 1, len(envelopes)):
            for left_segment in envelopes[left_plan]:
                for right_segment in envelopes[right_plan]:
                    overlap_left = max(left_segment.left, right_segment.left)
                    overlap_right = min(left_segment.right, right_segment.right)
                    if overlap_right + _NUMERICAL_ATOL < overlap_left:
                        continue
                    denominator = left_segment.slope - right_segment.slope
                    if abs(denominator) <= _NUMERICAL_ATOL:
                        continue
                    crossing = (
                        right_segment.intercept - left_segment.intercept
                    ) / denominator
                    if (
                        overlap_left - _NUMERICAL_ATOL
                        <= crossing
                        <= overlap_right + _NUMERICAL_ATOL
                    ):
                        candidates.append(crossing)
                    if len(candidates) > maximum_breakpoint_count * 8:
                        raise ValueError(
                            "candidate phase-breakpoint count exceeds the registered cap"
                        )

    result = _deduplicate_breakpoints(
        candidates,
        maximum_epsilon=maximum_epsilon,
    )
    if result.size > maximum_breakpoint_count:
        raise ValueError(
            "phase-breakpoint count exceeds maximum_breakpoint_count: "
            f"{result.size} > {maximum_breakpoint_count}"
        )
    return result


def _maximum_admissible_epsilon(
    segments: tuple[_EnvelopeSegment, ...],
    *,
    tolerance: float,
    maximum_epsilon: float,
) -> float:
    value_zero = max(
        segment.intercept for segment in segments if segment.left <= _NUMERICAL_ATOL
    )
    if value_zero > tolerance + _NUMERICAL_ATOL:
        return float("nan")
    value_maximum = max(
        segment.intercept + maximum_epsilon * segment.slope
        for segment in segments
        if segment.right >= maximum_epsilon - _NUMERICAL_ATOL
    )
    if value_maximum <= tolerance + _NUMERICAL_ATOL:
        return maximum_epsilon
    for segment in segments:
        left_value = segment.intercept + segment.left * segment.slope
        right_value = segment.intercept + segment.right * segment.slope
        if left_value > tolerance + _NUMERICAL_ATOL:
            return segment.left
        if right_value > tolerance + _NUMERICAL_ATOL:
            if segment.slope <= _NUMERICAL_ATOL:
                return segment.left
            crossing = (tolerance - segment.intercept) / segment.slope
            return min(max(float(crossing), segment.left), segment.right)
    raise RuntimeError("admissible support-miss boundary was not found")


def support_robust_phase_diagram(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    terminal_loss_by_hypothesis_action: object,
    probe_outcome_index_by_hypothesis: object,
    probe_costs: object,
    *,
    fallback_action_index: int,
    regret_tolerance: float = 0.0,
    maximum_support_miss_probability: float = 1.0,
    probe_names: Sequence[str] | None = None,
    max_plan_count: int = 100_000,
    maximum_phase_plan_count: int = 512,
    maximum_breakpoint_count: int = 200_000,
    unknown_plan_loss_lower: object | None = None,
    unknown_plan_loss_upper: object | None = None,
    unknown_terminal_loss_lower_by_action: object | None = None,
    unknown_terminal_loss_upper_by_action: object | None = None,
) -> SupportRobustPhaseDiagramV1:
    """Return every exact Act/Sense/Fallback transition over ``epsilon``.

    Exactly one unknown-loss specification must be supplied:

    * complete-plan lower and upper vectors, in the exact plan order produced by
      :func:`act_sense_fallback_certificate`; or
    * terminal-action lower and upper vectors.  In the latter case a direct
      plan inherits its action bounds, while a sensing plan adds its probe cost
      and allows unknown physics to realize any registered probe outcome.

    The represented-support certificate and complete plan roster are frozen
    once.  Pairwise contaminated-support gaps are affine in ``epsilon``.  The
    routine builds every plan-regret upper envelope and then enumerates all
    finite crossings required to partition the requested closed interval.
    """

    tolerance = _finite_nonnegative_real(
        regret_tolerance,
        name="regret_tolerance",
    )
    maximum_epsilon = _probability(
        maximum_support_miss_probability,
        name="maximum_support_miss_probability",
    )
    phase_plan_cap = _positive_integer(
        maximum_phase_plan_count,
        name="maximum_phase_plan_count",
    )
    breakpoint_cap = _positive_integer(
        maximum_breakpoint_count,
        name="maximum_breakpoint_count",
    )
    certificate = act_sense_fallback_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        terminal_loss_by_hypothesis_action,
        probe_outcome_index_by_hypothesis,
        probe_costs,
        fallback_action_index=fallback_action_index,
        regret_tolerance=tolerance,
        probe_names=probe_names,
        max_plan_count=max_plan_count,
    )
    if certificate.plan_count > phase_plan_cap:
        raise ValueError(
            "phase-diagram plan count exceeds maximum_phase_plan_count: "
            f"{certificate.plan_count} > {phase_plan_cap}"
        )

    has_plan_box = (
        unknown_plan_loss_lower is not None or unknown_plan_loss_upper is not None
    )
    has_terminal_box = (
        unknown_terminal_loss_lower_by_action is not None
        or unknown_terminal_loss_upper_by_action is not None
    )
    if has_plan_box == has_terminal_box:
        raise ValueError(
            "supply exactly one complete plan-loss box or terminal action-loss box"
        )
    if has_plan_box:
        if unknown_plan_loss_lower is None or unknown_plan_loss_upper is None:
            raise ValueError(
                "complete plan-loss lower and upper bounds are both required"
            )
        lower, upper = _loss_box(
            unknown_plan_loss_lower,
            unknown_plan_loss_upper,
            size=certificate.plan_count,
            prefix="unknown_plan",
        )
    else:
        if (
            unknown_terminal_loss_lower_by_action is None
            or unknown_terminal_loss_upper_by_action is None
        ):
            raise ValueError(
                "terminal action-loss lower and upper bounds are both required"
            )
        terminal_lower, terminal_upper = _loss_box(
            unknown_terminal_loss_lower_by_action,
            unknown_terminal_loss_upper_by_action,
            size=certificate.direct_plan_count,
            prefix="unknown_terminal",
        )
        costs = _nonnegative_vector(
            probe_costs,
            size=certificate.probe_count,
            name="probe_costs",
        )
        lower, upper = _derive_plan_loss_box(
            certificate,
            costs,
            terminal_lower,
            terminal_upper,
        )

    represented_gap = np.array(
        certificate.plan_certificate.pairwise_worst_case_loss_gap,
        dtype=np.float64,
        copy=True,
    )
    if represented_gap.shape != (certificate.plan_count, certificate.plan_count):
        raise RuntimeError("represented pairwise plan-gap dimensions changed")
    np.fill_diagonal(represented_gap, 0.0)
    unknown_box_gap = upper[:, None] - lower[None, :]
    slope = np.maximum(unknown_box_gap - represented_gap, 0.0)
    np.fill_diagonal(slope, 0.0)
    represented_gap = _immutable_float64(represented_gap)
    slope = _immutable_float64(slope)

    envelopes = tuple(
        _upper_envelope_segments(
            represented_gap[plan_index],
            slope[plan_index],
            maximum_epsilon,
        )
        for plan_index in range(certificate.plan_count)
    )
    breakpoints = _phase_breakpoints(
        envelopes,
        tolerance=tolerance,
        maximum_epsilon=maximum_epsilon,
        maximum_breakpoint_count=breakpoint_cap,
    )
    point_decisions = tuple(
        _decision_at(
            certificate,
            represented_gap,
            slope,
            tolerance,
            float(epsilon),
        )
        for epsilon in breakpoints
    )
    intervals: list[SupportRobustPhaseIntervalV1] = []
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        if right - left <= _NUMERICAL_ATOL:
            continue
        midpoint = float(left + 0.5 * (right - left))
        decision = _decision_at(
            certificate,
            represented_gap,
            slope,
            tolerance,
            midpoint,
        )
        for fraction in (0.25, 0.75):
            check_epsilon = float(left + fraction * (right - left))
            check = _decision_at(
                certificate,
                represented_gap,
                slope,
                tolerance,
                check_epsilon,
            )
            if (
                check.minimax_plan_index != decision.minimax_plan_index
                or check.output_plan_index != decision.output_plan_index
                or check.output_mode != decision.output_mode
                or check.active_benchmark_plan_index
                != decision.active_benchmark_plan_index
            ):
                raise RuntimeError(
                    "phase partition missed an interior decision transition"
                )
        intervals.append(
            SupportRobustPhaseIntervalV1(
                support_miss_left=float(left),
                support_miss_right=float(right),
                decision=decision,
            )
        )

    maximum_by_plan = _immutable_float64(
        [
            _maximum_admissible_epsilon(
                segments,
                tolerance=tolerance,
                maximum_epsilon=maximum_epsilon,
            )
            for segments in envelopes
        ]
    )
    finite = maximum_by_plan[np.isfinite(maximum_by_plan)]
    maximum_any = float(np.max(finite)) if finite.size else float("nan")
    return SupportRobustPhaseDiagramV1(
        base_certificate=certificate,
        represented_pairwise_worst_case_loss_gap=represented_gap,
        support_miss_pairwise_slope=slope,
        unknown_plan_loss_lower=lower,
        unknown_plan_loss_upper=upper,
        regret_tolerance=tolerance,
        maximum_support_miss_probability=maximum_epsilon,
        breakpoints=breakpoints,
        point_decisions=point_decisions,
        interval_decisions=tuple(intervals),
        plan_maximum_admissible_support_miss=maximum_by_plan,
        maximum_any_plan_admissible_support_miss=maximum_any,
    )


__all__ = [
    "SUPPORT_ROBUST_PHASE_DIAGRAM_CLAIM_BOUNDARY",
    "SUPPORT_ROBUST_PHASE_DIAGRAM_SEMANTICS",
    "SUPPORT_ROBUST_PHASE_DIAGRAM_VERSION",
    "SupportRobustPhaseDecisionV1",
    "SupportRobustPhaseDiagramV1",
    "SupportRobustPhaseIntervalV1",
    "support_robust_phase_diagram",
]
