"""Support-robust act--sense--fallback certificates.

This module extends the exact finite contingent-plan certificate with an
explicit upper bound on physical support miss. The represented physical belief
is the registered query quotient used by
:func:`act_sense_fallback_certificate`. Up to ``epsilon`` probability mass may
instead lie on unknown physics whose complete plan-loss vector belongs to a
registered axis-aligned box.

For represented pairwise plan gap ``Delta_0(p, b)`` and unknown-support box
bound ``M(p, b) = upper[p] - lower[b]``, the exact worst-case gap over every
mixture with unknown mass ``rho in [0, epsilon]`` is

    Delta_epsilon(p, b)
      = Delta_0(p, b) + epsilon * max(0, M(p, b) - Delta_0(p, b)).

The expression is exact for the declared contamination class because the
objective is linear in the mixture mass, the represented belief, and the
unknown plan-loss vector. It is not a claim that a supplied miss probability
or loss box is valid for a deployment domain.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    ActSenseFallbackCertificateV1,
    ContingentPlanV1,
    act_sense_fallback_certificate,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

SUPPORT_ROBUST_ACT_SENSE_FALLBACK_VERSION: Final = 1
SUPPORT_ROBUST_ACT_SENSE_FALLBACK_SEMANTICS: Final = (
    "exact-at-most-epsilon-rectangular-unknown-plan-loss-contamination-v1"
)
SUPPORT_ROBUST_ACT_SENSE_FALLBACK_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied represented finite support, "
    "query quotient, deterministic probe maps, plan-loss model, upper bound on "
    "unknown-support probability, rectangular unknown plan-loss envelope, and "
    "regret tolerance. It does not estimate or validate the support-miss bound, "
    "unknown-domain loss envelope, probe physics, reset semantics, target "
    "transport, calibration, deployment, or safety."
)

_NUMERICAL_ATOL: Final = 1e-12


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


def _unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be a finite real number in [0, 1]")
    return result


def _finite_vector(value: object, *, size: int, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    vector = np.ascontiguousarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size != size:
        raise ValueError(f"{name} must contain exactly {size} entries")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return _immutable_float64(vector)


def _validated_box(
    lower_value: object,
    upper_value: object,
    *,
    size: int,
    stem: str,
) -> tuple[FloatArray, FloatArray]:
    lower = _finite_vector(lower_value, size=size, name=f"{stem}_lower")
    upper = _finite_vector(upper_value, size=size, name=f"{stem}_upper")
    if np.any(lower > upper):
        raise ValueError(f"{stem}_lower must not exceed {stem}_upper")
    return lower, upper


def _unknown_plan_loss_box(
    base: ActSenseFallbackCertificateV1,
    terminal_lower: FloatArray,
    terminal_upper: FloatArray,
    probe_lower: FloatArray,
    probe_upper: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    lower = np.empty(base.plan_count, dtype=np.float64)
    upper = np.empty(base.plan_count, dtype=np.float64)
    for plan_index, plan in enumerate(base.plans):
        if plan.mode == "act":
            if plan.direct_action_index is None:
                raise RuntimeError("direct plan is missing its action")
            action = plan.direct_action_index
            lower[plan_index] = terminal_lower[action]
            upper[plan_index] = terminal_upper[action]
            continue
        if plan.mode != "sense" or plan.probe_index is None:
            raise RuntimeError("invalid contingent plan returned by base certificate")
        actions = plan.terminal_action_by_outcome
        if actions.size == 0:
            raise RuntimeError("sensing plan contains no terminal actions")
        probe = plan.probe_index
        lower[plan_index] = probe_lower[probe] + float(np.min(terminal_lower[actions]))
        upper[plan_index] = probe_upper[probe] + float(np.max(terminal_upper[actions]))
    if np.any(lower > upper + _NUMERICAL_ATOL):
        raise RuntimeError("derived unknown plan-loss box is invalid")
    return _immutable_float64(lower), _immutable_float64(upper)


def _maximum_admissible_support_miss(
    represented_pairwise: FloatArray,
    unknown_pairwise: FloatArray,
    tolerance: float,
) -> FloatArray:
    plan_count = represented_pairwise.shape[0]
    budgets = np.ones(plan_count, dtype=np.float64)
    for plan in range(plan_count):
        budget = 1.0
        for benchmark in range(plan_count):
            represented = float(represented_pairwise[plan, benchmark])
            unknown = float(unknown_pairwise[plan, benchmark])
            if represented > tolerance + _NUMERICAL_ATOL:
                budget = 0.0
                break
            slope = max(0.0, unknown - represented)
            if slope <= _NUMERICAL_ATOL:
                continue
            candidate = (tolerance - represented) / slope
            budget = min(budget, max(0.0, min(1.0, candidate)))
        budgets[plan] = budget
    return _immutable_float64(budgets)


class SupportRobustActSenseFallbackCertificateV1(NamedTuple):
    """Exact act/sense/fallback certificate under declared support miss."""

    represented_certificate: ActSenseFallbackCertificateV1
    support_miss_probability_upper: float
    unknown_terminal_loss_lower_by_action: FloatArray
    unknown_terminal_loss_upper_by_action: FloatArray
    unknown_probe_loss_lower: FloatArray
    unknown_probe_loss_upper: FloatArray
    unknown_plan_loss_lower: FloatArray
    unknown_plan_loss_upper: FloatArray
    unknown_pairwise_max_loss_gap: FloatArray
    support_robust_pairwise_worst_case_loss_gap: FloatArray
    support_robust_worst_case_regret: FloatArray
    maximum_admissible_support_miss_probability: FloatArray
    tolerance_admissible_plan_mask: BoolArray
    robustly_optimal_plan_mask: BoolArray
    minimax_plan_index: int
    minimax_worst_case_regret: float
    output_plan_index: int
    output_mode: str
    used_fallback: bool

    @property
    def plans(self) -> tuple[ContingentPlanV1, ...]:
        return self.represented_certificate.plans

    @property
    def plan_count(self) -> int:
        return len(self.plans)

    @property
    def regret_tolerance(self) -> float:
        return self.represented_certificate.plan_certificate.regret_tolerance

    @property
    def fallback_plan_index(self) -> int:
        return self.represented_certificate.fallback_plan_index

    @property
    def minimax_plan(self) -> ContingentPlanV1:
        return self.plans[self.minimax_plan_index]

    @property
    def output_plan(self) -> ContingentPlanV1:
        return self.plans[self.output_plan_index]

    @property
    def has_admissible_plan(self) -> bool:
        return bool(np.any(self.tolerance_admissible_plan_mask))

    @property
    def selected_probe_index(self) -> int | None:
        return self.output_plan.probe_index if self.output_mode == "sense" else None

    @property
    def selected_support_miss_budget(self) -> float:
        return float(
            self.maximum_admissible_support_miss_probability[self.output_plan_index]
        )

    def terminal_action(self, outcome_index: int | None = None) -> int:
        """Resolve the direct/fallback action or frozen post-probe action map."""

        return self.output_plan.terminal_action(outcome_index)

    def summary(self) -> dict[str, object]:
        return {
            "version": SUPPORT_ROBUST_ACT_SENSE_FALLBACK_VERSION,
            "semantics": SUPPORT_ROBUST_ACT_SENSE_FALLBACK_SEMANTICS,
            "hypothesis_count": (
                self.represented_certificate.plan_certificate.hypothesis_count
            ),
            "quotient_class_count": (
                self.represented_certificate.plan_certificate.quotient_class_count
            ),
            "terminal_action_count": self.represented_certificate.direct_plan_count,
            "probe_count": self.represented_certificate.probe_count,
            "plan_count": self.plan_count,
            "support_miss_probability_upper": self.support_miss_probability_upper,
            "regret_tolerance": self.regret_tolerance,
            "minimax_plan_index": self.minimax_plan_index,
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "has_admissible_plan": self.has_admissible_plan,
            "output_plan_index": self.output_plan_index,
            "output_mode": self.output_mode,
            "used_fallback": self.used_fallback,
            "selected_probe_index": self.selected_probe_index,
            "selected_support_miss_budget": self.selected_support_miss_budget,
            "claim_boundary": SUPPORT_ROBUST_ACT_SENSE_FALLBACK_CLAIM_BOUNDARY,
        }


def support_robust_act_sense_fallback_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    terminal_loss_by_hypothesis_action: object,
    probe_outcome_index_by_hypothesis: object,
    probe_costs: object,
    *,
    fallback_action_index: int,
    support_miss_probability_upper: float,
    unknown_terminal_loss_lower_by_action: object,
    unknown_terminal_loss_upper_by_action: object,
    unknown_probe_loss_lower: object,
    unknown_probe_loss_upper: object,
    regret_tolerance: float = 0.0,
    probe_names: Sequence[str] | None = None,
    max_plan_count: int = 100_000,
) -> SupportRobustActSenseFallbackCertificateV1:
    """Certify a finite act/sense plan under at-most-epsilon support miss.

    The represented contingent-plan roster is generated by
    :func:`act_sense_fallback_certificate`. Unknown physics is described only by
    an upper bound on its probability mass and lower/upper loss bounds for every
    terminal action and probe. For a sensing plan, an unknown state may realize
    any registered probe outcome, so the derived plan interval uses the minimum
    lower and maximum upper terminal loss across its frozen outcome map.

    The resulting pairwise formula is exact for the declared rectangular
    unknown-plan-loss ambiguity set. If no plan satisfies ``regret_tolerance``,
    the exact caller-owned fallback action is returned.
    """

    epsilon = _unit_interval(
        support_miss_probability_upper,
        name="support_miss_probability_upper",
    )
    base = act_sense_fallback_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        terminal_loss_by_hypothesis_action,
        probe_outcome_index_by_hypothesis,
        probe_costs,
        fallback_action_index=fallback_action_index,
        regret_tolerance=regret_tolerance,
        probe_names=probe_names,
        max_plan_count=max_plan_count,
    )
    action_count = base.direct_plan_count
    probe_count = base.probe_count
    terminal_lower, terminal_upper = _validated_box(
        unknown_terminal_loss_lower_by_action,
        unknown_terminal_loss_upper_by_action,
        size=action_count,
        stem="unknown_terminal_loss_by_action",
    )
    probe_lower, probe_upper = _validated_box(
        unknown_probe_loss_lower,
        unknown_probe_loss_upper,
        size=probe_count,
        stem="unknown_probe_loss",
    )
    plan_lower, plan_upper = _unknown_plan_loss_box(
        base,
        terminal_lower,
        terminal_upper,
        probe_lower,
        probe_upper,
    )

    unknown_pairwise = plan_upper[:, None] - plan_lower[None, :]
    np.fill_diagonal(unknown_pairwise, 0.0)
    represented_pairwise = base.plan_certificate.pairwise_worst_case_loss_gap
    robust_pairwise = represented_pairwise + epsilon * np.maximum(
        0.0,
        unknown_pairwise - represented_pairwise,
    )
    np.fill_diagonal(robust_pairwise, 0.0)
    robust_regret = np.maximum(np.max(robust_pairwise, axis=1), 0.0)

    minimum = float(np.min(robust_regret))
    minimax_candidates = np.flatnonzero(
        np.isclose(robust_regret, minimum, rtol=0.0, atol=_NUMERICAL_ATOL)
    )
    minimax_plan_index = int(minimax_candidates[0])
    tolerance = base.plan_certificate.regret_tolerance
    tolerance_mask = robust_regret <= tolerance + _NUMERICAL_ATOL
    robust_optimal_mask = np.all(robust_pairwise <= _NUMERICAL_ATOL, axis=1)
    budgets = _maximum_admissible_support_miss(
        represented_pairwise,
        unknown_pairwise,
        tolerance,
    )

    if not np.any(tolerance_mask):
        output_plan_index = base.fallback_plan_index
        output_mode = "fallback"
        used_fallback = True
    else:
        output_plan_index = minimax_plan_index
        plan = base.plans[output_plan_index]
        if plan.mode == "sense":
            output_mode = "sense"
            used_fallback = False
        elif plan.direct_action_index == base.fallback_action_index:
            output_mode = "fallback"
            used_fallback = True
        else:
            output_mode = "act"
            used_fallback = False

    if output_mode == "fallback" and output_plan_index != base.fallback_plan_index:
        raise RuntimeError("fallback output does not reproduce the caller-owned plan")
    if (
        output_mode != "fallback"
        and robust_regret[output_plan_index] > tolerance + _NUMERICAL_ATOL
    ):
        raise RuntimeError("nonfallback output exceeds the robust regret tolerance")

    return SupportRobustActSenseFallbackCertificateV1(
        represented_certificate=base,
        support_miss_probability_upper=epsilon,
        unknown_terminal_loss_lower_by_action=terminal_lower,
        unknown_terminal_loss_upper_by_action=terminal_upper,
        unknown_probe_loss_lower=probe_lower,
        unknown_probe_loss_upper=probe_upper,
        unknown_plan_loss_lower=plan_lower,
        unknown_plan_loss_upper=plan_upper,
        unknown_pairwise_max_loss_gap=_immutable_float64(unknown_pairwise),
        support_robust_pairwise_worst_case_loss_gap=_immutable_float64(robust_pairwise),
        support_robust_worst_case_regret=_immutable_float64(robust_regret),
        maximum_admissible_support_miss_probability=budgets,
        tolerance_admissible_plan_mask=_immutable_bool(tolerance_mask),
        robustly_optimal_plan_mask=_immutable_bool(robust_optimal_mask),
        minimax_plan_index=minimax_plan_index,
        minimax_worst_case_regret=minimum,
        output_plan_index=output_plan_index,
        output_mode=output_mode,
        used_fallback=used_fallback,
    )


__all__ = [
    "SUPPORT_ROBUST_ACT_SENSE_FALLBACK_CLAIM_BOUNDARY",
    "SUPPORT_ROBUST_ACT_SENSE_FALLBACK_SEMANTICS",
    "SUPPORT_ROBUST_ACT_SENSE_FALLBACK_VERSION",
    "SupportRobustActSenseFallbackCertificateV1",
    "support_robust_act_sense_fallback_certificate",
]
