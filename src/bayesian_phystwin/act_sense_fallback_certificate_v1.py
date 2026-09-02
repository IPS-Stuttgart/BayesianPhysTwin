"""Exact act--sense--fallback certificates on a registered quotient belief.

The terminal decision certificate treats each complete contingent sensing policy
as one finite action.  For a deterministic registered probe, a plan maps every
probe outcome to a terminal physical action.  Enumerating those plans and
passing their hypothesis-wise total losses to :func:`query_decision_certificate`
therefore yields an exact worst-case-regret certificate over every complete
belief compatible with the supplied quotient masses and prior support.

This construction never forms a within-class point state or posterior.  It
certifies the complete plan *before* the probe is executed.  After the registered
outcome is observed, the plan's frozen action map is applied.

The certificate does not validate probe physics, outcome repeatability, reset
semantics, action losses, probe costs, quotient correctness, target transport,
calibration, deployment, or safety.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from numbers import Integral
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.query_decision_certificate_v1 import (
    QueryDecisionCertificateV1,
    query_decision_certificate,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

ACT_SENSE_FALLBACK_CERTIFICATE_VERSION: Final = 1
ACT_SENSE_FALLBACK_CERTIFICATE_SEMANTICS: Final = (
    "exact-preprobe-contingent-plan-regret-over-registered-quotient-v1"
)
ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied finite hypotheses, prior "
    "support, quotient masses, terminal loss matrix, deterministic registered "
    "probe-outcome maps, scalar probe costs, enumerated contingent plans, and "
    "regret tolerance. It does not validate the physical probe model, reset or "
    "repeatability assumptions, action losses, costs, quotient, provider, "
    "target-domain transport, online execution, calibration, deployment, or safety."
)

_NUMERICAL_ATOL: Final = 1e-12


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_int64(value: object) -> IntArray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    array.setflags(write=False)
    return array


def _finite_loss_matrix(value: object) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(
            "terminal_loss_by_hypothesis_action must contain real numeric values"
        )
    losses = np.ascontiguousarray(raw, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] < 2:
        raise ValueError(
            "terminal_loss_by_hypothesis_action must have shape "
            "(positive_hypothesis_count, action_count>=2)"
        )
    if not np.all(np.isfinite(losses)):
        raise ValueError("terminal_loss_by_hypothesis_action must be finite")
    return _immutable_float64(losses)


def _probe_outcomes(value: object, *, hypothesis_count: int) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError("probe_outcome_index_by_hypothesis must contain integers")
    outcomes = np.ascontiguousarray(raw, dtype=np.int64)
    if outcomes.ndim != 2 or outcomes.shape[1] != hypothesis_count:
        raise ValueError(
            "probe_outcome_index_by_hypothesis must have shape "
            "(probe_count, hypothesis_count)"
        )
    if np.any(outcomes < 0):
        raise ValueError("probe outcome labels must be nonnegative")
    for probe_index, row in enumerate(outcomes):
        unique = np.unique(row)
        expected: IntArray = np.arange(int(unique[-1]) + 1, dtype=np.int64)
        if not np.array_equal(unique, expected):
            raise ValueError(
                f"probe {probe_index} outcome labels must be contiguous from zero"
            )
    return _immutable_int64(outcomes)


def _nonnegative_vector(
    value: object,
    *,
    expected_size: int,
    name: str,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    vector = np.ascontiguousarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size != expected_size:
        raise ValueError(f"{name} must contain exactly {expected_size} entries")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return _immutable_float64(vector)


def _index(value: object, *, upper: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result >= upper:
        raise ValueError(f"{name} must be in [0, {upper})")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _probe_names(value: Sequence[str] | None, *, probe_count: int) -> tuple[str, ...]:
    if value is None:
        return tuple(f"probe_{index}" for index in range(probe_count))
    names = tuple(value)
    if len(names) != probe_count:
        raise ValueError(f"probe_names must contain exactly {probe_count} entries")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("probe_names must contain nonempty strings")
    normalized = tuple(name.strip() for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("probe_names must be unique")
    return normalized


class ContingentPlanV1(NamedTuple):
    """One direct or probe-contingent physical decision plan."""

    mode: str
    direct_action_index: int | None
    probe_index: int | None
    probe_name: str | None
    terminal_action_by_outcome: IntArray

    @property
    def outcome_count(self) -> int:
        return int(self.terminal_action_by_outcome.size)

    def terminal_action(self, outcome_index: int | None = None) -> int:
        """Resolve the terminal action for a direct plan or observed probe outcome."""

        if self.mode == "act":
            if outcome_index is not None:
                raise ValueError("direct plans do not accept a probe outcome")
            if self.direct_action_index is None:
                raise RuntimeError("direct plan is missing its action")
            return self.direct_action_index
        if self.mode != "sense":
            raise RuntimeError(f"unsupported plan mode: {self.mode}")
        if outcome_index is None:
            raise ValueError("a sensing plan requires an observed outcome")
        index = _index(
            outcome_index,
            upper=self.outcome_count,
            name="outcome_index",
        )
        return int(self.terminal_action_by_outcome[index])


class ActSenseFallbackCertificateV1(NamedTuple):
    """Exact quotient certificate over direct and contingent sensing plans."""

    plan_certificate: QueryDecisionCertificateV1
    plans: tuple[ContingentPlanV1, ...]
    plan_loss_by_hypothesis: FloatArray
    direct_plan_count: int
    probe_count: int
    sensing_plan_count: int
    fallback_action_index: int
    fallback_plan_index: int
    minimax_plan_index: int
    output_plan_index: int
    output_mode: str
    used_fallback: bool

    @property
    def plan_count(self) -> int:
        return len(self.plans)

    @property
    def minimax_plan(self) -> ContingentPlanV1:
        return self.plans[self.minimax_plan_index]

    @property
    def output_plan(self) -> ContingentPlanV1:
        return self.plans[self.output_plan_index]

    @property
    def has_admissible_plan(self) -> bool:
        return self.plan_certificate.has_tolerance_admissible_action

    @property
    def selected_probe_index(self) -> int | None:
        return self.output_plan.probe_index if self.output_mode == "sense" else None

    def terminal_action(self, outcome_index: int | None = None) -> int:
        """Return the direct/fallback action or resolve the frozen sensing map."""

        return self.output_plan.terminal_action(outcome_index)

    def summary(self) -> dict[str, object]:
        return {
            "version": ACT_SENSE_FALLBACK_CERTIFICATE_VERSION,
            "semantics": ACT_SENSE_FALLBACK_CERTIFICATE_SEMANTICS,
            "hypothesis_count": self.plan_certificate.hypothesis_count,
            "quotient_class_count": self.plan_certificate.quotient_class_count,
            "terminal_action_count": self.direct_plan_count,
            "probe_count": self.probe_count,
            "plan_count": self.plan_count,
            "direct_plan_count": self.direct_plan_count,
            "sensing_plan_count": self.sensing_plan_count,
            "fallback_action_index": self.fallback_action_index,
            "minimax_plan_index": self.minimax_plan_index,
            "minimax_worst_case_regret": (
                self.plan_certificate.minimax_worst_case_regret
            ),
            "regret_tolerance": self.plan_certificate.regret_tolerance,
            "has_admissible_plan": self.has_admissible_plan,
            "output_plan_index": self.output_plan_index,
            "output_mode": self.output_mode,
            "used_fallback": self.used_fallback,
            "selected_probe_index": self.selected_probe_index,
            "claim_boundary": ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY,
        }


def act_sense_fallback_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    terminal_loss_by_hypothesis_action: object,
    probe_outcome_index_by_hypothesis: object,
    probe_costs: object,
    *,
    fallback_action_index: int,
    regret_tolerance: float = 0.0,
    probe_names: Sequence[str] | None = None,
    max_plan_count: int = 100_000,
) -> ActSenseFallbackCertificateV1:
    """Certify whether to act, execute a deterministic probe, or fall back.

    Every direct terminal action is included as one plan.  For each deterministic
    probe with ``K`` outcomes, all ``A**K`` mappings from outcomes to terminal
    actions are enumerated.  Probe cost is added to the terminal loss under every
    hypothesis.  The existing exact quotient certificate then evaluates the
    expanded finite plan set without constructing any within-class point belief.

    Direct plans are enumerated first, so exact numerical ties prefer acting over
    sensing.  If the minimax plan exceeds ``regret_tolerance``, the caller-owned
    fallback action is returned exactly.
    """

    losses = _finite_loss_matrix(terminal_loss_by_hypothesis_action)
    hypothesis_count, action_count = losses.shape
    outcomes = _probe_outcomes(
        probe_outcome_index_by_hypothesis,
        hypothesis_count=hypothesis_count,
    )
    probe_count = outcomes.shape[0]
    costs = _nonnegative_vector(
        probe_costs,
        expected_size=probe_count,
        name="probe_costs",
    )
    names = _probe_names(probe_names, probe_count=probe_count)
    fallback = _index(
        fallback_action_index,
        upper=action_count,
        name="fallback_action_index",
    )
    plan_cap = _positive_integer(max_plan_count, name="max_plan_count")

    projected_plan_count = action_count
    outcome_counts: list[int] = []
    for row in outcomes:
        outcome_count = int(np.max(row)) + 1
        outcome_counts.append(outcome_count)
        projected_plan_count += action_count**outcome_count
        if projected_plan_count > plan_cap:
            raise ValueError(
                "enumerated contingent plan count exceeds max_plan_count: "
                f"{projected_plan_count} > {plan_cap}"
            )

    plan_losses: list[FloatArray] = []
    plans: list[ContingentPlanV1] = []
    empty_map = _immutable_int64(np.empty(0, dtype=np.int64))
    for action_index in range(action_count):
        plan_losses.append(losses[:, action_index])
        plans.append(
            ContingentPlanV1(
                mode="act",
                direct_action_index=action_index,
                probe_index=None,
                probe_name=None,
                terminal_action_by_outcome=empty_map,
            )
        )

    hypothesis_index: IntArray = np.arange(hypothesis_count, dtype=np.int64)
    for probe_index, (outcome_row, outcome_count) in enumerate(
        zip(outcomes, outcome_counts, strict=True)
    ):
        for mapping_tuple in itertools.product(
            range(action_count),
            repeat=outcome_count,
        ):
            mapping = _immutable_int64(mapping_tuple)
            terminal_actions = mapping[outcome_row]
            total_loss = (
                costs[probe_index]
                + losses[
                    hypothesis_index,
                    terminal_actions,
                ]
            )
            plan_losses.append(_immutable_float64(total_loss))
            plans.append(
                ContingentPlanV1(
                    mode="sense",
                    direct_action_index=None,
                    probe_index=probe_index,
                    probe_name=names[probe_index],
                    terminal_action_by_outcome=mapping,
                )
            )

    plan_loss_matrix = _immutable_float64(np.column_stack(plan_losses))
    plan_certificate = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        plan_loss_matrix,
        regret_tolerance=regret_tolerance,
    )
    minimax_plan_index = plan_certificate.minimax_action_index
    fallback_plan_index = fallback

    if not plan_certificate.has_tolerance_admissible_action:
        output_plan_index = fallback_plan_index
        output_mode = "fallback"
        used_fallback = True
    else:
        output_plan_index = minimax_plan_index
        plan = plans[output_plan_index]
        if plan.mode == "sense":
            output_mode = "sense"
            used_fallback = False
        elif plan.direct_action_index == fallback:
            output_mode = "fallback"
            used_fallback = True
        else:
            output_mode = "act"
            used_fallback = False

    if output_mode == "fallback" and output_plan_index != fallback_plan_index:
        raise RuntimeError("fallback output does not reproduce the caller-owned action")
    if (
        output_mode != "fallback"
        and plan_certificate.worst_case_regret[output_plan_index]
        > plan_certificate.regret_tolerance + _NUMERICAL_ATOL
    ):
        raise RuntimeError("nonfallback output exceeds the registered regret tolerance")

    return ActSenseFallbackCertificateV1(
        plan_certificate=plan_certificate,
        plans=tuple(plans),
        plan_loss_by_hypothesis=plan_loss_matrix,
        direct_plan_count=action_count,
        probe_count=probe_count,
        sensing_plan_count=len(plans) - action_count,
        fallback_action_index=fallback,
        fallback_plan_index=fallback_plan_index,
        minimax_plan_index=minimax_plan_index,
        output_plan_index=output_plan_index,
        output_mode=output_mode,
        used_fallback=used_fallback,
    )


__all__ = [
    "ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY",
    "ACT_SENSE_FALLBACK_CERTIFICATE_SEMANTICS",
    "ACT_SENSE_FALLBACK_CERTIFICATE_VERSION",
    "ActSenseFallbackCertificateV1",
    "ContingentPlanV1",
    "act_sense_fallback_certificate",
]
