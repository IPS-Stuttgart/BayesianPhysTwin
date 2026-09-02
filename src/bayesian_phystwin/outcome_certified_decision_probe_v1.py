"""Outcome-wise robust finite-decision certificates after an active probe.

The ex-ante active certificate in :mod:`active_decision_probe_v1` controls
expected regret before the probe outcome is observed.  This module provides the
strict complementary statement needed for deployment: after each possible
outcome, certify the selected terminal action against every posterior induced by
a complete current belief compatible with the registered quotient masses.

Let ``q`` range over complete current beliefs with fixed quotient masses, let
``K[i,o]`` be the likelihood of probe outcome ``o`` under hypothesis ``i``, and
let ``D[i]`` be a pairwise terminal loss difference for that outcome.  The
largest compatible posterior loss gap is

    sup_q sum_i q[i] K[i,o] D[i] / sum_i q[i] K[i,o].

For a candidate threshold ``t``, the numerator of the shifted ratio maximizes as

    phi(t) = sum_c lambda[c] max_{i in c, p[i] > 0} K[i,o] (D[i] - t).

The exact supremum is ``inf {t : phi(t) <= 0}``.  A monotone scalar bisection
therefore avoids enumerating the product of within-class simplex vertices.

The certificate is conditional on the supplied finite support, quotient masses,
probe likelihood, outcome-conditioned loss table, action set, and numerical
root tolerance.  It does not validate probe physics, support completeness,
likelihood calibration, loss transport, continuous control, deployment, or
safety.
"""

from __future__ import annotations

from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.active_decision_probe_v1 import DecisionProbeCandidateV1

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

OUTCOME_CERTIFIED_DECISION_PROBE_VERSION: Final = 1
OUTCOME_CERTIFIED_DECISION_PROBE_SEMANTICS: Final = (
    "post-outcome-minimax-regret-over-every-quotient-compatible-"
    "probe-conditioned-complete-belief-v1"
)
OUTCOME_CERTIFIED_DECISION_PROBE_CLAIM_BOUNDARY: Final = (
    "The post-outcome certificate is exact up to its declared scalar root "
    "tolerance only for the supplied finite hypotheses, positive prior "
    "support, quotient masses, probe likelihood, outcome-conditioned losses, "
    "and finite action set. It does not validate probe physics, support "
    "completeness, likelihood calibration, target transport, continuous "
    "control, deployment, or safety."
)

_PROBABILITY_ATOL: Final = 1e-12
_NUMERICAL_ATOL: Final = 1e-12
_DEFAULT_ROOT_TOLERANCE: Final = 1e-12
_DEFAULT_ROOT_ITERATIONS: Final = 128


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_int64(value: object) -> IntArray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    array.setflags(write=False)
    return array


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


def _probability_vector(
    value: object,
    *,
    name: str,
    expected_size: int | None = None,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"{name} must contain exactly {expected_size} entries")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    total = float(np.sum(array, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL):
        raise ValueError(f"{name} must sum to one")
    return _immutable_float64(array / total)


def _class_index(value: object, *, expected_size: int) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError("class_index must contain integer class labels")
    array = np.ascontiguousarray(raw, dtype=np.int64)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError(f"class_index must contain exactly {expected_size} entries")
    if np.any(array < 0):
        raise ValueError("class_index labels must be nonnegative")
    unique = np.unique(array)
    if not np.array_equal(
        unique,
        np.arange(int(unique[-1]) + 1, dtype=np.int64),
    ):
        raise ValueError("class_index labels must be contiguous from zero")
    return _immutable_int64(array)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _outcome_likelihood(value: object, *, hypothesis_count: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("outcome_likelihood must contain real numeric values")
    likelihood = np.array(raw, dtype=np.float64, copy=True, order="C")
    if (
        likelihood.ndim != 2
        or likelihood.shape[0] != hypothesis_count
        or likelihood.shape[1] < 1
    ):
        raise ValueError(
            "outcome_likelihood must have shape (hypothesis_count, outcome_count)"
        )
    if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0.0):
        raise ValueError("outcome_likelihood must contain finite nonnegative values")
    totals = np.sum(likelihood, axis=1, dtype=np.float64)
    if not np.allclose(
        totals,
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_ATOL,
    ):
        raise ValueError("every outcome_likelihood row must sum to one")
    likelihood /= totals[:, None]
    return _immutable_float64(likelihood)


def _loss_tensor(
    value: object,
    *,
    hypothesis_count: int,
    outcome_count: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(
            "loss_by_hypothesis_outcome_action must contain real numeric values"
        )
    losses = np.ascontiguousarray(raw, dtype=np.float64)
    if losses.ndim == 2:
        if losses.shape[0] != hypothesis_count or losses.shape[1] < 2:
            raise ValueError(
                "two-dimensional losses must have shape "
                "(hypothesis_count, action_count) with at least two actions"
            )
        losses = np.repeat(losses[:, None, :], outcome_count, axis=1)
    elif (
        losses.ndim != 3
        or losses.shape[0] != hypothesis_count
        or losses.shape[1] != outcome_count
        or losses.shape[2] < 2
    ):
        raise ValueError(
            "loss_by_hypothesis_outcome_action must have shape "
            "(hypothesis_count, outcome_count, action_count), or "
            "(hypothesis_count, action_count)"
        )
    if not np.all(np.isfinite(losses)):
        raise ValueError("loss_by_hypothesis_outcome_action must be finite")
    return _immutable_float64(losses)


def _maximum_outcome_probability(
    likelihood: FloatArray,
    classes: IntArray,
    quotient: FloatArray,
    support: BoolArray,
) -> float:
    result = 0.0
    for class_id, mass in enumerate(quotient):
        if mass <= 0.0:
            continue
        members = (classes == class_id) & support
        result += float(mass) * float(np.max(likelihood[members]))
    return result


def _fractional_support_function(
    difference: FloatArray,
    likelihood: FloatArray,
    classes: IntArray,
    quotient: FloatArray,
    support: BoolArray,
    *,
    root_tolerance: float,
    root_iterations: int,
) -> float:
    positive = support & (quotient[classes] > 0.0) & (likelihood > 0.0)
    if not np.any(positive):
        raise ValueError("cannot condition on an impossible probe outcome")
    lower = float(np.min(difference[positive]))
    upper = float(np.max(difference[positive]))
    if np.isclose(lower, upper, rtol=0.0, atol=root_tolerance):
        return upper

    def shifted_support(threshold: float) -> float:
        result = 0.0
        for class_id, mass in enumerate(quotient):
            if mass <= 0.0:
                continue
            members = (classes == class_id) & support
            values = likelihood[members] * (difference[members] - threshold)
            result += float(mass) * float(np.max(values))
        return result

    scale = max(1.0, abs(lower), abs(upper))
    tolerance = root_tolerance * scale
    if shifted_support(lower) <= 0.0:
        return lower
    lo = lower
    hi = upper
    for _ in range(root_iterations):
        middle = 0.5 * (lo + hi)
        if shifted_support(middle) > 0.0:
            lo = middle
        else:
            hi = middle
        if hi - lo <= tolerance:
            break
    return hi


class OutcomeCertifiedDecisionProbeV1(NamedTuple):
    """Post-outcome robust action certificate for one finite probe."""

    prior_weights: FloatArray
    prior_support_mask: BoolArray
    quotient_weights: FloatArray
    class_index: IntArray
    outcome_likelihood: FloatArray
    loss_by_hypothesis_outcome_action: FloatArray
    reachable_outcome_mask: BoolArray
    maximum_outcome_probability: FloatArray
    outcome_pairwise_worst_case_gap: FloatArray
    outcome_action_worst_case_regret: FloatArray
    outcome_minimax_action_index: IntArray
    outcome_minimax_worst_case_regret: FloatArray
    outcome_tolerance_certified_mask: BoolArray
    regret_tolerance: float
    worst_reachable_outcome_regret: float
    all_reachable_outcomes_certified: bool

    @property
    def hypothesis_count(self) -> int:
        return int(self.class_index.size)

    @property
    def quotient_class_count(self) -> int:
        return int(self.quotient_weights.size)

    @property
    def outcome_count(self) -> int:
        return int(self.outcome_likelihood.shape[1])

    @property
    def action_count(self) -> int:
        return int(self.loss_by_hypothesis_outcome_action.shape[2])

    def summary(self) -> dict[str, object]:
        return {
            "version": OUTCOME_CERTIFIED_DECISION_PROBE_VERSION,
            "semantics": OUTCOME_CERTIFIED_DECISION_PROBE_SEMANTICS,
            "hypothesis_count": self.hypothesis_count,
            "prior_support_count": int(np.count_nonzero(self.prior_support_mask)),
            "quotient_class_count": self.quotient_class_count,
            "outcome_count": self.outcome_count,
            "action_count": self.action_count,
            "reachable_outcome_mask": self.reachable_outcome_mask.tolist(),
            "maximum_outcome_probability": (self.maximum_outcome_probability.tolist()),
            "outcome_minimax_action_index": (
                self.outcome_minimax_action_index.tolist()
            ),
            "outcome_minimax_worst_case_regret": (
                self.outcome_minimax_worst_case_regret.tolist()
            ),
            "outcome_tolerance_certified_mask": (
                self.outcome_tolerance_certified_mask.tolist()
            ),
            "regret_tolerance": self.regret_tolerance,
            "worst_reachable_outcome_regret": (self.worst_reachable_outcome_regret),
            "all_reachable_outcomes_certified": (self.all_reachable_outcomes_certified),
            "claim_boundary": (OUTCOME_CERTIFIED_DECISION_PROBE_CLAIM_BOUNDARY),
        }


def outcome_certified_decision_probe(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    outcome_likelihood: object,
    loss_by_hypothesis_outcome_action: object,
    *,
    regret_tolerance: float = 0.0,
    root_tolerance: float = _DEFAULT_ROOT_TOLERANCE,
    root_iterations: int = _DEFAULT_ROOT_ITERATIONS,
) -> OutcomeCertifiedDecisionProbeV1:
    """Certify a terminal action after every possible probe outcome."""

    prior = _probability_vector(prior_weights, name="prior_weights")
    classes = _class_index(class_index, expected_size=prior.size)
    class_count = int(np.max(classes)) + 1
    quotient = _probability_vector(
        quotient_weights,
        name="quotient_weights",
        expected_size=class_count,
    )
    likelihood = _outcome_likelihood(
        outcome_likelihood,
        hypothesis_count=prior.size,
    )
    losses = _loss_tensor(
        loss_by_hypothesis_outcome_action,
        hypothesis_count=prior.size,
        outcome_count=likelihood.shape[1],
    )
    tolerance = _finite_nonnegative(
        regret_tolerance,
        name="regret_tolerance",
    )
    ratio_tolerance = _finite_nonnegative(
        root_tolerance,
        name="root_tolerance",
    )
    if ratio_tolerance <= 0.0:
        raise ValueError("root_tolerance must be positive")
    iterations = _positive_integer(
        root_iterations,
        name="root_iterations",
    )

    support = prior > 0.0
    prior_quotient = np.bincount(
        classes,
        weights=prior,
        minlength=class_count,
    ).astype(np.float64, copy=False)
    unsupported_classes = (quotient > 0.0) & (prior_quotient <= 0.0)
    if np.any(unsupported_classes):
        raise ValueError(
            "positive quotient mass has zero prior support for classes "
            f"{np.flatnonzero(unsupported_classes).tolist()}"
        )

    outcome_count = int(likelihood.shape[1])
    action_count = int(losses.shape[2])
    reachable: BoolArray = np.zeros(outcome_count, dtype=np.bool_)
    maximum_probability: FloatArray = np.zeros(outcome_count, dtype=np.float64)
    pairwise: FloatArray = np.zeros(
        (outcome_count, action_count, action_count),
        dtype=np.float64,
    )
    action_regret: FloatArray = np.zeros(
        (outcome_count, action_count),
        dtype=np.float64,
    )
    minimax_action: IntArray = np.full(outcome_count, -1, dtype=np.int64)
    minimax_regret: FloatArray = np.zeros(outcome_count, dtype=np.float64)
    certified: BoolArray = np.zeros(outcome_count, dtype=np.bool_)

    for outcome in range(outcome_count):
        outcome_likelihood_vector = likelihood[:, outcome]
        maximum_probability[outcome] = _maximum_outcome_probability(
            outcome_likelihood_vector,
            classes,
            quotient,
            support,
        )
        if maximum_probability[outcome] <= _PROBABILITY_ATOL:
            continue
        reachable[outcome] = True
        for action in range(action_count):
            for benchmark in range(action_count):
                if action == benchmark:
                    continue
                pairwise[outcome, action, benchmark] = _fractional_support_function(
                    losses[:, outcome, action] - losses[:, outcome, benchmark],
                    outcome_likelihood_vector,
                    classes,
                    quotient,
                    support,
                    root_tolerance=ratio_tolerance,
                    root_iterations=iterations,
                )
        action_regret[outcome] = np.maximum(
            np.max(pairwise[outcome], axis=1),
            0.0,
        )
        minimum = float(np.min(action_regret[outcome]))
        candidates = np.flatnonzero(
            np.isclose(
                action_regret[outcome],
                minimum,
                rtol=0.0,
                atol=max(_NUMERICAL_ATOL, ratio_tolerance),
            )
        )
        minimax_action[outcome] = int(candidates[0])
        minimax_regret[outcome] = minimum
        certified[outcome] = minimum <= tolerance + max(
            _NUMERICAL_ATOL,
            ratio_tolerance,
        )

    if not np.any(reachable):
        raise ValueError("probe has no reachable outcome")
    worst_reachable = float(np.max(minimax_regret[reachable]))
    all_certified = bool(np.all(certified[reachable]))
    return OutcomeCertifiedDecisionProbeV1(
        prior_weights=prior,
        prior_support_mask=_immutable_bool(support),
        quotient_weights=quotient,
        class_index=classes,
        outcome_likelihood=likelihood,
        loss_by_hypothesis_outcome_action=losses,
        reachable_outcome_mask=_immutable_bool(reachable),
        maximum_outcome_probability=_immutable_float64(maximum_probability),
        outcome_pairwise_worst_case_gap=_immutable_float64(pairwise),
        outcome_action_worst_case_regret=_immutable_float64(action_regret),
        outcome_minimax_action_index=_immutable_int64(minimax_action),
        outcome_minimax_worst_case_regret=_immutable_float64(minimax_regret),
        outcome_tolerance_certified_mask=_immutable_bool(certified),
        regret_tolerance=tolerance,
        worst_reachable_outcome_regret=worst_reachable,
        all_reachable_outcomes_certified=all_certified,
    )


class OutcomeCertifiedProbeSelectionV1(NamedTuple):
    """Minimum-cost probe whose every reachable outcome is certified."""

    probe_names: tuple[str, ...]
    probe_costs: FloatArray
    certificates: tuple[OutcomeCertifiedDecisionProbeV1, ...]
    admissible_probe_mask: BoolArray
    selected_probe_index: int | None
    fallback_required: bool

    @property
    def selected_probe_name(self) -> str | None:
        if self.selected_probe_index is None:
            return None
        return self.probe_names[self.selected_probe_index]

    def summary(self) -> dict[str, object]:
        return {
            "version": OUTCOME_CERTIFIED_DECISION_PROBE_VERSION,
            "semantics": OUTCOME_CERTIFIED_DECISION_PROBE_SEMANTICS,
            "probe_names": list(self.probe_names),
            "probe_costs": self.probe_costs.tolist(),
            "probe_worst_reachable_outcome_regret": [
                certificate.worst_reachable_outcome_regret
                for certificate in self.certificates
            ],
            "admissible_probe_mask": self.admissible_probe_mask.tolist(),
            "selected_probe_index": self.selected_probe_index,
            "selected_probe_name": self.selected_probe_name,
            "fallback_required": self.fallback_required,
            "claim_boundary": (OUTCOME_CERTIFIED_DECISION_PROBE_CLAIM_BOUNDARY),
        }


def select_minimum_cost_outcome_certified_probe(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    probes: tuple[DecisionProbeCandidateV1, ...],
    *,
    regret_tolerance: float = 0.0,
    root_tolerance: float = _DEFAULT_ROOT_TOLERANCE,
    root_iterations: int = _DEFAULT_ROOT_ITERATIONS,
) -> OutcomeCertifiedProbeSelectionV1:
    """Select the cheapest probe certified after every reachable outcome."""

    if not isinstance(probes, tuple) or not probes:
        raise ValueError("probes must be a nonempty tuple")
    names = tuple(probe.name for probe in probes)
    if len(set(names)) != len(names):
        raise ValueError("probe names must be unique")
    costs = np.asarray([probe.cost for probe in probes], dtype=np.float64)
    certificates = tuple(
        outcome_certified_decision_probe(
            prior_weights,
            quotient_weights,
            class_index,
            probe.outcome_likelihood,
            probe.loss_by_hypothesis_outcome_action,
            regret_tolerance=regret_tolerance,
            root_tolerance=root_tolerance,
            root_iterations=root_iterations,
        )
        for probe in probes
    )
    admissible = np.asarray(
        [certificate.all_reachable_outcomes_certified for certificate in certificates],
        dtype=np.bool_,
    )
    candidates = np.flatnonzero(admissible)
    selected: int | None
    if candidates.size == 0:
        selected = None
    else:
        selected = min(
            (int(index) for index in candidates),
            key=lambda index: (
                float(costs[index]),
                certificates[index].worst_reachable_outcome_regret,
                index,
            ),
        )
    return OutcomeCertifiedProbeSelectionV1(
        probe_names=names,
        probe_costs=_immutable_float64(costs),
        certificates=certificates,
        admissible_probe_mask=_immutable_bool(admissible),
        selected_probe_index=selected,
        fallback_required=selected is None,
    )


__all__ = [
    "OUTCOME_CERTIFIED_DECISION_PROBE_CLAIM_BOUNDARY",
    "OUTCOME_CERTIFIED_DECISION_PROBE_SEMANTICS",
    "OUTCOME_CERTIFIED_DECISION_PROBE_VERSION",
    "OutcomeCertifiedDecisionProbeV1",
    "OutcomeCertifiedProbeSelectionV1",
    "outcome_certified_decision_probe",
    "select_minimum_cost_outcome_certified_probe",
]
