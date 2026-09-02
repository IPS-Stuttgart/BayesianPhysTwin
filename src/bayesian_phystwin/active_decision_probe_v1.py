"""Exact active finite-decision probes under quotient ambiguity.

The passive query-decision certificate asks whether one action is uniformly
acceptable over every complete belief compatible with registered quotient
masses.  This module adds a finite probe whose observed outcome may be used to
choose a contingent terminal action.

For a deterministic terminal policy ``pi: outcome -> action`` and benchmark
policy ``beta``, the exact worst-case expected loss gap is

    sum_c lambda_c max_{i in c: p_i > 0}
        sum_o P(o | i, probe)
            [L(i, o, pi(o)) - L(i, o, beta(o))].

Maximizing over benchmark policies gives the exact worst-case regret of ``pi``;
minimizing over terminal policies gives the optimal robust contingent policy.
No complete within-class belief is selected.

The result is exact only for the supplied finite support, quotient masses, probe
outcome model, and outcome-conditioned loss table.  It does not validate probe
physics, support completeness, likelihood calibration, target transport, probe
cost, or deployment safety.  The implementation enumerates ``A**O`` deterministic
terminal policies, so it is intended for small registered outcome alphabets.
"""

from __future__ import annotations

import itertools
from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

ACTIVE_DECISION_PROBE_VERSION: Final = 1
ACTIVE_DECISION_PROBE_SEMANTICS: Final = (
    "exact-minimax-regret-for-finite-outcome-contingent-actions-over-"
    "registered-quotient-and-prior-support-v1"
)
ACTIVE_DECISION_PROBE_CLAIM_BOUNDARY: Final = (
    "The active-probe certificate is exact only for the supplied finite "
    "hypotheses, positive prior support, quotient masses, probe-outcome "
    "likelihood, outcome-conditioned losses, and deterministic terminal "
    "policies. It does not validate the physical probe, support completeness, "
    "likelihood calibration, loss model, target transport, probe cost, "
    "continuous actions, deployment, or safety."
)

_PROBABILITY_ATOL: Final = 1e-12
_NUMERICAL_ATOL: Final = 1e-12
_DEFAULT_MAX_POLICY_COUNT: Final = 4096


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


def _enumerate_policies(
    outcome_count: int,
    action_count: int,
    *,
    max_policy_count: int,
) -> IntArray:
    policy_count = action_count**outcome_count
    if policy_count > max_policy_count:
        raise ValueError(
            "deterministic terminal-policy count exceeds max_policy_count: "
            f"{action_count}^{outcome_count}={policy_count} > "
            f"{max_policy_count}"
        )
    policies = np.asarray(
        tuple(itertools.product(range(action_count), repeat=outcome_count)),
        dtype=np.int64,
    )
    return _immutable_int64(policies)


class ActiveDecisionProbeCertificateV1(NamedTuple):
    """Exact robust certificate for one finite-outcome probe."""

    prior_weights: FloatArray
    prior_support_mask: BoolArray
    quotient_weights: FloatArray
    class_index: IntArray
    outcome_likelihood: FloatArray
    loss_by_hypothesis_outcome_action: FloatArray
    terminal_policies: IntArray
    policy_worst_case_regret: FloatArray
    minimax_policy_index: int
    minimax_terminal_policy: IntArray
    minimax_worst_case_regret: float
    adversary_policy_index: int
    adversary_terminal_policy: IntArray
    selected_policy_pairwise_worst_case_gap: FloatArray
    regret_tolerance: float
    tolerance_admissible_policy_mask: BoolArray
    robustly_optimal_policy_mask: BoolArray

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

    @property
    def policy_count(self) -> int:
        return int(self.terminal_policies.shape[0])

    @property
    def has_tolerance_admissible_policy(self) -> bool:
        return bool(np.any(self.tolerance_admissible_policy_mask))

    @property
    def has_robustly_optimal_policy(self) -> bool:
        return bool(np.any(self.robustly_optimal_policy_mask))

    def summary(self) -> dict[str, object]:
        return {
            "version": ACTIVE_DECISION_PROBE_VERSION,
            "semantics": ACTIVE_DECISION_PROBE_SEMANTICS,
            "hypothesis_count": self.hypothesis_count,
            "prior_support_count": int(np.count_nonzero(self.prior_support_mask)),
            "quotient_class_count": self.quotient_class_count,
            "outcome_count": self.outcome_count,
            "action_count": self.action_count,
            "policy_count": self.policy_count,
            "minimax_policy_index": self.minimax_policy_index,
            "minimax_terminal_policy": self.minimax_terminal_policy.tolist(),
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "adversary_policy_index": self.adversary_policy_index,
            "adversary_terminal_policy": (self.adversary_terminal_policy.tolist()),
            "regret_tolerance": self.regret_tolerance,
            "has_tolerance_admissible_policy": (self.has_tolerance_admissible_policy),
            "has_robustly_optimal_policy": (self.has_robustly_optimal_policy),
            "claim_boundary": ACTIVE_DECISION_PROBE_CLAIM_BOUNDARY,
        }


def active_decision_probe_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    outcome_likelihood: object,
    loss_by_hypothesis_outcome_action: object,
    *,
    regret_tolerance: float = 0.0,
    max_policy_count: int = _DEFAULT_MAX_POLICY_COUNT,
) -> ActiveDecisionProbeCertificateV1:
    """Certify a deterministic outcome-contingent terminal action policy.

    The returned minimax policy is selected before observing the probe outcome.
    Its component for the realized outcome is then the terminal action.  The
    regret is uniform over every complete current belief compatible with the
    registered quotient masses and positive prior support.
    """

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
    policy_limit = _positive_integer(
        max_policy_count,
        name="max_policy_count",
    )

    prior_support = prior > 0.0
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
    policies = _enumerate_policies(
        outcome_count,
        action_count,
        max_policy_count=policy_limit,
    )
    policy_count = int(policies.shape[0])

    # Expected loss under each hypothesis and deterministic terminal policy:
    # E[i, pi] = sum_o P(o | i) L(i, o, pi(o)).
    expected_loss = np.zeros(
        (prior.size, policy_count),
        dtype=np.float64,
    )
    for outcome in range(outcome_count):
        expected_loss += (
            likelihood[:, outcome, None]
            * losses[:, outcome, :][:, policies[:, outcome]]
        )

    # For a fixed candidate policy pi and benchmark policy beta, the objective
    # is linear in the unknown complete belief.  With fixed mass lambda_c in
    # each quotient class, the worst supported belief independently puts that
    # class mass on its maximizing member.
    pairwise_worst_case_gap: FloatArray = np.zeros(
        (policy_count, policy_count),
        dtype=np.float64,
    )
    for class_id in range(class_count):
        members = expected_loss[(classes == class_id) & prior_support]
        if members.shape[0] == 0:
            continue
        class_pairwise_max: FloatArray = np.full(
            (policy_count, policy_count),
            -np.inf,
            dtype=np.float64,
        )
        for member_loss in members:
            class_pairwise_max = np.maximum(
                class_pairwise_max,
                member_loss[:, None] - member_loss[None, :],
            )
        pairwise_worst_case_gap += quotient[class_id] * class_pairwise_max
    np.fill_diagonal(pairwise_worst_case_gap, 0.0)

    policy_regret = np.maximum(
        np.max(pairwise_worst_case_gap, axis=1),
        0.0,
    )
    minimum_regret = float(np.min(policy_regret))
    minimax_indices = np.flatnonzero(
        np.isclose(
            policy_regret,
            minimum_regret,
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        )
    )
    minimax_index = int(minimax_indices[0])
    selected_gaps = pairwise_worst_case_gap[minimax_index]
    adversary_indices = np.flatnonzero(
        np.isclose(
            selected_gaps,
            float(np.max(selected_gaps)),
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        )
    )
    adversary_index = int(adversary_indices[0])

    tolerance_mask = policy_regret <= tolerance + _NUMERICAL_ATOL
    robust_mask = policy_regret <= _NUMERICAL_ATOL

    return ActiveDecisionProbeCertificateV1(
        prior_weights=prior,
        prior_support_mask=_immutable_bool(prior_support),
        quotient_weights=quotient,
        class_index=classes,
        outcome_likelihood=likelihood,
        loss_by_hypothesis_outcome_action=losses,
        terminal_policies=policies,
        policy_worst_case_regret=_immutable_float64(policy_regret),
        minimax_policy_index=minimax_index,
        minimax_terminal_policy=_immutable_int64(policies[minimax_index]),
        minimax_worst_case_regret=minimum_regret,
        adversary_policy_index=adversary_index,
        adversary_terminal_policy=_immutable_int64(policies[adversary_index]),
        selected_policy_pairwise_worst_case_gap=_immutable_float64(selected_gaps),
        regret_tolerance=tolerance,
        tolerance_admissible_policy_mask=_immutable_bool(tolerance_mask),
        robustly_optimal_policy_mask=_immutable_bool(robust_mask),
    )


class DecisionProbeCandidateV1(NamedTuple):
    """One source-registered finite probe candidate."""

    name: str
    cost: float
    outcome_likelihood: FloatArray
    loss_by_hypothesis_outcome_action: FloatArray


class ActiveDecisionProbeSelectionV1(NamedTuple):
    """Minimum-cost certified probe selection."""

    probe_names: tuple[str, ...]
    probe_costs: FloatArray
    certificates: tuple[ActiveDecisionProbeCertificateV1, ...]
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
            "version": ACTIVE_DECISION_PROBE_VERSION,
            "semantics": ACTIVE_DECISION_PROBE_SEMANTICS,
            "probe_names": list(self.probe_names),
            "probe_costs": self.probe_costs.tolist(),
            "probe_minimax_regrets": [
                certificate.minimax_worst_case_regret
                for certificate in self.certificates
            ],
            "admissible_probe_mask": self.admissible_probe_mask.tolist(),
            "selected_probe_index": self.selected_probe_index,
            "selected_probe_name": self.selected_probe_name,
            "fallback_required": self.fallback_required,
            "claim_boundary": ACTIVE_DECISION_PROBE_CLAIM_BOUNDARY,
        }


def decision_probe_candidate(
    name: str,
    cost: float,
    outcome_likelihood: object,
    loss_by_hypothesis_outcome_action: object,
) -> DecisionProbeCandidateV1:
    """Construct a lightweight probe candidate.

    Full dimension and probability validation occurs when the candidate is
    evaluated against a particular hypothesis support.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("probe name must be a nonempty string")
    candidate_cost = _finite_nonnegative(cost, name="probe cost")
    likelihood = np.ascontiguousarray(outcome_likelihood, dtype=np.float64)
    losses = np.ascontiguousarray(
        loss_by_hypothesis_outcome_action,
        dtype=np.float64,
    )
    likelihood.setflags(write=False)
    losses.setflags(write=False)
    return DecisionProbeCandidateV1(
        name=name.strip(),
        cost=candidate_cost,
        outcome_likelihood=likelihood,
        loss_by_hypothesis_outcome_action=losses,
    )


def select_minimum_cost_decision_probe(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    probes: tuple[DecisionProbeCandidateV1, ...],
    *,
    regret_tolerance: float = 0.0,
    max_policy_count: int = _DEFAULT_MAX_POLICY_COUNT,
) -> ActiveDecisionProbeSelectionV1:
    """Return the cheapest probe with a tolerance-admissible terminal policy.

    Ties are resolved by smaller minimax regret and then declaration order.
    When no probe passes, ``selected_probe_index`` is ``None`` and
    ``fallback_required`` is true.
    """

    if not isinstance(probes, tuple) or not probes:
        raise ValueError("probes must be a nonempty tuple")
    names = tuple(probe.name for probe in probes)
    if len(set(names)) != len(names):
        raise ValueError("probe names must be unique")
    costs = np.asarray([probe.cost for probe in probes], dtype=np.float64)
    certificates = tuple(
        active_decision_probe_certificate(
            prior_weights,
            quotient_weights,
            class_index,
            probe.outcome_likelihood,
            probe.loss_by_hypothesis_outcome_action,
            regret_tolerance=regret_tolerance,
            max_policy_count=max_policy_count,
        )
        for probe in probes
    )
    admissible = np.asarray(
        [
            certificate.minimax_worst_case_regret <= regret_tolerance + _NUMERICAL_ATOL
            for certificate in certificates
        ],
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
                certificates[index].minimax_worst_case_regret,
                index,
            ),
        )
    return ActiveDecisionProbeSelectionV1(
        probe_names=names,
        probe_costs=_immutable_float64(costs),
        certificates=certificates,
        admissible_probe_mask=_immutable_bool(admissible),
        selected_probe_index=selected,
        fallback_required=selected is None,
    )


__all__ = [
    "ACTIVE_DECISION_PROBE_CLAIM_BOUNDARY",
    "ACTIVE_DECISION_PROBE_SEMANTICS",
    "ACTIVE_DECISION_PROBE_VERSION",
    "ActiveDecisionProbeCertificateV1",
    "ActiveDecisionProbeSelectionV1",
    "DecisionProbeCandidateV1",
    "active_decision_probe_certificate",
    "decision_probe_candidate",
    "select_minimum_cost_decision_probe",
]
