"""Exact finite-hypothesis certificates for act, probe, or fallback decisions.

A query quotient may identify posterior mass only at the class level.  The
existing :mod:`query_decision_certificate_v1` computes exact worst-case regret
for a direct finite action over every prior-supported complete lift.  This
module lifts the same construction to a registered finite probe.

For probe outcome likelihood ``O[i, z]`` and contingent policy ``delta(z)``,
define the hypothesis-wise policy loss

    G[i, delta] = sum_z O[i, z] L[i, delta(z)].

A contingent policy is simply a finite meta-action.  Therefore its exact
worst-case regret over all complete beliefs compatible with the registered
quotient masses is obtained by applying the direct certificate to ``G``.  In
expanded form,

    Reg_bar(delta)
      = max_delta' sum_c lambda[c]
          max_{i in c: p[i] > 0}
            sum_z O[i,z] (L[i,delta(z)] - L[i,delta'(z)]).

The result supports an auditable act/probe/fallback router.  It does not validate
the probe likelihood, physical hypothesis set, quotient, loss, cost, population,
or deployment context.
"""

from __future__ import annotations

import itertools
from numbers import Integral, Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from .query_decision_certificate_v1 import (
    QueryDecisionCertificateV1,
    query_decision_certificate,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

QUERY_PROBE_CERTIFICATE_VERSION: Final = 1
QUERY_PROBE_CERTIFICATE_SEMANTICS: Final = (
    "exact-ex-ante-worst-case-regret-for-finite-contingent-probe-policies-v1"
)
QUERY_PROBE_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied finite hypotheses, positive "
    "prior support, quotient masses, action losses, and probe-outcome likelihood. "
    "It is an ex-ante expected-regret statement. It does not validate the physical "
    "hypotheses, quotient, likelihood, costs, outcome model, exchangeability, "
    "transport, calibration, deployment context, or safety."
)

_PROBABILITY_ATOL: Final = 1e-12
_ARRAY_ATOL: Final = 1e-12
_DEFAULT_MAX_POLICY_COUNT: Final = 100_000


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_int64(value: object) -> IntArray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    array.setflags(write=False)
    return array


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _loss_matrix(value: object) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("loss_by_hypothesis_action must contain real numeric values")
    losses = np.ascontiguousarray(raw, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] < 2:
        raise ValueError(
            "loss_by_hypothesis_action must have shape "
            "(positive_hypothesis_count, action_count>=2)"
        )
    if not np.all(np.isfinite(losses)):
        raise ValueError("loss_by_hypothesis_action must be finite")
    return _immutable_float64(losses)


def _likelihood_matrix(value: object, *, hypothesis_count: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("outcome_likelihood_by_hypothesis must be numeric")
    likelihood = np.ascontiguousarray(raw, dtype=np.float64)
    if (
        likelihood.ndim != 2
        or likelihood.shape[0] != hypothesis_count
        or likelihood.shape[1] == 0
    ):
        raise ValueError(
            "outcome_likelihood_by_hypothesis must have shape "
            "(hypothesis_count, positive_outcome_count)"
        )
    if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0.0):
        raise ValueError("outcome likelihood must be finite and nonnegative")
    row_sum = np.sum(likelihood, axis=1, dtype=np.float64)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL):
        raise ValueError("every hypothesis likelihood row must sum to one")
    likelihood = likelihood / row_sum[:, None]
    return _immutable_float64(likelihood)


def _contingent_policies(
    action_count: int,
    outcome_count: int,
    *,
    max_policy_count: int,
) -> IntArray:
    policy_count = pow(action_count, outcome_count)
    if policy_count > max_policy_count:
        raise ValueError(
            f"contingent policy count {policy_count} exceeds cap {max_policy_count}"
        )
    policies = np.fromiter(
        (
            action
            for policy in itertools.product(range(action_count), repeat=outcome_count)
            for action in policy
        ),
        dtype=np.int64,
        count=policy_count * outcome_count,
    ).reshape(policy_count, outcome_count)
    return _immutable_int64(policies)


class QueryProbeCertificateV1(NamedTuple):
    """Exact ex-ante certificate for one registered finite probe."""

    outcome_likelihood_by_hypothesis: FloatArray
    contingent_action_indices: IntArray
    policy_loss_by_hypothesis: FloatArray
    policy_decision_certificate: QueryDecisionCertificateV1
    probe_cost: float
    minimax_contingent_action_indices: IntArray
    minimax_worst_case_regret: float
    total_regret_plus_cost: float

    @property
    def hypothesis_count(self) -> int:
        return int(self.outcome_likelihood_by_hypothesis.shape[0])

    @property
    def outcome_count(self) -> int:
        return int(self.outcome_likelihood_by_hypothesis.shape[1])

    @property
    def policy_count(self) -> int:
        return int(self.contingent_action_indices.shape[0])

    @property
    def action_count(self) -> int:
        return int(np.max(self.contingent_action_indices)) + 1

    @property
    def has_tolerance_admissible_policy(self) -> bool:
        return self.policy_decision_certificate.has_tolerance_admissible_action

    def summary(self) -> dict[str, object]:
        return {
            "version": QUERY_PROBE_CERTIFICATE_VERSION,
            "semantics": QUERY_PROBE_CERTIFICATE_SEMANTICS,
            "hypothesis_count": self.hypothesis_count,
            "outcome_count": self.outcome_count,
            "action_count": self.action_count,
            "policy_count": self.policy_count,
            "probe_cost": self.probe_cost,
            "minimax_policy_index": (
                self.policy_decision_certificate.minimax_action_index
            ),
            "minimax_contingent_action_indices": (
                self.minimax_contingent_action_indices.tolist()
            ),
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "total_regret_plus_cost": self.total_regret_plus_cost,
            "has_tolerance_admissible_policy": (self.has_tolerance_admissible_policy),
            "claim_boundary": QUERY_PROBE_CERTIFICATE_CLAIM_BOUNDARY,
        }


class ActProbeFallbackDecisionV1(NamedTuple):
    """Deterministic routing decision from direct and probe certificates."""

    route: str
    direct_action_index: int | None
    probe_index: int | None
    contingent_action_indices: IntArray | None
    fallback_action_index: int
    direct_minimax_worst_case_regret: float
    selected_value: float
    maximum_probe_total_value: float

    def summary(self) -> dict[str, object]:
        return {
            "version": QUERY_PROBE_CERTIFICATE_VERSION,
            "route": self.route,
            "direct_action_index": self.direct_action_index,
            "probe_index": self.probe_index,
            "contingent_action_indices": (
                None
                if self.contingent_action_indices is None
                else self.contingent_action_indices.tolist()
            ),
            "fallback_action_index": self.fallback_action_index,
            "direct_minimax_worst_case_regret": (self.direct_minimax_worst_case_regret),
            "selected_value": self.selected_value,
            "maximum_probe_total_value": self.maximum_probe_total_value,
            "claim_boundary": QUERY_PROBE_CERTIFICATE_CLAIM_BOUNDARY,
        }


def query_probe_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    outcome_likelihood_by_hypothesis: object,
    *,
    probe_cost: float = 0.0,
    regret_tolerance: float = 0.0,
    max_policy_count: int = _DEFAULT_MAX_POLICY_COUNT,
) -> QueryProbeCertificateV1:
    """Certify every contingent finite-action policy for one finite probe.

    ``outcome_likelihood_by_hypothesis[i, z]`` is the registered probability of
    outcome ``z`` under hypothesis ``i``.  A policy commits before probing to one
    action for each possible outcome.  The returned minimax policy minimizes
    exact ex-ante worst-case regret over every prior-supported complete belief
    compatible with the supplied quotient masses.
    """

    losses = _loss_matrix(loss_by_hypothesis_action)
    likelihood = _likelihood_matrix(
        outcome_likelihood_by_hypothesis,
        hypothesis_count=losses.shape[0],
    )
    cost = _finite_nonnegative(probe_cost, name="probe_cost")
    cap = _positive_integer(max_policy_count, name="max_policy_count")
    policies = _contingent_policies(
        losses.shape[1],
        likelihood.shape[1],
        max_policy_count=cap,
    )

    selected_loss = np.take(losses, policies, axis=1)
    policy_loss = np.sum(
        selected_loss * likelihood[:, None, :],
        axis=2,
        dtype=np.float64,
    )
    policy_loss = _immutable_float64(policy_loss)
    decision = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        policy_loss,
        regret_tolerance=regret_tolerance,
    )
    minimax_policy = policies[decision.minimax_action_index]
    regret = float(decision.minimax_worst_case_regret)
    return QueryProbeCertificateV1(
        outcome_likelihood_by_hypothesis=likelihood,
        contingent_action_indices=policies,
        policy_loss_by_hypothesis=policy_loss,
        policy_decision_certificate=decision,
        probe_cost=cost,
        minimax_contingent_action_indices=_immutable_int64(minimax_policy),
        minimax_worst_case_regret=regret,
        total_regret_plus_cost=cost + regret,
    )


def act_probe_fallback_decision(
    direct_certificate: QueryDecisionCertificateV1,
    probe_certificates: object,
    *,
    fallback_action_index: int,
    maximum_probe_total_value: float,
) -> ActProbeFallbackDecisionV1:
    """Route to a certified direct action, best affordable probe, or fallback.

    Direct action has priority when the direct certificate contains an action at
    its registered regret tolerance.  Otherwise the lowest-index probe among
    ties is selected when ``probe_cost + exact minimax regret`` does not exceed
    ``maximum_probe_total_value``.  If neither condition holds, the caller-owned
    fallback action is returned.
    """

    if (
        isinstance(fallback_action_index, (bool, np.bool_))
        or not isinstance(fallback_action_index, Integral)
        or not 0 <= int(fallback_action_index) < direct_certificate.action_count
    ):
        raise ValueError("fallback_action_index is outside the direct action set")
    threshold = _finite_nonnegative(
        maximum_probe_total_value,
        name="maximum_probe_total_value",
    )
    probes = tuple(probe_certificates)
    for probe in probes:
        if not isinstance(probe, QueryProbeCertificateV1):
            raise ValueError("probe_certificates must contain QueryProbeCertificateV1")
        if probe.hypothesis_count != direct_certificate.hypothesis_count:
            raise ValueError("direct and probe hypothesis counts differ")
        if probe.action_count != direct_certificate.action_count:
            raise ValueError("direct and probe action counts differ")
        nested = probe.policy_decision_certificate
        if (
            not np.array_equal(
                nested.prior_support_mask, direct_certificate.prior_support_mask
            )
            or not np.array_equal(nested.class_index, direct_certificate.class_index)
            or not np.allclose(
                nested.quotient_weights,
                direct_certificate.quotient_weights,
                rtol=0.0,
                atol=_ARRAY_ATOL,
            )
        ):
            raise ValueError("direct and probe ambiguity sets differ")

    if direct_certificate.has_tolerance_admissible_action:
        action = int(direct_certificate.minimax_action_index)
        return ActProbeFallbackDecisionV1(
            route="act",
            direct_action_index=action,
            probe_index=None,
            contingent_action_indices=None,
            fallback_action_index=int(fallback_action_index),
            direct_minimax_worst_case_regret=float(
                direct_certificate.minimax_worst_case_regret
            ),
            selected_value=float(direct_certificate.minimax_worst_case_regret),
            maximum_probe_total_value=threshold,
        )

    if probes:
        values = np.asarray(
            [probe.total_regret_plus_cost for probe in probes], dtype=np.float64
        )
        probe_index = int(np.argmin(values))
        selected = probes[probe_index]
        if selected.total_regret_plus_cost <= threshold + _ARRAY_ATOL:
            return ActProbeFallbackDecisionV1(
                route="probe",
                direct_action_index=None,
                probe_index=probe_index,
                contingent_action_indices=selected.minimax_contingent_action_indices,
                fallback_action_index=int(fallback_action_index),
                direct_minimax_worst_case_regret=float(
                    direct_certificate.minimax_worst_case_regret
                ),
                selected_value=float(selected.total_regret_plus_cost),
                maximum_probe_total_value=threshold,
            )

    return ActProbeFallbackDecisionV1(
        route="fallback",
        direct_action_index=None,
        probe_index=None,
        contingent_action_indices=None,
        fallback_action_index=int(fallback_action_index),
        direct_minimax_worst_case_regret=float(
            direct_certificate.minimax_worst_case_regret
        ),
        selected_value=float(
            direct_certificate.worst_case_regret[fallback_action_index]
        ),
        maximum_probe_total_value=threshold,
    )


__all__ = [
    "QUERY_PROBE_CERTIFICATE_CLAIM_BOUNDARY",
    "QUERY_PROBE_CERTIFICATE_SEMANTICS",
    "QUERY_PROBE_CERTIFICATE_VERSION",
    "ActProbeFallbackDecisionV1",
    "QueryProbeCertificateV1",
    "act_probe_fallback_decision",
    "query_probe_certificate",
]
