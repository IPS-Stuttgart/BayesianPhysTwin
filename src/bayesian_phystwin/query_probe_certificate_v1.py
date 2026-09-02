"""Exact finite certificates for acting, probing, or returning a fallback.

A registered query quotient may determine posterior mass only at the class level.
The existing :mod:`query_decision_certificate_v1` computes exact worst-case
regret for direct finite actions over every prior-supported complete lift. This
module extends the same support-function construction to finite probes.

For probe outcome likelihood ``O_e[i, z]``, probe cost ``c_e``, and contingent
policy ``delta(z)``, define one hypothesis-wise meta-action loss

    G[i, (e, delta)] = c_e + sum_z O_e[i, z] L[i, delta(z)].

Direct actions are meta-actions with ``G[i, a] = L[i, a]``. Applying the direct
certificate to the *union* of direct and probe-contingent meta-actions gives the
exact common-comparator regret

    Reg_bar(m)
      = max_m' sum_c lambda[c]
          max_{i in c: p[i] > 0} (G[i, m] - G[i, m']).

Using one union is essential: direct-action regret and within-probe regret use
different comparator classes and cannot in general be ranked against each
other. The resulting router acts or probes only when the selected union
meta-action satisfies the registered regret tolerance; otherwise it returns the
caller-registered fallback action.

The result does not validate the physical hypotheses, quotient, probe
likelihood, action loss, probe cost, population, transport, or deployment
context.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Final, NamedTuple, TypeAlias, cast

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
    "exact-common-comparator-regret-for-direct-and-finite-probe-meta-actions-v1"
)
QUERY_PROBE_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied finite hypotheses, positive "
    "prior support, quotient masses, direct-action losses, probe costs, and "
    "probe-outcome likelihoods. It is an ex-ante expected-loss statement. It "
    "does not validate the physical hypotheses, quotient, likelihoods, costs, "
    "outcome model, transport, calibration, deployment context, or safety."
)

_PROBABILITY_ATOL: Final = 1e-12
_DEFAULT_MAX_POLICY_COUNT: Final = 100_000
_DEFAULT_MAX_META_ACTION_COUNT: Final = 250_000


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
    scalar: float = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return scalar


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    scalar: int = int(value)
    if scalar <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return scalar


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
    normalized = likelihood / row_sum[:, None]
    return _immutable_float64(normalized)


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


def _probe_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("probe_likelihoods must be an iterable of matrices")
    return tuple(cast(Iterable[object], value))


def _probe_costs(value: object | None, *, probe_count: int) -> FloatArray:
    if value is None:
        return _immutable_float64(np.zeros(probe_count, dtype=np.float64))
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("probe_costs must contain real numeric values")
    costs = np.ascontiguousarray(raw, dtype=np.float64)
    if costs.ndim != 1 or costs.size != probe_count:
        raise ValueError("probe_costs must contain one entry per probe")
    if not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("probe_costs must be finite and nonnegative")
    return _immutable_float64(costs)


class QueryProbeCertificateV1(NamedTuple):
    """Exact within-probe certificate and hypothesis-wise meta-action losses."""

    outcome_likelihood_by_hypothesis: FloatArray
    contingent_action_indices: IntArray
    expected_action_loss_by_hypothesis_policy: FloatArray
    meta_loss_by_hypothesis_policy: FloatArray
    policy_decision_certificate: QueryDecisionCertificateV1
    probe_cost: float
    minimax_contingent_action_indices: IntArray
    minimax_within_probe_worst_case_regret: float

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

    def summary(self) -> dict[str, object]:
        return {
            "version": QUERY_PROBE_CERTIFICATE_VERSION,
            "semantics": QUERY_PROBE_CERTIFICATE_SEMANTICS,
            "hypothesis_count": self.hypothesis_count,
            "outcome_count": self.outcome_count,
            "action_count": self.action_count,
            "policy_count": self.policy_count,
            "probe_cost": self.probe_cost,
            "within_probe_minimax_policy_index": (
                self.policy_decision_certificate.minimax_action_index
            ),
            "within_probe_minimax_contingent_action_indices": (
                self.minimax_contingent_action_indices.tolist()
            ),
            "within_probe_minimax_worst_case_regret": (
                self.minimax_within_probe_worst_case_regret
            ),
            "claim_boundary": QUERY_PROBE_CERTIFICATE_CLAIM_BOUNDARY,
        }


class ActProbeFallbackCertificateV1(NamedTuple):
    """Exact certificate over one common direct/probe meta-action class."""

    direct_decision_certificate: QueryDecisionCertificateV1
    probe_certificates: tuple[QueryProbeCertificateV1, ...]
    meta_loss_by_hypothesis: FloatArray
    meta_action_kind: tuple[str, ...]
    meta_direct_action_index: IntArray
    meta_probe_index: IntArray
    meta_probe_policy_index: IntArray
    meta_decision_certificate: QueryDecisionCertificateV1
    fallback_action_index: int
    route: str
    selected_meta_action_index: int | None
    selected_direct_action_index: int | None
    selected_probe_index: int | None
    selected_contingent_action_indices: IntArray | None
    selected_worst_case_regret: float

    @property
    def certified(self) -> bool:
        return self.route != "fallback"

    @property
    def meta_action_count(self) -> int:
        return int(self.meta_loss_by_hypothesis.shape[1])

    def summary(self) -> dict[str, object]:
        return {
            "version": QUERY_PROBE_CERTIFICATE_VERSION,
            "semantics": QUERY_PROBE_CERTIFICATE_SEMANTICS,
            "route": self.route,
            "certified": self.certified,
            "fallback_action_index": self.fallback_action_index,
            "direct_action_count": self.direct_decision_certificate.action_count,
            "probe_count": len(self.probe_certificates),
            "meta_action_count": self.meta_action_count,
            "selected_meta_action_index": self.selected_meta_action_index,
            "selected_direct_action_index": self.selected_direct_action_index,
            "selected_probe_index": self.selected_probe_index,
            "selected_contingent_action_indices": (
                None
                if self.selected_contingent_action_indices is None
                else self.selected_contingent_action_indices.tolist()
            ),
            "selected_worst_case_regret": self.selected_worst_case_regret,
            "regret_tolerance": self.meta_decision_certificate.regret_tolerance,
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
    """Build every contingent policy and its exact within-probe certificate.

    The returned meta-action loss includes the probe cost and can be concatenated
    with direct actions and other probes. The within-probe regret is descriptive
    only; direct-versus-probe selection must use
    :func:`act_probe_fallback_certificate`, whose comparator class is common.
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
    expected_loss = np.sum(
        selected_loss * likelihood[:, None, :],
        axis=2,
        dtype=np.float64,
    )
    meta_loss = expected_loss + cost
    expected_loss_immutable = _immutable_float64(expected_loss)
    meta_loss_immutable = _immutable_float64(meta_loss)
    decision = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        meta_loss_immutable,
        regret_tolerance=regret_tolerance,
    )
    minimax_policy = policies[decision.minimax_action_index]
    return QueryProbeCertificateV1(
        outcome_likelihood_by_hypothesis=likelihood,
        contingent_action_indices=policies,
        expected_action_loss_by_hypothesis_policy=expected_loss_immutable,
        meta_loss_by_hypothesis_policy=meta_loss_immutable,
        policy_decision_certificate=decision,
        probe_cost=cost,
        minimax_contingent_action_indices=_immutable_int64(minimax_policy),
        minimax_within_probe_worst_case_regret=float(
            decision.minimax_worst_case_regret
        ),
    )


def act_probe_fallback_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    probe_likelihoods: object,
    *,
    probe_costs: object | None = None,
    fallback_action_index: int,
    regret_tolerance: float = 0.0,
    max_policy_count_per_probe: int = _DEFAULT_MAX_POLICY_COUNT,
    max_meta_action_count: int = _DEFAULT_MAX_META_ACTION_COUNT,
) -> ActProbeFallbackCertificateV1:
    """Certify the union of direct actions and finite contingent probe policies.

    Every direct action and every probe-contingent policy is represented by one
    hypothesis-wise expected-loss column. Probe costs therefore enter the same
    pairwise comparisons as direct actions and other probes. When no union
    meta-action satisfies ``regret_tolerance``, the result returns the supplied
    fallback action without representing it as certified.
    """

    losses = _loss_matrix(loss_by_hypothesis_action)
    if (
        isinstance(fallback_action_index, (bool, np.bool_))
        or not isinstance(fallback_action_index, Integral)
        or not 0 <= int(fallback_action_index) < losses.shape[1]
    ):
        raise ValueError("fallback_action_index is outside the direct action set")
    policy_cap = _positive_integer(
        max_policy_count_per_probe,
        name="max_policy_count_per_probe",
    )
    meta_cap = _positive_integer(
        max_meta_action_count,
        name="max_meta_action_count",
    )
    probes = _probe_sequence(probe_likelihoods)
    costs = _probe_costs(probe_costs, probe_count=len(probes))

    direct = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        losses,
        regret_tolerance=regret_tolerance,
    )
    probe_certificate_list: list[QueryProbeCertificateV1] = []
    for probe_index, likelihood in enumerate(probes):
        probe_certificate_list.append(
            query_probe_certificate(
                prior_weights,
                quotient_weights,
                class_index,
                losses,
                likelihood,
                probe_cost=float(costs[probe_index]),
                regret_tolerance=regret_tolerance,
                max_policy_count=policy_cap,
            )
        )
    probe_certificates = tuple(probe_certificate_list)

    meta_action_count = losses.shape[1] + sum(
        certificate.policy_count for certificate in probe_certificates
    )
    if meta_action_count > meta_cap:
        raise ValueError(
            f"meta-action count {meta_action_count} exceeds cap {meta_cap}"
        )

    columns: list[FloatArray] = [losses]
    kinds = ["act"] * losses.shape[1]
    direct_indices = list(range(losses.shape[1]))
    probe_indices = [-1] * losses.shape[1]
    policy_indices = [-1] * losses.shape[1]
    for probe_index, certificate in enumerate(probe_certificates):
        columns.append(certificate.meta_loss_by_hypothesis_policy)
        kinds.extend(["probe"] * certificate.policy_count)
        direct_indices.extend([-1] * certificate.policy_count)
        probe_indices.extend([probe_index] * certificate.policy_count)
        policy_indices.extend(range(certificate.policy_count))

    meta_loss = _immutable_float64(np.concatenate(columns, axis=1))
    meta_decision = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        meta_loss,
        regret_tolerance=regret_tolerance,
    )

    selected_meta: int | None
    selected_direct: int | None
    selected_probe: int | None
    selected_policy: IntArray | None
    if meta_decision.has_tolerance_admissible_action:
        selected_meta = int(meta_decision.minimax_action_index)
        selected_worst_case_regret = float(
            meta_decision.worst_case_regret[selected_meta]
        )
        if kinds[selected_meta] == "act":
            route = "act"
            selected_direct = direct_indices[selected_meta]
            selected_probe = None
            selected_policy = None
        else:
            route = "probe"
            selected_direct = None
            selected_probe_value = int(probe_indices[selected_meta])
            probe_policy_index = int(policy_indices[selected_meta])
            selected_probe = selected_probe_value
            selected_policy = _immutable_int64(
                probe_certificates[selected_probe_value].contingent_action_indices[
                    probe_policy_index
                ]
            )
    else:
        route = "fallback"
        selected_meta = None
        selected_direct = None
        selected_probe = None
        selected_policy = None
        selected_worst_case_regret = float(
            meta_decision.worst_case_regret[int(fallback_action_index)]
        )

    return ActProbeFallbackCertificateV1(
        direct_decision_certificate=direct,
        probe_certificates=probe_certificates,
        meta_loss_by_hypothesis=meta_loss,
        meta_action_kind=tuple(kinds),
        meta_direct_action_index=_immutable_int64(direct_indices),
        meta_probe_index=_immutable_int64(probe_indices),
        meta_probe_policy_index=_immutable_int64(policy_indices),
        meta_decision_certificate=meta_decision,
        fallback_action_index=int(fallback_action_index),
        route=route,
        selected_meta_action_index=selected_meta,
        selected_direct_action_index=selected_direct,
        selected_probe_index=selected_probe,
        selected_contingent_action_indices=selected_policy,
        selected_worst_case_regret=selected_worst_case_regret,
    )


__all__ = [
    "QUERY_PROBE_CERTIFICATE_CLAIM_BOUNDARY",
    "QUERY_PROBE_CERTIFICATE_SEMANTICS",
    "QUERY_PROBE_CERTIFICATE_VERSION",
    "ActProbeFallbackCertificateV1",
    "QueryProbeCertificateV1",
    "act_probe_fallback_certificate",
    "query_probe_certificate",
]
