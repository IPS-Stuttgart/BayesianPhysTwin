"""Exact certificates for finite-horizon stochastic act--sense policy trees.

A policy is fixed before sensing.  Each internal node selects one registered
probe and each outgoing edge is one registered probe outcome.  Leaves execute
terminal physical actions.  The complete tree is evaluated as one ex-ante
finite decision over every prior-supported physical hypothesis compatible with
registered quotient masses.

For a policy tree ``pi`` and hypothesis ``h``, let ``V[h, pi]`` contain the
expected terminal loss plus sensing costs.  The exact pairwise support function
is

    Delta(pi, rho) = sum_c lambda[c]
        max_{h in c, prior[h] > 0} (V[h, pi] - V[h, rho]).

Its maximum over comparator trees is the exact worst-case regret over every
complete belief with the supplied quotient masses.  The implementation selects
one unique minimax tree only when its regret is within a declared tolerance;
otherwise it reproduces the caller-owned fallback action without taking a
probe.

The guarantee is conditional on the finite hypothesis support, quotient masses,
probe likelihoods, conditional-independence semantics encoded by the tree,
probe costs, terminal losses, policy depth, candidate-tree compression, and
regret tolerance.  It does not validate physical hypotheses, likelihoods,
probe safety, reset semantics, target transport, exchangeability, deployment,
or safety.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final, NamedTuple, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.conformal_complete_plan_certificate_v1 import (
    ScaledTrajectoryConformalPlanEnvelopeV1,
    scaled_trajectory_conformal_plan_envelope,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

ADAPTIVE_STOCHASTIC_POLICY_TREE_VERSION: Final = 1
ADAPTIVE_STOCHASTIC_POLICY_TREE_SEMANTICS: Final = (
    "pre-probe-finite-horizon-stochastic-policy-tree-regret-v1"
)
ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY: Final = (
    "Exactness is conditional on the supplied finite physical hypotheses, "
    "positive prior support, quotient masses, stochastic probe likelihoods, "
    "terminal losses, sensing costs, finite policy depth, candidate-tree "
    "compression, and regret tolerance. The conformal wrapper additionally "
    "requires exchangeable complete calibration trajectories and a policy "
    "roster fixed before calibration outcomes. Neither layer validates the "
    "physical support, likelihoods, conditional independence, probe physics, "
    "reset semantics, target transport, deployment, or safety."
)

_NUMERICAL_ATOL: Final = 1e-12
_MAX_DEPTH_HARD_CAP: Final = 4
_MAX_POLICY_HARD_CAP: Final = 10_000
_MAX_RAW_TREE_HARD_CAP: Final = 2_000_000


def _bytes_backed_array(value: object, dtype: npt.DTypeLike) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    raw = contiguous.tobytes(order="C")
    result = np.frombuffer(raw, dtype=contiguous.dtype).reshape(contiguous.shape)
    if result.flags.writeable:
        raise RuntimeError("bytes-backed array unexpectedly remained writeable")
    return result


def _immutable_float64(value: object) -> FloatArray:
    return _bytes_backed_array(value, np.float64)


def _immutable_int64(value: object) -> IntArray:
    return _bytes_backed_array(value, np.int64)


def _immutable_bool(value: object) -> BoolArray:
    return _bytes_backed_array(value, np.bool_)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive_integer(value: object, *, name: str, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [1, {upper}]")
    result = int(value)
    if result < 1 or result > upper:
        raise ValueError(f"{name} must be an integer in [1, {upper}]")
    return result


def _depth(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("maximum_depth must be an integer")
    result = int(value)
    if result < 0 or result > _MAX_DEPTH_HARD_CAP:
        raise ValueError(
            f"maximum_depth must lie in [0, {_MAX_DEPTH_HARD_CAP}]"
        )
    return result


def _probability_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    total = float(np.sum(result))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    result /= total
    return _immutable_float64(result)


def _class_indices(value: object, *, hypothesis_count: int) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError("class_index_by_hypothesis must contain integers")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    if result.shape != (hypothesis_count,):
        raise ValueError(
            "class_index_by_hypothesis must contain one entry per hypothesis"
        )
    if np.any(result < 0):
        raise ValueError("class indices must be nonnegative")
    unique = np.unique(result)
    if not np.array_equal(unique, np.arange(unique.size, dtype=np.int64)):
        raise ValueError("class indices must be contiguous and start at zero")
    return _immutable_int64(result)


def _loss_matrix(value: object, *, hypothesis_count: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("terminal_loss_by_hypothesis_action must be numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] != hypothesis_count
        or result.shape[1] == 0
    ):
        raise ValueError(
            "terminal_loss_by_hypothesis_action must have shape "
            "(hypothesis_count, positive_action_count)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("terminal losses must be finite")
    return _immutable_float64(result)


def _probe_models(
    value: Sequence[object],
    *,
    hypothesis_count: int,
) -> tuple[FloatArray, ...]:
    result: list[FloatArray] = []
    for probe_index, item in enumerate(value):
        raw = np.asarray(item)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"probe {probe_index} likelihoods must be numeric")
        likelihood = np.ascontiguousarray(raw, dtype=np.float64)
        if (
            likelihood.ndim != 2
            or likelihood.shape[0] != hypothesis_count
            or likelihood.shape[1] < 2
        ):
            raise ValueError(
                f"probe {probe_index} must have shape "
                "(hypothesis_count, at_least_two_outcomes)"
            )
        if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0.0):
            raise ValueError(
                f"probe {probe_index} likelihoods must be finite and nonnegative"
            )
        row_sums = np.sum(likelihood, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"probe {probe_index} rows must sum to one")
        result.append(_immutable_float64(likelihood))
    return tuple(result)


def _probe_costs(value: object, *, probe_count: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("probe_costs must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.shape != (probe_count,):
        raise ValueError("probe_costs must contain one entry per probe")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("probe_costs must be finite and nonnegative")
    return _immutable_float64(result)


@dataclass(frozen=True, slots=True)
class AdaptivePolicyTreeV1:
    """One complete policy fixed before any registered probe is observed."""

    mode: str
    action_index: int | None
    probe_index: int | None
    children: tuple[AdaptivePolicyTreeV1, ...]
    expected_loss_by_hypothesis: FloatArray
    expected_probe_cost_by_hypothesis: FloatArray
    depth: int
    canonical_key: str

    def structure(self) -> dict[str, object]:
        if self.mode == "act":
            return {"mode": "act", "action_index": self.action_index}
        return {
            "mode": "sense",
            "probe_index": self.probe_index,
            "children": [child.structure() for child in self.children],
        }


class AdaptivePolicyTreeCertificateV1(NamedTuple):
    """Exact common-comparator certificate over a finite policy-tree roster."""

    prior_probabilities: FloatArray
    class_index_by_hypothesis: IntArray
    class_masses: FloatArray
    terminal_loss_by_hypothesis_action: FloatArray
    probe_likelihoods: tuple[FloatArray, ...]
    probe_costs: FloatArray
    probe_names: tuple[str, ...]
    policies: tuple[AdaptivePolicyTreeV1, ...]
    expected_loss_by_hypothesis_policy: FloatArray
    pairwise_worst_case_gap: FloatArray
    worst_case_regret: FloatArray
    regret_tolerance: float
    fallback_action_index: int
    fallback_policy_index: int
    minimizer_policy_mask: BoolArray
    minimizer_count: int
    candidate_policy_index: int | None
    output_policy_index: int
    output_mode: str
    used_fallback: bool
    fallback_reason: str | None
    raw_tree_count: int
    loss_equivalent_tree_count: int
    dominance_pruned_tree_count: int

    @property
    def policy_count(self) -> int:
        return len(self.policies)

    @property
    def output_policy(self) -> AdaptivePolicyTreeV1:
        return self.policies[self.output_policy_index]

    @property
    def selected_first_probe_index(self) -> int | None:
        if self.output_mode != "sense":
            return None
        return self.output_policy.probe_index

    def summary(self) -> dict[str, object]:
        candidate_regret = (
            None
            if self.candidate_policy_index is None
            else float(self.worst_case_regret[self.candidate_policy_index])
        )
        return {
            "version": ADAPTIVE_STOCHASTIC_POLICY_TREE_VERSION,
            "semantics": ADAPTIVE_STOCHASTIC_POLICY_TREE_SEMANTICS,
            "hypothesis_count": int(self.prior_probabilities.size),
            "class_count": int(self.class_masses.size),
            "action_count": int(self.terminal_loss_by_hypothesis_action.shape[1]),
            "probe_count": len(self.probe_likelihoods),
            "policy_count": self.policy_count,
            "raw_tree_count": self.raw_tree_count,
            "loss_equivalent_tree_count": self.loss_equivalent_tree_count,
            "dominance_pruned_tree_count": self.dominance_pruned_tree_count,
            "regret_tolerance": self.regret_tolerance,
            "candidate_policy_index": self.candidate_policy_index,
            "candidate_worst_case_regret": candidate_regret,
            "minimizer_count": self.minimizer_count,
            "output_policy_index": self.output_policy_index,
            "output_mode": self.output_mode,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "selected_first_probe_index": self.selected_first_probe_index,
            "output_policy": self.output_policy.structure(),
            "claim_boundary": ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY,
        }


class LoggedPolicyTreeRegretTensorV1(NamedTuple):
    """Logged realized losses for every precommitted adaptive policy tree."""

    policy_loss_by_trajectory_decision_policy: FloatArray
    best_policy_loss_by_trajectory_decision: FloatArray
    realized_regret_by_trajectory_decision_policy: FloatArray


class ConformalAdaptivePolicyTreeDecisionV1(NamedTuple):
    """Trajectory-calibrated pre-probe policy-tree decision."""

    certificate: AdaptivePolicyTreeCertificateV1
    envelope: ScaledTrajectoryConformalPlanEnvelopeV1
    calibrated_regret_upper_by_policy: FloatArray
    regret_tolerance: float
    minimizer_policy_mask: BoolArray
    minimizer_count: int
    candidate_policy_index: int | None
    output_policy_index: int
    output_mode: str
    used_fallback: bool
    fallback_reason: str | None

    @property
    def output_policy(self) -> AdaptivePolicyTreeV1:
        return self.certificate.policies[self.output_policy_index]

    def summary(self) -> dict[str, object]:
        return {
            "version": ADAPTIVE_STOCHASTIC_POLICY_TREE_VERSION,
            "semantics": (
                "trajectory-conformal-pre-probe-adaptive-policy-tree-regret-v1"
            ),
            "policy_count": self.certificate.policy_count,
            "regret_tolerance": self.regret_tolerance,
            "conformal_score_quantile": self.envelope.score_quantile,
            "finite_sample_coverage_lower_bound": (
                self.envelope.finite_sample_coverage_lower_bound
            ),
            "candidate_policy_index": self.candidate_policy_index,
            "minimizer_count": self.minimizer_count,
            "output_policy_index": self.output_policy_index,
            "output_mode": self.output_mode,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "output_policy": self.output_policy.structure(),
            "claim_boundary": ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY,
        }


def _action_tree(
    action_index: int,
    terminal_losses: FloatArray,
) -> AdaptivePolicyTreeV1:
    hypothesis_count = terminal_losses.shape[0]
    return AdaptivePolicyTreeV1(
        mode="act",
        action_index=action_index,
        probe_index=None,
        children=(),
        expected_loss_by_hypothesis=_immutable_float64(
            terminal_losses[:, action_index]
        ),
        expected_probe_cost_by_hypothesis=_immutable_float64(
            np.zeros(hypothesis_count, dtype=np.float64)
        ),
        depth=0,
        canonical_key=f"act:{action_index}",
    )


def _sense_tree(
    probe_index: int,
    children: tuple[AdaptivePolicyTreeV1, ...],
    likelihood: FloatArray,
    cost: float,
) -> AdaptivePolicyTreeV1:
    expected_loss = np.full(likelihood.shape[0], cost, dtype=np.float64)
    expected_cost = np.full(likelihood.shape[0], cost, dtype=np.float64)
    for outcome_index, child in enumerate(children):
        probability = likelihood[:, outcome_index]
        expected_loss += probability * child.expected_loss_by_hypothesis
        expected_cost += probability * child.expected_probe_cost_by_hypothesis
    return AdaptivePolicyTreeV1(
        mode="sense",
        action_index=None,
        probe_index=probe_index,
        children=children,
        expected_loss_by_hypothesis=_immutable_float64(expected_loss),
        expected_probe_cost_by_hypothesis=_immutable_float64(expected_cost),
        depth=1 + max(child.depth for child in children),
        canonical_key=(
            f"sense:{probe_index}["
            + ",".join(child.canonical_key for child in children)
            + "]"
        ),
    )


def _equivalent_loss(
    left: AdaptivePolicyTreeV1,
    right: AdaptivePolicyTreeV1,
    tolerance: float,
) -> bool:
    return bool(
        np.allclose(
            left.expected_loss_by_hypothesis,
            right.expected_loss_by_hypothesis,
            rtol=0.0,
            atol=tolerance,
        )
    )


def _dominates(
    left: AdaptivePolicyTreeV1,
    right: AdaptivePolicyTreeV1,
    tolerance: float,
) -> bool:
    left_loss = left.expected_loss_by_hypothesis
    right_loss = right.expected_loss_by_hypothesis
    return bool(
        np.all(left_loss <= right_loss + tolerance)
        and np.any(left_loss < right_loss - tolerance)
    )


class _PolicyEnumerator:
    def __init__(
        self,
        terminal_losses: FloatArray,
        likelihoods: tuple[FloatArray, ...],
        costs: FloatArray,
        *,
        loss_tolerance: float,
        max_raw_trees: int,
        max_policies: int,
    ) -> None:
        self._terminal_losses = terminal_losses
        self._likelihoods = likelihoods
        self._costs = costs
        self._loss_tolerance = loss_tolerance
        self._max_raw_trees = max_raw_trees
        self._max_policies = max_policies
        self._memo: dict[
            tuple[tuple[int, ...], int], tuple[AdaptivePolicyTreeV1, ...]
        ] = {}
        self.raw_tree_count = 0
        self.loss_equivalent_tree_count = 0
        self.dominance_pruned_tree_count = 0

    def _insert(
        self,
        retained: list[AdaptivePolicyTreeV1],
        candidate: AdaptivePolicyTreeV1,
    ) -> None:
        for existing in retained:
            if _equivalent_loss(existing, candidate, self._loss_tolerance):
                self.loss_equivalent_tree_count += 1
                return
            if _dominates(existing, candidate, self._loss_tolerance):
                self.dominance_pruned_tree_count += 1
                return

        survivors: list[AdaptivePolicyTreeV1] = []
        for existing in retained:
            if (
                existing.mode == "sense"
                and _dominates(candidate, existing, self._loss_tolerance)
            ):
                self.dominance_pruned_tree_count += 1
                continue
            survivors.append(existing)
        survivors.append(candidate)
        survivors.sort(
            key=lambda tree: (
                tree.mode != "act",
                tree.depth,
                tree.canonical_key,
            )
        )
        if len(survivors) > self._max_policies:
            raise ValueError(
                "retained policy count exceeds max_policy_count; reduce the "
                "probe roster/depth or raise the reviewed cap"
            )
        retained[:] = survivors

    def enumerate(
        self,
        available_probes: tuple[int, ...],
        remaining_depth: int,
    ) -> tuple[AdaptivePolicyTreeV1, ...]:
        key = (available_probes, remaining_depth)
        if key in self._memo:
            return self._memo[key]

        retained = [
            _action_tree(action_index, self._terminal_losses)
            for action_index in range(self._terminal_losses.shape[1])
        ]
        self.raw_tree_count += len(retained)
        if self.raw_tree_count > self._max_raw_trees:
            raise ValueError("raw policy-tree count exceeds max_raw_tree_count")

        if remaining_depth > 0:
            for probe_index in available_probes:
                remaining = tuple(
                    value for value in available_probes if value != probe_index
                )
                child_roster = self.enumerate(remaining, remaining_depth - 1)
                outcome_count = self._likelihoods[probe_index].shape[1]
                combination_count = len(child_roster) ** outcome_count
                if self.raw_tree_count + combination_count > self._max_raw_trees:
                    raise ValueError(
                        "raw policy-tree count exceeds max_raw_tree_count; "
                        "reduce the probe roster/depth or raise the reviewed cap"
                    )
                for children_raw in itertools.product(
                    child_roster,
                    repeat=outcome_count,
                ):
                    self.raw_tree_count += 1
                    children = tuple(children_raw)
                    candidate = _sense_tree(
                        probe_index,
                        children,
                        self._likelihoods[probe_index],
                        float(self._costs[probe_index]),
                    )
                    self._insert(retained, candidate)
        result = tuple(retained)
        self._memo[key] = result
        return result


def _pairwise_support_gaps(
    expected_loss: FloatArray,
    prior: FloatArray,
    classes: IntArray,
    class_masses: FloatArray,
) -> FloatArray:
    policy_count = expected_loss.shape[1]
    pairwise = np.zeros((policy_count, policy_count), dtype=np.float64)
    support = prior > 0.0
    for class_index, mass in enumerate(class_masses):
        if mass <= 0.0:
            continue
        indices = np.flatnonzero(support & (classes == class_index))
        if indices.size == 0:
            raise ValueError(
                "every positive-mass quotient class requires prior-supported "
                "physical hypotheses"
            )
        class_gap = np.full(
            (policy_count, policy_count),
            -math.inf,
            dtype=np.float64,
        )
        for hypothesis_index in indices:
            losses = expected_loss[hypothesis_index]
            class_gap = np.maximum(
                class_gap,
                losses[:, None] - losses[None, :],
            )
        pairwise += float(mass) * class_gap
    np.fill_diagonal(pairwise, 0.0)
    return _immutable_float64(pairwise)


def adaptive_stochastic_policy_tree_certificate(
    prior_probabilities: object,
    class_masses: object,
    class_index_by_hypothesis: object,
    terminal_loss_by_hypothesis_action: object,
    probe_likelihoods: Sequence[object],
    probe_costs: object,
    *,
    fallback_action_index: int,
    maximum_depth: int,
    regret_tolerance: float,
    probe_names: Sequence[str] | None = None,
    loss_compression_tolerance: float = 1e-12,
    max_policy_count: int = 5000,
    max_raw_tree_count: int = 500000,
) -> AdaptivePolicyTreeCertificateV1:
    """Enumerate, compress, and certify stochastic adaptive policy trees.

    A registered probe is used at most once along any root-to-leaf path.  Trees
    with equal represented-support expected-loss vectors are reduced to the
    first canonical representative.  A sensing tree that is componentwise
    dominated on every represented hypothesis is removed.  Direct action trees
    are always retained, including the caller-owned fallback.
    """

    prior = _probability_vector(prior_probabilities, name="prior_probabilities")
    classes = _class_indices(
        class_index_by_hypothesis,
        hypothesis_count=prior.size,
    )
    masses = _probability_vector(class_masses, name="class_masses")
    if masses.size != int(np.max(classes)) + 1:
        raise ValueError("class_masses does not match the registered classes")
    terminal_losses = _loss_matrix(
        terminal_loss_by_hypothesis_action,
        hypothesis_count=prior.size,
    )
    action_count = terminal_losses.shape[1]
    if (
        isinstance(fallback_action_index, (bool, np.bool_))
        or not isinstance(fallback_action_index, Integral)
        or int(fallback_action_index) < 0
        or int(fallback_action_index) >= action_count
    ):
        raise ValueError("fallback_action_index is outside the action roster")
    fallback_action = int(fallback_action_index)
    likelihoods = _probe_models(
        probe_likelihoods,
        hypothesis_count=prior.size,
    )
    costs = _probe_costs(probe_costs, probe_count=len(likelihoods))
    depth = _depth(maximum_depth)
    tolerance = _finite_nonnegative(
        regret_tolerance,
        name="regret_tolerance",
    )
    compression_tolerance = _finite_nonnegative(
        loss_compression_tolerance,
        name="loss_compression_tolerance",
    )
    policy_cap = _positive_integer(
        max_policy_count,
        name="max_policy_count",
        upper=_MAX_POLICY_HARD_CAP,
    )
    raw_cap = _positive_integer(
        max_raw_tree_count,
        name="max_raw_tree_count",
        upper=_MAX_RAW_TREE_HARD_CAP,
    )
    if probe_names is None:
        names = tuple(f"probe_{index}" for index in range(len(likelihoods)))
    else:
        names = tuple(str(item) for item in probe_names)
        if len(names) != len(likelihoods) or any(not item for item in names):
            raise ValueError("probe_names must be nonempty and match the probe roster")
        if len(set(names)) != len(names):
            raise ValueError("probe_names must be unique")

    enumerator = _PolicyEnumerator(
        terminal_losses,
        likelihoods,
        costs,
        loss_tolerance=compression_tolerance,
        max_raw_trees=raw_cap,
        max_policies=policy_cap,
    )
    policies = enumerator.enumerate(tuple(range(len(likelihoods))), depth)
    expected_loss = _immutable_float64(
        np.column_stack(
            [policy.expected_loss_by_hypothesis for policy in policies]
        )
    )
    pairwise = _pairwise_support_gaps(
        expected_loss,
        prior,
        classes,
        masses,
    )
    regrets = _immutable_float64(np.max(pairwise, axis=1))

    fallback_key = f"act:{fallback_action}"
    fallback_matches = [
        index
        for index, policy in enumerate(policies)
        if policy.canonical_key == fallback_key
    ]
    if len(fallback_matches) != 1:
        raise RuntimeError("caller-owned fallback action tree was not retained exactly")
    fallback_policy = fallback_matches[0]

    minimum = float(np.min(regrets))
    minimizers = np.isclose(
        regrets,
        minimum,
        rtol=0.0,
        atol=_NUMERICAL_ATOL,
    )
    minimizer_count = int(np.count_nonzero(minimizers))
    candidate = int(np.flatnonzero(minimizers)[0]) if minimizer_count else None
    fallback_reason: str | None = None
    if candidate is None:
        output = fallback_policy
        fallback_reason = "no-finite-policy-minimizer"
    elif minimizer_count != 1:
        output = fallback_policy
        fallback_reason = "nonunique-minimax-policy"
    elif regrets[candidate] > tolerance + _NUMERICAL_ATOL:
        output = fallback_policy
        fallback_reason = "minimax-regret-exceeds-tolerance"
    else:
        output = candidate
        if output == fallback_policy:
            fallback_reason = "registered-fallback-is-minimax-policy"

    selected_policy = policies[output]
    used_fallback = output == fallback_policy
    output_mode = "fallback" if used_fallback else selected_policy.mode
    return AdaptivePolicyTreeCertificateV1(
        prior_probabilities=prior,
        class_index_by_hypothesis=classes,
        class_masses=masses,
        terminal_loss_by_hypothesis_action=terminal_losses,
        probe_likelihoods=likelihoods,
        probe_costs=costs,
        probe_names=names,
        policies=policies,
        expected_loss_by_hypothesis_policy=expected_loss,
        pairwise_worst_case_gap=pairwise,
        worst_case_regret=regrets,
        regret_tolerance=tolerance,
        fallback_action_index=fallback_action,
        fallback_policy_index=fallback_policy,
        minimizer_policy_mask=_immutable_bool(minimizers),
        minimizer_count=minimizer_count,
        candidate_policy_index=candidate,
        output_policy_index=output,
        output_mode=output_mode,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        raw_tree_count=enumerator.raw_tree_count,
        loss_equivalent_tree_count=enumerator.loss_equivalent_tree_count,
        dominance_pruned_tree_count=enumerator.dominance_pruned_tree_count,
    )


def terminal_action_for_probe_outcomes(
    policy: AdaptivePolicyTreeV1,
    outcome_index_by_probe: Sequence[int],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Traverse one frozen tree without any post-outcome re-optimization."""

    outcomes = tuple(int(value) for value in outcome_index_by_probe)
    node = policy
    trace: list[tuple[int, int]] = []
    while node.mode == "sense":
        if node.probe_index is None:
            raise RuntimeError("sensing tree is missing its probe index")
        if node.probe_index >= len(outcomes):
            raise ValueError("outcome_index_by_probe does not cover the policy probes")
        outcome = outcomes[node.probe_index]
        if outcome < 0 or outcome >= len(node.children):
            raise ValueError("probe outcome is outside the registered outcome roster")
        trace.append((node.probe_index, outcome))
        node = node.children[outcome]
    if node.mode != "act" or node.action_index is None:
        raise RuntimeError("policy tree did not terminate in a physical action")
    return node.action_index, tuple(trace)


def _logged_policy_loss(
    policy: AdaptivePolicyTreeV1,
    terminal_losses: FloatArray,
    probe_outcomes: IntArray,
    probe_costs: FloatArray,
    cache: dict[str, FloatArray],
) -> FloatArray:
    if policy.canonical_key in cache:
        return cache[policy.canonical_key]
    if policy.mode == "act":
        if policy.action_index is None:
            raise RuntimeError("terminal policy is missing its action index")
        result = np.asarray(terminal_losses[:, :, policy.action_index])
    else:
        if policy.probe_index is None:
            raise RuntimeError("sensing policy is missing its probe index")
        labels = probe_outcomes[:, :, policy.probe_index]
        result = np.full(labels.shape, probe_costs[policy.probe_index], dtype=float)
        for outcome_index, child in enumerate(policy.children):
            child_loss = _logged_policy_loss(
                child,
                terminal_losses,
                probe_outcomes,
                probe_costs,
                cache,
            )
            mask = labels == outcome_index
            result[mask] += child_loss[mask]
    frozen = _immutable_float64(result)
    cache[policy.canonical_key] = frozen
    return frozen


def logged_policy_tree_regret_tensor(
    certificate: AdaptivePolicyTreeCertificateV1,
    terminal_loss_by_trajectory_decision_action: object,
    probe_outcome_index_by_trajectory_decision_probe: object,
) -> LoggedPolicyTreeRegretTensorV1:
    """Evaluate every precommitted tree on logged complete trajectories."""

    terminal_raw = np.asarray(terminal_loss_by_trajectory_decision_action)
    if terminal_raw.dtype.kind not in "iuf":
        raise ValueError("logged terminal losses must be numeric")
    terminal = np.ascontiguousarray(terminal_raw, dtype=np.float64)
    action_count = certificate.terminal_loss_by_hypothesis_action.shape[1]
    if (
        terminal.ndim != 3
        or terminal.shape[2] != action_count
        or terminal.shape[0] == 0
        or terminal.shape[1] == 0
        or not np.all(np.isfinite(terminal))
    ):
        raise ValueError(
            "logged terminal losses must have shape "
            "(positive_trajectory_count, positive_decision_count, action_count)"
        )
    outcome_raw = np.asarray(probe_outcome_index_by_trajectory_decision_probe)
    if outcome_raw.dtype.kind not in "iu":
        raise ValueError("logged probe outcomes must contain integers")
    outcomes = np.ascontiguousarray(outcome_raw, dtype=np.int64)
    expected_shape = terminal.shape[:2] + (len(certificate.probe_likelihoods),)
    if outcomes.shape != expected_shape or np.any(outcomes < 0):
        raise ValueError(f"logged probe outcomes must have shape {expected_shape}")
    for probe_index, likelihood in enumerate(certificate.probe_likelihoods):
        if np.any(outcomes[:, :, probe_index] >= likelihood.shape[1]):
            raise ValueError(
                f"logged outcome for probe {probe_index} is outside its roster"
            )

    cache: dict[str, FloatArray] = {}
    policy_losses = np.stack(
        [
            _logged_policy_loss(
                policy,
                terminal,
                outcomes,
                certificate.probe_costs,
                cache,
            )
            for policy in certificate.policies
        ],
        axis=2,
    )
    best = np.min(policy_losses, axis=2)
    regret = np.maximum(policy_losses - best[:, :, None], 0.0)
    return LoggedPolicyTreeRegretTensorV1(
        policy_loss_by_trajectory_decision_policy=_immutable_float64(policy_losses),
        best_policy_loss_by_trajectory_decision=_immutable_float64(best),
        realized_regret_by_trajectory_decision_policy=_immutable_float64(regret),
    )


def trajectory_conformal_policy_tree_envelope(
    certificate: AdaptivePolicyTreeCertificateV1,
    logged: LoggedPolicyTreeRegretTensorV1,
    policy_scales: object,
    *,
    miscoverage: float,
    candidate_policy_mask: object | None = None,
) -> ScaledTrajectoryConformalPlanEnvelopeV1:
    """Calibrate one simultaneous envelope over decisions and policy trees."""

    registered = np.broadcast_to(
        certificate.worst_case_regret,
        logged.realized_regret_by_trajectory_decision_policy.shape,
    )
    return scaled_trajectory_conformal_plan_envelope(
        logged.realized_regret_by_trajectory_decision_policy,
        registered,
        policy_scales,
        miscoverage=miscoverage,
        candidate_plan_mask=candidate_policy_mask,
    )


def conformal_adaptive_policy_tree_decision(
    certificate: AdaptivePolicyTreeCertificateV1,
    envelope: ScaledTrajectoryConformalPlanEnvelopeV1,
    *,
    regret_tolerance: float | None = None,
) -> ConformalAdaptivePolicyTreeDecisionV1:
    """Choose one whole calibrated tree before probing, or return fallback."""

    if envelope.plan_count != certificate.policy_count:
        raise ValueError("conformal envelope policy count does not match certificate")
    tolerance = (
        certificate.regret_tolerance
        if regret_tolerance is None
        else _finite_nonnegative(regret_tolerance, name="regret_tolerance")
    )
    calibrated = certificate.worst_case_regret + envelope.inflation_by_plan
    finite_candidates = envelope.candidate_plan_mask & np.isfinite(calibrated)
    minimizers = np.zeros(certificate.policy_count, dtype=np.bool_)
    candidate: int | None = None
    if np.any(finite_candidates):
        minimum = float(np.min(calibrated[finite_candidates]))
        minimizers = finite_candidates & np.isclose(
            calibrated,
            minimum,
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        )
        candidate = int(np.flatnonzero(minimizers)[0])
    minimizer_count = int(np.count_nonzero(minimizers))
    fallback_reason: str | None = None
    if candidate is None:
        output = certificate.fallback_policy_index
        fallback_reason = "infinite-conformal-policy-envelope"
    elif minimizer_count != 1:
        output = certificate.fallback_policy_index
        fallback_reason = "nonunique-calibrated-minimax-policy"
    elif calibrated[candidate] > tolerance + _NUMERICAL_ATOL:
        output = certificate.fallback_policy_index
        fallback_reason = "calibrated-policy-regret-exceeds-tolerance"
    else:
        output = candidate
        if output == certificate.fallback_policy_index:
            fallback_reason = "registered-fallback-is-calibrated-minimax-policy"
    used_fallback = output == certificate.fallback_policy_index
    output_mode = "fallback" if used_fallback else certificate.policies[output].mode
    return ConformalAdaptivePolicyTreeDecisionV1(
        certificate=certificate,
        envelope=envelope,
        calibrated_regret_upper_by_policy=_immutable_float64(calibrated),
        regret_tolerance=tolerance,
        minimizer_policy_mask=_immutable_bool(minimizers),
        minimizer_count=minimizer_count,
        candidate_policy_index=candidate,
        output_policy_index=output,
        output_mode=output_mode,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
    )


__all__ = [
    "ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY",
    "ADAPTIVE_STOCHASTIC_POLICY_TREE_SEMANTICS",
    "ADAPTIVE_STOCHASTIC_POLICY_TREE_VERSION",
    "AdaptivePolicyTreeCertificateV1",
    "AdaptivePolicyTreeV1",
    "ConformalAdaptivePolicyTreeDecisionV1",
    "LoggedPolicyTreeRegretTensorV1",
    "adaptive_stochastic_policy_tree_certificate",
    "conformal_adaptive_policy_tree_decision",
    "logged_policy_tree_regret_tensor",
    "terminal_action_for_probe_outcomes",
    "trajectory_conformal_policy_tree_envelope",
]
