"""Exact decision certificates from a registered query-quotient posterior.

A physical observation may determine only quotient-class masses over latent
hypotheses. This module certifies whether that information is nevertheless
sufficient for a downstream finite decision, without selecting an unsupported
within-class latent explanation.

For action ``a`` and benchmark action ``b``, the exact worst-case loss gap over
all complete beliefs ``q`` compatible with quotient masses ``lambda`` and prior
support ``q << p`` is

    sup_q E_q[L(a, H) - L(b, H)]
      = sum_c lambda_c max_{i in c: p_i > 0} (L(a, i) - L(b, i)).

The exact worst-case regret of action ``a`` is the maximum of this quantity over
all benchmark actions. An action may therefore be admitted whenever its
worst-case regret is below a separately declared tolerance, even when the full
latent state—or even every scalar query endpoint—is not identifiable.

The certificate does not validate the quotient, provider, loss model, tolerance,
or deployment context. Those remain separately registered responsibilities.
"""

from __future__ import annotations

from numbers import Real
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

QUERY_DECISION_CERTIFICATE_VERSION: Final = 1
QUERY_DECISION_CERTIFICATE_SEMANTICS: Final = (
    "exact-worst-case-regret-over-registered-query-quotient-and-prior-support-v1"
)
QUERY_DECISION_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied finite hypotheses, prior "
    "support, registered quotient masses, and loss matrix. It does not establish "
    "that the quotient is physically correct, validate a provider, identify a "
    "unique physical cause, calibrate uncertainty, justify the loss or regret "
    "tolerance, certify held-out transport, authorize deployment, or certify safety."
)

_PROBABILITY_ATOL: Final = 1e-12
_NUMERICAL_ATOL: Final = 1e-12


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
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(array, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL):
        raise ValueError(f"{name} must sum to one")
    return _immutable_float64(array / total)


def _class_index(value: object, *, expected_size: int | None = None) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError("class_index must contain integer class labels")
    array = np.ascontiguousarray(raw, dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("class_index must be a nonempty one-dimensional vector")
    if expected_size is not None and array.size != expected_size:
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


def _loss_matrix(value: object, *, hypothesis_count: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("loss_by_hypothesis_action must contain real numeric values")
    losses = np.ascontiguousarray(raw, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] != hypothesis_count:
        raise ValueError(
            "loss_by_hypothesis_action must have shape (hypothesis_count, action_count)"
        )
    if losses.shape[1] < 2:
        raise ValueError("at least two candidate actions are required")
    if not np.all(np.isfinite(losses)):
        raise ValueError("loss_by_hypothesis_action must be finite")
    return _immutable_float64(losses)


class QueryDecisionCertificateV1(NamedTuple):
    """Exact finite-action decision certificate for one quotient posterior."""

    prior_weights: FloatArray
    prior_support_mask: BoolArray
    quotient_weights: FloatArray
    class_index: IntArray
    class_pairwise_max_loss_gap: FloatArray
    pairwise_worst_case_loss_gap: FloatArray
    worst_case_regret: FloatArray
    minimax_action_index: int
    minimax_worst_case_regret: float
    regret_tolerance: float
    tolerance_admissible_action_mask: BoolArray
    robustly_optimal_action_mask: BoolArray

    @property
    def hypothesis_count(self) -> int:
        return int(self.class_index.size)

    @property
    def prior_support_count(self) -> int:
        return int(np.count_nonzero(self.prior_support_mask))

    @property
    def quotient_class_count(self) -> int:
        return int(self.quotient_weights.size)

    @property
    def action_count(self) -> int:
        return int(self.worst_case_regret.size)

    @property
    def has_tolerance_admissible_action(self) -> bool:
        return bool(np.any(self.tolerance_admissible_action_mask))

    @property
    def uniquely_tolerance_identified(self) -> bool:
        return bool(np.count_nonzero(self.tolerance_admissible_action_mask) == 1)

    @property
    def has_robustly_optimal_action(self) -> bool:
        return bool(np.any(self.robustly_optimal_action_mask))

    @property
    def uniquely_robustly_optimal(self) -> bool:
        return bool(np.count_nonzero(self.robustly_optimal_action_mask) == 1)

    def summary(self) -> dict[str, object]:
        return {
            "version": QUERY_DECISION_CERTIFICATE_VERSION,
            "semantics": QUERY_DECISION_CERTIFICATE_SEMANTICS,
            "hypothesis_count": self.hypothesis_count,
            "prior_support_count": self.prior_support_count,
            "quotient_class_count": self.quotient_class_count,
            "action_count": self.action_count,
            "minimax_action_index": self.minimax_action_index,
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "regret_tolerance": self.regret_tolerance,
            "has_tolerance_admissible_action": self.has_tolerance_admissible_action,
            "uniquely_tolerance_identified": self.uniquely_tolerance_identified,
            "has_robustly_optimal_action": self.has_robustly_optimal_action,
            "uniquely_robustly_optimal": self.uniquely_robustly_optimal,
            "claim_boundary": QUERY_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
        }


def query_decision_certificate(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    *,
    regret_tolerance: float = 0.0,
) -> QueryDecisionCertificateV1:
    """Return exact worst-case finite-action regret over every supported lift.

    Parameters
    ----------
    prior_weights:
        Prior probability for every finite physical hypothesis. Its positive
        entries define the admissible support ``q << p``. The certificate is
        invariant to changes in positive prior magnitudes that preserve support.
    quotient_weights:
        Posterior mass for every contiguous quotient class.
    class_index:
        One class label per physical hypothesis. Labels must be contiguous from
        zero. A class with positive quotient mass must have positive prior mass.
    loss_by_hypothesis_action:
        Matrix with shape ``(hypothesis_count, action_count)``. Entry ``[i, a]``
        is the registered loss of action ``a`` under hypothesis ``i``.
    regret_tolerance:
        Separately declared maximum acceptable worst-case regret.

    Returns
    -------
    QueryDecisionCertificateV1
        Exact pairwise worst-case gaps, exact action-wise worst-case regret, the
        deterministic lowest-index minimax action, and exact/tolerance masks.
    """

    prior = _probability_vector(prior_weights, name="prior_weights")
    classes = _class_index(class_index, expected_size=prior.size)
    class_count = int(np.max(classes)) + 1
    quotient = _probability_vector(
        quotient_weights,
        name="quotient_weights",
        expected_size=class_count,
    )
    losses = _loss_matrix(
        loss_by_hypothesis_action,
        hypothesis_count=classes.size,
    )
    tolerance = _finite_nonnegative(
        regret_tolerance,
        name="regret_tolerance",
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

    # diff[i, a, b] = loss(a | i) - loss(b | i)
    pairwise_loss_difference = losses[:, :, None] - losses[:, None, :]

    action_count = losses.shape[1]
    class_pairwise_max = np.zeros(
        (class_count, action_count, action_count),
        dtype=np.float64,
    )
    for class_id in range(class_count):
        members = pairwise_loss_difference[(classes == class_id) & prior_support]
        if members.shape[0] == 0:
            # Necessarily a zero-posterior class due to the support check above.
            continue
        class_pairwise_max[class_id] = np.max(members, axis=0)

    # Fixed quotient masses allow each class conditional to concentrate on its
    # own supported maximizing hypothesis, making this support function exact.
    pairwise_worst_case = np.tensordot(
        quotient,
        class_pairwise_max,
        axes=(0, 0),
    )
    np.fill_diagonal(pairwise_worst_case, 0.0)

    # sup_q [R_q(a) - min_b R_q(b)]
    # = sup_q max_b [R_q(a) - R_q(b)]
    # = max_b sup_q [R_q(a) - R_q(b)].
    worst_case_regret = np.max(pairwise_worst_case, axis=1)
    worst_case_regret = np.maximum(worst_case_regret, 0.0)

    minimum_regret = float(np.min(worst_case_regret))
    minimax_indices = np.flatnonzero(
        np.isclose(
            worst_case_regret,
            minimum_regret,
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        )
    )
    minimax_action_index = int(minimax_indices[0])

    tolerance_mask = worst_case_regret <= tolerance + _NUMERICAL_ATOL
    robust_mask = np.all(
        pairwise_worst_case <= _NUMERICAL_ATOL,
        axis=1,
    )

    return QueryDecisionCertificateV1(
        prior_weights=prior,
        prior_support_mask=_immutable_bool(prior_support),
        quotient_weights=quotient,
        class_index=classes,
        class_pairwise_max_loss_gap=_immutable_float64(class_pairwise_max),
        pairwise_worst_case_loss_gap=_immutable_float64(pairwise_worst_case),
        worst_case_regret=_immutable_float64(worst_case_regret),
        minimax_action_index=minimax_action_index,
        minimax_worst_case_regret=minimum_regret,
        regret_tolerance=tolerance,
        tolerance_admissible_action_mask=_immutable_bool(tolerance_mask),
        robustly_optimal_action_mask=_immutable_bool(robust_mask),
    )


__all__ = [
    "QUERY_DECISION_CERTIFICATE_CLAIM_BOUNDARY",
    "QUERY_DECISION_CERTIFICATE_SEMANTICS",
    "QUERY_DECISION_CERTIFICATE_VERSION",
    "QueryDecisionCertificateV1",
    "query_decision_certificate",
]
