"""Query-quotient posterior lifting without unsupported physical specificity.

A registered physical query may identify an equivalence class of hypotheses
without identifying one latent physical cause. This module updates only those
class masses, preserves prior conditionals within every class, and exposes the
remaining ambiguity of downstream physical quantities.
"""

from __future__ import annotations

from numbers import Real
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

QUERY_QUOTIENT_BELIEF_VERSION: Final = 1
QUERY_QUOTIENT_BELIEF_SEMANTICS: Final = (
    "minimum-forward-kl-lift-of-registered-query-quotient-v1"
)
QUERY_QUOTIENT_BELIEF_CLAIM_BOUNDARY: Final = (
    "The lift changes only registered quotient-class masses and preserves the "
    "prior conditional belief inside each class. It does not establish that the "
    "registered quotient is physically correct, identify a unique physical "
    "cause, authorize candidate admission, validate a provider, calibrate "
    "uncertainty, certify held-out transport, or certify safety."
)

_PROBABILITY_ATOL: Final = 1e-12
_INFORMATION_ATOL: Final = 1e-11


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
        raise ValueError(
            f"class_index must contain exactly {expected_size} entries"
        )
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


def _class_count(classes: IntArray) -> int:
    return int(np.max(classes)) + 1


def _aggregate(weights: FloatArray, classes: IntArray) -> FloatArray:
    result = np.bincount(
        classes,
        weights=weights,
        minlength=_class_count(classes),
    ).astype(np.float64, copy=False)
    result /= float(np.sum(result, dtype=np.float64))
    return _immutable_float64(result)


def _relative_entropy(posterior: FloatArray, prior: FloatArray, *, name: str) -> float:
    positive = posterior > 0.0
    if np.any(prior[positive] <= 0.0):
        raise ValueError(
            f"{name} is not absolutely continuous with respect to prior"
        )
    result = float(
        np.sum(
            posterior[positive]
            * (np.log(posterior[positive]) - np.log(prior[positive])),
            dtype=np.float64,
        )
    )
    if result < -_INFORMATION_ATOL:
        raise RuntimeError(f"{name} relative entropy became materially negative")
    return max(0.0, result)


class QueryQuotientInformationV1(NamedTuple):
    """KL chain-rule audit for one finite-hypothesis update."""

    prior_quotient_weights: FloatArray
    posterior_quotient_weights: FloatArray
    class_unsupported_specificity_nats: FloatArray
    total_information_nats: float
    quotient_information_nats: float
    unsupported_specificity_nats: float
    chain_rule_residual_nats: float

    @property
    def supported_information_fraction(self) -> float:
        if self.total_information_nats == 0.0:
            return 1.0
        return min(1.0, self.quotient_information_nats / self.total_information_nats)

    def summary(self) -> dict[str, object]:
        return {
            "semantics": QUERY_QUOTIENT_BELIEF_SEMANTICS,
            "total_information_nats": self.total_information_nats,
            "quotient_information_nats": self.quotient_information_nats,
            "unsupported_specificity_nats": self.unsupported_specificity_nats,
            "supported_information_fraction": self.supported_information_fraction,
            "chain_rule_residual_nats": self.chain_rule_residual_nats,
            "claim_boundary": QUERY_QUOTIENT_BELIEF_CLAIM_BOUNDARY,
        }


class MinimumInformationQueryLiftV1(NamedTuple):
    """Complete belief induced by a registered quotient posterior and prior."""

    prior_weights: FloatArray
    class_index: IntArray
    quotient_posterior_weights: FloatArray
    lifted_weights: FloatArray
    information: QueryQuotientInformationV1

    @property
    def hypothesis_count(self) -> int:
        return int(self.prior_weights.size)

    @property
    def quotient_class_count(self) -> int:
        return int(self.quotient_posterior_weights.size)

    def summary(self) -> dict[str, object]:
        return {
            "version": QUERY_QUOTIENT_BELIEF_VERSION,
            "hypothesis_count": self.hypothesis_count,
            "quotient_class_count": self.quotient_class_count,
            **self.information.summary(),
        }


class QueryAmbiguityEnvelopeV1(NamedTuple):
    """Exact componentwise expectation envelope under a quotient posterior."""

    quotient_weights: FloatArray
    class_minimum: FloatArray
    class_maximum: FloatArray
    lower: FloatArray
    upper: FloatArray
    width: FloatArray
    identifiability_tolerance: float
    identified_mask: BoolArray

    @property
    def endpoint_dimension(self) -> int:
        return int(self.width.size)

    @property
    def all_identified(self) -> bool:
        return bool(np.all(self.identified_mask))

    @property
    def maximum_width(self) -> float:
        return float(np.max(self.width))

    def summary(self) -> dict[str, object]:
        return {
            "version": QUERY_QUOTIENT_BELIEF_VERSION,
            "semantics": QUERY_QUOTIENT_BELIEF_SEMANTICS,
            "endpoint_dimension": self.endpoint_dimension,
            "all_identified": self.all_identified,
            "maximum_width": self.maximum_width,
            "identifiability_tolerance": self.identifiability_tolerance,
            "claim_boundary": QUERY_QUOTIENT_BELIEF_CLAIM_BOUNDARY,
        }


def aggregate_to_query_quotient(
    weights: object,
    class_index: object,
) -> FloatArray:
    """Aggregate a finite-hypothesis belief over registered query classes."""

    probabilities = _probability_vector(weights, name="weights")
    classes = _class_index(class_index, expected_size=probabilities.size)
    return _aggregate(probabilities, classes)


def query_quotient_information_decomposition(
    prior_weights: object,
    posterior_weights: object,
    class_index: object,
) -> QueryQuotientInformationV1:
    """Split full-update KL into quotient and within-class information.

    The identity is
    ``KL(q || p) = KL(q_C || p_C) + sum_C q_C KL(q(.|C) || p(.|C))``.
    The final term is unsupported specificity: a quotient posterior alone does
    not determine it.
    """

    prior = _probability_vector(prior_weights, name="prior_weights")
    posterior = _probability_vector(
        posterior_weights,
        name="posterior_weights",
        expected_size=prior.size,
    )
    classes = _class_index(class_index, expected_size=prior.size)
    if np.any((posterior > 0.0) & (prior <= 0.0)):
        raise ValueError(
            "posterior_weights are not absolutely continuous with respect to prior"
        )

    prior_quotient = _aggregate(prior, classes)
    posterior_quotient = _aggregate(posterior, classes)
    total = _relative_entropy(posterior, prior, name="posterior_weights")
    quotient = _relative_entropy(
        posterior_quotient,
        prior_quotient,
        name="posterior_quotient_weights",
    )

    contributions = np.zeros(_class_count(classes), dtype=np.float64)
    for class_id in range(contributions.size):
        posterior_mass = float(posterior_quotient[class_id])
        if posterior_mass == 0.0:
            continue
        prior_mass = float(prior_quotient[class_id])
        members = classes == class_id
        contributions[class_id] = posterior_mass * _relative_entropy(
            _immutable_float64(posterior[members] / posterior_mass),
            _immutable_float64(prior[members] / prior_mass),
            name=f"posterior conditional for class {class_id}",
        )

    unsupported = float(np.sum(contributions, dtype=np.float64))
    residual = total - quotient - unsupported
    if not np.isclose(residual, 0.0, rtol=1e-10, atol=_INFORMATION_ATOL):
        raise RuntimeError("KL chain-rule residual exceeds numerical tolerance")
    return QueryQuotientInformationV1(
        prior_quotient_weights=prior_quotient,
        posterior_quotient_weights=posterior_quotient,
        class_unsupported_specificity_nats=_immutable_float64(contributions),
        total_information_nats=total,
        quotient_information_nats=quotient,
        unsupported_specificity_nats=unsupported,
        chain_rule_residual_nats=residual,
    )


def minimum_information_query_lift(
    prior_weights: object,
    class_index: object,
    quotient_posterior_weights: object,
) -> MinimumInformationQueryLiftV1:
    """Return the unique minimum-forward-KL lift of a quotient posterior.

    For hypothesis ``i`` in class ``c(i)``,
    ``q_i = quotient_posterior[c(i)] * prior_i / prior_quotient[c(i)]``.
    A class assigned positive posterior mass must have positive prior mass.
    """

    prior = _probability_vector(prior_weights, name="prior_weights")
    classes = _class_index(class_index, expected_size=prior.size)
    prior_quotient = _aggregate(prior, classes)
    quotient_posterior = _probability_vector(
        quotient_posterior_weights,
        name="quotient_posterior_weights",
        expected_size=prior_quotient.size,
    )
    unsupported = (
        quotient_posterior > 0.0
    ) & (prior_quotient <= 0.0)
    if np.any(unsupported):
        raise ValueError(
            "positive quotient posterior mass has zero prior support for classes "
            f"{np.flatnonzero(unsupported).tolist()}"
        )

    scale = np.zeros_like(prior_quotient)
    supported = prior_quotient > 0.0
    scale[supported] = quotient_posterior[supported] / prior_quotient[supported]
    lifted = prior * scale[classes]
    lifted /= float(np.sum(lifted, dtype=np.float64))
    lifted = _immutable_float64(lifted)
    information = query_quotient_information_decomposition(prior, lifted, classes)
    if information.unsupported_specificity_nats > _INFORMATION_ATOL:
        raise RuntimeError("minimum-information lift added within-class specificity")
    return MinimumInformationQueryLiftV1(
        prior_weights=prior,
        class_index=classes,
        quotient_posterior_weights=quotient_posterior,
        lifted_weights=lifted,
        information=information,
    )


def query_ambiguity_envelope(
    quotient_weights: object,
    class_index: object,
    downstream_values: object,
    *,
    identifiability_tolerance: float = 0.0,
) -> QueryAmbiguityEnvelopeV1:
    """Bound expected endpoints over every full lift of a quotient belief.

    ``downstream_values`` has shape ``(hypotheses,)`` or
    ``(hypotheses, endpoints)``. Intervals are exact componentwise; extrema for
    different vector components need not be jointly attainable.
    """

    classes = _class_index(class_index)
    quotient = _probability_vector(
        quotient_weights,
        name="quotient_weights",
        expected_size=_class_count(classes),
    )
    raw_values = np.asarray(downstream_values)
    if raw_values.dtype.kind not in "iuf":
        raise ValueError("downstream_values must contain real numeric values")
    values = np.ascontiguousarray(raw_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != classes.size:
        raise ValueError(
            "downstream_values must have one row per physical hypothesis"
        )
    if values.shape[1] == 0:
        raise ValueError("downstream_values must contain at least one endpoint")
    if not np.all(np.isfinite(values)):
        raise ValueError("downstream_values must be finite")
    tolerance = _finite_nonnegative(
        identifiability_tolerance,
        name="identifiability_tolerance",
    )

    class_minimum = np.empty((quotient.size, values.shape[1]), dtype=np.float64)
    class_maximum = np.empty_like(class_minimum)
    for class_id in range(quotient.size):
        members = values[classes == class_id]
        class_minimum[class_id] = np.min(members, axis=0)
        class_maximum[class_id] = np.max(members, axis=0)

    lower = quotient @ class_minimum
    upper = quotient @ class_maximum
    width = np.maximum(0.0, upper - lower)
    return QueryAmbiguityEnvelopeV1(
        quotient_weights=quotient,
        class_minimum=_immutable_float64(class_minimum),
        class_maximum=_immutable_float64(class_maximum),
        lower=_immutable_float64(lower),
        upper=_immutable_float64(upper),
        width=_immutable_float64(width),
        identifiability_tolerance=tolerance,
        identified_mask=_immutable_bool(width <= tolerance),
    )


__all__ = [
    "QUERY_QUOTIENT_BELIEF_CLAIM_BOUNDARY",
    "QUERY_QUOTIENT_BELIEF_SEMANTICS",
    "QUERY_QUOTIENT_BELIEF_VERSION",
    "MinimumInformationQueryLiftV1",
    "QueryAmbiguityEnvelopeV1",
    "QueryQuotientInformationV1",
    "aggregate_to_query_quotient",
    "minimum_information_query_lift",
    "query_ambiguity_envelope",
    "query_quotient_information_decomposition",
]
