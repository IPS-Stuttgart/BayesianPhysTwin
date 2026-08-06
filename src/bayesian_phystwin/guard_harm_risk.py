"""Finite-group certification of harmful accepted-update risk.

A baseline-relative guard may be evaluated at many score thresholds, but a
claim-bearing threshold must be frozen before certification outcomes are opened.
This module certifies one such fixed threshold on independent physical groups
using an exact one-sided binomial confidence bound. It does not select a
threshold from the certification outcomes and does not replace exact fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)

GUARD_HARM_RISK_CERTIFICATE_SCHEMA = (
    "bayesian_phystwin.guard_harm_risk_certificate"
)
GUARD_HARM_RISK_CERTIFICATE_VERSION = 1
RISK_SCORE_SEMANTICS = "lower-is-safer-inclusive-threshold-v1"
BOUND_METHOD = "one-sided-clopper-pearson-binomial-v1"

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _open_probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def _float_vector(value: object, *, name: str) -> FloatArray:
    try:
        result: FloatArray = np.array(
            value,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real vector") from error
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector")
    return result


def _boolean_vector(value: object, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean vector")
    return np.array(raw, dtype=np.bool_, copy=True, order="C")


def _immutable_float(value: FloatArray) -> FloatArray:
    canonical = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    return np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.dtype("<f8"),
    ).reshape(canonical.shape)


def _immutable_bool(value: BoolArray) -> BoolArray:
    canonical = np.asarray(value, dtype=np.bool_, order="C")
    return np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.bool_,
    ).reshape(canonical.shape)


def _group_id_tuple(value: object, *, expected_count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("group_ids must be a sequence of canonical strings")
    result = tuple(
        _canonical_string(item, name=f"group_ids[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(result) != expected_count:
        raise ValueError("group_ids length must match certification arrays")
    if len(set(result)) != len(result):
        raise ValueError("group_ids must not contain duplicates")
    return result


def _log_binomial_probability(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    return (
        math.lgamma(trials + 1.0)
        - math.lgamma(successes + 1.0)
        - math.lgamma(trials - successes + 1.0)
        + successes * math.log(probability)
        + (trials - successes) * math.log1p(-probability)
    )


def _binomial_cdf(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes == trials else 0.0
    logs = np.asarray(
        [
            _log_binomial_probability(index, trials, probability)
            for index in range(successes + 1)
        ],
        dtype=np.float64,
    )
    maximum = float(np.max(logs))
    return float(math.exp(maximum) * np.sum(np.exp(logs - maximum)))


def one_sided_binomial_upper_bound(
    harmful_count: int,
    accepted_count: int,
    confidence_level: float,
) -> float:
    """Return the exact one-sided Clopper--Pearson upper confidence bound."""

    accepted = genuine_integer(
        accepted_count,
        name="accepted_count",
        minimum=0,
    )
    harmful = genuine_integer(
        harmful_count,
        name="harmful_count",
        minimum=0,
    )
    if harmful > accepted:
        raise ValueError("harmful_count cannot exceed accepted_count")
    confidence = _open_probability(confidence_level, name="confidence_level")
    if accepted == 0 or harmful == accepted:
        return 1.0
    tail_probability = 1.0 - confidence
    if harmful == 0:
        return float(-math.expm1(math.log(tail_probability) / accepted))

    lower = harmful / accepted
    upper = 1.0
    for _ in range(160):
        midpoint = 0.5 * (lower + upper)
        if _binomial_cdf(harmful, accepted, midpoint) > tail_probability:
            lower = midpoint
        else:
            upper = midpoint
    return float(upper)


def minimum_zero_harm_groups_for_certificate(
    target_harm_probability: float,
    confidence_level: float,
) -> int:
    """Return the smallest zero-harm accepted-group count that can certify."""

    target = _open_probability(
        target_harm_probability,
        name="target_harm_probability",
    )
    confidence = _open_probability(confidence_level, name="confidence_level")
    tail_probability = 1.0 - confidence
    estimate = max(
        1,
        int(
            math.ceil(
                math.log(tail_probability) / math.log1p(-target)
            )
        ),
    )
    while (
        one_sided_binomial_upper_bound(0, estimate, confidence)
        > target
    ):
        estimate += 1
    while (
        estimate > 1
        and one_sided_binomial_upper_bound(0, estimate - 1, confidence)
        <= target
    ):
        estimate -= 1
    return estimate


@dataclass(frozen=True, slots=True)
class GuardHarmRiskCertificateV1:
    """Exact finite-group harm-risk certificate for one frozen guard."""

    guard_policy_id: str
    threshold_source_artifact_id: str
    certification_partition_id: str
    statistical_unit: str
    metric: str
    group_ids: Sequence[str]
    risk_scores: FloatArray
    candidate_losses: FloatArray
    fallback_losses: FloatArray
    fallback_identity_verified: BoolArray
    threshold: float
    harm_margin: float
    target_harm_probability: float
    confidence_level: float
    minimum_accepted_group_count: int
    accepted_mask: BoolArray
    harmful_mask: BoolArray
    accepted_count: int
    harmful_accepted_count: int
    one_sided_upper_bound: float
    certified: bool
    threshold_frozen_before_certification_outcomes: bool
    certification_outcomes_used_for_threshold_selection: bool
    certification_groups_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        guard_policy_id = sha256_digest(
            self.guard_policy_id,
            name="guard_policy_id",
        )
        threshold_source_id = sha256_digest(
            self.threshold_source_artifact_id,
            name="threshold_source_artifact_id",
        )
        partition_id = sha256_digest(
            self.certification_partition_id,
            name="certification_partition_id",
        )
        statistical_unit = _canonical_string(
            self.statistical_unit,
            name="statistical_unit",
        )
        metric = _canonical_string(self.metric, name="metric")
        risk_scores = _float_vector(self.risk_scores, name="risk_scores")
        candidate_losses = _float_vector(
            self.candidate_losses,
            name="candidate_losses",
        )
        fallback_losses = _float_vector(
            self.fallback_losses,
            name="fallback_losses",
        )
        fallback_verified = _boolean_vector(
            self.fallback_identity_verified,
            name="fallback_identity_verified",
        )
        count = len(risk_scores)
        if not count:
            raise ValueError("at least one independent certification group is required")
        if any(
            len(value) != count
            for value in (
                candidate_losses,
                fallback_losses,
                fallback_verified,
            )
        ):
            raise ValueError("certification arrays must have equal length")
        if np.any(candidate_losses < 0.0) or np.any(fallback_losses < 0.0):
            raise ValueError("candidate and fallback losses must be nonnegative")
        group_ids = _group_id_tuple(self.group_ids, expected_count=count)
        order = np.argsort(np.asarray(group_ids, dtype=object), kind="mergesort")
        group_ids = tuple(group_ids[int(index)] for index in order)
        risk_scores = risk_scores[order]
        candidate_losses = candidate_losses[order]
        fallback_losses = fallback_losses[order]
        fallback_verified = fallback_verified[order]

        threshold = _finite_real(self.threshold, name="threshold")
        harm_margin = _finite_real(
            self.harm_margin,
            name="harm_margin",
            minimum=0.0,
        )
        target = _open_probability(
            self.target_harm_probability,
            name="target_harm_probability",
        )
        confidence = _open_probability(
            self.confidence_level,
            name="confidence_level",
        )
        minimum_accepted = genuine_integer(
            self.minimum_accepted_group_count,
            name="minimum_accepted_group_count",
            minimum=1,
        )
        threshold_frozen = genuine_boolean(
            self.threshold_frozen_before_certification_outcomes,
            name="threshold_frozen_before_certification_outcomes",
        )
        outcomes_used = genuine_boolean(
            self.certification_outcomes_used_for_threshold_selection,
            name="certification_outcomes_used_for_threshold_selection",
        )
        groups_independent = genuine_boolean(
            self.certification_groups_independent,
            name="certification_groups_independent",
        )
        if not threshold_frozen:
            raise ValueError(
                "threshold must be frozen before certification outcomes are opened"
            )
        if outcomes_used:
            raise ValueError(
                "certification outcomes cannot select the certified threshold"
            )
        if not groups_independent:
            raise ValueError(
                "certification groups must be independent physical units"
            )

        expected_accepted = risk_scores <= threshold
        expected_harmful = candidate_losses > fallback_losses + harm_margin
        accepted_mask = _boolean_vector(
            self.accepted_mask,
            name="accepted_mask",
        )
        harmful_mask = _boolean_vector(
            self.harmful_mask,
            name="harmful_mask",
        )
        accepted_mask = accepted_mask[order]
        harmful_mask = harmful_mask[order]
        if not np.array_equal(accepted_mask, expected_accepted):
            raise ValueError("accepted_mask does not match the inclusive threshold")
        if not np.array_equal(harmful_mask, expected_harmful):
            raise ValueError("harmful_mask does not match baseline-relative losses")
        if np.any(~fallback_verified[~expected_accepted]):
            raise ValueError(
                "every rejected certification group must verify exact fallback"
            )

        accepted_count = int(np.sum(expected_accepted))
        harmful_count = int(np.sum(expected_accepted & expected_harmful))
        supplied_accepted_count = genuine_integer(
            self.accepted_count,
            name="accepted_count",
            minimum=0,
        )
        supplied_harmful_count = genuine_integer(
            self.harmful_accepted_count,
            name="harmful_accepted_count",
            minimum=0,
        )
        if supplied_accepted_count != accepted_count:
            raise ValueError("accepted_count does not match accepted_mask")
        if supplied_harmful_count != harmful_count:
            raise ValueError(
                "harmful_accepted_count does not match accepted harmful groups"
            )
        expected_upper = one_sided_binomial_upper_bound(
            harmful_count,
            accepted_count,
            confidence,
        )
        supplied_upper = _finite_real(
            self.one_sided_upper_bound,
            name="one_sided_upper_bound",
            minimum=0.0,
        )
        if supplied_upper > 1.0 or not np.isclose(
            supplied_upper,
            expected_upper,
            atol=1e-14,
            rtol=1e-13,
        ):
            raise ValueError(
                "one_sided_upper_bound does not match exact binomial inversion"
            )
        expected_certified = (
            accepted_count >= minimum_accepted
            and expected_upper <= target
        )
        certified = genuine_boolean(self.certified, name="certified")
        if certified != expected_certified:
            raise ValueError("certified does not match support and risk gates")

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="guard harm-risk certificate metadata",
        )
        object.__setattr__(self, "guard_policy_id", guard_policy_id)
        object.__setattr__(
            self,
            "threshold_source_artifact_id",
            threshold_source_id,
        )
        object.__setattr__(self, "certification_partition_id", partition_id)
        object.__setattr__(self, "statistical_unit", statistical_unit)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "risk_scores", _immutable_float(risk_scores))
        object.__setattr__(
            self,
            "candidate_losses",
            _immutable_float(candidate_losses),
        )
        object.__setattr__(
            self,
            "fallback_losses",
            _immutable_float(fallback_losses),
        )
        object.__setattr__(
            self,
            "fallback_identity_verified",
            _immutable_bool(fallback_verified),
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "harm_margin", harm_margin)
        object.__setattr__(self, "target_harm_probability", target)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "minimum_accepted_group_count",
            minimum_accepted,
        )
        object.__setattr__(
            self,
            "accepted_mask",
            _immutable_bool(expected_accepted),
        )
        object.__setattr__(
            self,
            "harmful_mask",
            _immutable_bool(expected_harmful),
        )
        object.__setattr__(self, "accepted_count", accepted_count)
        object.__setattr__(
            self,
            "harmful_accepted_count",
            harmful_count,
        )
        object.__setattr__(self, "one_sided_upper_bound", expected_upper)
        object.__setattr__(self, "certified", expected_certified)
        object.__setattr__(
            self,
            "threshold_frozen_before_certification_outcomes",
            threshold_frozen,
        )
        object.__setattr__(
            self,
            "certification_outcomes_used_for_threshold_selection",
            outcomes_used,
        )
        object.__setattr__(
            self,
            "certification_groups_independent",
            groups_independent,
        )
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match certificate content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_count / self.group_count

    @property
    def observed_harm_rate(self) -> float | None:
        if self.accepted_count == 0:
            return None
        return self.harmful_accepted_count / self.accepted_count

    @property
    def minimum_zero_harm_accepted_groups(self) -> int:
        return minimum_zero_harm_groups_for_certificate(
            self.target_harm_probability,
            self.confidence_level,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GUARD_HARM_RISK_CERTIFICATE_SCHEMA,
            "schema_version": GUARD_HARM_RISK_CERTIFICATE_VERSION,
            "bound_method": BOUND_METHOD,
            "risk_score_semantics": RISK_SCORE_SEMANTICS,
            "guard_policy_id": self.guard_policy_id,
            "threshold_source_artifact_id": (
                self.threshold_source_artifact_id
            ),
            "certification_partition_id": self.certification_partition_id,
            "statistical_unit": self.statistical_unit,
            "metric": self.metric,
            "group_ids": list(self.group_ids),
            "risk_scores": self.risk_scores.tolist(),
            "candidate_losses": self.candidate_losses.tolist(),
            "fallback_losses": self.fallback_losses.tolist(),
            "fallback_identity_verified": (
                self.fallback_identity_verified.tolist()
            ),
            "threshold": self.threshold,
            "harm_margin": self.harm_margin,
            "target_harm_probability": self.target_harm_probability,
            "confidence_level": self.confidence_level,
            "minimum_accepted_group_count": (
                self.minimum_accepted_group_count
            ),
            "accepted_mask": self.accepted_mask.tolist(),
            "harmful_mask": self.harmful_mask.tolist(),
            "group_count": self.group_count,
            "accepted_count": self.accepted_count,
            "acceptance_rate": self.acceptance_rate,
            "harmful_accepted_count": self.harmful_accepted_count,
            "observed_harm_rate": self.observed_harm_rate,
            "one_sided_upper_bound": self.one_sided_upper_bound,
            "minimum_zero_harm_accepted_groups": (
                self.minimum_zero_harm_accepted_groups
            ),
            "certified": self.certified,
            "threshold_frozen_before_certification_outcomes": (
                self.threshold_frozen_before_certification_outcomes
            ),
            "certification_outcomes_used_for_threshold_selection": (
                self.certification_outcomes_used_for_threshold_selection
            ),
            "certification_groups_independent": (
                self.certification_groups_independent
            ),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "guard harm-risk certificate",
    ) -> GuardHarmRiskCertificateV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        expected_fields = frozenset(
            {
                "schema",
                "schema_version",
                "bound_method",
                "risk_score_semantics",
                "guard_policy_id",
                "threshold_source_artifact_id",
                "certification_partition_id",
                "statistical_unit",
                "metric",
                "group_ids",
                "risk_scores",
                "candidate_losses",
                "fallback_losses",
                "fallback_identity_verified",
                "threshold",
                "harm_margin",
                "target_harm_probability",
                "confidence_level",
                "minimum_accepted_group_count",
                "accepted_mask",
                "harmful_mask",
                "group_count",
                "accepted_count",
                "acceptance_rate",
                "harmful_accepted_count",
                "observed_harm_rate",
                "one_sided_upper_bound",
                "minimum_zero_harm_accepted_groups",
                "certified",
                "threshold_frozen_before_certification_outcomes",
                "certification_outcomes_used_for_threshold_selection",
                "certification_groups_independent",
                "metadata",
                "artifact_id",
            }
        )
        require_exact_fields(value, expected=expected_fields, name=name)
        if value["schema"] != GUARD_HARM_RISK_CERTIFICATE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != GUARD_HARM_RISK_CERTIFICATE_VERSION:
            raise ValueError(f"{name} version changed")
        if value["bound_method"] != BOUND_METHOD:
            raise ValueError(f"{name} bound method changed")
        if value["risk_score_semantics"] != RISK_SCORE_SEMANTICS:
            raise ValueError(f"{name} risk-score semantics changed")
        certificate = cls(
            guard_policy_id=cast(str, value["guard_policy_id"]),
            threshold_source_artifact_id=cast(
                str,
                value["threshold_source_artifact_id"],
            ),
            certification_partition_id=cast(
                str,
                value["certification_partition_id"],
            ),
            statistical_unit=cast(str, value["statistical_unit"]),
            metric=cast(str, value["metric"]),
            group_ids=cast(Sequence[str], value["group_ids"]),
            risk_scores=cast(FloatArray, value["risk_scores"]),
            candidate_losses=cast(FloatArray, value["candidate_losses"]),
            fallback_losses=cast(FloatArray, value["fallback_losses"]),
            fallback_identity_verified=cast(
                BoolArray,
                value["fallback_identity_verified"],
            ),
            threshold=cast(float, value["threshold"]),
            harm_margin=cast(float, value["harm_margin"]),
            target_harm_probability=cast(
                float,
                value["target_harm_probability"],
            ),
            confidence_level=cast(float, value["confidence_level"]),
            minimum_accepted_group_count=cast(
                int,
                value["minimum_accepted_group_count"],
            ),
            accepted_mask=cast(BoolArray, value["accepted_mask"]),
            harmful_mask=cast(BoolArray, value["harmful_mask"]),
            accepted_count=cast(int, value["accepted_count"]),
            harmful_accepted_count=cast(
                int,
                value["harmful_accepted_count"],
            ),
            one_sided_upper_bound=cast(
                float,
                value["one_sided_upper_bound"],
            ),
            certified=cast(bool, value["certified"]),
            threshold_frozen_before_certification_outcomes=cast(
                bool,
                value[
                    "threshold_frozen_before_certification_outcomes"
                ],
            ),
            certification_outcomes_used_for_threshold_selection=cast(
                bool,
                value[
                    "certification_outcomes_used_for_threshold_selection"
                ],
            ),
            certification_groups_independent=cast(
                bool,
                value["certification_groups_independent"],
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
        if value["group_count"] != certificate.group_count:
            raise ValueError(f"{name} group_count changed")
        if value["acceptance_rate"] != certificate.acceptance_rate:
            raise ValueError(f"{name} acceptance_rate changed")
        if value["observed_harm_rate"] != certificate.observed_harm_rate:
            raise ValueError(f"{name} observed_harm_rate changed")
        if (
            value["minimum_zero_harm_accepted_groups"]
            != certificate.minimum_zero_harm_accepted_groups
        ):
            raise ValueError(
                f"{name} minimum zero-harm support changed"
            )
        return certificate


def certify_guard_harm_risk(
    *,
    guard_policy_id: str,
    threshold_source_artifact_id: str,
    certification_partition_id: str,
    statistical_unit: str,
    metric: str,
    group_ids: Sequence[str],
    risk_scores: object,
    candidate_losses: object,
    fallback_losses: object,
    fallback_identity_verified: object,
    threshold: float,
    harm_margin: float,
    target_harm_probability: float,
    confidence_level: float,
    minimum_accepted_group_count: int,
    threshold_frozen_before_certification_outcomes: bool,
    certification_outcomes_used_for_threshold_selection: bool,
    certification_groups_independent: bool,
    metadata: Mapping[str, Any] | None = None,
) -> GuardHarmRiskCertificateV1:
    """Build a certificate from one independent certification partition."""

    scores = _float_vector(risk_scores, name="risk_scores")
    candidate = _float_vector(candidate_losses, name="candidate_losses")
    fallback = _float_vector(fallback_losses, name="fallback_losses")
    fallback_verified = _boolean_vector(
        fallback_identity_verified,
        name="fallback_identity_verified",
    )
    threshold_value = _finite_real(threshold, name="threshold")
    margin = _finite_real(harm_margin, name="harm_margin", minimum=0.0)
    accepted = scores <= threshold_value
    harmful = candidate > fallback + margin
    accepted_count = int(np.sum(accepted))
    harmful_count = int(np.sum(accepted & harmful))
    upper = one_sided_binomial_upper_bound(
        harmful_count,
        accepted_count,
        confidence_level,
    )
    minimum_accepted = genuine_integer(
        minimum_accepted_group_count,
        name="minimum_accepted_group_count",
        minimum=1,
    )
    target = _open_probability(
        target_harm_probability,
        name="target_harm_probability",
    )
    return GuardHarmRiskCertificateV1(
        guard_policy_id=guard_policy_id,
        threshold_source_artifact_id=threshold_source_artifact_id,
        certification_partition_id=certification_partition_id,
        statistical_unit=statistical_unit,
        metric=metric,
        group_ids=group_ids,
        risk_scores=scores,
        candidate_losses=candidate,
        fallback_losses=fallback,
        fallback_identity_verified=fallback_verified,
        threshold=threshold_value,
        harm_margin=margin,
        target_harm_probability=target,
        confidence_level=confidence_level,
        minimum_accepted_group_count=minimum_accepted,
        accepted_mask=accepted,
        harmful_mask=harmful,
        accepted_count=accepted_count,
        harmful_accepted_count=harmful_count,
        one_sided_upper_bound=upper,
        certified=(
            accepted_count >= minimum_accepted and upper <= target
        ),
        threshold_frozen_before_certification_outcomes=(
            threshold_frozen_before_certification_outcomes
        ),
        certification_outcomes_used_for_threshold_selection=(
            certification_outcomes_used_for_threshold_selection
        ),
        certification_groups_independent=(
            certification_groups_independent
        ),
        metadata={} if metadata is None else metadata,
    )


def save_guard_harm_risk_certificate(
    certificate: GuardHarmRiskCertificateV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(certificate, GuardHarmRiskCertificateV1):
        raise TypeError("certificate must be a GuardHarmRiskCertificateV1")
    write_atomic_json(certificate.to_record(), path, overwrite=overwrite)


def load_guard_harm_risk_certificate(
    path: str | Path,
) -> GuardHarmRiskCertificateV1:
    payload = load_strict_json_object(
        path,
        label="guard harm-risk certificate",
    )
    return GuardHarmRiskCertificateV1.from_mapping(payload)


__all__ = [
    "BOUND_METHOD",
    "GUARD_HARM_RISK_CERTIFICATE_SCHEMA",
    "GUARD_HARM_RISK_CERTIFICATE_VERSION",
    "RISK_SCORE_SEMANTICS",
    "GuardHarmRiskCertificateV1",
    "certify_guard_harm_risk",
    "load_guard_harm_risk_certificate",
    "minimum_zero_harm_groups_for_certificate",
    "one_sided_binomial_upper_bound",
    "save_guard_harm_risk_certificate",
]
