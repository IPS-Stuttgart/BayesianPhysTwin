"""Finite-group value certification for exact-mean covariance candidates.

This module separates uncertainty admission from point-update admission. It
certifies one already frozen covariance-only policy on independent physical
objects or acquisition sessions. Candidate and reference point predictions
must have byte-identical per-group digests; the certificate then evaluates
bounded proper-score regret, full interval width, and harmful-group frequency.
It does not select a policy or threshold from the certification outcomes.
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
from .guard_harm_risk import one_sided_binomial_upper_bound

COVARIANCE_ONLY_VALUE_CERTIFICATE_SCHEMA = (
    "bayesian_phystwin.covariance_only_value_certificate"
)
COVARIANCE_ONLY_VALUE_CERTIFICATE_VERSION = 1
SCORE_DIFFERENCE_SEMANTICS = "candidate-minus-reference-lower-is-better-v1"
WIDTH_SEMANTICS = "group-mean-full-interval-width-v1"
MEAN_IDENTITY_SEMANTICS = "per-group-byte-digest-equality-v1"
MEAN_BOUND_METHOD = "one-sided-hoeffding-bounded-mean-v1"
FAMILYWISE_METHOD = "bonferroni-three-one-sided-gates-v1"

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
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real vector") from error
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector")
    try:
        result: FloatArray = np.array(
            raw,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real vector") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector")
    return result


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


def _selection_group_id_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("selection_group_ids must be a sequence of strings")
    result = tuple(
        _canonical_string(item, name=f"selection_group_ids[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(set(result)) != len(result):
        raise ValueError("selection_group_ids must not contain duplicates")
    return tuple(sorted(result))


def _digest_tuple(
    value: object,
    *,
    name: str,
    expected_count: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of SHA-256 digests")
    result = tuple(
        sha256_digest(item, name=f"{name}[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(result) != expected_count:
        raise ValueError(f"{name} length must match certification arrays")
    return result


def _exact_json_value_equal(supplied: object, expected: object) -> bool:
    if type(supplied) is not type(expected):
        return False
    if isinstance(expected, list):
        supplied_list = cast(list[object], supplied)
        return len(supplied_list) == len(expected) and all(
            _exact_json_value_equal(left, right)
            for left, right in zip(supplied_list, expected, strict=True)
        )
    return supplied == expected


def _bounded_interval(
    lower_bound: object,
    upper_bound: object,
    *,
    name: str,
) -> tuple[float, float]:
    lower = _finite_real(lower_bound, name=f"{name}_lower_bound")
    upper = _finite_real(upper_bound, name=f"{name}_upper_bound")
    if not lower < upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound")
    return lower, upper


def bonferroni_gate_confidence_level(
    familywise_confidence_level: float,
    *,
    gate_count: int = 3,
) -> float:
    """Return the per-gate level for a Bonferroni familywise certificate."""

    familywise = _open_probability(
        familywise_confidence_level,
        name="familywise_confidence_level",
    )
    gates = genuine_integer(gate_count, name="gate_count", minimum=1)
    return 1.0 - (1.0 - familywise) / gates


def hoeffding_mean_upper_bound(
    values: object,
    *,
    lower_bound: float,
    upper_bound: float,
    confidence_level: float,
) -> float:
    """Return a one-sided Hoeffding bound for an independent bounded mean."""

    vector = _float_vector(values, name="values")
    if len(vector) == 0:
        raise ValueError("values must contain at least one independent group")
    lower, upper = _bounded_interval(
        lower_bound,
        upper_bound,
        name="values",
    )
    if np.any(vector < lower) or np.any(vector > upper):
        raise ValueError("values must lie inside the frozen bounded interval")
    confidence = _open_probability(confidence_level, name="confidence_level")
    delta = 1.0 - confidence
    radius = (upper - lower) * math.sqrt(math.log(1.0 / delta) / (2.0 * len(vector)))
    return min(upper, float(np.mean(vector)) + radius)


@dataclass(frozen=True, slots=True)
class CovarianceOnlyValueCertificateV1:
    """Finite-group score, width, and harm certificate for fixed point means."""

    candidate_policy_id: str
    reference_policy_id: str
    query_set_id: str
    policy_freeze_artifact_id: str
    certification_partition_id: str
    statistical_unit: str
    score_metric: str
    width_metric: str
    selection_group_ids: Sequence[str]
    group_ids: Sequence[str]
    candidate_mean_sha256: Sequence[str]
    reference_mean_sha256: Sequence[str]
    candidate_scores: FloatArray
    reference_scores: FloatArray
    candidate_full_widths: FloatArray
    reference_full_widths: FloatArray
    score_difference_lower_bound: float
    score_difference_upper_bound: float
    full_width_upper_bound: float
    maximum_expected_score_regret: float
    maximum_expected_full_width: float
    harm_margin: float
    target_harm_probability: float
    familywise_confidence_level: float
    minimum_group_count: int
    thresholds_frozen_before_certification_outcomes: bool
    certification_outcomes_used_for_policy_selection: bool
    certification_groups_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    mean_identity_mask: BoolArray = field(init=False, repr=False)
    score_differences: FloatArray = field(init=False, repr=False)
    harmful_mask: BoolArray = field(init=False, repr=False)
    mean_score_difference: float = field(init=False)
    score_upper_confidence_bound: float = field(init=False)
    mean_candidate_full_width: float = field(init=False)
    width_upper_confidence_bound: float = field(init=False)
    harmful_group_count: int = field(init=False)
    harm_probability_upper_bound: float = field(init=False)
    certified: bool = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "candidate_policy_id",
            "reference_policy_id",
            "query_set_id",
            "policy_freeze_artifact_id",
            "certification_partition_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in ("statistical_unit", "score_metric", "width_metric"):
            object.__setattr__(
                self,
                name,
                _canonical_string(getattr(self, name), name=name),
            )

        selection_group_ids = _selection_group_id_tuple(self.selection_group_ids)
        candidate_scores = _float_vector(
            self.candidate_scores,
            name="candidate_scores",
        )
        reference_scores = _float_vector(
            self.reference_scores,
            name="reference_scores",
        )
        candidate_widths = _float_vector(
            self.candidate_full_widths,
            name="candidate_full_widths",
        )
        reference_widths = _float_vector(
            self.reference_full_widths,
            name="reference_full_widths",
        )
        count = len(candidate_scores)
        if count == 0:
            raise ValueError("at least one independent certification group is required")
        if any(
            len(value) != count
            for value in (reference_scores, candidate_widths, reference_widths)
        ):
            raise ValueError("certification arrays must have equal length")
        if np.any(candidate_widths < 0.0) or np.any(reference_widths < 0.0):
            raise ValueError("full interval widths must be nonnegative")

        group_ids = _group_id_tuple(self.group_ids, expected_count=count)
        candidate_digests = _digest_tuple(
            self.candidate_mean_sha256,
            name="candidate_mean_sha256",
            expected_count=count,
        )
        reference_digests = _digest_tuple(
            self.reference_mean_sha256,
            name="reference_mean_sha256",
            expected_count=count,
        )
        order = np.argsort(np.asarray(group_ids, dtype=object), kind="mergesort")
        group_ids = tuple(group_ids[int(index)] for index in order)
        overlap = sorted(set(group_ids) & set(selection_group_ids))
        if overlap:
            raise ValueError(f"selection and certification groups overlap: {overlap}")
        candidate_digests = tuple(candidate_digests[int(index)] for index in order)
        reference_digests = tuple(reference_digests[int(index)] for index in order)
        candidate_scores = candidate_scores[order]
        reference_scores = reference_scores[order]
        candidate_widths = candidate_widths[order]
        reference_widths = reference_widths[order]

        score_lower, score_upper = _bounded_interval(
            self.score_difference_lower_bound,
            self.score_difference_upper_bound,
            name="score_difference",
        )
        width_upper = _finite_real(
            self.full_width_upper_bound,
            name="full_width_upper_bound",
            minimum=0.0,
        )
        if width_upper <= 0.0:
            raise ValueError("full_width_upper_bound must be positive")
        maximum_score_regret = _finite_real(
            self.maximum_expected_score_regret,
            name="maximum_expected_score_regret",
        )
        if not score_lower <= maximum_score_regret <= score_upper:
            raise ValueError(
                "maximum_expected_score_regret must lie inside score bounds"
            )
        maximum_width = _finite_real(
            self.maximum_expected_full_width,
            name="maximum_expected_full_width",
            minimum=0.0,
        )
        if maximum_width > width_upper:
            raise ValueError(
                "maximum_expected_full_width cannot exceed its frozen bound"
            )
        margin = _finite_real(self.harm_margin, name="harm_margin", minimum=0.0)
        target_harm = _open_probability(
            self.target_harm_probability,
            name="target_harm_probability",
        )
        familywise_confidence = _open_probability(
            self.familywise_confidence_level,
            name="familywise_confidence_level",
        )
        minimum_groups = genuine_integer(
            self.minimum_group_count,
            name="minimum_group_count",
            minimum=1,
        )
        thresholds_frozen = genuine_boolean(
            self.thresholds_frozen_before_certification_outcomes,
            name="thresholds_frozen_before_certification_outcomes",
        )
        outcomes_used = genuine_boolean(
            self.certification_outcomes_used_for_policy_selection,
            name="certification_outcomes_used_for_policy_selection",
        )
        groups_independent = genuine_boolean(
            self.certification_groups_independent,
            name="certification_groups_independent",
        )
        if not thresholds_frozen:
            raise ValueError(
                "thresholds must be frozen before certification outcomes are opened"
            )
        if outcomes_used:
            raise ValueError(
                "certification outcomes cannot select the certified policy"
            )
        if not groups_independent:
            raise ValueError("certification groups must be independent physical units")

        score_differences = candidate_scores - reference_scores
        if np.any(score_differences < score_lower) or np.any(
            score_differences > score_upper
        ):
            raise ValueError(
                "score differences must lie inside the frozen bounded interval"
            )
        if np.any(candidate_widths > width_upper) or np.any(
            reference_widths > width_upper
        ):
            raise ValueError(
                "full interval widths must lie inside the frozen bounded interval"
            )
        mean_identity = np.asarray(
            [
                candidate == reference
                for candidate, reference in zip(
                    candidate_digests,
                    reference_digests,
                    strict=True,
                )
            ],
            dtype=np.bool_,
        )
        harmful = score_differences > margin
        per_gate_confidence = bonferroni_gate_confidence_level(familywise_confidence)
        score_upper_confidence = hoeffding_mean_upper_bound(
            score_differences,
            lower_bound=score_lower,
            upper_bound=score_upper,
            confidence_level=per_gate_confidence,
        )
        width_upper_confidence = hoeffding_mean_upper_bound(
            candidate_widths,
            lower_bound=0.0,
            upper_bound=width_upper,
            confidence_level=per_gate_confidence,
        )
        harmful_count = int(np.sum(harmful))
        harm_upper = one_sided_binomial_upper_bound(
            harmful_count,
            count,
            per_gate_confidence,
        )
        certified = (
            bool(np.all(mean_identity))
            and count >= minimum_groups
            and score_upper_confidence <= maximum_score_regret
            and width_upper_confidence <= maximum_width
            and harm_upper <= target_harm
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="covariance-only value certificate metadata",
        )

        object.__setattr__(self, "selection_group_ids", selection_group_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "candidate_mean_sha256", candidate_digests)
        object.__setattr__(self, "reference_mean_sha256", reference_digests)
        object.__setattr__(
            self,
            "candidate_scores",
            _immutable_float(candidate_scores),
        )
        object.__setattr__(
            self,
            "reference_scores",
            _immutable_float(reference_scores),
        )
        object.__setattr__(
            self,
            "candidate_full_widths",
            _immutable_float(candidate_widths),
        )
        object.__setattr__(
            self,
            "reference_full_widths",
            _immutable_float(reference_widths),
        )
        object.__setattr__(self, "score_difference_lower_bound", score_lower)
        object.__setattr__(self, "score_difference_upper_bound", score_upper)
        object.__setattr__(self, "full_width_upper_bound", width_upper)
        object.__setattr__(
            self,
            "maximum_expected_score_regret",
            maximum_score_regret,
        )
        object.__setattr__(self, "maximum_expected_full_width", maximum_width)
        object.__setattr__(self, "harm_margin", margin)
        object.__setattr__(self, "target_harm_probability", target_harm)
        object.__setattr__(
            self,
            "familywise_confidence_level",
            familywise_confidence,
        )
        object.__setattr__(self, "minimum_group_count", minimum_groups)
        object.__setattr__(
            self,
            "thresholds_frozen_before_certification_outcomes",
            thresholds_frozen,
        )
        object.__setattr__(
            self,
            "certification_outcomes_used_for_policy_selection",
            outcomes_used,
        )
        object.__setattr__(
            self,
            "certification_groups_independent",
            groups_independent,
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "mean_identity_mask",
            _immutable_bool(mean_identity),
        )
        object.__setattr__(
            self,
            "score_differences",
            _immutable_float(score_differences),
        )
        object.__setattr__(self, "harmful_mask", _immutable_bool(harmful))
        object.__setattr__(
            self,
            "mean_score_difference",
            float(np.mean(score_differences)),
        )
        object.__setattr__(
            self,
            "score_upper_confidence_bound",
            score_upper_confidence,
        )
        object.__setattr__(
            self,
            "mean_candidate_full_width",
            float(np.mean(candidate_widths)),
        )
        object.__setattr__(
            self,
            "width_upper_confidence_bound",
            width_upper_confidence,
        )
        object.__setattr__(self, "harmful_group_count", harmful_count)
        object.__setattr__(self, "harm_probability_upper_bound", harm_upper)
        object.__setattr__(self, "certified", certified)

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
    def mean_identity_count(self) -> int:
        return int(np.sum(self.mean_identity_mask))

    @property
    def per_gate_confidence_level(self) -> float:
        return bonferroni_gate_confidence_level(self.familywise_confidence_level)

    @property
    def mean_reference_full_width(self) -> float:
        return float(np.mean(self.reference_full_widths))

    @property
    def mean_full_width_difference(self) -> float:
        return self.mean_candidate_full_width - self.mean_reference_full_width

    @property
    def observed_harm_rate(self) -> float:
        return self.harmful_group_count / self.group_count

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": COVARIANCE_ONLY_VALUE_CERTIFICATE_SCHEMA,
            "schema_version": COVARIANCE_ONLY_VALUE_CERTIFICATE_VERSION,
            "score_difference_semantics": SCORE_DIFFERENCE_SEMANTICS,
            "width_semantics": WIDTH_SEMANTICS,
            "mean_identity_semantics": MEAN_IDENTITY_SEMANTICS,
            "mean_bound_method": MEAN_BOUND_METHOD,
            "familywise_method": FAMILYWISE_METHOD,
            "candidate_policy_id": self.candidate_policy_id,
            "reference_policy_id": self.reference_policy_id,
            "query_set_id": self.query_set_id,
            "policy_freeze_artifact_id": self.policy_freeze_artifact_id,
            "certification_partition_id": self.certification_partition_id,
            "statistical_unit": self.statistical_unit,
            "score_metric": self.score_metric,
            "width_metric": self.width_metric,
            "selection_group_ids": list(self.selection_group_ids),
            "group_ids": list(self.group_ids),
            "candidate_mean_sha256": list(self.candidate_mean_sha256),
            "reference_mean_sha256": list(self.reference_mean_sha256),
            "candidate_scores": self.candidate_scores.tolist(),
            "reference_scores": self.reference_scores.tolist(),
            "candidate_full_widths": self.candidate_full_widths.tolist(),
            "reference_full_widths": self.reference_full_widths.tolist(),
            "score_difference_lower_bound": self.score_difference_lower_bound,
            "score_difference_upper_bound": self.score_difference_upper_bound,
            "full_width_upper_bound": self.full_width_upper_bound,
            "maximum_expected_score_regret": self.maximum_expected_score_regret,
            "maximum_expected_full_width": self.maximum_expected_full_width,
            "harm_margin": self.harm_margin,
            "target_harm_probability": self.target_harm_probability,
            "familywise_confidence_level": self.familywise_confidence_level,
            "per_gate_confidence_level": self.per_gate_confidence_level,
            "minimum_group_count": self.minimum_group_count,
            "mean_identity_mask": self.mean_identity_mask.tolist(),
            "mean_identity_count": self.mean_identity_count,
            "score_differences": self.score_differences.tolist(),
            "mean_score_difference": self.mean_score_difference,
            "score_upper_confidence_bound": self.score_upper_confidence_bound,
            "mean_candidate_full_width": self.mean_candidate_full_width,
            "mean_reference_full_width": self.mean_reference_full_width,
            "mean_full_width_difference": self.mean_full_width_difference,
            "width_upper_confidence_bound": self.width_upper_confidence_bound,
            "harmful_mask": self.harmful_mask.tolist(),
            "harmful_group_count": self.harmful_group_count,
            "observed_harm_rate": self.observed_harm_rate,
            "harm_probability_upper_bound": self.harm_probability_upper_bound,
            "group_count": self.group_count,
            "certified": self.certified,
            "thresholds_frozen_before_certification_outcomes": (
                self.thresholds_frozen_before_certification_outcomes
            ),
            "certification_outcomes_used_for_policy_selection": (
                self.certification_outcomes_used_for_policy_selection
            ),
            "certification_groups_independent": (self.certification_groups_independent),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "covariance-only value certificate",
    ) -> CovarianceOnlyValueCertificateV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        expected_fields = frozenset(
            {
                *cls._record_field_names(),
                "artifact_id",
            }
        )
        require_exact_fields(value, expected=expected_fields, name=name)
        if value["schema"] != COVARIANCE_ONLY_VALUE_CERTIFICATE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != COVARIANCE_ONLY_VALUE_CERTIFICATE_VERSION:
            raise ValueError(f"{name} version changed")
        constants = {
            "score_difference_semantics": SCORE_DIFFERENCE_SEMANTICS,
            "width_semantics": WIDTH_SEMANTICS,
            "mean_identity_semantics": MEAN_IDENTITY_SEMANTICS,
            "mean_bound_method": MEAN_BOUND_METHOD,
            "familywise_method": FAMILYWISE_METHOD,
        }
        for field_name, expected in constants.items():
            if value[field_name] != expected:
                raise ValueError(f"{name} {field_name} changed")

        certificate = cls(
            candidate_policy_id=cast(str, value["candidate_policy_id"]),
            reference_policy_id=cast(str, value["reference_policy_id"]),
            query_set_id=cast(str, value["query_set_id"]),
            policy_freeze_artifact_id=cast(
                str,
                value["policy_freeze_artifact_id"],
            ),
            certification_partition_id=cast(
                str,
                value["certification_partition_id"],
            ),
            statistical_unit=cast(str, value["statistical_unit"]),
            score_metric=cast(str, value["score_metric"]),
            width_metric=cast(str, value["width_metric"]),
            selection_group_ids=cast(
                Sequence[str],
                value["selection_group_ids"],
            ),
            group_ids=cast(Sequence[str], value["group_ids"]),
            candidate_mean_sha256=cast(
                Sequence[str],
                value["candidate_mean_sha256"],
            ),
            reference_mean_sha256=cast(
                Sequence[str],
                value["reference_mean_sha256"],
            ),
            candidate_scores=cast(FloatArray, value["candidate_scores"]),
            reference_scores=cast(FloatArray, value["reference_scores"]),
            candidate_full_widths=cast(
                FloatArray,
                value["candidate_full_widths"],
            ),
            reference_full_widths=cast(
                FloatArray,
                value["reference_full_widths"],
            ),
            score_difference_lower_bound=cast(
                float,
                value["score_difference_lower_bound"],
            ),
            score_difference_upper_bound=cast(
                float,
                value["score_difference_upper_bound"],
            ),
            full_width_upper_bound=cast(
                float,
                value["full_width_upper_bound"],
            ),
            maximum_expected_score_regret=cast(
                float,
                value["maximum_expected_score_regret"],
            ),
            maximum_expected_full_width=cast(
                float,
                value["maximum_expected_full_width"],
            ),
            harm_margin=cast(float, value["harm_margin"]),
            target_harm_probability=cast(
                float,
                value["target_harm_probability"],
            ),
            familywise_confidence_level=cast(
                float,
                value["familywise_confidence_level"],
            ),
            minimum_group_count=cast(int, value["minimum_group_count"]),
            thresholds_frozen_before_certification_outcomes=cast(
                bool,
                value["thresholds_frozen_before_certification_outcomes"],
            ),
            certification_outcomes_used_for_policy_selection=cast(
                bool,
                value["certification_outcomes_used_for_policy_selection"],
            ),
            certification_groups_independent=cast(
                bool,
                value["certification_groups_independent"],
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
        expected_record = certificate.descriptor()
        for field_name in cls._derived_field_names():
            if not _exact_json_value_equal(
                value[field_name],
                expected_record[field_name],
            ):
                raise ValueError(f"{name} derived field changed: {field_name}")
        return certificate

    @staticmethod
    def _derived_field_names() -> frozenset[str]:
        return frozenset(
            {
                "per_gate_confidence_level",
                "mean_identity_mask",
                "mean_identity_count",
                "score_differences",
                "mean_score_difference",
                "score_upper_confidence_bound",
                "mean_candidate_full_width",
                "mean_reference_full_width",
                "mean_full_width_difference",
                "width_upper_confidence_bound",
                "harmful_mask",
                "harmful_group_count",
                "observed_harm_rate",
                "harm_probability_upper_bound",
                "group_count",
                "certified",
            }
        )

    @staticmethod
    def _record_field_names() -> frozenset[str]:
        return frozenset(
            {
                "schema",
                "schema_version",
                "score_difference_semantics",
                "width_semantics",
                "mean_identity_semantics",
                "mean_bound_method",
                "familywise_method",
                "candidate_policy_id",
                "reference_policy_id",
                "query_set_id",
                "policy_freeze_artifact_id",
                "certification_partition_id",
                "statistical_unit",
                "score_metric",
                "width_metric",
                "selection_group_ids",
                "group_ids",
                "candidate_mean_sha256",
                "reference_mean_sha256",
                "candidate_scores",
                "reference_scores",
                "candidate_full_widths",
                "reference_full_widths",
                "score_difference_lower_bound",
                "score_difference_upper_bound",
                "full_width_upper_bound",
                "maximum_expected_score_regret",
                "maximum_expected_full_width",
                "harm_margin",
                "target_harm_probability",
                "familywise_confidence_level",
                "minimum_group_count",
                "thresholds_frozen_before_certification_outcomes",
                "certification_outcomes_used_for_policy_selection",
                "certification_groups_independent",
                "metadata",
                *CovarianceOnlyValueCertificateV1._derived_field_names(),
            }
        )


def certify_covariance_only_value(
    *,
    candidate_policy_id: str,
    reference_policy_id: str,
    query_set_id: str,
    policy_freeze_artifact_id: str,
    certification_partition_id: str,
    statistical_unit: str,
    score_metric: str,
    width_metric: str,
    selection_group_ids: Sequence[str],
    group_ids: Sequence[str],
    candidate_mean_sha256: Sequence[str],
    reference_mean_sha256: Sequence[str],
    candidate_scores: object,
    reference_scores: object,
    candidate_full_widths: object,
    reference_full_widths: object,
    score_difference_lower_bound: float,
    score_difference_upper_bound: float,
    full_width_upper_bound: float,
    maximum_expected_score_regret: float,
    maximum_expected_full_width: float,
    harm_margin: float,
    target_harm_probability: float,
    familywise_confidence_level: float,
    minimum_group_count: int,
    thresholds_frozen_before_certification_outcomes: bool,
    certification_outcomes_used_for_policy_selection: bool,
    certification_groups_independent: bool,
    metadata: Mapping[str, Any] | None = None,
) -> CovarianceOnlyValueCertificateV1:
    """Build a certificate for one frozen covariance-only policy."""

    return CovarianceOnlyValueCertificateV1(
        candidate_policy_id=candidate_policy_id,
        reference_policy_id=reference_policy_id,
        query_set_id=query_set_id,
        policy_freeze_artifact_id=policy_freeze_artifact_id,
        certification_partition_id=certification_partition_id,
        statistical_unit=statistical_unit,
        score_metric=score_metric,
        width_metric=width_metric,
        selection_group_ids=selection_group_ids,
        group_ids=group_ids,
        candidate_mean_sha256=candidate_mean_sha256,
        reference_mean_sha256=reference_mean_sha256,
        candidate_scores=candidate_scores,
        reference_scores=reference_scores,
        candidate_full_widths=candidate_full_widths,
        reference_full_widths=reference_full_widths,
        score_difference_lower_bound=score_difference_lower_bound,
        score_difference_upper_bound=score_difference_upper_bound,
        full_width_upper_bound=full_width_upper_bound,
        maximum_expected_score_regret=maximum_expected_score_regret,
        maximum_expected_full_width=maximum_expected_full_width,
        harm_margin=harm_margin,
        target_harm_probability=target_harm_probability,
        familywise_confidence_level=familywise_confidence_level,
        minimum_group_count=minimum_group_count,
        thresholds_frozen_before_certification_outcomes=(
            thresholds_frozen_before_certification_outcomes
        ),
        certification_outcomes_used_for_policy_selection=(
            certification_outcomes_used_for_policy_selection
        ),
        certification_groups_independent=certification_groups_independent,
        metadata={} if metadata is None else metadata,
    )


def save_covariance_only_value_certificate(
    certificate: CovarianceOnlyValueCertificateV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(certificate, CovarianceOnlyValueCertificateV1):
        raise TypeError("certificate must be a CovarianceOnlyValueCertificateV1")
    write_atomic_json(certificate.to_record(), path, overwrite=overwrite)


def load_covariance_only_value_certificate(
    path: str | Path,
) -> CovarianceOnlyValueCertificateV1:
    payload = load_strict_json_object(
        path,
        label="covariance-only value certificate",
    )
    return CovarianceOnlyValueCertificateV1.from_mapping(payload)


__all__ = [
    "COVARIANCE_ONLY_VALUE_CERTIFICATE_SCHEMA",
    "COVARIANCE_ONLY_VALUE_CERTIFICATE_VERSION",
    "FAMILYWISE_METHOD",
    "MEAN_BOUND_METHOD",
    "MEAN_IDENTITY_SEMANTICS",
    "SCORE_DIFFERENCE_SEMANTICS",
    "WIDTH_SEMANTICS",
    "CovarianceOnlyValueCertificateV1",
    "bonferroni_gate_confidence_level",
    "certify_covariance_only_value",
    "hoeffding_mean_upper_bound",
    "load_covariance_only_value_certificate",
    "save_covariance_only_value_certificate",
]
