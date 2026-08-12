"""Exact paired inference over independent physical object or session groups.

Every row is one already frozen, equal-weight independent group and every
column is one preregistered paired candidate-minus-comparator estimand. Negative
effects favor the candidate. Exact joint sign-flip enumeration and a shared-index
paired bootstrap are replayed before an artifact identity is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, literal_lower_hex
from ._independent_group_inference_common import (
    BOOTSTRAP_RNG,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    EFFECT_DIRECTION,
    GROUP_WEIGHTING,
    INDEPENDENT_GROUP_INFERENCE_SCHEMA,
    INDEPENDENT_GROUP_INFERENCE_VERSION,
    MAXIMUM_BOOTSTRAP_DRAWS,
    MAXIMUM_BOOTSTRAP_REPLICATES,
    MAXIMUM_BOOTSTRAP_RESULT_VALUES,
    MAXIMUM_ESTIMANDS,
    MAXIMUM_EXACT_GROUPS,
    RESAMPLING_UNIT,
    SIGN_FLIP_ASSUMPTION,
    FloatArray,
    IntArray,
    canonical_axis_order,
    canonical_identifiers,
    canonical_string,
    finite_effect_matrix,
    finite_real,
    immutable_float,
    immutable_int,
    strict_integer,
)
from ._independent_group_inference_payload import (
    inference_descriptor,
    validated_constructor_payload,
)
from ._independent_group_inference_resampling import (
    bootstrap_inference,
    exact_sign_flip_inference,
)
from ._portable_contracts import (
    canonical_json_bytes,
    content_id,
    load_strict_json_object,
    write_atomic_json,
)


@dataclass(frozen=True, slots=True)
class IndependentGroupInferenceV1:
    """Content-addressed exact analysis of paired independent-group effects.

    ``group_effects[g, e]`` is the preregistered candidate-minus-comparator
    effect for independent group ``g`` and estimand ``e``. Negative values favor
    the candidate. The class sorts groups by exact identity, replays exact
    sign-flip enumeration, and uses one shared paired bootstrap index stream for
    every estimand in the family.
    """

    protocol_id: str
    family_id: str
    statistical_unit: str
    within_group_aggregation: str
    group_ids: Sequence[str]
    estimand_ids: Sequence[str]
    group_effects: object
    confidence: float = DEFAULT_CONFIDENCE
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    metadata: Mapping[str, Any] | None = None
    artifact_id: str | None = None

    observed_mean: FloatArray = field(init=False, repr=False)
    standard_error: FloatArray = field(init=False, repr=False)
    root_mean_square_scale: FloatArray = field(init=False, repr=False)
    standardized_mean: FloatArray = field(init=False, repr=False)
    exact_unadjusted_p_value: FloatArray = field(init=False, repr=False)
    exact_familywise_p_value: FloatArray = field(init=False, repr=False)
    exact_global_family_p_value: float = field(init=False)
    pointwise_interval_lower: FloatArray = field(init=False, repr=False)
    pointwise_interval_upper: FloatArray = field(init=False, repr=False)
    simultaneous_interval_lower: FloatArray = field(init=False, repr=False)
    simultaneous_interval_upper: FloatArray = field(init=False, repr=False)
    simultaneous_superiority_upper: FloatArray = field(init=False, repr=False)
    simultaneous_two_sided_critical_value: float = field(init=False)
    simultaneous_one_sided_critical_value: float = field(init=False)
    win_count: IntArray = field(init=False, repr=False)
    tie_count: IntArray = field(init=False, repr=False)
    harm_count: IntArray = field(init=False, repr=False)
    best_group_effect: FloatArray = field(init=False, repr=False)
    worst_group_effect: FloatArray = field(init=False, repr=False)
    median_group_effect: FloatArray = field(init=False, repr=False)
    sign_pattern_count: int = field(init=False)
    bootstrap_index_sha256: str = field(init=False)
    bootstrap_mean_sha256: str = field(init=False)
    sign_pattern_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        protocol_id = canonical_string(self.protocol_id, name="protocol_id")
        family_id = canonical_string(self.family_id, name="family_id")
        statistical_unit = canonical_string(
            self.statistical_unit,
            name="statistical_unit",
        )
        aggregation = canonical_string(
            self.within_group_aggregation,
            name="within_group_aggregation",
        )
        group_ids = canonical_identifiers(
            self.group_ids,
            name="group_ids",
            minimum_count=2,
            maximum_count=MAXIMUM_EXACT_GROUPS,
        )
        estimand_ids = canonical_identifiers(
            self.estimand_ids,
            name="estimand_ids",
            minimum_count=1,
            maximum_count=MAXIMUM_ESTIMANDS,
        )
        effects = finite_effect_matrix(
            self.group_effects,
            group_count=len(group_ids),
            estimand_count=len(estimand_ids),
        )
        group_ids, effects = canonical_axis_order(group_ids, effects, axis=0)
        estimand_ids, effects = canonical_axis_order(estimand_ids, effects, axis=1)
        confidence = finite_real(
            self.confidence,
            name="confidence",
            minimum=0.0,
        )
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly inside (0, 1)")
        replicates = strict_integer(
            self.bootstrap_replicates,
            name="bootstrap_replicates",
            minimum=1,
            maximum=MAXIMUM_BOOTSTRAP_REPLICATES,
        )
        seed = strict_integer(
            self.bootstrap_seed,
            name="bootstrap_seed",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        if replicates * len(group_ids) > MAXIMUM_BOOTSTRAP_DRAWS:
            raise ValueError("bootstrap_replicates exceed the group-draw budget")
        if replicates * len(estimand_ids) > MAXIMUM_BOOTSTRAP_RESULT_VALUES:
            raise ValueError("bootstrap_replicates exceed the result-value budget")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="independent-group inference metadata",
        )

        observed_mean = np.mean(effects, axis=0)
        standard_error = np.std(effects, axis=0, ddof=1) / np.sqrt(len(group_ids))
        rms_scale = np.sqrt(np.mean(np.square(effects), axis=0))
        standardized_mean = np.zeros_like(observed_mean)
        positive_scale = rms_scale > 0.0
        standardized_mean[positive_scale] = (
            observed_mean[positive_scale] / rms_scale[positive_scale]
        )
        (
            unadjusted_p,
            familywise_p,
            global_p,
            pattern_count,
            pattern_digest,
        ) = exact_sign_flip_inference(effects, standardized_mean)
        (
            point_lower,
            point_upper,
            simultaneous_lower,
            simultaneous_upper,
            superiority_upper,
            two_sided_critical,
            one_sided_critical,
            index_digest,
            mean_digest,
        ) = bootstrap_inference(
            effects,
            observed_mean,
            standard_error,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
        )

        immutable_effects = immutable_float(effects)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "statistical_unit", statistical_unit)
        object.__setattr__(self, "within_group_aggregation", aggregation)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "estimand_ids", estimand_ids)
        object.__setattr__(self, "group_effects", immutable_effects)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bootstrap_replicates", replicates)
        object.__setattr__(self, "bootstrap_seed", seed)
        object.__setattr__(self, "metadata", metadata)
        for name, value in (
            ("observed_mean", observed_mean),
            ("standard_error", standard_error),
            ("root_mean_square_scale", rms_scale),
            ("standardized_mean", standardized_mean),
            ("exact_unadjusted_p_value", unadjusted_p),
            ("exact_familywise_p_value", familywise_p),
            ("pointwise_interval_lower", point_lower),
            ("pointwise_interval_upper", point_upper),
            ("simultaneous_interval_lower", simultaneous_lower),
            ("simultaneous_interval_upper", simultaneous_upper),
            ("simultaneous_superiority_upper", superiority_upper),
            ("best_group_effect", np.min(effects, axis=0)),
            ("worst_group_effect", np.max(effects, axis=0)),
            ("median_group_effect", np.median(effects, axis=0)),
        ):
            object.__setattr__(self, name, immutable_float(value))
        for name, value in (
            ("win_count", np.sum(effects < 0.0, axis=0, dtype=np.int64)),
            ("tie_count", np.sum(effects == 0.0, axis=0, dtype=np.int64)),
            ("harm_count", np.sum(effects > 0.0, axis=0, dtype=np.int64)),
        ):
            object.__setattr__(self, name, immutable_int(value))
        object.__setattr__(self, "exact_global_family_p_value", float(global_p))
        object.__setattr__(
            self,
            "simultaneous_two_sided_critical_value",
            float(two_sided_critical),
        )
        object.__setattr__(
            self,
            "simultaneous_one_sided_critical_value",
            float(one_sided_critical),
        )
        object.__setattr__(self, "sign_pattern_count", pattern_count)
        object.__setattr__(self, "bootstrap_index_sha256", index_digest)
        object.__setattr__(self, "bootstrap_mean_sha256", mean_digest)
        object.__setattr__(self, "sign_pattern_sha256", pattern_digest)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = literal_lower_hex(
                supplied_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match inference content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def estimand_count(self) -> int:
        return len(self.estimand_ids)

    def descriptor(self) -> dict[str, Any]:
        return inference_descriptor(self)

    def to_payload(self) -> dict[str, Any]:
        payload = self.descriptor()
        artifact_id = self.artifact_id
        assert artifact_id is not None
        payload["artifact_id"] = artifact_id
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> IndependentGroupInferenceV1:
        result = cls(**validated_constructor_payload(payload))
        if canonical_json_bytes(result.to_payload()) != canonical_json_bytes(payload):
            raise ValueError("independent-group inference payload does not replay")
        return result


def analyze_independent_group_inference_v1(
    *,
    protocol_id: str,
    family_id: str,
    statistical_unit: str,
    within_group_aggregation: str,
    group_ids: Sequence[str],
    estimand_ids: Sequence[str],
    group_effects: object,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    metadata: Mapping[str, Any] | None = None,
) -> IndependentGroupInferenceV1:
    """Run and content-address exact paired independent-group inference."""

    return IndependentGroupInferenceV1(
        protocol_id=protocol_id,
        family_id=family_id,
        statistical_unit=statistical_unit,
        within_group_aggregation=within_group_aggregation,
        group_ids=group_ids,
        estimand_ids=estimand_ids,
        group_effects=group_effects,
        confidence=confidence,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        metadata=metadata,
    )


def save_independent_group_inference_v1(
    result: IndependentGroupInferenceV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one canonical JSON inference record."""

    if not isinstance(result, IndependentGroupInferenceV1):
        raise TypeError("result must be an IndependentGroupInferenceV1")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a literal Boolean")
    write_atomic_json(result.to_payload(), path, overwrite=overwrite)


def load_independent_group_inference_v1(
    path: str | Path,
) -> IndependentGroupInferenceV1:
    """Load and fully replay one independent-group inference record."""

    payload = load_strict_json_object(
        path,
        label="independent-group inference record",
    )
    return IndependentGroupInferenceV1.from_payload(payload)


__all__ = [
    "BOOTSTRAP_RNG",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "EFFECT_DIRECTION",
    "GROUP_WEIGHTING",
    "INDEPENDENT_GROUP_INFERENCE_SCHEMA",
    "INDEPENDENT_GROUP_INFERENCE_VERSION",
    "IndependentGroupInferenceV1",
    "RESAMPLING_UNIT",
    "SIGN_FLIP_ASSUMPTION",
    "analyze_independent_group_inference_v1",
    "load_independent_group_inference_v1",
    "save_independent_group_inference_v1",
]
