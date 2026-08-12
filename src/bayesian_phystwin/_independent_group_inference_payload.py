"""Canonical payload helpers for independent-group inference v1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._canonical_contracts import plain_json
from ._independent_group_inference_common import (
    BOOTSTRAP_CHUNK_SIZE,
    BOOTSTRAP_INDEX_DIGEST_DTYPE,
    BOOTSTRAP_MEAN_DIGEST_DTYPE,
    BOOTSTRAP_RNG,
    EFFECT_DIRECTION,
    GROUP_WEIGHTING,
    INDEPENDENT_GROUP_INFERENCE_SCHEMA,
    INDEPENDENT_GROUP_INFERENCE_VERSION,
    MAXIMUM_EXACT_GROUPS,
    PAYLOAD_FIELDS,
    POINTWISE_INTERVAL_METHOD,
    RESAMPLING_UNIT,
    SIGN_FLIP_ASSUMPTION,
    SIGN_FLIP_COMPARISON_EPSILON_MULTIPLIER,
    SIGN_FLIP_STATISTIC,
    SIGN_PATTERN_DIGEST_DTYPE,
    SIMULTANEOUS_INTERVAL_METHOD,
    array_payload,
)
from ._portable_contracts import require_exact_fields

if TYPE_CHECKING:
    from .independent_group_inference_v1 import IndependentGroupInferenceV1


def inference_descriptor(
    record: IndependentGroupInferenceV1,
) -> dict[str, Any]:
    """Return the canonical descriptor used for content addressing."""

    return {
        "schema": INDEPENDENT_GROUP_INFERENCE_SCHEMA,
        "schema_version": INDEPENDENT_GROUP_INFERENCE_VERSION,
        "protocol_id": record.protocol_id,
        "family_id": record.family_id,
        "statistical_unit": record.statistical_unit,
        "within_group_aggregation": record.within_group_aggregation,
        "effect_direction": EFFECT_DIRECTION,
        "resampling_unit": RESAMPLING_UNIT,
        "group_weighting": GROUP_WEIGHTING,
        "sign_flip_assumption": SIGN_FLIP_ASSUMPTION,
        "sign_flip_statistic": SIGN_FLIP_STATISTIC,
        "sign_flip_comparison_epsilon_multiplier": (
            SIGN_FLIP_COMPARISON_EPSILON_MULTIPLIER
        ),
        "pointwise_interval_method": POINTWISE_INTERVAL_METHOD,
        "simultaneous_interval_method": SIMULTANEOUS_INTERVAL_METHOD,
        "bootstrap_rng": BOOTSTRAP_RNG,
        "bootstrap_index_digest_dtype": BOOTSTRAP_INDEX_DIGEST_DTYPE,
        "bootstrap_mean_digest_dtype": BOOTSTRAP_MEAN_DIGEST_DTYPE,
        "sign_pattern_digest_dtype": SIGN_PATTERN_DIGEST_DTYPE,
        "bootstrap_chunk_size": BOOTSTRAP_CHUNK_SIZE,
        "group_ids": list(record.group_ids),
        "estimand_ids": list(record.estimand_ids),
        "group_effects": array_payload(cast(np.ndarray, record.group_effects)),
        "confidence": record.confidence,
        "bootstrap_replicates": record.bootstrap_replicates,
        "bootstrap_seed": record.bootstrap_seed,
        "maximum_exact_groups": MAXIMUM_EXACT_GROUPS,
        "sign_pattern_count": record.sign_pattern_count,
        "observed_mean": array_payload(record.observed_mean),
        "standard_error": array_payload(record.standard_error),
        "root_mean_square_scale": array_payload(record.root_mean_square_scale),
        "standardized_mean": array_payload(record.standardized_mean),
        "exact_unadjusted_p_value": array_payload(record.exact_unadjusted_p_value),
        "exact_familywise_p_value": array_payload(record.exact_familywise_p_value),
        "exact_global_family_p_value": record.exact_global_family_p_value,
        "pointwise_interval_lower": array_payload(record.pointwise_interval_lower),
        "pointwise_interval_upper": array_payload(record.pointwise_interval_upper),
        "simultaneous_interval_lower": array_payload(
            record.simultaneous_interval_lower
        ),
        "simultaneous_interval_upper": array_payload(
            record.simultaneous_interval_upper
        ),
        "simultaneous_superiority_upper": array_payload(
            record.simultaneous_superiority_upper
        ),
        "simultaneous_two_sided_critical_value": (
            record.simultaneous_two_sided_critical_value
        ),
        "simultaneous_one_sided_critical_value": (
            record.simultaneous_one_sided_critical_value
        ),
        "win_count": array_payload(record.win_count),
        "tie_count": array_payload(record.tie_count),
        "harm_count": array_payload(record.harm_count),
        "best_group_effect": array_payload(record.best_group_effect),
        "worst_group_effect": array_payload(record.worst_group_effect),
        "median_group_effect": array_payload(record.median_group_effect),
        "bootstrap_index_sha256": record.bootstrap_index_sha256,
        "bootstrap_mean_sha256": record.bootstrap_mean_sha256,
        "sign_pattern_sha256": record.sign_pattern_sha256,
        "metadata": plain_json(record.metadata),
    }


def validated_constructor_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate payload structure and return only constructor-owned fields."""

    require_exact_fields(
        payload,
        expected=PAYLOAD_FIELDS,
        name="independent-group inference payload",
    )
    schema = payload.get("schema")
    if type(schema) is not str or schema != INDEPENDENT_GROUP_INFERENCE_SCHEMA:
        raise ValueError("independent-group inference schema changed")
    schema_version = payload.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != INDEPENDENT_GROUP_INFERENCE_VERSION
    ):
        raise ValueError("independent-group inference schema version changed")
    expected_constants = {
        "effect_direction": EFFECT_DIRECTION,
        "resampling_unit": RESAMPLING_UNIT,
        "group_weighting": GROUP_WEIGHTING,
        "sign_flip_assumption": SIGN_FLIP_ASSUMPTION,
        "sign_flip_statistic": SIGN_FLIP_STATISTIC,
        "sign_flip_comparison_epsilon_multiplier": (
            SIGN_FLIP_COMPARISON_EPSILON_MULTIPLIER
        ),
        "pointwise_interval_method": POINTWISE_INTERVAL_METHOD,
        "simultaneous_interval_method": SIMULTANEOUS_INTERVAL_METHOD,
        "bootstrap_rng": BOOTSTRAP_RNG,
        "bootstrap_index_digest_dtype": BOOTSTRAP_INDEX_DIGEST_DTYPE,
        "bootstrap_mean_digest_dtype": BOOTSTRAP_MEAN_DIGEST_DTYPE,
        "sign_pattern_digest_dtype": SIGN_PATTERN_DIGEST_DTYPE,
        "bootstrap_chunk_size": BOOTSTRAP_CHUNK_SIZE,
        "maximum_exact_groups": MAXIMUM_EXACT_GROUPS,
    }
    for name, expected in expected_constants.items():
        supplied = payload.get(name)
        if type(supplied) is not type(expected) or supplied != expected:
            raise ValueError(f"{name} changed")
    return {
        "protocol_id": payload.get("protocol_id"),
        "family_id": payload.get("family_id"),
        "statistical_unit": payload.get("statistical_unit"),
        "within_group_aggregation": payload.get("within_group_aggregation"),
        "group_ids": payload.get("group_ids"),
        "estimand_ids": payload.get("estimand_ids"),
        "group_effects": payload.get("group_effects"),
        "confidence": payload.get("confidence"),
        "bootstrap_replicates": payload.get("bootstrap_replicates"),
        "bootstrap_seed": payload.get("bootstrap_seed"),
        "metadata": payload.get("metadata"),
        "artifact_id": payload.get("artifact_id"),
    }


__all__ = ["inference_descriptor", "validated_constructor_payload"]
