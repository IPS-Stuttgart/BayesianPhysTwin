"""Frozen target-free sample-admissibility contract for Deform360 Prob4D."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, cast

from ._canonical_contracts import genuine_boolean, genuine_integer, plain_json
from ._portable_contracts import content_id, nonempty_string, require_exact_fields

SAMPLE_ADMISSIBILITY_POLICY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-sample-admissibility-policy"
)
SAMPLE_ADMISSIBILITY_POLICY_VERSION: Final = 1
SAMPLE_ADMISSIBILITY_POLICY_SEMANTICS: Final = (
    "target-free-provider-mask-and-released-metric-window-support-v1"
)
SAMPLE_ADMISSIBLE_PLAN_VERSION: Final = 3
SAMPLE_ADMISSIBLE_PLAN_SEMANTICS: Final = (
    "target-free-sample-admissible-integrity-bound-streams-with-causal-public-"
    "metric-prefix-v3"
)
SAMPLE_SUPPORT_NEGATIVE_REASON: Final = (
    "insufficient-target-free-held-prefix-sample-support"
)

_POLICY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "artifact_id",
        "protocol_id",
        "eligibility_evidence",
        "eligible_status",
        "support_negative_action",
        "technical_failure_action",
        "allowed_support_negative_reason",
        "minimum_metric_gauge_correspondences_per_window",
        "minimum_metric_gauge_spatial_clusters_per_window",
        "maximum_metric_fit_correspondences",
        "minimum_held_prefix_point_rows_per_window",
        "covariance_cluster_size_pixels",
        "minimum_supported_streams_per_object",
        "minimum_supported_object_count",
        "minimum_supported_stream_fraction",
        "prediction_support_masks_used_for_eligibility",
        "prediction_point_values_used_for_eligibility",
        "prediction_residuals_used_for_eligibility",
        "calibration_outcomes_used_for_eligibility",
        "replacement_allowed",
        "confirmation_payloads_opened",
        "future_frames_used",
        "target_outcomes_used",
        "human_approval_required",
        "new_measurements_required",
        "claim_boundary",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    _require(math.isfinite(result) and 0.0 < result <= 1.0, f"{name} is invalid")
    return result


def validate_deform360_prob4d_sample_admissibility_policy(
    value: object,
) -> dict[str, Any]:
    """Validate and normalize the frozen target-free v3 policy."""

    if not isinstance(value, Mapping):
        raise ValueError("sample admissibility policy must be a JSON object")
    require_exact_fields(
        value,
        expected=_POLICY_FIELDS,
        name="sample admissibility policy",
    )
    _require(
        value["schema"] == SAMPLE_ADMISSIBILITY_POLICY_SCHEMA
        and value["schema_version"] == SAMPLE_ADMISSIBILITY_POLICY_VERSION
        and value["semantics"] == SAMPLE_ADMISSIBILITY_POLICY_SEMANTICS,
        "unsupported sample admissibility policy",
    )
    identity = dict(value)
    declared_id = nonempty_string(identity.pop("artifact_id"), name="artifact_id")
    _require(
        content_id(identity) == declared_id,
        "sample admissibility policy ID changed",
    )
    _require(
        value["eligible_status"] == "admissible"
        and value["support_negative_action"] == "retain-and-exclude"
        and value["technical_failure_action"] == "terminal"
        and value["allowed_support_negative_reason"] == SAMPLE_SUPPORT_NEGATIVE_REASON,
        "sample admissibility decision rule changed",
    )
    true_fields = ("prediction_support_masks_used_for_eligibility",)
    false_fields = (
        "prediction_point_values_used_for_eligibility",
        "prediction_residuals_used_for_eligibility",
        "calibration_outcomes_used_for_eligibility",
        "replacement_allowed",
        "confirmation_payloads_opened",
        "future_frames_used",
        "target_outcomes_used",
        "human_approval_required",
        "new_measurements_required",
    )
    for field in true_fields:
        _require(
            genuine_boolean(value[field], name=field) is True,
            f"sample admissibility boundary changed: {field}",
        )
    for field in false_fields:
        _require(
            genuine_boolean(value[field], name=field) is False,
            f"sample admissibility boundary changed: {field}",
        )

    normalized = dict(value)
    normalized["protocol_id"] = nonempty_string(
        value["protocol_id"], name="protocol_id"
    )
    normalized["eligibility_evidence"] = nonempty_string(
        value["eligibility_evidence"], name="eligibility_evidence"
    )
    for field, minimum in (
        ("minimum_metric_gauge_correspondences_per_window", 8),
        ("minimum_metric_gauge_spatial_clusters_per_window", 8),
        ("maximum_metric_fit_correspondences", 8),
        ("minimum_held_prefix_point_rows_per_window", 1),
        ("covariance_cluster_size_pixels", 1),
        ("minimum_supported_streams_per_object", 2),
        ("minimum_supported_object_count", 1),
    ):
        normalized[field] = genuine_integer(value[field], name=field, minimum=minimum)
    normalized["minimum_supported_stream_fraction"] = _finite_fraction(
        value["minimum_supported_stream_fraction"],
        name="minimum_supported_stream_fraction",
    )
    normalized["claim_boundary"] = nonempty_string(
        value["claim_boundary"], name="claim_boundary"
    )
    return cast(dict[str, Any], plain_json(normalized))


__all__ = [
    "SAMPLE_ADMISSIBILITY_POLICY_SCHEMA",
    "SAMPLE_ADMISSIBILITY_POLICY_SEMANTICS",
    "SAMPLE_ADMISSIBILITY_POLICY_VERSION",
    "SAMPLE_ADMISSIBLE_PLAN_SEMANTICS",
    "SAMPLE_ADMISSIBLE_PLAN_VERSION",
    "SAMPLE_SUPPORT_NEGATIVE_REASON",
    "validate_deform360_prob4d_sample_admissibility_policy",
]
