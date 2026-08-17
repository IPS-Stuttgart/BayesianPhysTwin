"""Target-free camera eligibility for public Deform360 Prob4D calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, cast

from ._canonical_contracts import genuine_boolean, genuine_integer, plain_json
from ._portable_contracts import content_id, nonempty_string, require_exact_fields

CAMERA_ELIGIBILITY_POLICY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-camera-eligibility-policy"
)
CAMERA_ELIGIBILITY_POLICY_VERSION: Final = 1
CAMERA_ELIGIBILITY_POLICY_SEMANTICS: Final = (
    "target-free-released-robot-prefix-visibility-v1"
)
VISIBLE_STREAM_PLAN_SEMANTICS: Final = (
    "target-free-robot-visible-integrity-bound-streams-with-causal-public-"
    "metric-prefix-v2"
)
VISIBLE_STREAM_PLAN_VERSION: Final = 2
SUPPORT_NEGATIVE_REASON: Final = "released-robot-geometry-outside-fixed-camera-prefix"

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
        "minimum_supported_streams_per_object",
        "minimum_supported_object_count",
        "minimum_supported_stream_fraction",
        "camera_images_used_for_eligibility",
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


def validate_deform360_prob4d_camera_eligibility_policy(
    value: object,
) -> dict[str, Any]:
    """Validate and normalize the frozen target-free eligibility policy."""

    if not isinstance(value, Mapping):
        raise ValueError("camera eligibility policy must be a JSON object")
    require_exact_fields(
        value,
        expected=_POLICY_FIELDS,
        name="camera eligibility policy",
    )
    _require(
        value["schema"] == CAMERA_ELIGIBILITY_POLICY_SCHEMA
        and value["schema_version"] == CAMERA_ELIGIBILITY_POLICY_VERSION
        and value["semantics"] == CAMERA_ELIGIBILITY_POLICY_SEMANTICS,
        "unsupported camera eligibility policy",
    )
    identity = dict(value)
    declared_id = nonempty_string(identity.pop("artifact_id"), name="artifact_id")
    _require(
        content_id(identity) == declared_id, "camera eligibility policy ID changed"
    )
    _require(
        value["eligible_status"] == "supported"
        and value["support_negative_action"] == "retain-and-exclude"
        and value["technical_failure_action"] == "terminal"
        and value["allowed_support_negative_reason"] == SUPPORT_NEGATIVE_REASON,
        "camera eligibility decision rule changed",
    )
    for field in (
        "camera_images_used_for_eligibility",
        "prediction_residuals_used_for_eligibility",
        "calibration_outcomes_used_for_eligibility",
        "replacement_allowed",
        "confirmation_payloads_opened",
        "future_frames_used",
        "target_outcomes_used",
        "human_approval_required",
        "new_measurements_required",
    ):
        _require(
            genuine_boolean(value[field], name=field) is False,
            f"camera eligibility boundary changed: {field}",
        )
    normalized = dict(value)
    normalized["protocol_id"] = nonempty_string(
        value["protocol_id"], name="protocol_id"
    )
    normalized["eligibility_evidence"] = nonempty_string(
        value["eligibility_evidence"], name="eligibility_evidence"
    )
    normalized["minimum_supported_streams_per_object"] = genuine_integer(
        value["minimum_supported_streams_per_object"],
        name="minimum_supported_streams_per_object",
        minimum=2,
    )
    normalized["minimum_supported_object_count"] = genuine_integer(
        value["minimum_supported_object_count"],
        name="minimum_supported_object_count",
        minimum=1,
    )
    normalized["minimum_supported_stream_fraction"] = _finite_fraction(
        value["minimum_supported_stream_fraction"],
        name="minimum_supported_stream_fraction",
    )
    normalized["claim_boundary"] = nonempty_string(
        value["claim_boundary"], name="claim_boundary"
    )
    return cast(dict[str, Any], plain_json(normalized))


__all__ = [
    "CAMERA_ELIGIBILITY_POLICY_SCHEMA",
    "CAMERA_ELIGIBILITY_POLICY_SEMANTICS",
    "CAMERA_ELIGIBILITY_POLICY_VERSION",
    "SUPPORT_NEGATIVE_REASON",
    "VISIBLE_STREAM_PLAN_SEMANTICS",
    "VISIBLE_STREAM_PLAN_VERSION",
    "validate_deform360_prob4d_camera_eligibility_policy",
]
