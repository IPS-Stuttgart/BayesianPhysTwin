"""Validation for the frozen PokeFlex independent-depth source study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .pokeflex_independent_depth_protocol import (
    EXPECTED_DEVELOPMENT_OBJECTS,
    POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
)
from .pokeflex_registration_protocol import POKEFLEX_REGISTRATION_PROTOCOL_SHA256


POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_ID = (
    "pokeflex-independent-depth-source-validation-v2"
)
POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_SHA256 = (
    "04ecf64f4b15e543825732b4f9296119ebd5fe47d99dcfc8230cfb70f0c86051"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pokeflex_independent_depth_source_validation_sha256(
    payload: Mapping[str, Any],
) -> str:
    """Return the canonical checksum without the embedded digest."""

    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_pokeflex_independent_depth_source_validation_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the T3 design and T1/T4/T5/T6 source-validation boundary."""

    _require(payload.get("schema_version") == 1, "unsupported protocol schema")
    _require(
        payload.get("artifact_kind")
        == "PokeFlexIndependentDepthSourceValidationProtocol",
        "unexpected source-validation protocol kind",
    )
    _require(
        payload.get("protocol_id")
        == POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_ID,
        "source-validation protocol id changed",
    )
    observed = pokeflex_independent_depth_source_validation_sha256(payload)
    _require(
        payload.get("protocol_sha256") == observed,
        "source-validation protocol checksum mismatch",
    )
    if POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_SHA256 != "TO_BE_FILLED":
        _require(
            observed == POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_SHA256,
            "source-validation protocol differs from canonical lock",
        )

    parent = payload.get("parent_protocol")
    predecessor = payload.get("predecessor_protocol")
    _require(isinstance(parent, Mapping), "parent protocol is missing")
    _require(isinstance(predecessor, Mapping), "predecessor protocol is missing")
    _require(
        parent.get("protocol_sha256") == POKEFLEX_REGISTRATION_PROTOCOL_SHA256,
        "parent protocol checksum changed",
    )
    _require(
        predecessor.get("protocol_sha256")
        == POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
        "predecessor protocol checksum changed",
    )

    boundary = payload.get("evidence_boundary")
    _require(isinstance(boundary, Mapping), "evidence boundary is missing")
    _require(
        tuple(boundary.get("development_objects", ())) == EXPECTED_DEVELOPMENT_OBJECTS,
        "development object boundary changed",
    )
    _require(boundary.get("method_design_take") == "T3", "design take changed")
    _require(
        boundary.get("source_validation_takes") == ["T1", "T4", "T5", "T6"],
        "source-validation take inventory changed",
    )
    _require(
        boundary.get("prospective_development_validation_take") == "T2",
        "prospective T2 boundary changed",
    )
    _require(
        boundary.get("calibration_objects_remain_sealed") is True
        and boundary.get("target_objects_remain_sealed") is True,
        "calibration or target objects were unsealed",
    )
    _require(boundary.get("replacement_allowed") is False, "replacement changed")

    method = payload.get("method_lock")
    _require(isinstance(method, Mapping), "method lock is missing")
    _require(
        float(method.get("static_template_support_radius_mm", 0.0)) == 15.0,
        "template support radius changed",
    )
    _require(
        float(method.get("maximum_calibration_median_residual_mm", 0.0)) == 10.0,
        "sensor calibration gate changed",
    )
    _require(
        method.get("unknown_cross_camera_correlation_rule")
        == "maximum regret over calibration-qualified RealSense sensors",
        "unknown-correlation rule changed",
    )
    _require(
        method.get("correction_fields")
        == [
            "action_local_state_relative_0.4",
            "action_local_state_relative_0.55",
            "action_local_state_relative_0.7",
        ],
        "candidate field family changed",
    )
    _require(
        method.get("correction_scales") == [0.0, 0.125, 0.25, 0.5, 1.0],
        "candidate scale family changed",
    )
    _require(
        method.get("exact_fallback")
        == "released Kinect checkpoint vertices byte-for-byte",
        "exact fallback changed",
    )

    validation = payload.get("source_validation")
    _require(isinstance(validation, Mapping), "source-validation gates are missing")
    _require(
        validation.get("maximum_frame") is None,
        "source-validation horizon changed",
    )
    _require(
        float(validation.get("minimum_regret_sign_agreement", 0.0)) >= 0.65,
        "sign-agreement gate weakened",
    )
    _require(
        float(validation.get("maximum_false_safe_rate", 1.0)) <= 0.1,
        "false-safe gate weakened",
    )
    _require(
        float(validation.get("minimum_regret_spearman", 0.0)) >= 0.2,
        "rank-correlation gate weakened",
    )
    _require(
        float(
            validation.get(
                "minimum_object_balanced_CD_UL1_relative_improvement", 0.0
            )
        )
        >= 0.05,
        "source transfer gate weakened",
    )
    _require(
        int(validation.get("minimum_object_wins", 0)) >= 4,
        "object-win gate weakened",
    )
    _require(
        float(validation.get("maximum_per_object_relative_regression", 1.0)) <= 0.1,
        "object-regression gate weakened",
    )
    _require(
        validation.get("all_required_before_T2_access") is True,
        "T2 access gate changed",
    )
    return {
        "passed": True,
        "protocol_sha256": observed,
        "development_objects": EXPECTED_DEVELOPMENT_OBJECTS,
        "source_validation_takes": ("T1", "T4", "T5", "T6"),
        "prospective_development_validation_take": "T2",
    }
