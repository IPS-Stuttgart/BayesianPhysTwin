"""Validation for the source-only PokeFlex RealSense anchor protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .pokeflex_registration_protocol import (
    POKEFLEX_REGISTRATION_PROTOCOL_SHA256,
)


POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_ID = (
    "pokeflex-independent-depth-development-v1"
)
POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256 = (
    "30e00c23a00e0b4e80af969d3190703f4d9d0b11d21cf8bbf88a9d9306049653"
)
EXPECTED_DEVELOPMENT_OBJECTS = (
    "FoamDice",
    "MemoryFoam",
    "PlushOctopus",
    "3dPrintedHeart",
    "ToiletPaperRoll",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pokeflex_independent_depth_protocol_sha256(
    payload: Mapping[str, Any],
) -> str:
    """Return the canonical checksum without the embedded digest."""

    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_pokeflex_independent_depth_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate sensor roles, causal timing, and unopened partitions."""

    _require(payload.get("schema_version") == 1, "unsupported protocol schema")
    _require(
        payload.get("artifact_kind")
        == "PokeFlexIndependentDepthDevelopmentProtocol",
        "unexpected protocol kind",
    )
    _require(
        payload.get("protocol_id") == POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_ID,
        "independent-depth protocol id changed",
    )
    observed = pokeflex_independent_depth_protocol_sha256(payload)
    _require(
        payload.get("protocol_sha256") == observed,
        "independent-depth protocol checksum mismatch",
    )
    if POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256 != "TO_BE_FILLED":
        _require(
            observed == POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
            "independent-depth protocol differs from canonical lock",
        )

    parent = payload.get("parent_protocol")
    _require(isinstance(parent, Mapping), "parent protocol is missing")
    _require(
        parent.get("protocol_sha256") == POKEFLEX_REGISTRATION_PROTOCOL_SHA256,
        "parent protocol checksum changed",
    )
    upstream = payload.get("upstream")
    _require(isinstance(upstream, Mapping), "upstream provenance is missing")
    _require(
        upstream.get("code_commit") == "aaa8726072834a95bbe97e1a113588968c36e185",
        "PokeFlex upstream commit changed",
    )

    boundary = payload.get("evidence_boundary")
    _require(isinstance(boundary, Mapping), "evidence boundary is missing")
    _require(
        tuple(boundary.get("development_objects", ()))
        == EXPECTED_DEVELOPMENT_OBJECTS,
        "development object boundary changed",
    )
    _require(
        boundary.get("outcome_open_design_takes")
        == ["T1", "T3", "T4", "T5", "T6"],
        "outcome-open take inventory changed",
    )
    _require(
        boundary.get("prospective_development_validation_take") == "T2",
        "prospective development take changed",
    )
    _require(
        boundary.get("calibration_objects_remain_sealed") is True
        and boundary.get("target_objects_remain_sealed") is True,
        "calibration or target objects were unsealed",
    )
    _require(boundary.get("replacement_allowed") is False, "replacement changed")

    causal = payload.get("causal_input_contract")
    _require(isinstance(causal, Mapping), "causal input contract is missing")
    expected_causal = {
        "kinect_history": "f-5 through f-1 only",
        "realsense_history": "f-5 through f-1 only",
        "robot_history": "through f-1 only",
        "frame_f_kinect_or_realsense_allowed_before_prediction": False,
        "frame_f_mesh_allowed_before_scoring": False,
        "template_mesh_allowed_for_static_sensor_calibration": True,
        "synthetic_points_from_frame_f_mesh_allowed": False,
    }
    for key, expected in expected_causal.items():
        _require(causal.get(key) == expected, f"causal contract changed: {key}")

    anchor = payload.get("anchor_contract")
    _require(isinstance(anchor, Mapping), "anchor contract is missing")
    expected_anchor = {
        "depth_scale": 10000.0,
        "depth_unit_after_conversion": "m",
        "variance_unit": "m^2",
        "invalid_depth_value": 65535,
        "extrinsic_convention": (
            "per-frame world-to-camera matrix; invert for back-projection"
        ),
        "camera_clusters_have_fixed_information_mass": True,
        "unknown_cross-camera_correlation_rule": (
            "maximum per-camera baseline-relative regret"
        ),
        "camera_bias_is_not_assumed_zero": True,
        "state_innovation_is_processed_once": True,
        "exact_fallback": "released Kinect checkpoint vertices byte-for-byte",
    }
    for key, expected in expected_anchor.items():
        _require(anchor.get(key) == expected, f"anchor contract changed: {key}")

    gates = payload.get("source_competence_gates")
    _require(isinstance(gates, Mapping), "source competence gates are missing")
    _require(
        gates.get("all_required_before_T2_access") is True,
        "T2 access gate changed",
    )
    _require(
        float(gates.get("maximum_false_safe_rate", 1.0)) <= 0.1,
        "false-safe gate weakened",
    )
    _require(
        int(gates.get("minimum_clustered_points_per_camera", 0)) >= 64,
        "anchor support gate weakened",
    )
    return {
        "passed": True,
        "protocol_sha256": observed,
        "development_objects": EXPECTED_DEVELOPMENT_OBJECTS,
        "prospective_development_validation_take": "T2",
    }


def load_pokeflex_independent_depth_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the canonical source-only protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("artifact_kind")
        == "PokeFlexIndependentDepthSourceValidationProtocol"
    ):
        from .pokeflex_independent_depth_source_validation_protocol import (
            validate_pokeflex_independent_depth_source_validation_protocol,
        )

        result = validate_pokeflex_independent_depth_source_validation_protocol(
            payload
        )
    else:
        result = validate_pokeflex_independent_depth_protocol(payload)
    result["path"] = str(source)
    result["payload"] = payload
    return result
