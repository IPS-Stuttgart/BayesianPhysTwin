"""Validation for the reusable-PhysTwin contact-transition addendum."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONTACT_TRANSITION_ADDENDUM_SCHEMA_VERSION = 1
CONTACT_TRANSITION_ADDENDUM_ID = "deform360-reusable-contact-transition-v1"
PARENT_PROTOCOL_ID = "deform360-reusable-sota-v1"
PARENT_CONFIG_SHA256 = (
    "64e41773d3e333987dc97a11c52e3474fd34fe65139e6c7d799b3e8f4db188cd"
)
CANONICAL_CONTACT_TRANSITION_CONFIG_SHA256 = (
    "7c0711e130e32d030e1c82e45c2249ed8d7654473bdf8eee889310daa89ed334"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def contact_transition_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_contact_transition_addendum(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == CONTACT_TRANSITION_ADDENDUM_SCHEMA_VERSION,
        "unsupported contact-transition addendum schema",
    )
    observed = contact_transition_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "contact-transition addendum checksum mismatch",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == CONTACT_TRANSITION_ADDENDUM_ID,
        "contact-transition addendum id changed",
    )
    parent = config.get("parent_protocol", {})
    _require(
        parent.get("protocol_id") == PARENT_PROTOCOL_ID
        and parent.get("config_sha256") == PARENT_CONFIG_SHA256
        and parent.get("modified") is False,
        "contact-transition parent lock changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("fit_episode_tactile_may_train_transition") is True
        and boundary.get("held_initial_object_frame_count") == 1
        and boundary.get("held_future_robot_trajectory_allowed") is True
        and boundary.get("held_future_object_geometry_before_seal") is False
        and boundary.get("held_future_tactile_before_seal") is False
        and boundary.get("confirmatory_data_access_before_method_freeze") is False,
        "contact-transition information boundary changed",
    )
    model = config.get("model", {})
    _require(
        tuple(model.get("features", ()))
        == (
            "gripper_openness_m",
            "gripper_to_predicted_object_proximity_m",
            "relative_closing_speed_m_s",
        )
        and model.get("initial_contact_state")
        == "infer from frame-zero onset hazard; no target tactile"
        and model.get("feature_rollout")
        == "one geometry-latched Warp rollout followed by one hazard-conditioned Warp rerun"
        and model.get("learned_geometry_residual") is False,
        "contact-transition model boundary changed",
    )
    fallback = config.get("fallback", {})
    _require(
        fallback.get("invalid_or_unsupported_contact") == "exact persistence"
        and fallback.get("failed_development_gate")
        == "retain frozen static-contact trusted arm"
        and fallback.get("held_outcome_may_choose_fallback") is False,
        "contact-transition fallback changed",
    )
    gate = config.get("development_gate", {})
    _require(
        gate.get("development_object_count") == 12
        and gate.get("development_held_episode_count") == 48
        and gate.get("minimum_cd_improvement_vs_static_fraction") == 0.02
        and gate.get("minimum_track_improvement_vs_static_fraction") == 0.02
        and gate.get("minimum_episode_win_count_out_of_48") == 28
        and gate.get("maximum_episode_degradation_fraction") == 0.10
        and gate.get("no_category_median_degradation") is True
        and gate.get("contact_brier_must_improve") is True
        and gate.get("all_gates_conjunctive") is True,
        "contact-transition development gate changed",
    )
    _require(
        observed == CANONICAL_CONTACT_TRANSITION_CONFIG_SHA256,
        "contact-transition addendum differs from canonical lock",
    )
    return {
        "passed": True,
        "protocol_id": CONTACT_TRANSITION_ADDENDUM_ID,
        "config_sha256": observed,
        "development_held_episode_count": 48,
        "confirmatory_access_authorized": False,
    }


def load_contact_transition_addendum(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "contact-transition addendum must be an object")
    validate_contact_transition_addendum(payload)
    return payload


__all__ = [
    "CANONICAL_CONTACT_TRANSITION_CONFIG_SHA256",
    "CONTACT_TRANSITION_ADDENDUM_ID",
    "PARENT_CONFIG_SHA256",
    "PARENT_PROTOCOL_ID",
    "contact_transition_config_sha256",
    "load_contact_transition_addendum",
    "validate_contact_transition_addendum",
]
