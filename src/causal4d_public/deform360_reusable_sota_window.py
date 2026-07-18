"""Locked action-window addendum for the reusable-PhysTwin SOTA panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_action_audit import summarize_robot_action
from .deform360_reusable_sota_protocol import (
    CANONICAL_REUSABLE_SOTA_CONFIG_SHA256,
    EXPECTED_DEVELOPMENT_OBJECTS,
    EXPECTED_FIT_EPISODES,
    EXPECTED_HELD_EPISODES,
    REUSABLE_SOTA_PROTOCOL_ID,
    validate_reusable_sota_config,
)


REUSABLE_SOTA_WINDOW_SCHEMA_VERSION = 1
REUSABLE_SOTA_WINDOW_PROTOCOL_ID = "deform360-reusable-sota-window-v1"
CANONICAL_REUSABLE_SOTA_WINDOW_SHA256 = (
    "331faa06dd3d81d1ddea615828eae2b4a040c3a56b92058e5b037a6129f1f9c8"
)

EXPECTED_INPUT_FIELDS = (
    "robot/robot.npz:actions",
    "robot/robot.npz:openings",
    "episode frame count",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reusable_sota_window_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an addendum without its self-declared digest."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reusable_sota_window(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject changes to the inherited action-only temporal protocol."""

    _require(
        payload.get("schema_version") == REUSABLE_SOTA_WINDOW_SCHEMA_VERSION,
        "unsupported reusable-SOTA window schema",
    )
    observed = reusable_sota_window_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "reusable-SOTA window checksum mismatch",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == REUSABLE_SOTA_WINDOW_PROTOCOL_ID,
        "reusable-SOTA window protocol id changed",
    )
    parent = config.get("parent_protocol", {})
    _require(
        parent.get("protocol_id") == REUSABLE_SOTA_PROTOCOL_ID
        and parent.get("config_sha256") == CANONICAL_REUSABLE_SOTA_CONFIG_SHA256,
        "reusable-SOTA window parent changed",
    )
    selection = config.get("window_selection", {})
    candidates = selection.get("candidate_starts", {})
    _require(
        selection.get("window_length_frames") == 81
        and candidates.get("first") == 8
        and candidates.get("stride") == 6
        and candidates.get("stop") == "last start whose complete 81-frame window exists"
        and selection.get("tie_break") == "earliest start"
        and tuple(selection.get("input_fields", ())) == EXPECTED_INPUT_FIELDS
        and selection.get("known_future_action_is_conditioning_input") is True
        and selection.get("object_geometry_or_tactile_used_for_selection") is False
        and str(selection.get("rule", "")).startswith(
            "maximize mean gripper-centre path"
        ),
        "reusable-SOTA action-window rule changed",
    )
    frames = config.get("frame_protocol", {})
    _require(
        tuple(frames.get("comparison_fixed_raw_range_half_open", ())) == (110, 191)
        and frames.get("tracking_tail_frames_skipped") == 5
        and frames.get("processed_frame_count") == 76
        and tuple(frames.get("evaluation_range_half_open", ())) == (1, 76)
        and frames.get("horizon_ranges_half_open")
        == {"early": [1, 26], "middle": [26, 51], "late": [51, 76]},
        "reusable-SOTA temporal evaluation changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("held_window_selection_may_read_full_robot_action") is True
        and boundary.get("held_window_selection_may_read_object_geometry") is False
        and boundary.get("held_window_selection_may_read_object_tracks") is False
        and boundary.get("held_window_selection_may_read_tactile") is False
        and boundary.get("held_prediction_object_input_frame_count") == 1
        and boundary.get("held_future_outcomes_sealed_until_prediction_hash") is True,
        "reusable-SOTA window information boundary changed",
    )
    claim = config.get("claim_boundary", {})
    _require(
        claim.get("compute_addendum_only") is True
        and claim.get("direct_deform360_table4_claim") is False
        and claim.get("official_evaluator_parity_unresolved") is True,
        "reusable-SOTA window claim boundary changed",
    )
    _require(
        observed == CANONICAL_REUSABLE_SOTA_WINDOW_SHA256,
        "reusable-SOTA window differs from the canonical lock",
    )
    return {
        "passed": True,
        "protocol_id": REUSABLE_SOTA_WINDOW_PROTOCOL_ID,
        "config_sha256": observed,
        "window_length_frames": 81,
        "processed_frame_count": 76,
    }


def load_reusable_sota_window(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "reusable-SOTA window must be an object")
    validate_reusable_sota_window(payload)
    return payload


def authorize_development_fit_window(
    parent: Mapping[str, Any],
    addendum: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Authorize only a development fit episode for outcome-visible staging."""

    validate_reusable_sota_config(parent)
    validate_reusable_sota_window(addendum)
    development = {
        value for values in EXPECTED_DEVELOPMENT_OBJECTS.values() for value in values
    }
    _require(object_id in development, "window staging is not a development object")
    _require(
        int(episode_id) in EXPECTED_FIT_EPISODES,
        "window staging is not a development fit episode",
    )
    return {
        "passed": True,
        "operation": "development-fit-staging",
        "protocol_id": REUSABLE_SOTA_WINDOW_PROTOCOL_ID,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "parent_protocol_id": REUSABLE_SOTA_PROTOCOL_ID,
        "parent_config_sha256": CANONICAL_REUSABLE_SOTA_CONFIG_SHA256,
        "window_protocol_id": REUSABLE_SOTA_WINDOW_PROTOCOL_ID,
        "window_config_sha256": addendum["config_sha256"],
        "held_outcome_read": False,
        "confirmatory_object_read": False,
    }


def authorize_development_held_prediction_window(
    parent: Mapping[str, Any],
    addendum: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Authorize one-frame, outcome-sealed development prediction staging."""

    validate_reusable_sota_config(parent)
    validate_reusable_sota_window(addendum)
    development = {
        value for values in EXPECTED_DEVELOPMENT_OBJECTS.values() for value in values
    }
    _require(object_id in development, "window staging is not a development object")
    _require(
        int(episode_id) in EXPECTED_HELD_EPISODES,
        "window staging is not a development held episode",
    )
    return {
        "passed": True,
        "operation": "development-held-prediction-staging",
        "protocol_id": REUSABLE_SOTA_WINDOW_PROTOCOL_ID,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "parent_protocol_id": REUSABLE_SOTA_PROTOCOL_ID,
        "parent_config_sha256": CANONICAL_REUSABLE_SOTA_CONFIG_SHA256,
        "window_protocol_id": REUSABLE_SOTA_WINDOW_PROTOCOL_ID,
        "window_config_sha256": addendum["config_sha256"],
        "held_action_read": True,
        "held_object_input_frame_count": 1,
        "held_future_object_read": False,
        "held_tactile_read": False,
        "prediction_seal_required_before_outcome_reveal": True,
        "confirmatory_object_read": False,
    }


def select_reusable_sota_action_window(
    actions: np.ndarray,
    openings: np.ndarray,
    addendum: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the locked window from action and aperture streams only."""

    validate_reusable_sota_window(addendum)
    config = addendum["config"]
    selection = config["window_selection"]
    fixed_start, fixed_stop = config["frame_protocol"][
        "comparison_fixed_raw_range_half_open"
    ]
    summary = summarize_robot_action(
        actions,
        openings,
        locked_start=int(fixed_start),
        locked_stop=int(fixed_stop),
        candidate_start_frame=int(selection["candidate_starts"]["first"]),
        candidate_stride_frames=int(selection["candidate_starts"]["stride"]),
    )
    selected = summary["best_contact_conditioned_path_window"]["frame_range_half_open"]
    _require(
        int(selected[1]) - int(selected[0]) == int(selection["window_length_frames"]),
        "selected action window has the wrong length",
    )
    return {
        "selected_raw_frame_range_half_open": [int(selected[0]), int(selected[1])],
        "selection_rule": selection,
        "action_summary": summary,
        "object_geometry_read": False,
        "tactile_read": False,
        "future_action_is_conditioning_input": True,
    }


__all__ = [
    "CANONICAL_REUSABLE_SOTA_WINDOW_SHA256",
    "REUSABLE_SOTA_WINDOW_PROTOCOL_ID",
    "authorize_development_fit_window",
    "authorize_development_held_prediction_window",
    "load_reusable_sota_window",
    "reusable_sota_window_sha256",
    "select_reusable_sota_action_window",
    "validate_reusable_sota_window",
]
