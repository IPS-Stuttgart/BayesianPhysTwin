"""Locked state semantics for the fresh Deform360 reusable-twin panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .deform360_reusable_trust_masks import (
    SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
    load_reusable_trust_mask_addendum,
    sha256_file,
)


STATE_ADDENDUM_ID = "deform360-reusable-trust-state-addendum-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_reusable_trust_state_addendum(
    parent_path: str | Path,
    physics_path: str | Path,
    execution_path: str | Path,
    mask_path: str | Path,
    state_path: str | Path,
) -> dict[str, Any]:
    """Validate the source-only state policy against all preceding locks."""

    protocol = load_reusable_trust_mask_addendum(
        parent_path,
        physics_path,
        execution_path,
        mask_path,
    )
    state_file = Path(state_path).resolve()
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "state addendum must contain an object")
    _require(payload.get("schema_version") == 1, "state addendum schema changed")
    _require(
        payload.get("protocol_id") == STATE_ADDENDUM_ID,
        "state addendum identity changed",
    )
    parents = payload.get("parent_locks", {})
    _require(
        parents.get("fresh_protocol_file_sha256") == protocol["parent_file_sha256"]
        and parents.get("physics_addendum_file_sha256")
        == protocol["addendum_file_sha256"]
        and parents.get("execution_lock_file_sha256")
        == protocol["execution_file_sha256"]
        and parents.get("mask_addendum_id") == SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID
        and parents.get("mask_addendum_file_sha256")
        == protocol["mask_addendum_file_sha256"],
        "state addendum uses another parent lock",
    )
    timing = payload.get("lock_timing", {})
    _require(
        timing.get("source_future_object_outcomes_inspected") is False
        and timing.get("held_out_media_inspected") is False
        and timing.get("held_out_outcomes_inspected") is False,
        "state policy was not locked before outcome access",
    )
    policy = payload.get("state_policy", {})
    _require(
        policy.get("mode") == "rigid-rest-preserving"
        and policy.get("object_rest_lengths_changed") is False
        and policy.get("object_topology_changed") is False
        and policy.get("episode_readout_is_external") is True
        and policy.get("readout_covariance_includes_assignment_spread") is True
        and policy.get("simulator_residual_used") is False
        and policy.get("post_initial_object_observation_used") is False
        and policy.get("target_tactile_used") is False,
        "state semantics crossed the frozen information boundary",
    )
    gate = payload.get("source_gate", {})
    _require(
        gate.get("required_episode_ids") == [1, 3, 4, 6, 7, 9]
        and gate.get("all_frame_zero_state_gates_must_pass") is True
        and gate.get("all_reference_physics_rollouts_must_be_finite") is True
        and float(gate.get("maximum_p99_warp_edge_strain", -1.0)) == 0.5
        and float(gate.get("contact_attachment_must_be_within_m", -1.0)) == 0.03,
        "state source-admission gate changed",
    )
    return {
        **protocol,
        "state_addendum": payload,
        "state_addendum_path": str(state_file),
        "state_addendum_file_sha256": sha256_file(state_file),
    }


__all__ = ["STATE_ADDENDUM_ID", "load_reusable_trust_state_addendum"]
