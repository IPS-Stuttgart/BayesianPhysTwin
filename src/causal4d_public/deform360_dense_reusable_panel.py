"""Prospective boundary for the dense reusable Deform360 panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DENSE_REUSABLE_PANEL_SCHEMA_VERSION = 1
DENSE_REUSABLE_PANEL_PROTOCOL_ID = "deform360-dense-reusable-panel-v1"
CANONICAL_DENSE_REUSABLE_PANEL_CONFIG_SHA256 = (
    "1a78b8d74679ebf65768cc5078b34d034a2fcac55f7e0c0a00e50e1967a1c9bd"
)

_EXPECTED_COHORT = {
    "002-rope-silk": ("filament", (0, 2, 5, 6, 7, 9), (3, 4, 8), 1),
    "085-scarf-cloth": ("sheet", (1, 3, 4, 6, 8, 9), (0, 5, 7), 2),
    "083-blanket-cloth": ("sheet", (1, 2, 4, 5, 8, 9), (0, 3, 6), 7),
    "092-squirrel": ("volumetric", (0, 4, 5, 7, 8, 9), (2, 3, 6), 1),
    "170-spider": ("volumetric", (0, 1, 3, 5, 8, 9), (2, 4, 7), 6),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dense_reusable_panel_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dense_reusable_panel_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable five-target panel and its evidence order."""

    _require(
        payload.get("schema_version") == DENSE_REUSABLE_PANEL_SCHEMA_VERSION,
        "unsupported dense reusable panel schema",
    )
    observed = dense_reusable_panel_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "dense reusable panel checksum mismatch",
    )
    _require(
        observed == CANONICAL_DENSE_REUSABLE_PANEL_CONFIG_SHA256,
        "dense reusable panel differs from the canonical lock",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == DENSE_REUSABLE_PANEL_PROTOCOL_ID,
        "dense reusable panel protocol id changed",
    )

    development = config.get("development_object", {})
    _require(
        development.get("object_id") == "081-stripe-rope"
        and development.get("target_episode_id") == 5
        and development.get("excluded_from_confirmatory_panel") is True
        and development.get("target_must_remain_sealed") is True,
        "development object boundary changed",
    )

    cohort = config.get("cohort", [])
    _require(len(cohort) == len(_EXPECTED_COHORT), "panel cohort size changed")
    observed_cohort: dict[str, tuple[str, tuple[int, ...], tuple[int, ...], int]] = {}
    for row in cohort:
        object_id = str(row.get("object_id"))
        source = tuple(int(value) for value in row.get("source_episode_ids", ()))
        calibration = tuple(
            int(value) for value in row.get("calibration_episode_ids", ())
        )
        target = int(row.get("target_episode_id", -1))
        _require(
            set(source).isdisjoint(calibration)
            and target not in source
            and target not in calibration
            and set(source) | set(calibration) | {target} == set(range(10)),
            f"episode partition changed for {object_id}",
        )
        _require(
            int(row.get("canonical_reference_episode_id", -1)) in source,
            f"canonical reference is not source-only for {object_id}",
        )
        observed_cohort[object_id] = (
            str(row.get("stratum")),
            source,
            calibration,
            target,
        )
    _require(observed_cohort == _EXPECTED_COHORT, "panel cohort or split changed")

    method = config.get("dense_reusable_method", {})
    _require(
        method.get("association_uses_simulator_residual") is False
        and method.get("association_uses_future_object_frames") is False,
        "association crossed the causal information boundary",
    )
    registration = method.get("canonical_episode_registration", {})
    _require(
        method.get("canonical_surface_node_count") == 192
        and method.get("minimum_canonical_surface_node_count") <= 192
        and "topology and object rest lengths are never rebuilt per episode"
        in str(method.get("canonical_geometry", "")),
        "canonical graph reuse is no longer explicit",
    )
    _require(
        registration.get("one_to_one_assignment") is False
        and registration.get("partial_visibility_is_explicit") is True
        and registration.get("assignment_mixture_spread_enters_observation_covariance")
        is True
        and registration.get("simulator_state_innovation_used_as_prior_reliability")
        is False,
        "cross-episode association uncertainty contract changed",
    )
    state_completion = method.get("partial_graph_state_completion", {})
    _require(
        state_completion.get("uses_simulator_residual") is False
        and state_completion.get("uses_future_object_frames") is False
        and state_completion.get("uses_only_object_geometry_frames") == [0]
        and method.get("temporal_prefix_frame_count") == 1
        and state_completion.get("uses_prefix_visibility_frame_count") == 1
        and state_completion.get("bridge_strain_weight") == 3.0
        and state_completion.get("contact_anchor_weight") == 10.0
        and state_completion.get("maximum_p99_relative_edge_strain") == 0.5
        and state_completion.get("maximum_bridge_relative_edge_strain") == 0.5
        and state_completion.get("maximum_contact_anchor_error_m") == 0.015
        and method.get("controller_input_group_size") == 768,
        "partial state-completion boundary changed",
    )
    _require(
        str(method.get("unsupported_controller_group_policy", "")).startswith(
            "reject episode before rollout"
        ),
        "unsupported contacts no longer fail closed",
    )
    _require(
        method.get("model_averaging_promoted") is False,
        "failed model averaging was silently promoted",
    )

    source_gate = config.get("source_admission", {})
    calibration_gate = config.get("calibration_admission", {})
    target = config.get("target_panel", {})
    frame_protocol = config.get("frame_protocol", {})
    window_selection = frame_protocol.get("window_selection", {})
    _require(
        window_selection.get("window_length_frames") == 81
        and window_selection.get("tie_break") == "earliest start"
        and window_selection.get("candidate_starts", {}).get("first") == 8
        and window_selection.get("candidate_starts", {}).get("stride") == 6
        and window_selection.get("input_fields")
        == [
            "robot/robot.npz:actions",
            "robot/robot.npz:openings",
            "episode frame count",
        ]
        and str(window_selection.get("rule", "")).startswith(
            "maximize mean gripper-centre path weighted"
        )
        and window_selection.get("static_aperture_fallback")
        == "confidence one, reducing exactly to unweighted path when aperture cannot identify closure"
        and window_selection.get("known_future_action_is_conditioning_input") is True
        and window_selection.get("object_geometry_or_tactile_used_for_selection")
        is False
        and frame_protocol.get("superseded_fixed_raw_aligned_range_half_open")
        == [110, 191],
        "action-only episode alignment contract changed",
    )
    _require(
        source_gate.get("all_five_objects_must_pass") is True,
        "source gate is no longer cohort-conjunctive",
    )
    _require(
        calibration_gate.get("all_five_objects_must_be_scored") is True
        and calibration_gate.get("all_gates_conjunctive") is True,
        "calibration gate is no longer cohort-conjunctive",
    )
    _require(
        target.get("target_count") == 5
        and target.get("open_as_one_conjunctive_panel") is True
        and target.get("partial_target_opening_allowed") is False
        and target.get("target_prefix_allowed_before_calibration_pass") is False
        and target.get("target_initial_frame_allowed_after_calibration_pass") is True
        and target.get("target_action_trajectory_allowed_after_calibration_pass")
        is True
        and target.get("target_post_initial_object_observations_allowed") is False
        and target.get("target_future_allowed_before_prediction_seal") is False,
        "target opening boundary changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("five_target_prefixes_read") is False
        and boundary.get("five_target_action_trajectories_read") is False
        and boundary.get("five_target_initial_frames_read") is False
        and boundary.get("five_target_post_initial_object_observations_read") is False
        and boundary.get("five_target_future_geometry_read") is False
        and boundary.get("five_target_future_tactile_read") is False
        and boundary.get("development_081_target_read") is False
        and boundary.get("target_requests_authorized_by_this_config") is False,
        "target evidence was read or authorized",
    )
    return {
        "passed": True,
        "protocol_id": DENSE_REUSABLE_PANEL_PROTOCOL_ID,
        "config_sha256": observed,
        "object_ids": sorted(_EXPECTED_COHORT),
        "target_episode_ids": {
            object_id: values[3] for object_id, values in _EXPECTED_COHORT.items()
        },
    }


def load_dense_reusable_panel_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "dense panel config must be an object")
    validate_dense_reusable_panel_config(payload)
    return payload


def authorize_dense_panel_episode(
    payload: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    phase: str,
    source_admission_passed: bool = False,
) -> dict[str, Any]:
    """Authorize source or calibration work; this lock never opens targets."""

    validated = validate_dense_reusable_panel_config(payload)
    _require(object_id in _EXPECTED_COHORT, "object is outside the dense panel")
    _, source, calibration, target = _EXPECTED_COHORT[object_id]
    episode_id = int(episode_id)
    _require(episode_id != target, "target episode remains sealed")
    if phase == "source":
        _require(episode_id in source, "episode is outside the source partition")
    elif phase == "calibration":
        _require(source_admission_passed, "source admission has not passed")
        _require(
            episode_id in calibration,
            "episode is outside the calibration partition",
        )
    else:
        raise ValueError("phase must be source or calibration")
    return {
        **validated,
        "object_id": object_id,
        "episode_id": episode_id,
        "phase": phase,
        "target_access": False,
    }


def audit_dense_panel_target_boundary(
    payload: Mapping[str, Any],
    *,
    replication_root: str | Path,
) -> dict[str, Any]:
    """Fail when a sealed target appears in a protected derived-data tree."""

    validated = validate_dense_reusable_panel_config(payload)
    root = Path(replication_root)
    protected = tuple(payload["config"]["protected_stage_roots"])
    records = []
    for object_id, (_, _, _, target) in sorted(_EXPECTED_COHORT.items()):
        for stage in protected:
            path = root / stage / object_id / f"episode_{target:04d}"
            records.append(
                {
                    "object_id": object_id,
                    "target_episode_id": target,
                    "stage": stage,
                    "path": str(path),
                    "exists": path.exists(),
                }
            )
    leaked = [record for record in records if record["exists"]]
    _require(not leaked, "a sealed target exists in a protected stage tree")
    return {
        "schema_version": DENSE_REUSABLE_PANEL_SCHEMA_VERSION,
        "artifact_kind": "Deform360DenseReusablePanelTargetBoundaryAudit",
        "protocol_id": validated["protocol_id"],
        "config_sha256": validated["config_sha256"],
        "replication_root": str(root),
        "records": records,
        "target_prefix_read": False,
        "target_future_geometry_read": False,
        "target_future_tactile_read": False,
        "passed": True,
    }


__all__ = [
    "CANONICAL_DENSE_REUSABLE_PANEL_CONFIG_SHA256",
    "DENSE_REUSABLE_PANEL_PROTOCOL_ID",
    "audit_dense_panel_target_boundary",
    "authorize_dense_panel_episode",
    "dense_reusable_panel_config_sha256",
    "load_dense_reusable_panel_config",
    "validate_dense_reusable_panel_config",
]
