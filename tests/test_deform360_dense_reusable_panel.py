from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_dense_reusable_panel import (
    audit_dense_panel_target_boundary,
    authorize_dense_panel_episode,
    load_dense_reusable_panel_config,
    validate_dense_reusable_panel_config,
)


CONFIG = (
    Path(__file__).parents[1]
    / "configs/causal4d_public/deform360_dense_reusable_panel_v1.json"
)


def test_dense_panel_config_locks_five_fresh_targets() -> None:
    payload = load_dense_reusable_panel_config(CONFIG)
    validated = validate_dense_reusable_panel_config(payload)

    assert len(validated["object_ids"]) == 5
    assert validated["target_episode_ids"]["002-rope-silk"] == 1
    assert "081-stripe-rope" not in validated["object_ids"]
    method = payload["config"]["dense_reusable_method"]
    assert method["temporal_prefix_frame_count"] == 1
    assert (
        method["partial_graph_state_completion"]["uses_prefix_visibility_frame_count"]
        == 1
    )
    target = payload["config"]["target_panel"]
    assert target["target_initial_frame_allowed_after_calibration_pass"] is True
    assert target["target_action_trajectory_allowed_after_calibration_pass"] is True
    assert target["target_post_initial_object_observations_allowed"] is False
    window = payload["config"]["frame_protocol"]["window_selection"]
    assert window["window_length_frames"] == 81
    assert window["input_fields"] == [
        "robot/robot.npz:actions",
        "robot/robot.npz:openings",
        "episode frame count",
    ]
    assert window["candidate_starts"]["first"] == 8
    assert window["candidate_starts"]["stride"] == 6
    assert window["object_geometry_or_tactile_used_for_selection"] is False


def test_dense_panel_rejects_mutation_and_target_requests() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["config"]["target_panel"]["partial_target_opening_allowed"] = True
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_dense_reusable_panel_config(changed)

    with pytest.raises(ValueError, match="target episode remains sealed"):
        authorize_dense_panel_episode(
            payload,
            object_id="002-rope-silk",
            episode_id=1,
            phase="source",
        )


def test_dense_panel_requires_source_admission_before_calibration() -> None:
    payload = load_dense_reusable_panel_config(CONFIG)

    source = authorize_dense_panel_episode(
        payload,
        object_id="170-spider",
        episode_id=0,
        phase="source",
    )
    assert source["target_access"] is False

    with pytest.raises(ValueError, match="source admission has not passed"):
        authorize_dense_panel_episode(
            payload,
            object_id="170-spider",
            episode_id=2,
            phase="calibration",
        )
    calibration = authorize_dense_panel_episode(
        payload,
        object_id="170-spider",
        episode_id=2,
        phase="calibration",
        source_admission_passed=True,
    )
    assert calibration["target_access"] is False


def test_target_boundary_audit_fails_on_any_derived_target(tmp_path: Path) -> None:
    payload = load_dense_reusable_panel_config(CONFIG)
    clean = audit_dense_panel_target_boundary(payload, replication_root=tmp_path)
    assert clean["passed"] is True
    assert len(clean["records"]) == 15

    leaked = tmp_path / "observations/092-squirrel/episode_0001"
    leaked.mkdir(parents=True)
    with pytest.raises(ValueError, match="sealed target exists"):
        audit_dense_panel_target_boundary(payload, replication_root=tmp_path)
