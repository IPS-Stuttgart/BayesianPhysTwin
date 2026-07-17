from __future__ import annotations

import json

import numpy as np
import pytest

from causal4d_public.deform360_contact_conditioned_action import (
    condition_controller_action,
    controller_spring_group_indices,
    load_contact_conditioned_action_artifact,
    write_contact_conditioned_action_artifact,
)


def _controllers() -> np.ndarray:
    controls = np.zeros((6, 4, 3), dtype=np.float64)
    controls[:, :2, 0] = np.arange(6)[:, None] * 0.01
    controls[:, 2:, 0] = 0.2 + np.arange(6)[:, None] * 0.01
    controls[:, [1, 3], 1] = 0.005
    return controls


def test_conditioned_action_holds_each_group_at_contact_onset() -> None:
    controls = _controllers()
    active = np.zeros((6, 2), dtype=bool)
    active[2:5, 0] = True
    active[4:, 1] = True
    initial = np.array([[0.02, 0.0, 0.0], [0.24, 0.0, 0.0]])

    result = condition_controller_action(
        controls,
        active,
        initial,
        controller_group_size=2,
        maximum_contact_distance_m=0.01,
    )

    assert result.source_group_indices == (0, 1)
    assert result.onset_frames == (2, 4)
    assert result.contact_active.shape == (6, 2)
    np.testing.assert_allclose(
        result.controller_points_m[:3, :2],
        np.repeat(controls[2:3, :2], 3, axis=0),
    )
    np.testing.assert_allclose(
        result.controller_points_m[:5, 2:],
        np.repeat(controls[4:5, 2:], 5, axis=0),
    )
    np.testing.assert_allclose(result.controller_points_m[3:, :2], controls[3:, :2])
    np.testing.assert_allclose(result.controller_points_m[5:, 2:], controls[5:, 2:])


def test_conditioned_action_rejects_missing_and_spatially_implausible_groups() -> None:
    controls = _controllers()
    active = np.zeros((6, 2), dtype=bool)
    active[2:, 0] = True
    initial = np.array([[1.0, 0.0, 0.0]])

    result = condition_controller_action(
        controls,
        active,
        initial,
        controller_group_size=2,
        maximum_contact_distance_m=0.03,
    )

    assert result.falls_back_to_persistence
    assert result.controller_points_m.shape == (6, 0, 3)
    assert result.contact_active.shape == (6, 0)
    assert result.source_group_indices == ()


def test_conditioned_action_keeps_only_admissible_original_group_indices() -> None:
    controls = _controllers()
    active = np.ones((6, 2), dtype=bool)
    initial = np.array([[0.2, 0.0, 0.0]])

    result = condition_controller_action(
        controls,
        active,
        initial,
        controller_group_size=2,
        maximum_contact_distance_m=0.01,
    )

    assert result.source_group_indices == (1,)
    assert result.onset_frames == (0,)
    np.testing.assert_allclose(result.controller_points_m, controls[:, 2:])


def test_contact_conditioned_action_artifact_roundtrip(tmp_path) -> None:
    controls = _controllers()
    active = np.zeros((6, 2), dtype=bool)
    active[2:, 0] = True
    active[4:, 1] = True
    initial = np.array([[0.02, 0.0, 0.0], [0.24, 0.0, 0.0]])
    action = condition_controller_action(
        controls,
        active,
        initial,
        controller_group_size=2,
        maximum_contact_distance_m=0.01,
    )
    archive = tmp_path / "conditioned_action.npz"
    payload = write_contact_conditioned_action_artifact(
        archive,
        action,
        object_id="object",
        episode_id=3,
        source_controller_sha256="1" * 64,
        contact_model_result_sha256="2" * 64,
        information_boundary={
            "known_future_robot_action_used": True,
            "future_object_observations_used": False,
            "target_tactile_used": False,
        },
    )
    metadata = tmp_path / "conditioned_action.json"
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_contact_conditioned_action_artifact(
        json.loads(metadata.read_text(encoding="utf-8"))
    )

    np.testing.assert_array_equal(
        loaded.controller_points_m, action.controller_points_m
    )
    np.testing.assert_array_equal(loaded.contact_active, action.contact_active)
    assert loaded.source_group_indices == action.source_group_indices
    assert loaded.onset_frames == action.onset_frames


def test_conditioned_action_rejects_mismatched_group_schedule() -> None:
    with pytest.raises(ValueError, match="contact schedule"):
        condition_controller_action(
            _controllers(),
            np.zeros((6, 1), dtype=bool),
            np.zeros((1, 3)),
            controller_group_size=2,
            maximum_contact_distance_m=0.03,
        )


def test_controller_spring_groups_follow_packed_controller_vertices() -> None:
    springs = np.array(
        [
            [0, 1],
            [1, 2],
            [3, 0],
            [2, 5],
            [4, 1],
            [0, 6],
        ]
    )

    groups = controller_spring_group_indices(
        springs,
        num_object_springs=2,
        controller_vertex_start=3,
        controller_point_count=4,
        controller_group_size=2,
        retained_group_count=2,
    )

    np.testing.assert_array_equal(groups, np.array([0, 1, 0, 1]))


def test_controller_spring_groups_reject_unattached_group() -> None:
    with pytest.raises(ValueError, match="not every retained"):
        controller_spring_group_indices(
            np.array([[0, 1], [2, 0], [3, 1]]),
            num_object_springs=1,
            controller_vertex_start=2,
            controller_point_count=4,
            controller_group_size=2,
            retained_group_count=2,
        )
