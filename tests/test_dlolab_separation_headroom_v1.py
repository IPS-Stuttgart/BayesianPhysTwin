from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_separation_headroom_v1 import (
    ACTION_ANGLES_DEG,
    MEMORY_NAMES,
    UNIQUE_ACTION_COUNT,
    WORLD_ANGLES_DEG,
    action_bank,
    development_metrics,
    native_qa,
    native_reward,
    protocol,
    task,
    worlds,
)


def test_worlds_and_actions_are_geometry_only_and_deterministic() -> None:
    assert [world["rotation_deg"] for world in worlds()] == list(WORLD_ANGLES_DEG)
    assert len(set(WORLD_ANGLES_DEG)) == 9
    actions = action_bank()
    assert actions.shape == (11, 6, 12)
    assert np.array_equal(actions[9], actions[1])
    assert np.array_equal(actions[10], actions[8])
    assert np.array_equal(actions[:, :2], np.repeat(actions[:1, :2], 11, axis=0))
    assert np.all(actions[..., 6:] == 0)
    assert len(ACTION_ANGLES_DEG) == 8


def test_protocol_is_bounded_development_only() -> None:
    value = protocol()
    assert value["role"] == (
        "bounded_public_simulator_development_screen_not_scientific_evidence"
    )
    assert value["worlds"] == worlds()
    assert value["unique_actions"] == UNIQUE_ACTION_COUNT
    assert not value["source_transfer_automatically_authorized"]
    assert not value["prospective_replication_automatically_authorized"]
    assert not value["retry_authorized"]
    assert not value["protected_data_read"]
    assert not value["gpu_work"]


def test_tasks_are_exactly_registered_worlds() -> None:
    assert [task(index)["world"] for index in range(9)] == worlds()
    with pytest.raises(ValueError, match="unregistered"):
        task(9)
    with pytest.raises(ValueError, match="unregistered"):
        task(True)


def test_native_reward_is_symmetric_nearest_point_distance() -> None:
    x = np.linspace(0.0, 0.29, 30)
    a = np.zeros((2, 30, 3), dtype=np.float64)
    b = np.zeros_like(a)
    a[..., 0] = x
    b[..., 0] = x
    b[0, :, 1] = 0.05
    b[1, :, 1] = 0.12
    assert native_reward(a, b).tolist() == pytest.approx([0.10, 0.24])


def _qualified_bundle():
    a = np.zeros((11, 30, 3), dtype=np.float64)
    b = np.zeros_like(a)
    b[..., 1] = 0.1
    arrays = {
        "controls": action_bank(),
        "prefix_rope_a_m": np.zeros((2, 11, 5, 3), dtype=np.float64),
        "prefix_rope_b_m": np.zeros((2, 11, 5, 3), dtype=np.float64),
        "final_rope_a_m": a,
        "final_rope_b_m": b,
        "final_rope_a_velocity_m_s": np.zeros_like(a),
        "final_rope_b_velocity_m_s": np.zeros_like(b),
        "final_gripper_a_m": np.zeros((11, 3), dtype=np.float64),
        "final_gripper_b_m": np.zeros((11, 3), dtype=np.float64),
        "joint_targets": np.zeros((11, 61, 18), dtype=np.float32),
    }
    arrays.update({name: np.zeros(1, dtype=np.float64) for name in MEMORY_NAMES})
    world = worlds()[0]
    native = {
        "native_steps": 1200,
        "native_final_reward_m": native_reward(a, b).tolist(),
        "world": world,
        "world_realization": {
            "rotation_deg": world["rotation_deg"],
            "maximum_rotation_realization_error_m": 0.0,
        },
        "measurements": {
            "maximum_common_prefix_error_m": 0.0,
            "maximum_duplicate_coordinate_error_m": 0.0,
            "maximum_segment_relative_error": 0.0,
            "minimum_rod_height_m": 0.0,
            "maximum_attachment_distance_m": 0.0,
        },
    }
    return arrays, native, world


def test_native_qa_rederives_reward_and_world_rotation() -> None:
    arrays, native, world = _qualified_bundle()
    assert native_qa(arrays, native, world)["passed"] is True
    native["world_realization"]["rotation_deg"] = 0.0
    value = native_qa(arrays, native, world)
    assert value["passed"] is False
    assert value["checks"]["registered_world_rotation"] is False


def test_informative_bank_can_pass_headroom_gate() -> None:
    rewards = np.full((9, UNIQUE_ACTION_COUNT), 0.15, dtype=np.float64)
    rewards[:, 0] = 0.10
    for world in range(9):
        rewards[world, 1 + min(world, 7)] = 0.20
    value = development_metrics(rewards)
    assert value["development_gate_passed"] is True
    assert value["distinct_oracle_actions"] == 8
    assert value["adjusted_oracle_headroom_m"] > 0.03
    assert not value["source_transfer_automatically_authorized"]


def test_dominant_action_fails_without_authorizing_transfer() -> None:
    rewards = np.full((9, UNIQUE_ACTION_COUNT), 0.10, dtype=np.float64)
    rewards[:, 1] = 0.20
    value = development_metrics(rewards)
    assert value["development_gate_passed"] is False
    assert value["distinct_oracle_actions"] == 1
    assert value["oracle_headroom_m"] == pytest.approx(0.0)
    assert not value["source_transfer_automatically_authorized"]
