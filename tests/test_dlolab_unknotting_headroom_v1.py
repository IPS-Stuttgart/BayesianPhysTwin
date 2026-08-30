from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_unknotting_headroom_v1 import (
    ACTION_OFFSETS_DEG,
    BASE_PULL_ANGLE_DEG,
    MEMORY_NAMES,
    UNIQUE_ACTION_COUNT,
    WORLD_ROTATIONS_DEG,
    action_bank,
    development_metrics,
    native_qa,
    native_reward,
    protocol,
    task,
    worlds,
)


def test_worlds_and_actions_are_geometry_only_and_deterministic() -> None:
    assert [world["rotation_deg"] for world in worlds()] == list(WORLD_ROTATIONS_DEG)
    actions = action_bank()
    assert actions.shape == (11, 6, 12)
    assert np.array_equal(actions[9], actions[1])
    assert np.array_equal(actions[10], actions[8])
    assert np.array_equal(actions[:, :2], np.repeat(actions[:1, :2], 11, axis=0))
    assert np.all(actions[..., 6:] == 0)
    direction = actions[1, 2, :2]
    assert np.degrees(np.arctan2(direction[1], direction[0])) == pytest.approx(
        BASE_PULL_ANGLE_DEG + ACTION_OFFSETS_DEG[0]
    )


def test_protocol_is_bounded_development_only() -> None:
    value = protocol()
    assert value["role"] == (
        "bounded_public_simulator_development_screen_not_scientific_evidence"
    )
    assert value["task"] == "unknotting"
    assert value["worlds"] == worlds()
    assert value["unique_actions"] == UNIQUE_ACTION_COUNT
    assert value["runtime_preflight_before_attempt_consumption"]
    assert value["external_write_once_attempt_ledger"]
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


def test_native_reward_is_rotation_invariant_and_bounded() -> None:
    x = np.linspace(-0.25, 0.25, 50)
    rope = np.zeros((2, 50, 3), dtype=np.float64)
    rope[..., 0] = x
    angle = np.deg2rad(37.0)
    rotated = rope.copy()
    rotated[..., 0] = np.cos(angle) * rope[..., 0]
    rotated[..., 1] = np.sin(angle) * rope[..., 0]
    reward = native_reward(rope)
    assert np.all((reward >= 0) & (reward <= 1))
    assert native_reward(rotated).tolist() == pytest.approx(reward.tolist())


def _qualified_bundle():
    x = np.linspace(-0.25, 0.25, 50)
    rope = np.zeros((11, 50, 3), dtype=np.float64)
    rope[..., 0] = x
    arrays = {
        "controls": action_bank(),
        "prefix_rope_m": np.zeros((2, 11, 5, 3), dtype=np.float64),
        "final_rope_m": rope,
        "final_rope_velocity_m_s": np.zeros_like(rope),
        "final_gripper_a_m": np.zeros((11, 3), dtype=np.float64),
        "final_gripper_b_m": np.zeros((11, 3), dtype=np.float64),
        "joint_targets": np.zeros((11, 61, 18), dtype=np.float32),
    }
    arrays.update({name: np.zeros(1, dtype=np.float64) for name in MEMORY_NAMES})
    world = worlds()[0]
    native = {
        "native_steps": 1200,
        "native_final_reward": native_reward(rope).tolist(),
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
            "maximum_attachment_offset_drift_m": 0.0,
        },
    }
    return arrays, native, world


def test_native_qa_rederives_reward_and_attachment_drift() -> None:
    arrays, native, world = _qualified_bundle()
    assert native_qa(arrays, native, world)["passed"] is True
    native["measurements"]["maximum_attachment_offset_drift_m"] = 0.0011
    value = native_qa(arrays, native, world)
    assert value["passed"] is False
    assert value["checks"]["material_attachment"] is False


def test_informative_bank_can_pass_headroom_gate() -> None:
    rewards = np.full((9, UNIQUE_ACTION_COUNT), 0.10, dtype=np.float64)
    rewards[:, 0] = 0.02
    for world in range(9):
        rewards[world, 1 + min(world, 7)] = 0.50
    value = development_metrics(rewards)
    assert value["development_gate_passed"] is True
    assert value["distinct_oracle_actions"] == 8
    assert value["adjusted_oracle_headroom"] > 0.30
    assert not value["source_transfer_automatically_authorized"]


def test_dominant_action_fails_without_authorizing_transfer() -> None:
    rewards = np.full((9, UNIQUE_ACTION_COUNT), 0.10, dtype=np.float64)
    rewards[:, 1] = 0.50
    value = development_metrics(rewards)
    assert value["development_gate_passed"] is False
    assert value["distinct_oracle_actions"] == 1
    assert value["oracle_headroom"] == pytest.approx(0.0)
    assert not value["source_transfer_automatically_authorized"]
