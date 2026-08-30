import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_coiling_query_competence_v1 import (
    ACTION_NAMES,
    MEMORY_NAMES,
    action_bank,
    native_qa,
    native_reward,
    protocol,
    protocol_v1_1,
    source_value,
    task,
    worlds,
)


def test_action_bank_is_geometry_fixed_common_prefix_and_duplicate_safe():
    controls = action_bank()
    assert controls.shape == (8, 10, 6)
    assert controls.dtype == np.float64
    assert np.linalg.norm(controls[..., :3], axis=-1).max() < 0.1
    assert np.array_equal(controls[:, :4], np.repeat(controls[:1, :4], 8, 0))
    assert np.array_equal(controls[1], controls[7])
    assert not np.array_equal(controls[0], controls[1])
    assert np.all(controls[..., 3:] == 0)


def test_protocol_is_development_only_and_complete():
    value = protocol()
    assert value["role"].endswith("not_scientific_evidence")
    assert value["worlds"] == worlds()
    assert value["action_names"] == list(ACTION_NAMES)
    assert not value["prospective_replication_automatically_authorized"]
    assert not value["retry_authorized"]
    assert not value["protected_data_read"]
    assert not value["gpu_work"]
    assert len({(x["bending_E"], x["twisting_G"]) for x in worlds()}) == 9


def test_v1_1_changes_only_replacement_metadata():
    original = protocol()
    replacement = protocol_v1_1()
    assert replacement["replacement"]["parent_native_scene_steps_completed"] == 0
    assert not replacement["replacement"]["scientific_fields_changed"]
    assert not replacement["replacement"]["further_replacement_authorized"]
    original.pop("schema")
    replacement.pop("schema")
    replacement.pop("replacement")
    assert replacement == original


def test_tasks_are_exactly_the_registered_worlds():
    assert [task(index)["world"] for index in range(9)] == worlds()
    with pytest.raises(ValueError, match="unregistered"):
        task(9)
    with pytest.raises(ValueError, match="unregistered"):
        task(True)


def test_native_reward_reproduces_public_formula():
    positions = np.zeros((2, 60, 3), dtype=np.float64)
    positions[0, :, 2] = 0.15
    positions[1, :, 0] = 0.1
    positions[1, :, 2] = 0.15
    value = native_reward(positions)
    assert value[0] == 1.0
    assert value[1] == pytest.approx(np.exp(-0.6))


def _qualified_bundle():
    final = np.zeros((8, 60, 3), dtype=np.float64)
    final[..., 2] = 0.15
    arrays = {
        "controls": action_bank(),
        "prefix_positions_m": np.zeros((3, 8, 5, 3), dtype=np.float64),
        "final_positions_m": final,
        "final_velocities_m_s": np.zeros((8, 60, 3), dtype=np.float64),
        "final_gripper_positions_m": np.zeros((8, 3), dtype=np.float64),
        "joint_targets": np.zeros((8, 101, 9), dtype=np.float32),
    }
    arrays.update({name: np.zeros(1, dtype=np.float64) for name in MEMORY_NAMES})
    world = worlds()[0]
    native = {
        "native_steps": 2000,
        "native_final_reward": native_reward(final).tolist(),
        "world": world,
        "world_realization": {
            "bending": [world["bending_E"]] * 8,
            "twisting": [world["twisting_G"]] * 8,
        },
        "measurements": {
            "maximum_common_prefix_error_m": 0.0,
            "maximum_duplicate_coordinate_error_m": 0.0,
            "maximum_segment_relative_error": 0.0,
            "minimum_rod_height_m": 0.0,
            "maximum_attachment_distance_m": 0.0,
            "maximum_fixed_cone_error_m": 0.0,
        },
    }
    return arrays, native, world


def test_native_qa_rederives_reward_and_all_gates():
    arrays, native, world = _qualified_bundle()
    value = native_qa(arrays, native, world)
    assert value["passed"]
    assert all(value["checks"].values())
    native["native_final_reward"][0] -= 0.01
    assert not native_qa(arrays, native, world)["checks"]["native_final_reward"]


def _synthetic_value_bank(informative: bool):
    prefix = np.zeros((9, 3, 5, 3), dtype=np.float64)
    if informative:
        prefix[:, :, :, 0] = np.arange(9)[:, None, None] * 0.05
    reward = np.full((9, 7), 0.5, dtype=np.float64)
    reward[:, 0] = 0.4
    reward[:, 1] = 0.6
    for world in range(9):
        reward[world, 1 + world % 3] = 0.75
    return prefix, reward


def test_informative_prefix_can_pass_frozen_development_gate():
    prefix, reward = _synthetic_value_bank(True)
    value = source_value(prefix, reward)
    assert value["development_gate_passed"]
    assert value["distinct_oracle_actions"] == 3
    assert value["bayes_gain_over_best_fixed"] > 0.05
    assert not value["prospective_replication_automatically_authorized"]


def test_uninformative_prefix_fails_without_promoting_itself():
    prefix, reward = _synthetic_value_bank(False)
    value = source_value(prefix, reward)
    assert not value["development_gate_passed"]
    assert value["bayes_gain_over_best_fixed"] == pytest.approx(0, abs=1e-12)
    assert not value["prospective_replication_automatically_authorized"]
