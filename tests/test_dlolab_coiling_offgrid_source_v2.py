import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_coiling_offgrid_source_v2 import (
    SOURCE_WORLD_ROWS,
    guarded_action_distribution,
    protocol,
    rederive_native_qa,
    source_crossfit,
    source_worlds,
    task,
)
from bayesian_phystwin_experiments.dlolab_coiling_query_competence_v1 import (
    MEMORY_NAMES,
    action_bank,
    native_reward,
)


def test_source_worlds_are_exact_off_grid_and_disjoint_from_opened_world():
    worlds = source_worlds()
    assert len(worlds) == 12
    assert [
        (x["bending_E"], x["twisting_G"], x["offset_x_m"], x["offset_y_m"])
        for x in worlds
    ] == list(SOURCE_WORLD_ROWS)
    assert (
        len(
            {
                (x["bending_E"], x["twisting_G"], x["offset_x_m"], x["offset_y_m"])
                for x in worlds
            }
        )
        == 12
    )
    assert all((x["bending_E"], x["twisting_G"]) != (500.0, 500.0) for x in worlds)
    assert all(-0.05 <= x["offset_x_m"] <= 0.05 for x in worlds)
    assert all(-0.01 <= x["offset_y_m"] <= 0.01 for x in worlds)


def test_protocol_is_source_only_and_cannot_open_prospective_worlds():
    value = protocol()
    assert value["role"] == "source_only_transfer_gate_not_prospective_evidence"
    assert value["source_worlds"] == source_worlds()
    assert value["source_world_count"] == 12
    assert value["shared_prefix_and_observation_policy_unchanged_from_v1"]
    assert value["native_reward_unchanged"]
    assert not value["prospective_worlds_selected"]
    assert not value["prospective_execution_authorized"]
    assert not value["retry_authorized"]
    assert not value["protected_data_read"]


def test_tasks_are_exactly_the_registered_source_worlds():
    assert [task(index)["world"] for index in range(12)] == source_worlds()
    with pytest.raises(ValueError, match="unregistered"):
        task(12)
    with pytest.raises(ValueError, match="unregistered"):
        task(True)


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
    world = source_worlds()[0]
    native = {
        "native_steps": 2000,
        "native_final_reward": native_reward(final).tolist(),
        "world": world,
        "world_realization": {
            "bending": [world["bending_E"]] * 8,
            "twisting": [world["twisting_G"]] * 8,
        },
        "state_realization": {
            "offset_m": [world["offset_x_m"], world["offset_y_m"], 0.0]
        },
        "state_measurements": {"maximum_offset_realization_error_m": 0.0},
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


def test_native_qa_rederives_state_offset_and_emits_builtin_booleans():
    arrays, native, world = _qualified_bundle()
    value = rederive_native_qa(arrays, native, world)
    assert value["passed"] is True
    assert all(type(check) is bool for check in value["checks"].values())
    native["state_realization"] = {"offset_m": [0.0, 0.0, 0.0]}
    value = rederive_native_qa(arrays, native, world)
    assert value["passed"] is False
    assert value["checks"]["registered_state_offset"] is False


def test_ambiguous_posterior_falls_back_exactly_to_fixed_action():
    truth = np.zeros((3, 5, 3), dtype=np.float64)
    particles = np.zeros((4, 3, 5, 3), dtype=np.float64)
    rewards = np.full((4, 7), 0.5, dtype=np.float64)
    rewards[:, 1] = 0.6
    value = guarded_action_distribution(
        truth, particles, rewards, baseline_action=1, seed=271003
    )
    expected = [0.0] * 7
    expected[1] = 1.0
    assert value["action_probabilities"] == expected
    assert value["admission_probability"] == 0.0
    assert value["exact_fallback_probability"] == 1.0


def _informative_bank():
    prefix = np.zeros((12, 3, 5, 3), dtype=np.float64)
    rewards = np.full((12, 7), 0.45, dtype=np.float64)
    rewards[:, 0] = 0.50
    rewards[:, 1] = 0.55
    for world in range(12):
        group = world % 3
        prefix[world, :, :, 0] = group * 0.08
        rewards[world, 2 + group] = 0.65
    return prefix, rewards


def test_informative_off_grid_bank_can_pass_all_source_gates():
    prefix, rewards = _informative_bank()
    value = source_crossfit(prefix, rewards)
    assert value["source_gate_passed"] is True
    assert value["distinct_oracle_actions"] == 3
    assert value["crossfit_guarded_mean_gain"] > 0.05
    assert value["maximum_observation_draw_harm_probability"] == 0.0
    assert value["prospective_execution_authorized"] is False


def test_uninformative_bank_fails_without_selecting_prospective_worlds():
    prefix, rewards = _informative_bank()
    prefix[:] = 0.0
    value = source_crossfit(prefix, rewards)
    assert value["source_gate_passed"] is False
    assert value["crossfit_guarded_mean_gain"] == pytest.approx(0.0, abs=1e-12)
    assert value["prospective_execution_authorized"] is False
