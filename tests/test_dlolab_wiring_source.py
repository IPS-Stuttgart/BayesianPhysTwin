import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_wiring_source import (
    ACTION_NAMES,
    MEMORY_NAMES,
    NATIVE_STEPS,
    PREFIX_STEPS,
    TRACE_SHAPES,
    action_bank,
    information_value,
    native_qa,
    native_reward,
    prefix_observation,
    protocol,
    repeat_qa,
    task,
    worlds,
)


def fixture():
    data = {
        key: np.zeros(shape, dtype=np.float64) for key, shape in TRACE_SHAPES.items()
    }
    line = np.column_stack(
        (0.3 + 0.02 * np.arange(30), np.full(30, 0.05), np.full(30, 0.012))
    )
    data["rod_pos_m"][:] = line
    data["gripper_pos_m"][:] = line[3]
    data["post_pos_m"][:] = [[0.28, 0.14, 0.02], [0.1, 0.275, 0.02]]
    data["hidden_post_pos_m"][:] = [
        [[0.28, 0.14, z] for z in (-0.02, 0, 0.02)],
        [[0.1, 0.275, z] for z in (-0.02, 0, 0.02)],
    ]
    data.update({key: np.zeros((8, 1)) for key in MEMORY_NAMES})
    data.update(
        controls=action_bank(),
        joint_targets=np.zeros((8, 91, 9)),
        target_pos_m=line.copy(),
    )
    final = native_reward(data["rod_pos_m"][-1], line)
    cumulative = np.zeros(8, dtype=np.float32)
    for _ in range(90):
        cumulative += final.astype(np.float32)
    native = {
        "native_steps": NATIVE_STEPS,
        "native_final_reward": final.astype(np.float32).tolist(),
        "native_cumulative_reward": cumulative.tolist(),
        "world": worlds()[4],
        "world_realization": {"bending": [1e4] * 8, "twisting": [1e3] * 8},
    }
    return data, native


def test_actions_have_fixed_common_prefix_and_duplicate():
    bank = action_bank()
    assert bank.shape == (8, 9, 6)
    assert bank.dtype == np.float64
    assert np.all(bank[:, :3] == bank[1, :3])
    assert np.array_equal(bank[1], bank[7])
    assert np.unique(bank.reshape(8, -1), axis=0).shape[0] == 7
    assert np.all(bank[0, 3:] == 0)
    assert np.linalg.norm(bank[:, :, :3], axis=-1).max() <= 0.1
    assert np.all(bank[..., 2:] == 0)


def test_frozen_public_native_and_claim_boundaries():
    p = protocol()
    assert p["native_reward_controller_scene_solver_unchanged"]
    assert not p["stretching_parameter_varied"]
    assert not p["gpu_work"] and not p["protected_data_read"]
    assert not p["fresh_method_evaluation_authorized"]
    assert not p["published_controller_comparator_available"]
    assert p["native_trajectories"] == 88
    assert len(ACTION_NAMES) == 8
    assert task(0)["world"] == task(1)["world"] == task(2)["world"] == worlds()[4]
    assert sorted(
        task(i)["world"]["index"] for i in range(11) if i not in (1, 2)
    ) == list(range(9))


@pytest.mark.parametrize("index", [-1, 11, True, 1.0])
def test_task_rejects_unregistered_identity(index):
    with pytest.raises(ValueError, match="unregistered"):
        task(index)


def test_reward_matches_independent_loop_and_native_goal():
    line = np.column_stack(
        (0.198 + 0.02 * (np.arange(30) - 15), np.full(30, 0.198), np.full(30, 0.01))
    )
    assert native_reward(line[None], line)[0] == 1
    moved = line + [0.03, -0.04, 0.05]
    pair = [[np.linalg.norm(a - b) for b in line] for a in moved]
    cost = np.mean([min(row) for row in pair]) + np.mean(np.min(pair, axis=0))
    cost += 0.02 + 5 * (0.05 - 0.02)
    assert native_reward(moved[None], line)[0] == pytest.approx(
        np.exp(-cost), abs=1e-14
    )


@pytest.mark.parametrize("shape", [(30, 3), (8, 29, 3), (8, 30, 2)])
def test_reward_rejects_wrong_geometry(shape):
    with pytest.raises(ValueError, match="geometry"):
        native_reward(np.zeros(shape), np.zeros((30, 3)))


def test_prefix_observation_cannot_receive_future():
    prefix = np.arange(PREFIX_STEPS * 8 * 30 * 3).reshape(PREFIX_STEPS, 8, 30, 3)
    selected = prefix_observation(prefix)
    assert selected.shape == (3, 5, 3)
    assert np.array_equal(selected[0, 0], prefix[199, 1, 6])
    with pytest.raises(ValueError, match="prefix"):
        prefix_observation(np.zeros((PREFIX_STEPS + 1, 8, 30, 3)))


def test_native_qa_roundtrip():
    data, native = fixture()
    assert native_qa(data, native, worlds()[4])["passed"]


@pytest.mark.parametrize(
    "mutation", ["controls", "world", "memory", "material", "steps"]
)
def test_native_contract_mutations_rejected(mutation):
    data, native = fixture()
    if mutation == "controls":
        data["controls"][1, 0, 0] += 0.001
    elif mutation == "world":
        native["world"] = worlds()[0]
    elif mutation == "memory":
        del data[MEMORY_NAMES[0]]
    elif mutation == "material":
        native["world_realization"]["twisting"][3] *= 2
    else:
        native["native_steps"] -= 1
    with pytest.raises(ValueError):
        native_qa(data, native, worlds()[4])


@pytest.mark.parametrize(
    "mutation,check",
    [
        ("post", "fixed_posts"),
        ("stretch", "segment_length"),
        ("prefix", "common_prefix"),
        ("duplicate", "duplicate_positions"),
        ("reward", "native_final_reward"),
        ("attachment", "attached_material_point"),
    ],
)
def test_native_physical_gates_fail_closed(mutation, check):
    data, native = fixture()
    if mutation == "post":
        data["post_pos_m"][:, :, 0, 0] += 0.001
    elif mutation == "stretch":
        data["rod_pos_m"][:, :, 15, 0] += 0.01
    elif mutation == "prefix":
        data["rod_pos_m"][200, 2, 12, 0] += 0.0001
    elif mutation == "duplicate":
        data["rod_pos_m"][-1, 7, :, 0] += 0.002
    elif mutation == "reward":
        native["native_final_reward"][0] += 0.01
    else:
        data["gripper_pos_m"][:, :, 0] += 0.02
    result = native_qa(data, native, worlds()[4])
    assert not result["passed"] and not result["checks"][check]


def test_repeatability_checks_all_actions_and_is_not_a_population_bound():
    data, native = fixture()
    rows = [{k: v.copy() for k, v in data.items()} for _ in range(3)]
    rewards = np.asarray([native["native_final_reward"]] * 3)
    result = repeat_qa(rows, rewards)
    assert result["passed"] and not result["population_bound_claimed"]
    rows[2]["gripper_pos_m"][1799, 6, 0] += 0.002
    assert not repeat_qa(rows, rewards)["passed"]


def test_no_information_cannot_improve_best_fixed():
    rewards = np.tile([0.2, 0.5, 0.4, 0.3, 0.45, 0.3, 0.25], (9, 1))
    result = information_value(np.zeros((9, 3, 5, 3)), rewards)
    assert result["best_fixed_action"] == 1
    assert result["oracle_reward"] == 0.5
    assert result["arms"]["bias_aware_bayes"][
        "expected_native_final_reward"
    ] == pytest.approx(0.5)
    assert not result["source_gate_passed"]


def test_identifiable_materials_recover_oracle_but_do_not_invent_bayes_map_gain():
    prefix = np.repeat((np.arange(9) * 0.1)[:, None], 45, axis=1).reshape(9, 3, 5, 3)
    rewards = np.full((9, 7), 0.3)
    rewards[np.arange(9), 1 + np.arange(9) % 3] = 0.8
    result = information_value(prefix, rewards)
    assert result["arms"]["bias_aware_bayes"][
        "expected_native_final_reward"
    ] == pytest.approx(0.8)
    assert result["arms"]["bias_aware_map"][
        "expected_native_final_reward"
    ] == pytest.approx(0.8)
    assert result["checks"]["adjusted_bayes_gain_over_best_fixed"]
    assert not result["checks"]["adjusted_bayes_gain_over_map"]
    assert not result["method_promotion_authorized"]
