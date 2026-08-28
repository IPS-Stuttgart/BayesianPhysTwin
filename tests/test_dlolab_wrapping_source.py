import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_wrapping_native import observe, run_world
from bayesian_phystwin_experiments.dlolab_wrapping_source import (
    ACTION_NAMES,
    MEMORY_NAMES,
    N_ENVS,
    NATIVE_STEPS,
    POSTS,
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
    validate_world,
    winding_components,
    worlds,
)

SPEC = importlib.util.spec_from_file_location(
    "wrapping_checker",
    Path(__file__).resolve().parents[1] / "scripts/verify_dlolab_wrapping_source.py",
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def circle(center=(0.6, 0, 0.012), radius=0.14):
    angle = np.arange(50) * 2 * np.pi / 50 + 0.03
    return (
        np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros(50)))
        + center
    )


def fixture():
    data = {
        key: np.zeros(shape, dtype=np.float64) for key, shape in TRACE_SHAPES.items()
    }
    loop = circle()
    data["rod_pos_m"][:] = loop
    data["gripper_pos_m"][:] = loop[[17, 33]]
    data["post_pos_m"][:] = POSTS
    data.update({key: np.zeros((N_ENVS, 1)) for key in MEMORY_NAMES})
    data.update(
        controls=action_bank(),
        joint_targets=np.zeros((N_ENVS, 111, 18)),
        initial_rod_pos_m=np.tile(loop[None], (N_ENVS, 1, 1)),
    )
    final = native_reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    cumulative = np.zeros(N_ENVS, dtype=np.float32)
    for _ in range(110):
        cumulative += final.astype(np.float32) + np.float32(1)
    native = {
        "native_steps": NATIVE_STEPS,
        "native_final_reward": final.astype(np.float32).tolist(),
        "native_cumulative_reward": cumulative.tolist(),
        "world": worlds()[4],
        "world_realization": {"bending": [1e4] * N_ENVS, "stretching": [1e5] * N_ENVS},
        "device": "cpu",
        "twisting_stiffness_zero_preserved": True,
        "runtime_camera_rendered": False,
        "native_source_modified": False,
    }
    return data, native


def test_actions_have_fixed_common_prefix_and_duplicate():
    bank = action_bank()
    assert bank.shape == (9, 11, 12) and bank.dtype == np.float64
    assert np.all(bank[:, :3] == bank[1, :3])
    assert np.array_equal(bank[1], bank[8])
    assert np.unique(bank.reshape(9, -1), axis=0).shape[0] == 8
    assert np.all(bank[0, 3:] == 0) and np.all(bank[..., 6:] == 0)
    assert (
        np.linalg.norm(bank[..., :6].reshape(9, 11, 2, 3), axis=-1).max()
        <= 0.1000000001
    )
    assert np.allclose(bank[1, :, :6].sum(axis=0), [-0.37, 0, 0.02, -0.37, 0, 0.02])


def test_frozen_public_native_and_claim_boundaries():
    p = protocol()
    assert p["native_reward_controller_scene_solver_unchanged"]
    assert p["native_extensible_closed_loop_preserved"]
    assert not p["twisting_parameter_varied"]
    for key in (
        "gpu_work",
        "protected_data_read",
        "fresh_method_evaluation_authorized",
        "published_controller_comparator_available",
        "new_recordings",
        "retry_authorized",
    ):
        assert not p[key]
    assert p["native_trajectories"] == 99 and len(ACTION_NAMES) == 9
    assert task(0)["world"] == task(1)["world"] == task(2)["world"] == worlds()[4]
    assert sorted(
        task(i)["world"]["index"] for i in range(11) if i not in (1, 2)
    ) == list(range(9))


@pytest.mark.parametrize("index", [-1, 11, True, 1.0])
def test_unregistered_task_rejected(index):
    with pytest.raises(ValueError, match="unregistered"):
        task(index)


@pytest.mark.parametrize("mutation", ["index", "bool", "string", "parameter", "extra"])
def test_unregistered_world_rejected_before_native_import(tmp_path, mutation):
    world = worlds()[4]
    if mutation == "index":
        world["index"] = 4.0
    elif mutation == "bool":
        world["index"] = True
    elif mutation == "string":
        world["bending_E"] = "10000"
    elif mutation == "parameter":
        world["stretching_K"] = 2e5
    else:
        world["extra"] = 0
    with pytest.raises(ValueError, match="unregistered"):
        validate_world(world)
    with pytest.raises(ValueError, match="unregistered"):
        run_world(tmp_path / "absent", tmp_path / "not_created", world)
    assert not (tmp_path / "not_created").exists()


def test_winding_uses_closed_loop_and_both_orientations():
    points = circle((0, 0, 0), 1)[None]
    posts = np.asarray([[[0, 0, 0], [0.3, 0.2, 0], [2, 0, 0]]])
    turns, penalty = winding_components(points, posts)
    assert np.allclose(turns, [[1, 1, 0]], atol=1e-14)
    assert np.all(penalty >= 0)
    reverse_turns, _ = winding_components(points[:, ::-1], posts)
    assert np.allclose(reverse_turns, -turns, atol=1e-14)
    assert np.allclose(
        native_reward(points[:, ::-1], posts), native_reward(points, posts), atol=1e-14
    )
    assert np.allclose(
        checker.reward(points, posts), native_reward(points, posts), atol=1e-14
    )


def test_proximity_penalty_uses_3d_distance_not_xy_only():
    points = circle((0, 0, 0), 0.14)[None]
    posts = np.zeros((1, 3, 3))
    _, flat = winding_components(points, posts)
    posts[..., 2] = 0.1
    _, high = winding_components(points, posts)
    assert np.all(high > flat)


@pytest.mark.parametrize("shape", [(50, 3), (9, 49, 3), (9, 50, 2)])
def test_reward_rejects_wrong_geometry(shape):
    with pytest.raises(ValueError, match="geometry"):
        native_reward(np.zeros(shape), np.zeros((9, 3, 3)))


def test_second_reward_formula_matches_on_random_polygons():
    rng = np.random.default_rng(43)
    points = rng.normal(0.1, 0.05, (3, 9, 50, 3))
    posts = rng.normal(0.1, 0.05, (3, 9, 3, 3))
    assert np.allclose(
        checker.reward(points, posts), native_reward(points, posts), rtol=0, atol=1e-13
    )


def test_prefix_adapter_refuses_future_and_preserves_identities():
    data = np.arange(PREFIX_STEPS * 9 * 50 * 3).reshape(PREFIX_STEPS, 9, 50, 3)
    selected = prefix_observation(data)
    assert selected.shape == (3, 5, 3)
    assert np.array_equal(selected[2, 3], data[599, 1, 41])
    selected[:] = -1
    assert np.all(data >= 0)
    with pytest.raises(ValueError, match="prefix"):
        prefix_observation(np.zeros((PREFIX_STEPS + 1, 9, 50, 3)))


def test_native_qa_roundtrip_accepts_ordinary_negative_reward():
    data, native = fixture()
    assert np.all(np.asarray(native["native_final_reward"]) < 0)
    first = native_qa(data, native, worlds()[4])
    assert first["passed"]
    assert checker.compare(checker.qa(data, native, protocol()), first) < 1e-10


@pytest.mark.parametrize(
    "mutation",
    [
        "controls",
        "world",
        "memory",
        "material",
        "steps",
        "gpu",
        "render",
        "native",
        "twist",
        "nonfinite",
        "reward_shape",
    ],
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
        native["world_realization"]["stretching"][3] *= 2
    elif mutation == "steps":
        native["native_steps"] -= 1
    elif mutation == "gpu":
        native["device"] = "cuda"
    elif mutation == "render":
        native["runtime_camera_rendered"] = True
    elif mutation == "native":
        native["native_source_modified"] = True
    elif mutation == "twist":
        native["twisting_stiffness_zero_preserved"] = False
    elif mutation == "nonfinite":
        native["native_final_reward"][0] = float("nan")
    else:
        native["native_final_reward"] = [0]
    with pytest.raises(ValueError):
        native_qa(data, native, worlds()[4])


@pytest.mark.parametrize(
    "mutation,check",
    [
        ("post", "fixed_posts"),
        ("stretch", "finite_extensible_segments"),
        ("prefix", "common_prefix"),
        ("duplicate", "duplicate_positions"),
        ("reward", "native_final_reward"),
        ("attachment", "attached_material_points"),
        ("floor", "above_floor"),
        ("cumulative", "native_cumulative_reward"),
        ("penalty", "ordinary_native_success"),
    ],
)
def test_native_physical_gates_fail_closed(mutation, check):
    data, native = fixture()
    if mutation == "post":
        data["post_pos_m"][:, :, 0, 0] += 0.001
    elif mutation == "stretch":
        data["rod_pos_m"][:, :, 15, 0] += 0.1
    elif mutation == "prefix":
        data["rod_pos_m"][200, 2, 12, 0] += 0.0001
    elif mutation == "duplicate":
        data["rod_pos_m"][-1, 8, :, 0] += 0.002
    elif mutation == "reward":
        native["native_final_reward"][0] += 0.01
    elif mutation == "attachment":
        data["gripper_pos_m"][:, :, 0, 0] += 0.02
    elif mutation == "floor":
        data["rod_pos_m"][800, :, :, 2] = -0.02
    elif mutation == "cumulative":
        native["native_cumulative_reward"][0] -= 110
    else:
        native["native_final_reward"][0] = -99
    result = native_qa(data, native, worlds()[4])
    assert not result["passed"] and not result["checks"][check]


def test_collapsed_initial_loop_rejected():
    data, native = fixture()
    data["initial_rod_pos_m"][:, 0] = data["initial_rod_pos_m"][:, 1]
    with pytest.raises(ValueError, match="collapsed"):
        native_qa(data, native, worlds()[4])


def test_repeatability_covers_all_actions_not_only_nominal():
    data, native = fixture()
    rows = [data, data, {k: v.copy() for k, v in data.items()}]
    rewards = np.asarray([native["native_final_reward"]] * 3)
    result = repeat_qa(rows, rewards)
    assert result["passed"] and not result["population_bound_claimed"]
    rows[2]["gripper_pos_m"][-1, 6, 1, 0] += 0.002
    assert not repeat_qa(rows, rewards)["passed"]


def test_no_information_cannot_improve_best_fixed():
    rewards = np.tile([0.2, 0.5, 0.4, 0.3, 0.45, 0.3, 0.25, 0.1], (9, 1))
    result = information_value(np.zeros((9, 3, 5, 3)), rewards)
    assert result["best_fixed_action"] == 1 and result["oracle_reward"] == 0.5
    assert result["arms"]["bias_aware_bayes"][
        "expected_native_final_reward"
    ] == pytest.approx(0.5)
    assert not result["source_gate_passed"]


def test_identifiable_materials_recover_oracle_not_bayes_map_gain():
    prefix = np.repeat((np.arange(9) * 0.1)[:, None], 45, axis=1).reshape(9, 3, 5, 3)
    rewards = np.full((9, 8), 0.3)
    rewards[np.arange(9), 1 + np.arange(9) % 3] = 0.8
    result = information_value(prefix, rewards)
    for name in ("bias_aware_bayes", "bias_aware_map"):
        assert result["arms"][name]["expected_native_final_reward"] == pytest.approx(
            0.8
        )
    assert result["checks"]["adjusted_bayes_gain_over_best_fixed"]
    assert not result["checks"]["adjusted_bayes_gain_over_map"]
    assert not result["method_promotion_authorized"]


def test_sherman_morrison_matches_production_belief_value():
    rng = np.random.default_rng(44)
    prefix = rng.normal(0.1, 0.001, (9, 3, 5, 3))
    rewards = rng.uniform(-0.2, 0.9, (9, 8))
    assert (
        checker.compare(
            checker.belief_value(prefix, rewards, protocol()),
            information_value(prefix, rewards),
        )
        < 1e-10
    )


def test_ambiguous_materials_use_expected_reward_not_map():
    prefix = np.zeros((9, 3, 5, 3))
    prefix[..., 0] = (0.04 * (np.arange(9) // 3))[:, None, None]
    rewards = np.full((9, 8), -0.6)
    rewards[:, 0] = -0.2
    for world in range(9):
        rewards[world, 1 + world // 3] = 0.7
        rewards[world, 4 + world % 3] = 0.99
    result = information_value(prefix, rewards)
    bayes = result["arms"]["bias_aware_bayes"]["expected_native_final_reward"]
    point = result["arms"]["bias_aware_map"]["expected_native_final_reward"]
    assert bayes > 0.69 and bayes - point > 0.6
    assert result["checks"]["adjusted_bayes_gain_over_map"]
    assert result["checks"]["adjusted_bayes_gain_over_best_fixed"]
    assert (
        checker.compare(checker.belief_value(prefix, rewards, protocol()), result)
        < 1e-10
    )


@pytest.mark.parametrize("mutation", ["missing", "future", "penalty", "nonfinite"])
def test_incomplete_or_failure_source_bank_rejected(mutation):
    prefix, rewards = np.zeros((9, 3, 5, 3)), np.zeros((9, 8))
    if mutation == "missing":
        rewards = rewards[:-1]
    elif mutation == "future":
        prefix = np.zeros((9, 4, 5, 3))
    elif mutation == "penalty":
        rewards[0, 1] = -99
    else:
        prefix[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="ordinary"):
        information_value(prefix, rewards)


def test_verifier_rejects_forged_decision_and_record(tmp_path):
    with pytest.raises(ValueError, match="value differs"):
        checker.compare({"passed": True}, {"passed": False})
    path = tmp_path / "record.json"
    path.write_text(
        '{"value":2,"artifact_id":"' + checker.canonical({"value": 1}) + '"}'
    )
    with pytest.raises(ValueError, match="digest differs"):
        checker.read(path)


def test_observer_reports_native_two_robot_shapes_and_rejects_nan():
    def point():
        return np.zeros((9, 3))

    env = SimpleNamespace(
        rope=SimpleNamespace(
            get_all_verts=lambda: np.zeros((9, 50, 3)),
            get_all_vels=lambda: np.zeros((9, 50, 3)),
        ),
        **{f"post{i}": SimpleNamespace(get_pos=point) for i in range(1, 4)},
        **{f"c{i}": SimpleNamespace(ef=SimpleNamespace(get_pos=point)) for i in (1, 2)},
        **{
            f"franka{i}": SimpleNamespace(get_qpos=lambda: np.zeros((9, 9)))
            for i in (1, 2)
        },
    )
    result = observe(env)
    assert result["post_pos_m"].shape == (9, 3, 3)
    assert result["gripper_pos_m"].shape == (9, 2, 3)
    assert result["robot_qpos"].shape == (9, 18)
    env.post1.get_pos = lambda: np.full((9, 3), np.nan)
    with pytest.raises(RuntimeError, match="nonfinite"):
        observe(env)
