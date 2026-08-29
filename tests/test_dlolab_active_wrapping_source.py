import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_active_wrapping_native import run_world
from bayesian_phystwin_experiments.dlolab_active_wrapping_source import (
    FULL_MACROS,
    FULL_STEPS,
    MEMORY_NAMES,
    N_ENVS,
    N_PROBES,
    OBSERVATION_FRAMES,
    PROBE_MACROS,
    PROBE_NAMES,
    PROBE_SLOT,
    PROBE_STEPS,
    active_decision_gate,
    decision_value,
    full_action_controls,
    native_qa,
    prefix_observation,
    probe_bank_controls,
    probe_information,
    protocol,
    repeat_qa,
    task,
    worlds,
)
from bayesian_phystwin_experiments.dlolab_wrapping_source import POSTS, native_reward

SPEC = importlib.util.spec_from_file_location(
    "active_wrapping_checker",
    Path(__file__).resolve().parents[1]
    / "scripts/verify_dlolab_active_probe_wrapping_source.py",
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

PARTIAL_SPEC = importlib.util.spec_from_file_location(
    "active_wrapping_partial_checker",
    Path(__file__).resolve().parents[1]
    / "scripts/verify_dlolab_active_probe_wrapping_failure.py",
)
assert PARTIAL_SPEC is not None and PARTIAL_SPEC.loader is not None
partial_checker = importlib.util.module_from_spec(PARTIAL_SPEC)
PARTIAL_SPEC.loader.exec_module(partial_checker)


def circle(center=(0.6, 0, 0.012), radius=0.14):
    angle = np.arange(50) * 2 * np.pi / 50 + 0.03
    return (
        np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros(50)))
        + center
    )


def fixture(stage="probe", probe_index=None):
    steps = PROBE_STEPS if stage == "probe" else FULL_STEPS
    controls = (
        probe_bank_controls() if stage == "probe" else full_action_controls(probe_index)
    )
    loop = circle()
    data = {
        "rod_pos_m": np.tile(loop, (steps, N_ENVS, 1, 1)),
        "rod_vel_m_s": np.zeros((steps, N_ENVS, 50, 3)),
        "post_pos_m": np.tile(POSTS, (steps, N_ENVS, 1, 1)),
        "gripper_pos_m": np.tile(loop[[17, 33]], (steps, N_ENVS, 1, 1)),
        "robot_qpos": np.zeros((steps, N_ENVS, 18)),
        "controls": controls,
        "joint_targets": np.zeros((N_ENVS, controls.shape[1] * 10 + 1, 18)),
        "initial_rod_pos_m": np.tile(loop, (N_ENVS, 1, 1)),
    }
    data.update({key: np.zeros((N_ENVS, 1)) for key in MEMORY_NAMES})
    final = native_reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    cumulative = np.zeros(N_ENVS, dtype=np.float32)
    for _ in range(steps // 20):
        cumulative += final.astype(np.float32) + np.float32(1)
    native = {
        "native_steps": steps,
        "native_final_reward": final.tolist(),
        "native_cumulative_reward": cumulative.tolist(),
        "world": worlds()[4],
        "world_realization": {"bending": [1e4] * N_ENVS, "stretching": [1e5] * N_ENVS},
        "stage": stage,
        "probe_index": probe_index,
        "device": "cpu",
        "twisting_stiffness_zero_preserved": True,
        "runtime_camera_rendered": False,
        "native_source_modified": False,
    }
    return data, native


def test_probe_bank_is_distinct_reversible_and_bounded():
    controls = probe_bank_controls()
    assert controls.shape == (N_ENVS, PROBE_MACROS, 12)
    assert controls.dtype == np.float64
    assert np.unique(controls.reshape(N_ENVS, -1), axis=0).shape[0] == N_PROBES
    for slot, probe in enumerate(PROBE_SLOT):
        assert np.array_equal(controls[slot], controls[probe])
    assert np.allclose(
        controls[:, :PROBE_MACROS, :6].sum(axis=1), controls[0, :, :6].sum(axis=0)
    )
    assert (
        np.linalg.norm(
            controls[..., :6].reshape(N_ENVS, PROBE_MACROS, 2, 3), axis=-1
        ).max()
        <= 0.1
    )


@pytest.mark.parametrize("probe", range(N_PROBES))
def test_full_bank_preserves_prefix_duplicate_and_continuation_count(probe):
    controls = full_action_controls(probe)
    assert controls.shape == (N_ENVS, FULL_MACROS, 12)
    assert np.all(controls[:, :PROBE_MACROS] == controls[0, :PROBE_MACROS])
    assert np.array_equal(controls[1], controls[8])
    assert np.unique(controls.reshape(N_ENVS, -1), axis=0).shape[0] == 8


def test_protocol_is_reward_blind_source_only_and_exact_fallback():
    row = protocol()
    assert row["probe_selection_uses_future_reward"] is False
    assert row["old_wrapping_outcomes_not_used_for_probe_selection"] is True
    assert row["native_trajectories_if_complete"] == 297
    assert row["probe_names"] == list(PROBE_NAMES)
    for key in (
        "automatic_promotion",
        "fresh_evaluation_authorized",
        "new_recordings",
        "gpu_work",
        "protected_data_read",
        "retry_authorized",
    ):
        assert row[key] is False


@pytest.mark.parametrize("stage", ["bad", "", "PROBE"])
def test_unregistered_stage_rejected(stage):
    with pytest.raises(ValueError, match="unregistered"):
        task(stage, 0)


@pytest.mark.parametrize("index", [-1, 11, True, 1.0])
def test_unregistered_task_index_rejected(index):
    with pytest.raises(ValueError, match="unregistered"):
        task("probe", index)


def test_native_validation_precedes_import(tmp_path):
    world = worlds()[4]
    world["stretching_K"] = 3e5
    with pytest.raises(ValueError, match="unregistered"):
        run_world(tmp_path / "missing", tmp_path / "out", world, "probe", None)
    assert not (tmp_path / "out").exists()


def test_prefix_adapter_cannot_receive_future_and_preserves_probe_identity():
    data = np.arange(PROBE_STEPS * N_ENVS * 50 * 3).reshape(PROBE_STEPS, N_ENVS, 50, 3)
    selected = prefix_observation(data, "probe")
    assert selected.shape == (4, 5, 5, 3)
    assert np.array_equal(selected[3, 2, 4], data[OBSERVATION_FRAMES[2], 3, 49])
    selected[:] = -1
    assert np.all(data >= 0)
    with pytest.raises(ValueError, match="trace"):
        prefix_observation(np.zeros((PROBE_STEPS + 1, N_ENVS, 50, 3)), "probe")


def test_full_prefix_adapter_uses_one_common_action_without_future():
    data = np.arange(FULL_STEPS * N_ENVS * 50 * 3).reshape(FULL_STEPS, N_ENVS, 50, 3)
    selected = prefix_observation(data, "active")
    assert selected.shape == (5, 5, 3)
    assert np.array_equal(selected[-1, -1], data[1199, 1, 49])


def test_independent_reward_formulas_match_native_on_random_polygons():
    rng = np.random.default_rng(261001)
    points = rng.normal(0.1, 0.05, (3, 9, 50, 3))
    posts = rng.normal(0.1, 0.05, (3, 9, 3, 3))
    expected = native_reward(points, posts)
    assert np.allclose(checker.angular_reward(points, posts), expected, atol=1e-13)
    assert np.allclose(
        partial_checker.angular_reward(points, posts), expected, atol=1e-13
    )


def synthetic_probe_prefix():
    result = np.zeros((9, 4, 5, 5, 3))
    pattern = np.linspace(-1, 1, 25).reshape(5, 5)
    for world in range(9):
        result[world, 0, ..., 0] = (world - 4) * 0.0002 * pattern
        result[world, 1, ..., 0] = (world - 4) * 0.008 * pattern
        result[world, 2, ..., 0] = (world - 4) * 0.002 * pattern
        result[world, 3, ..., 0] = (world - 4) * 0.004 * pattern
    return result


def test_reward_blind_probe_information_selects_identifiable_excitation():
    result = probe_information(synthetic_probe_prefix())
    assert result["selected_probe_index"] == 1
    assert result["passed"]
    assert result["future_reward_used"] is False
    assert result["mutual_information_nats"][1] > result["mutual_information_nats"][0]
    assert (
        checker.maximum_difference(
            result, checker.probe_information(synthetic_probe_prefix())
        )
        < 1e-10
    )


def test_uninformative_probe_bank_fails_without_reward_rescue():
    result = probe_information(np.zeros((9, 4, 5, 5, 3)))
    assert result["selected_probe_index"] == 0
    assert not result["passed"]


def synthetic_decision_bank():
    prefix = synthetic_probe_prefix()[:, 1]
    reward = np.full((9, 8), 0.6)
    for world in range(9):
        reward[world, world % 3] = 0.95
    return prefix, reward


def test_decision_value_uses_distribution_and_strong_fixed_comparator():
    prefix, reward = synthetic_decision_bank()
    result = decision_value(prefix, reward, 7)
    assert (
        result["arms"]["bias_aware_bayes"]["expected_native_final_reward"]
        > result["best_fixed_reward"]
    )
    assert len(result["oracle_actions"]) == 9
    assert sum(
        result["arms"]["bias_aware_bayes"]["action_probability"]
    ) == pytest.approx(1)
    assert (
        checker.maximum_difference(result, checker.decision_value(prefix, reward, 7))
        < 1e-10
    )


def decision_fixture(bayes, fixed, world=None):
    if world is None:
        world = [bayes] * 9
    return {
        "arms": {
            "bias_aware_bayes": {
                "expected_native_final_reward": bayes,
                "source_world_expected_rewards": world,
            },
            "bias_aware_map": {"expected_native_final_reward": bayes - 0.001},
        },
        "best_fixed_reward": fixed,
        "prefix_hold_reward": fixed - 0.1,
        "oracle_reward": fixed + 0.05,
        "oracle_actions": [0, 1] * 4 + [0],
    }


def test_active_gate_requires_difference_in_decision_value_not_only_reward():
    passive = {
        "metrics": {"arms": {"bias_aware_bayes": {"expected_native_final_reward": 0.9}}}
    }
    active = decision_fixture(0.94, 0.90, [0.94] * 9)
    null = decision_fixture(0.92, 0.895, [0.92] * 9)
    assert active_decision_gate(active, null, passive)["passed"]
    shifted = decision_fixture(0.94, 0.915, [0.94] * 9)
    result = active_decision_gate(shifted, null, passive)
    assert not result["passed"]
    assert not result["checks"]["gain_difference_over_null"]


@pytest.mark.parametrize(
    "stage,probe", [("probe", None), ("baseline", 0), ("active", 1)]
)
def test_native_qa_roundtrip(stage, probe):
    data, native = fixture(stage, probe)
    result = native_qa(data, native, worlds()[4], stage, probe)
    assert result["passed"]
    if stage == "probe":
        assert (
            partial_checker.compare(
                result, partial_checker.independent_qa(data, native)
            )
            < 1e-10
        )


@pytest.mark.parametrize(
    "mutation,check",
    [
        ("endpoint", "common_probe_endpoint_tools"),
        ("post", "fixed_posts"),
        ("reward", "native_final_reward"),
        ("failure", "ordinary_native_success"),
    ],
)
def test_native_gates_fail_closed(mutation, check):
    data, native = fixture("probe", None)
    if mutation == "endpoint":
        data["gripper_pos_m"][-1, 3, 0, 0] += 0.001
    elif mutation == "post":
        data["post_pos_m"][:, :, 0, 0] += 0.001
    elif mutation == "reward":
        native["native_final_reward"][0] += 0.01
    else:
        native["native_final_reward"][0] = -99
    result = native_qa(data, native, worlds()[4], "probe", None)
    assert not result["passed"] and not result["checks"][check]


def test_repeat_qualification_detects_changed_native_batch():
    rows = [fixture("probe", None)[0] for _ in range(3)]
    rewards = np.zeros((3, N_ENVS))
    assert repeat_qa(rows, rewards)["passed"]
    rows[2]["rod_pos_m"][0, 0, 0, 0] += 0.002
    assert not repeat_qa(rows, rewards)["passed"]


def test_active_gate_rejects_malformed_summary():
    with pytest.raises(ValueError, match="complete"):
        active_decision_gate({}, {}, {})
