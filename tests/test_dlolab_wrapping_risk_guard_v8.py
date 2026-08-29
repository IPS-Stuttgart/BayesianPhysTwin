from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from bayesian_phystwin_experiments import dlolab_wrapping_risk_guard_v4 as terminal_v4
from bayesian_phystwin_experiments import dlolab_wrapping_risk_guard_v8 as study
from bayesian_phystwin_experiments.dlolab_wrapping_continuous_bayes_v1 import (
    continuous_worlds as failed_v1_worlds,
)
from bayesian_phystwin_experiments.dlolab_wrapping_continuous_interp_v2 import (
    continuous_worlds as development_v2_worlds,
)
from bayesian_phystwin_experiments.dlolab_wrapping_resolution_ensemble_v3 import (
    continuous_worlds as development_v3_worlds,
)
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_v4 import (
    continuous_worlds as terminal_v4_worlds,
)
from bayesian_phystwin_experiments.dlolab_wrapping_source import (
    MEMORY_NAMES,
    N_ENVS,
    POSTS,
    action_bank,
    worlds,
)


def _material_keys(rows: list[dict[str, object]]) -> set[tuple[object, object]]:
    return {(row["stretching_K"], row["bending_E"]) for row in rows}


def _circle() -> np.ndarray:
    angle = np.arange(50) * 2 * np.pi / 50 + 0.03
    return np.column_stack(
        (
            0.6 + 0.14 * np.cos(angle),
            0.14 * np.sin(angle),
            np.full(50, 0.012),
        )
    )


def _prefix_fixture() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    loop = _circle()
    roster = study.continuous_worlds()[:N_ENVS]
    steps = 600
    data = {
        "rod_pos_m": np.tile(loop, (steps, N_ENVS, 1, 1)),
        "rod_vel_m_s": np.zeros((steps, N_ENVS, 50, 3)),
        "post_pos_m": np.tile(POSTS, (steps, N_ENVS, 1, 1)),
        "gripper_pos_m": np.tile(loop[[17, 33]], (steps, N_ENVS, 1, 1)),
        "robot_qpos": np.zeros((steps, N_ENVS, 18)),
        "controls": action_bank()[:, :3],
        "joint_targets": np.zeros((N_ENVS, 31, 18)),
        "initial_rod_pos_m": np.tile(loop, (N_ENVS, 1, 1)),
    }
    data.update({name: np.zeros((N_ENVS, 1)) for name in MEMORY_NAMES})
    native: dict[str, object] = {
        "native_steps": steps,
        "worlds": roster,
        "world_realization": {
            "bending": [row["bending_E"] for row in roster],
            "stretching": [row["stretching_K"] for row in roster],
        },
        "prefix_only": True,
        "future_simulated": False,
        "reward_exposed": False,
        "prefix_reward_excluded": True,
        "twisting_stiffness_zero_preserved": True,
        "device": "cpu",
        "runtime_camera_rendered": False,
        "native_source_modified": False,
    }
    return data, native


def _future_fixture() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    loop = _circle()
    world = study.continuous_worlds()[0]
    steps = 2200
    data = {
        "rod_pos_m": np.broadcast_to(loop, (steps, N_ENVS, 50, 3)),
        "rod_vel_m_s": np.broadcast_to(np.zeros((1, 1, 50, 3)), (steps, N_ENVS, 50, 3)),
        "post_pos_m": np.broadcast_to(POSTS, (steps, N_ENVS, 3, 3)),
        "gripper_pos_m": np.broadcast_to(loop[[17, 33]], (steps, N_ENVS, 2, 3)),
        "robot_qpos": np.broadcast_to(np.zeros((1, 1, 18)), (steps, N_ENVS, 18)),
        "controls": action_bank(),
        "joint_targets": np.zeros((N_ENVS, 111, 18)),
        "initial_rod_pos_m": np.broadcast_to(loop, (N_ENVS, 50, 3)),
    }
    data.update({name: np.zeros((N_ENVS, 1)) for name in MEMORY_NAMES})
    final = study.native_reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    cumulative = np.zeros(N_ENVS, dtype=np.float32)
    for _ in range(110):
        cumulative += final.astype(np.float32) + np.float32(1)
    native: dict[str, object] = {
        "native_steps": steps,
        "worlds": [world] * N_ENVS,
        "world_realization": {
            "bending": [world["bending_E"]] * N_ENVS,
            "stretching": [world["stretching_K"]] * N_ENVS,
        },
        "prefix_only": False,
        "future_simulated": True,
        "reward_exposed": True,
        "prefix_reward_excluded": False,
        "native_final_reward": final.tolist(),
        "native_cumulative_reward": cumulative.tolist(),
        "twisting_stiffness_zero_preserved": True,
        "device": "cpu",
        "runtime_camera_rendered": False,
        "native_source_modified": False,
    }
    return data, native


def test_worlds_are_fresh_deterministic_and_in_registered_stress_region() -> None:
    roster = study.continuous_worlds()
    assert len(roster) == study.WORLD_COUNT == 144
    assert len(_material_keys(roster)) == 144
    previous = (
        _material_keys(worlds())
        | _material_keys(failed_v1_worlds())
        | _material_keys(development_v2_worlds())
        | _material_keys(development_v3_worlds())
        | _material_keys(terminal_v4_worlds())
    )
    assert not _material_keys(roster) & previous
    x = np.asarray([np.log(row["stretching_K"] / 2e4) / np.log(25.0) for row in roster])
    y = np.asarray([np.log(row["bending_E"] / 1e3) / np.log(100.0) for row in roster])
    assert np.all((x >= 0.60) & (x <= 0.995))
    assert np.all((y >= 0.02) & (y <= 0.70))
    assert [
        index
        for batch in range(study.PREFIX_BATCH_COUNT)
        for index in study.prefix_task(batch)["world_indices"]
    ] == list(range(144))


@pytest.mark.parametrize("field", ["index", "stretching_K", "extra"])
def test_unregistered_world_is_rejected(field: str) -> None:
    world = study.continuous_worlds()[0]
    if field == "index":
        world["index"] = True
    elif field == "stretching_K":
        world["stretching_K"] *= 1.01
    else:
        world["extra"] = 1
    with pytest.raises(ValueError, match="registered continuous"):
        study.validate_continuous_world(world)


def test_native_qa_preserves_prefix_future_boundary_and_json_booleans() -> None:
    prefix, prefix_native = _prefix_fixture()
    prefix_result = study.prefix_native_qa(
        prefix, prefix_native, study.continuous_worlds()[:N_ENVS]
    )
    assert prefix_result["qa_passed"]
    assert prefix_result["checks"]["no_future_reward_exposed"]
    future, future_native = _future_fixture()
    future_result = study.future_native_qa(
        future, future_native, study.continuous_worlds()[0]
    )
    assert future_result["qa_passed"]
    assert all(type(value) is bool for value in future_result["checks"].values())
    json.dumps(future_result, sort_keys=True, allow_nan=False)


def test_posterior_guard_uses_probability_then_highest_expected_reward() -> None:
    expected = np.asarray([[0.5, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2]])
    probability = np.asarray([[1.0, 0.97, 0.98, 0.99, 1.0, 0.4, 0.3, 0.2]])
    assert study.posterior_guard_actions(
        expected, probability, threshold=0.975, fixed_action=0
    ).tolist() == [2]
    assert study.posterior_guard_actions(
        expected, probability, threshold=0.99, fixed_action=0
    ).tolist() == [3]
    probability[:, 1:] = 0
    assert study.posterior_guard_actions(
        expected, probability, threshold=0.975, fixed_action=0
    ).tolist() == [0]


def test_inference_integrates_posterior_and_processes_partial_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "SENSOR_DRAWS", 257)
    model = np.zeros((9, 3, 5, 3), dtype=np.float64)
    model[..., 0] = (np.arange(9) * 0.02)[:, None, None]
    reward = np.zeros((9, 8), dtype=np.float64)
    reward[:, 0] = 0.5
    reward[np.arange(9), 1 + np.arange(9) % 3] = 1.0
    truth = np.stack([model[index % 9] for index in range(study.WORLD_COUNT)])
    result = study.infer_decisions(model, reward, truth)
    assert result["decisions"].shape == (study.WORLD_COUNT, 257, len(study.ARM_NAMES))
    assert result["continuous_posterior_expected_reward"].shape == (
        study.WORLD_COUNT,
        257,
        study.N_ACTIONS,
    )
    nonfixed = result["decisions"][:, :, 2] != result["decisions"][:, :, 0]
    assert np.all(
        result["guarded_posterior_improvement_probability"][nonfixed]
        >= study.PRIMARY_PROBABILITY
    )
    fixed = int(result["continuous_prior_best_fixed_action"])
    assert np.array_equal(
        result["continuous_posterior_improvement_probability"][:, :, fixed],
        np.zeros((study.WORLD_COUNT, 257)),
    )


def test_pre_future_gate_and_score_capture_downside_tradeoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "SENSOR_DRAWS", 16)
    decisions = np.zeros((study.WORLD_COUNT, 16, len(study.ARM_NAMES)), dtype=np.int64)
    decisions[:, :, 1] = np.where(np.arange(study.WORLD_COUNT)[:, None] < 2, 2, 1)
    decisions[: study.WORLD_COUNT // 2, :, 2] = 1
    decisions[study.WORLD_COUNT // 2 :, :, 2] = 3
    decisions[:, :, 3] = decisions[:, :, 2]
    decisions[:, :, 4] = decisions[:, :, 2]
    decisions[:, :, 5] = decisions[:, :, 1]
    decisions[:, :, 6] = decisions[:, :, 1]
    probability = np.full((study.WORLD_COUNT, 16), 0.98)
    gate = study.pre_future_checks(decisions, probability, all_prefix_qa=True)
    assert gate["pre_future_gate_passed"]

    reward = np.full((study.WORLD_COUNT, study.N_ACTIONS), 0.49)
    reward[:, 0] = 0.50
    reward[:, 1] = 0.52
    reward[:, 3] = 0.52
    reward[study.WORLD_COUNT // 2 :, 3] = 0.521
    reward[:2, 2] = 0.45
    reward[2:, 2] = 0.52
    result = study.score(decisions, reward, all_native_qa=True)
    assert result["source_gate_passed"]
    assert result["guard_harmed_worlds"] == 0
    assert result["continuous_harmed_worlds"] == 2
    assert result["guard_downside_reduction_fraction_vs_continuous"] == 1.0


def test_protocol_freezes_development_use_and_public_only_boundaries() -> None:
    value = study.protocol()
    assert value["development_v2_source_gate_passed"] is False
    assert value["development_v3_source_gate_passed"] is False
    assert value["development_v4_diagnostic_id"] == study.DEVELOPMENT_V4_DIAGNOSTIC_ID
    assert value["posterior_improvement_probability_threshold"] == 0.975
    assert value["registered_fixed_action_index"] == 4
    assert value["candidate_selection_used_open_development_outcomes"] is True
    assert value["worlds_disjoint_from_source_v1_v2_v3_and_terminal_v4"]
    assert value["terminal_v4_retried_or_scored"] is False
    assert value["runtime_v7_qualification_passed"] is True
    assert value["method_class_changed_from_v4"] is False
    for field in (
        "retry_authorized",
        "replacement_authorized",
        "fresh_successor_automatically_authorized",
        "official_benchmark_or_sota_claim",
        "real_robot_or_physical_safety_claim",
        "protected_data_read",
        "held_v8_read",
        "dlo4_dlo5_read",
        "official_dlo3_evaluation",
        "new_recordings",
        "gpu_work",
        "push_or_merge",
    ):
        assert value[field] is False


def test_v8_preserves_the_v4_scientific_method_exactly() -> None:
    unchanged_functions = (
        "_base_native_qa",
        "prefix_native_qa",
        "future_native_qa",
        "prefix_observation",
        "_covariance_cholesky",
        "_whiten",
        "_bilinear",
        "interpolated_bank",
        "posterior_guard_actions",
        "risk_actions",
        "infer_risk_decisions",
    )
    for name in unchanged_functions:
        assert inspect.getsource(getattr(study, name)) == inspect.getsource(
            getattr(terminal_v4, name)
        )
    for name in (
        "SENSOR_DRAWS",
        "BOOTSTRAP_REPLICATES",
        "QUADRATURE_POINTS_PER_AXIS",
        "SHARED_BIAS_STD_M",
        "INDEPENDENT_NOISE_STD_M",
        "REWARD_MARGIN",
        "PRIMARY_PROBABILITY",
        "SENSITIVITY_PROBABILITIES",
        "NORMALIZED_LOG_STRETCHING_RANGE",
        "NORMALIZED_LOG_BENDING_RANGE",
        "ARM_NAMES",
    ):
        assert getattr(study, name) == getattr(terminal_v4, name)
    v4_protocol = terminal_v4.protocol()
    v8_protocol = study.protocol()
    assert v8_protocol["method"] == v4_protocol["method"]
    assert v8_protocol["posterior_improvement_probability_threshold"] == v4_protocol[
        "posterior_improvement_probability_threshold"
    ]
    assert v8_protocol["posterior_improvement_margin"] == v4_protocol[
        "posterior_improvement_margin"
    ]
    assert v8_protocol["registered_fixed_action_index"] == v4_protocol[
        "registered_fixed_action_index"
    ]
    assert v8_protocol["arms"] == v4_protocol["arms"]
