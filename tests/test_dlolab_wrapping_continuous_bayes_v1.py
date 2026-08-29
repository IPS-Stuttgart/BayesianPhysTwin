from __future__ import annotations

import json

import numpy as np
import pytest

from bayesian_phystwin_experiments import dlolab_wrapping_continuous_bayes_v1 as study
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


def test_worlds_are_fresh_deterministic_and_covered_once() -> None:
    roster = study.continuous_worlds()
    assert len(roster) == study.WORLD_COUNT
    assert len(_material_keys(roster)) == study.WORLD_COUNT
    assert not _material_keys(roster) & _material_keys(worlds())
    assert roster == study.continuous_worlds()
    assert [
        index
        for batch in range(4)
        for index in study.prefix_task(batch)["world_indices"]
    ] == list(range(study.WORLD_COUNT))
    assert [
        study.future_task(index)["world_index"]
        for index in range(study.WORLD_COUNT)
    ] == list(range(study.WORLD_COUNT))
    assert study.prefix_task(3)["native_world_indices"] == [
        27,
        28,
        29,
        30,
        31,
        31,
        31,
        31,
        31,
    ]


@pytest.mark.parametrize("field", ["index", "stretching_K", "extra"])
def test_unregistered_continuous_world_is_rejected(field: str) -> None:
    world = study.continuous_worlds()[0]
    if field == "index":
        world["index"] = True
    elif field == "stretching_K":
        world["stretching_K"] *= 1.01
    else:
        world["extra"] = 1
    with pytest.raises(ValueError, match="registered continuous"):
        study.validate_continuous_world(world)


def test_prefix_qa_excludes_future_and_reward() -> None:
    data, native = _prefix_fixture()
    roster = study.continuous_worlds()[:N_ENVS]
    result = study.prefix_native_qa(data, native, roster)
    assert result["qa_passed"]
    assert result["checks"]["no_future_reward_exposed"]
    native["future_simulated"] = True
    with pytest.raises(ValueError, match="native execution contract"):
        study.prefix_native_qa(data, native, roster)


def test_prefix_adapter_preserves_registered_frames_and_identities() -> None:
    trace = np.arange(600 * N_ENVS * 50 * 3).reshape(600, N_ENVS, 50, 3)
    selected = study.prefix_observation(trace)
    assert selected.shape == (N_ENVS, 3, 5, 3)
    assert np.array_equal(selected[4, 2, 3], trace[599, 4, 41])
    selected[:] = -1
    assert np.all(trace >= 0)
    with pytest.raises(ValueError, match="prefix"):
        study.prefix_observation(np.zeros((601, N_ENVS, 50, 3)))


def test_inference_handles_partial_final_chunk_and_integrates_posterior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "SENSOR_DRAWS", 257)
    model = np.zeros((9, 3, 5, 3), dtype=np.float64)
    model[..., 0] = (np.arange(9) * 0.02)[:, None, None]
    reward = np.zeros((9, 8), dtype=np.float64)
    reward[:, 0] = 0.5
    reward[np.arange(9), 1 + np.arange(9) % 3] = 1.0
    truth = np.stack(
        [model[index % 9] for index in range(study.WORLD_COUNT)]
    )
    result = study.infer_decisions(model, reward, truth)
    assert result["decisions"].shape == (
        study.WORLD_COUNT,
        257,
        len(study.ARM_NAMES),
    )
    assert np.isfinite(result["posterior_entropy_nats"]).all()
    assert np.count_nonzero(result["decisions"][:, :, 2]) > 0
    assert int(result["source_best_fixed_action"]) == 0


def test_pre_future_gate_and_score_use_equal_world_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "SENSOR_DRAWS", 16)
    decisions = np.zeros(
        (study.WORLD_COUNT, 16, len(study.ARM_NAMES)), dtype=np.int64
    )
    decisions[:, :, 1] = 1
    decisions[::2, :, 2] = 2
    decisions[1::2, :, 2] = 3
    decisions[:, :, 3] = 4
    gate = study.pre_future_checks(decisions, all_prefix_qa=True)
    assert gate["pre_future_gate_passed"]
    assert not study.pre_future_checks(
        decisions, all_prefix_qa=False
    )["pre_future_gate_passed"]

    reward = np.zeros((study.WORLD_COUNT, 8), dtype=np.float64)
    reward[:, 0] = 0.5
    reward[:, 1] = 0.51
    reward[::2, 2] = 0.55
    reward[1::2, 3] = 0.56
    reward[:, 4] = 0.50
    reward[::2, 5] = 0.58
    reward[1::2, 6] = 0.59
    result = study.score(decisions, reward, all_native_qa=True)
    assert result["ordinary_worlds"] == study.WORLD_COUNT
    assert result["sensor_draws_per_world"] == 16
    assert result["source_gate_passed"]
    assert result["paired_bayes_gain"]["source_best_fixed"]["mean_gain"] == pytest.approx(
        0.055
    )
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_protocol_preserves_parent_failure_and_public_only_boundaries() -> None:
    value = study.protocol()
    assert value["parent_source_gate_passed"] is False
    assert value["parent_gate_reclassified"] is False
    assert value["worlds_disjoint_from_nine_particle_support"] is True
    assert value["future_before_decision_barrier"] is False
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
