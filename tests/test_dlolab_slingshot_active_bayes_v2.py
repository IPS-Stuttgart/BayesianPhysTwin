from __future__ import annotations

import json

import numpy as np

from bayesian_phystwin_experiments.dlolab_slingshot_active_bayes import (
    continuous_worlds as v1_continuous_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_active_bayes_v2 import (
    ARM_NAMES,
    SENSOR_DRAWS,
    WORLD_COUNT,
    continuous_worlds,
    future_task,
    infer_decisions,
    pre_future_checks,
    prefix_task,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    particle_worlds,
    sample_worlds,
)


def test_worlds_are_fresh_and_tasks_cover_them_once() -> None:
    worlds = continuous_worlds()

    def keys(rows: list[dict[str, object]]) -> set[tuple[object, object, object]]:
        return {
            (row["x_offset_m"], row["bending_E"], row["stretching_K"])
            for row in rows
        }
    assert len(worlds) == WORLD_COUNT
    assert len(keys(worlds)) == WORLD_COUNT
    assert not keys(worlds) & keys(particle_worlds())
    assert not keys(worlds) & keys(sample_worlds("calibration"))
    assert not keys(worlds) & keys(sample_worlds("evaluation"))
    assert not keys(worlds) & keys(v1_continuous_worlds())
    assert [
        index
        for probe in range(2)
        for batch in range(4)
        for index in prefix_task(probe, batch)["world_indices"]
    ] == list(range(WORLD_COUNT)) * 2
    assert [future_task(index)["world_index"] for index in range(WORLD_COUNT)] == list(
        range(WORLD_COUNT)
    )


def test_decision_inference_preserves_posterior_integration() -> None:
    history = np.zeros((2, 27, 3, 4, 3), dtype=np.float64)
    history[1, :, 0, 0, 0] = np.arange(27) * 0.02
    reward = np.zeros((27, 7), dtype=np.float64)
    reward[:, 0] = 1.0
    reward[np.arange(27), 1 + np.arange(27) % 6] = 3.0
    truth = np.zeros((2, WORLD_COUNT, 3, 4, 3), dtype=np.float64)
    truth[1, :, 0, 0, 0] = np.arange(WORLD_COUNT) % 27 * 0.02
    result = infer_decisions(history, reward, truth)
    assert result["decisions"].shape == (
        WORLD_COUNT,
        SENSOR_DRAWS,
        len(ARM_NAMES),
    )
    assert result["posterior_weights"].shape == (2, WORLD_COUNT, SENSOR_DRAWS, 27)
    np.testing.assert_allclose(result["posterior_weights"].sum(axis=-1), 1.0)


def test_pre_future_gate_and_score_use_world_level_aggregation() -> None:
    decisions = np.zeros((WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES)), dtype=np.int64)
    decisions[:, :, 3] = 1
    decisions[:, :, 4] = 2
    gate = pre_future_checks(decisions, all_prefix_qa=True)
    assert gate["pre_future_gate_passed"] is False
    decisions[::2, :, 4] = 3
    gate = pre_future_checks(decisions, all_prefix_qa=True)
    assert gate["pre_future_gate_passed"] is True
    rewards = np.zeros((WORLD_COUNT, 7), dtype=np.float64)
    rewards[:, 0] = 1.0
    rewards[:, 1] = 0.9
    rewards[:, 2] = 1.01
    rewards[:, 3] = 1.02
    rewards[:, 4] = 1.03
    value = score(decisions, rewards, all_native_qa=True)
    assert value["ordinary_worlds"] == WORLD_COUNT
    assert value["sensor_draws_per_world"] == SENSOR_DRAWS
    assert all(
        sum(arm["action_counts"]) == WORLD_COUNT * SENSOR_DRAWS
        for arm in value["arms"].values()
    )
    json.dumps(value, sort_keys=True, allow_nan=False)


def test_protocol_retains_parent_failure_and_information_boundaries() -> None:
    value = protocol()
    assert value["parent_particle_gate_passed"] is False
    assert value["parent_particle_gate_reclassified"] is False
    assert value["parent_output_retried"] is False
    assert value["v1_retried"] is False
    assert value["v1_scientific_result"] is False
    assert value["runtime_preflight_passed"] is True
    assert value["task_future_before_decision_barrier"] is False
    assert value["continuous_truth_protocol_automatically_authorized"] is False
    assert value["retry_authorized"] is False
