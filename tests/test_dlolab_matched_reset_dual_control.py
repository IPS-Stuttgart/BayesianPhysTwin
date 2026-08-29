from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments import dlolab_matched_reset_native as native
from bayesian_phystwin_experiments.dlolab_matched_reset_dual_control import (
    ACTION_AMPLITUDES_M,
    ACTION_STEPS,
    ARMS,
    GOALS_Y_M,
    PARTICLE_SCALES,
    PROBE_NAMES,
    PROBE_NODES,
    PROBE_STEPS,
    PROBE_TIMES,
    TRUTH_COUNT,
    action_commands,
    particle_bending,
    probe_commands,
    probe_features,
    probe_information,
    protocol,
    score_source,
    seal_decisions,
    task_headroom,
    task_losses,
    truth_partition,
)
from bayesian_phystwin_experiments.dlolab_native import DloLabConfig


def clamps(count: int = 3) -> np.ndarray:
    value = np.zeros((count, 2, 3), dtype=np.float64)
    value[:, 0, 0] = 0.0
    value[:, 1, 0] = DloLabConfig().interval_m
    value[..., 2] = DloLabConfig().height_m
    return value


def test_protocol_freezes_matched_reset_and_information_boundary() -> None:
    value = protocol()
    assert value["matched_reset"]["probe_mechanics_cannot_enter_task_state"]
    assert not value["probe_selection_uses_task_reward_or_future"]
    assert value["staged_information_boundary"].index("decision_seal") < value[
        "staged_information_boundary"
    ].index("truth_task_futures")
    assert value["retry_authorized"] is False


@pytest.mark.parametrize("probe", range(len(PROBE_NAMES)))
def test_probe_commands_are_bounded_and_return_exactly(probe: int) -> None:
    base = clamps()
    value = probe_commands(base, probe)
    assert value.shape == (PROBE_STEPS, len(base), 2, 3)
    assert np.array_equal(value[0], base)
    assert np.array_equal(value[-1], base)
    if probe == 0:
        assert np.array_equal(value, np.broadcast_to(base, value.shape))


@pytest.mark.parametrize("action", range(len(ACTION_AMPLITUDES_M)))
def test_action_commands_return_exactly(action: int) -> None:
    base = clamps()
    value = action_commands(base, action)
    assert value.shape == (ACTION_STEPS, len(base), 2, 3)
    assert np.array_equal(value[0], base)
    assert np.array_equal(value[-1], base)


def test_partition_is_balanced_continuous_and_reproducible() -> None:
    first = truth_partition()
    second = truth_partition()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert len(first["bending"]) == TRUTH_COUNT
    assert np.bincount(first["goal_index"]).tolist() == [TRUTH_COUNT // 2] * 2
    assert not set(first["bending"]).issubset(set(particle_bending()))


def test_probe_features_are_displacements_at_registered_rows() -> None:
    trajectory = np.zeros((2, PROBE_STEPS, DloLabConfig().node_count, 3))
    initial = np.zeros((2, DloLabConfig().node_count, 3))
    trajectory[..., 1] = np.arange(PROBE_STEPS)[None, :, None]
    value = probe_features(trajectory, initial)
    assert value.shape == (2, len(PROBE_TIMES), len(PROBE_NODES), 3)
    assert np.array_equal(value[0, :, 0, 1], np.asarray(PROBE_TIMES))


def test_reward_blind_information_control_detects_active_probe() -> None:
    value = np.zeros(
        (
            len(PROBE_NAMES),
            len(PARTICLE_SCALES),
            len(PROBE_TIMES),
            len(PROBE_NODES),
            3,
        )
    )
    pattern = np.arange(len(PROBE_TIMES) * len(PROBE_NODES)).reshape(
        len(PROBE_TIMES), len(PROBE_NODES)
    )
    pattern = pattern - pattern.mean()
    for world in range(len(PARTICLE_SCALES)):
        value[1, world, ..., 1] = world * 0.0002 * pattern
        value[2, world, ..., 1] = world * 0.0015 * pattern
    result = probe_information(value)
    assert result["task_reward_read"] is False
    assert result["selected_probe_index"] == 2
    assert result["passed"]


def test_task_loss_is_goal_conditioned_and_headroom_gate_detects_material_value() -> None:
    future = np.zeros(
        (
            len(PARTICLE_SCALES),
            len(ACTION_AMPLITUDES_M),
            ACTION_STEPS,
            DloLabConfig().node_count,
            3,
        )
    )
    loss = task_losses(future, GOALS_Y_M)
    assert loss.shape == (
        len(PARTICLE_SCALES),
        len(GOALS_Y_M),
        len(ACTION_AMPLITUDES_M),
    )

    synthetic = np.full_like(loss, 2.0)
    for world in range(len(PARTICLE_SCALES)):
        synthetic[world, 0, world % 3] = 0.0
        synthetic[world, 1, 6 + world % 3] = 0.0
    assert task_headroom(synthetic)["passed"]


def test_decisions_use_probe_observation_and_not_truth_future() -> None:
    shape = (len(PROBE_TIMES), len(PROBE_NODES), 3)
    predictions = np.zeros((len(PARTICLE_SCALES),) + shape)
    pattern = np.arange(np.prod(shape[:-1])).reshape(shape[:-1])
    pattern = pattern - pattern.mean()
    for world in range(len(PARTICLE_SCALES)):
        predictions[world, ..., 1] = world * 0.001 * pattern
    observations = np.stack(
        [predictions[index % len(PARTICLE_SCALES)] for index in range(TRUTH_COUNT)]
    )
    losses = np.ones(
        (len(PARTICLE_SCALES), len(GOALS_Y_M), len(ACTION_AMPLITUDES_M))
    )
    for world in range(len(PARTICLE_SCALES)):
        losses[world, :, world] = 0.0
    goals = np.arange(TRUTH_COUNT) % len(GOALS_Y_M)
    decision = seal_decisions(
        observations,
        np.zeros_like(observations),
        np.zeros_like(observations),
        predictions,
        np.zeros_like(predictions),
        np.zeros_like(predictions),
        losses,
        goals,
    )
    assert set(decision) == set(ARMS[:-1]) | {
        "active_weights",
        "null_weights",
        "fixed_probe_weights",
        "goal_index",
    }
    assert decision["active_bayes"].shape == (TRUTH_COUNT,)


def test_positive_source_value_control_passes_all_gates() -> None:
    decisions = {
        arm: np.full(TRUTH_COUNT, 5 if arm.startswith("active") else 4, dtype=np.int64)
        for arm in ARMS[:-1]
    }
    truth = np.full((TRUTH_COUNT, len(ACTION_AMPLITUDES_M)), 2.0)
    truth[:, 4] = 1.0
    truth[:, 5] = 0.5
    result = score_source(decisions, truth)
    assert result["source_gate_passed"]
    assert all(result["checks"].values())


def test_incomplete_denominators_fail_closed() -> None:
    with pytest.raises(ValueError, match="complete truth action losses"):
        score_source({}, np.zeros((TRUTH_COUNT - 1, len(ACTION_AMPLITUDES_M))))


def test_particle_transport_keeps_different_probe_and_action_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes = [
        np.zeros((len(PARTICLE_SCALES), PROBE_STEPS, DloLabConfig().node_count, 3))
        for _ in PROBE_NAMES
    ]
    actions = [
        np.zeros((len(PARTICLE_SCALES), ACTION_STEPS, DloLabConfig().node_count, 3))
        for _ in ACTION_AMPLITUDES_M
    ]
    monkeypatch.setattr(native, "_branches", lambda *args: (probes + actions, {}))
    arrays, _ = native.generate_particle_bank(Path("/unused"))
    assert arrays["probe_trajectory_m"].shape[2] == PROBE_STEPS
    assert arrays["action_trajectory_m"].shape[2] == ACTION_STEPS
