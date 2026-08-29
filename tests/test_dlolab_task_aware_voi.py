from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments import dlolab_task_aware_native as native
from bayesian_phystwin_experiments.dlolab_native import DloLabConfig
from bayesian_phystwin_experiments.dlolab_task_aware_voi import (
    ACTION_ROOT_Z_M,
    ACTION_STEPS,
    BENDING_SCALES,
    FIXED_PROBE_INDEX,
    GOAL_TIP_Z_M,
    PROBE_NAMES,
    PROBE_NODES,
    PROBE_STEPS,
    PROBE_TIMES,
    TRUTH_COUNT,
    TWISTING_SCALES,
    action_commands,
    particle_count,
    particle_parameters,
    probe_commands,
    protocol,
    score_source,
    selector_analysis,
    task_headroom,
    truth_partition,
)


def _clamps(count: int) -> np.ndarray:
    value = np.zeros((count, 2, 3), dtype=np.float64)
    value[:, 1, 0] = DloLabConfig().interval_m
    value[..., 2] = DloLabConfig().height_m
    return value


def _synthetic_features_and_losses() -> tuple[np.ndarray, np.ndarray]:
    parameters = particle_parameters()
    bend = parameters["bending_index"]
    twist = parameters["twisting_index"]
    features = np.zeros(
        (
            len(PROBE_NAMES),
            particle_count(),
            len(PROBE_TIMES),
            len(PROBE_NODES),
            3,
        ),
        dtype=np.float64,
    )
    node_pattern = np.asarray([-1.0, -0.3, 0.3, 1.0])
    time_pattern = np.asarray([0.5, 1.0, -0.5, -1.0])
    pattern = time_pattern[:, None] * node_pattern[None]
    features[1, ..., 2] = bend[:, None, None] * 0.004 * pattern
    features[2, ..., 2] = np.minimum(bend, 2)[:, None, None] * 0.002 * pattern
    features[3, ..., 1] = twist[:, None, None] * 0.006 * pattern
    features[4, ..., 1] = twist[:, None, None] * 0.006 * pattern
    features[4, ..., 2] = (bend // 2)[:, None, None] * 0.003 * pattern

    actions = np.arange(len(ACTION_ROOT_Z_M))
    losses = np.zeros((particle_count(), len(GOAL_TIP_Z_M), len(actions)))
    for world in range(particle_count()):
        for goal in range(len(GOAL_TIP_Z_M)):
            target = np.clip(bend[world] + goal + 1, 0, len(actions) - 1)
            losses[world, goal] = 0.01 + (actions - target) ** 2
    return features, losses


def test_protocol_declares_task_aware_selection_and_closed_boundaries() -> None:
    value = protocol()
    assert value["primary_probe_selection"] == "minimum expected downstream Bayes task loss"
    assert value["generic_information_control"] == "maximum full-latent mutual information"
    assert value["primary_probe_uses_truth_futures"] is False
    assert value["distinct_from_closed_mi_only_protocol"] is True
    assert value["retry_authorized"] is False
    assert value["protected_data_read"] is False


def test_particle_and_truth_partitions_are_complete_and_deterministic() -> None:
    particle = particle_parameters()
    assert particle["bending"].shape == (len(BENDING_SCALES) * len(TWISTING_SCALES),)
    assert particle["twisting"].shape == particle["bending"].shape
    assert np.unique(particle["bending_index"]).size == len(BENDING_SCALES)
    assert np.unique(particle["twisting_index"]).size == len(TWISTING_SCALES)
    first = truth_partition()
    second = truth_partition()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert first["bending"].shape == (TRUTH_COUNT,)
    assert first["twisting"].shape == (TRUTH_COUNT,)


@pytest.mark.parametrize("probe", range(len(PROBE_NAMES)))
def test_probe_commands_start_and_end_at_exact_matched_state(probe: int) -> None:
    base = _clamps(3)
    value = probe_commands(base, probe)
    assert value.shape == (PROBE_STEPS, 3, 2, 3)
    assert np.array_equal(value[0], base)
    assert np.array_equal(value[-1], base)


@pytest.mark.parametrize("action", range(len(ACTION_ROOT_Z_M)))
def test_task_actions_are_smooth_vertical_tilts(action: int) -> None:
    base = _clamps(2)
    value = action_commands(base, action)
    assert value.shape == (ACTION_STEPS, 2, 2, 3)
    assert np.array_equal(value[0], base)
    assert np.all(value[:, :, 0] == base[:, 0])
    assert np.allclose(
        value[-1, :, 1, 2] - base[:, 1, 2], ACTION_ROOT_Z_M[action], rtol=0, atol=1e-15
    )


def test_synthetic_task_aware_selector_can_reject_higher_information_probe() -> None:
    features, losses = _synthetic_features_and_losses()
    result = selector_analysis(features, losses, draws_per_world=64, seed=401)
    assert result["task_aware_probe_index"] == 1
    assert result["generic_mi_probe_index"] == 4
    assert result["mutual_information_nats"][4] > result["mutual_information_nats"][1]
    assert result["expected_task_loss"][1] < result["expected_task_loss"][4]
    assert result["passed"] is True


def test_synthetic_task_headroom_identifies_bending_not_twisting() -> None:
    _, losses = _synthetic_features_and_losses()
    result = task_headroom(losses)
    assert result["passed"] is True
    assert min(result["distinct_bending_conditioned_oracle_actions_per_goal"]) >= 2
    assert max(result["twisting_only_oracle_disagreement_fraction"]) == 0


def test_positive_source_value_control_passes_all_gates() -> None:
    actions = len(ACTION_ROOT_Z_M)
    truth = np.full((TRUTH_COUNT, actions), 2.0)
    truth[:, 0] = 1.0
    truth[:, 1] = 0.0
    decisions = {
        "best_fixed": np.zeros(TRUTH_COUNT, dtype=np.int64),
        "null_bayes": np.zeros(TRUTH_COUNT, dtype=np.int64),
        "fixed_probe_bayes": np.zeros(TRUTH_COUNT, dtype=np.int64),
        "mi_probe_bayes": np.zeros(TRUTH_COUNT, dtype=np.int64),
        "task_aware_map": np.ones(TRUTH_COUNT, dtype=np.int64),
        "task_aware_bayes": np.ones(TRUTH_COUNT, dtype=np.int64),
        "task_aware_guarded": np.ones(TRUTH_COUNT, dtype=np.int64),
    }
    result = score_source(decisions, truth)
    assert result["source_gate_passed"] is True
    assert all(result["checks"].values())


def test_incomplete_source_denominator_fails_closed() -> None:
    with pytest.raises(ValueError, match="complete truth task-loss table"):
        score_source({}, np.zeros((TRUTH_COUNT - 1, len(ACTION_ROOT_Z_M))))


def test_native_transport_keeps_probe_and_action_horizons_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes = [
        np.zeros((particle_count(), PROBE_STEPS, DloLabConfig().node_count, 3))
        for _ in PROBE_NAMES
    ]
    actions = [
        np.zeros((particle_count(), ACTION_STEPS, DloLabConfig().node_count, 3))
        for _ in ACTION_ROOT_Z_M
    ]
    monkeypatch.setattr(native, "_branches", lambda *args: (probes + actions, {}))
    arrays, _ = native.generate_particle_bank(Path("/unused"))
    assert arrays["probe_trajectory_m"].shape[2] == PROBE_STEPS
    assert arrays["action_trajectory_m"].shape[2] == ACTION_STEPS


def test_fixed_probe_is_not_null() -> None:
    assert FIXED_PROBE_INDEX != 0
