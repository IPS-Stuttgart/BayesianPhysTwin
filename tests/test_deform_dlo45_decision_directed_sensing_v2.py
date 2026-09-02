from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from experiments.deform_dlo45_decision_directed_sensing_v2 import (
    evaluate as module,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "experiments" / "deform_dlo45_decision_directed_sensing_v2" / "protocol.json"
)
SCRIPT = (
    ROOT / "experiments" / "deform_dlo45_decision_directed_sensing_v2" / "evaluate.py"
)


def _protocol() -> module.Protocol:
    return module.load_protocol(PROTOCOL)


def _trajectory() -> np.ndarray:
    trajectory = np.zeros((module.FRAME_COUNT, module.NODE_COUNT, 3))
    trajectory[:, module.NODE_COUNT - 2 :, 0] = 1.0
    weights = np.linspace(1.0 / 9.0, 8.0 / 9.0, 8)
    trajectory[:, 2:10, 0] = weights[None, :]
    return trajectory


def _context() -> module.CaseContext:
    sensor = np.zeros((4, 8, 1))
    sensor[:, 0, 0] = np.asarray([-2.0, -2.0, 2.0, 2.0])
    sensor[:, 1, 0] = np.asarray([-2.0, 2.0, -2.0, 2.0])
    losses = np.asarray(
        [
            [3.0, 1.0, 3.0],
            [2.0, 0.0, 0.0],
            [5.0, 4.0, 5.0],
            [0.0, 5.0, 2.0],
        ]
    )
    return module.CaseContext(
        support_indices=np.arange(4, dtype=np.int64),
        base_logits=np.zeros(4),
        support_sensor_features=sensor,
        target_sensor_features=sensor[0],
        support_classes=np.asarray([0, 0, 1, 1], dtype=np.int64),
        support_global_classes=np.asarray([0, 0, 1, 1], dtype=np.int64),
        support_state_representation=np.asarray([[-1.0], [-0.5], [0.5], [1.0]]),
        support_query_representation=np.asarray([[-1.0], [-0.5], [0.5], [1.0]]),
        support_task_residuals=np.zeros((4, 2)),
        actions=np.zeros((3, 2)),
        action_labels=("fallback", "left", "right"),
        relative_losses=losses,
        length_scale=1.0,
    )


def test_protocol_freezes_competing_actions_and_source_only_evaluation() -> None:
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol = _protocol()
    assert raw["evaluation"]["evaluation_split_opened"] is False
    assert raw["evaluation"]["target_tuning"] is False
    assert raw["evaluation"]["new_data_collection"] is False
    assert raw["calibration"]["shared_across_dlos"] is True
    assert protocol.policies == module.EXPECTED_POLICIES
    assert protocol.task_internal_nodes == (4, 5, 6, 7)
    assert protocol.budgets == (0, 1, 2, 3, 4, 6, 8)


def test_endpoint_observation_does_not_touch_future_internal_outcomes() -> None:
    protocol = _protocol()
    trajectory = _trajectory()
    current = protocol.first_current_frame
    future = slice(current + 1, current + 1 + protocol.horizon_frames)
    trajectory[future, module.INTERNAL, :] = np.nan

    observation = module.extract_endpoint_observation(trajectory, current, protocol)

    assert np.all(np.isfinite(observation.base_feature))
    assert np.all(np.isfinite(observation.sensor_features))
    assert np.all(np.isfinite(observation.baseline))
    target = module.extract_full_target_residual(
        trajectory, current, observation, protocol
    )
    assert np.any(~np.isfinite(target))


def test_task_residual_selects_only_registered_central_nodes() -> None:
    protocol = _protocol()
    full = np.arange(protocol.horizon_frames * 8 * 3, dtype=np.float64).reshape(
        protocol.horizon_frames, 8, 3
    )
    selected = module.task_residuals(full.reshape(-1), protocol)
    expected = full[:, 2:6, :].reshape(-1)
    np.testing.assert_array_equal(selected, expected)


def test_source_split_is_deterministic_and_disjoint() -> None:
    protocol = _protocol()
    names = tuple(f"{index}.pkl" for index in range(56))
    first = module.split_names(names, "DLO4", protocol)
    second = module.split_names(names, "DLO4", protocol)
    assert first == second
    assert len(first["fit"]) == 39
    assert len(first["calibration"]) == 9
    assert len(first["source_test"]) == 8
    assert not (set(first["fit"]) & set(first["calibration"]))
    assert not (set(first["fit"]) & set(first["source_test"]))
    assert not (set(first["calibration"]) & set(first["source_test"]))


def test_decision_regret_selects_action_relevant_node() -> None:
    protocol = dataclasses.replace(_protocol(), measurement_costs=np.ones(8))
    context = _context()
    observations: dict[int, np.ndarray] = {}
    selected = module.choose_candidate(
        "decision_regret",
        context,
        observations,
        (0, 1),
        "synthetic",
        2.0,
        protocol,
    )
    class_sensor = module.expected_candidate_metric(
        context, observations, 0, "decision_regret", 2.0
    )
    within_class_sensor = module.expected_candidate_metric(
        context, observations, 1, "decision_regret", 2.0
    )
    assert selected == 0
    assert class_sensor < within_class_sensor


def test_posterior_update_is_normalized_and_informative() -> None:
    context = _context()
    prior = module.posterior_weights(context, {}, 2.0)
    posterior = module.posterior_weights(
        context, {0: context.target_sensor_features[0]}, 2.0
    )
    assert np.isclose(np.sum(prior), 1.0)
    assert np.isclose(np.sum(posterior), 1.0)
    assert posterior[:2].sum() > 0.99
    assert module._weighted_variance(
        context.support_state_representation, posterior
    ) < module._weighted_variance(context.support_state_representation, prior)


def test_action_competition_reports_multiple_pointwise_winners() -> None:
    prototypes = np.asarray([[-1.0, 1.0], [1.0, 1.0]])
    task = np.asarray(
        [
            [-1.0, 1.0],
            [-0.8, 1.1],
            [1.0, 1.0],
            [0.9, 1.2],
        ]
    )
    model = module.SourceModel(
        base_features=np.zeros((4, 1)),
        sensor_features=np.zeros((4, 8, 1)),
        full_residuals=np.zeros((4, 2)),
        task_residuals=task,
        class_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        action_prototypes=prototypes,
        state_representation=np.zeros((4, 1)),
        query_representation=np.zeros((4, 1)),
        base_mean=np.zeros(1),
        base_scale=np.ones(1),
        sensor_mean=np.zeros((8, 1)),
        sensor_scale=np.ones((8, 1)),
        loss_floor=1e-6,
        class_counts=np.asarray([2, 2], dtype=np.int64),
    )
    summary = module.action_competition_summary(model)
    assert summary["action_count"] == 3
    assert summary["pointwise_active_action_count"] >= 2
    assert summary["minimum_pairwise_action_rmse"] > 0.0


def test_calibration_selection_respects_harm_gate_and_tie_breaks() -> None:
    summaries = [
        {
            "sensor_log_likelihood_scale": 1.0,
            "action_prototype_scale": 1.0,
            "regret_tolerance": 0.2,
            "eligible": False,
            "mean_trajectory_improvement": 0.20,
            "nonfallback_fraction": 0.9,
            "mean_measurement_cost": 1.0,
            "nonfallback_harmful_fraction": 0.2,
        },
        {
            "sensor_log_likelihood_scale": 2.0,
            "action_prototype_scale": 0.75,
            "regret_tolerance": 0.05,
            "eligible": True,
            "mean_trajectory_improvement": 0.10,
            "nonfallback_fraction": 0.5,
            "mean_measurement_cost": 2.0,
            "nonfallback_harmful_fraction": 0.01,
        },
        {
            "sensor_log_likelihood_scale": 1.0,
            "action_prototype_scale": 0.5,
            "regret_tolerance": 0.05,
            "eligible": True,
            "mean_trajectory_improvement": 0.10,
            "nonfallback_fraction": 0.5,
            "mean_measurement_cost": 1.0,
            "nonfallback_harmful_fraction": 0.01,
        },
    ]
    selected = module.select_calibration(summaries)
    assert selected.gate_passed
    assert selected.sensor_log_likelihood_scale == 1.0
    assert selected.action_prototype_scale == 0.5


def test_scoring_uses_the_frozen_action() -> None:
    context = dataclasses.replace(
        _context(),
        actions=np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
    )
    state = module.decision_state(context, {}, 1.0)
    plan = module.FrozenPlan(
        policy="decision_regret",
        budget=1,
        certified=True,
        action_index=1,
        sensor_count=1,
        measurement_cost=1.0,
        selected_internal_nodes=(5,),
        state=state,
    )
    model = module.SourceModel(
        base_features=np.zeros((1, 1)),
        sensor_features=np.zeros((1, 8, 1)),
        full_residuals=np.zeros((1, 2)),
        task_residuals=np.zeros((1, 2)),
        class_labels=np.zeros(1, dtype=np.int64),
        action_prototypes=np.zeros((1, 2)),
        state_representation=np.zeros((1, 1)),
        query_representation=np.zeros((1, 1)),
        base_mean=np.zeros(1),
        base_scale=np.ones(1),
        sensor_mean=np.zeros((8, 1)),
        sensor_scale=np.ones((8, 1)),
        loss_floor=1e-6,
        class_counts=np.ones(1, dtype=np.int64),
    )
    scored = module.score_plan(plan, context, np.asarray([1.0, 0.0]), model)
    assert scored["physical_task_mse"] == 0.0
    assert scored["fallback_task_mse"] > 0.0
    assert scored["harmful_vs_fallback"] is False


def test_source_text_freezes_plans_before_target_residual_slice() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index("plans: list[FrozenPlan]")
    target = text.index("full_target = extract_full_target_residual", start)
    assert start < target
    prefix = text[start:target]
    assert "acquisition_path(" in prefix
    assert "score_plan(" not in prefix
