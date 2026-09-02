from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from experiments.deform_dlo45_decision_directed_sensing_v1 import (
    evaluate as module,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "deform_dlo45_decision_directed_sensing_v1"
    / "protocol.json"
)
SCRIPT = (
    ROOT
    / "experiments"
    / "deform_dlo45_decision_directed_sensing_v1"
    / "evaluate.py"
)


def _protocol() -> module.Protocol:
    return module.load_protocol(PROTOCOL)


def _trajectory() -> np.ndarray:
    trajectory = np.zeros((module.FRAME_COUNT, module.NODE_COUNT, 3))
    trajectory[:, module.NODE_COUNT - 2 :, 0] = 1.0
    weights = np.linspace(1.0 / 9.0, 8.0 / 9.0, 8)
    trajectory[:, 2:10, 0] = weights[None, :]
    return trajectory


def test_protocol_is_source_test_only_and_collects_no_new_data() -> None:
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol = _protocol()
    assert raw["evaluation"]["primary_stage"] == "source-test-only-pilot"
    assert raw["evaluation"]["evaluation_split_opened"] is False
    assert raw["evaluation"]["new_data_collection"] is False
    assert protocol.policies == module.EXPECTED_POLICIES
    assert protocol.budgets == (0, 1, 2, 3, 4, 6, 8)


def test_endpoint_observation_does_not_touch_future_internal_outcomes() -> None:
    protocol = _protocol()
    trajectory = _trajectory()
    current = protocol.first_current_frame
    future = slice(current + 1, current + 1 + protocol.horizon_frames)
    trajectory[future, module.INTERNAL, :] = np.nan

    observation = module.extract_endpoint_observation(
        trajectory,
        current,
        protocol,
    )

    assert np.all(np.isfinite(observation.base_feature))
    assert np.all(np.isfinite(observation.sensor_features))
    assert np.all(np.isfinite(observation.baseline))
    assert np.any(
        ~np.isfinite(
            module.extract_target_residual(
                trajectory,
                current,
                observation,
                protocol,
            )
        )
    )


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


def test_random_acquisition_order_is_deterministic_per_case() -> None:
    protocol = _protocol()
    first = module._stable_random_order("DLO4/1.pkl/4", protocol)
    second = module._stable_random_order("DLO4/1.pkl/4", protocol)
    other = module._stable_random_order("DLO4/2.pkl/4", protocol)
    assert first == second
    assert sorted(first) == list(range(8))
    assert first != other


def test_posterior_sensor_update_is_normalized_and_informative() -> None:
    protocol = dataclasses.replace(
        _protocol(),
        sensor_log_likelihood_scale=2.0,
    )
    context = module.CaseContext(
        support_indices=np.arange(4, dtype=np.int64),
        base_logits=np.zeros(4),
        support_sensor_features=np.asarray(
            [
                [[-2.0, 0.0]],
                [[-1.8, 0.0]],
                [[1.8, 0.0]],
                [[2.0, 0.0]],
            ]
        ),
        target_sensor_features=np.asarray([[-2.0, 0.0]]),
        support_residuals=np.zeros((4, 3)),
        support_classes=np.asarray([0, 0, 1, 1], dtype=np.int64),
        support_state_representation=np.asarray(
            [[-1.0], [-0.8], [0.8], [1.0]]
        ),
        support_query_representation=np.asarray(
            [[-1.0], [-0.8], [0.8], [1.0]]
        ),
        fixed_actions=np.zeros((3, 3)),
        relative_losses=np.asarray(
            [
                [1.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        ),
        length_scale=1.0,
    )
    prior = module.posterior_weights(context, {}, protocol)
    posterior = module.posterior_weights(
        context,
        {0: context.target_sensor_features[0]},
        protocol,
    )
    assert np.isclose(np.sum(prior), 1.0)
    assert np.isclose(np.sum(posterior), 1.0)
    assert posterior[:2].sum() > 0.99
    assert module._weighted_variance(
        context.support_state_representation,
        posterior,
    ) < module._weighted_variance(
        context.support_state_representation,
        prior,
    )


def test_scoring_uses_truth_only_after_the_plan_is_fixed() -> None:
    context = module.CaseContext(
        support_indices=np.arange(2, dtype=np.int64),
        base_logits=np.zeros(2),
        support_sensor_features=np.zeros((2, 8, 6)),
        target_sensor_features=np.zeros((8, 6)),
        support_residuals=np.zeros((2, 3)),
        support_classes=np.asarray([0, 0], dtype=np.int64),
        support_state_representation=np.zeros((2, 1)),
        support_query_representation=np.zeros((2, 1)),
        fixed_actions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        relative_losses=np.zeros((2, 3)),
        length_scale=0.1,
    )
    plan = {
        "action_index": 1,
        "nonfallback": True,
        "certified": True,
        "sensor_count": 1,
        "selected_internal_nodes": [5],
    }
    scored = module.score_plan(
        plan,
        context,
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert scored["physical_mse"] == 0.0
    assert scored["fallback_mse"] > 0.0
    assert scored["harmful_vs_fallback"] is False


def test_source_text_freezes_plans_before_target_residual_slice() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    plans = text.index("plans: list[dict[str, object]]")
    target = text.index("target_residual = extract_target_residual")
    assert plans < target
    prefix = text[plans:target]
    assert "acquisition_path(" in prefix
    assert "score_plan(" not in prefix
