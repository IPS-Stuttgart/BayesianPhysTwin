from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.deform_dlo45_decision_directed_sensing_v1 import evaluate as core
from experiments.deform_dlo45_decision_directed_sensing_v2 import evaluate as module

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "experiments" / "deform_dlo45_decision_directed_sensing_v2" / "protocol.json"
)
SCRIPT = (
    ROOT / "experiments" / "deform_dlo45_decision_directed_sensing_v2" / "evaluate.py"
)
WORKFLOW = (
    ROOT / ".github" / "workflows" / "deform-dlo45-decision-directed-sensing-v2.yml"
)


def test_protocol_uses_disjoint_source_calibration_and_test_splits() -> None:
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol = module.load_protocol(PROTOCOL)

    assert raw["evaluation"]["evaluation_split_opened"] is False
    assert raw["evaluation"]["new_data_collection"] is False
    assert protocol.core_template.fit_count == 39
    assert protocol.core_template.calibration_count == 9
    assert protocol.core_template.source_test_count == 8
    assert protocol.task_nodes == (4, 5, 6, 7)
    assert protocol.likelihood_scales == (1.0, 2.0, 4.0)
    assert protocol.regret_tolerances[-1] == 0.4


def test_task_index_projection_selects_only_registered_central_nodes() -> None:
    protocol = module.load_protocol(PROTOCOL)
    indices = module.task_flat_indices(protocol)
    shaped = np.zeros(
        (
            protocol.core_template.horizon_frames,
            8,
            3,
        ),
        dtype=bool,
    )
    shaped.reshape(-1)[indices] = True

    assert int(np.sum(shaped)) == (
        protocol.core_template.horizon_frames * len(protocol.task_nodes) * 3
    )
    assert np.all(shaped[:, [2, 3, 4, 5], :])
    assert not np.any(shaped[:, [0, 1, 6, 7], :])


def _synthetic_model() -> core.SourceModel:
    count = 16
    residual_dimension = 25 * 8 * 3
    base_features = np.linspace(-1.0, 1.0, count)[:, None]
    sensor_features = np.zeros((count, 8, 6), dtype=np.float64)
    residuals = np.zeros((count, residual_dimension), dtype=np.float64)
    for index in range(count):
        class_id = index // 4
        sensor_features[index, :, 0] = class_id
        residuals[index] = 0.1 * class_id
    return core.SourceModel(
        base_features=base_features,
        sensor_features=sensor_features,
        residuals=residuals,
        class_labels=np.repeat(np.arange(4), 4),
        state_representation=np.arange(count, dtype=np.float64)[:, None],
        query_representation=np.repeat(
            np.arange(4, dtype=np.float64),
            4,
        )[:, None],
        base_mean=np.zeros(1),
        base_scale=np.ones(1),
        sensor_mean=np.zeros((8, 6)),
        sensor_scale=np.ones((8, 6)),
        loss_floor=1e-6,
    )


def test_competing_context_contains_fallback_and_class_actions() -> None:
    protocol = module.load_protocol(PROTOCOL)
    observation = core.Observation(
        base_feature=np.zeros(1),
        sensor_features=np.zeros((8, 6)),
        baseline=np.zeros((25, 8, 3)),
        length_scale=0.5,
    )
    context = module.make_competing_context(
        observation,
        _synthetic_model(),
        protocol,
    )

    assert context.fixed_actions.shape[0] >= 3
    assert np.all(context.fixed_actions[0] == 0.0)
    assert context.action_labels[0] == "physical_fallback"
    assert context.relative_losses.shape == (
        protocol.core_template.support_neighbors,
        context.fixed_actions.shape[0],
    )
    assert np.array_equal(
        np.unique(context.support_classes),
        np.arange(len(np.unique(context.support_classes))),
    )


def test_scoring_preserves_task_and_full_fallbacks() -> None:
    protocol = module.load_protocol(PROTOCOL)
    dimension = 25 * 8 * 3
    indices = module.task_flat_indices(protocol)
    context = module.CompetingContext(
        support_indices=np.arange(2, dtype=np.int64),
        base_logits=np.zeros(2),
        support_sensor_features=np.zeros((2, 8, 6)),
        target_sensor_features=np.zeros((8, 6)),
        support_residuals=np.zeros((2, dimension)),
        support_classes=np.array([0, 1], dtype=np.int64),
        support_state_representation=np.zeros((2, 1)),
        support_query_representation=np.zeros((2, 1)),
        fixed_actions=np.vstack(
            (
                np.zeros(dimension),
                np.ones(dimension),
            )
        ),
        relative_losses=np.zeros((2, 2)),
        length_scale=0.1,
        task_flat_indices=indices,
        action_labels=("physical_fallback", "class_0"),
    )
    target = np.ones(dimension)
    plan = {
        "action_index": 1,
        "nonfallback": True,
        "certified": True,
        "sensor_count": 1,
        "selected_internal_nodes": [5],
    }
    scored = module.score_plan(plan, context, target)

    assert scored["task_mse"] == 0.0
    assert scored["full_mse"] == 0.0
    assert scored["task_fallback_mse"] > 0.0
    assert scored["full_fallback_mse"] > 0.0
    assert scored["harmful_vs_task_fallback"] is False


def test_source_freezes_paths_before_slicing_future_internal_truth() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    path_freeze = text.index("plans: list[dict[str, object]]")
    target_slice = text.index("target = core.extract_target_residual")
    assert path_freeze < target_slice
    prefix = text[path_freeze:target_slice]
    assert "full_acquisition_path(" in prefix
    assert "score_plan(" not in prefix


def test_workflow_is_file_triggered_and_excludes_official_evaluation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "deform-dlo45-decision-directed-sensing-v2.json" in text
    assert "runs-on: [self-hosted, gpuserver4090]" in text
    assert 'test "$RUNNER_NAME" = "workstation1"' in text
    assert 'test ! -e "$isolated/$dlo/eval"' in text
    assert "new_data_collection_authorized" in text
    assert "future_internal_outcome_access_before_action_selection" in text
    assert "contents: write" not in text
    assert "secrets." not in text
