from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bayesian_phystwin.pokeflex_action_discrepancy import (
    apply_bounded_translation,
    causal_action_features,
    fit_translation_ridge,
    robust_nearest_translation_m,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = (
    ROOT / "scripts" / "development" / "evaluate_pokeflex_action_discrepancy_v1.py"
)


def _evaluator() -> object:
    spec = importlib.util.spec_from_file_location(
        "evaluate_pokeflex_action_discrepancy_v1",
        EVALUATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records() -> list[dict[str, object]]:
    result = []
    for index in range(5):
        tool = np.eye(4)
        tool[:3, 3] = [0.01 * index, 0.0, 0.0]
        end_effector = np.eye(4)
        end_effector[:3, 3] = [0.0, 0.02 * index, 0.0]
        result.append(
            {
                "forces": [float(index), 2.0 * index, -float(index)],
                "T_WT": tool.tolist(),
                "T_WE": end_effector.tolist(),
            }
        )
    return result


def test_robust_nearest_translation_recovers_shift_with_outlier() -> None:
    prediction = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [50.0, 0.0, 0.0]]
    )
    shift = np.asarray([0.1, -0.2, 0.3])
    target = prediction[:3] + shift

    result = robust_nearest_translation_m(
        prediction,
        target,
        retained_fraction=0.75,
    )

    assert np.allclose(result, shift)
    assert not result.flags.writeable


def test_causal_action_features_are_residual_independent_and_fixed_size() -> None:
    template = np.zeros((4, 3), dtype=np.float64)
    prediction = np.full((4, 3), [0.01, -0.02, 0.0])

    result = causal_action_features(
        _records(),
        template_vertices_m=template,
        predicted_vertices_m=prediction,
    )

    assert result.shape == (24,)
    assert np.all(np.isfinite(result))
    assert np.allclose(result[18:21], [0.01, -0.02, 0.0])


def test_translation_ridge_equalizes_groups_and_caps_predictions() -> None:
    features = np.asarray([[0.0], [0.0], [0.0], [1.0]])
    target = np.asarray(
        [
            [0.001, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.003, 0.0, 0.0],
        ]
    )
    model = fit_translation_ridge(
        features,
        target,
        np.asarray(["a", "a", "a", "b"]),
        ridge_penalty=0.0,
        maximum_translation_m=0.002,
    )

    prediction = model.predict(np.asarray([[0.0], [1.0], [10.0]]))

    assert np.allclose(prediction[:2, 0], [0.001, 0.002])
    assert np.all(np.linalg.norm(prediction, axis=1) <= 0.002 + 1e-15)
    assert not prediction.flags.writeable


def test_scale_zero_is_exact_fallback() -> None:
    values = np.asarray(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    translation = np.full((2, 3), 0.1)

    result = apply_bounded_translation(values, translation, scale=0.0)

    assert np.array_equal(result, values)


def test_evaluator_translation_score_and_object_balance() -> None:
    evaluator = _evaluator()
    baseline = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    shift = np.asarray([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3]])
    target = baseline + shift[:, None]

    score = evaluator._score_translation(baseline, target, shift, 1.0)
    rows = [
        {"object": "a", "take": "1", "metric": 1.0},
        {"object": "a", "take": "1", "metric": 3.0},
        {"object": "a", "take": "2", "metric": 6.0},
        {"object": "b", "take": "1", "metric": 10.0},
    ]

    assert np.allclose(score, 0.0)
    assert evaluator._object_balanced(rows, "metric") == 7.0
