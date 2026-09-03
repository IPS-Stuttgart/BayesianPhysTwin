from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

controls = importlib.import_module(
    "experiments.deform_dlo45_adapter_controls_v1.evaluate"
)
PROTOCOL = Path("experiments/deform_dlo45_adapter_controls_v1/protocol.json")


def test_protocol_and_feature_masks_are_frozen() -> None:
    protocol = controls.load_protocol(PROTOCOL)
    assert protocol["contract"] == "deform-dlo45-adapter-controls-v1"
    assert np.array_equal(controls.ALL_FEATURES, np.arange(92))
    assert controls.NO_EXPLICIT_ACTION_FEATURES.size == 36
    assert controls.INITIAL_ACTION_ONLY_FEATURES.size == 48
    for mask in (
        controls.NO_EXPLICIT_ACTION_FEATURES,
        controls.INITIAL_ACTION_ONLY_FEATURES,
    ):
        assert np.array_equal(mask, np.unique(mask))
        assert np.all(mask >= 0)
        assert np.all(mask < 92)


def test_hash_order_and_balanced_folds_are_deterministic() -> None:
    names = tuple(f"{index}.pkl" for index in range(14))
    first = controls._hash_order(names, domain="test", dlo="DLO4", replicate=3)
    second = controls._hash_order(names, domain="test", dlo="DLO4", replicate=3)
    changed = controls._hash_order(names, domain="test", dlo="DLO4", replicate=4)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert sorted(first.tolist()) == list(range(14))

    folds = controls._balanced_folds(
        names,
        domain="test-fold",
        dlo="DLO5",
        folds=7,
    )
    assert folds.shape == (14,)
    assert sorted(np.bincount(folds).tolist()) == [2] * 7


def test_score_prediction_uses_complete_trajectory_units() -> None:
    target = np.zeros((2, 3, 12, 3), dtype=np.float64)
    baseline = np.ones_like(target)
    candidate = baseline.copy()
    candidate[0] *= 0.5
    candidate[1] *= 1.5
    summary = controls.score_prediction(candidate, baseline, target, ("a", "b"))
    assert summary["candidate_mean_l1_m"] == 1.0
    assert summary["baseline_mean_l1_m"] == 1.0
    assert summary["relative_improvement"] == 0.0
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 0
    assert summary["case_names"] == ["a", "b"]


def test_trivial_templates_have_declared_shapes() -> None:
    residual = np.arange(4 * 5 * 8 * 3, dtype=np.float64).reshape(4, 5, 8, 3)
    expected = {
        "global_bias": (3,),
        "node_bias": (8, 3),
        "time_node_mean": (5, 8, 3),
    }
    for kind, shape in expected.items():
        template = controls._fit_trivial_template(residual, kind)
        assert template.shape == shape
        broadcast = controls._broadcast_trivial_template(
            template,
            kind=kind,
            trajectory_count=2,
            horizon=5,
            internal_count=8,
        )
        assert broadcast.shape == (2, 5, 8, 3)


def test_duplicate_queries_average_only_the_targets() -> None:
    initial = np.zeros((3, 2, 12, 3), dtype=np.float64)
    action = np.zeros((3, 4, 4, 3), dtype=np.float64)
    baseline = np.zeros((3, 4, 12, 3), dtype=np.float64)
    target = np.zeros_like(baseline)
    initial[2] = 1.0
    action[2] = 1.0
    baseline[2] = 1.0
    target[0] = 2.0
    target[1] = 4.0
    target[2] = 7.0

    grouped = controls._collapse_duplicate_queries(initial, action, baseline, target)
    assert grouped[0].shape[0] == 2
    assert np.all(grouped[3][0] == 3.0)
    assert np.all(grouped[3][1] == 7.0)
