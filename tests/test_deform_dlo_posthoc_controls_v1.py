from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.deform_dlo_posthoc_controls_v1.evaluate import load_protocol
from experiments.deform_dlo_posthoc_controls_v1.model import (
    FEATURE_COUNT,
    candidate_from_canonical,
    deterministic_subset_indices,
    equal_dlo_bootstrap,
    feature_indices,
    feature_layout,
    fit_linear_residual,
    predict_linear_residual,
    score_arm,
)


def test_protocol_is_frozen_post_open_control() -> None:
    protocol = load_protocol(
        Path("experiments/deform_dlo_posthoc_controls_v1/protocol.json")
    )
    assert protocol["evidence_class"] == "retrospective-post-open-control-study"
    assert protocol["data"]["new_data_collection"] is False
    assert protocol["parent"]["both_target_results_already_open"] is True
    assert protocol["runtime"]["runner_labels"][-1] == "gpuserver4090"


def test_feature_layout_and_ablation_masks_are_fixed() -> None:
    layout = feature_layout()
    assert layout["time"] == slice(0, 4)
    assert layout["dynamic_action_velocity"].stop == FEATURE_COUNT
    full = feature_indices("full")
    time_only = feature_indices("time_only_ridge")
    no_action = feature_indices("no_explicit_action_features")
    no_dynamics = feature_indices("no_baseline_dynamics_features")
    assert np.array_equal(full, np.arange(FEATURE_COUNT))
    assert np.array_equal(time_only, np.arange(4))
    assert 24 not in no_action
    assert 15 not in no_dynamics
    assert no_action.size < full.size
    assert no_dynamics.size < full.size


def test_source_subsets_are_deterministic_nested_and_target_independent() -> None:
    names = tuple(f"trajectory-{index:02d}.pkl" for index in range(12))
    small = deterministic_subset_indices(
        names,
        dlo="DLO4",
        repeat=2,
        count=4,
        domain="test-domain",
    )
    large = deterministic_subset_indices(
        names,
        dlo="DLO4",
        repeat=2,
        count=8,
        domain="test-domain",
    )
    repeated = deterministic_subset_indices(
        names,
        dlo="DLO4",
        repeat=2,
        count=4,
        domain="test-domain",
    )
    assert np.array_equal(small, repeated)
    assert set(small).issubset(set(large))
    assert not np.array_equal(
        small,
        deterministic_subset_indices(
            names,
            dlo="DLO5",
            repeat=2,
            count=4,
            domain="test-domain",
        ),
    )


def test_linear_residual_fit_predict_and_clamped_identity() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(5, 7, 3, FEATURE_COUNT))
    residual = np.zeros((5, 7, 3, 3), dtype=np.float64)
    residual[..., 0] = 0.5 + 1.25 * features[..., 0]
    residual[..., 1] = -0.25 * features[..., 1]
    residual[..., 2] = features[..., 2] - features[..., 3]
    model = fit_linear_residual(
        features,
        residual,
        np.arange(5),
        selected_features=np.arange(4),
        ridge=1e-8,
    )
    predicted = predict_linear_residual(model, features)
    assert np.max(np.abs(predicted - residual)) < 1e-6

    baseline = rng.normal(size=(5, 7, 7, 3))
    frames = np.broadcast_to(np.eye(3), (5, 3, 3)).copy()
    candidate = candidate_from_canonical(
        baseline,
        predicted,
        frames,
        shrinkage=0.25,
    )
    assert np.array_equal(candidate[:, :, :2], baseline[:, :, :2])
    assert np.array_equal(candidate[:, :, -2:], baseline[:, :, -2:])
    assert not np.array_equal(candidate[:, :, 2:-2], baseline[:, :, 2:-2])


def test_scoring_and_equal_dlo_bootstrap_use_complete_cases() -> None:
    target = np.zeros((4, 3, 6, 3), dtype=np.float64)
    baseline = np.ones_like(target)
    candidate = baseline * 0.8
    names = [f"case-{index}" for index in range(4)]
    summary = score_arm(candidate, baseline, target, names)
    assert summary["wins"] == 4
    assert summary["relative_improvement"] == np.testing.assert_allclose(
        summary["relative_improvement"], 0.2, rtol=0.0, atol=1e-15
    )

    interval = equal_dlo_bootstrap(
        {"DLO4": [0.8, 0.9, 0.7], "DLO5": [1.0, 0.8, 0.9]},
        {"DLO4": [1.0, 1.0, 1.0], "DLO5": [1.0, 1.0, 1.0]},
        replicates=500,
        seed=3,
    )
    assert interval["relative_improvement"] > 0.0
    assert interval["bootstrap_low"] <= interval["relative_improvement"]
    assert interval["bootstrap_high"] >= interval["relative_improvement"]
