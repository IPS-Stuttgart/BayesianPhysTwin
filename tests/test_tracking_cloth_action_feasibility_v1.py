from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.tracking_cloth_action_feasibility_v1._decision import (
    _probe_binary_outcomes,
    decision_grid,
)
from experiments.tracking_cloth_action_feasibility_v1._metrics import (
    causal_fill_truth,
    cloth_grid_edges,
    nonneighbor_pairs,
    pairwise_shape_change,
    physical_action_metrics,
    read_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "experiments" / "tracking_cloth_action_feasibility_v1" / "protocol.json"
)


def _regular_cloth(time_count: int = 8) -> np.ndarray:
    grid = np.zeros((20, 3), dtype=np.float64)
    for index in range(20):
        row, column = divmod(index, 4)
        grid[index, :2] = [0.08 * column, 0.08 * row]
    return np.repeat(grid[None, :, :], time_count, axis=0)


def test_registered_grid_topology_is_complete() -> None:
    edges = cloth_grid_edges()
    pairs = nonneighbor_pairs()

    assert edges.shape == (31, 2)
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert len({tuple(row) for row in edges.tolist()}) == 31
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_pairwise_probe_feature_is_rigid_translation_invariant() -> None:
    points = _regular_cloth(2)
    points[1] += np.asarray([1.0, -2.0, 0.5])

    assert pairwise_shape_change(points, diameter=0.5) == pytest.approx(0.0)


def test_zero_deformation_has_zero_registered_task_loss() -> None:
    truth = _regular_cloth()
    metrics = physical_action_metrics(
        truth,
        cutoff=3,
        contact_distance_m=0.035,
        edge_strain_weight=0.25,
        edge_strain_quantile=0.95,
        initial_diameter_m=0.5,
    )

    assert metrics["task_loss"] == pytest.approx(0.0)
    assert metrics["contact_fraction"] == pytest.approx(0.0)
    assert metrics["contact_depth_rms"] == pytest.approx(0.0)
    assert metrics["edge_strain_quantile"] == pytest.approx(0.0)
    assert metrics["probe_feature"] == pytest.approx(0.0)


def test_causal_truth_fill_preserves_observed_values() -> None:
    truth = _regular_cloth(4)
    truth[1, 3, 2] = np.nan
    truth[2, 3, 2] = 0.25

    filled, missing_fraction = causal_fill_truth(truth)

    assert filled[1, 3, 2] == pytest.approx(filled[0, 3, 2])
    assert filled[2, 3, 2] == pytest.approx(0.25)
    assert missing_fraction == pytest.approx(1.0 / truth.size)


def test_contact_and_edge_stretch_both_increase_task_loss() -> None:
    truth = _regular_cloth()
    truth[4:, 19] = truth[4:, 0]
    truth[4:, 1, 0] += 0.08
    metrics = physical_action_metrics(
        truth,
        cutoff=3,
        contact_distance_m=0.035,
        edge_strain_weight=0.25,
        edge_strain_quantile=0.95,
        initial_diameter_m=0.5,
    )

    assert metrics["task_loss"] > 0.0
    assert metrics["contact_depth_rms"] > 0.0
    assert metrics["edge_strain_peak"] > 0.0


def test_probe_thresholds_are_source_only_and_binary() -> None:
    features = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [1.0] * 8,
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    outcomes, thresholds = _probe_binary_outcomes(features)

    assert outcomes.shape == features.shape
    assert thresholds[0] == pytest.approx(0.45)
    assert set(np.unique(outcomes[0])) == {0, 1}
    assert set(np.unique(outcomes[1])) == {0}
    assert set(np.unique(outcomes[2])) == {0, 1}


def test_source_grid_can_exercise_a_decision_probe() -> None:
    protocol = read_protocol(PROTOCOL)
    blocks = [
        (material, repetition)
        for material in protocol["materials"]
        for repetition in protocol["source_repetitions"]
    ]
    losses = np.asarray(
        [
            [0.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    probes = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
        ],
        dtype=np.int64,
    )

    records, summary = decision_grid(blocks, losses, probes, protocol)

    assert len(records) == 16
    assert summary["informative_probe_indices"] == [0, 2]
    assert summary["informative_probe_names"] == [
        "four_corners_normal",
        "two_corners_normal",
    ]
    selected = summary["selected_source_setting"]
    assert selected["mode_counts"]["sense"] >= 1
    assert selected["relative_gain_vs_fallback"] > 0.0


def test_protocol_keeps_rep3_numerically_closed() -> None:
    protocol = read_protocol(PROTOCOL)
    boundary = protocol["information_boundary"]

    assert protocol["source_repetitions"] == [1, 2]
    assert protocol["reserved_target_repetition"] == 3
    assert boundary["target_rep3_numeric_outcomes_read"] is False
    assert boundary["target_protocol_authorized"] is False
    assert boundary["raw_trajectory_upload"] is False
