from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli.phystwin_tapip3d_competence import build_parser
from bayesian_phystwin.phystwin_tapip3d_competence import (
    IdentityTrajectory,
    build_same_query_cotracker3_trajectory,
    evaluate_tapip3d_competence_gates,
    identity_trajectory_metrics,
    load_canonical_tapip3d_prediction,
    load_tapip3d_prediction,
    save_canonical_tapip3d_prediction,
    shared_support_displacement_metrics,
    validate_tapip3d_prediction_contract,
)


def _query_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _coords(frame_count: int = 4) -> np.ndarray:
    query = _query_points()[:, 1:]
    coords = np.repeat(query[None], frame_count, axis=0)
    coords[:, :, 0] += 0.01 * np.arange(frame_count)[:, None]
    return coords


def test_official_tapip3d_archive_loads_and_round_trips(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.npz"
    np.savez(
        source,
        coords=_coords(),
        visibs=np.ones((4, 2), dtype=bool),
        query_points=_query_points(),
        video=np.zeros((4, 3, 8, 8), dtype=np.float32),
    )

    prediction = load_tapip3d_prediction(source)
    validate_tapip3d_prediction_contract(
        prediction,
        _query_points(),
        expected_frame_count=4,
    )
    canonical = tmp_path / "canonical.npz"
    save_canonical_tapip3d_prediction(canonical, prediction)
    restored = load_canonical_tapip3d_prediction(canonical)

    np.testing.assert_allclose(restored.coords_world_m, _coords())
    np.testing.assert_array_equal(restored.valid, np.ones((4, 2), dtype=bool))
    np.testing.assert_allclose(restored.query_points, _query_points())


def test_tapip3d_archive_rejects_malformed_visibility(
    tmp_path: Path,
) -> None:
    source = tmp_path / "malformed.npz"
    np.savez(
        source,
        coords=_coords(),
        visibs=np.ones((4, 2), dtype=np.float32),
        query_points=_query_points(),
    )

    with pytest.raises(ValueError, match="boolean visibility"):
        load_tapip3d_prediction(source)


def test_prediction_contract_rejects_query_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "official.npz"
    np.savez(
        source,
        coords=_coords(),
        visibs=np.ones((4, 2), dtype=bool),
        query_points=_query_points(),
    )
    prediction = load_tapip3d_prediction(source)
    changed = _query_points()
    changed[1, 1] += 0.001

    with pytest.raises(ValueError, match="queries differ"):
        validate_tapip3d_prediction_contract(
            prediction,
            changed,
            expected_frame_count=4,
        )


def test_seal_parser_has_no_score_target_or_cue_argument() -> None:
    args = build_parser().parse_args(
        [
            "seal-prediction",
            "--protocol",
            "protocol.json",
            "--input-manifest",
            "input.json",
            "--tapip3d-result",
            "result.npz",
            "--output-dir",
            "output",
        ]
    )

    assert not hasattr(args, "case_dir")
    assert not hasattr(args, "cues")
    assert not hasattr(args, "gt_track")
    assert not hasattr(args, "manual_tracks")


def test_same_query_cotracker_uses_only_frame_zero_nearest_nodes() -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    queries = np.asarray(
        [
            [1.9, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ]
    )
    points = np.repeat(nodes[None], 3, axis=0)
    points[1:, :, 1] += np.asarray([0.1, 0.2])[:, None]
    valid = np.ones((3, 3), dtype=bool)

    trajectory, association = build_same_query_cotracker3_trajectory(
        points,
        valid,
        nodes,
        queries,
    )

    np.testing.assert_array_equal(association.node_indices, [2, 0])
    np.testing.assert_allclose(association.distance_m, [0.1, 0.1])
    np.testing.assert_allclose(trajectory.coords_world_m[0], queries)
    np.testing.assert_allclose(
        trajectory.coords_world_m[2, :, 1],
        [0.2, 0.2],
    )


def test_identity_metrics_use_euclidean_displacement_rmse() -> None:
    target = _coords(frame_count=3).astype(float)
    candidate = target.copy()
    candidate[1:, :, 1] += 0.003
    valid = np.ones((3, 2), dtype=bool)

    metrics = identity_trajectory_metrics(
        IdentityTrajectory(candidate, valid),
        target,
    )

    expected = np.sqrt((0.0 + 4.0 * 0.003**2) / 6.0)
    assert metrics["support_fraction"] == 1.0
    assert metrics["frame_zero_anchor_rmse_m"] == 0.0
    assert metrics["displacement_rmse_m"] == pytest.approx(expected)
    assert metrics["translation_diagnostics"][
        "translation_removed_rmse_m"
    ] == pytest.approx(0.0)


def test_shared_support_comparison_is_matched() -> None:
    target = _coords(frame_count=4).astype(float)
    first = target.copy()
    second = target.copy()
    first[1:, :, 1] += 0.001
    second[1:, :, 1] += 0.002
    first_valid = np.ones((4, 2), dtype=bool)
    second_valid = np.ones((4, 2), dtype=bool)
    second_valid[2, 1] = False

    result = shared_support_displacement_metrics(
        IdentityTrajectory(first, first_valid),
        IdentityTrajectory(second, second_valid),
        target,
    )

    assert result["shared_count"] == 7
    assert result["first_relative_improvement_fraction"] == pytest.approx(0.5)


def test_competence_gate_requires_every_predeclared_condition() -> None:
    tapip3d = {
        "support_fraction": 0.8,
        "displacement_rmse_m": 0.004,
        "frame_zero_anchor_rmse_m": 0.001,
    }
    late = {
        "support_fraction": 0.6,
        "displacement_rmse_m": 0.009,
    }
    shared = {"first_relative_improvement_fraction": 0.25}

    passed = evaluate_tapip3d_competence_gates(
        tapip3d,
        late,
        shared,
        minimum_support_fraction=0.7,
        minimum_shared_rmse_improvement_fraction=0.2,
        maximum_displacement_rmse_m=0.005,
        maximum_frame_zero_anchor_rmse_m=0.002,
        minimum_late_support_fraction=0.5,
        maximum_late_displacement_rmse_m=0.010,
    )
    assert passed["competence_gate_passed"]

    late["support_fraction"] = 0.49
    failed = evaluate_tapip3d_competence_gates(
        tapip3d,
        late,
        shared,
        minimum_support_fraction=0.7,
        minimum_shared_rmse_improvement_fraction=0.2,
        maximum_displacement_rmse_m=0.005,
        maximum_frame_zero_anchor_rmse_m=0.002,
        minimum_late_support_fraction=0.5,
        maximum_late_displacement_rmse_m=0.010,
    )
    assert not failed["competence_gate_passed"]
