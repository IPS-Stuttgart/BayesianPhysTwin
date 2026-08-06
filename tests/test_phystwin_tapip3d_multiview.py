from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli.phystwin_tapip3d_multiview import (
    build_parser,
    frame_zero_depth_eligibility,
)
from bayesian_phystwin.phystwin_tapip3d_competence import Tapip3dPrediction
from bayesian_phystwin.phystwin_tapip3d_multiview import (
    MultiviewTapip3dPrediction,
    apply_exact_identity_fallback,
    evaluate_multiview_tapip3d_gates,
    fuse_tapip3d_views,
    load_multiview_tapip3d_prediction,
    save_multiview_tapip3d_prediction,
)


def _queries() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )


def _view(offset_m: float, *, query_indices: tuple[int, ...] = (0, 1)) -> Tapip3dPrediction:
    queries = _queries()[list(query_indices)]
    coords = np.repeat(queries[None, :, 1:], 3, axis=0)
    coords[0, :, 0] += offset_m
    coords[1:, :, 1] += np.asarray([0.01, 0.02])[:, None]
    coords[1:, :, 0] += offset_m
    return Tapip3dPrediction(
        coords_world_m=coords,
        valid=np.ones(coords.shape[:2], dtype=bool),
        query_points=queries,
    )


def test_multiview_reanchors_static_per_view_bias() -> None:
    fused = fuse_tapip3d_views([_view(0.003), _view(-0.004)], _queries())

    np.testing.assert_allclose(fused.coords_world_m[0], _queries()[:, 1:])
    np.testing.assert_allclose(fused.coords_world_m[2, :, 1], [0.02, 0.02])
    np.testing.assert_array_equal(fused.valid, np.ones((3, 2), dtype=bool))


def test_frame_zero_depth_eligibility_is_geometry_only() -> None:
    queries = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    intrinsics = np.asarray([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])
    depth = np.ones((3, 3), dtype=float)

    eligible, pixels, error = frame_zero_depth_eligibility(
        queries,
        intrinsics,
        np.eye(4),
        depth,
        maximum_depth_error_m=0.01,
    )

    np.testing.assert_array_equal(eligible, [True, False])
    np.testing.assert_allclose(pixels[0], [1.0, 1.0])
    assert error[0] == pytest.approx(0.0)


def test_seal_fusion_parser_has_no_score_target_argument() -> None:
    args = build_parser().parse_args(
        [
            "seal-fusion",
            "--protocol",
            "protocol.json",
            "--prediction-manifest",
            "camera0.json",
            "--prediction-manifest",
            "camera1.json",
            "--output-dir",
            "sealed",
        ]
    )

    assert not hasattr(args, "case_dir")
    assert not hasattr(args, "gt_track")
    assert not hasattr(args, "manual_trajectory")


def test_partial_view_query_sets_map_to_global_identities() -> None:
    fused = fuse_tapip3d_views(
        [_view(0.0), _view(0.0, query_indices=(1,))],
        _queries(),
    )

    assert not np.any(fused.valid[:, 0])
    assert np.all(fused.valid[:, 1])
    np.testing.assert_array_equal(fused.view_count[:, 0], [1, 1, 1])
    np.testing.assert_array_equal(fused.view_count[:, 1], [2, 2, 2])


def test_disagreement_gate_rejects_without_hiding_view_support() -> None:
    first = _view(0.0)
    second = _view(0.0)
    changed = second.coords_world_m.copy()
    changed[1:, :, 2] += 0.03
    second = Tapip3dPrediction(changed, second.valid, second.query_points)

    fused = fuse_tapip3d_views(
        [first, second],
        _queries(),
        maximum_pairwise_disagreement_m=0.02,
    )

    assert np.all(fused.valid[0])
    assert not np.any(fused.valid[1:])
    np.testing.assert_array_equal(fused.view_count, 2)
    assert np.all(fused.max_pairwise_disagreement_m[1:] == pytest.approx(0.03))


def test_duplicate_correlated_camera_does_not_tighten_covariance() -> None:
    two = fuse_tapip3d_views([_view(0.0), _view(0.0)], _queries())
    three = fuse_tapip3d_views(
        [_view(0.0), _view(0.0), _view(0.0)],
        _queries(),
    )

    np.testing.assert_array_equal(
        two.observation_covariance_m2,
        three.observation_covariance_m2,
    )
    floor_variance = 0.005**2
    assert np.all(np.diagonal(two.observation_covariance_m2, axis1=-2, axis2=-1) >= floor_variance)
    assert floor_variance > floor_variance / 2.0


def test_unknown_correlation_is_more_conservative_than_independent_fusion() -> None:
    fused = fuse_tapip3d_views([_view(0.0), _view(0.0)], _queries())
    variance = fused.observation_covariance_m2[0, 0, 0, 0]

    assert variance == pytest.approx(0.005**2)
    assert variance >= 0.005**2 / 2.0


def test_exact_fallback_preserves_unsupported_baseline_bits() -> None:
    fused = fuse_tapip3d_views(
        [_view(0.0), _view(0.0, query_indices=(1,))],
        _queries(),
    )
    baseline = np.arange(18, dtype=np.float32).reshape(3, 2, 3)

    output, weight = apply_exact_identity_fallback(fused, baseline)

    assert output.dtype == baseline.dtype
    assert output[:, 0].tobytes() == baseline[:, 0].tobytes()
    np.testing.assert_array_equal(weight[:, 0], 0.0)
    np.testing.assert_array_equal(weight[:, 1], 1.0)


def test_multiview_carrier_round_trip(tmp_path: Path) -> None:
    source = fuse_tapip3d_views([_view(0.0), _view(0.001)], _queries())
    path = tmp_path / "multiview.npz"

    save_multiview_tapip3d_prediction(path, source)
    restored = load_multiview_tapip3d_prediction(path)

    np.testing.assert_allclose(restored.coords_world_m, source.coords_world_m)
    np.testing.assert_array_equal(restored.valid, source.valid)
    np.testing.assert_allclose(
        restored.observation_covariance_m2,
        source.observation_covariance_m2,
    )


def test_multiview_gate_requires_all_conditions() -> None:
    metrics = {
        "support_fraction": 0.75,
        "displacement_rmse_m": 0.004,
        "frame_zero_anchor_rmse_m": 0.0,
    }
    late = {"support_fraction": 0.6, "displacement_rmse_m": 0.009}
    passed = evaluate_multiview_tapip3d_gates(
        metrics,
        late,
        0.25,
        minimum_support_fraction=0.7,
        maximum_displacement_rmse_m=0.005,
        maximum_frame_zero_anchor_rmse_m=0.002,
        minimum_late_support_fraction=0.5,
        maximum_late_displacement_rmse_m=0.01,
        minimum_best_single_shared_improvement_fraction=0.2,
    )
    assert passed["competence_gate_passed"]

    failed = evaluate_multiview_tapip3d_gates(
        metrics,
        late,
        0.19,
        minimum_support_fraction=0.7,
        maximum_displacement_rmse_m=0.005,
        maximum_frame_zero_anchor_rmse_m=0.002,
        minimum_late_support_fraction=0.5,
        maximum_late_displacement_rmse_m=0.01,
        minimum_best_single_shared_improvement_fraction=0.2,
    )
    assert not failed["competence_gate_passed"]


def test_loader_rejects_nonpositive_covariance(tmp_path: Path) -> None:
    prediction = MultiviewTapip3dPrediction(
        coords_world_m=np.zeros((1, 1, 3)),
        valid=np.ones((1, 1), dtype=bool),
        query_points=_queries()[:1],
        observation_covariance_m2=np.zeros((1, 1, 3, 3)),
        view_count=np.full((1, 1), 2, dtype=np.int16),
        max_pairwise_disagreement_m=np.zeros((1, 1)),
    )
    path = tmp_path / "bad.npz"
    save_multiview_tapip3d_prediction(path, prediction)

    with pytest.raises(ValueError, match="positive definite"):
        load_multiview_tapip3d_prediction(path)
