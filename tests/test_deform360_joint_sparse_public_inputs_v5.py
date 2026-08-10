from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseExtractionConfigV5,
    Deform360JointSparsePrefixFitV5,
)
from bayesian_phystwin.deform360_joint_sparse_public_inputs_v5 import (
    estimate_deform360_last_causal_residual_v5,
    prepare_deform360_joint_sparse_visual_window_v5,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack(covariance: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            covariance[..., 0, 0],
            covariance[..., 0, 1],
            covariance[..., 0, 2],
            covariance[..., 1, 1],
            covariance[..., 1, 2],
            covariance[..., 2, 2],
        ),
        axis=-1,
    )


def _fixture(
    root: Path,
    *,
    contributors: int = 2,
    future_bias_m: float = 0.0,
    metric_cluster_count: int = 8,
) -> tuple[Path, Path, Deform360JointSparsePrefixFitV5]:
    frames = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    height = width = 16
    yy, xx = np.indices((height, width))
    base = np.stack(
        (
            0.004 * xx,
            0.004 * yy,
            0.30 + 0.001 * xx,
        ),
        axis=-1,
    )
    local = np.stack([base + [0.001 * index, 0.0, 0.0] for index in range(5)])
    # Row-vector convention used by MotionCrafter association.
    linear = 1.5 * np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.asarray([0.20, -0.10, 0.05])
    world = local @ linear + translation
    world[-1, ..., 0] += future_bias_m
    valid = np.ones((5, height, width), dtype=np.bool_)
    deform = np.ones_like(valid)
    covariance = np.broadcast_to(2e-6 * np.eye(3), (*valid.shape, 3, 3)).copy()
    decoded = root / "decoded.npz"
    np.savez(
        decoded,
        point_map=local.astype(np.float32),
        valid_mask=valid,
        scene_flow=np.zeros_like(local, dtype=np.float32),
        deform_mask=deform,
        frame_indices=frames,
        point_covariance_packed=_pack(covariance),
        flow_covariance_packed=_pack(covariance),
        contributors=np.full(valid.shape, contributors, dtype=np.uint16),
    )

    metric_points = np.full((4, height, width, 3), np.nan, dtype=np.float64)
    metric_valid = np.zeros((4, height, width), dtype=np.bool_)
    candidates = [
        (row, column)
        for row in (1, 5, 9, 13)
        for column in (1, 5, 9, 13)
    ][:metric_cluster_count]
    for row, column in candidates:
        metric_valid[0, row, column] = True
        metric_points[0, row, column] = world[0, row, column]
    metric = root / "metric.npz"
    np.savez(
        metric,
        frame_indices=np.arange(10, 14, dtype=np.int64),
        points_world_m=metric_points,
        valid_mask=metric_valid,
    )
    fit = Deform360JointSparsePrefixFitV5(
        fit_object_ids=("source-a",),
        source_artifact_ids={"fit.json": "a" * 64},
    )
    return decoded, metric, fit


def _prepare(
    decoded: Path,
    metric: Path,
    fit: Deform360JointSparsePrefixFitV5,
):
    return prepare_deform360_joint_sparse_visual_window_v5(
        camera_id="camera-0",
        decoded_uniform_path=decoded,
        metric_prefix_path=metric,
        raw_prefix_range_half_open=(10, 14),
        fit=fit,
        source_artifact_ids={"provider.json": "b" * 64},
        extraction_config=Deform360JointSparseExtractionConfigV5(
            measurement_stride_pixels=1,
            maximum_rows_per_window=4096,
        ),
        metric_cluster_size_pixels=4,
    )


def test_public_adapter_uses_only_causal_frames_and_metric_covariance(
    tmp_path: Path,
) -> None:
    decoded, metric, fit = _fixture(tmp_path, future_bias_m=20.0)
    rows, gauge = _prepare(decoded, metric, fit)

    assert set(rows.frame_indices) == {0, 1, 2, 3}
    assert np.all(rows.frame_indices < 4)
    assert gauge.raw_frame_index == 10
    assert gauge.independent_cluster_count == 8
    assert gauge.inlier_independent_cluster_count == 8
    assert gauge.inlier_rmse_m < 1e-6
    assert np.all(np.linalg.eigvalsh(rows.point_covariance_m2) > 0.0)
    assert rows.source_artifact_ids["prob4d-decoded-uniform/camera-0.npz"] == _sha256(
        decoded
    )
    assert "metric-gauge/camera-0.json" in rows.source_artifact_ids


def test_contributor_duplication_does_not_raise_prior_reliability(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    decoded_a, metric_a, fit_a = _fixture(first, contributors=1)
    decoded_b, metric_b, fit_b = _fixture(second, contributors=500)

    rows_a, _ = _prepare(decoded_a, metric_a, fit_a)
    rows_b, _ = _prepare(decoded_b, metric_b, fit_b)

    np.testing.assert_array_equal(rows_a.source_confidence, rows_b.source_confidence)
    assert np.all(rows_a.source_confidence == 1.0)
    assert np.all(rows_a.contributor_count == 1)
    assert np.all(rows_b.contributor_count == 500)


def test_public_adapter_rejects_weak_metric_cluster_support(tmp_path: Path) -> None:
    decoded, metric, fit = _fixture(tmp_path, metric_cluster_count=7)

    with pytest.raises(ValueError, match="eight independent causal clusters"):
        _prepare(decoded, metric, fit)


def test_public_adapter_rejects_metric_prefix_that_omits_a_causal_frame(
    tmp_path: Path,
) -> None:
    decoded, metric, fit = _fixture(tmp_path)
    with np.load(metric, allow_pickle=False) as archive:
        points = np.asarray(archive["points_world_m"][:-1])
        valid = np.asarray(archive["valid_mask"][:-1])
    np.savez(
        metric,
        frame_indices=np.arange(10, 13, dtype=np.int64),
        points_world_m=points,
        valid_mask=valid,
    )

    with pytest.raises(ValueError, match="complete causal range"):
        _prepare(decoded, metric, fit)


def test_last_causal_residual_is_duplicate_invariant_and_capped(
    tmp_path: Path,
) -> None:
    decoded, metric, fit = _fixture(tmp_path)
    rows, _ = _prepare(decoded, metric, fit)
    selected = np.flatnonzero(rows.frame_indices == 3)
    reference = np.asarray(rows.point_world_m[selected[:12]]) - np.asarray(
        [0.050, 0.0, 0.0]
    )
    physical = np.broadcast_to(reference[None], (76, *reference.shape)).copy()

    single = estimate_deform360_last_causal_residual_v5(
        visual_windows=(rows,),
        physical_prediction_m=physical,
        causal_frame_stop=4,
    )
    duplicate = estimate_deform360_last_causal_residual_v5(
        visual_windows=(rows, replace(rows, window_id="duplicate-window")),
        physical_prediction_m=physical,
        causal_frame_stop=4,
    )

    np.testing.assert_allclose(single, duplicate, atol=1e-14, rtol=0.0)
    assert np.max(np.linalg.norm(single, axis=1)) <= 0.030 + 1e-14


def test_last_causal_residual_rejects_post_cutoff_rows(tmp_path: Path) -> None:
    decoded, metric, fit = _fixture(tmp_path)
    rows, _ = _prepare(decoded, metric, fit)
    physical = np.zeros((76, 8, 3), dtype=np.float64)
    future = replace(rows, frame_indices=rows.frame_indices + 1)

    with pytest.raises(ValueError, match="post-cutoff"):
        estimate_deform360_last_causal_residual_v5(
            visual_windows=(future,),
            physical_prediction_m=physical,
            causal_frame_stop=4,
        )
