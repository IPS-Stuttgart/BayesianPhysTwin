from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_frame_zero_depth_initializer import (
    DepthSupportedFrameZeroInitializerConfig,
    build_depth_supported_multiview_surface,
    metric_depth_support_counts,
    select_depth_supported_frame_zero_point_cloud,
)
from bayesian_phystwin.deform360_frame_zero_initializer import (
    FrameZeroInitializerConfig,
    FrameZeroPointCloud,
)


def _visual_hull_config() -> FrameZeroInitializerConfig:
    return FrameZeroInitializerConfig(
        minimum_original_point_count=3,
        minimum_camera_count=3,
        cube_half_extent_m=0.5,
        voxel_resolution=16,
        consensus_fraction_of_peak=0.75,
        minimum_consensus_votes=3,
        maximum_mask_dilation_pixels=0,
        minimum_fallback_point_count=3,
        maximum_output_point_count=32,
    )


def _depth_config() -> DepthSupportedFrameZeroInitializerConfig:
    return DepthSupportedFrameZeroInitializerConfig(
        visual_hull=_visual_hull_config(),
        depth_tolerance_m=0.05,
        minimum_depth_camera_count=2,
        minimum_depth_support_views=1,
        minimum_depth_supported_point_count=3,
    )


def _camera_inputs() -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.24, 0.0, 1.2]],
        dtype=np.float32,
    )
    intrinsics = np.array(
        [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]
    )
    depth_a = np.zeros((12, 12), dtype=np.float32)
    depth_a[5, 5] = 1.0
    depth_a[5, 6] = 1.0
    depth_a[5, 7] = 1.2
    depth_b = depth_a.copy()
    depth_b[5, 6] = 1.2
    depths = {"a": depth_a, "b": depth_b}
    intrinsics_by_camera = {"a": intrinsics, "b": intrinsics.copy()}
    extrinsics = {"a": np.eye(4), "b": np.eye(4)}
    return points, depths, intrinsics_by_camera, extrinsics


def test_source_v2_config_is_hash_locked_and_reuses_the_open_panel() -> None:
    path = Path(
        "configs/sota/deform360_frame_zero_depth_initializer_source_v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload["config"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == payload["config_sha256"]
    assert payload["config"]["claim_boundary"]["source_development_only"] is True
    assert payload["config"]["information_boundary"]["reserved_target_cases_read"] is False
    counts = {
        stratum: sum(len(episodes) for episodes in objects.values())
        for stratum, objects in payload["config"]["source_objects"].items()
    }
    assert counts == {"filament": 4, "sheet": 4, "volumetric": 4}


def test_metric_depth_support_is_metric_and_deterministic() -> None:
    points, depths, intrinsics, extrinsics = _camera_inputs()
    first, diagnostics = metric_depth_support_counts(
        points,
        depths,
        intrinsics,
        extrinsics,
        depth_tolerance_m=0.05,
    )
    second, _ = metric_depth_support_counts(
        points,
        depths,
        intrinsics,
        extrinsics,
        depth_tolerance_m=0.05,
    )
    assert np.array_equal(first, np.array([2, 1, 2], dtype=np.uint16))
    assert np.array_equal(first, second)
    assert diagnostics["informative_camera_count"] == 2
    assert diagnostics["maximum_support_view_count"] == 2


def test_depth_supported_surface_filters_the_frozen_hull(monkeypatch: pytest.MonkeyPatch) -> None:
    points, depths, intrinsics, extrinsics = _camera_inputs()
    colors = np.arange(9, dtype=np.float32).reshape(3, 3) / 8.0
    hull = FrameZeroPointCloud(
        points_m=points,
        colors=colors,
        method="strict-multiview-visual-hull-surface",
        diagnostics={"frozen": True},
    )
    monkeypatch.setattr(
        "bayesian_phystwin.deform360_frame_zero_depth_initializer."
        "build_strict_multiview_surface",
        lambda *args, **kwargs: hull,
    )
    masks = {camera: depth > 0 for camera, depth in depths.items()}
    images = {
        camera: np.zeros((*depth.shape, 3), dtype=np.uint8)
        for camera, depth in depths.items()
    }
    selected = build_depth_supported_multiview_surface(
        masks,
        images,
        depths,
        intrinsics,
        extrinsics,
        config=_depth_config(),
    )
    assert selected.method == "depth-supported-strict-multiview-surface"
    assert np.array_equal(selected.points_m, points)
    assert np.array_equal(selected.colors, colors)
    assert selected.diagnostics["depth_is_independent_modality"] is False
    assert selected.diagnostics["retained_minimum_support_views"] == 1


def test_depth_supported_surface_rejects_insufficient_metric_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points, depths, intrinsics, extrinsics = _camera_inputs()
    hull = FrameZeroPointCloud(
        points_m=points,
        colors=np.ones_like(points),
        method="strict-multiview-visual-hull-surface",
        diagnostics={},
    )
    monkeypatch.setattr(
        "bayesian_phystwin.deform360_frame_zero_depth_initializer."
        "build_strict_multiview_surface",
        lambda *args, **kwargs: hull,
    )
    depths = {camera: np.zeros_like(depth) for camera, depth in depths.items()}
    masks = {camera: np.ones_like(depth, dtype=bool) for camera, depth in depths.items()}
    images = {
        camera: np.zeros((*depth.shape, 3), dtype=np.uint8)
        for camera, depth in depths.items()
    }
    with pytest.raises(ValueError, match="informative frame-zero depth cameras"):
        build_depth_supported_multiview_surface(
            masks,
            images,
            depths,
            intrinsics,
            extrinsics,
            config=_depth_config(),
        )


def test_depth_selector_preserves_admitted_original_and_is_lazy() -> None:
    points = np.arange(12, dtype=np.float32).reshape(4, 3)
    colors = np.arange(12, dtype=np.uint8).reshape(4, 3)

    def forbidden_fallback() -> FrameZeroPointCloud:
        raise AssertionError("fallback must not be evaluated")

    selected = select_depth_supported_frame_zero_point_cloud(
        points,
        colors,
        forbidden_fallback,
        config=_depth_config(),
    )
    assert selected.method == "original-splat"
    assert selected.points_m is points
    assert selected.colors is colors
    assert selected.points_m.tobytes() == points.tobytes()
    assert selected.colors.tobytes() == colors.tobytes()
    assert selected.diagnostics["fallback_evaluated"] is False


def test_depth_selector_recovers_a_failed_original() -> None:
    fallback_points = np.arange(9, dtype=np.float32).reshape(3, 3)
    fallback = FrameZeroPointCloud(
        points_m=fallback_points,
        colors=np.ones_like(fallback_points),
        method="depth-supported-strict-multiview-surface",
        diagnostics={"source": "test"},
    )
    selected = select_depth_supported_frame_zero_point_cloud(
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        lambda: fallback,
        config=_depth_config(),
    )
    assert selected.method == "depth-supported-strict-multiview-surface"
    assert np.array_equal(selected.points_m, fallback_points)
    assert selected.diagnostics["fallback_evaluated"] is True
