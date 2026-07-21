from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_frame_zero_initializer import (
    FrameZeroInitializerConfig,
    FrameZeroPointCloud,
    build_strict_multiview_surface,
    multiview_mask_votes,
    original_point_cloud_admissible,
    select_frame_zero_point_cloud,
)


def _look_at(position: np.ndarray) -> np.ndarray:
    forward = -position / np.linalg.norm(position)
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(forward, up_hint))) > 0.9:
        up_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, up, forward])
    transform[:3, 3] = position
    return transform


def _synthetic_views() -> tuple[dict[str, np.ndarray], ...]:
    image_size = 64
    intrinsics = np.array(
        [[48.0, 0.0, 31.5], [0.0, 48.0, 31.5], [0.0, 0.0, 1.0]]
    )
    positions = (
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-0.7, -0.4, -0.7]),
    )
    rows, columns = np.ogrid[:image_size, :image_size]
    disc = (rows - 31.5) ** 2 + (columns - 31.5) ** 2 <= 11.0**2
    masks: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    intrinsics_by_camera: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    for index, position in enumerate(positions):
        camera = f"camera-{index}"
        masks[camera] = disc.copy()
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image[..., index % 3] = 64 + 32 * index
        images[camera] = image
        intrinsics_by_camera[camera] = intrinsics.copy()
        extrinsics[camera] = _look_at(position)
    return masks, images, intrinsics_by_camera, extrinsics


def _config(*, maximum_output_point_count: int = 512) -> FrameZeroInitializerConfig:
    return FrameZeroInitializerConfig(
        minimum_original_point_count=32,
        minimum_camera_count=4,
        cube_half_extent_m=0.5,
        voxel_resolution=32,
        consensus_fraction_of_peak=0.75,
        minimum_consensus_votes=3,
        maximum_mask_dilation_pixels=1,
        minimum_fallback_point_count=32,
        maximum_output_point_count=maximum_output_point_count,
    )


def test_source_config_is_hash_locked_and_balanced() -> None:
    path = Path("configs/sota/deform360_frame_zero_initializer_source_v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == payload["config_sha256"]
    counts = {
        stratum: sum(len(episodes) for episodes in objects.values())
        for stratum, objects in config["source_objects"].items()
    }
    assert counts == {"filament": 4, "sheet": 4, "volumetric": 4}


def test_postopen_config_reuses_the_source_frozen_candidate() -> None:
    source = json.loads(
        Path(
            "configs/sota/deform360_frame_zero_initializer_source_v1.json"
        ).read_text(encoding="utf-8")
    )
    postopen = json.loads(
        Path(
            "configs/sota/deform360_frame_zero_postopen_failures_v1.json"
        ).read_text(encoding="utf-8")
    )
    encoded = json.dumps(
        postopen["config"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == postopen["config_sha256"]
    candidate = postopen["config"]["source_candidate"]
    assert candidate["source_config_sha256"] == source["config_sha256"]
    assert len(postopen["config"]["cases"]) == 4


def test_physical_config_reuses_the_frozen_postopen_result() -> None:
    postopen = json.loads(
        Path(
            "configs/sota/deform360_frame_zero_postopen_failures_v1.json"
        ).read_text(encoding="utf-8")
    )
    physical = json.loads(
        Path(
            "configs/sota/deform360_frame_zero_fallback_physical_v1.json"
        ).read_text(encoding="utf-8")
    )
    encoded = json.dumps(
        physical["config"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == physical["config_sha256"]
    assert physical["config"]["candidate"]["initializer_module_sha256"] == (
        postopen["config"]["source_candidate"]["initializer_module_sha256"]
    )
    assert physical["config"]["gate"]["required_warp_twin_count"] == 4


def test_persistence_only_integration_config_is_hash_locked_and_scoped() -> None:
    path = Path(
        "configs/sota/deform360_reconstruction_failure_persistence_fallback_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload["config"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == payload["config_sha256"]
    config = payload["config"]
    assert config["fallback_contract"]["physical_policy"] == "persistence_only"
    assert config["gates"]["future_object_or_outcome_read"] is False
    assert config["scoring_boundary"]["absolute_identity_rmse"].startswith(
        "Not available"
    )


def test_original_admission_matches_point_only_gate() -> None:
    points = np.zeros((32, 3), dtype=np.float32)
    assert original_point_cloud_admissible(points, minimum_point_count=32)
    assert not original_point_cloud_admissible(points[:31], minimum_point_count=32)
    points[0, 0] = np.nan
    assert not original_point_cloud_admissible(points, minimum_point_count=32)


def test_admitted_original_is_exact_and_fallback_is_lazy() -> None:
    points = np.arange(120, dtype=np.float32).reshape(40, 3)
    colors = np.arange(120, dtype=np.uint8).reshape(40, 3)

    def forbidden_fallback() -> FrameZeroPointCloud:
        raise AssertionError("fallback must not be evaluated")

    selected = select_frame_zero_point_cloud(
        points,
        colors,
        forbidden_fallback,
        config=_config(),
    )
    assert selected.method == "original-splat"
    assert selected.points_m is points
    assert selected.colors is colors
    assert selected.points_m.tobytes() == points.tobytes()
    assert selected.colors.tobytes() == colors.tobytes()
    assert selected.diagnostics["fallback_evaluated"] is False


def test_strict_surface_is_deterministic_connected_and_multiview_supported() -> None:
    masks, images, intrinsics, extrinsics = _synthetic_views()
    first = build_strict_multiview_surface(
        masks,
        images,
        intrinsics,
        extrinsics,
        config=_config(),
    )
    second = build_strict_multiview_surface(
        masks,
        images,
        intrinsics,
        extrinsics,
        config=_config(),
    )
    assert first.method == "strict-multiview-visual-hull-surface"
    assert np.array_equal(first.points_m, second.points_m)
    assert np.array_equal(first.colors, second.colors)
    assert len(first.points_m) >= 32
    assert first.points_m.dtype == np.float32
    assert first.colors.dtype == np.float32
    assert np.all(np.isfinite(first.points_m))
    assert np.all((first.colors >= 0.0) & (first.colors <= 1.0))
    selected = first.diagnostics["attempts"][-1]
    assert selected["components"]["largest_component_point_count"] > selected[
        "components"
    ]["surface_point_count_before_subsampling"]
    votes, _ = multiview_mask_votes(
        first.points_m,
        masks,
        intrinsics,
        extrinsics,
        mask_dilation_radius_pixels=first.diagnostics[
            "selected_mask_dilation_radius_pixels"
        ],
    )
    assert np.all(votes >= selected["required_vote_count"])


def test_surface_cap_is_spatial_and_fallback_selector_recovers() -> None:
    masks, images, intrinsics, extrinsics = _synthetic_views()
    config = _config(maximum_output_point_count=48)
    fallback = build_strict_multiview_surface(
        masks,
        images,
        intrinsics,
        extrinsics,
        config=config,
    )
    assert len(fallback.points_m) == 48
    failed_points = np.zeros((8, 3), dtype=np.float32)
    failed_colors = np.zeros_like(failed_points)
    selected = select_frame_zero_point_cloud(
        failed_points,
        failed_colors,
        lambda: fallback,
        config=config,
    )
    assert selected.method == "strict-multiview-visual-hull-surface"
    assert selected.diagnostics["fallback_evaluated"] is True
    assert np.array_equal(selected.points_m, fallback.points_m)
    assert np.ptp(selected.points_m, axis=0).min() > 0.1
