from __future__ import annotations

import numpy as np
import pytest

from causal4d_public.deform360_filament_initialization import (
    backproject_pixels_to_plane,
    extract_filament_mask_centerline,
    initialize_filament_from_multiview_masks,
)
from causal4d_public.deform360_filament_registration import (
    FilamentRegistrationConfig,
    FilamentRegistrationQAConfig,
    _decode_filament_parameters,
    _encode_filament_parameters,
    _project_centerline_equal_edge_lengths,
    audit_filament_registration,
    filament_local_geometry_diagnostics,
    filament_multiview_support_diagnostics,
    fit_multiview_filament_centerline,
    sample_filament_centerline,
)


def _camera(center_x: float, center_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray([[180.0, 0.0, 96.0], [0.0, 180.0, 96.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.eye(4)
    camera_to_world[:2, 3] = (center_x, center_y)
    return intrinsics, camera_to_world


def _project(
    points: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    return np.column_stack(
        (
            camera[:, 0] / camera[:, 2] * intrinsics[0, 0] + intrinsics[0, 2],
            camera[:, 1] / camera[:, 2] * intrinsics[1, 1] + intrinsics[1, 2],
        )
    )


def _render_curve_mask(
    centerline: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    radius_px: float = 3.5,
) -> np.ndarray:
    samples = sample_filament_centerline(centerline, samples_per_edge=10)
    pixels = _project(samples, intrinsics, camera_to_world)
    rows, columns = np.indices((192, 192))
    squared = (columns[..., None] - pixels[:, 0]) ** 2 + (
        rows[..., None] - pixels[:, 1]
    ) ** 2
    return np.min(squared, axis=2) <= radius_px**2


def _synthetic_views(
    centerline: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    masks = {}
    intrinsics = {}
    poses = {}
    for name, center in (("left", -0.12), ("middle", 0.0), ("right", 0.12)):
        intrinsic, pose = _camera(center)
        masks[name] = _render_curve_mask(centerline, intrinsic, pose)
        intrinsics[name] = intrinsic
        poses[name] = pose
    return masks, intrinsics, poses


def test_local_thickness_does_not_confuse_curvature_with_width() -> None:
    parameter = np.linspace(-1.0, 1.0, 160)
    curve = np.column_stack(
        (
            0.18 * parameter,
            0.13 * parameter**2,
            np.full_like(parameter, 1.0),
        )
    )
    rng = np.random.default_rng(5)
    points = np.repeat(curve, 8, axis=0) + rng.normal(
        scale=0.0015, size=(len(curve) * 8, 3)
    )

    diagnostics = filament_local_geometry_diagnostics(points, neighbor_count=16)

    assert diagnostics["global_pca_q01_to_q99_spans_m_descending"][1] > 0.08
    assert diagnostics["local_radial_distance_m"]["p95"] < 0.008


def test_equal_edge_projection_preserves_a_kinked_initial_shape() -> None:
    parameter = np.linspace(0.0, 1.0, 21) ** 1.8
    initial = np.column_stack(
        (
            0.45 * parameter,
            0.08 * np.sin(3.0 * np.pi * parameter),
            1.0 + 0.04 * np.cos(2.0 * np.pi * parameter),
        )
    )
    target_length = float(np.linalg.norm(np.diff(initial, axis=0), axis=1).sum())

    projected = _project_centerline_equal_edge_lengths(
        initial, target_length, iterations=64
    )
    cumulative, _ = _decode_filament_parameters(
        _encode_filament_parameters(initial),
        node_count=len(initial),
        target_length_m=target_length,
    )

    edge = np.linalg.norm(np.diff(projected, axis=0), axis=1)
    projected_movement = np.median(np.linalg.norm(projected - initial, axis=1))
    cumulative_movement = np.median(np.linalg.norm(cumulative - initial, axis=1))
    assert np.std(edge) / np.mean(edge) < 1e-5
    assert projected_movement < 0.5 * cumulative_movement


def test_gripper_occlusion_is_missing_evidence_not_a_registration_failure() -> None:
    x = np.linspace(-0.22, 0.22, 21)
    centerline = np.column_stack((x, np.zeros_like(x), np.ones_like(x)))
    masks, intrinsics, poses = _synthetic_views(centerline)
    gripper_masks = {}
    for camera, mask in masks.items():
        pixels = _project(centerline, intrinsics[camera], poses[camera])
        columns = np.arange(mask.shape[1])[None, :]
        rows = np.arange(mask.shape[0])[:, None]
        occluder = (np.abs(columns - pixels[len(pixels) // 2, 0]) <= 18) & (
            np.abs(rows - pixels[len(pixels) // 2, 1]) <= 10
        )
        masks[camera] = mask & ~occluder
        gripper_masks[camera] = occluder

    without = filament_multiview_support_diagnostics(
        centerline, masks, intrinsics, poses, samples_per_edge=5
    )
    with_occlusion = filament_multiview_support_diagnostics(
        centerline,
        masks,
        intrinsics,
        poses,
        gripper_masks_by_camera=gripper_masks,
        samples_per_edge=5,
    )

    assert with_occlusion["visibility_aware_mask_support"]["median"] > 0.95
    assert (
        with_occlusion["visibility_aware_mask_support"]["median"]
        > without["visibility_aware_mask_support"]["median"] + 0.1
    )


def test_source_camera_reliability_downweights_a_disjoint_mask() -> None:
    x = np.linspace(-0.22, 0.22, 21)
    centerline = np.column_stack((x, np.zeros_like(x), np.ones_like(x)))
    masks, intrinsics, poses = _synthetic_views(centerline)
    masks["right"] = np.roll(masks["right"], 45, axis=0)

    diagnostics = filament_multiview_support_diagnostics(
        centerline,
        masks,
        intrinsics,
        poses,
        camera_reliability_by_camera={
            "left": 1.0,
            "middle": 1.0,
            "right": 0.05,
        },
    )

    assert diagnostics["visibility_aware_mask_support"]["lower_quartile"] < 0.6
    assert (
        diagnostics["reliability_weighted_visibility_aware_mask_support"][
            "lower_quartile"
        ]
        > 0.95
    )
    assert diagnostics["camera_reliability"]["effective_camera_count"] < 2.2


def test_multiview_fit_completes_a_curve_from_disconnected_seed_geometry() -> None:
    parameter = np.linspace(-1.0, 1.0, 21)
    truth = np.column_stack(
        (
            0.20 * parameter,
            0.055 * (1.0 - parameter**2),
            1.0 + 0.025 * np.sin(np.pi * parameter),
        )
    )
    masks, intrinsics, poses = _synthetic_views(truth)
    observed_centerlines = {
        camera: _project(
            sample_filament_centerline(truth, samples_per_edge=5),
            intrinsics[camera],
            poses[camera],
        )
        for camera in masks
    }
    initial = np.column_stack(
        (
            0.20 * parameter,
            np.full_like(parameter, -0.035),
            np.ones_like(parameter),
        )
    )
    rng = np.random.default_rng(9)
    seed = np.concatenate((truth[:6], truth[-6:]), axis=0)
    seed = np.repeat(seed, 5, axis=0) + rng.normal(scale=0.002, size=(60, 3))
    target_length = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
    config = FilamentRegistrationConfig(
        maximum_function_evaluations=120,
        initialization_weight=0.005,
        point_cloud_weight=0.05,
    )

    fitted, diagnostics = fit_multiview_filament_centerline(
        initial,
        target_length,
        masks,
        intrinsics,
        poses,
        observed_centerline_pixels_by_camera=observed_centerlines,
        seed_points_world_m=seed,
        config=config,
    )

    initial_error = np.mean(np.linalg.norm(initial - truth, axis=1))
    fitted_error = np.mean(np.linalg.norm(fitted - truth, axis=1))
    assert diagnostics["optimizer"]["final_mean_squared_residual"] < (
        0.55 * diagnostics["optimizer"]["initial_mean_squared_residual"]
    )
    assert fitted_error < 0.7 * initial_error
    assert diagnostics["relative_length_error"] < 0.04
    assert (
        diagnostics["fitted_multiview_support"]["visibility_aware_mask_support"][
            "median"
        ]
        > diagnostics["initial_multiview_support"]["visibility_aware_mask_support"][
            "median"
        ]
    )


def test_topology_aware_audit_accepts_a_curved_thin_filament() -> None:
    parameter = np.linspace(-1.0, 1.0, 21)
    centerline = np.column_stack(
        (
            0.20 * parameter,
            0.07 * parameter**2,
            np.ones_like(parameter),
        )
    )
    masks, intrinsics, poses = _synthetic_views(centerline)
    samples = sample_filament_centerline(centerline, samples_per_edge=6)
    rng = np.random.default_rng(18)
    points = np.repeat(samples, 6, axis=0) + rng.normal(
        scale=0.001, size=(len(samples) * 6, 3)
    )
    target_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())

    audit = audit_filament_registration(
        centerline,
        target_length,
        masks,
        intrinsics,
        poses,
        reconstructed_points_world_m=points,
        config=FilamentRegistrationQAConfig(maximum_median_mask_coverage_p95_px=5.0),
    )

    assert audit["passed"] is True
    assert audit["acceptance_gates"]["local_filament_thickness"] is True
    assert audit["local_geometry"]["global_pca_q01_to_q99_spans_m_descending"][1] > 0.04


def test_mask_centerline_bridges_only_the_gripper_covered_gap() -> None:
    parameter = np.linspace(-1.0, 1.0, 21)
    centerline = np.column_stack(
        (
            0.20 * parameter,
            0.06 * (1.0 - parameter**2),
            np.ones_like(parameter),
        )
    )
    intrinsic, pose = _camera(0.0)
    mask = _render_curve_mask(centerline, intrinsic, pose, radius_px=5.0)
    truth_pixels = _project(
        sample_filament_centerline(centerline, samples_per_edge=8), intrinsic, pose
    )
    middle = truth_pixels[len(truth_pixels) // 2]
    columns = np.arange(mask.shape[1])[None, :]
    rows = np.arange(mask.shape[0])[:, None]
    gripper = (np.abs(columns - middle[0]) <= 9) & (np.abs(rows - middle[1]) <= 12)
    occluded_mask = mask & ~gripper

    extracted, diagnostics = extract_filament_mask_centerline(
        occluded_mask, gripper_mask=gripper
    )

    nearest = np.linalg.norm(
        extracted[:, None, :] - truth_pixels[None, :, :], axis=2
    ).min(axis=1)
    assert diagnostics["bridge_count"] >= 1
    assert np.median(nearest) < 4.0
    assert diagnostics["diameter_path_length_px"] > 60.0


def test_backprojection_recovers_points_on_a_known_plane() -> None:
    intrinsic, pose = _camera(0.04)
    points = np.asarray([[-0.12, -0.03, 1.0], [0.0, 0.04, 1.0], [0.17, -0.02, 1.0]])
    pixels = _project(points, intrinsic, pose)

    recovered = backproject_pixels_to_plane(
        pixels,
        intrinsic,
        pose,
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(recovered, points, atol=1e-10)


def test_multiview_mask_initializer_recovers_missing_middle_geometry() -> None:
    parameter = np.linspace(-1.0, 1.0, 21)
    truth = np.column_stack(
        (
            0.20 * parameter,
            0.07 * (1.0 - parameter**2),
            np.ones_like(parameter),
        )
    )
    masks, intrinsics, poses = _synthetic_views(truth)
    rng = np.random.default_rng(27)
    seed = np.concatenate((truth[:7], truth[-7:]), axis=0)
    seed = np.repeat(seed, 8, axis=0) + rng.normal(
        scale=(0.0015, 0.0015, 0.0003), size=(len(seed) * 8, 3)
    )

    target_length = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
    initial, diagnostics = initialize_filament_from_multiview_masks(
        seed,
        masks,
        intrinsics,
        poses,
        camera_reliability_by_camera={
            "left": 0.1,
            "middle": 1.0,
            "right": 1.0,
        },
        target_length_m=target_length,
    )

    forward = np.mean(np.linalg.norm(initial - truth, axis=1))
    reverse = np.mean(np.linalg.norm(initial[::-1] - truth, axis=1))
    assert min(forward, reverse) < 0.012
    assert diagnostics["passing_candidate_count"] >= 3
    assert "left" not in diagnostics["selected_source_cameras"]
    assert diagnostics["selected_centerline_length_m"] == pytest.approx(target_length)
