from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "science"
    / "audit_deform360_source_visual_hull_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_deform360_source_visual_hull_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _look_at(position: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    direction = -position
    direction /= np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(direction, up)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    down = np.cross(direction, right)
    down /= np.linalg.norm(down)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, down, direction])
    transform[:3, 3] = position
    return transform


def test_binary_morphology_and_silhouette_metrics() -> None:
    truth = np.zeros((9, 9), dtype=bool)
    truth[3:6, 3:6] = True
    dilated = MODULE._binary_dilate(truth, 1)
    eroded = MODULE._binary_erode(truth, 1)
    assert int(np.count_nonzero(dilated)) == 25
    assert int(np.count_nonzero(eroded)) == 1
    metrics = MODULE._silhouette_metrics(
        truth, truth.copy(), boundary_tolerance_px=1
    )
    assert metrics["iou"] == 1.0
    assert metrics["boundary_f1"] == 1.0


def test_farthest_point_holdout_is_disjoint_and_spatially_dispersed() -> None:
    extrinsics = {}
    for index, angle in enumerate(
        np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ):
        position = np.array(
            [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
        )
        extrinsics[f"camera_{index:02d}"] = _look_at(position)
    training, heldout, order = MODULE._farthest_point_holdout(
        extrinsics, sorted(extrinsics), holdout_count=4
    )
    assert len(training) == 8
    assert len(heldout) == 4
    assert len(order) == 4
    assert set(training).isdisjoint(heldout)
    assert set(training) | set(heldout) == set(extrinsics)
    centers = MODULE._camera_centers(extrinsics, heldout)
    pairwise = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    nonzero = pairwise[pairwise > 0.0]
    assert float(np.min(nonzero)) > 1.5


def test_contact_frame_selection_is_quantile_based_and_unique() -> None:
    signal = np.zeros(30)
    signal[5:25] = np.arange(1.0, 21.0)
    frames = MODULE._select_contact_frames(signal, [0.1, 0.5, 0.9])
    assert frames == [7, 15, 22]
    assert len(set(frames)) == 3


def test_visual_hull_generalizes_to_heldout_views_and_beats_yaw_control() -> None:
    intrinsics_matrix = np.array(
        [[150.0, 0.0, 64.0], [0.0, 150.0, 64.0], [0.0, 0.0, 1.0]]
    )
    extrinsics = {}
    for index, angle in enumerate(
        np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ):
        position = np.array(
            [
                2.0 * np.cos(angle),
                2.0 * np.sin(angle),
                0.35 * np.sin(2.0 * angle),
            ]
        )
        extrinsics[f"camera_{index:02d}"] = _look_at(position)
    intrinsics = {
        camera: intrinsics_matrix.copy() for camera in extrinsics
    }

    axis = np.linspace(-0.5, 0.5, 34)
    candidate_points = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    truth_points = candidate_points[
        (
            (candidate_points[:, 0] / 0.35) ** 2
            + (candidate_points[:, 1] / 0.22) ** 2
            + (candidate_points[:, 2] / 0.18) ** 2
            < 1.0
        )
        & (
            candidate_points[:, 0]
            + 0.15 * candidate_points[:, 1]
            > -0.28
        )
    ]
    masks = {}
    for camera in extrinsics:
        masks[camera], _ = MODULE._render_points(
            truth_points,
            intrinsics[camera],
            extrinsics[camera],
            (128, 128),
            voxel_size_m=0.03,
            maximum_radius_px=3,
        )

    training, heldout, _order = MODULE._farthest_point_holdout(
        extrinsics, sorted(extrinsics), holdout_count=2
    )
    rig_center = MODULE._camera_centers(
        extrinsics, sorted(extrinsics)
    ).mean(axis=0)
    hull, metadata = MODULE._visual_hull_points(
        {camera: masks[camera] for camera in training},
        intrinsics,
        extrinsics,
        grid_center_world=rig_center,
        cube_half_extent_m=0.6,
        voxel_resolution=28,
        minimum_hull_points=100,
        minimum_consensus_fraction=0.5,
    )
    assert len(hull) >= 100
    correct_iou = []
    perturbed_iou = []
    for camera in heldout:
        correct, _ = MODULE._render_points(
            hull,
            intrinsics[camera],
            extrinsics[camera],
            masks[camera].shape,
            voxel_size_m=metadata["voxel_size_m"],
            maximum_radius_px=4,
        )
        perturbed, _ = MODULE._render_points(
            hull,
            intrinsics[camera],
            MODULE._perturb_camera_yaw(
                extrinsics[camera], yaw_degrees=5.0
            ),
            masks[camera].shape,
            voxel_size_m=metadata["voxel_size_m"],
            maximum_radius_px=4,
        )
        correct_iou.append(
            MODULE._silhouette_metrics(
                masks[camera], correct, boundary_tolerance_px=2
            )["iou"]
        )
        perturbed_iou.append(
            MODULE._silhouette_metrics(
                masks[camera], perturbed, boundary_tolerance_px=2
            )["iou"]
        )
    assert float(np.median(correct_iou)) > 0.60
    assert (
        float(np.median(correct_iou)) - float(np.median(perturbed_iou))
        > 0.15
    )


def test_camera_block_bootstrap_is_deterministic() -> None:
    records = [
        {"camera": "a", "iou": 0.4},
        {"camera": "a", "iou": 0.6},
        {"camera": "b", "iou": 0.8},
        {"camera": "b", "iou": 1.0},
        {"camera": "c", "iou": 0.5},
        {"camera": "c", "iou": 0.7},
    ]
    first = MODULE._camera_block_bootstrap(
        records, metric="iou", replicates=200, seed=7
    )
    second = MODULE._camera_block_bootstrap(
        records, metric="iou", replicates=200, seed=7
    )
    assert first == second
    assert np.isclose(first["estimate"], 2.0 / 3.0)
    assert first["ci95"][0] <= first["estimate"] <= first["ci95"][1]
