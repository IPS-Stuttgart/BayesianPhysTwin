from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_dynamic_query import (
    QUERY_BIRTH_FRAMES,
    UPDATE_FRAMES,
    DynamicQueryConfig,
    build_dynamic_query_schedule,
    project_visibility,
    projection_matrices,
    select_camera_panel,
)


def _cameras(count: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    intrinsics = np.repeat(np.eye(3)[None], count, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 64.0
    intrinsics[:, 1, 2] = 64.0
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)):
        center = np.asarray([2.0 * np.sin(angle), 0.0, -2.0 * np.cos(angle)])
        forward = -center / np.linalg.norm(center)
        up = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        poses[index, :3, :3] = np.column_stack([right, up, forward])
        poses[index, :3, 3] = center
    shapes = np.repeat(np.asarray([[128, 128]], dtype=np.int64), count, axis=0)
    names = tuple(f"camera-{index:02d}" for index in range(count))
    return intrinsics, poses, shapes, names


def _trajectory(node_count: int = 96) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-0.35, 0.35, node_count)
    frame_zero = np.column_stack(
        [
            x,
            0.04 * np.sin(4.0 * np.pi * x),
            0.02 * np.cos(3.0 * np.pi * x),
        ]
    )
    positions = np.repeat(frame_zero[None], 76, axis=0)
    amplitude = np.linspace(0.08, 0.16, node_count)
    for frame in range(76):
        positions[frame, :, 1] += amplitude * (frame / 75.0)
        positions[frame, :, 0] += 0.01 * np.sin(frame / 12.0) * (1.0 + x)
    rank = 8
    basis = np.zeros((node_count, 3, rank), dtype=np.float64)
    normalized = np.linspace(-1.0, 1.0, node_count)
    for mode in range(rank):
        basis[:, mode % 3, mode] = np.cos(
            (mode + 1) * np.pi * (normalized + 1.0) / 2.0
        )
        basis[:, (mode + 1) % 3, mode] += 0.25 * np.sin(
            (mode + 1) * np.pi * (normalized + 1.0) / 2.0
        )
    return positions, basis


def test_projection_visibility_has_expected_shape_and_rejects_behind_camera() -> None:
    intrinsics, poses, shapes, _ = _cameras(3)
    projections = projection_matrices(intrinsics, poses)
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0]])
    pixels, depth, visible = project_visibility(points, projections, shapes)
    assert pixels.shape == (3, 2, 2)
    assert depth.shape == (3, 2)
    assert visible.shape == (3, 2)
    assert np.all(visible[:, 0])
    assert not np.all(visible[:, 1])


def test_camera_panel_is_deterministic_and_diverse() -> None:
    positions, _ = _trajectory()
    intrinsics, poses, shapes, names = _cameras(10)
    config = replace(
        DynamicQueryConfig(),
        selected_camera_count=8,
        minimum_eligible_camera_count=8,
    )
    first = select_camera_panel(
        positions[0],
        intrinsics,
        poses,
        shapes,
        names,
        config=config,
    )
    second = select_camera_panel(
        positions[0],
        intrinsics,
        poses,
        shapes,
        names,
        config=config,
    )
    assert np.array_equal(first.camera_indices, second.camera_indices)
    assert first.camera_names == second.camera_names
    assert len(first.camera_indices) == 8
    assert len(set(first.camera_names)) == 8


def test_schedule_is_complete_unique_and_target_future_blind() -> None:
    positions, basis = _trajectory()
    intrinsics, poses, shapes, names = _cameras()
    config = replace(
        DynamicQueryConfig(),
        minimum_spatial_separation_m=0.0,
    )
    schedule = build_dynamic_query_schedule(
        positions,
        basis,
        intrinsics,
        poses,
        shapes,
        names,
        config=config,
    )
    expected_count = 3 * 3 * config.queries_per_birth
    assert len(schedule.entity_ids) == expected_count
    assert len(set(map(int, schedule.entity_ids))) == expected_count
    assert set(map(int, schedule.update_frames)) == set(UPDATE_FRAMES)
    assert set(map(int, schedule.birth_frames)) == {
        frame for group in QUERY_BIRTH_FRAMES for frame in group
    }
    assert np.all(
        schedule.predicted_motion_m >= config.minimum_predicted_motion_m
    )
    assert np.all(
        schedule.predicted_visible_views
        >= config.minimum_predicted_visible_views
    )

    changed_future = positions.copy()
    changed_future[58:] += 1000.0
    repeated = build_dynamic_query_schedule(
        changed_future,
        basis,
        intrinsics,
        poses,
        shapes,
        names,
        config=config,
    )
    assert np.array_equal(schedule.entity_ids, repeated.entity_ids)
    assert np.array_equal(schedule.birth_frames, repeated.birth_frames)
    assert np.array_equal(schedule.update_frames, repeated.update_frames)
    assert schedule.artifact_sha256 == repeated.artifact_sha256


def test_schedule_abstains_when_physical_motion_has_no_headroom() -> None:
    positions, basis = _trajectory()
    positions[:] = positions[0]
    intrinsics, poses, shapes, names = _cameras()
    with pytest.raises(ValueError, match="too few active"):
        build_dynamic_query_schedule(
            positions,
            basis,
            intrinsics,
            poses,
            shapes,
            names,
            config=replace(
                DynamicQueryConfig(),
                minimum_spatial_separation_m=0.0,
            ),
        )


def test_schedule_rejects_changed_update_or_birth_contract() -> None:
    positions, basis = _trajectory()
    intrinsics, poses, shapes, names = _cameras()
    with pytest.raises(ValueError, match="update frames changed"):
        build_dynamic_query_schedule(
            positions,
            basis,
            intrinsics,
            poses,
            shapes,
            names,
            update_frames=(18, 38, 57),
        )
    with pytest.raises(ValueError, match="birth frames changed"):
        build_dynamic_query_schedule(
            positions,
            basis,
            intrinsics,
            poses,
            shapes,
            names,
            query_birth_frames=((0, 6, 11), (20, 26, 32), (39, 45, 51)),
        )
