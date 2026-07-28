from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_dynamic_query import (
    DynamicQueryConfig,
    build_dynamic_query_schedule,
)
from bayesian_phystwin.deform360_dynamic_query_v2 import (
    AdaptiveDynamicQueryConfig,
    build_adaptive_dynamic_query_schedule,
)


def _cameras(
    count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    intrinsics = np.repeat(np.eye(3)[None], count, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 64.0
    intrinsics[:, 1, 2] = 64.0
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    for index, angle in enumerate(
        np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ):
        center = np.asarray(
            [2.0 * np.sin(angle), 0.0, -2.0 * np.cos(angle)]
        )
        forward = -center / np.linalg.norm(center)
        up = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        poses[index, :3, :3] = np.column_stack([right, up, forward])
        poses[index, :3, 3] = center
    shapes = np.repeat(
        np.asarray([[128, 128]], dtype=np.int64),
        count,
        axis=0,
    )
    names = tuple(f"camera-{index:02d}" for index in range(count))
    return intrinsics, poses, shapes, names


def _basis(node_count: int, rank: int = 8) -> np.ndarray:
    basis = np.zeros((node_count, 3, rank), dtype=np.float64)
    normalized = np.linspace(-1.0, 1.0, node_count)
    for mode in range(rank):
        basis[:, mode % 3, mode] = np.cos(
            (mode + 1) * np.pi * (normalized + 1.0) / 2.0
        )
        basis[:, (mode + 1) % 3, mode] += 0.25 * np.sin(
            (mode + 1) * np.pi * (normalized + 1.0) / 2.0
        )
    return basis


def _sparse_response_trajectory(
    node_count: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-0.35, 0.35, node_count)
    frame_zero = np.column_stack(
        [
            x,
            0.04 * np.sin(4.0 * np.pi * x),
            0.02 * np.cos(3.0 * np.pi * x),
        ]
    )
    positions = np.repeat(frame_zero[None], 76, axis=0)
    active = np.arange(24)
    positions[1:, active, 1] += 0.012
    positions[21:, active, 0] += 0.010
    return positions, _basis(node_count)


def test_adaptive_schedule_skips_inactive_waves_and_is_future_blind() -> None:
    positions, basis = _sparse_response_trajectory()
    intrinsics, poses, shapes, names = _cameras()
    config = replace(
        AdaptiveDynamicQueryConfig(),
        minimum_spatial_separation_m=0.0,
    )
    schedule = build_adaptive_dynamic_query_schedule(
        positions,
        basis,
        intrinsics,
        poses,
        shapes,
        names,
        config=config,
    )

    assert 8 <= len(schedule.entity_ids) <= 16
    assert len(set(map(int, schedule.entity_ids))) == len(schedule.entity_ids)
    assert set(map(int, schedule.birth_frames)) == {0, 20}
    assert set(map(int, schedule.skipped_birth_frames)) == {
        6,
        12,
        26,
        32,
        39,
        45,
        51,
    }
    assert all(
        config.minimum_queries_per_active_birth
        <= np.count_nonzero(schedule.birth_frames == birth)
        <= config.maximum_queries_per_active_birth
        for birth in set(map(int, schedule.birth_frames))
    )

    changed_future = positions.copy()
    changed_future[58:] += 1000.0
    repeated = build_adaptive_dynamic_query_schedule(
        changed_future,
        basis,
        intrinsics,
        poses,
        shapes,
        names,
        config=config,
    )
    assert repeated.artifact_sha256 == schedule.artifact_sha256
    assert repeated.descriptor() == schedule.descriptor()


def test_adaptive_schedule_rejects_too_few_active_waves() -> None:
    positions, basis = _sparse_response_trajectory()
    positions[20:] = positions[19]
    intrinsics, poses, shapes, names = _cameras()
    with pytest.raises(ValueError, match="too few active"):
        build_adaptive_dynamic_query_schedule(
            positions,
            basis,
            intrinsics,
            poses,
            shapes,
            names,
            config=replace(
                AdaptiveDynamicQueryConfig(),
                minimum_spatial_separation_m=0.0,
            ),
        )


def test_v1_fixed_schedule_remains_72_queries() -> None:
    node_count = 128
    x = np.linspace(-0.35, 0.35, node_count)
    frame_zero = np.column_stack([x, np.zeros(node_count), np.zeros(node_count)])
    positions = np.repeat(frame_zero[None], 76, axis=0)
    positions[:, :, 1] += (
        np.linspace(0.0, 0.20, 76)[:, None]
        * np.linspace(1.0, 2.0, node_count)[None]
    )
    intrinsics, poses, shapes, names = _cameras()
    config = replace(
        DynamicQueryConfig(),
        minimum_spatial_separation_m=0.0,
    )
    schedule = build_dynamic_query_schedule(
        positions,
        _basis(node_count),
        intrinsics,
        poses,
        shapes,
        names,
        config=config,
    )
    assert len(schedule.entity_ids) == 72
    assert schedule.config == config
