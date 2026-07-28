from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_sentinel_query_schedule import (
    PREFIX_END_FRAME,
    Deform360SentinelQueryConfig,
    build_deform360_sentinel_query_schedule,
)
from bayesian_phystwin.phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
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


def _trajectory(
    node_count: int = 64,
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
    active_count = 40
    amplitude = np.linspace(0.004, 0.012, active_count)
    for frame in range(1, 76):
        phase = frame / 75.0
        positions[frame, :active_count, 1] += amplitude * phase
        positions[frame, active_count:, 2] += 0.0002 * phase
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


def test_schedule_reserves_disjoint_active_and_sentinel_roles() -> None:
    positions, basis = _trajectory()
    intrinsics, poses, shapes, names = _cameras()
    schedule = build_deform360_sentinel_query_schedule(
        positions,
        basis,
        intrinsics,
        poses,
        shapes,
        names,
    )

    roles = schedule.query_roles
    active = roles == ACTIVE_QUERY_ROLE
    sentinel = roles == SENTINEL_QUERY_ROLE
    assert len(schedule.entity_ids) == 12
    assert int(np.sum(active)) == 9
    assert int(np.sum(sentinel)) == 3
    assert len(set(map(int, schedule.entity_ids))) == 12
    assert np.all(schedule.birth_frames == 0)
    assert np.all(schedule.update_frames == PREFIX_END_FRAME)
    assert np.all(schedule.predicted_motion_m[active] >= 0.002)
    assert np.all(schedule.predicted_motion_m[sentinel] <= 0.0005)
    assert np.all(schedule.predicted_visible_views >= 3)
    assert not schedule.entity_ids.flags.writeable
    assert not schedule.query_roles.flags.writeable


def test_schedule_is_future_blind_and_candidate_order_invariant() -> None:
    positions, basis = _trajectory()
    intrinsics, poses, shapes, names = _cameras()
    shuffled_candidates = np.random.default_rng(7).permutation(
        positions.shape[1]
    )
    first = build_deform360_sentinel_query_schedule(
        positions,
        basis,
        intrinsics,
        poses,
        shapes,
        names,
        candidate_entity_ids=shuffled_candidates,
    )
    changed_future = positions.copy()
    changed_future[PREFIX_END_FRAME + 1 :] += 1000.0
    second = build_deform360_sentinel_query_schedule(
        changed_future,
        basis,
        intrinsics,
        poses,
        shapes,
        names,
        candidate_entity_ids=np.arange(positions.shape[1]),
    )

    np.testing.assert_array_equal(first.entity_ids, second.entity_ids)
    np.testing.assert_array_equal(first.query_roles, second.query_roles)
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.descriptor()["information_boundary"] == {
        "maximum_physical_frame_read": PREFIX_END_FRAME,
        "maximum_observed_tracker_frame_used_for_planning": None,
        "observed_object_trajectory_read": False,
        "target_metric_read": False,
        "future_frame_after_update_used_for_that_update": False,
    }


@pytest.mark.parametrize(
    ("active_amplitude_m", "sentinel_amplitude_m"),
    [(0.001, 0.0002), (0.004, 0.001)],
)
def test_schedule_abstains_if_either_role_budget_is_missing(
    active_amplitude_m: float,
    sentinel_amplitude_m: float,
) -> None:
    positions, basis = _trajectory()
    positions[1:, :40, 1] = (
        positions[0, :40, 1][None]
        + np.linspace(0.0, active_amplitude_m, 75)[:, None]
    )
    positions[1:, 40:, 2] = (
        positions[0, 40:, 2][None]
        + np.linspace(0.0, sentinel_amplitude_m, 75)[:, None]
    )
    intrinsics, poses, shapes, names = _cameras()

    with pytest.raises(
        ValueError,
        match="motion-stratified frame-zero query budget is incomplete",
    ):
        build_deform360_sentinel_query_schedule(
            positions,
            basis,
            intrinsics,
            poses,
            shapes,
            names,
            config=replace(
                Deform360SentinelQueryConfig(),
                graph_basis_rank=4,
            ),
        )
