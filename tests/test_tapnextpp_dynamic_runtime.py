from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bayesian_phystwin.deform360_dynamic_query import (
    CameraPanel,
    DynamicQueryConfig,
    DynamicQuerySchedule,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    DynamicBirthAssociations,
    DynamicTAPNextPPRuntimeConfig,
    build_dynamic_birth_associations,
    run_dynamic_tapnextpp_births,
)


def _digest(character: str) -> str:
    return character * 64


def _schedule() -> DynamicQuerySchedule:
    config = DynamicQueryConfig(
        selected_camera_count=3,
        minimum_eligible_camera_count=3,
        queries_per_birth=1,
        graph_basis_rank=1,
        minimum_predicted_visible_views=3,
        minimum_spatial_separation_m=0.0,
    )
    panel = CameraPanel(
        camera_indices=np.asarray([0, 1, 2]),
        camera_names=("camera-a", "camera-b", "camera-c"),
        frame_zero_coverage=np.ones(3),
        selection_scores=np.ones(3),
    )
    return DynamicQuerySchedule(
        update_frames=np.asarray([2, 5]),
        birth_frames=np.asarray([0, 3]),
        entity_ids=np.asarray([0, 1]),
        predicted_motion_m=np.asarray([0.01, 0.01]),
        predicted_visible_views=np.asarray([3, 3]),
        information_gain=np.asarray([1.0, 1.0]),
        config=config,
        camera_panel=panel,
        physical_prefix_sha256=_digest("a"),
        graph_basis_sha256=_digest("b"),
        artifact_sha256=_digest("c"),
    )


def _associations(*, second_camera_second_entity: bool = True) -> DynamicBirthAssociations:
    valid = np.asarray(
        [
            [True, True],
            [True, second_camera_second_entity],
        ]
    )
    pixels = np.asarray(
        [
            [[10.0, 11.0], [20.0, 21.0]],
            [[12.0, 13.0], [22.0, 23.0]],
        ]
    )
    covariance = np.repeat(
        np.eye(2)[None, None],
        4,
        axis=0,
    ).reshape(2, 2, 2, 2)
    return DynamicBirthAssociations(
        query_points_world_m=np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]
        ),
        query_points_xy=pixels,
        valid=valid,
        association_probability=np.where(valid, 0.8, 0.0),
        association_entropy=np.where(valid, 0.1, 1.0),
        candidate_pixel_covariance_px2=covariance,
        candidate_count=np.where(valid, 4, 0),
        camera_indices=np.asarray([0, 1]),
        camera_names=("camera-a", "camera-b"),
    )


def test_dynamic_runtime_uses_independent_states_and_causal_intervals() -> None:
    calls: list[tuple[bool, int]] = []

    def fake_step(
        _model: object,
        frame_bgr: np.ndarray,
        query: np.ndarray | None,
        state: object,
        _utils: object,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        frame = int(frame_bgr[0, 0, 0])
        if query is not None:
            state = {"query": query.copy(), "start": frame}
        assert isinstance(state, dict)
        calls.append((query is not None, frame))
        elapsed = frame - int(state["start"])
        positions = np.asarray(state["query"]) + np.asarray([elapsed, 0.0])
        probability = np.full(len(positions), 0.75)
        return positions, probability, state

    rgbs = np.zeros((2, 6, 32, 48, 3), dtype=np.uint8)
    for frame in range(6):
        rgbs[:, frame] = frame
    result = run_dynamic_tapnextpp_births(
        object(),
        rgbs,
        _associations(),
        np.asarray([0, 3]),
        np.asarray([2, 5]),
        SimpleNamespace(),
        config=DynamicTAPNextPPRuntimeConfig(
            support_points_per_query=2,
        ),
        tracker_step=fake_step,
    )

    assert result.rollout_count == 4
    assert result.model_frame_count == 12
    assert sum(fresh for fresh, _ in calls) == 4
    assert np.all(np.isnan(result.tracks_xy[:, 0:3, 1]))
    assert np.all(np.isnan(result.tracks_xy[:, 3:, 0]))
    np.testing.assert_allclose(result.tracks_xy[0, 2, 0], [12.0, 11.0])
    np.testing.assert_allclose(result.tracks_xy[1, 5, 1], [24.0, 23.0])
    assert np.all(result.visibility_probability[result.active] == 0.75)


def test_dynamic_runtime_skips_invalid_camera_association() -> None:
    calls = 0

    def fake_step(
        _model: object,
        _frame: np.ndarray,
        query: np.ndarray | None,
        state: object,
        _utils: object,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        nonlocal calls
        calls += 1
        if query is not None:
            state = query.copy()
        assert isinstance(state, np.ndarray)
        return state, np.ones(len(state)), state

    result = run_dynamic_tapnextpp_births(
        object(),
        np.zeros((2, 6, 16, 16, 3), dtype=np.uint8),
        _associations(second_camera_second_entity=False),
        np.asarray([0, 3]),
        np.asarray([2, 5]),
        SimpleNamespace(),
        config=DynamicTAPNextPPRuntimeConfig(
            support_points_per_query=0,
        ),
        tracker_step=fake_step,
    )
    assert result.rollout_count == 3
    assert calls == 9
    assert np.all(np.isnan(result.tracks_xy[1, :, 1]))


def test_birth_association_reads_each_entity_at_its_birth() -> None:
    schedule = _schedule()
    positions = np.zeros((6, 2, 3), dtype=np.float64)
    positions[..., 2] = 1.0
    positions[3, 1, 0] = 0.1
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    camera_to_world = np.repeat(np.eye(4)[None], 3, axis=0)
    depths = np.ones((3, 6, 8, 8), dtype=np.float64)
    masks = np.ones_like(depths, dtype=bool)

    associations = build_dynamic_birth_associations(
        schedule,
        positions,
        intrinsics,
        camera_to_world,
        depths,
        masks,
    )

    np.testing.assert_allclose(
        associations.query_points_world_m,
        np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
    )
    assert associations.valid.shape == (3, 2)
    assert np.all(associations.valid)
    assert np.all(associations.candidate_count > 0)


def test_birth_association_is_independent_of_future_depth() -> None:
    schedule = _schedule()
    positions = np.zeros((6, 2, 3), dtype=np.float64)
    positions[..., 2] = 1.0
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    camera_to_world = np.repeat(np.eye(4)[None], 3, axis=0)
    depths = np.ones((3, 6, 8, 8), dtype=np.float64)
    masks = np.ones_like(depths, dtype=bool)
    first = build_dynamic_birth_associations(
        schedule,
        positions,
        intrinsics,
        camera_to_world,
        depths,
        masks,
    )
    mutated = depths.copy()
    mutated[:, [1, 2, 4, 5]] = 100.0
    second = build_dynamic_birth_associations(
        schedule,
        positions,
        intrinsics,
        camera_to_world,
        mutated,
        masks,
    )

    np.testing.assert_array_equal(first.query_points_xy, second.query_points_xy)
    np.testing.assert_array_equal(
        first.association_probability,
        second.association_probability,
    )
