import numpy as np
import pytest

from bayesian_phystwin.phystwin_active_queries import (
    PhysicsGuidedQueryConfig,
    geometric_view_support,
    plan_physics_guided_queries,
)


def _projected_pixels(camera_count: int, rollout: np.ndarray) -> np.ndarray:
    frame_count, node_count, _ = rollout.shape
    pixels = np.empty((camera_count, frame_count, node_count, 2), dtype=float)
    for camera in range(camera_count):
        pixels[camera, ..., 0] = 10.0 + rollout[..., 0] + camera
        pixels[camera, ..., 1] = 20.0 + rollout[..., 1]
    return pixels


def test_geometric_view_support_requires_positive_in_frame_projection() -> None:
    pixels = np.asarray(
        [
            [
                [[2.0, 3.0], [-1.0, 3.0], [9.0, 9.0]],
                [[2.0, 3.0], [2.0, 3.0], [np.nan, 9.0]],
            ]
        ]
    )
    depth = np.asarray([[[1.0, 1.0, -1.0], [0.0, 1.0, 1.0]]])

    visible = geometric_view_support(pixels, depth, np.asarray([10, 10]))

    np.testing.assert_array_equal(
        visible,
        np.asarray([[[True, False, False], [False, True, False]]]),
    )
    assert not visible.flags.writeable


def test_selection_filters_contact_static_and_single_view_nodes() -> None:
    rollout = np.zeros((3, 6, 3), dtype=float)
    rollout[0, :, 0] = np.asarray([0.0, 0.2, 0.4, 1.0, 2.0, 1.05])
    rollout[1:, 0, 1] = 1.0  # moving but at the contact
    rollout[1:, 2, 1] = 1.0  # moving but visible from only one camera
    rollout[1:, 3, 1] = 1.0
    rollout[1:, 4, 2] = 1.0
    rollout[1:, 5, 1] = 1.0
    pixels = _projected_pixels(3, rollout)
    support = np.ones((3, 3, 6), dtype=float)
    support[1:, :, 2] = 0.0
    mode_basis = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    config = PhysicsGuidedQueryConfig(
        query_count=2,
        maximum_reseeds=0,
        minimum_motion_m=0.5,
        contact_exclusion_radius_m=0.2,
        contact_exclusion_fraction=0.0,
    )

    plan = plan_physics_guided_queries(
        rollout,
        pixels,
        support,
        mode_basis=mode_basis,
        contact_position_m=np.zeros(3),
        candidate_ids=np.asarray([5, 4, 3, 2, 1, 0]),
        config=config,
    )

    np.testing.assert_array_equal(plan.node_ids, np.asarray([4, 3]))
    np.testing.assert_array_equal(plan.seed_frames, np.asarray([0, 0]))
    assert plan.initial_budget_met
    assert plan.reseed_count == 0
    assert np.all(np.sum(plan.camera_mask, axis=1) == 3)
    assert not plan.node_ids.flags.writeable


def test_selection_is_independent_of_candidate_order() -> None:
    rollout = np.zeros((2, 4, 3), dtype=float)
    rollout[0, :, 0] = np.arange(4)
    rollout[1, :, 1] = 1.0
    pixels = _projected_pixels(2, rollout)
    support = np.ones((2, 2, 4), dtype=float)
    mode_basis = np.eye(4, 2)
    config = PhysicsGuidedQueryConfig(
        query_count=2,
        maximum_reseeds=0,
        contact_exclusion_fraction=0.0,
    )

    first = plan_physics_guided_queries(
        rollout,
        pixels,
        support,
        mode_basis=mode_basis,
        candidate_ids=np.asarray([3, 1, 0, 2]),
        config=config,
    )
    second = plan_physics_guided_queries(
        rollout,
        pixels,
        support,
        mode_basis=mode_basis,
        candidate_ids=np.asarray([0, 1, 2, 3]),
        config=config,
    )

    np.testing.assert_array_equal(first.node_ids, second.node_ids)
    np.testing.assert_array_equal(first.total_score, second.total_score)


def test_support_collapse_reseeds_a_new_identity_causally() -> None:
    rollout = np.zeros((5, 3, 3), dtype=float)
    rollout[:, :, 0] = np.asarray([0.0, 1.0, 2.0])[None]
    rollout[:, 0, 1] = np.linspace(0.0, 2.0, 5)
    rollout[:, 1, 2] = np.linspace(0.0, 1.5, 5)
    rollout[:, 2, 1] = np.linspace(0.0, 0.5, 5)
    pixels = _projected_pixels(2, rollout)
    predicted = np.ones((2, 5, 3), dtype=float)
    tracker = np.ones_like(predicted)
    tracker[:, 1:3, 0] = 0.0
    config = PhysicsGuidedQueryConfig(
        query_count=1,
        maximum_reseeds=1,
        minimum_motion_m=0.1,
        contact_exclusion_fraction=0.0,
        reseed_patience_frames=2,
        minimum_reseed_interval_frames=1,
    )

    plan = plan_physics_guided_queries(
        rollout,
        pixels,
        predicted,
        tracker_support_probability=tracker,
        config=config,
    )

    np.testing.assert_array_equal(plan.node_ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(plan.seed_frames, np.asarray([0, 2]))
    np.testing.assert_array_equal(plan.replaces_node_ids, np.asarray([-1, 0]))
    assert plan.reseed_count == 1
    node_ids, queries_txy, replacements = plan.camera_queries_txy(0)
    np.testing.assert_array_equal(node_ids, np.asarray([0, 1]))
    np.testing.assert_allclose(queries_txy[:, 0], np.asarray([0.0, 2.0]))
    np.testing.assert_array_equal(replacements, np.asarray([-1, 0]))
    assert not queries_txy.flags.writeable


def test_delayed_reseed_retains_the_retired_identity_link() -> None:
    rollout = np.zeros((5, 2, 3), dtype=float)
    rollout[:, 0, 0] = np.linspace(0.0, 4.0, 5)
    rollout[:, 1, 1] = np.asarray([0.0, 0.1, 0.2, 0.3, 1.3])
    pixels = _projected_pixels(2, rollout)
    predicted = np.ones((2, 5, 2), dtype=float)
    predicted[:, 2, 1] = 0.0
    tracker = np.ones_like(predicted)
    tracker[:, 1:3, 0] = 0.0
    config = PhysicsGuidedQueryConfig(
        query_count=1,
        maximum_reseeds=1,
        minimum_motion_m=0.1,
        contact_exclusion_fraction=0.0,
        reseed_patience_frames=2,
        minimum_reseed_interval_frames=1,
    )

    plan = plan_physics_guided_queries(
        rollout,
        pixels,
        predicted,
        tracker_support_probability=tracker,
        config=config,
    )

    np.testing.assert_array_equal(plan.node_ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(plan.seed_frames, np.asarray([0, 3]))
    np.testing.assert_array_equal(plan.replaces_node_ids, np.asarray([-1, 0]))


def test_two_view_gate_is_not_weakened_when_support_is_insufficient() -> None:
    rollout = np.zeros((3, 2, 3), dtype=float)
    rollout[1:, :, 0] = 1.0
    pixels = _projected_pixels(3, rollout)
    support = np.zeros((3, 3, 2), dtype=float)
    support[0] = 1.0
    config = PhysicsGuidedQueryConfig(
        query_count=2,
        maximum_reseeds=0,
        contact_exclusion_fraction=0.0,
    )

    plan = plan_physics_guided_queries(rollout, pixels, support, config=config)

    assert len(plan.node_ids) == 0
    assert not plan.initial_budget_met
    assert plan.camera_mask.shape == (0, 3)
    with pytest.raises(ValueError, match="outside"):
        plan.camera_queries_txy(3)


def test_invalid_mode_basis_is_rejected() -> None:
    rollout = np.zeros((2, 3, 3), dtype=float)
    pixels = _projected_pixels(2, rollout)
    support = np.ones((2, 2, 3), dtype=float)

    with pytest.raises(ValueError, match="mode_basis"):
        plan_physics_guided_queries(
            rollout,
            pixels,
            support,
            mode_basis=np.ones((2, 1)),
        )
