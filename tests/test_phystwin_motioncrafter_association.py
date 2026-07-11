import numpy as np

from bayesian_phystwin.phystwin_motioncrafter_association import (
    DenseMotionTrajectories,
    MotionCrafterPrediction,
    apply_graph_association,
    concatenate_dense_trajectories,
    compose_dense_trajectories,
    dense_graph_error_by_frame,
    infer_graph_association,
    manual_track_association_audit,
    resample_cover_grid,
    reverse_dense_trajectories,
    robust_icp_transform,
    robust_similarity_transform,
)


def test_resample_cover_grid_is_identity_at_equal_resolution() -> None:
    values = np.arange(4 * 6).reshape(4, 6)

    result = resample_cover_grid(values, (4, 6))

    np.testing.assert_array_equal(result, values)


def test_robust_similarity_transform_rejects_large_outliers() -> None:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(100, 3))
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = 1.7 * source @ rotation + np.array([0.4, -0.2, 1.1])
    target[:10] += rng.normal(scale=20.0, size=(10, 3))

    transform = robust_similarity_transform(
        source,
        target,
        trim_fraction=0.8,
        iterations=8,
    )

    fitted = source[10:] @ np.asarray(transform["linear"]) + np.asarray(
        transform["translation"]
    )
    np.testing.assert_allclose(fitted, target[10:], atol=1e-10)
    assert transform["inlier_pair_count"] == 80


def test_robust_icp_transform_recovers_small_rigid_offset() -> None:
    rng = np.random.default_rng(7)
    source = rng.uniform(-0.1, 0.1, size=(200, 3))
    offset = np.array([0.008, -0.005, 0.003])
    target = source + offset

    transform = robust_icp_transform(
        source,
        target,
        mode="se3",
        trim_fraction=1.0,
        iterations=8,
        maximum_correspondence_m=0.03,
    )

    fitted = source @ np.asarray(transform["linear"]) + np.asarray(
        transform["translation"]
    )
    np.testing.assert_allclose(fitted, target, atol=1e-10)


def test_reverse_dense_trajectories_restores_original_time_order() -> None:
    reverse_time = DenseMotionTrajectories(
        positions=np.arange(3, dtype=np.float32)[:, None, None]
        * np.ones((3, 1, 3), dtype=np.float32),
        valid=np.array([[True], [True], [False]]),
        step_error_m=np.array([[1.0], [2.0]], dtype=np.float32),
        pixel_indices=np.array([[2], [1], [-1]]),
        seed_pixels_yx=np.array([[0, 2]], dtype=np.int32),
    )

    original_time = reverse_dense_trajectories(reverse_time)

    np.testing.assert_allclose(original_time.positions[:, 0, 0], [2.0, 1.0, 0.0])
    np.testing.assert_array_equal(original_time.valid[:, 0], [False, True, True])
    np.testing.assert_allclose(original_time.step_error_m[:, 0], [2.0, 1.0])


def test_compose_dense_trajectories_reindexes_flow_and_rejects_collisions() -> None:
    point_map = np.zeros((3, 1, 4, 3), dtype=np.float32)
    point_map[:, 0, :, 0] = 0.01 * np.arange(4)
    scene_flow = np.zeros_like(point_map)
    scene_flow[:, :, :, 0] = 0.01
    prediction = MotionCrafterPrediction(
        point_map=point_map,
        valid_mask=np.ones((3, 1, 4), dtype=bool),
        scene_flow=scene_flow,
        deform_mask=np.ones((3, 1, 4), dtype=bool),
    )

    trajectories = compose_dense_trajectories(
        prediction,
        np.ones((3, 1, 4), dtype=bool),
        seed_stride_pixels=1,
        maximum_transport_error_m=0.004,
    )

    np.testing.assert_allclose(trajectories.positions[:, 0, 0], [0.0, 0.01, 0.02])
    np.testing.assert_array_equal(trajectories.valid[:, 0], [True, True, True])
    np.testing.assert_array_equal(
        np.sum(trajectories.valid, axis=1), [4, 3, 2]
    )
    assert len(np.unique(trajectories.pixel_indices[1, trajectories.valid[1]])) == 3


def test_compose_dense_trajectories_uses_alternate_collision_candidate() -> None:
    point_map = np.zeros((2, 1, 2, 3), dtype=np.float32)
    point_map[0, 0, :, 0] = [0.0, 0.01]
    point_map[1, 0, :, 0] = [0.005, 0.008]
    scene_flow = np.zeros_like(point_map)
    scene_flow[0, 0, :, 0] = [0.006, -0.004]
    prediction = MotionCrafterPrediction(
        point_map=point_map,
        valid_mask=np.ones((2, 1, 2), dtype=bool),
        scene_flow=scene_flow,
        deform_mask=np.ones((2, 1, 2), dtype=bool),
    )

    trajectories = compose_dense_trajectories(
        prediction,
        np.ones((2, 1, 2), dtype=bool),
        seed_stride_pixels=1,
        maximum_transport_error_m=0.01,
        transport_candidate_count=2,
    )

    assert np.all(trajectories.valid)
    assert len(np.unique(trajectories.pixel_indices[1])) == 2


def test_graph_association_persistence_uses_only_allowed_prefix() -> None:
    trajectories = DenseMotionTrajectories(
        positions=np.array(
            [
                [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
                [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            ],
            dtype=np.float32,
        ),
        valid=np.array([[True, True], [True, False], [True, False]]),
        step_error_m=np.zeros((2, 2), dtype=np.float32),
        pixel_indices=np.array([[0, 1], [0, -1], [0, -1]]),
        seed_pixels_yx=np.array([[0, 0], [0, 1]], dtype=np.int32),
    )

    association = infer_graph_association(
        np.array([[0.01, 0.0, 0.0]]),
        np.empty((0, 2), dtype=np.int32),
        trajectories,
        candidate_count=1,
        minimum_trajectory_valid_fraction=1.0,
        association_frame_count=1,
    )

    assert association.trajectory_indices[0, 0] == 1
    assert association.candidate_valid_fraction[0, 0] == 1.0


def test_concatenate_dense_trajectories_preserves_camera_ids() -> None:
    def make_trajectories(offset: float) -> DenseMotionTrajectories:
        return DenseMotionTrajectories(
            positions=np.full((2, 2, 3), offset, dtype=np.float32),
            valid=np.ones((2, 2), dtype=bool),
            step_error_m=np.zeros((1, 2), dtype=np.float32),
            pixel_indices=np.tile(np.arange(2), (2, 1)),
            seed_pixels_yx=np.column_stack([np.zeros(2), np.arange(2)]).astype(
                np.int32
            ),
        )

    combined, camera_indices = concatenate_dense_trajectories(
        {2: make_trajectories(2.0), 0: make_trajectories(0.0)}
    )

    np.testing.assert_array_equal(camera_indices, [0, 0, 2, 2])
    np.testing.assert_allclose(
        combined.positions[0, :, 0], [0.0, 0.0, 2.0, 2.0]
    )


def test_training_motion_disambiguates_nearby_automatic_identities() -> None:
    trajectories = DenseMotionTrajectories(
        positions=np.array(
            [
                [[-0.001, 0.0, 0.0], [0.001, 0.0, 0.0]],
                [[0.019, 0.0, 0.0], [-0.019, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        valid=np.ones((2, 2), dtype=bool),
        step_error_m=np.zeros((1, 2), dtype=np.float32),
        pixel_indices=np.tile(np.arange(2), (2, 1)),
        seed_pixels_yx=np.array([[0, 0], [0, 1]], dtype=np.int32),
    )
    target = np.array([[[0.0, 0.0, 0.0]], [[0.02, 0.0, 0.0]]])

    association = infer_graph_association(
        np.array([[0.0, 0.0, 0.0]]),
        np.empty((0, 2), dtype=np.int32),
        trajectories,
        candidate_count=2,
        motion_scale_m=0.005,
        motion_strength=5.0,
        association_frame_count=2,
        graph_training_trajectory=target,
    )

    first_candidate = np.flatnonzero(association.trajectory_indices[0] == 0)[0]
    assert association.weights[0, first_candidate] > 0.99
    assert association.training_motion_error_m[0] < 1e-4


def test_graph_association_and_manual_audit_recover_dense_identities() -> None:
    graph = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]
    )
    dense_initial = np.array(
        [
            [0.0001, 0.0, 0.0],
            [0.0099, 0.0, 0.0],
            [0.0201, 0.0, 0.0],
            [0.0050, 0.01, 0.0],
            [0.0150, -0.01, 0.0],
        ]
    )
    dense_positions = np.stack(
        [dense_initial, dense_initial + np.array([0.0, 0.002, 0.0])]
    ).astype(np.float32)
    trajectories = DenseMotionTrajectories(
        positions=dense_positions,
        valid=np.ones((2, 5), dtype=bool),
        step_error_m=np.zeros((1, 5), dtype=np.float32),
        pixel_indices=np.tile(np.arange(5), (2, 1)),
        seed_pixels_yx=np.column_stack([np.zeros(5), np.arange(5)]).astype(
            np.int32
        ),
    )
    springs = np.array([[0, 1], [1, 2]], dtype=np.int32)

    association = infer_graph_association(
        graph,
        springs,
        trajectories,
        candidate_count=2,
        position_scale_m=0.003,
        graph_scale_m=0.005,
        graph_strength=0.5,
        collision_strength=0.2,
        mean_field_iterations=5,
        minimum_trajectory_valid_fraction=1.0,
    )
    observations, valid, reliability = apply_graph_association(
        trajectories, association, minimum_observation_mass=0.5
    )

    assert np.all(valid)
    assert np.all(reliability > 0.0)
    np.testing.assert_allclose(observations[0], graph, atol=7e-4)
    moved_graph = np.stack([graph, graph + np.array([0.0, 0.002, 0.0])])
    error = dense_graph_error_by_frame(
        moved_graph, observations, valid, reliability
    )
    assert np.max(error) < 7e-4

    manual = moved_graph.copy()
    audit = manual_track_association_audit(
        graph,
        observations,
        valid,
        manual,
        np.array([0, 1]),
    )
    assert audit["manual_track_count"] == 3
    assert audit["error_distribution_m"]["maximum"] < 7e-4
