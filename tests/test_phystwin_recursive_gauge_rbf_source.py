import numpy as np

from bayesian_phystwin.phystwin_recursive_gauge_rbf_source import (
    PhysTwinRecursiveGaugeRbfSourceConfig,
    action_conditioned_rigid_transition,
    build_dense_temporal_comparator,
    fixed_update_frames,
    run_recursive_gauge_rbf_source_prediction,
    sparse_frame_observation_belief,
)
from bayesian_phystwin.phystwin_sparse_identity_observation import (
    SparseIdentityObservations,
)
from bayesian_phystwin.recursive_gauge_rbf_belief import (
    initialize_recursive_gauge_rbf_belief,
)


def _geometry() -> np.ndarray:
    return np.asarray(
        [
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [-0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )


def _sparse_observations(points: np.ndarray) -> SparseIdentityObservations:
    frame_count, point_count, _ = points.shape
    shape = (frame_count, point_count)
    return SparseIdentityObservations(
        points_world_m=points,
        observation_covariance_m2=np.broadcast_to(
            np.eye(3) * 1e-6,
            (*shape, 3, 3),
        ),
        observation_variance_m2=np.full(shape, 1e-6),
        prior_reliability=np.ones(shape),
        valid=np.ones(shape, dtype=bool),
        raw_camera_count=np.full(shape, 3),
        effective_camera_count=np.full(shape, 3),
        reprojection_error_px=np.zeros(shape),
        redundant_view_disagreement_m=np.zeros(shape),
        two_view_fallback=np.zeros(shape, dtype=bool),
    )


def _trajectory(frame_count: int) -> np.ndarray:
    initial = _geometry()
    trajectory = np.repeat(initial[None], frame_count, axis=0)
    trajectory[:, :, 0] += np.arange(frame_count)[:, None] * 0.001
    return trajectory.astype(np.float32)


def test_fixed_update_frames_are_positive_unique_and_end_at_prefix() -> None:
    frames = fixed_update_frames(10, 4)

    assert len(frames) == 4
    assert np.all(np.diff(frames) > 0)
    assert frames[0] > 0
    assert frames[-1] == 9


def test_action_conditioned_transition_rotates_local_vectors() -> None:
    config = PhysTwinRecursiveGaugeRbfSourceConfig(center_count=4)
    source = _geometry()
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = source @ rotation.T
    snapshot = initialize_recursive_gauge_rbf_belief(
        np.arange(4),
        source,
        source,
        config=config.recursive,
    )

    transition = action_conditioned_rigid_transition(
        snapshot,
        source,
        target,
        neighbor_count=4,
    )

    np.testing.assert_allclose(transition[:3, :3], np.eye(3))
    for center in range(4):
        start = 3 + 3 * center
        np.testing.assert_allclose(
            transition[start : start + 3, start : start + 3],
            rotation,
            atol=1e-12,
        )


def test_sparse_frame_adapter_does_not_use_physical_innovation() -> None:
    points = _trajectory(5).astype(np.float64)
    observations = _sparse_observations(points)

    belief = sparse_frame_observation_belief(
        observations,
        case_id="synthetic",
        frame_index=3,
        entity_ids=np.asarray([0, 2]),
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
    )

    assert belief is not None
    np.testing.assert_array_equal(belief.entity_ids, [0, 2])
    np.testing.assert_array_equal(belief.prior_reliability, [1.0, 1.0])
    assert belief.metadata["reliability_uses_physical_innovation"] is False
    assert belief.factor_rank == 0


def test_zero_dense_residual_preserves_raw_comparator() -> None:
    raw = _trajectory(12)
    observed = raw[:8].astype(np.float64)
    visible = np.ones(observed.shape[:2], dtype=bool)
    motion_valid = np.ones((7, observed.shape[1]), dtype=bool)
    config = PhysTwinRecursiveGaugeRbfSourceConfig(center_count=4)

    comparator, _ = build_dense_temporal_comparator(
        raw,
        observed,
        visible,
        motion_valid,
        fit_end_frame=5,
        train_end_frame=8,
        config=config,
    )

    assert comparator.dtype == raw.dtype
    assert comparator.tobytes() == raw.tobytes()


def test_failed_prefix_gate_returns_dense_baseline_byte_exact() -> None:
    raw = _trajectory(12)
    observed = raw[:8].astype(np.float64)
    visible = np.ones(observed.shape[:2], dtype=bool)
    motion_valid = np.ones((7, observed.shape[1]), dtype=bool)
    sparse_points = observed.copy()
    local_direction = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    sparse_points += (
        np.linspace(0.0, 0.02, len(sparse_points))[:, None, None]
        * local_direction[None]
    )
    observations = _sparse_observations(sparse_points)
    config = PhysTwinRecursiveGaugeRbfSourceConfig(
        center_count=4,
        update_count=2,
        minimum_center_availability_fraction=0.0,
        minimum_rows_per_update=4,
        transport_neighbor_count=4,
        minimum_prefix_cd_improvement_fraction=0.01,
    )

    result = run_recursive_gauge_rbf_source_prediction(
        raw,
        observed,
        visible,
        motion_valid,
        observations,
        case_id="synthetic",
        fit_end_frame=5,
        train_end_frame=8,
        num_surface_points=4,
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        config=config,
    )

    assert not result.prefix_admitted
    assert result.candidate.dtype == result.dense_baseline.dtype
    assert result.candidate.tobytes() == result.dense_baseline.tobytes()
    assert result.diagnostics["prefix_gate_uses_manual_tracks"] is False
