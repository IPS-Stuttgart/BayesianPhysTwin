import numpy as np

from bayesian_phystwin.deform360_dynamic_query import (
    CameraPanel,
    DynamicQueryConfig,
    DynamicQuerySchedule,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    SELECTED_BACKBONE_ARM,
    build_birth_anchored_measurements,
    predict_dynamic_tapnextpp_candidate,
)
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    DynamicMultiviewConfig,
    DynamicMultiviewResult,
)


def _physical() -> np.ndarray:
    count = 30
    base = np.stack(
        (
            np.linspace(-0.15, 0.15, count),
            0.02 * np.sin(np.linspace(0.0, 3.0 * np.pi, count)),
            np.linspace(0.8, 0.9, count),
        ),
        axis=1,
    )
    trajectory = np.repeat(base[None], 76, axis=0)
    trajectory[:, :, 1] += np.arange(76)[:, None] * np.linspace(
        0.0001,
        0.0004,
        count,
    )[None]
    return trajectory


def _schedule() -> DynamicQuerySchedule:
    entities = np.arange(27, dtype=np.int64)
    updates = np.repeat(np.asarray([19, 38, 57]), 9)
    births = np.concatenate(
        (
            np.repeat(18, 9),
            np.repeat(37, 9),
            np.repeat(56, 9),
        )
    )
    panel = CameraPanel(
        camera_indices=np.arange(8),
        camera_names=tuple(f"camera-{index}" for index in range(8)),
        frame_zero_coverage=np.ones(8),
        selection_scores=np.ones(8),
    )
    return DynamicQuerySchedule(
        update_frames=updates,
        birth_frames=births,
        entity_ids=entities,
        predicted_motion_m=np.full(27, 0.01),
        predicted_visible_views=np.full(27, 3),
        information_gain=np.ones(27),
        config=DynamicQueryConfig(),
        camera_panel=panel,
        physical_prefix_sha256="a" * 64,
        graph_basis_sha256="b" * 64,
        artifact_sha256="c" * 64,
    )


def _result(
    physical: np.ndarray,
    schedule: DynamicQuerySchedule,
    *,
    covariance_scale: float = 1.0,
    support_count: int = 27,
) -> DynamicMultiviewResult:
    frame_count = 58
    entity_count = len(schedule.entity_ids)
    trajectory = np.zeros((frame_count, entity_count, 3))
    accepted = np.zeros((frame_count, entity_count), dtype=bool)
    correction = np.asarray([0.012, -0.004, 0.003])
    shared_bias = np.asarray([0.18, -0.09, 0.04])
    for row, (entity, birth, update) in enumerate(
        zip(
            schedule.entity_ids,
            schedule.birth_frames,
            schedule.update_frames,
            strict=True,
        )
    ):
        trajectory[:, row] = physical[:frame_count, entity] + shared_bias
        trajectory[update, row] += correction
        if row < support_count:
            accepted[birth, row] = True
            accepted[update, row] = True
    covariance = np.repeat(
        (np.eye(3) * 1e-6 * covariance_scale)[None, None],
        frame_count * entity_count,
        axis=0,
    ).reshape(frame_count, entity_count, 3, 3)
    support = accepted.astype(np.int64) * 3
    config = DynamicMultiviewConfig()
    return DynamicMultiviewResult(
        trajectory_world_m=trajectory,
        proposal_available=accepted.copy(),
        accepted_support=accepted,
        prior_reliability=np.where(accepted, 0.9, 0.0),
        association_probability=np.where(accepted, 0.8, 0.0),
        local_covariance_m2=covariance,
        naive_independent_covariance_m2=covariance * 0.5,
        assignment_mixture_spread_m2=np.zeros_like(covariance),
        independent_support_count=support,
        raw_support_count=support,
        reprojection_rmse_px=np.zeros((frame_count, entity_count)),
        depth_residual_rmse_m=np.zeros((frame_count, entity_count)),
        inlier_camera_mask=np.repeat(
            accepted[None],
            3,
            axis=0,
        ),
        camera_cluster_ids=np.arange(3),
        shared_bias_standard_deviation_m=(
            config.shared_bias_standard_deviation_m
        ),
        config=config,
    )


def test_birth_anchor_cancels_a_large_shared_absolute_bias() -> None:
    physical = _physical()
    schedule = _schedule()
    result = _result(physical, schedule)

    measurements = build_birth_anchored_measurements(
        result,
        schedule,
        physical,
    )

    expected = np.asarray([0.012, -0.004, 0.003])
    for entity, update in zip(
        schedule.entity_ids,
        schedule.update_frames,
        strict=True,
    ):
        np.testing.assert_allclose(
            measurements.measurement_m[update, entity] - physical[update, entity],
            expected,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            measurements.covariance_m2[update, entity],
            np.eye(3) * 4e-6,
        )


def test_covariance_aware_dynamic_candidate_updates_and_keeps_measurements_hidden() -> None:
    physical = _physical().astype(np.float32)
    persistence = np.repeat(physical[:1], 76, axis=0)
    schedule = _schedule()
    measurements = build_birth_anchored_measurements(
        _result(physical, schedule),
        schedule,
        physical,
    )

    report, arrays = predict_dynamic_tapnextpp_candidate(
        physical,
        persistence,
        measurements,
    )

    assert all(item["pairwise_gate"]["accepted"] for item in report["updates"])
    assert not np.array_equal(
        arrays[CANDIDATE_ARM][20:],
        arrays[SELECTED_BACKBONE_ARM][20:],
    )
    assert set(report["information_boundary"]) == {
        "future_target_read",
        "future_object_geometry_read",
        "prediction_depends_on",
    }
    assert report["information_boundary"]["future_target_read"] is False


def test_larger_metric_covariance_reduces_the_candidate_update() -> None:
    physical = _physical().astype(np.float32)
    persistence = np.repeat(physical[:1], 76, axis=0)
    schedule = _schedule()
    low = build_birth_anchored_measurements(
        _result(physical, schedule, covariance_scale=1.0),
        schedule,
        physical,
    )
    high = build_birth_anchored_measurements(
        _result(physical, schedule, covariance_scale=10_000.0),
        schedule,
        physical,
    )

    _, low_arrays = predict_dynamic_tapnextpp_candidate(
        physical,
        persistence,
        low,
    )
    _, high_arrays = predict_dynamic_tapnextpp_candidate(
        physical,
        persistence,
        high,
    )
    low_update = np.linalg.norm(
        low_arrays[CANDIDATE_ARM][20] - low_arrays[SELECTED_BACKBONE_ARM][20]
    )
    high_update = np.linalg.norm(
        high_arrays[CANDIDATE_ARM][20] - high_arrays[SELECTED_BACKBONE_ARM][20]
    )
    assert high_update < low_update


def test_insufficient_pairwise_support_is_bit_exact_fallback() -> None:
    physical = _physical().astype(np.float32)
    persistence = np.repeat(physical[:1], 76, axis=0)
    schedule = _schedule()
    measurements = build_birth_anchored_measurements(
        _result(physical, schedule, support_count=8),
        schedule,
        physical,
    )

    report, arrays = predict_dynamic_tapnextpp_candidate(
        physical,
        persistence,
        measurements,
    )

    assert not any(item["pairwise_gate"]["accepted"] for item in report["updates"])
    np.testing.assert_array_equal(
        arrays[CANDIDATE_ARM],
        arrays[SELECTED_BACKBONE_ARM],
    )
