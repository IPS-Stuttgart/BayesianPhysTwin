from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    DirectDepthEndpointObservations,
    build_direct_depth_birth_anchored_measurements,
    build_direct_depth_endpoint_observations,
)
from bayesian_phystwin.deform360_dynamic_query import CameraPanel
from bayesian_phystwin.deform360_sentinel_assimilation import (
    build_sentinel_debiased_measurements,
)
from bayesian_phystwin.deform360_sentinel_query_schedule import (
    DIRECT_DEPTH_PROTOCOL_ID,
    Deform360SentinelQueryConfig,
    Deform360SentinelQuerySchedule,
)
from bayesian_phystwin.phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
)


def _schedule() -> Deform360SentinelQuerySchedule:
    config = Deform360SentinelQueryConfig(
        selected_camera_count=3,
        minimum_eligible_camera_count=3,
        total_query_count=4,
        sentinel_query_count=2,
        minimum_camera_support=3,
        graph_basis_rank=2,
        query_birth_frame=51,
        protocol_id=DIRECT_DEPTH_PROTOCOL_ID,
    )
    return Deform360SentinelQuerySchedule(
        update_frames=np.full(4, 57, dtype=np.int64),
        birth_frames=np.full(4, 51, dtype=np.int64),
        entity_ids=np.arange(4, dtype=np.int64),
        query_roles=np.asarray(
            [
                ACTIVE_QUERY_ROLE,
                ACTIVE_QUERY_ROLE,
                SENTINEL_QUERY_ROLE,
                SENTINEL_QUERY_ROLE,
            ]
        ),
        predicted_motion_m=np.asarray([0.02, 0.03, 0.0002, 0.0003]),
        predicted_visible_views=np.full(4, 3, dtype=np.int64),
        information_gain=np.ones(4),
        config=config,
        camera_panel=CameraPanel(
            camera_indices=np.arange(3),
            camera_names=("a", "b", "c"),
            frame_zero_coverage=np.ones(3),
            selection_scores=np.ones(3),
        ),
        physical_prefix_sha256="0" * 64,
        graph_basis_sha256="1" * 64,
        artifact_sha256="2" * 64,
    )


def _inputs(
    *,
    camera_count: int = 3,
    update_depth_bias_m: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    physical: np.ndarray = np.zeros((76, 6, 3), dtype=np.float64)
    physical[:, :4, 0] = np.asarray([-0.15, -0.05, 0.05, 0.15])
    physical[:, :4, 2] = 1.0
    physical[57:, 0, 0] += 0.02
    physical[57:, 1, 1] += 0.03
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    depths: np.ndarray = np.ones(
        (camera_count, 58, 64, 64),
        dtype=np.float32,
    )
    depths[:, 57] += update_depth_bias_m
    masks = np.zeros_like(depths, dtype=bool)
    for camera in range(camera_count):
        for frame in (51, 57):
            for entity in range(4):
                point = physical[frame, entity]
                x = int(np.rint(100.0 * point[0] / point[2] + 32.0))
                y = int(np.rint(100.0 * point[1] / point[2] + 32.0))
                masks[camera, frame, y, x] = True
    return physical, intrinsics, poses, depths, masks


def _provider(
    *,
    camera_count: int = 3,
    update_depth_bias_m: float = 0.0,
):
    physical, intrinsics, poses, depths, masks = _inputs(
        camera_count=camera_count,
        update_depth_bias_m=update_depth_bias_m,
    )
    observations = build_direct_depth_endpoint_observations(
        physical,
        _schedule(),
        intrinsics,
        poses,
        depths,
        masks,
        config=DirectDepthEndpointConfig(search_radius_px=3),
    )
    return physical, observations


def test_direct_depth_builds_metric_endpoint_beliefs() -> None:
    _, observations = _provider()

    assert np.all(observations.accepted_support)
    assert np.all(observations.support_count == 3)
    assert np.all(np.isfinite(observations.point_world_m))
    assert np.all(np.linalg.eigvalsh(observations.covariance_m2) > 0.0)
    assert np.all(
        (observations.association_probability > 0.0)
        & (observations.association_probability <= 1.0)
    )


def test_duplicate_correlated_cameras_do_not_shrink_covariance() -> None:
    _, three = _provider(camera_count=3)
    _, six = _provider(camera_count=6)

    np.testing.assert_allclose(
        six.covariance_m2,
        three.covariance_m2,
        rtol=0.0,
        atol=1e-15,
    )


def test_unknown_correlation_is_not_naively_independent() -> None:
    _, observations = _provider(camera_count=3)
    naive_independent = observations.covariance_m2 / 3.0

    difference = observations.covariance_m2 - naive_independent
    assert np.all(np.linalg.eigvalsh(difference) >= -1e-15)


def test_state_innovation_does_not_change_prior_reliability() -> None:
    physical, nominal = _provider(update_depth_bias_m=0.0)
    _, shifted = _provider(update_depth_bias_m=0.02)
    nominal_measurements = build_direct_depth_birth_anchored_measurements(
        nominal,
        physical,
    )
    shifted_measurements = build_direct_depth_birth_anchored_measurements(
        shifted,
        physical,
    )

    np.testing.assert_array_equal(
        nominal_measurements.prior_reliability,
        shifted_measurements.prior_reliability,
    )
    assert np.all(
        shifted_measurements.prior_reliability[
            57,
            shifted_measurements.entity_ids,
        ]
        == 1.0
    )
    assert not np.allclose(
        nominal_measurements.measurement_m[57, :4],
        shifted_measurements.measurement_m[57, :4],
    )
    assert np.all(
        shifted_measurements.association_probability[57, :4]
        <= nominal_measurements.association_probability[57, :4]
    )


def test_sentinels_remove_a_shared_depth_endpoint_bias() -> None:
    physical, observations = _provider()
    biased_points = observations.point_world_m.copy()
    biased_points[1] += np.asarray([0.0, 0.0, 0.01])
    observations = DirectDepthEndpointObservations(
        endpoint_frames=observations.endpoint_frames,
        entity_ids=observations.entity_ids,
        point_world_m=biased_points,
        covariance_m2=observations.covariance_m2,
        accepted_support=observations.accepted_support,
        association_probability=observations.association_probability,
        support_count=observations.support_count,
        maximum_view_scatter_m=observations.maximum_view_scatter_m,
        config=observations.config,
    )
    measurements = build_direct_depth_birth_anchored_measurements(
        observations,
        physical,
    )
    result = build_sentinel_debiased_measurements(
        measurements,
        _schedule(),
        physical,
    )

    assert result.applied
    np.testing.assert_allclose(
        result.estimate.bias_m,
        np.asarray([0.0, 0.0, 0.01]),
        atol=2e-4,
    )
    np.testing.assert_allclose(
        result.measurements.measurement_m[57, :2],
        physical[57, :2],
        atol=2e-4,
    )
    assert not np.any(result.measurements.available[:, 2:])
