from __future__ import annotations

import numpy as np

from bayesian_phystwin.tapnextpp_depth_completion import (
    PerCameraMetricTracks,
    TAPNextPPDepthCompletionConfig,
    complete_strict_multiview_carrier,
    lift_per_camera_rgbd_tracks,
)


def _metric_tracks(
    points: np.ndarray,
    valid: np.ndarray,
) -> PerCameraMetricTracks:
    rows = valid.shape
    covariance = np.broadcast_to(np.eye(3) * 1e-6, (*rows, 3, 3)).copy()
    return PerCameraMetricTracks(
        points_world_m=points,
        valid=valid,
        prior_reliability=np.where(valid, 0.8, 0.0),
        covariance_m2=covariance,
        local_depth_mad_m=np.zeros(rows),
        object_mask_fraction=np.where(valid, 1.0, 0.0),
    )


def _strict_carrier() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.zeros((4, 2, 3), dtype=np.float64)
    points[:, :, 0] = np.arange(4)[:, None] * 0.01
    support = np.zeros((4, 2), dtype=bool)
    support[:2] = True
    reliability = np.where(support, 0.9, 0.0)
    covariance = np.broadcast_to(
        np.eye(3) * 25e-6,
        (*support.shape, 3, 3),
    ).copy()
    return points, support, reliability, covariance


def test_lift_anchors_every_camera_at_frame_zero() -> None:
    tracks = np.asarray([[[[1.0, 1.0]], [[1.0, 1.0]]]])
    visibility = np.ones((1, 2, 1))
    depths = np.ones((1, 2, 3, 3))
    depths[:, 1] = 2.0
    masks = np.ones_like(depths, dtype=bool)
    intrinsics = np.eye(3)[None]
    poses = np.eye(4)[None]
    frame_zero = np.asarray([[1.0, 1.0, 1.0]])

    result = lift_per_camera_rgbd_tracks(
        tracks,
        visibility,
        depths,
        masks,
        intrinsics,
        poses,
        frame_zero,
    )

    np.testing.assert_allclose(result.points_world_m[0, 0, 0], frame_zero[0])
    np.testing.assert_allclose(result.points_world_m[0, 1, 0], [2.0, 2.0, 2.0])
    assert np.all(result.valid)
    assert np.all(result.prior_reliability > 0.0)
    assert np.min(np.linalg.eigvalsh(result.covariance_m2)) >= 0.0


def test_overlap_penalty_selects_broad_target_free_camera() -> None:
    carrier, support, reliability, covariance = _strict_carrier()
    camera_zero = carrier.copy()
    camera_zero[:2, :, 1] += np.asarray([[0.0, 0.002], [0.0, 0.002]])
    camera_zero[2:, :, 0] += 0.03
    camera_one = carrier.copy()
    camera_one[0, :, 1] += np.asarray([0.0, 0.0016])
    camera_one[2:, :, 0] += 0.20
    points = np.stack((camera_zero, camera_one))
    valid = np.ones((2, 4, 2), dtype=bool)
    valid[1, 1] = False
    config = TAPNextPPDepthCompletionConfig(
        minimum_carrier_overlap_rows=2,
        minimum_carrier_overlap_fraction=0.25,
        maximum_penalized_agreement_m=0.010,
    )

    result = complete_strict_multiview_carrier(
        carrier,
        support,
        reliability,
        covariance,
        _metric_tracks(points, valid),
        config=config,
    )

    assert result.accepted
    assert result.selected_camera == 0
    assert np.all(result.support)
    assert np.array_equal(result.points_world_m[support], carrier[support])
    assert np.all(result.source_camera[support] == -1)
    assert np.all(result.source_camera[~support] == 0)


def test_duplicate_correlated_camera_does_not_increase_confidence() -> None:
    carrier, support, reliability, covariance = _strict_carrier()
    camera = carrier.copy()
    camera[:2, :, 1] += np.asarray([[0.0, 0.002], [0.0, 0.002]])
    camera[2:, :, 0] += 0.03
    config = TAPNextPPDepthCompletionConfig(
        minimum_carrier_overlap_rows=2,
        maximum_penalized_agreement_m=0.010,
    )
    single = _metric_tracks(camera[None], np.ones((1, 4, 2), dtype=bool))
    duplicated = _metric_tracks(
        np.stack((camera, camera)),
        np.ones((2, 4, 2), dtype=bool),
    )

    one = complete_strict_multiview_carrier(
        carrier,
        support,
        reliability,
        covariance,
        single,
        config=config,
    )
    two = complete_strict_multiview_carrier(
        carrier,
        support,
        reliability,
        covariance,
        duplicated,
        config=config,
    )

    assert one.selected_camera == two.selected_camera == 0
    assert np.array_equal(one.points_world_m, two.points_world_m)
    assert np.array_equal(one.prior_reliability, two.prior_reliability)
    assert np.array_equal(one.covariance_m2, two.covariance_m2)


def test_completed_covariance_retains_shared_bias_floor() -> None:
    carrier, support, reliability, covariance = _strict_carrier()
    camera = carrier.copy()
    camera[2:, :, 0] += 0.03
    config = TAPNextPPDepthCompletionConfig(
        shared_bias_std_m=0.005,
        minimum_carrier_overlap_rows=2,
        maximum_penalized_agreement_m=0.010,
    )
    result = complete_strict_multiview_carrier(
        carrier,
        support,
        reliability,
        covariance,
        _metric_tracks(camera[None], np.ones((1, 4, 2), dtype=bool)),
        config=config,
    )

    fallback_covariance = result.covariance_m2[~support]
    minimum = np.min(np.linalg.eigvalsh(fallback_covariance))
    assert minimum >= config.shared_bias_std_m**2


def test_failed_camera_gate_is_exact_carrier_fallback() -> None:
    carrier, support, reliability, covariance = _strict_carrier()
    camera = carrier.copy()
    camera[:2, :, 1] += np.asarray([[0.0, 0.10], [0.0, 0.10]])
    valid = np.ones((1, 4, 2), dtype=bool)
    config = TAPNextPPDepthCompletionConfig(
        minimum_carrier_overlap_rows=2,
        maximum_penalized_agreement_m=0.001,
    )

    result = complete_strict_multiview_carrier(
        carrier,
        support,
        reliability,
        covariance,
        _metric_tracks(camera[None], valid),
        config=config,
    )

    assert not result.accepted
    assert result.selected_camera is None
    assert np.array_equal(result.points_world_m, carrier)
    assert np.array_equal(result.support, support)
    assert np.array_equal(result.prior_reliability, reliability)
    assert np.array_equal(result.covariance_m2, covariance)
