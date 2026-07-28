from pathlib import Path

import numpy as np

from bayesian_phystwin import PseudoMeasurementBatch
from bayesian_phystwin.observation_belief import (
    load_observation_belief,
    save_observation_belief,
)
from bayesian_phystwin.robust_likelihood import robust_mixture_likelihood
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    TAPNEXT_CHECKPOINT_SHA256,
    build_dynamic_tapnextpp_observation_belief,
    camera_projection_matrix,
    fuse_dynamic_tapnextpp_multiview,
    project_world_point,
)


def _look_at_pose(center: np.ndarray) -> np.ndarray:
    forward = -center / np.linalg.norm(center)
    up = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, up, forward])
    pose[:3, 3] = center
    return pose


def _synthetic_input(
    camera_count: int,
    *,
    duplicate_first: bool = False,
    frame_count: int = 4,
) -> dict[str, np.ndarray]:
    base_count = camera_count - int(duplicate_first)
    angles = np.linspace(0.0, 2.0 * np.pi, base_count, endpoint=False)
    poses = np.stack(
        [
            _look_at_pose(
                np.asarray([2.0 * np.sin(angle), 0.3, -2.0 * np.cos(angle)])
            )
            for angle in angles
        ]
    )
    if duplicate_first:
        poses = np.concatenate([poses, poses[:1]], axis=0)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 300.0
    intrinsics[:, 1, 1] = 300.0
    intrinsics[:, 0, 2] = 64.0
    intrinsics[:, 1, 2] = 64.0
    projections = np.stack(
        [
            camera_projection_matrix(intrinsics[index], poses[index])
            for index in range(camera_count)
        ]
    )
    points = np.asarray([[0.0, 0.0, 0.0]])
    tracks = np.zeros((camera_count, frame_count, 1, 2))
    depths = np.zeros((camera_count, frame_count, 128, 128))
    masks = np.ones_like(depths, dtype=bool)
    for camera in range(camera_count):
        pixel, _ = project_world_point(points[0], projections[camera])
        tracks[camera, :, 0] = pixel
        camera_depth = (
            np.linalg.inv(poses[camera]) @ np.asarray([0.0, 0.0, 0.0, 1.0])
        )[2]
        depths[camera] = camera_depth
    return {
        "tracks_xy": tracks,
        "visibility_probability": np.ones((camera_count, frame_count, 1)),
        "depths_m": depths,
        "object_masks": masks,
        "intrinsics": intrinsics,
        "camera_to_world": poses,
        "query_points_world_m": points,
        "association_valid": np.ones((camera_count, 1), dtype=bool),
        "association_probability": np.full((camera_count, 1), 0.8),
        "association_entropy": np.full((camera_count, 1), 0.1),
        "assignment_pixel_covariance_px2": np.repeat(
            (np.eye(2) * 0.25)[None, None],
            camera_count,
            axis=0,
        ),
    }


def _fuse(values: dict[str, np.ndarray]):
    return fuse_dynamic_tapnextpp_multiview(**values)


def test_two_views_are_proposals_but_not_claim_bearing() -> None:
    result = _fuse(_synthetic_input(2))
    assert np.all(result.proposal_available)
    assert not np.any(result.accepted_support)
    assert np.all(result.independent_support_count == 2)


def test_no_claim_bearing_rows_seal_as_prior_only_fallback(
    tmp_path: Path,
) -> None:
    result = _fuse(_synthetic_input(2))
    belief = build_dynamic_tapnextpp_observation_belief(
        result,
        case_id="synthetic-prior-only",
        frame_ids=np.asarray([0, 1, 2, 3]),
        entity_ids=np.asarray([7]),
        entity_birth_frames=np.asarray([0]),
        entity_update_frames=np.asarray([3]),
        camera_names=("camera-0", "camera-1"),
        query_schedule_sha256="a" * 64,
        tracker_checkpoint_sha256=TAPNEXT_CHECKPOINT_SHA256,
    )

    assert belief.observation_count == 0
    assert belief.metadata["prior_only_fallback"] is True
    path = tmp_path / "prior-only-belief.npz"
    save_observation_belief(path, belief)
    restored = load_observation_belief(path)
    assert restored.artifact_id == belief.artifact_id


def test_three_independent_views_are_claim_bearing() -> None:
    result = _fuse(_synthetic_input(3))
    assert np.all(result.proposal_available)
    assert np.all(result.accepted_support)
    assert np.all(result.independent_support_count == 3)
    np.testing.assert_allclose(result.trajectory_world_m, 0.0, atol=1e-10)


def test_duplicate_camera_adds_no_support_or_confidence() -> None:
    base = _fuse(_synthetic_input(3))
    duplicated = _fuse(_synthetic_input(4, duplicate_first=True))
    assert np.all(duplicated.independent_support_count == 3)
    assert np.all(duplicated.raw_support_count == 4)
    np.testing.assert_allclose(
        duplicated.local_covariance_m2,
        base.local_covariance_m2,
        rtol=1e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        duplicated.prior_reliability,
        base.prior_reliability,
        rtol=1e-12,
        atol=1e-14,
    )


def test_unknown_correlation_is_not_more_confident_than_independence() -> None:
    result = _fuse(_synthetic_input(3))
    difference = (
        result.local_covariance_m2[0, 0]
        - result.naive_independent_covariance_m2[0, 0]
    )
    assert np.min(np.linalg.eigvalsh(difference)) >= -1e-12
    assert np.trace(result.local_covariance_m2[0, 0]) > np.trace(
        result.naive_independent_covariance_m2[0, 0]
    )


def test_shared_bias_is_low_rank_not_a_local_floor(tmp_path: Path) -> None:
    result = _fuse(_synthetic_input(3))
    belief = build_dynamic_tapnextpp_observation_belief(
        result,
        case_id="synthetic",
        frame_ids=np.asarray([0, 1, 2, 3]),
        entity_ids=np.asarray([7]),
        entity_birth_frames=np.asarray([0]),
        entity_update_frames=np.asarray([3]),
        camera_names=("camera-0", "camera-1", "camera-2"),
        query_schedule_sha256="a" * 64,
        tracker_checkpoint_sha256=TAPNEXT_CHECKPOINT_SHA256,
    )
    expected_factor = np.eye(3) * 0.005
    np.testing.assert_allclose(
        belief.low_rank_factor_m,
        np.repeat(expected_factor[None], belief.observation_count, axis=0),
    )
    assert belief.metadata["local_covariance_definition"].endswith(
        "excludes coherent camera bias"
    )
    assert np.all(belief.factor_group_ids == 0)
    assert belief.group_composite_weight[0] == 0.75

    path = tmp_path / "dynamic_tapnextpp_belief.npz"
    save_observation_belief(path, belief)
    restored = load_observation_belief(path)
    assert restored.artifact_id == belief.artifact_id


def test_physical_residual_does_not_change_prior_reliability() -> None:
    result = _fuse(_synthetic_input(3))
    prior = result.prior_reliability.copy()
    observed = result.trajectory_world_m.reshape(-1, 3)
    near = PseudoMeasurementBatch(
        observed=observed,
        predicted=observed + 1e-4,
        variance=1e-6,
    )
    gross = PseudoMeasurementBatch(
        observed=observed,
        predicted=observed + 0.2,
        variance=1e-6,
    )
    near_result = robust_mixture_likelihood(
        near,
        prior_reliability=prior.ravel(),
    )
    gross_result = robust_mixture_likelihood(
        gross,
        prior_reliability=prior.ravel(),
    )
    np.testing.assert_array_equal(result.prior_reliability, prior)
    assert np.max(gross_result.posterior_inlier_probability) < np.min(
        near_result.posterior_inlier_probability
    )


def test_assignment_ambiguity_increases_metric_covariance() -> None:
    unambiguous_input = _synthetic_input(3)
    ambiguous_input = _synthetic_input(3)
    ambiguous_input["assignment_pixel_covariance_px2"] *= 100.0
    unambiguous = _fuse(unambiguous_input)
    ambiguous = _fuse(ambiguous_input)
    assert np.trace(ambiguous.local_covariance_m2[0, 0]) > np.trace(
        unambiguous.local_covariance_m2[0, 0]
    )
    assert np.trace(
        ambiguous.assignment_mixture_spread_m2[0, 0]
    ) > np.trace(unambiguous.assignment_mixture_spread_m2[0, 0])
