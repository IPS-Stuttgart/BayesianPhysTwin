from __future__ import annotations

import copy

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_metric_object_carrier import (
    DEFORM360_METRIC_OBJECT_CARRIER_LOCK_SCHEMA,
    METRIC_OBJECT_CARRIER_INFORMATION_BOUNDARY,
    METRIC_OBJECT_CARRIER_POLICY,
    BlockPointCandidates,
    build_metric_object_carrier,
    cover_resize_mask_nearest,
    deterministic_farthest_point_indices,
    fuse_unknown_correlated_points,
    mutual_nearest_mapping,
    reduce_masked_point_map,
    validate_metric_object_carrier_lock,
)
from bayesian_phystwin.deform360_tactile_metric_gauge import SimilarityTransform


def _lock() -> dict[str, object]:
    descriptor: dict[str, object] = {
        "schema": DEFORM360_METRIC_OBJECT_CARRIER_LOCK_SCHEMA,
        "schema_version": 1,
        "status": "locked-source-only-pre-mask",
        "implementation": {
            "revision": "a" * 40,
            "runner_source_sha256": "b" * 64,
            "module_source_sha256": "c" * 64,
        },
        "source_case": {
            "object_id": "026-sock-cloth",
            "processing_episode_index": 0,
            "causal_frame_stop": 150,
        },
        "parents": {
            "metric_gauge": {
                "artifact_id": "d" * 64,
                "sha256": "e" * 64,
            }
        },
        "cameras": ["reference", "left", "right"],
        "reference_camera": "reference",
        "providers": [
            {
                "camera": camera,
                "video_path": f"{camera}.mp4",
                "video_sha256": str(index) * 64,
                "prediction_manifest_path": f"{camera}.json",
                "prediction_manifest_sha256": str(index + 3) * 64,
                "window_path": f"{camera}.npz",
                "window_sha256": str(index + 6) * 64,
                "window_source_frames": [125, 150],
            }
            for index, camera in enumerate(("reference", "left", "right"), start=1)
        ],
        "sam2": {
            "repository_path": "/sam2",
            "repository_revision": "f" * 40,
            "checkpoint_path": "/sam2/checkpoint.pt",
            "checkpoint_sha256": "1" * 64,
            "selector_source_path": "/selector.py",
            "selector_source_sha256": "2" * 64,
        },
        "policy": METRIC_OBJECT_CARRIER_POLICY,
        "information_boundary": METRIC_OBJECT_CARRIER_INFORMATION_BOUNDARY,
        "claim_boundary": "source only",
    }
    return {"artifact_id": content_id(descriptor), **descriptor}


def test_carrier_lock_is_content_addressed_and_fail_closed() -> None:
    value = _lock()
    assert validate_metric_object_carrier_lock(value) == value["artifact_id"]
    changed = copy.deepcopy(value)
    changed["policy"]["carrier_node_count"] = 64
    with np.testing.assert_raises_regex(ValueError, "identity changed"):
        validate_metric_object_carrier_lock(changed)


def test_cover_resize_mask_matches_center_crop() -> None:
    source = np.zeros((4, 4), dtype=bool)
    source[1:3] = True
    result = cover_resize_mask_nearest(source, target_shape=(2, 4))
    assert result.shape == (2, 4)
    assert np.all(result)


def test_prior_reliability_does_not_depend_on_state_innovation() -> None:
    point_map = np.zeros((8, 8, 3), dtype=np.float64)
    yy, xx = np.mgrid[:8, :8]
    point_map[..., 0] = xx * 0.001
    point_map[..., 1] = yy * 0.001
    valid = np.ones((8, 8), dtype=bool)
    mask = np.ones((8, 8), dtype=bool)
    deform = np.ones((8, 8), dtype=bool)
    transform = SimilarityTransform(1.0, np.eye(3), np.zeros(3))
    kwargs = {
        "transform": transform,
        "gauge_covariance_m2": np.eye(3) * 1e-4,
        "block_size_px": 4,
        "minimum_mask_pixels": 4,
        "minimum_valid_fraction": 0.5,
        "full_reliability_deform_fraction": 0.5,
        "covariance_floor_m": 0.005,
    }
    first = reduce_masked_point_map(point_map, valid, mask, deform, **kwargs)
    shifted = reduce_masked_point_map(
        point_map + np.asarray([10.0, -3.0, 2.0]),
        valid,
        mask,
        deform,
        **kwargs,
    )
    assert np.array_equal(first.prior_reliability, shifted.prior_reliability)
    assert not np.allclose(first.points_world_m, shifted.points_world_m)


def test_unknown_correlation_does_not_gain_confidence_from_duplicates() -> None:
    point = np.asarray([[0.1, 0.2, 0.3]])
    covariance = np.asarray([np.diag([1e-4, 2e-4, 3e-4])])
    one_mean, one_covariance = fuse_unknown_correlated_points(point, covariance)
    many_mean, many_covariance = fuse_unknown_correlated_points(
        np.repeat(point, 20, axis=0),
        np.repeat(covariance, 20, axis=0),
    )
    independent = covariance[0] / 20
    assert np.allclose(one_mean, many_mean)
    assert np.allclose(one_covariance, many_covariance)
    assert np.all(np.linalg.eigvalsh(many_covariance - independent) >= -1e-12)


def test_mutual_nearest_rejects_nonmutual_and_distant_rows() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]])
    candidate = np.asarray([[0.001, 0.0, 0.0], [0.012, 0.0, 0.0]])
    mapping, distance = mutual_nearest_mapping(
        reference, candidate, maximum_distance_m=0.02
    )
    assert mapping.tolist() == [0, 1, -1]
    assert np.isinf(distance[-1])


def test_deterministic_fps_is_order_stable_for_unique_grid() -> None:
    grid = np.asarray([[x, y] for y in range(4) for x in range(5)], dtype=float)
    first = deterministic_farthest_point_indices(grid, 8)
    repeated = deterministic_farthest_point_indices(grid, 8)
    assert np.array_equal(first, repeated)
    assert len(np.unique(first)) == 8


def _candidates(points: np.ndarray, offset: np.ndarray) -> BlockPointCandidates:
    values = points + offset
    count = len(values)
    return BlockPointCandidates(
        block_yx=np.column_stack((np.arange(count) // 16, np.arange(count) % 16)),
        pixel_xy=np.column_stack((np.arange(count) % 16, np.arange(count) // 16)).astype(float),
        points_world_m=values,
        covariance_m2=np.repeat((np.eye(3) * 1e-4)[None], count, axis=0),
        prior_reliability=np.full(count, 0.8),
        deform_fraction=np.ones(count),
    )


def test_carrier_preserves_assignment_spread_and_three_view_support() -> None:
    yy, xx = np.mgrid[:9, :16]
    base = np.column_stack((xx.ravel() * 0.01, yy.ravel() * 0.01, np.zeros(xx.size)))
    cameras = ("reference", "left", "right")
    assignments = []
    for assignment_shift in (0.0, 0.004):
        assignments.append(
            {
                "reference": _candidates(base, np.asarray([assignment_shift, 0.0, 0.0])),
                "left": _candidates(base, np.asarray([assignment_shift + 0.001, 0.0, 0.0])),
                "right": _candidates(base, np.asarray([assignment_shift - 0.001, 0.0, 0.0])),
            }
        )
    carrier = build_metric_object_carrier(
        assignments,
        camera_order=cameras,
        reference_camera="reference",
        maximum_distance_m=0.005,
        node_count=128,
    )
    assert carrier.points_world_m.shape == (2, 128, 3)
    assert carrier.contributor_indices.shape == (2, 128, 3)
    assert np.all(carrier.contributor_indices >= 0)
    assert np.all(carrier.prior_reliability <= 0.8)
    assert np.all(carrier.assignment_mixture_covariance_m2[:, 0, 0] > 0.0)
    assert np.all(
        np.linalg.eigvalsh(
            carrier.marginal_covariance_m2 - carrier.covariance_m2
        )
        >= -1e-12
    )
