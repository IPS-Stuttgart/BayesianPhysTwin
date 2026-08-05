from __future__ import annotations

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_tactile_metric_gauge import (
    CONTACT_CAMERA_POLICY,
    DEFORM360_TACTILE_METRIC_GAUGE_LOCK_SCHEMA,
    DEFORM360_TACTILE_METRIC_GAUGE_TWO_VIEW_LOCK_SCHEMA,
    TACTILE_METRIC_GAUGE_INFORMATION_BOUNDARY,
    TACTILE_METRIC_GAUGE_QUALITY_GATE,
    TACTILE_METRIC_GAUGE_TWO_VIEW_QUALITY_GATE,
    TWO_VIEW_CONTACT_CAMERA_POLICY,
    ContactCameraCandidate,
    apply_similarity,
    contact_camera_candidates,
    covariance_intersection_equal_weight,
    fit_robust_similarity,
    held_frame_gauge_quality,
    project_world_points_to_target,
    sample_point_map_bilinear,
    select_contact_camera_panel,
    unknown_correlation_covariance_union,
    validate_tactile_metric_gauge_lock,
)


def _camera(name: str, direction: tuple[float, float, float], margin: float):
    values = np.asarray(direction, dtype=np.float64)
    values /= np.linalg.norm(values)
    return ContactCameraCandidate(name, 1.0, margin, values)


def test_contact_panel_is_order_invariant_and_angularly_diverse() -> None:
    candidates = [
        _camera("front", (0.0, 0.0, 1.0), 100.0),
        _camera("back", (0.0, 0.0, -1.0), 90.0),
        _camera("right", (1.0, 0.0, 0.0), 80.0),
        _camera("near-front", (0.1, 0.0, 1.0), 99.0),
    ]
    expected = ("front", "back", "right")
    for values in (candidates, list(reversed(candidates))):
        selected = select_contact_camera_panel(
            values,
            panel_size=3,
            minimum_coverage=1.0,
            minimum_margin_px=64.0,
            minimum_angular_separation_deg=45.0,
        )
        assert tuple(item.camera for item in selected) == expected


def test_contact_projection_preserves_assignment_worst_case() -> None:
    points = np.asarray(
        [
            [[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]],
            [[-0.1, 0.1, 1.0], [0.1, 0.1, 1.0]],
        ]
    )
    intrinsics = {
        "camera": np.asarray([[100.0, 0.0, 99.5], [0.0, 100.0, 49.5], [0.0, 0.0, 1.0]])
    }
    extrinsics = {"camera": np.eye(4)}
    candidate = contact_camera_candidates(
        points,
        intrinsics_by_camera=intrinsics,
        world_from_camera_by_camera=extrinsics,
        source_shape=(100, 200),
        target_shape=(100, 200),
    )[0]
    assert candidate.minimum_assignment_coverage == 1.0
    assert candidate.minimum_margin_px > 0.0


def test_world_projection_matches_cover_resize_geometry() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [0.2, -0.1, 1.0]])
    intrinsics = np.asarray([[100.0, 0.0, 99.5], [0.0, 100.0, 49.5], [0.0, 0.0, 1.0]])

    xy, depth, visible = project_world_points_to_target(
        points,
        intrinsics=intrinsics,
        world_from_camera=np.eye(4),
        source_shape=(100, 200),
        target_shape=(50, 100),
    )

    assert np.allclose(xy, [[49.5, 24.5], [59.5, 19.5]])
    assert np.allclose(depth, 1.0)
    assert np.all(visible)


def test_bilinear_point_map_requires_all_four_valid_neighbors() -> None:
    point_map = np.zeros((1, 2, 2, 3), dtype=np.float64)
    point_map[0, 0, 0, 0] = 0.0
    point_map[0, 0, 1, 0] = 2.0
    point_map[0, 1, 0, 0] = 4.0
    point_map[0, 1, 1, 0] = 6.0
    valid = np.ones((1, 2, 2), dtype=bool)

    sampled, support = sample_point_map_bilinear(
        point_map,
        valid,
        np.asarray([125]),
        np.asarray([125]),
        np.asarray([[0.5, 0.5]]),
    )
    assert support.tolist() == [True]
    assert np.allclose(sampled, [[3.0, 0.0, 0.0]])

    valid[0, 1, 1] = False
    sampled, support = sample_point_map_bilinear(
        point_map,
        valid,
        np.asarray([125]),
        np.asarray([125]),
        np.asarray([[0.5, 0.5]]),
    )
    assert support.tolist() == [False]
    assert np.all(np.isnan(sampled))


def test_robust_similarity_recovers_transform_with_gross_outlier() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(30, 3))
    angle = 0.3
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = (1.7 * (rotation @ source.T)).T + np.asarray([0.2, -0.1, 0.3])
    target[-1] += np.asarray([2.0, -3.0, 1.0])
    transform = fit_robust_similarity(source, target, huber_delta_m=0.01)
    errors = np.linalg.norm(
        apply_similarity(transform, source[:-1]) - target[:-1], axis=1
    )
    assert np.median(errors) < 0.01


def test_held_frame_validation_rejects_frame_specific_bias() -> None:
    rng = np.random.default_rng(8)
    frames = np.repeat(np.arange(6), 6)
    source = rng.normal(scale=0.1, size=(len(frames), 3))
    target = source.copy()
    target[frames == 5] += np.asarray([0.03, 0.0, 0.0])
    quality = held_frame_gauge_quality(
        source,
        target,
        frames,
        huber_delta_m=0.005,
        covariance_floor_m=0.005,
        maximum_median_error_m=0.005,
        maximum_percentile_90_error_m=0.015,
    )
    assert not quality.admitted
    assert "tail-held-frame-error-too-large" in quality.reason_codes
    assert np.all(np.linalg.eigvalsh(quality.covariance_m2) >= 0.005**2)


def test_unknown_correlation_is_less_confident_than_independence() -> None:
    covariance = np.diag([1.0, 2.0, 3.0])
    values = np.stack((covariance, covariance))
    intersection = covariance_intersection_equal_weight(values)
    independent = np.linalg.inv(sum(np.linalg.inv(item) for item in values))
    assert np.allclose(intersection, covariance)
    assert np.all(np.linalg.eigvalsh(intersection - independent) >= -1e-12)


def test_two_view_covariance_union_never_adds_precision() -> None:
    values = np.asarray(
        [
            [[4.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 5.0, 1.0], [0.0, 1.0, 2.0]],
        ]
    ) * 1e-4
    union = unknown_correlation_covariance_union(
        values,
        shared_bias_floor_m=0.01,
    )
    for covariance in values:
        assert np.min(np.linalg.eigvalsh(union - covariance)) >= -1e-12
    independent = np.linalg.inv(sum(np.linalg.inv(item) for item in values))
    assert np.min(np.linalg.eigvalsh(union - independent)) >= -1e-12


def _lock() -> dict[str, object]:
    job_descriptor = {
        "object_id": "source",
        "episode": "episode_0000",
        "camera": "new",
        "source_video": {"path": "source.mp4", "sha256": "a" * 64, "bytes": 1},
        "source_frame_start": 0,
        "source_frame_stop_exclusive": 42,
        "windows": [],
        "seed_schedule": [],
        "output_relative_path": "source/episode_0000/new",
    }
    descriptor: dict[str, object] = {
        "schema": DEFORM360_TACTILE_METRIC_GAUGE_LOCK_SCHEMA,
        "schema_version": 1,
        "status": "locked-source-only-pre-supplement-provider",
        "implementation": {
            "revision": "b" * 40,
            "runner_source_sha256": "c" * 64,
        },
        "source_case": {
            "object_id": "source",
            "processing_episode_index": 0,
            "causal_frame_stop": 42,
        },
        "parents": {"parent": {"artifact_id": "d" * 64, "sha256": "e" * 64}},
        "camera_selection": {
            "policy": CONTACT_CAMERA_POLICY,
            "selected_cameras": ["old", "new", "third"],
            "reused_provider_cameras": ["old", "third"],
            "supplemental_provider_cameras": ["new"],
            "candidate_count": 4,
            "eligible_camera_count": 3,
            "candidate_inventory_sha256": "f" * 64,
            "selected_candidate_records": [
                {
                    "camera": camera,
                    "minimum_assignment_coverage": 1.0,
                    "minimum_margin_px": 80.0,
                    "view_direction": direction,
                }
                for camera, direction in (
                    ("old", [1.0, 0.0, 0.0]),
                    ("new", [0.0, 1.0, 0.0]),
                    ("third", [0.0, 0.0, 1.0]),
                )
            ],
        },
        "provider": {"quality_gate": TACTILE_METRIC_GAUGE_QUALITY_GATE},
        "supplemental_jobs": [{"job_id": content_id(job_descriptor), **job_descriptor}],
        "information_boundary": TACTILE_METRIC_GAUGE_INFORMATION_BOUNDARY,
    }
    return {"artifact_id": content_id(descriptor), **descriptor}


def test_metric_gauge_lock_binds_boundary_and_jobs() -> None:
    value = _lock()
    assert validate_tactile_metric_gauge_lock(value) == value["artifact_id"]
    changed = dict(value)
    changed["information_boundary"] = {
        **TACTILE_METRIC_GAUGE_INFORMATION_BOUNDARY,
        "calibration_scores_opened": True,
    }
    descriptor = dict(changed)
    descriptor.pop("artifact_id")
    changed["artifact_id"] = content_id(descriptor)
    try:
        validate_tactile_metric_gauge_lock(changed)
    except ValueError as error:
        assert "information boundary" in str(error)
    else:
        raise AssertionError("opened score boundary was accepted")


def test_metric_gauge_lock_rejects_selected_camera_below_margin() -> None:
    value = _lock()
    changed = dict(value)
    selection = dict(changed["camera_selection"])
    records = [dict(record) for record in selection["selected_candidate_records"]]
    records[0]["minimum_margin_px"] = 63.0
    selection["selected_candidate_records"] = records
    changed["camera_selection"] = selection
    descriptor = dict(changed)
    descriptor.pop("artifact_id")
    changed["artifact_id"] = content_id(descriptor)
    try:
        validate_tactile_metric_gauge_lock(changed)
    except ValueError as error:
        assert "margin" in str(error)
    else:
        raise AssertionError("under-margin selected camera was accepted")


def test_two_view_metric_gauge_lock_requires_no_precision_gain_policy() -> None:
    value = _lock()
    value["schema"] = DEFORM360_TACTILE_METRIC_GAUGE_TWO_VIEW_LOCK_SCHEMA
    selection = dict(value["camera_selection"])
    selection["policy"] = TWO_VIEW_CONTACT_CAMERA_POLICY
    selection["selected_cameras"] = ["old", "third"]
    selection["reused_provider_cameras"] = ["old", "third"]
    selection["supplemental_provider_cameras"] = []
    selection["selected_candidate_records"] = [
        selection["selected_candidate_records"][0],
        selection["selected_candidate_records"][2],
    ]
    value["camera_selection"] = selection
    value["provider"] = {
        "quality_gate": TACTILE_METRIC_GAUGE_TWO_VIEW_QUALITY_GATE
    }
    value["supplemental_jobs"] = []
    descriptor = dict(value)
    descriptor.pop("artifact_id")
    value["artifact_id"] = content_id(descriptor)
    assert validate_tactile_metric_gauge_lock(value) == value["artifact_id"]
