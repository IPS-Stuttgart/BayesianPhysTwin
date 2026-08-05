from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_tactile_metric_gauge import (
    ContactCameraCandidate,
    apply_similarity,
    contact_camera_candidates,
    covariance_intersection_equal_weight,
    fit_robust_similarity,
    held_frame_gauge_quality,
    select_contact_camera_panel,
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
    intrinsics = {"camera": np.asarray([[100.0, 0.0, 99.5], [0.0, 100.0, 49.5], [0.0, 0.0, 1.0]])}
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
    errors = np.linalg.norm(apply_similarity(transform, source[:-1]) - target[:-1], axis=1)
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
