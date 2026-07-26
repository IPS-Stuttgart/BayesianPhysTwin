from __future__ import annotations

import numpy as np

from bayesian_phystwin.static_scene_gauge import (
    StaticSceneGaugeConfig,
    apply_static_scene_gauge,
    estimate_static_scene_gauge,
    select_static_scene_queries,
)


def _synthetic_tracks(
    *,
    duplicate: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.arange(10.0, 111.0, 20.0), np.arange(10.0, 91.0, 20.0))
    background = np.column_stack((x.ravel(), y.ravel()))
    background = np.repeat(background, duplicate, axis=0)
    frames = np.arange(8.0)
    drift = np.empty((len(frames), len(background), 2), dtype=float)
    drift[:, :, 0] = (
        0.10 * frames[:, None]
        + 0.004 * frames[:, None] * background[None, :, 0]
    )
    drift[:, :, 1] = (
        -0.05 * frames[:, None]
        + 0.003 * frames[:, None] * background[None, :, 1]
    )
    tracks = background[None] + drift
    quality = np.ones(tracks.shape[:2], dtype=float)
    requested = np.array([[30.0, 30.0], [70.0, 50.0], [90.0, 70.0]])
    requested_drift = np.empty((len(frames), len(requested), 2), dtype=float)
    requested_drift[:, :, 0] = (
        0.10 * frames[:, None]
        + 0.004 * frames[:, None] * requested[None, :, 0]
    )
    requested_drift[:, :, 1] = (
        -0.05 * frames[:, None]
        + 0.003 * frames[:, None] * requested[None, :, 1]
    )
    return background, tracks, quality, requested, requested_drift


def _config(**overrides: object) -> StaticSceneGaugeConfig:
    values: dict[str, object] = {
        "correlation_cell_size_px": 10,
        "cross_validation_cell_size_px": 20,
        "neighbor_count": 12,
        "rbf_bandwidth_px": 40.0,
        "maximum_query_distance_px": 60.0,
        "minimum_effective_support": 2.0,
        "minimum_cross_validation_count": 20,
        "minimum_cross_validation_gain": 0.05,
    }
    values.update(overrides)
    return StaticSceneGaugeConfig(**values)


def test_static_scene_gauge_recovers_and_removes_local_drift() -> None:
    background, tracks, quality, requested, expected = _synthetic_tracks()
    estimate = estimate_static_scene_gauge(
        background,
        tracks,
        quality,
        requested,
        config=_config(),
    )

    assert estimate.accepted
    assert np.all(estimate.supported[1:])
    assert estimate.cross_validation_relative_gain is not None
    assert estimate.cross_validation_relative_gain > 0.75
    np.testing.assert_allclose(
        estimate.correction_px,
        expected,
        atol=0.35,
    )

    object_tracks = requested[None] + expected
    corrected = apply_static_scene_gauge(object_tracks, estimate)
    assert np.mean(np.linalg.norm(corrected - requested[None], axis=2)) < 0.2


def test_rejected_gauge_is_byte_identical() -> None:
    background, tracks, quality, requested, expected = _synthetic_tracks()
    estimate = estimate_static_scene_gauge(
        background,
        tracks,
        quality,
        requested,
        config=_config(minimum_cross_validation_gain=0.99),
    )
    object_tracks = (requested[None] + expected).astype(np.float32)
    before = object_tracks.tobytes()

    corrected = apply_static_scene_gauge(object_tracks, estimate)

    assert not estimate.accepted
    assert corrected.dtype == object_tracks.dtype
    assert corrected.tobytes() == before
    assert not np.any(estimate.supported)
    assert not np.any(estimate.correction_px)


def test_duplicate_correlated_background_tracks_add_no_confidence() -> None:
    original = _synthetic_tracks(duplicate=1)
    duplicated = _synthetic_tracks(duplicate=5)
    estimate = estimate_static_scene_gauge(
        original[0],
        original[1],
        original[2],
        original[3],
        config=_config(),
    )
    duplicate_estimate = estimate_static_scene_gauge(
        duplicated[0],
        duplicated[1],
        duplicated[2],
        duplicated[3],
        config=_config(),
    )

    assert estimate.background_cluster_count == duplicate_estimate.background_cluster_count
    np.testing.assert_allclose(
        estimate.correction_px,
        duplicate_estimate.correction_px,
    )
    np.testing.assert_allclose(
        estimate.variance_px2,
        duplicate_estimate.variance_px2,
    )
    np.testing.assert_allclose(
        estimate.effective_support,
        duplicate_estimate.effective_support,
    )


def test_far_object_queries_fall_back_locally() -> None:
    background, tracks, quality, _, _ = _synthetic_tracks()
    requested = np.array([[50.0, 50.0], [1000.0, 1000.0]])
    estimate = estimate_static_scene_gauge(
        background,
        tracks,
        quality,
        requested,
        config=_config(),
    )

    assert estimate.accepted
    assert np.all(estimate.supported[1:, 0])
    assert not np.any(estimate.supported[:, 1])
    assert not np.any(estimate.correction_px[:, 1])


def test_state_residual_cannot_enter_prior_gauge_estimation() -> None:
    background, tracks, quality, requested, _ = _synthetic_tracks()
    estimate = estimate_static_scene_gauge(
        background,
        tracks,
        quality,
        requested,
        config=_config(),
    )
    changed_object_state = np.full((len(tracks), len(requested), 3), 1000.0)

    repeated = estimate_static_scene_gauge(
        background,
        tracks,
        quality,
        requested,
        config=_config(),
    )

    assert changed_object_state.shape[2] == 3
    assert estimate.content_sha256 == repeated.content_sha256
    np.testing.assert_array_equal(estimate.correction_px, repeated.correction_px)


def test_static_query_selection_excludes_every_prefix_dynamic_mask() -> None:
    masks = np.zeros((3, 20, 24), dtype=bool)
    masks[1, 8:12, 8:12] = True
    depth = np.ones((20, 24), dtype=bool)
    config = StaticSceneGaugeConfig(
        query_stride_px=4,
        dynamic_margin_px=2,
        maximum_query_count=128,
    )

    queries = select_static_scene_queries(masks, depth, config=config)

    assert len(queries) > 0
    for x, y in queries.astype(int):
        assert not (6 <= y < 14 and 6 <= x < 14)
