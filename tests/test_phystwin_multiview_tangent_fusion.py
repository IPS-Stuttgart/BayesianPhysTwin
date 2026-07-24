import numpy as np

from bayesian_phystwin.phystwin_multiview_tangent_fusion import (
    fuse_source_normal_multiview_tangent,
    local_surface_tangent_projectors,
)


def _planar_points() -> np.ndarray:
    xy = np.array(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    return np.column_stack((xy, np.zeros(len(xy))))


def test_tangent_projector_recovers_planar_normal() -> None:
    projectors = local_surface_tangent_projectors(
        _planar_points(),
        neighbor_count=9,
    )

    tangent_x = np.einsum("nij,j->ni", projectors, [1.0, 0.0, 0.0])
    normal_z = np.einsum("nij,j->ni", projectors, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        tangent_x,
        np.tile([1.0, 0.0, 0.0], (len(projectors), 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(normal_z, 0.0, atol=1e-12)


def test_fusion_accepts_tangent_motion_and_preserves_source_normal() -> None:
    initial = _planar_points()
    source = np.repeat(initial[None], 2, axis=0)
    multiview = source.copy()
    source[1, :, 2] += 0.002
    multiview[1, :, 0] += 0.010
    multiview[1, :, 2] += 0.020
    valid = np.ones(source.shape[:2], dtype=bool)

    result = fuse_source_normal_multiview_tangent(
        source,
        valid,
        multiview,
        valid,
        initial,
        minimum_multiview_availability_fraction=0.4,
        neighbor_count=9,
    )

    np.testing.assert_allclose(
        result.points_world_m[1, :, 0],
        initial[:, 0] + 0.010,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.points_world_m[1, :, 2],
        0.002,
        atol=1e-12,
    )
    assert np.all(result.valid)
    assert np.all(result.fused_update)


def test_below_priority_threshold_is_exact_source_fallback() -> None:
    initial = _planar_points()
    source = np.repeat(initial[None], 3, axis=0)
    multiview = source + 0.1
    source_valid = np.ones(source.shape[:2], dtype=bool)
    multiview_valid = np.zeros(source.shape[:2], dtype=bool)
    multiview_valid[0] = True

    result = fuse_source_normal_multiview_tangent(
        source,
        source_valid,
        multiview,
        multiview_valid,
        initial,
        minimum_multiview_availability_fraction=0.5,
        neighbor_count=9,
    )

    assert np.array_equal(result.points_world_m, source)
    assert np.array_equal(result.valid, source_valid)
    assert not np.any(result.priority_identities)
    assert not np.any(result.fused_update)


def test_missing_channel_never_adds_source_support() -> None:
    initial = _planar_points()
    source = np.repeat(initial[None], 2, axis=0)
    multiview = source + np.array([0.1, 0.0, 0.0])
    source_valid = np.ones(source.shape[:2], dtype=bool)
    multiview_valid = np.ones(source.shape[:2], dtype=bool)
    multiview_valid[0, 0] = False
    source_valid[1, 1] = False

    result = fuse_source_normal_multiview_tangent(
        source,
        source_valid,
        multiview,
        multiview_valid,
        initial,
        minimum_multiview_availability_fraction=0.4,
        neighbor_count=9,
    )

    assert np.array_equal(result.valid, source_valid)
    np.testing.assert_array_equal(result.points_world_m[0, 0], source[0, 0])
    np.testing.assert_array_equal(result.points_world_m[1, 1], source[1, 1])
    assert not result.fused_update[0, 0]
    assert not result.fused_update[1, 1]
    assert result.fused_update[1, 0]
