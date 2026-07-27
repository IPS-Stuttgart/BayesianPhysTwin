from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_dynamic_query import projection_matrices
from bayesian_phystwin.tapnextpp_birth_association import (
    BirthAssociationConfig,
    propose_birth_query_pixels,
)


def _scene() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    camera_count = 3
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    poses[0, :3, 3] = [-0.05, 0.0, 0.0]
    poses[1, :3, 3] = [0.0, 0.0, 0.0]
    poses[2, :3, 3] = [0.05, 0.0, 0.0]
    depth = np.ones((camera_count, 64, 64), dtype=np.float64)
    masks = np.zeros_like(depth, dtype=bool)
    masks[:, 20:45, 20:45] = True
    points = np.asarray([[0.0, 0.0, 1.0], [0.05, 0.02, 1.0]])
    return points, projection_matrices(intrinsics, poses), poses, depth, masks


def test_birth_association_has_no_reliability_channel() -> None:
    points, projections, poses, depth, masks = _scene()
    result = propose_birth_query_pixels(
        points,
        projections,
        poses,
        depth,
        masks,
        config=BirthAssociationConfig(search_radius_px=4),
    )
    assert "prior_reliability" not in result
    assert "observation_reliability" not in result
    assert result["query_points_xy"].shape == (3, 2, 2)
    assert np.all(result["valid"])
    assert np.all(
        (result["association_probability"] >= 0.0)
        & (result["association_probability"] <= 1.0)
    )


def test_state_geometry_changes_association_but_cannot_create_reliability() -> None:
    points, projections, poses, depth, masks = _scene()
    depth[:, :, 32:] = 1.04
    first = propose_birth_query_pixels(
        points,
        projections,
        poses,
        depth,
        masks,
        config=BirthAssociationConfig(search_radius_px=6),
    )
    changed = points.copy()
    changed[:, 2] += 0.04
    second = propose_birth_query_pixels(
        changed,
        projections,
        poses,
        depth,
        masks,
        config=BirthAssociationConfig(search_radius_px=6),
    )
    assert not np.array_equal(
        first["association_probability"],
        second["association_probability"],
    )
    assert set(first) == set(second)
    assert "prior_reliability" not in first


def test_birth_association_rejects_pixels_outside_object_mask() -> None:
    points, projections, poses, depth, masks = _scene()
    masks[:] = False
    result = propose_birth_query_pixels(points, projections, poses, depth, masks)
    assert not np.any(result["valid"])
    assert np.all(result["association_probability"] == 0.0)
    assert np.all(result["association_entropy"] == 1.0)
