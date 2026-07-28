from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_prefix_support_screen import (
    PrefixAssociationSupportConfig,
    build_prefix_association_support_screen,
)


def _inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    physical: np.ndarray = np.zeros((58, 4, 3), dtype=np.float64)
    physical[:, :, 0] = np.asarray([-0.15, -0.05, 0.05, 0.15])
    physical[:, :, 2] = 1.0
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    depths: np.ndarray = np.ones(
        (3, 58, 64, 64),
        dtype=np.float32,
    )
    masks = np.zeros_like(depths, dtype=bool)
    pixel_x = np.asarray([17, 27, 37, 47])

    def mark(frame: int, camera: int, entity: int) -> None:
        x = int(pixel_x[entity])
        masks[camera, frame, 30:35, x - 2 : x + 3] = True

    for camera in range(3):
        for entity in range(4):
            mark(51, camera, entity)
        mark(57, camera, 0)
        mark(57, camera, 2)
    for camera in range(2):
        mark(57, camera, 1)
    return physical, intrinsics, poses, depths, masks


def test_screen_requires_support_at_both_prefix_endpoints() -> None:
    physical, intrinsics, poses, depths, masks = _inputs()
    result = build_prefix_association_support_screen(
        physical,
        intrinsics,
        poses,
        depths,
        masks,
        config=PrefixAssociationSupportConfig(search_radius_px=3),
    )

    np.testing.assert_array_equal(result.birth_support_count, [3, 3, 3, 3])
    np.testing.assert_array_equal(result.update_support_count, [3, 2, 3, 0])
    np.testing.assert_array_equal(result.eligible, [True, False, True, False])
    np.testing.assert_array_equal(result.eligible_entity_ids, [0, 2])
    assert not result.eligible_entity_ids.flags.writeable


def test_screen_is_candidate_order_invariant_and_prefix_endpoint_only() -> None:
    physical, intrinsics, poses, depths, masks = _inputs()
    config = PrefixAssociationSupportConfig(search_radius_px=3)
    first = build_prefix_association_support_screen(
        physical,
        intrinsics,
        poses,
        depths,
        masks,
        candidate_entity_ids=np.asarray([3, 1, 0, 2]),
        config=config,
    )
    changed = depths.copy()
    changed[:, :51] = 1000.0
    second = build_prefix_association_support_screen(
        physical,
        intrinsics,
        poses,
        changed,
        masks,
        candidate_entity_ids=np.arange(4),
        config=config,
    )

    np.testing.assert_array_equal(first.entity_ids, second.entity_ids)
    np.testing.assert_array_equal(first.eligible, second.eligible)
    assert first.artifact_sha256 == second.artifact_sha256
    boundary = first.descriptor()["information_boundary"]
    assert boundary["observed_frames_read"] == [51, 57]
    assert not boundary["association_probability_used_for_selection"]
    assert not boundary["state_innovation_used_for_reliability"]


def test_screen_does_not_lower_the_three_camera_requirement() -> None:
    physical, intrinsics, poses, depths, masks = _inputs()
    masks[2, 57] = masks[1, 57]
    masks[2, 57, :, :] = False
    result = build_prefix_association_support_screen(
        physical,
        intrinsics,
        poses,
        depths,
        masks,
        config=PrefixAssociationSupportConfig(search_radius_px=3),
    )

    assert np.all(result.update_support_count <= 2)
    assert not np.any(result.eligible)
