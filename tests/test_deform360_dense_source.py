from __future__ import annotations

import numpy as np
import pytest

from causal4d_public.deform360_dense_source import (
    associate_controller_material_patch,
    fit_source_controller_patch,
    fit_phystwin_support_frame,
    select_sparse_controller_patch,
    support_dynamics_reverse_factor,
    unpack_sampled_mask,
)


def test_unpack_sampled_mask_recovers_original_bits() -> None:
    mask = np.array(
        [
            [True, False, True, False, True, False, True, False, True],
            [False, True, False, True, False, True, False, True, False],
        ],
        dtype=bool,
    )
    archive = {
        "cameras": np.array(["cam-a"]),
        "frame_indices": np.array([7], dtype=np.int32),
        "image_shape": np.array(mask.shape, dtype=np.int32),
        "packed_masks": np.packbits(mask[None, None], axis=-1),
    }

    decoded = unpack_sampled_mask(archive, "cam-a", 7)

    np.testing.assert_array_equal(decoded, mask)


def test_unpack_sampled_mask_rejects_unsealed_frame() -> None:
    archive = {
        "cameras": np.array(["cam-a"]),
        "frame_indices": np.array([7], dtype=np.int32),
        "image_shape": np.array([2, 8], dtype=np.int32),
        "packed_masks": np.zeros((1, 1, 2, 1), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="frame 8"):
        unpack_sampled_mask(archive, "cam-a", 8)


def test_sparse_controller_patch_uses_nearest_diverse_frame_zero_points() -> None:
    objects = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    controllers = np.asarray(
        [
            [0.001, 0.0, 0.0],
            [0.0015, 0.0, 0.0],
            [0.019, 0.0, 0.0],
            [0.10, 0.0, 0.0],
        ]
    )

    patch = select_sparse_controller_patch(
        objects,
        controllers,
        count=2,
        minimum_separation_m=0.004,
    )

    np.testing.assert_array_equal(patch.controller_indices, [0, 2])
    np.testing.assert_array_equal(patch.nearest_object_indices, [0, 1])
    np.testing.assert_allclose(patch.initial_distances_m, [0.001, 0.001])


def test_sparse_controller_patch_reads_only_supplied_frame_zero() -> None:
    objects = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    trajectory = np.asarray(
        [
            [[0.001, 0.0, 0.0], [0.019, 0.0, 0.0], [0.10, 0.0, 0.0]],
            [[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    first = select_sparse_controller_patch(objects, trajectory[0], count=1)
    trajectory[1] *= -7.0
    second = select_sparse_controller_patch(objects, trajectory[0], count=1)

    np.testing.assert_array_equal(first.controller_indices, second.controller_indices)


def test_source_controller_patch_prefers_source_prefix_comotion() -> None:
    objects = np.zeros((4, 2, 3))
    objects[:, 0, 0] = [0.0, 0.0, 0.0, 0.0]
    objects[:, 1, 0] = [0.01, 0.02, 0.03, 0.04]
    controllers = np.zeros((4, 2, 3))
    controllers[:, 0, 0] = [0.001, 0.011, 0.021, 0.031]
    controllers[:, 1, 0] = [0.009, 0.019, 0.029, 0.039]

    patch, diagnostics = fit_source_controller_patch(
        objects,
        controllers,
        count=1,
        maximum_initial_distance_m=0.02,
    )

    np.testing.assert_array_equal(patch.controller_indices, [1])
    np.testing.assert_array_equal(patch.nearest_object_indices, [1])
    assert diagnostics["fit_frame_range"] == [0, 4]
    assert diagnostics["held_out_motion_read"] is False


def test_source_controller_patch_does_not_read_unsupplied_holdout() -> None:
    objects = np.zeros((6, 2, 3))
    controllers = np.zeros((6, 2, 3))
    objects[:4, 1, 0] = [0.01, 0.02, 0.03, 0.04]
    controllers[:4, 1, 0] = [0.009, 0.019, 0.029, 0.039]
    first, _ = fit_source_controller_patch(
        objects[:4], controllers[:4], count=1
    )
    objects[4:] = 1000.0
    controllers[4:] = -1000.0
    second, _ = fit_source_controller_patch(
        objects[:4], controllers[:4], count=1
    )

    np.testing.assert_array_equal(first.controller_indices, second.controller_indices)


def test_controller_material_patch_reuses_index_and_refits_object_side() -> None:
    objects = np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    controllers = np.asarray(
        [[0.001, 0.0, 0.0], [0.099, 0.0, 0.0], [0.050, 0.0, 0.0]]
    )

    patch = associate_controller_material_patch(objects, controllers, np.asarray([1]))

    np.testing.assert_array_equal(patch.controller_indices, [1])
    np.testing.assert_array_equal(patch.nearest_object_indices, [1])
    np.testing.assert_allclose(patch.initial_distances_m, [0.001])


def test_controller_material_patch_rejects_repeated_indices() -> None:
    objects = np.zeros((1, 3))
    controllers = np.zeros((2, 3))

    with pytest.raises(ValueError, match="material indices"):
        associate_controller_material_patch(objects, controllers, np.asarray([1, 1]))


def test_support_frame_maps_positive_y_free_space_to_positive_z() -> None:
    points = np.asarray(
        [
            [0.0, 0.10, 0.0],
            [0.2, 0.11, 0.3],
            [0.4, 0.12, 0.1],
        ]
    )
    frame = fit_phystwin_support_frame(
        points,
        support_axis=1,
        free_space_sign=1,
        support_quantile=0.0,
        clearance_m=0.002,
    )

    transformed = frame.transform(points)

    np.testing.assert_allclose(transformed[:, 2], [0.002, 0.012, 0.022])
    np.testing.assert_allclose(
        np.linalg.norm(transformed[1] - transformed[0]),
        np.linalg.norm(points[1] - points[0]),
    )
    assert np.linalg.det(frame.rotation_world_to_sim) == pytest.approx(1.0)


def test_support_dynamics_preserves_frozen_ground_orientation() -> None:
    assert support_dynamics_reverse_factor(
        "official-ground", reverse_z=False
    ) == 1.0
    assert support_dynamics_reverse_factor(
        "official-ground", reverse_z=True
    ) == -1.0


def test_gravity_neutral_planar_support_has_no_gravity_factor() -> None:
    assert support_dynamics_reverse_factor(
        "gravity-neutral-planar", reverse_z=False
    ) == 0.0
    assert support_dynamics_reverse_factor(
        "gravity-neutral-planar", reverse_z=True
    ) == 0.0


def test_support_dynamics_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError, match="unknown Deform360 support dynamics"):
        support_dynamics_reverse_factor("magic", reverse_z=False)
