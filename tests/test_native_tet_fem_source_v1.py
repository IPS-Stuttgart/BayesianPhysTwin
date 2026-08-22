from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
)
from bayesian_phystwin.native_tet_fem_source_v1 import (
    attached_targets_from_transform_v1,
    contact_transform_at_fraction_v1,
    interpolate_rotation_so3_v1,
    prepare_native_tet_source_geometry_v1,
    replay_deformation_determinants_v1,
    tetrahedral_nodal_masses_v1,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, RigidContactProjectionV1]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    attachments = np.asarray([0, 1, 2, 3], dtype=np.int64)
    angle = np.pi / 2.0
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotations = np.stack((np.eye(3), rotation))[:, None]
    translations = np.asarray([[[0.0, 0.0, 0.0]], [[0.2, -0.1, 0.3]]])
    projected = np.empty((2, len(attachments), 3), dtype=np.float64)
    for frame in range(2):
        projected[frame] = (
            points[attachments] @ rotations[frame, 0].T + translations[frame, 0]
        )
    contact = RigidContactProjectionV1(
        projected_targets_m=projected,
        rotations=rotations,
        translations_m=translations,
        patch_local_indices=(np.arange(4, dtype=np.int64),),
        patch_ranks=(3,),
    )
    return points, cells, attachments, contact


def test_geometry_and_lumped_mass_preserve_fixed_source_identity() -> None:
    points, cells, attachments, contact = _fixture()
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=cells,
        attachment_indices=attachments,
        contact=contact,
    )

    np.testing.assert_array_equal(geometry.points_m, points)
    np.testing.assert_array_equal(geometry.cells, cells)
    np.testing.assert_array_equal(geometry.patch_node_indices[0], attachments)
    masses, total_mass = tetrahedral_nodal_masses_v1(
        geometry,
        density_kg_m3=12.0,
    )
    expected_volume = sum(
        abs(np.linalg.det((points[cell[1:]] - points[cell[0]]).T)) / 6.0
        for cell in cells
    )
    assert total_mass == pytest.approx(12.0 * expected_volume)
    assert float(np.sum(masses)) == pytest.approx(total_mass)
    assert np.all(masses > 0.0)


def test_contact_continuation_stays_on_so3_and_projects_exact_targets() -> None:
    points, cells, attachments, contact = _fixture()
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=cells,
        attachment_indices=attachments,
        contact=contact,
    )

    midpoint = interpolate_rotation_so3_v1(
        contact.rotations[0, 0],
        contact.rotations[1, 0],
        0.5,
    )
    np.testing.assert_allclose(midpoint.T @ midpoint, np.eye(3), atol=1e-12)
    assert np.linalg.det(midpoint) == pytest.approx(1.0)
    rotations, translations = contact_transform_at_fraction_v1(
        contact,
        previous_frame=0,
        target_frame=1,
        fraction=1.0,
        driven=True,
    )
    targets = attached_targets_from_transform_v1(
        geometry,
        contact,
        rotations=rotations,
        translations_m=translations,
    )
    np.testing.assert_allclose(targets, contact.projected_targets_m[1], atol=1e-15)


def test_zero_action_transform_and_fixed_reference_determinants() -> None:
    points, cells, attachments, contact = _fixture()
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=cells,
        attachment_indices=attachments,
        contact=contact,
    )

    rotations, translations = contact_transform_at_fraction_v1(
        contact,
        previous_frame=0,
        target_frame=1,
        fraction=0.75,
        driven=False,
    )
    np.testing.assert_array_equal(rotations, np.eye(3)[None])
    np.testing.assert_array_equal(translations, np.zeros((1, 3)))
    trajectory = np.stack((points, 2.0 * points))
    determinants = replay_deformation_determinants_v1(geometry, trajectory)
    np.testing.assert_allclose(determinants[0], 1.0, atol=1e-15)
    np.testing.assert_allclose(determinants[1], 8.0, atol=1e-14)


def test_contact_patches_must_partition_each_attachment_once() -> None:
    points, cells, attachments, contact = _fixture()
    invalid = RigidContactProjectionV1(
        projected_targets_m=contact.projected_targets_m,
        rotations=contact.rotations,
        translations_m=contact.translations_m,
        patch_local_indices=(np.asarray([0, 1, 2, 2], dtype=np.int64),),
        patch_ranks=(3,),
    )
    with pytest.raises(ValueError, match="partition every attachment"):
        prepare_native_tet_source_geometry_v1(
            points_m=points,
            cells=cells,
            attachment_indices=attachments,
            contact=invalid,
        )
