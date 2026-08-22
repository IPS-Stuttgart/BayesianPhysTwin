from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.sofa_fem_canonical_source_v3 as canonical_module
from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
    rigid_contact_projection_v1,
    rigid_transform_v1,
)
from bayesian_phystwin.sofa_fem_canonical_source_v3 import (
    BACKEND_VARIANT,
    CANONICAL_ROUNDING_M,
    COORDINATE_POLICY,
    MINIMUM_RELATIVE_EIGENGAP,
    canonicalize_sofa_source_v3,
    run_sofa_fem_canonical_source_replay_v3,
)
from bayesian_phystwin.sofa_fem_kinematic_source_v2 import (
    SofaKinematicSourceReplayV2,
)


def _source() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    RigidContactProjectionV1,
]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.014, 0.0, 0.0],
            [0.0, 0.009, 0.0],
            [0.0, 0.0, 0.006],
            [0.002, 0.003, 0.001],
        ],
        dtype=np.float64,
    )
    cells = np.asarray([[0, 1, 2, 4], [0, 1, 4, 3]], dtype=np.int32)
    indices = np.arange(4, dtype=np.int64)
    local_rotation = rigid_transform_v1([2.0, -1.0, 3.0], 0.11)
    translation = np.asarray([0.001, -0.0004, 0.0002], dtype=np.float64)
    targets = np.stack(
        (
            points[indices],
            points[indices] @ local_rotation.T + translation,
        )
    )
    contact = rigid_contact_projection_v1(
        points,
        indices,
        targets,
        (np.arange(4, dtype=np.int64),),
    )
    return points, cells, indices, contact


def _transform_source(
    points: np.ndarray,
    indices: np.ndarray,
    contact: RigidContactProjectionV1,
) -> tuple[np.ndarray, RigidContactProjectionV1, np.ndarray, np.ndarray]:
    rotation = rigid_transform_v1([1.0, 2.0, 3.0], 0.37)
    translation = np.asarray([0.13, -0.08, 0.11], dtype=np.float64)
    transformed_points = points @ rotation.T + translation
    transformed_targets = contact.projected_targets_m @ rotation.T + translation
    transformed_contact = rigid_contact_projection_v1(
        transformed_points,
        indices,
        transformed_targets,
        contact.patch_local_indices,
    )
    return transformed_points, transformed_contact, rotation, translation


def test_canonical_gauge_is_exactly_invariant_to_registered_rigid_pose() -> None:
    points, cells, indices, contact = _source()
    transformed_points, transformed_contact, _, _ = _transform_source(
        points, indices, contact
    )

    left = canonicalize_sofa_source_v3(
        points_m=points,
        cells=cells,
        attachment_indices=indices,
        contact=contact,
    )
    right = canonicalize_sofa_source_v3(
        points_m=transformed_points,
        cells=cells,
        attachment_indices=indices,
        contact=transformed_contact,
    )

    assert BACKEND_VARIANT.endswith("canonical-gauge-keyed-dirichlet-v3")
    assert COORDINATE_POLICY.startswith("principal-axis-right-handed")
    assert CANONICAL_ROUNDING_M == 1.0e-11
    assert MINIMUM_RELATIVE_EIGENGAP == 1.0e-6
    assert left.gauge_sha256 == right.gauge_sha256
    np.testing.assert_array_equal(left.canonical_points_m, right.canonical_points_m)
    np.testing.assert_array_equal(
        left.canonical_contact.projected_targets_m,
        right.canonical_contact.projected_targets_m,
    )
    np.testing.assert_allclose(
        left.canonical_contact.rotations,
        right.canonical_contact.rotations,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert left.maximum_point_quantization_error_m < 1.0e-11
    assert left.maximum_target_quantization_error_m < 1.0e-11
    assert left.maximum_contact_reprojection_error_m < 5.0e-11


def test_canonical_gauge_rejects_rotationally_ambiguous_geometry() -> None:
    symmetric = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="stable principal-axis gauge"):
        canonical_module._oriented_principal_axes(
            symmetric,
            minimum_relative_eigengap=1.0e-6,
        )


def test_canonical_gauge_identity_binds_contact_patch_membership() -> None:
    first = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.014, 0.0, 0.0],
            [0.0, 0.009, 0.0],
            [0.0, 0.0, 0.006],
        ],
        dtype=np.float64,
    )
    second = first + np.asarray([0.019, 0.004, 0.002], dtype=np.float64)
    points = np.concatenate((first, second), axis=0)
    cells = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    indices = np.arange(8, dtype=np.int64)
    rotation = rigid_transform_v1([3.0, -1.0, 2.0], 0.07)
    translation = np.asarray([0.0003, -0.0002, 0.0001], dtype=np.float64)
    targets = np.stack((points, points @ rotation.T + translation))
    left_contact = rigid_contact_projection_v1(
        points,
        indices,
        targets,
        (np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64)),
    )
    right_contact = rigid_contact_projection_v1(
        points,
        indices,
        targets,
        (np.arange(4, 8, dtype=np.int64), np.arange(4, dtype=np.int64)),
    )

    left = canonicalize_sofa_source_v3(
        points_m=points,
        cells=cells,
        attachment_indices=indices,
        contact=left_contact,
    )
    right = canonicalize_sofa_source_v3(
        points_m=points,
        cells=cells,
        attachment_indices=indices,
        contact=right_contact,
    )

    np.testing.assert_array_equal(left.canonical_points_m, right.canonical_points_m)
    np.testing.assert_array_equal(
        left.canonical_contact.projected_targets_m,
        right.canonical_contact.projected_targets_m,
    )
    assert left.gauge_sha256 != right.gauge_sha256


def _fake_native_replay(**kwargs: Any) -> SofaKinematicSourceReplayV2:
    points = np.asarray(kwargs["points_m"], dtype=np.float64)
    indices = np.asarray(kwargs["attachment_indices"], dtype=np.int64)
    contact = kwargs["contact"]
    positions = np.repeat(points[None], 2, axis=0)
    positions[-1, indices] = contact.projected_targets_m[-1]
    positions[-1, 4, 0] += 0.001
    cells = np.asarray(kwargs["cells"])
    return SofaKinematicSourceReplayV2(
        positions_m=np.ascontiguousarray(positions),
        deformation_determinants=np.ones((2, len(cells)), dtype=np.float64),
        minimum_continuation_deformation_determinant=1.0,
        maximum_attachment_error_m=0.0,
        native_step_count=int(kwargs["interval_substeps"]),
        scene_sha256="1" * 64,
        schedule_sha256="2" * 64,
        material_vertex_count=len(points),
        tetrahedron_count=len(cells),
        attachment_count=len(indices),
        total_reference_mass_kg=1.0,
    )


def _run_wrapper(
    points: np.ndarray,
    cells: np.ndarray,
    indices: np.ndarray,
    contact: RigidContactProjectionV1,
) -> Any:
    return run_sofa_fem_canonical_source_replay_v3(
        native=SimpleNamespace(),
        points_m=points,
        cells=cells,
        attachment_indices=indices,
        contact=contact,
        driven=True,
        integrator_time_step_s=0.001,
        interval_substeps=4,
        young_modulus_pa=100_000.0,
        poisson_ratio=0.3,
        density_kg_m3=1000.0,
        rayleigh_stiffness=0.1,
        rayleigh_mass=0.1,
        hard_minimum_deformation_determinant=0.35,
    )


def test_world_replay_recovers_rigid_equivariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points, cells, indices, contact = _source()
    transformed_points, transformed_contact, rotation, translation = _transform_source(
        points, indices, contact
    )
    monkeypatch.setattr(
        canonical_module,
        "run_sofa_fem_kinematic_source_replay_v2",
        _fake_native_replay,
    )

    base = _run_wrapper(points, cells, indices, contact)
    transformed = _run_wrapper(
        transformed_points,
        cells,
        indices,
        transformed_contact,
    )

    assert base.gauge_sha256 == transformed.gauge_sha256
    np.testing.assert_allclose(
        (transformed.positions_m - translation) @ rotation,
        base.positions_m,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert base.maximum_attachment_error_m == 0.0
    assert base.maximum_world_attachment_approximation_error_m < 2.0e-11
