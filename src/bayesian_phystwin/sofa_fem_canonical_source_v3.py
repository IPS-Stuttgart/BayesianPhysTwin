"""Pose-canonical SOFA replay for numerically stable native FEM interchange."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from .jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
    rigid_contact_projection_v1,
)
from .native_tet_fem_source_v1 import prepare_native_tet_source_geometry_v1
from .sofa_fem_kinematic_source_v2 import run_sofa_fem_kinematic_source_replay_v2
from .sofa_fem_source_v1 import NativeSofaFemModulesV1

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

BACKEND_VARIANT = "sofa-stable-neo-hookean-canonical-gauge-keyed-dirichlet-v3"
COORDINATE_POLICY = "principal-axis-right-handed-ten-picometer-canonicalization-v1"
CANONICAL_ROUNDING_M = 1.0e-11
MINIMUM_RELATIVE_EIGENGAP = 1.0e-6
_SIGN_MOMENT_RELATIVE_FLOOR = 1.0e-12


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


class _HashLike(Protocol):
    def update(self, value: bytes) -> object: ...


def _array_identity(digest: _HashLike, array: npt.ArrayLike) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.tobytes(order="C"))


def _quantize_metric(
    values: npt.ArrayLike,
    *,
    quantum_m: float,
) -> tuple[FloatArray, IntArray]:
    array = np.asarray(values, dtype=np.float64)
    quantum = _finite(quantum_m, name="quantum_m", positive=True)
    scaled = array / quantum
    _require(
        np.all(np.isfinite(scaled)) and np.max(np.abs(scaled)) < np.iinfo(np.int64).max,
        "canonical metric coordinates exceed the integer lattice",
    )
    lattice = np.ascontiguousarray(np.rint(scaled), dtype=np.int64)
    return (
        np.ascontiguousarray(lattice.astype(np.float64) * quantum),
        lattice,
    )


def _oriented_principal_axes(
    centered_points_m: FloatArray,
    *,
    minimum_relative_eigengap: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    points = np.asarray(centered_points_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[0] >= 4
        and points.shape[1] == 3
        and np.all(np.isfinite(points)),
        "centered points must have shape (N,3)",
    )
    eigengap_floor = _finite(
        minimum_relative_eigengap,
        name="minimum_relative_eigengap",
        positive=True,
    )
    covariance = (points.T @ points) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    values = np.ascontiguousarray(eigenvalues[order], dtype=np.float64)
    axes = np.ascontiguousarray(eigenvectors[:, order], dtype=np.float64)
    _require(values[-1] > 0.0, "source geometry is not three-dimensional")
    relative_gaps = np.ascontiguousarray(
        (values[:-1] - values[1:]) / values[0], dtype=np.float64
    )
    _require(
        np.all(relative_gaps >= eigengap_floor),
        "source geometry does not define a stable principal-axis gauge",
    )

    for axis_index in (0, 1):
        projection = points @ axes[:, axis_index]
        moment = float(np.sum(projection**3))
        moment_scale = float(np.sum(np.abs(projection) ** 3))
        if abs(moment) <= _SIGN_MOMENT_RELATIVE_FLOOR * max(
            moment_scale, np.finfo(np.float64).tiny
        ):
            pivot = int(np.argmax(np.abs(projection)))
            sign_value = float(projection[pivot])
        else:
            sign_value = moment
        _require(sign_value != 0.0, "principal-axis sign is not identifiable")
        if sign_value < 0.0:
            axes[:, axis_index] *= -1.0
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    _require(
        np.allclose(axes.T @ axes, np.eye(3), atol=1.0e-12, rtol=0.0)
        and np.linalg.det(axes) > 0.0,
        "canonical frame left SO(3)",
    )
    return axes, values, relative_gaps


@dataclass(frozen=True, slots=True)
class SofaCanonicalGaugeV3:
    """Rigid-pose gauge and quantized source state passed to native SOFA."""

    center_m: FloatArray
    world_from_canonical: FloatArray
    eigenvalues_m2: FloatArray
    relative_eigengaps: FloatArray
    canonical_points_m: FloatArray
    canonical_contact: RigidContactProjectionV1
    maximum_point_quantization_error_m: float
    maximum_target_quantization_error_m: float
    maximum_contact_reprojection_error_m: float
    gauge_sha256: str


def canonicalize_sofa_source_v3(
    *,
    points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    contact: RigidContactProjectionV1,
    canonical_rounding_m: float = CANONICAL_ROUNDING_M,
    minimum_relative_eigengap: float = MINIMUM_RELATIVE_EIGENGAP,
) -> SofaCanonicalGaugeV3:
    """Map one registered source into a pose-invariant metric solver gauge."""

    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points_m,
        cells=cells,
        attachment_indices=attachment_indices,
        contact=contact,
    )
    center = np.ascontiguousarray(np.mean(geometry.points_m, axis=0))
    centered = np.ascontiguousarray(geometry.points_m - center)
    axes, eigenvalues, relative_gaps = _oriented_principal_axes(
        centered,
        minimum_relative_eigengap=minimum_relative_eigengap,
    )
    raw_points = np.ascontiguousarray(centered @ axes)
    raw_targets = np.ascontiguousarray((contact.projected_targets_m - center) @ axes)
    canonical_points, point_lattice = _quantize_metric(
        raw_points,
        quantum_m=canonical_rounding_m,
    )
    canonical_targets, target_lattice = _quantize_metric(
        raw_targets,
        quantum_m=canonical_rounding_m,
    )
    canonical_contact = rigid_contact_projection_v1(
        canonical_points,
        geometry.attachment_indices,
        canonical_targets,
        contact.patch_local_indices,
    )
    maximum_point_error = float(
        np.max(np.linalg.norm(canonical_points - raw_points, axis=1))
    )
    maximum_target_error = float(
        np.max(np.linalg.norm(canonical_targets - raw_targets, axis=2))
    )
    maximum_reprojection_error = float(
        np.max(
            np.linalg.norm(
                canonical_contact.projected_targets_m - canonical_targets,
                axis=2,
            )
        )
    )
    quantum = _finite(
        canonical_rounding_m,
        name="canonical_rounding_m",
        positive=True,
    )
    _require(
        maximum_point_error <= np.sqrt(3.0) * quantum
        and maximum_target_error <= np.sqrt(3.0) * quantum
        and maximum_reprojection_error <= 4.0 * np.sqrt(3.0) * quantum,
        "canonicalization exceeded its picometer approximation boundary",
    )
    digest = hashlib.sha256()
    for array in (
        point_lattice,
        target_lattice,
        geometry.cells,
        geometry.attachment_indices,
        np.asarray(
            [len(patch) for patch in contact.patch_local_indices],
            dtype=np.int64,
        ),
    ):
        _array_identity(digest, array)
    for patch in contact.patch_local_indices:
        _array_identity(digest, patch)
    digest.update(COORDINATE_POLICY.encode())
    digest.update(f"{quantum:.17g}".encode())
    return SofaCanonicalGaugeV3(
        center_m=center,
        world_from_canonical=axes,
        eigenvalues_m2=eigenvalues,
        relative_eigengaps=relative_gaps,
        canonical_points_m=canonical_points,
        canonical_contact=canonical_contact,
        maximum_point_quantization_error_m=maximum_point_error,
        maximum_target_quantization_error_m=maximum_target_error,
        maximum_contact_reprojection_error_m=maximum_reprojection_error,
        gauge_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class SofaCanonicalSourceReplayV3:
    """World-frame result from one pose-canonical native SOFA replay."""

    positions_m: FloatArray
    deformation_determinants: FloatArray
    minimum_continuation_deformation_determinant: float
    maximum_attachment_error_m: float
    maximum_world_attachment_approximation_error_m: float
    native_step_count: int
    scene_sha256: str
    schedule_sha256: str
    gauge_sha256: str
    material_vertex_count: int
    tetrahedron_count: int
    attachment_count: int
    total_reference_mass_kg: float
    maximum_point_quantization_error_m: float
    maximum_target_quantization_error_m: float
    maximum_contact_reprojection_error_m: float


def run_sofa_fem_canonical_source_replay_v3(
    *,
    native: NativeSofaFemModulesV1,
    points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    contact: RigidContactProjectionV1,
    driven: bool,
    integrator_time_step_s: float,
    interval_substeps: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    density_kg_m3: float,
    rayleigh_stiffness: float,
    rayleigh_mass: float,
    hard_minimum_deformation_determinant: float,
    canonical_rounding_m: float = CANONICAL_ROUNDING_M,
    minimum_relative_eigengap: float = MINIMUM_RELATIVE_EIGENGAP,
) -> SofaCanonicalSourceReplayV3:  # pragma: no cover - exact native runtime
    """Run native SOFA in the deterministic source gauge and return world state."""

    gauge = canonicalize_sofa_source_v3(
        points_m=points_m,
        cells=cells,
        attachment_indices=attachment_indices,
        contact=contact,
        canonical_rounding_m=canonical_rounding_m,
        minimum_relative_eigengap=minimum_relative_eigengap,
    )
    canonical = run_sofa_fem_kinematic_source_replay_v2(
        native=native,
        points_m=gauge.canonical_points_m,
        cells=cells,
        attachment_indices=attachment_indices,
        contact=gauge.canonical_contact,
        driven=driven,
        integrator_time_step_s=integrator_time_step_s,
        interval_substeps=interval_substeps,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
        density_kg_m3=density_kg_m3,
        rayleigh_stiffness=rayleigh_stiffness,
        rayleigh_mass=rayleigh_mass,
        hard_minimum_deformation_determinant=hard_minimum_deformation_determinant,
    )
    positions = np.ascontiguousarray(
        canonical.positions_m @ gauge.world_from_canonical.T + gauge.center_m
    )
    indices = np.asarray(attachment_indices, dtype=np.int64)
    expected_targets = (
        contact.projected_targets_m
        if driven
        else np.repeat(contact.projected_targets_m[:1], len(positions), axis=0)
    )
    endpoint_attachment_error = float(
        np.max(np.linalg.norm(positions[:, indices] - expected_targets, axis=2))
    )
    scene_digest = hashlib.sha256()
    scene_digest.update(canonical.scene_sha256.encode())
    scene_digest.update(gauge.gauge_sha256.encode())
    scene_digest.update(BACKEND_VARIANT.encode())
    schedule_digest = hashlib.sha256()
    schedule_digest.update(canonical.schedule_sha256.encode())
    schedule_digest.update(gauge.gauge_sha256.encode())
    return SofaCanonicalSourceReplayV3(
        positions_m=positions,
        deformation_determinants=np.ascontiguousarray(
            canonical.deformation_determinants
        ),
        minimum_continuation_deformation_determinant=(
            canonical.minimum_continuation_deformation_determinant
        ),
        maximum_attachment_error_m=canonical.maximum_attachment_error_m,
        maximum_world_attachment_approximation_error_m=(endpoint_attachment_error),
        native_step_count=canonical.native_step_count,
        scene_sha256=scene_digest.hexdigest(),
        schedule_sha256=schedule_digest.hexdigest(),
        gauge_sha256=gauge.gauge_sha256,
        material_vertex_count=canonical.material_vertex_count,
        tetrahedron_count=canonical.tetrahedron_count,
        attachment_count=canonical.attachment_count,
        total_reference_mass_kg=canonical.total_reference_mass_kg,
        maximum_point_quantization_error_m=(gauge.maximum_point_quantization_error_m),
        maximum_target_quantization_error_m=(gauge.maximum_target_quantization_error_m),
        maximum_contact_reprojection_error_m=(
            gauge.maximum_contact_reprojection_error_m
        ),
    )


__all__ = [
    "BACKEND_VARIANT",
    "CANONICAL_ROUNDING_M",
    "COORDINATE_POLICY",
    "MINIMUM_RELATIVE_EIGENGAP",
    "SofaCanonicalGaugeV3",
    "SofaCanonicalSourceReplayV3",
    "canonicalize_sofa_source_v3",
    "run_sofa_fem_canonical_source_replay_v3",
]
