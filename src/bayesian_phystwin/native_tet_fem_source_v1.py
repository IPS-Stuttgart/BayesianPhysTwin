"""Shared source-geometry mechanics for native tetrahedral FEM backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
    deformation_determinants_v1,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


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


@dataclass(frozen=True, slots=True)
class NativeTetSourceGeometryV1:
    """Validated fixed-identity volume mesh and rigid attachment partition."""

    points_m: FloatArray
    cells: IntArray
    attachment_indices: IntArray
    patch_node_indices: tuple[IntArray, ...]
    patch_centers_m: FloatArray


def prepare_native_tet_source_geometry_v1(
    *,
    points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    contact: RigidContactProjectionV1,
) -> NativeTetSourceGeometryV1:
    """Validate one source mesh and its complete non-overlapping patch roster."""

    points = np.ascontiguousarray(np.asarray(points_m, dtype=np.float64))
    tetrahedra = np.ascontiguousarray(np.asarray(cells, dtype=np.int64))
    indices = np.ascontiguousarray(np.asarray(attachment_indices, dtype=np.int64))
    _require(points.ndim == 2 and points.shape[1] == 3, "points changed")
    _require(
        tetrahedra.ndim == 2 and tetrahedra.shape[1] == 4 and len(tetrahedra) > 0,
        "TET4 cells changed",
    )
    _require(indices.ndim == 1 and len(indices) > 0, "attachments changed")
    _require(
        np.all(np.isfinite(points))
        and np.all(tetrahedra >= 0)
        and np.all(tetrahedra < len(points)),
        "source mesh is invalid",
    )
    _require(
        len(np.unique(indices)) == len(indices)
        and np.all(indices >= 0)
        and np.all(indices < len(points)),
        "attachment identities changed",
    )
    _require(
        all(len(np.unique(cell)) == 4 for cell in tetrahedra),
        "tetrahedra contain repeated nodes",
    )
    _require(
        len(np.unique(np.sort(tetrahedra, axis=1), axis=0)) == len(tetrahedra),
        "source mesh contains duplicate tetrahedra",
    )
    _require(
        len(np.unique(tetrahedra)) == len(points),
        "source mesh contains unused nodes",
    )
    reference = points[tetrahedra]
    matrices = np.stack(
        (
            reference[:, 1] - reference[:, 0],
            reference[:, 2] - reference[:, 0],
            reference[:, 3] - reference[:, 0],
        ),
        axis=2,
    )
    determinants = np.linalg.det(matrices)
    _require(
        np.all(np.isfinite(determinants)) and np.all(determinants > 0.0),
        "source tetrahedra must have positive orientation",
    )
    _require(
        contact.rotations.ndim == 4
        and contact.rotations.shape[2:] == (3, 3)
        and contact.translations_m.shape == contact.rotations.shape[:2] + (3,)
        and len(contact.patch_local_indices) == contact.rotations.shape[1],
        "contact transform roster changed",
    )
    _require(
        contact.rotations.shape[0] >= 2
        and contact.projected_targets_m.shape == (len(contact.rotations), len(indices), 3)
        and len(contact.patch_ranks) == contact.rotations.shape[1]
        and all(rank == 3 for rank in contact.patch_ranks),
        "contact trajectory or patch rank changed",
    )
    _require(
        np.all(np.isfinite(contact.rotations))
        and np.all(np.isfinite(contact.translations_m))
        and np.all(np.isfinite(contact.projected_targets_m)),
        "contact trajectory is not finite",
    )
    identities = np.einsum(
        "...ji,...jk->...ik",
        contact.rotations,
        contact.rotations,
        optimize=True,
    )
    _require(
        np.allclose(identities, np.eye(3), atol=1.0e-10, rtol=0.0)
        and np.all(np.linalg.det(contact.rotations) > 0.0),
        "contact rotations left SO(3)",
    )
    patch_nodes: list[IntArray] = []
    covered: list[int] = []
    centers: list[FloatArray] = []
    for local in contact.patch_local_indices:
        local_indices = np.ascontiguousarray(np.asarray(local, dtype=np.int64))
        _require(
            local_indices.ndim == 1
            and len(local_indices) > 0
            and np.all(local_indices >= 0)
            and np.all(local_indices < len(indices)),
            "contact patch indices changed",
        )
        nodes = np.ascontiguousarray(indices[local_indices])
        patch_nodes.append(nodes)
        covered.extend(int(value) for value in local_indices)
        centers.append(np.ascontiguousarray(np.mean(points[nodes], axis=0)))
    _require(
        sorted(covered) == list(range(len(indices))),
        "contact patches do not partition every attachment",
    )
    expected_targets = np.empty_like(contact.projected_targets_m)
    for patch_index, local in enumerate(contact.patch_local_indices):
        local_indices = np.asarray(local, dtype=np.int64)
        source = points[indices[local_indices]]
        expected_targets[:, local_indices] = (
            np.einsum(
                "tij,pj->tpi",
                contact.rotations[:, patch_index],
                source,
                optimize=True,
            )
            + contact.translations_m[:, patch_index, None]
        )
    _require(
        np.allclose(
            expected_targets,
            contact.projected_targets_m,
            atol=1.0e-10,
            rtol=0.0,
        ),
        "contact projected targets changed",
    )
    _require(
        np.allclose(
            contact.projected_targets_m[0],
            points[indices],
            atol=1.0e-12,
            rtol=0.0,
        ),
        "frame-zero contact registration changed",
    )
    return NativeTetSourceGeometryV1(
        points_m=points,
        cells=tetrahedra,
        attachment_indices=indices,
        patch_node_indices=tuple(patch_nodes),
        patch_centers_m=np.ascontiguousarray(np.stack(centers)),
    )


def interpolate_rotation_so3_v1(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    fraction: float,
) -> FloatArray:
    """Linearly blend two rotations and project deterministically onto SO(3)."""

    start = np.asarray(left, dtype=np.float64)
    end = np.asarray(right, dtype=np.float64)
    alpha = _finite(fraction, name="fraction")
    _require(start.shape == (3, 3) and end.shape == (3, 3), "rotations changed")
    _require(0.0 <= alpha <= 1.0, "fraction must lie in [0,1]")
    if alpha == 0.0:
        return np.ascontiguousarray(start)
    if alpha == 1.0:
        return np.ascontiguousarray(end)
    left_singular, _, right_singular = np.linalg.svd(
        (1.0 - alpha) * start + alpha * end,
        full_matrices=False,
    )
    rotation = left_singular @ right_singular
    if np.linalg.det(rotation) < 0.0:
        left_singular[:, -1] *= -1.0
        rotation = left_singular @ right_singular
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12, rtol=0.0)
        and np.linalg.det(rotation) > 0.0,
        "interpolated contact rotation left SO(3)",
    )
    return np.ascontiguousarray(rotation)


def contact_transform_at_fraction_v1(
    contact: RigidContactProjectionV1,
    *,
    previous_frame: int,
    target_frame: int,
    fraction: float,
    driven: bool,
) -> tuple[FloatArray, FloatArray]:
    """Return one registered rigid-patch continuation transform."""

    patch_count = len(contact.patch_local_indices)
    if driven:
        left_rotations = np.asarray(contact.rotations[previous_frame], dtype=np.float64)
        right_rotations = np.asarray(contact.rotations[target_frame], dtype=np.float64)
        left_translations = np.asarray(
            contact.translations_m[previous_frame], dtype=np.float64
        )
        right_translations = np.asarray(
            contact.translations_m[target_frame], dtype=np.float64
        )
    else:
        left_rotations = right_rotations = np.repeat(
            np.eye(3, dtype=np.float64)[None], patch_count, axis=0
        )
        left_translations = right_translations = np.zeros(
            (patch_count, 3), dtype=np.float64
        )
    rotations = np.stack(
        [
            interpolate_rotation_so3_v1(left, right, fraction)
            for left, right in zip(
                left_rotations,
                right_rotations,
                strict=True,
            )
        ]
    )
    translations = (
        (1.0 - fraction) * left_translations + fraction * right_translations
    )
    return (
        np.ascontiguousarray(rotations),
        np.ascontiguousarray(translations),
    )


def attached_targets_from_transform_v1(
    geometry: NativeTetSourceGeometryV1,
    contact: RigidContactProjectionV1,
    *,
    rotations: npt.ArrayLike,
    translations_m: npt.ArrayLike,
) -> FloatArray:
    """Project patch transforms into the original attachment-index order."""

    rotation_array = np.asarray(rotations, dtype=np.float64)
    translation_array = np.asarray(translations_m, dtype=np.float64)
    _require(
        rotation_array.shape == (len(geometry.patch_node_indices), 3, 3)
        and translation_array.shape == (len(geometry.patch_node_indices), 3),
        "contact transform shape changed",
    )
    targets: FloatArray = np.empty(
        (len(geometry.attachment_indices), 3),
        dtype=np.float64,
    )
    for patch_index, local in enumerate(contact.patch_local_indices):
        local_indices = np.asarray(local, dtype=np.int64)
        source = geometry.points_m[geometry.attachment_indices[local_indices]]
        targets[local_indices] = (
            source @ rotation_array[patch_index].T + translation_array[patch_index]
        )
    return np.ascontiguousarray(targets)


def tetrahedral_nodal_masses_v1(
    geometry: NativeTetSourceGeometryV1,
    *,
    density_kg_m3: float,
) -> tuple[FloatArray, float]:
    """Lump exact reference tetrahedral volume mass equally to its four nodes."""

    density = _finite(density_kg_m3, name="density_kg_m3", positive=True)
    cells = geometry.points_m[geometry.cells]
    matrices = np.stack(
        (
            cells[:, 1] - cells[:, 0],
            cells[:, 2] - cells[:, 0],
            cells[:, 3] - cells[:, 0],
        ),
        axis=2,
    )
    volumes = np.abs(np.linalg.det(matrices)) / 6.0
    masses: FloatArray = np.zeros(len(geometry.points_m), dtype=np.float64)
    np.add.at(
        masses,
        geometry.cells.reshape(-1),
        np.repeat(density * volumes / 4.0, 4),
    )
    _require(np.all(masses > 0.0), "source mesh contains massless nodes")
    return np.ascontiguousarray(masses), float(np.sum(density * volumes))


def replay_deformation_determinants_v1(
    geometry: NativeTetSourceGeometryV1,
    positions_m: npt.ArrayLike,
) -> FloatArray:
    """Compute fixed-reference TET4 determinants for a native replay."""

    return cast(
        FloatArray,
        deformation_determinants_v1(
            geometry.points_m,
            geometry.cells,
            positions_m,
        ),
    )


__all__ = [
    "NativeTetSourceGeometryV1",
    "attached_targets_from_transform_v1",
    "contact_transform_at_fraction_v1",
    "interpolate_rotation_so3_v1",
    "prepare_native_tet_source_geometry_v1",
    "replay_deformation_determinants_v1",
    "tetrahedral_nodal_masses_v1",
]
