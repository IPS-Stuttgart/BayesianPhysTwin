"""Deterministic volumetric particles and query readout for Newton MPM.

The physical discretization is deliberately separate from the benchmark
identities.  MPM evolves a regular particle volume; benchmark points are read
out from weighted material displacements.  This avoids treating a sparse,
surface-heavy observation cloud as if it were a volumetric material sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, QhullError, cKDTree


@dataclass(frozen=True)
class MaterialQueryMapV2:
    """Fixed sparse displacement map from material particles to query points."""

    indices: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]
    maximum_distance_m: float


@dataclass(frozen=True)
class MaterialContactMapV2:
    """Controller displacement weights for the material contact particles."""

    material_indices: npt.NDArray[np.int64]
    controller_weights: npt.NDArray[np.float64]


def _points(
    value: object,
    *,
    name: str,
    minimum_count: int,
) -> npt.NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or len(raw) < minimum_count:
        raise ValueError(f"{name} must have shape (N>={minimum_count}, 3)")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(
        npt.NDArray[np.float64],
        np.ascontiguousarray(result, dtype=np.float64),
    )


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def regular_convex_hull_particles(
    source_points_m: object,
    *,
    spacing_m: float,
    maximum_particle_count: int = 100_000,
) -> npt.NDArray[np.float64]:
    """Fill a source point cloud's convex hull with a deterministic grid.

    The grid is anchored to the source bounding-box minimum, making the
    particleization translation equivariant.  A convex hull is intentionally a
    conservative first volume contract; future non-convex occupancy models must
    use a separately versioned protocol.
    """

    source = _points(source_points_m, name="source_points_m", minimum_count=4)
    spacing = _finite_positive(spacing_m, name="spacing_m")
    maximum = _positive_integer(
        maximum_particle_count,
        name="maximum_particle_count",
    )
    try:
        hull = ConvexHull(source)
    except QhullError as error:
        raise ValueError(
            "source_points_m must span a non-degenerate 3D hull"
        ) from error

    lower = np.min(source, axis=0)
    upper = np.max(source, axis=0)
    counts = np.floor((upper - lower) / spacing).astype(np.int64) + 1
    if np.any(counts < 2):
        raise ValueError("spacing_m is too large for the source hull")
    grid_count = int(np.prod(counts, dtype=np.int64))
    if grid_count > 20 * maximum:
        raise ValueError("candidate grid is too large for maximum_particle_count")

    axes = [lower[axis] + spacing * np.arange(counts[axis]) for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    normals = hull.equations[:, :3]
    offsets = hull.equations[:, 3]
    scale = max(1.0, float(np.max(np.abs(source), initial=0.0)))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    inside = np.all(grid @ normals.T + offsets[None] <= tolerance, axis=1)
    particles = np.ascontiguousarray(grid[inside], dtype=np.float64)
    if len(particles) < 4:
        raise ValueError(
            "volumetric particleization produced fewer than four particles"
        )
    if len(particles) > maximum:
        raise ValueError("volumetric particleization exceeds maximum_particle_count")
    return particles


def build_material_query_map(
    query_points_m: object,
    material_points_m: object,
    *,
    neighbour_count: int = 8,
    inverse_distance_power: float = 2.0,
    minimum_distance_m: float = 1.0e-9,
) -> MaterialQueryMapV2:
    """Build a deterministic inverse-distance material displacement readout."""

    queries = _points(query_points_m, name="query_points_m", minimum_count=1)
    material = _points(
        material_points_m,
        name="material_points_m",
        minimum_count=1,
    )
    neighbours = min(
        _positive_integer(neighbour_count, name="neighbour_count"),
        len(material),
    )
    power = _finite_positive(
        inverse_distance_power,
        name="inverse_distance_power",
    )
    minimum_distance = _finite_positive(
        minimum_distance_m,
        name="minimum_distance_m",
    )

    distances, indices = cKDTree(material).query(queries, k=neighbours, workers=1)
    distances = np.asarray(distances, dtype=np.float64).reshape(
        len(queries), neighbours
    )
    indices = np.asarray(indices, dtype=np.int64).reshape(len(queries), neighbours)
    order = np.argsort(indices, axis=1, kind="stable")
    indices = np.take_along_axis(indices, order, axis=1)
    distances = np.take_along_axis(distances, order, axis=1)

    weights = np.zeros_like(distances)
    exact = distances <= minimum_distance
    for row in range(len(queries)):
        exact_columns = np.flatnonzero(exact[row])
        if len(exact_columns):
            weights[row, exact_columns[0]] = 1.0
        else:
            raw = np.power(np.maximum(distances[row], minimum_distance), -power)
            weights[row] = raw / np.sum(raw)
    return MaterialQueryMapV2(
        indices=np.ascontiguousarray(indices),
        weights=np.ascontiguousarray(weights),
        maximum_distance_m=float(np.max(distances, initial=0.0)),
    )


def read_material_displacements(
    material_trajectory_m: object,
    material_rest_points_m: object,
    query_rest_points_m: object,
    query_map: MaterialQueryMapV2,
) -> npt.NDArray[np.float64]:
    """Read material displacement trajectories at fixed benchmark identities."""

    trajectory_raw = np.asarray(material_trajectory_m)
    if trajectory_raw.ndim != 3 or trajectory_raw.shape[2:] != (3,):
        raise ValueError("material_trajectory_m must have shape (T, P, 3)")
    if trajectory_raw.dtype.kind not in "iuf":
        raise ValueError("material_trajectory_m must contain real numeric values")
    trajectory = np.asarray(trajectory_raw, dtype=np.float64)
    material = _points(
        material_rest_points_m,
        name="material_rest_points_m",
        minimum_count=1,
    )
    queries = _points(
        query_rest_points_m,
        name="query_rest_points_m",
        minimum_count=1,
    )
    if trajectory.shape[1] != len(material):
        raise ValueError("material trajectory and rest particle counts differ")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("material_trajectory_m must contain only finite values")
    indices = np.asarray(query_map.indices)
    weights = np.asarray(query_map.weights, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[0] != len(queries):
        raise ValueError("query-map indices do not match query points")
    if weights.shape != indices.shape:
        raise ValueError("query-map weights do not match indices")
    if (
        indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(material))
    ):
        raise ValueError("query-map indices are outside the material particles")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("query-map weights must be finite and nonnegative")
    if not np.allclose(np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("query-map weights must sum to one")

    displacement = trajectory[:, indices, :] - material[indices][None]
    readout = queries[None] + np.sum(weights[None, ..., None] * displacement, axis=2)
    return np.ascontiguousarray(readout, dtype=np.float64)


def transfer_query_contacts_to_material(
    attached_query_indices: object,
    query_controller_weights: object,
    query_map: MaterialQueryMapV2,
    *,
    material_particle_count: int,
) -> MaterialContactMapV2:
    """Transfer a frozen query contact patch onto finite-mass particles."""

    material_count = _positive_integer(
        material_particle_count,
        name="material_particle_count",
    )
    attached = np.asarray(attached_query_indices)
    controller_raw = np.asarray(query_controller_weights)
    if attached.ndim != 1 or attached.dtype.kind not in "iu" or len(attached) < 1:
        raise ValueError("attached_query_indices must be a nonempty integer vector")
    if controller_raw.ndim != 2 or controller_raw.shape[0] != len(attached):
        raise ValueError("query_controller_weights must match attached queries")
    if controller_raw.dtype.kind not in "iuf":
        raise ValueError("query_controller_weights must contain numeric values")
    controller = np.asarray(controller_raw, dtype=np.float64)
    if not np.all(np.isfinite(controller)) or np.any(controller < 0.0):
        raise ValueError("query_controller_weights must be finite and nonnegative")
    if not np.allclose(np.sum(controller, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("query_controller_weights must sum to one")

    query_indices = np.asarray(query_map.indices)
    query_weights = np.asarray(query_map.weights, dtype=np.float64)
    if np.any(attached < 0) or np.any(attached >= len(query_indices)):
        raise ValueError("attached query index is outside the query map")
    if np.any(query_indices < 0) or np.any(query_indices >= material_count):
        raise ValueError("query map references an unavailable material particle")

    controller_count = controller.shape[1]
    accumulated = np.zeros((material_count, controller_count), dtype=np.float64)
    influence: npt.NDArray[np.float64] = np.zeros(
        material_count,
        dtype=np.float64,
    )
    for row, query_index in enumerate(attached.astype(np.int64, copy=False)):
        for column, material_index in enumerate(query_indices[query_index]):
            weight = float(query_weights[query_index, column])
            accumulated[material_index] += weight * controller[row]
            influence[material_index] += weight
    selected = np.flatnonzero(influence > 0.0)
    if len(selected) < 1:
        raise ValueError("contact transfer produced no material particles")
    accumulated[selected] /= influence[selected, None]
    return MaterialContactMapV2(
        material_indices=np.ascontiguousarray(selected, dtype=np.int64),
        controller_weights=np.ascontiguousarray(accumulated[selected]),
    )


def compliant_contact_projection(
    current_positions_m: object,
    current_velocities_m_s: object,
    target_positions_m: object,
    target_velocities_m_s: object,
    *,
    coupling: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reference finite-mass contact projection used by the CUDA kernel."""

    current_positions = _points(
        current_positions_m,
        name="current_positions_m",
        minimum_count=1,
    )
    current_velocities = _points(
        current_velocities_m_s,
        name="current_velocities_m_s",
        minimum_count=1,
    )
    target_positions = _points(
        target_positions_m,
        name="target_positions_m",
        minimum_count=1,
    )
    target_velocities = _points(
        target_velocities_m_s,
        name="target_velocities_m_s",
        minimum_count=1,
    )
    if not (
        current_positions.shape
        == current_velocities.shape
        == target_positions.shape
        == target_velocities.shape
    ):
        raise ValueError("contact projection arrays must have identical shapes")
    if isinstance(coupling, (bool, np.bool_)) or not isinstance(
        coupling, (int, float, np.integer, np.floating)
    ):
        raise ValueError("coupling must be finite and in [0, 1]")
    gain = float(coupling)
    if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("coupling must be finite and in [0, 1]")
    positions = current_positions + gain * (target_positions - current_positions)
    velocities = current_velocities + gain * (target_velocities - current_velocities)
    return np.ascontiguousarray(positions), np.ascontiguousarray(velocities)


__all__ = [
    "MaterialContactMapV2",
    "MaterialQueryMapV2",
    "build_material_query_map",
    "compliant_contact_projection",
    "read_material_displacements",
    "regular_convex_hull_particles",
    "transfer_query_contacts_to_material",
]
