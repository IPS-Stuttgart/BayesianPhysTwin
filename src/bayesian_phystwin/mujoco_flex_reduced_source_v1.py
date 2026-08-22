"""Reduced-coordinate MuJoCo Flex replay for registered source meshes.

The original tetrahedral vertices and elements remain the native Flex geometry,
but their motion is interpolated from a regular trilinear background grid.  A
rigid contact patch owns every background node in the interpolation support of
its attached vertices.  Disjoint support is required, so each patch can impose
its registered SE(3) motion exactly while the remaining active grid nodes evolve
under MuJoCo's native Saint-Venant--Kirchhoff elasticity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, TypeAlias
from xml.etree import ElementTree

import numpy as np
import numpy.typing as npt

from .jax_fem_source_qualification_v1 import RigidContactProjectionV1
from .mujoco_flex_source_v1 import NativeMujocoFlexModulesV1
from .native_tet_fem_source_v1 import (
    NativeTetSourceGeometryV1,
    attached_targets_from_transform_v1,
    contact_transform_at_fraction_v1,
    prepare_native_tet_source_geometry_v1,
    replay_deformation_determinants_v1,
    tetrahedral_nodal_masses_v1,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

BACKEND_VARIANT = "mujoco-volumetric-trilinear-flex-v1"
CONSTITUTIVE_MODEL = "MuJoCo-native-Saint-Venant-Kirchhoff-flex-elasticity"
ATTACHMENT_MODEL = "disjoint-rigid-patch-trilinear-node-Dirichlet-v1"
INTERPOLATION_MODEL = "axis-aligned-multicell-trilinear-v1"
GRID_FRAME_POLICY = "identity-world-frame-with-empty-cell-sentinel-v1"
MASS_MODEL = "reference-mass-conserving-occupied-cell-lumping-v1"
EMPTY_CELL_STIFFNESS_SENTINEL = float(np.nextafter(0.0, 1.0))


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


def _numbers(values: npt.ArrayLike) -> str:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return " ".join(str(int(value)) for value in array.reshape(-1))
    return " ".join(f"{float(value):.17g}" for value in array.reshape(-1))


def _warning_count(data: Any) -> int:
    return sum(int(warning.number) for warning in data.warning)


@dataclass(frozen=True, slots=True)
class MujocoTrilinearGridV1:
    """Fixed interpolation, ownership, and mass map for one source mesh."""

    cellcount: tuple[int, int, int]
    grid_from_world_rotation: FloatArray
    grid_lower_m: FloatArray
    grid_upper_m: FloatArray
    grid_spacing_m: FloatArray
    cell_zero_tetrahedron_count: int
    node_positions_backend_m: FloatArray
    node_positions_world_m: FloatArray
    vertex_node_indices: IntArray
    vertex_node_weights: FloatArray
    occupied_cell_indices: IntArray
    active_node_indices: IntArray
    patch_node_indices: tuple[IntArray, ...]
    free_node_indices: IntArray
    inactive_node_indices: IntArray
    node_patch_index: IntArray
    node_masses_kg: FloatArray
    total_reference_mass_kg: float


def _cell_zero_tetrahedron_count_v1(
    points_grid_m: FloatArray,
    cells: IntArray,
    counts: IntArray,
) -> int:
    lower = np.min(points_grid_m, axis=0)
    upper = np.max(points_grid_m, axis=0)
    extent = upper - lower
    _require(np.all(extent > 0.0), "source bounds must span all three axes")
    tetrahedron_minima = np.min(points_grid_m[cells], axis=1)
    minimum_cells = np.floor((tetrahedron_minima - lower) / extent * counts).astype(
        np.int64
    )
    minimum_cells = np.clip(minimum_cells, 0, counts - 1)
    return int(np.sum(np.all(minimum_cells == 0, axis=1)))


def _occupied_trilinear_cells_v1(
    points_grid_m: FloatArray,
    cells: IntArray,
    counts: IntArray,
) -> IntArray:
    lower = np.min(points_grid_m, axis=0)
    upper = np.max(points_grid_m, axis=0)
    extent = upper - lower
    _require(np.all(extent > 0.0), "source bounds must span all three axes")
    tetrahedra = points_grid_m[cells]
    low = ((np.min(tetrahedra, axis=1) - lower) / extent * counts).astype(np.int64)
    high = ((np.max(tetrahedra, axis=1) - lower) / extent * counts).astype(np.int64)
    low = np.clip(low, 0, counts - 1)
    high = np.clip(high, 0, counts - 1)
    occupied: set[int] = set()
    for cell_low, cell_high in zip(low, high, strict=True):
        for i in range(int(cell_low[0]), int(cell_high[0]) + 1):
            for j in range(int(cell_low[1]), int(cell_high[1]) + 1):
                for k in range(int(cell_low[2]), int(cell_high[2]) + 1):
                    occupied.add((i * int(counts[1]) + j) * int(counts[2]) + k)
    _require(bool(occupied), "trilinear grid has no occupied cells")
    return np.asarray(sorted(occupied), dtype=np.int64)


def build_mujoco_trilinear_grid_v1(
    geometry: NativeTetSourceGeometryV1,
    *,
    contact: RigidContactProjectionV1,
    cellcount: tuple[int, int, int],
    density_kg_m3: float,
) -> MujocoTrilinearGridV1:
    """Build a mass-conserving grid with disjoint exact patch support."""

    _require(
        type(cellcount) is tuple
        and len(cellcount) == 3
        and all(type(value) is int and value >= 1 for value in cellcount),
        "cellcount must contain three positive integers",
    )
    counts: IntArray = np.asarray(cellcount, dtype=np.int64)
    points = geometry.points_m
    grid_from_world: FloatArray = np.eye(3, dtype=np.float64)
    points_grid = np.ascontiguousarray(points)
    cell_zero_tetrahedron_count = _cell_zero_tetrahedron_count_v1(
        points_grid,
        geometry.cells,
        counts,
    )
    lower = np.min(points_grid, axis=0)
    upper = np.max(points_grid, axis=0)
    extent = upper - lower
    _require(np.all(extent > 0.0), "source bounds must span all three axes")
    spacing = extent / counts

    axes = tuple(
        np.linspace(lower[axis], upper[axis], int(counts[axis]) + 1)
        for axis in range(3)
    )
    mesh = np.meshgrid(*axes, indexing="ij")
    node_positions_backend = np.ascontiguousarray(
        np.stack(mesh, axis=-1).reshape(-1, 3),
        dtype=np.float64,
    )
    node_positions_world = np.ascontiguousarray(
        node_positions_backend @ grid_from_world
    )
    scaled = (points_grid - lower) / extent * counts
    base = np.floor(scaled).astype(np.int64)
    base = np.minimum(base, counts - 1)
    fraction = scaled - base
    offsets: IntArray = np.asarray(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.int64,
    )
    support = base[:, None, :] + offsets[None, :, :]
    node_indices = (support[:, :, 0] * (counts[1] + 1) + support[:, :, 1]) * (
        counts[2] + 1
    ) + support[:, :, 2]
    factors = np.where(
        offsets[None, :, :] == 1,
        fraction[:, None, :],
        1.0 - fraction[:, None, :],
    )
    weights = np.ascontiguousarray(np.prod(factors, axis=2), dtype=np.float64)
    _require(
        np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-14, rtol=0.0),
        "trilinear weights changed",
    )
    reconstructed = np.sum(
        node_positions_backend[node_indices] * weights[:, :, None],
        axis=1,
    )
    _require(
        np.allclose(reconstructed, points_grid, atol=1.0e-12, rtol=0.0),
        "trilinear rest reconstruction changed",
    )

    _, total_mass = tetrahedral_nodal_masses_v1(
        geometry,
        density_kg_m3=density_kg_m3,
    )
    occupied_cells = _occupied_trilinear_cells_v1(
        points_grid,
        geometry.cells,
        counts,
    )
    node_masses: FloatArray = np.zeros(
        len(node_positions_backend),
        dtype=np.float64,
    )
    cell_mass = total_mass / len(occupied_cells)
    for flat_cell in occupied_cells:
        i = int(flat_cell) // (int(counts[1]) * int(counts[2]))
        remainder = int(flat_cell) % (int(counts[1]) * int(counts[2]))
        j = remainder // int(counts[2])
        k = remainder % int(counts[2])
        cell_nodes: IntArray = np.asarray(
            [
                ((i + di) * (int(counts[1]) + 1) + (j + dj)) * (int(counts[2]) + 1)
                + (k + dk)
                for di in (0, 1)
                for dj in (0, 1)
                for dk in (0, 1)
            ],
            dtype=np.int64,
        )
        node_masses[cell_nodes] += cell_mass / 8.0
    _require(
        np.isclose(np.sum(node_masses), total_mass, atol=1.0e-12, rtol=1.0e-12),
        "trilinear mass projection changed total reference mass",
    )
    active: IntArray = np.flatnonzero(node_masses > 0.0).astype(np.int64)

    patch_supports: list[IntArray] = []
    node_patch: IntArray = np.full(
        len(node_positions_backend),
        -1,
        dtype=np.int64,
    )
    for patch_index, material_nodes in enumerate(geometry.patch_node_indices):
        patch_weights = weights[material_nodes]
        patch_nodes = np.unique(
            node_indices[material_nodes][patch_weights > 1.0e-14]
        ).astype(np.int64)
        _require(len(patch_nodes) >= 4, "contact patch grid support is degenerate")
        _require(
            np.all(node_patch[patch_nodes] == -1),
            "contact patches overlap trilinear grid support",
        )
        node_patch[patch_nodes] = patch_index
        patch_supports.append(np.ascontiguousarray(patch_nodes))
    contact_nodes = np.concatenate(patch_supports)
    _require(
        np.all(node_masses[contact_nodes] > 0.0),
        "contact patch owns inactive trilinear nodes",
    )
    free: IntArray = np.setdiff1d(
        active,
        contact_nodes,
        assume_unique=True,
    ).astype(np.int64)
    inactive: IntArray = np.setdiff1d(
        np.arange(len(node_positions_backend), dtype=np.int64),
        active,
        assume_unique=True,
    ).astype(np.int64)
    _require(len(free) >= 1, "trilinear grid has no free dynamic nodes")

    transformed_nodes = np.array(node_positions_world, copy=True)
    for frame in range(len(contact.rotations)):
        transformed_nodes[:] = node_positions_world
        for patch_index, patch_nodes in enumerate(patch_supports):
            transformed_nodes[patch_nodes] = (
                node_positions_world[patch_nodes]
                @ contact.rotations[frame, patch_index].T
                + contact.translations_m[frame, patch_index]
            )
        projected = np.sum(
            transformed_nodes[node_indices[geometry.attachment_indices]]
            * weights[geometry.attachment_indices, :, None],
            axis=1,
        )
        _require(
            np.allclose(
                projected,
                contact.projected_targets_m[frame],
                atol=1.0e-10,
                rtol=0.0,
            ),
            "trilinear contact support does not reproduce rigid patch targets",
        )

    return MujocoTrilinearGridV1(
        cellcount=cellcount,
        grid_from_world_rotation=grid_from_world,
        grid_lower_m=np.ascontiguousarray(lower),
        grid_upper_m=np.ascontiguousarray(upper),
        grid_spacing_m=np.ascontiguousarray(spacing),
        cell_zero_tetrahedron_count=cell_zero_tetrahedron_count,
        node_positions_backend_m=node_positions_backend,
        node_positions_world_m=node_positions_world,
        vertex_node_indices=np.ascontiguousarray(node_indices, dtype=np.int64),
        vertex_node_weights=weights,
        occupied_cell_indices=occupied_cells,
        active_node_indices=np.ascontiguousarray(active),
        patch_node_indices=tuple(patch_supports),
        free_node_indices=np.ascontiguousarray(free),
        inactive_node_indices=np.ascontiguousarray(inactive),
        node_patch_index=np.ascontiguousarray(node_patch),
        node_masses_kg=np.ascontiguousarray(node_masses),
        total_reference_mass_kg=total_mass,
    )


@dataclass(frozen=True, slots=True)
class MujocoTrilinearSceneV1:
    """Deterministic low-level trilinear MJCF and body ownership roster."""

    xml: str
    xml_sha256: str
    grid: MujocoTrilinearGridV1
    node_body_names: tuple[str, ...]
    patch_body_names: tuple[tuple[str, ...], ...]
    free_body_names: tuple[str, ...]


def build_mujoco_trilinear_scene_v1(
    geometry: NativeTetSourceGeometryV1,
    *,
    contact: RigidContactProjectionV1,
    cellcount: tuple[int, int, int],
    integrator: str,
    integrator_time_step_s: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    density_kg_m3: float,
    edge_damping: float,
    elasticity_damping: float,
    joint_damping: float,
    solver_iterations: int,
    solver_tolerance: float,
) -> MujocoTrilinearSceneV1:
    """Build exact native MJCF for a reduced-coordinate source replay."""

    _require(
        integrator in {"implicit", "implicitfast"},
        "integrator must be implicit or implicitfast",
    )
    time_step = _finite(
        integrator_time_step_s,
        name="integrator_time_step_s",
        positive=True,
    )
    young = _finite(young_modulus_pa, name="young_modulus_pa", positive=True)
    poisson = _finite(poisson_ratio, name="poisson_ratio")
    _require(0.0 < poisson < 0.5, "poisson_ratio must lie in (0,0.5)")
    edge = _finite(edge_damping, name="edge_damping")
    elasticity = _finite(elasticity_damping, name="elasticity_damping")
    joint = _finite(joint_damping, name="joint_damping")
    _require(
        edge >= 0.0 and elasticity >= 0.0 and joint >= 0.0,
        "damping values must be non-negative",
    )
    _require(
        type(solver_iterations) is int and solver_iterations >= 1,
        "solver_iterations must be a positive integer",
    )
    tolerance = _finite(solver_tolerance, name="solver_tolerance", positive=True)
    grid = build_mujoco_trilinear_grid_v1(
        geometry,
        contact=contact,
        cellcount=cellcount,
        density_kg_m3=density_kg_m3,
    )

    root = ElementTree.Element("mujoco", {"model": "bpt_source_trilinear_flex_v1"})
    ElementTree.SubElement(
        root,
        "option",
        {
            "timestep": f"{time_step:.17g}",
            "gravity": "0 0 0",
            "integrator": integrator,
            "iterations": str(solver_iterations),
            "tolerance": f"{tolerance:.17g}",
        },
    )
    world = ElementTree.SubElement(root, "worldbody")
    node_body_names = [""] * len(grid.node_positions_backend_m)
    for node in grid.inactive_node_indices:
        node_index = int(node)
        name = f"inactive_grid_node_{node_index}"
        node_body_names[node_index] = name
        ElementTree.SubElement(
            world,
            "body",
            {
                "name": name,
                "pos": _numbers(grid.node_positions_backend_m[node]),
            },
        )
    patch_body_names: list[tuple[str, ...]] = []
    for patch_index, patch_nodes in enumerate(grid.patch_node_indices):
        names: list[str] = []
        for node in patch_nodes:
            name = f"contact_patch_{patch_index}_node_{int(node)}"
            names.append(name)
            node_body_names[int(node)] = name
            ElementTree.SubElement(
                world,
                "body",
                {
                    "name": name,
                    "mocap": "true",
                    "pos": _numbers(grid.node_positions_backend_m[node]),
                },
            )
        patch_body_names.append(tuple(names))

    characteristic_length = max(
        float(np.linalg.norm(grid.grid_spacing_m)),
        1.0e-9,
    )
    free_body_names: list[str] = []
    for node in grid.free_node_indices:
        node_index = int(node)
        name = f"free_grid_node_{node_index}"
        free_body_names.append(name)
        node_body_names[node_index] = name
        body = ElementTree.SubElement(
            world,
            "body",
            {
                "name": name,
                "pos": _numbers(grid.node_positions_backend_m[node]),
            },
        )
        mass = float(grid.node_masses_kg[node])
        inertia = max(mass * characteristic_length**2 / 6.0, 1.0e-15)
        ElementTree.SubElement(
            body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": f"{mass:.17g}",
                "diaginertia": _numbers(np.repeat(inertia, 3)),
            },
        )
        joint_axes: IntArray = np.eye(3, dtype=np.int64)
        for axis_index, axis in enumerate(joint_axes):
            ElementTree.SubElement(
                body,
                "joint",
                {
                    "name": f"{name}_slide_{axis_index}",
                    "type": "slide",
                    "axis": _numbers(axis),
                    "damping": f"{joint:.17g}",
                },
            )

    _require(
        all(node_body_names),
        "trilinear node ownership roster is incomplete",
    )

    deformable = ElementTree.SubElement(root, "deformable")
    flex = ElementTree.SubElement(
        deformable,
        "flex",
        {
            "name": "soft",
            "dim": "3",
            "dof": "trilinear",
            "cellcount": _numbers(np.asarray(cellcount, dtype=np.int64)),
            "body": " ".join(["world"] * len(geometry.points_m)),
            "vertex": _numbers(geometry.points_m @ grid.grid_from_world_rotation.T),
            "element": _numbers(geometry.cells),
            "node": " ".join(node_body_names),
            "radius": "0",
        },
    )
    ElementTree.SubElement(
        flex,
        "contact",
        {"contype": "0", "conaffinity": "0", "selfcollide": "none"},
    )
    ElementTree.SubElement(flex, "edge", {"damping": f"{edge:.17g}"})
    ElementTree.SubElement(
        flex,
        "elasticity",
        {
            "young": f"{young:.17g}",
            "poisson": f"{poisson:.17g}",
            "damping": f"{elasticity:.17g}",
        },
    )
    xml = ElementTree.tostring(root, encoding="unicode", short_empty_elements=True)
    return MujocoTrilinearSceneV1(
        xml=xml,
        xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        grid=grid,
        node_body_names=tuple(node_body_names),
        patch_body_names=tuple(patch_body_names),
        free_body_names=tuple(free_body_names),
    )


@dataclass(frozen=True, slots=True)
class MujocoTrilinearSourceReplayV1:
    """Native reduced trajectory plus interpolation and safety diagnostics."""

    positions_m: FloatArray
    deformation_determinants: FloatArray
    minimum_continuation_deformation_determinant: float
    maximum_attachment_error_m: float
    maximum_free_grid_node_displacement_m: float
    grid_from_world_rotation: FloatArray
    cell_zero_tetrahedron_count: int
    empty_cell_stiffness_sentinel_applied: bool
    empty_cell_stiffness_sentinel: float
    native_step_count: int
    model_xml_sha256: str
    material_vertex_count: int
    tetrahedron_count: int
    grid_node_count: int
    occupied_grid_cell_count: int
    active_grid_node_count: int
    free_grid_node_count: int
    contact_grid_node_count: int
    contact_patch_count: int
    total_reference_mass_kg: float


def run_mujoco_trilinear_source_replay_v1(
    *,
    native: NativeMujocoFlexModulesV1,
    points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    contact: RigidContactProjectionV1,
    driven: bool,
    cellcount: tuple[int, int, int],
    integrator: str,
    integrator_time_step_s: float,
    interval_substeps: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    density_kg_m3: float,
    edge_damping: float,
    elasticity_damping: float,
    joint_damping: float,
    solver_iterations: int,
    solver_tolerance: float,
    hard_minimum_deformation_determinant: float,
) -> MujocoTrilinearSourceReplayV1:  # pragma: no cover - exact native runtime
    """Run one registered source replay on the reduced native Flex state."""

    _require(
        type(interval_substeps) is int and interval_substeps >= 1,
        "interval_substeps must be a positive integer",
    )
    determinant_floor = _finite(
        hard_minimum_deformation_determinant,
        name="hard_minimum_deformation_determinant",
        positive=True,
    )
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points_m,
        cells=cells,
        attachment_indices=attachment_indices,
        contact=contact,
    )
    scene = build_mujoco_trilinear_scene_v1(
        geometry,
        contact=contact,
        cellcount=cellcount,
        integrator=integrator,
        integrator_time_step_s=integrator_time_step_s,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
        density_kg_m3=density_kg_m3,
        edge_damping=edge_damping,
        elasticity_damping=elasticity_damping,
        joint_damping=joint_damping,
        solver_iterations=solver_iterations,
        solver_tolerance=solver_tolerance,
    )
    mujoco = native.mujoco
    model = mujoco.MjModel.from_xml_string(scene.xml)
    data = mujoco.MjData(model)
    flex_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "soft"))
    _require(flex_id == 0, "MuJoCo trilinear flex identity changed")
    _require(
        int(model.nflexvert) == len(geometry.points_m)
        and int(model.nflexelem) == len(geometry.cells)
        and int(model.flex_nodenum[flex_id])
        == len(scene.grid.node_positions_backend_m),
        "MuJoCo compiled trilinear topology changed",
    )
    _require(
        int(model.flex_rigid[flex_id]) == 0,
        "MuJoCo compiled trilinear flex became rigid",
    )
    stiffness_address = int(model.flex_stiffnessadr[flex_id])
    _require(stiffness_address >= 0, "MuJoCo trilinear stiffness is unavailable")
    stiffness = np.asarray(model.flex_stiffness, dtype=np.float64)
    cell_stiffness_size = (3 * 8) ** 2
    compiled_occupied_cells = np.flatnonzero(
        np.asarray(
            [
                np.linalg.norm(
                    stiffness[
                        stiffness_address
                        + cell_index * cell_stiffness_size : stiffness_address
                        + (cell_index + 1) * cell_stiffness_size
                    ]
                )
                > 0.0
                for cell_index in range(int(np.prod(scene.grid.cellcount)))
            ],
            dtype=np.bool_,
        )
    )
    _require(
        np.array_equal(
            compiled_occupied_cells,
            scene.grid.occupied_cell_indices,
        ),
        "MuJoCo compiled occupied-cell roster changed",
    )
    cell_zero_occupied = bool(
        len(compiled_occupied_cells) > 0 and compiled_occupied_cells[0] == 0
    )
    _require(
        cell_zero_occupied == (scene.grid.cell_zero_tetrahedron_count > 0),
        "MuJoCo compiled cell-zero occupancy changed",
    )
    sentinel_applied = not cell_zero_occupied
    if sentinel_applied:
        _require(
            np.all(
                stiffness[stiffness_address : stiffness_address + cell_stiffness_size]
                == 0.0
            ),
            "MuJoCo empty cell-zero stiffness block changed",
        )
        model.flex_stiffness[stiffness_address] = EMPTY_CELL_STIFFNESS_SENTINEL
        _require(
            float(model.flex_stiffness[stiffness_address])
            == EMPTY_CELL_STIFFNESS_SENTINEL,
            "MuJoCo empty-cell stiffness sentinel was not installed",
        )
    else:
        _require(
            float(model.flex_stiffness[stiffness_address]) != 0.0,
            "MuJoCo occupied cell-zero stiffness is unavailable",
        )
    expected_node_body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in scene.node_body_names
        ],
        dtype=np.int64,
    )
    _require(
        np.array_equal(
            np.asarray(model.flex_nodebodyid, dtype=np.int64),
            expected_node_body_ids,
        ),
        "MuJoCo compiled trilinear node/body roster changed",
    )
    _require(
        np.allclose(
            np.asarray(model.flex_node0, dtype=np.float64),
            scene.grid.node_positions_backend_m,
            atol=1.0e-12,
            rtol=0.0,
        ),
        "MuJoCo compiled trilinear rest grid changed",
    )
    free_qpos_addresses: list[IntArray] = []
    for body_id in expected_node_body_ids[scene.grid.free_node_indices]:
        joint_address = int(model.body_jntadr[body_id])
        _require(
            int(model.body_jntnum[body_id]) == 3,
            "MuJoCo trilinear free node lost its three translational joints",
        )
        addresses = np.asarray(
            model.jnt_qposadr[joint_address : joint_address + 3],
            dtype=np.int64,
        )
        _require(len(addresses) == 3, "MuJoCo trilinear qpos roster changed")
        free_qpos_addresses.append(addresses)
    free_qpos = np.stack(free_qpos_addresses)
    patch_mocap_ids: list[IntArray] = []
    for names in scene.patch_body_names:
        ids = np.asarray(
            [
                model.body_mocapid[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                ]
                for name in names
            ],
            dtype=np.int64,
        )
        _require(np.all(ids >= 0), "MuJoCo trilinear patch lost mocap ownership")
        patch_mocap_ids.append(ids)
    mujoco.mj_forward(model, data)
    initial_backend = np.asarray(data.flexvert_xpos, dtype=np.float64).copy()
    _require(
        np.allclose(
            initial_backend,
            geometry.points_m @ scene.grid.grid_from_world_rotation.T,
            atol=1.0e-12,
            rtol=0.0,
        ),
        "MuJoCo trilinear initial material state changed",
    )
    initial = np.ascontiguousarray(
        initial_backend @ scene.grid.grid_from_world_rotation
    )

    frame_count = int(contact.rotations.shape[0])
    positions: FloatArray = np.empty(
        (frame_count, len(geometry.points_m), 3),
        dtype=np.float64,
    )
    positions[0] = initial
    minimum_determinant = 1.0
    maximum_attachment_error = 0.0
    maximum_free_grid_node_displacement = 0.0
    native_step_count = 0
    for frame in range(1, frame_count):
        for substep in range(1, interval_substeps + 1):
            rotations, translations = contact_transform_at_fraction_v1(
                contact,
                previous_frame=frame - 1,
                target_frame=frame,
                fraction=substep / interval_substeps,
                driven=driven,
            )
            targets = attached_targets_from_transform_v1(
                geometry,
                contact,
                rotations=rotations,
                translations_m=translations,
            )
            for patch_index, (patch_nodes, mocap_ids) in enumerate(
                zip(scene.grid.patch_node_indices, patch_mocap_ids, strict=True)
            ):
                node_targets_world = (
                    scene.grid.node_positions_world_m[patch_nodes]
                    @ rotations[patch_index].T
                    + translations[patch_index]
                )
                node_targets = (
                    node_targets_world @ scene.grid.grid_from_world_rotation.T
                )
                for node_target, mocap_id in zip(
                    node_targets,
                    mocap_ids,
                    strict=True,
                ):
                    data.mocap_pos[mocap_id] = node_target
                    data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            previous_time = float(data.time)
            previous_warning_count = _warning_count(data)
            mujoco.mj_step(model, data)
            native_step_count += 1
            current_backend = np.asarray(
                data.flexvert_xpos,
                dtype=np.float64,
            ).copy()
            current = np.ascontiguousarray(
                current_backend @ scene.grid.grid_from_world_rotation
            )
            _require(
                float(data.time) > previous_time,
                "MuJoCo trilinear replay reset or failed to advance time "
                f"at frame={frame}, substep={substep}, "
                f"previous_time_s={previous_time:.17g}, "
                f"observed_time_s={float(data.time):.17g}",
            )
            warning_count = _warning_count(data)
            _require(
                warning_count == previous_warning_count,
                "MuJoCo trilinear replay emitted a native numerical warning "
                f"at frame={frame}, substep={substep}, "
                f"warning_delta={warning_count - previous_warning_count}",
            )
            _require(
                np.all(np.isfinite(current_backend))
                and np.all(np.isfinite(np.asarray(data.qpos)))
                and np.all(np.isfinite(np.asarray(data.qvel))),
                "MuJoCo trilinear replay produced non-finite state",
            )
            determinants = replay_deformation_determinants_v1(
                geometry,
                current[None],
            )[0]
            step_minimum = float(np.min(determinants))
            minimum_determinant = min(minimum_determinant, step_minimum)
            _require(
                step_minimum >= determinant_floor,
                "MuJoCo trilinear replay violated its hard orientation threshold "
                f"at frame={frame}, substep={substep}, "
                f"minimum_determinant={step_minimum:.17g}, "
                f"required_minimum={determinant_floor:.17g}",
            )
            attachment_error = float(
                np.max(
                    np.linalg.norm(
                        current[geometry.attachment_indices] - targets,
                        axis=1,
                    )
                )
            )
            maximum_attachment_error = max(
                maximum_attachment_error,
                attachment_error,
            )
            free_grid_displacement = float(
                np.max(np.linalg.norm(np.asarray(data.qpos)[free_qpos], axis=1))
            )
            maximum_free_grid_node_displacement = max(
                maximum_free_grid_node_displacement,
                free_grid_displacement,
            )
        positions[frame] = current

    deformation_determinants = replay_deformation_determinants_v1(
        geometry,
        positions,
    )
    contact_grid_node_count = sum(map(len, scene.grid.patch_node_indices))
    return MujocoTrilinearSourceReplayV1(
        positions_m=np.ascontiguousarray(positions),
        deformation_determinants=np.ascontiguousarray(deformation_determinants),
        minimum_continuation_deformation_determinant=minimum_determinant,
        maximum_attachment_error_m=maximum_attachment_error,
        maximum_free_grid_node_displacement_m=maximum_free_grid_node_displacement,
        grid_from_world_rotation=np.ascontiguousarray(
            scene.grid.grid_from_world_rotation
        ),
        cell_zero_tetrahedron_count=scene.grid.cell_zero_tetrahedron_count,
        empty_cell_stiffness_sentinel_applied=sentinel_applied,
        empty_cell_stiffness_sentinel=(
            EMPTY_CELL_STIFFNESS_SENTINEL if sentinel_applied else 0.0
        ),
        native_step_count=native_step_count,
        model_xml_sha256=scene.xml_sha256,
        material_vertex_count=len(geometry.points_m),
        tetrahedron_count=len(geometry.cells),
        grid_node_count=len(scene.grid.node_positions_backend_m),
        occupied_grid_cell_count=len(scene.grid.occupied_cell_indices),
        active_grid_node_count=len(scene.grid.active_node_indices),
        free_grid_node_count=len(scene.grid.free_node_indices),
        contact_grid_node_count=contact_grid_node_count,
        contact_patch_count=len(scene.grid.patch_node_indices),
        total_reference_mass_kg=scene.grid.total_reference_mass_kg,
    )


__all__ = [
    "ATTACHMENT_MODEL",
    "BACKEND_VARIANT",
    "CONSTITUTIVE_MODEL",
    "EMPTY_CELL_STIFFNESS_SENTINEL",
    "GRID_FRAME_POLICY",
    "INTERPOLATION_MODEL",
    "MASS_MODEL",
    "MujocoTrilinearGridV1",
    "MujocoTrilinearSceneV1",
    "MujocoTrilinearSourceReplayV1",
    "build_mujoco_trilinear_grid_v1",
    "build_mujoco_trilinear_scene_v1",
    "run_mujoco_trilinear_source_replay_v1",
]
