"""Exact-runtime MuJoCo volumetric Flex replay for registered source meshes.

The adapter uses a low-level tetrahedral ``flex`` roster. Every free material
vertex receives one translational body, while every attached vertex is owned
directly by one mocap body. Their targets come from the registered rigid patch
projection, supplying exact moving Dirichlet data without a spring, equality
weld, or post-step position overwrite.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
from xml.etree import ElementTree

import numpy as np
import numpy.typing as npt

from .jax_fem_source_qualification_v1 import RigidContactProjectionV1
from .native_tet_fem_source_v1 import (
    NativeTetSourceGeometryV1,
    attached_targets_from_transform_v1,
    contact_transform_at_fraction_v1,
    prepare_native_tet_source_geometry_v1,
    replay_deformation_determinants_v1,
    tetrahedral_nodal_masses_v1,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]

MUJOCO_REVISION = "237c17e48539b6c90bf90d3161547cbdcbfaa1e0"
MUJOCO_VERSION = "3.9.0"
MUJOCO_REPOSITORY = "https://github.com/google-deepmind/mujoco"
MUJOCO_WHEEL_FILENAME = (
    "mujoco-3.9.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
MUJOCO_WHEEL_SHA256 = "c148824d73487fe5ee29c371eff981645f372ccada1f20ea331288323e37c65e"
MUJOCO_INSTALLED_FILE_SHA256 = {
    "mujoco/__init__.py": (
        "f8e5b528617004b6215e16cb0c945faf1f8c7b5d798e5b92e3aabce19c838497"
    ),
    "mujoco/_structs.cpython-310-x86_64-linux-gnu.so": (
        "b6875dcfef3f895f8c293f9d5e20d4da8e7e37df1f3bda201124c1f2951b6c63"
    ),
    "mujoco/libmujoco.so.3.9.0": (
        "526773636a795dad11e094c8655d2375984a5cd7090f254d86bb71074651b852"
    ),
}
RUNTIME_SCHEMA = "bayesian-phystwin.mujoco-flex-source-runtime-v1"
BACKEND_VARIANT = "mujoco-volumetric-flex-v1"
CONSTITUTIVE_MODEL = "MuJoCo-native-Saint-Venant-Kirchhoff-flex-elasticity"
ATTACHMENT_MODEL = "direct-rigid-projected-vertex-mocap-Dirichlet-v2"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbers(values: npt.ArrayLike) -> str:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return " ".join(str(int(value)) for value in array.reshape(-1))
    return " ".join(f"{float(value):.17g}" for value in array.reshape(-1))


@dataclass(frozen=True, slots=True)
class NativeMujocoFlexModulesV1:
    """Exact MuJoCo modules admitted by the frozen wheel identity."""

    mujoco: Any
    package_root: Path
    installed_records: Mapping[str, Mapping[str, str]]


def load_native_mujoco_flex_modules_v1(
    wheel_path: str | Path,
) -> NativeMujocoFlexModulesV1:
    """Import MuJoCo only after its wheel, ABI, and installed files match."""

    wheel = Path(wheel_path).absolute()
    _require(
        wheel.is_file() and not wheel.is_symlink(),
        "MuJoCo wheel must be an ordinary file",
    )
    _require(
        wheel.name == MUJOCO_WHEEL_FILENAME,
        "MuJoCo wheel filename differs from the frozen runtime",
    )
    _require(
        _sha256_file(wheel) == MUJOCO_WHEEL_SHA256,
        "MuJoCo wheel SHA-256 differs from the frozen runtime",
    )
    _require(
        platform.python_implementation() == "CPython"
        and platform.python_version_tuple()[:2] == ("3", "10"),
        "MuJoCo source replay requires the frozen CPython 3.10 ABI",
    )
    _require(
        importlib.metadata.version("mujoco") == MUJOCO_VERSION,
        "MuJoCo installed version differs from the frozen runtime",
    )
    mujoco = importlib.import_module("mujoco")
    _require(
        str(getattr(mujoco, "__version__", "")) == MUJOCO_VERSION,
        "MuJoCo reported version differs from the frozen runtime",
    )
    distribution = importlib.metadata.distribution("mujoco")
    package_root = Path(str(distribution.locate_file(""))).resolve()
    module_path = Path(str(getattr(mujoco, "__file__", ""))).resolve()
    _require(
        package_root == module_path or package_root in module_path.parents,
        "MuJoCo module was imported outside the frozen distribution",
    )
    records: dict[str, Mapping[str, str]] = {}
    for relative_path, expected in MUJOCO_INSTALLED_FILE_SHA256.items():
        path = Path(str(distribution.locate_file(relative_path))).resolve()
        _require(
            path.is_file() and not path.is_symlink(),
            f"MuJoCo runtime member is unavailable: {relative_path}",
        )
        observed = _sha256_file(path)
        _require(
            observed == expected,
            f"MuJoCo runtime member changed: {relative_path}",
        )
        records[relative_path] = {"sha256": observed}
    return NativeMujocoFlexModulesV1(
        mujoco=mujoco,
        package_root=package_root,
        installed_records=records,
    )


@dataclass(frozen=True, slots=True)
class MujocoFlexSceneV1:
    """Deterministic low-level MJCF and its persistent vertex/body roster."""

    xml: str
    xml_sha256: str
    vertex_body_names: tuple[str, ...]
    attachment_body_names: tuple[str, ...]
    free_body_names: tuple[str, ...]
    total_reference_mass_kg: float


def build_mujoco_flex_scene_v1(
    geometry: NativeTetSourceGeometryV1,
    *,
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
) -> MujocoFlexSceneV1:
    """Build fixed-identity MJCF for an arbitrary registered tetrahedral mesh."""

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
    masses, total_mass = tetrahedral_nodal_masses_v1(
        geometry,
        density_kg_m3=density_kg_m3,
    )
    node_attachment: npt.NDArray[np.int64] = np.full(
        len(geometry.points_m),
        -1,
        dtype=np.int64,
    )
    attachment_names = tuple(
        f"contact_vertex_{int(node)}" for node in geometry.attachment_indices
    )
    node_attachment[geometry.attachment_indices] = np.arange(
        len(geometry.attachment_indices),
        dtype=np.int64,
    )

    root = ElementTree.Element("mujoco", {"model": "bpt_source_volumetric_flex_v1"})
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
    for name, node in zip(
        attachment_names,
        geometry.attachment_indices,
        strict=True,
    ):
        ElementTree.SubElement(
            world,
            "body",
            {
                "name": name,
                "mocap": "true",
                "pos": _numbers(geometry.points_m[node]),
            },
        )

    free_names: list[str] = []
    body_names: list[str] = []
    local_vertices = np.empty_like(geometry.points_m)
    characteristic_length = max(
        float(np.linalg.norm(np.ptp(geometry.points_m, axis=0))), 1e-9
    )
    for node_index, point in enumerate(geometry.points_m):
        attachment_index = int(node_attachment[node_index])
        if attachment_index >= 0:
            body_names.append(attachment_names[attachment_index])
            local_vertices[node_index] = 0.0
            continue
        name = f"free_vertex_{node_index}"
        free_names.append(name)
        body_names.append(name)
        local_vertices[node_index] = 0.0
        body = ElementTree.SubElement(
            world,
            "body",
            {"name": name, "pos": _numbers(point)},
        )
        inertia = max(float(masses[node_index]) * characteristic_length**2 / 6.0, 1e-15)
        ElementTree.SubElement(
            body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": f"{float(masses[node_index]):.17g}",
                "diaginertia": _numbers(np.repeat(inertia, 3)),
            },
        )
        axes: npt.NDArray[np.int64] = np.eye(3, dtype=np.int64)
        for axis_index, axis in enumerate(axes):
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

    deformable = ElementTree.SubElement(root, "deformable")
    flex = ElementTree.SubElement(
        deformable,
        "flex",
        {
            "name": "soft",
            "dim": "3",
            "body": " ".join(body_names),
            "vertex": _numbers(local_vertices),
            "element": _numbers(geometry.cells),
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
    return MujocoFlexSceneV1(
        xml=xml,
        xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        vertex_body_names=tuple(body_names),
        attachment_body_names=attachment_names,
        free_body_names=tuple(free_names),
        total_reference_mass_kg=total_mass,
    )


@dataclass(frozen=True, slots=True)
class MujocoFlexSourceReplayV1:
    """Native trajectory plus fixed-reference and attachment diagnostics."""

    positions_m: FloatArray
    deformation_determinants: FloatArray
    minimum_continuation_deformation_determinant: float
    maximum_attachment_error_m: float
    native_step_count: int
    model_xml_sha256: str
    material_vertex_count: int
    tetrahedron_count: int
    free_vertex_count: int
    contact_patch_count: int
    total_reference_mass_kg: float


def _warning_count(data: Any) -> int:
    return sum(int(warning.number) for warning in data.warning)


def run_mujoco_flex_source_replay_v1(
    *,
    native: NativeMujocoFlexModulesV1,
    points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    contact: RigidContactProjectionV1,
    driven: bool,
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
) -> MujocoFlexSourceReplayV1:  # pragma: no cover - exact native runtime
    """Run one complete registered source replay with exact patch ownership."""

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
    scene = build_mujoco_flex_scene_v1(
        geometry,
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
    _require(flex_id == 0, "MuJoCo flex identity changed")
    _require(
        int(model.nflexvert) == len(geometry.points_m)
        and int(model.nflexelem) == len(geometry.cells),
        "MuJoCo compiled flex topology changed",
    )
    expected_body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in scene.vertex_body_names
        ],
        dtype=np.int64,
    )
    _require(
        np.array_equal(
            np.asarray(model.flex_vertbodyid, dtype=np.int64),
            expected_body_ids,
        ),
        "MuJoCo compiled vertex/body roster changed",
    )
    mocap_ids = np.asarray(
        [
            model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
            for name in scene.attachment_body_names
        ],
        dtype=np.int64,
    )
    _require(
        np.array_equal(mocap_ids, np.arange(len(mocap_ids), dtype=np.int64)),
        "MuJoCo mocap attachment roster changed",
    )
    mujoco.mj_forward(model, data)
    initial = np.asarray(data.flexvert_xpos, dtype=np.float64).copy()
    _require(
        np.allclose(initial, geometry.points_m, atol=1e-12, rtol=0.0),
        "MuJoCo initial material state changed",
    )

    frame_count = int(contact.rotations.shape[0])
    positions: FloatArray = np.empty(
        (frame_count, len(geometry.points_m), 3),
        dtype=np.float64,
    )
    positions[0] = initial
    minimum_determinant = 1.0
    maximum_attachment_error = 0.0
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
            for attachment_index, mocap_id in enumerate(mocap_ids):
                data.mocap_pos[mocap_id] = targets[attachment_index]
                data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            previous_time = float(data.time)
            previous_warning_count = _warning_count(data)
            mujoco.mj_step(model, data)
            native_step_count += 1
            current = np.asarray(data.flexvert_xpos, dtype=np.float64).copy()
            _require(
                float(data.time) > previous_time,
                "MuJoCo reset or failed to advance time "
                f"at frame={frame}, substep={substep}, "
                f"previous_time_s={previous_time:.17g}, "
                f"observed_time_s={float(data.time):.17g}",
            )
            warning_count = _warning_count(data)
            _require(
                warning_count == previous_warning_count,
                "MuJoCo emitted a native numerical warning "
                f"at frame={frame}, substep={substep}, "
                f"warning_delta={warning_count - previous_warning_count}",
            )
            _require(
                np.all(np.isfinite(current))
                and np.all(np.isfinite(np.asarray(data.qpos)))
                and np.all(np.isfinite(np.asarray(data.qvel))),
                "MuJoCo source replay produced non-finite state",
            )
            determinants = replay_deformation_determinants_v1(
                geometry,
                current[None],
            )[0]
            step_minimum = float(np.min(determinants))
            minimum_determinant = min(minimum_determinant, step_minimum)
            _require(
                step_minimum >= determinant_floor,
                "MuJoCo source replay violated its hard orientation threshold "
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
        positions[frame] = current

    deformation_determinants = replay_deformation_determinants_v1(
        geometry,
        positions,
    )
    return MujocoFlexSourceReplayV1(
        positions_m=np.ascontiguousarray(positions),
        deformation_determinants=np.ascontiguousarray(deformation_determinants),
        minimum_continuation_deformation_determinant=minimum_determinant,
        maximum_attachment_error_m=maximum_attachment_error,
        native_step_count=native_step_count,
        model_xml_sha256=scene.xml_sha256,
        material_vertex_count=len(geometry.points_m),
        tetrahedron_count=len(geometry.cells),
        free_vertex_count=len(scene.free_body_names),
        contact_patch_count=len(geometry.patch_node_indices),
        total_reference_mass_kg=scene.total_reference_mass_kg,
    )


__all__ = [
    "ATTACHMENT_MODEL",
    "BACKEND_VARIANT",
    "CONSTITUTIVE_MODEL",
    "MUJOCO_INSTALLED_FILE_SHA256",
    "MUJOCO_REVISION",
    "MUJOCO_VERSION",
    "MUJOCO_WHEEL_FILENAME",
    "MUJOCO_WHEEL_SHA256",
    "MujocoFlexSceneV1",
    "MujocoFlexSourceReplayV1",
    "NativeMujocoFlexModulesV1",
    "build_mujoco_flex_scene_v1",
    "load_native_mujoco_flex_modules_v1",
    "run_mujoco_flex_source_replay_v1",
]
