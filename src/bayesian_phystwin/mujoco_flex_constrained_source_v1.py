"""MuJoCo volumetric Flex replay with native point constraints.

Every material vertex remains a dynamic three-translation body. Registered
contact vertices are connected to independent mocap targets with MuJoCo's
native point-equality constraints, so attachment reactions enter the physical
state instead of replacing its degrees of freedom.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, TypeAlias
from xml.etree import ElementTree

import numpy as np
import numpy.typing as npt

from .jax_fem_source_qualification_v1 import RigidContactProjectionV1
from .mujoco_flex_source_v1 import (
    NativeMujocoFlexModulesV1,
    build_mujoco_flex_scene_v1,
)
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

BACKEND_VARIANT = "mujoco-volumetric-flex-point-constraint-v1"
CONSTITUTIVE_MODEL = "MuJoCo-native-Saint-Venant-Kirchhoff-flex-elasticity"
ATTACHMENT_MODEL = "dynamic-vertex-native-connect-to-rigid-target-v1"
CONSTRAINT_POLICY = "critical-native-connect-minimum-safe-timeconstant-v1"
MASS_POLICY = "minimal-conservative-uniform-blend-condition-cap-v1"
MASS_CONDITION_NUMBER_CAP = 100.0


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _numbers(values: npt.ArrayLike) -> str:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return " ".join(str(int(value)) for value in array.reshape(-1))
    return " ".join(f"{float(value):.17g}" for value in array.reshape(-1))


def _warning_count(data: Any) -> int:
    return sum(int(warning.number) for warning in data.warning)


def condition_lumped_vertex_masses_v1(
    masses_kg: npt.ArrayLike,
    *,
    condition_number_cap: float = MASS_CONDITION_NUMBER_CAP,
) -> tuple[FloatArray, float]:
    """Conservatively apply the minimum uniform blend meeting a mass cap."""

    masses = np.asarray(masses_kg, dtype=np.float64)
    _require(
        masses.ndim == 1
        and len(masses) >= 1
        and np.all(np.isfinite(masses))
        and np.all(masses > 0.0),
        "lumped masses must be a positive finite vector",
    )
    cap = float(condition_number_cap)
    _require(np.isfinite(cap) and cap >= 1.0, "mass condition cap is invalid")
    minimum = float(np.min(masses))
    maximum = float(np.max(masses))
    mean = float(np.mean(masses))
    if maximum / minimum <= cap:
        return np.ascontiguousarray(masses), 0.0
    numerator = maximum - cap * minimum
    denominator = numerator + (cap - 1.0) * mean
    blend = numerator / denominator
    _require(0.0 < blend < 1.0, "mass-conditioning blend is invalid")
    conditioned = np.ascontiguousarray((1.0 - blend) * masses + blend * mean)
    target_total = float(np.sum(masses, dtype=np.float64))
    conditioned[int(np.argmax(conditioned))] += target_total - float(
        np.sum(conditioned, dtype=np.float64)
    )
    _require(
        np.isclose(
            np.sum(conditioned, dtype=np.float64),
            target_total,
            atol=np.spacing(target_total),
            rtol=0.0,
        ),
        "mass conditioning changed total reference mass",
    )
    _require(
        float(np.max(conditioned) / np.min(conditioned)) <= cap * (1.0 + 1.0e-12),
        "mass conditioning did not meet its cap",
    )
    return conditioned, blend


@dataclass(frozen=True, slots=True)
class MujocoConstrainedFlexSceneV1:
    """Deterministic native point-constrained Flex scene."""

    xml: str
    xml_sha256: str
    vertex_body_names: tuple[str, ...]
    attachment_vertex_body_names: tuple[str, ...]
    target_body_names: tuple[str, ...]
    equality_names: tuple[str, ...]
    total_reference_mass_kg: float
    constraint_time_constant_s: float
    mass_uniform_blend_fraction: float
    conditioned_mass_condition_number: float


def build_mujoco_constrained_flex_scene_v1(
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
) -> MujocoConstrainedFlexSceneV1:
    """Replace direct mocap ownership with dynamic native connect constraints."""

    base = build_mujoco_flex_scene_v1(
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
    raw_masses, total_mass = tetrahedral_nodal_masses_v1(
        geometry,
        density_kg_m3=density_kg_m3,
    )
    masses, mass_blend = condition_lumped_vertex_masses_v1(raw_masses)
    _require(
        np.isclose(total_mass, base.total_reference_mass_kg, atol=0.0, rtol=0.0),
        "MuJoCo constrained reference mass changed",
    )
    root = ElementTree.fromstring(base.xml)
    root.attrib["model"] = "bpt_source_constrained_volumetric_flex_v1"
    world = root.find("./worldbody")
    if world is None:
        raise ValueError("MuJoCo constrained scene lost worldbody")
    characteristic_length = max(
        float(np.linalg.norm(np.ptp(geometry.points_m, axis=0))),
        1.0e-9,
    )
    for node_index, vertex_name in enumerate(base.vertex_body_names):
        vertex_body = world.find(f"./body[@name='{vertex_name}']")
        if vertex_body is None:
            raise ValueError("MuJoCo dynamic vertex body roster changed")
        inertial = vertex_body.find("./inertial")
        if inertial is None:
            continue
        mass = float(masses[node_index])
        inertia = max(mass * characteristic_length**2 / 6.0, 1.0e-15)
        inertial.attrib["mass"] = f"{mass:.17g}"
        inertial.attrib["diaginertia"] = _numbers(np.repeat(inertia, 3))
    attachment_vertex_names: list[str] = []
    target_names: list[str] = []
    equality_names: list[str] = []
    equality = ElementTree.SubElement(root, "equality")
    constraint_time_constant = 2.0 * float(integrator_time_step_s)
    for attachment_index, node in enumerate(geometry.attachment_indices):
        vertex_name = base.attachment_body_names[attachment_index]
        vertex_body = world.find(f"./body[@name='{vertex_name}']")
        if vertex_body is None:
            raise ValueError("MuJoCo attachment body roster changed")
        _require(
            vertex_body.attrib.pop("mocap", None) == "true",
            "MuJoCo direct attachment body was not mocap-owned",
        )
        mass = float(masses[int(node)])
        inertia = max(mass * characteristic_length**2 / 6.0, 1.0e-15)
        ElementTree.SubElement(
            vertex_body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": f"{mass:.17g}",
                "diaginertia": _numbers(np.repeat(inertia, 3)),
            },
        )
        axes: IntArray = np.eye(3, dtype=np.int64)
        for axis_index, axis in enumerate(axes):
            ElementTree.SubElement(
                vertex_body,
                "joint",
                {
                    "name": f"{vertex_name}_slide_{axis_index}",
                    "type": "slide",
                    "axis": _numbers(axis),
                    "damping": f"{float(joint_damping):.17g}",
                },
            )
        target_name = f"target_vertex_{int(node)}"
        ElementTree.SubElement(
            world,
            "body",
            {
                "name": target_name,
                "mocap": "true",
                "pos": _numbers(geometry.points_m[int(node)]),
            },
        )
        equality_name = f"attach_vertex_{int(node)}"
        ElementTree.SubElement(
            equality,
            "connect",
            {
                "name": equality_name,
                "body1": vertex_name,
                "body2": target_name,
                "anchor": "0 0 0",
                "solref": f"{constraint_time_constant:.17g} 1",
            },
        )
        attachment_vertex_names.append(vertex_name)
        target_names.append(target_name)
        equality_names.append(equality_name)

    xml = ElementTree.tostring(root, encoding="unicode", short_empty_elements=True)
    return MujocoConstrainedFlexSceneV1(
        xml=xml,
        xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        vertex_body_names=base.vertex_body_names,
        attachment_vertex_body_names=tuple(attachment_vertex_names),
        target_body_names=tuple(target_names),
        equality_names=tuple(equality_names),
        total_reference_mass_kg=total_mass,
        constraint_time_constant_s=constraint_time_constant,
        mass_uniform_blend_fraction=mass_blend,
        conditioned_mass_condition_number=float(np.max(masses) / np.min(masses)),
    )


@dataclass(frozen=True, slots=True)
class MujocoConstrainedFlexReplayV1:
    """Native constrained trajectory and source-safety diagnostics."""

    positions_m: FloatArray
    deformation_determinants: FloatArray
    minimum_continuation_deformation_determinant: float
    maximum_attachment_error_m: float
    maximum_constraint_residual_m: float
    native_step_count: int
    model_xml_sha256: str
    material_vertex_count: int
    tetrahedron_count: int
    constrained_vertex_count: int
    contact_patch_count: int
    total_reference_mass_kg: float
    constraint_time_constant_s: float
    mass_uniform_blend_fraction: float
    conditioned_mass_condition_number: float


def run_mujoco_constrained_flex_source_replay_v1(
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
) -> MujocoConstrainedFlexReplayV1:  # pragma: no cover - exact native runtime
    """Run one registered source replay with native point constraints."""

    _require(
        type(interval_substeps) is int and interval_substeps >= 1,
        "interval_substeps must be a positive integer",
    )
    determinant_floor = float(hard_minimum_deformation_determinant)
    _require(
        np.isfinite(determinant_floor) and determinant_floor > 0.0,
        "hard_minimum_deformation_determinant must be positive and finite",
    )
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points_m,
        cells=cells,
        attachment_indices=attachment_indices,
        contact=contact,
    )
    scene = build_mujoco_constrained_flex_scene_v1(
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
    _require(flex_id == 0, "MuJoCo constrained flex identity changed")
    _require(
        int(model.nflexvert) == len(geometry.points_m)
        and int(model.nflexelem) == len(geometry.cells)
        and int(model.neq) == len(geometry.attachment_indices),
        "MuJoCo constrained topology changed",
    )
    expected_body_ids: IntArray = np.asarray(
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
        "MuJoCo constrained vertex/body roster changed",
    )
    target_mocap_ids: IntArray = np.asarray(
        [
            model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
            for name in scene.target_body_names
        ],
        dtype=np.int64,
    )
    _require(
        np.array_equal(
            target_mocap_ids,
            np.arange(len(target_mocap_ids), dtype=np.int64),
        ),
        "MuJoCo constrained target roster changed",
    )
    _require(
        np.all(np.asarray(model.eq_type, dtype=np.int64) == 0),
        "MuJoCo attachment constraints are not native connect constraints",
    )
    mujoco.mj_forward(model, data)
    initial = np.asarray(data.flexvert_xpos, dtype=np.float64).copy()
    _require(
        np.allclose(initial, geometry.points_m, atol=1.0e-12, rtol=0.0),
        "MuJoCo constrained initial material state changed",
    )

    frame_count = int(contact.rotations.shape[0])
    positions: FloatArray = np.empty(
        (frame_count, len(geometry.points_m), 3),
        dtype=np.float64,
    )
    positions[0] = initial
    minimum_determinant = 1.0
    maximum_attachment_error = 0.0
    maximum_constraint_residual = 0.0
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
            for target, mocap_id in zip(targets, target_mocap_ids, strict=True):
                data.mocap_pos[mocap_id] = target
                data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            previous_time = float(data.time)
            previous_warning_count = _warning_count(data)
            mujoco.mj_step(model, data)
            native_step_count += 1
            current = np.asarray(data.flexvert_xpos, dtype=np.float64).copy()
            _require(
                float(data.time) > previous_time,
                "MuJoCo constrained replay reset or failed to advance time "
                f"at frame={frame}, substep={substep}, "
                f"previous_time_s={previous_time:.17g}, "
                f"observed_time_s={float(data.time):.17g}",
            )
            warning_count = _warning_count(data)
            _require(
                warning_count == previous_warning_count,
                "MuJoCo constrained replay emitted a numerical warning "
                f"at frame={frame}, substep={substep}, "
                f"warning_delta={warning_count - previous_warning_count}",
            )
            _require(
                np.all(np.isfinite(current))
                and np.all(np.isfinite(np.asarray(data.qpos)))
                and np.all(np.isfinite(np.asarray(data.qvel))),
                "MuJoCo constrained replay produced non-finite state",
            )
            determinants = replay_deformation_determinants_v1(
                geometry,
                current[None],
            )[0]
            step_minimum = float(np.min(determinants))
            minimum_determinant = min(minimum_determinant, step_minimum)
            _require(
                step_minimum >= determinant_floor,
                "MuJoCo constrained replay violated its orientation threshold "
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
            constraint_residual = float(
                np.max(np.abs(np.asarray(data.efc_pos)[: 3 * len(targets)]))
            )
            maximum_constraint_residual = max(
                maximum_constraint_residual,
                constraint_residual,
            )
        positions[frame] = current

    deformation_determinants = replay_deformation_determinants_v1(
        geometry,
        positions,
    )
    return MujocoConstrainedFlexReplayV1(
        positions_m=np.ascontiguousarray(positions),
        deformation_determinants=np.ascontiguousarray(deformation_determinants),
        minimum_continuation_deformation_determinant=minimum_determinant,
        maximum_attachment_error_m=maximum_attachment_error,
        maximum_constraint_residual_m=maximum_constraint_residual,
        native_step_count=native_step_count,
        model_xml_sha256=scene.xml_sha256,
        material_vertex_count=len(geometry.points_m),
        tetrahedron_count=len(geometry.cells),
        constrained_vertex_count=len(geometry.attachment_indices),
        contact_patch_count=len(geometry.patch_node_indices),
        total_reference_mass_kg=scene.total_reference_mass_kg,
        constraint_time_constant_s=scene.constraint_time_constant_s,
        mass_uniform_blend_fraction=scene.mass_uniform_blend_fraction,
        conditioned_mass_condition_number=scene.conditioned_mass_condition_number,
    )


__all__ = [
    "ATTACHMENT_MODEL",
    "BACKEND_VARIANT",
    "CONSTRAINT_POLICY",
    "CONSTITUTIVE_MODEL",
    "MASS_CONDITION_NUMBER_CAP",
    "MASS_POLICY",
    "MujocoConstrainedFlexReplayV1",
    "MujocoConstrainedFlexSceneV1",
    "build_mujoco_constrained_flex_scene_v1",
    "condition_lumped_vertex_masses_v1",
    "run_mujoco_constrained_flex_source_replay_v1",
]
