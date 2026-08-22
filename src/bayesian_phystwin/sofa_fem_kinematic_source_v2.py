"""SOFA hyperelastic replay with native keyed moving Dirichlet constraints.

Each attached material vertex owns one ``LinearMovementProjectiveConstraint``.
Its complete key schedule is derived from the registered rigid-patch action
before the scene is initialized. SOFA therefore projects the material degrees
of freedom directly, without a mutable target state or post-step correction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

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
from .sofa_fem_source_v1 import (
    CONSTITUTIVE_MODEL,
    NativeSofaFemModulesV1,
    stable_neo_hookean_lame_parameters_v1,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

BACKEND_VARIANT = "sofa-stable-neo-hookean-keyed-dirichlet-v2"
ATTACHMENT_MODEL = "LinearMovementProjectiveConstraint-per-vertex-keyed-Dirichlet-v2"
CONTINUATION_POLICY = "registered-rigid-patch-substep-key-schedule-v2"


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


def _array_identity(digest: Any, array: npt.ArrayLike) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.tobytes(order="C"))


@dataclass(frozen=True, slots=True)
class SofaKinematicScheduleV2:
    """Complete native key schedule for every registered attachment vertex."""

    key_times_s: FloatArray
    attached_targets_m: FloatArray
    relative_movements_m: FloatArray
    frame_step_indices: IntArray
    schedule_sha256: str


def build_sofa_kinematic_schedule_v2(
    geometry: NativeTetSourceGeometryV1,
    contact: RigidContactProjectionV1,
    *,
    driven: bool,
    integrator_time_step_s: float,
    interval_substeps: int,
) -> SofaKinematicScheduleV2:
    """Derive all substep keys from the registered rigid-patch trajectory."""

    time_step = _finite(
        integrator_time_step_s,
        name="integrator_time_step_s",
        positive=True,
    )
    _require(
        type(interval_substeps) is int and interval_substeps >= 1,
        "interval_substeps must be a positive integer",
    )
    frame_count = int(contact.rotations.shape[0])
    total_steps = (frame_count - 1) * interval_substeps
    targets: FloatArray = np.empty(
        (total_steps + 1, len(geometry.attachment_indices), 3),
        dtype=np.float64,
    )
    targets[0] = geometry.points_m[geometry.attachment_indices]
    step = 0
    for frame in range(1, frame_count):
        for substep in range(1, interval_substeps + 1):
            step += 1
            rotations, translations = contact_transform_at_fraction_v1(
                contact,
                previous_frame=frame - 1,
                target_frame=frame,
                fraction=substep / interval_substeps,
                driven=driven,
            )
            targets[step] = attached_targets_from_transform_v1(
                geometry,
                contact,
                rotations=rotations,
                translations_m=translations,
            )
    key_times = np.arange(total_steps + 1, dtype=np.float64) * time_step
    frame_steps: IntArray = np.arange(frame_count, dtype=np.int64) * interval_substeps
    target_relative = targets - geometry.points_m[geometry.attachment_indices][None]
    relative = np.concatenate((target_relative[1:], target_relative[-1:]), axis=0)
    digest = hashlib.sha256()
    for array in (key_times, targets, relative, frame_steps):
        _array_identity(digest, array)
    return SofaKinematicScheduleV2(
        key_times_s=np.ascontiguousarray(key_times),
        attached_targets_m=np.ascontiguousarray(targets),
        relative_movements_m=np.ascontiguousarray(relative),
        frame_step_indices=np.ascontiguousarray(frame_steps),
        schedule_sha256=digest.hexdigest(),
    )


def _scene_identity(
    *,
    geometry: NativeTetSourceGeometryV1,
    schedule: SofaKinematicScheduleV2,
    parameters: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    for array in (
        geometry.points_m,
        geometry.cells,
        geometry.attachment_indices,
        schedule.key_times_s,
        schedule.relative_movements_m,
    ):
        _array_identity(digest, array)
    for key, value in sorted(parameters.items()):
        digest.update(key.encode())
        digest.update(repr(value).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SofaKinematicSourceReplayV2:
    """Native keyed-Dirichlet trajectory and numerical diagnostics."""

    positions_m: FloatArray
    deformation_determinants: FloatArray
    minimum_continuation_deformation_determinant: float
    maximum_attachment_error_m: float
    native_step_count: int
    scene_sha256: str
    schedule_sha256: str
    material_vertex_count: int
    tetrahedron_count: int
    attachment_count: int
    total_reference_mass_kg: float


def run_sofa_fem_kinematic_source_replay_v2(
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
) -> SofaKinematicSourceReplayV2:  # pragma: no cover - exact native runtime
    """Run one source replay through native keyed SOFA constraints."""

    time_step = _finite(
        integrator_time_step_s,
        name="integrator_time_step_s",
        positive=True,
    )
    density = _finite(density_kg_m3, name="density_kg_m3", positive=True)
    stiffness_damping = _finite(rayleigh_stiffness, name="rayleigh_stiffness")
    mass_damping = _finite(rayleigh_mass, name="rayleigh_mass")
    _require(
        stiffness_damping >= 0.0 and mass_damping >= 0.0,
        "Rayleigh damping values must be non-negative",
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
    schedule = build_sofa_kinematic_schedule_v2(
        geometry,
        contact,
        driven=driven,
        integrator_time_step_s=time_step,
        interval_substeps=interval_substeps,
    )
    shear, first_lame = stable_neo_hookean_lame_parameters_v1(
        young_modulus_pa,
        poisson_ratio,
    )
    _, total_mass = tetrahedral_nodal_masses_v1(
        geometry,
        density_kg_m3=density,
    )
    parameters: dict[str, object] = {
        "attachment_model": ATTACHMENT_MODEL,
        "backend_variant": BACKEND_VARIANT,
        "constitutive_model": CONSTITUTIVE_MODEL,
        "continuation_policy": CONTINUATION_POLICY,
        "density_kg_m3": density,
        "first_lame_pa": first_lame,
        "integrator_time_step_s": time_step,
        "rayleigh_mass": mass_damping,
        "rayleigh_stiffness": stiffness_damping,
        "shear_modulus_pa": shear,
    }
    scene_sha256 = _scene_identity(
        geometry=geometry,
        schedule=schedule,
        parameters=parameters,
    )

    sofa = native.sofa
    root = sofa.Core.Node("root")
    initialized = False
    try:
        root.dt = time_step
        root.gravity = [0.0, 0.0, 0.0]
        root.addObject("DefaultAnimationLoop")
        body = root.addChild("body")
        body.addObject(
            "EulerImplicitSolver",
            rayleighStiffness=stiffness_damping,
            rayleighMass=mass_damping,
        )
        linear_solver = body.addObject(
            "SparseLDLSolver",
            template="CompressedRowSparseMatrixMat3x3d",
        )
        topology = body.addObject(
            "TetrahedronSetTopologyContainer",
            name="topology",
            position=geometry.points_m.tolist(),
            tetrahedra=geometry.cells.tolist(),
        )
        mechanical = body.addObject(
            "MechanicalObject",
            name="dofs",
            template="Vec3d",
            position=geometry.points_m.tolist(),
        )
        mass = body.addObject(
            "MeshMatrixMass",
            name="mass",
            massDensity=density,
        )
        force_field = body.addObject(
            "TetrahedronHyperelasticityFEMForceField",
            name="stable_neo_hookean_fem",
            materialName="StableNeoHookean",
            ParameterSet=f"{shear:.17g} {first_lame:.17g}",
        )
        constraints: list[Any] = []
        key_times = schedule.key_times_s.tolist()
        inactive_movements = np.zeros_like(schedule.relative_movements_m[:, 0]).tolist()
        for node_index in geometry.attachment_indices:
            constraints.append(
                body.addObject(
                    "LinearMovementProjectiveConstraint",
                    name=f"moving_dirichlet_{int(node_index)}",
                    indices=[int(node_index)],
                    keyTimes=key_times,
                    movements=inactive_movements,
                    relativeMovements=True,
                    topology="@topology",
                )
            )
        sofa.Simulation.init(root)
        initialized = True
        for component in (linear_solver, mass, force_field, *constraints):
            state = str(component.componentState.value)
            _require(
                state == "Valid",
                f"SOFA v2 source component {component.getName()} initialized {state}",
            )
        _require(
            len(np.asarray(topology.tetrahedra.value)) == len(geometry.cells)
            and np.asarray(mechanical.position.value).shape == geometry.points_m.shape,
            "SOFA v2 source topology or mechanical-state roster changed",
        )

        frame_count = int(contact.rotations.shape[0])
        positions: FloatArray = np.empty(
            (frame_count, len(geometry.points_m), 3),
            dtype=np.float64,
        )
        positions[0] = np.asarray(mechanical.position.value, dtype=np.float64)
        _require(
            np.allclose(positions[0], geometry.points_m, atol=1e-12, rtol=0.0),
            "SOFA v2 initial material state changed",
        )
        for local_index, constraint in enumerate(constraints):
            movements = schedule.relative_movements_m[:, local_index]
            constraint.movements.value = movements.tolist()
            _require(
                np.array_equal(
                    np.asarray(constraint.movements.value, dtype=np.float64),
                    movements,
                ),
                "SOFA v2 native key schedule activation changed",
            )
        minimum_determinant = 1.0
        maximum_attachment_error = 0.0
        total_steps = int(schedule.frame_step_indices[-1])
        for step in range(1, total_steps + 1):
            previous_time = float(root.time.value)
            sofa.Simulation.animate(root, time_step)
            current = np.asarray(mechanical.position.value, dtype=np.float64).copy()
            _require(
                float(root.time.value) > previous_time,
                "SOFA v2 failed to advance source-scene time",
            )
            _require(
                np.all(np.isfinite(current))
                and np.all(
                    np.isfinite(np.asarray(mechanical.velocity.value, dtype=np.float64))
                ),
                "SOFA v2 source replay produced non-finite state",
            )
            determinants = replay_deformation_determinants_v1(
                geometry,
                current[None],
            )[0]
            step_minimum = float(np.min(determinants))
            minimum_determinant = min(minimum_determinant, step_minimum)
            _require(
                step_minimum >= determinant_floor,
                "SOFA v2 source replay violated its hard orientation threshold "
                f"at step={step}, minimum_determinant={step_minimum:.17g}, "
                f"required_minimum={determinant_floor:.17g}",
            )
            attachment_error = float(
                np.max(
                    np.linalg.norm(
                        current[geometry.attachment_indices]
                        - schedule.attached_targets_m[step],
                        axis=1,
                    )
                )
            )
            maximum_attachment_error = max(
                maximum_attachment_error,
                attachment_error,
            )
            _require(
                attachment_error <= 1.0e-12,
                "SOFA v2 native moving Dirichlet projection changed "
                f"at step={step}, attachment_error_m={attachment_error:.17g}",
            )
            if step % interval_substeps == 0:
                positions[step // interval_substeps] = current

        deformation_determinants = replay_deformation_determinants_v1(
            geometry,
            positions,
        )
        return SofaKinematicSourceReplayV2(
            positions_m=np.ascontiguousarray(positions),
            deformation_determinants=np.ascontiguousarray(deformation_determinants),
            minimum_continuation_deformation_determinant=minimum_determinant,
            maximum_attachment_error_m=maximum_attachment_error,
            native_step_count=total_steps,
            scene_sha256=scene_sha256,
            schedule_sha256=schedule.schedule_sha256,
            material_vertex_count=len(geometry.points_m),
            tetrahedron_count=len(geometry.cells),
            attachment_count=len(geometry.attachment_indices),
            total_reference_mass_kg=total_mass,
        )
    finally:
        if initialized:
            sofa.Simulation.unload(root)


__all__ = [
    "ATTACHMENT_MODEL",
    "BACKEND_VARIANT",
    "CONTINUATION_POLICY",
    "SofaKinematicScheduleV2",
    "SofaKinematicSourceReplayV2",
    "build_sofa_kinematic_schedule_v2",
    "run_sofa_fem_kinematic_source_replay_v2",
]
