"""Optional Newton runtime for volumetric MPM source experiments.

This module intentionally depends on the optional ``mpm`` extra.  It keeps
the MPM material particles separate from benchmark query identities and uses
a finite-mass compliant contact projection instead of kinematic particles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import numpy.typing as npt
import warp as wp
from newton.solvers import SolverImplicitMPM

from .newton_mpm_volumetric_bridge_v2 import (
    MaterialContactMapV2,
    MaterialQueryMapV2,
    build_material_query_map,
    read_material_displacements,
    regular_convex_hull_particles,
    transfer_query_contacts_to_material,
)


@dataclass(frozen=True, slots=True)
class VolumetricMpmConfigV2:
    """Numerical and material settings for one fixed source candidate."""

    fps: float = 30.0
    substeps: int = 4
    voxel_size_m: float = 0.02
    particle_spacing_m: float = 0.01
    maximum_particle_count: int = 25_000
    density_kg_m3: float = 1000.0
    young_modulus_pa: float = 25_000.0
    poisson_ratio: float = 0.35
    damping: float = 0.002
    contact_coupling_per_frame: float = 0.35
    max_iterations: int = 50
    tolerance: float = 1.0e-5
    query_neighbour_count: int = 8
    query_inverse_distance_power: float = 2.0
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        positive = {
            "fps": self.fps,
            "voxel_size_m": self.voxel_size_m,
            "particle_spacing_m": self.particle_spacing_m,
            "density_kg_m3": self.density_kg_m3,
            "young_modulus_pa": self.young_modulus_pa,
            "tolerance": self.tolerance,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.substeps < 1 or isinstance(self.substeps, bool):
            raise ValueError("substeps must be a positive integer")
        if self.maximum_particle_count < 4 or isinstance(
            self.maximum_particle_count, bool
        ):
            raise ValueError("maximum_particle_count must be at least four")
        if self.max_iterations < 1 or isinstance(self.max_iterations, bool):
            raise ValueError("max_iterations must be a positive integer")
        if self.query_neighbour_count < 1 or isinstance(
            self.query_neighbour_count, bool
        ):
            raise ValueError("query_neighbour_count must be a positive integer")
        if not np.isfinite(self.query_inverse_distance_power) or (
            self.query_inverse_distance_power <= 0.0
        ):
            raise ValueError("query_inverse_distance_power must be positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and nonnegative")
        if not np.isfinite(self.contact_coupling_per_frame) or not (
            0.0 < self.contact_coupling_per_frame <= 1.0
        ):
            raise ValueError("contact_coupling_per_frame must be in (0, 1]")
        gravity = np.asarray(self.gravity_m_s2, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_m_s2 must contain three finite values")


@dataclass(frozen=True, slots=True)
class VolumetricMpmRolloutV2:
    """Material trajectory and fixed-identity readout from one MPM run."""

    material_rest_points_m: npt.NDArray[np.float32]
    material_trajectory_m: npt.NDArray[np.float32]
    query_trajectory_m: npt.NDArray[np.float32]
    query_map: MaterialQueryMapV2
    contact_map: MaterialContactMapV2


@wp.kernel  # type: ignore[untyped-decorator]
def _project_compliant_contacts(  # pragma: no cover - exercised on CUDA
    indices: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    target_positions: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    target_velocities: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    coupling: float,
    output_positions: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_velocities: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_velocity_gradients: wp.array(  # type: ignore[valid-type]
        dtype=wp.mat33
    ),
) -> None:
    local_index = wp.tid()
    particle_index = indices[local_index]
    current_position = output_positions[particle_index]
    current_velocity = output_velocities[particle_index]
    output_positions[particle_index] = current_position + coupling * (
        target_positions[local_index] - current_position
    )
    output_velocities[particle_index] = current_velocity + coupling * (
        target_velocities[local_index] - current_velocity
    )
    output_velocity_gradients[particle_index] = wp.mat33(0.0)


def _points(value: object, *, name: str, minimum_count: int) -> npt.NDArray[np.float32]:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) < minimum_count:
        raise ValueError(f"{name} must have shape (N>={minimum_count}, 3)")
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _controllers(value: object) -> npt.NDArray[np.float32]:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2:] != (3,) or array.shape[0] < 2:
        raise ValueError("controller_points_m must have shape (T>=2, C, 3)")
    if array.shape[1] < 1 or array.dtype.kind not in "iuf":
        raise ValueError("controller_points_m must contain controller points")
    if not np.all(np.isfinite(array)):
        raise ValueError("controller_points_m must contain finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _substep_endpoint_fraction(substep: int, substeps: int) -> float:
    return (substep + 1) / substeps


def _build_model(
    material_points_m: npt.NDArray[np.float32],
    config: VolumetricMpmConfigV2,
    *,
    device: str,
) -> Any:  # pragma: no cover - exercised on CUDA
    particle_mass = config.density_kg_m3 * config.particle_spacing_m**3
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[tuple(float(component) for component in row) for row in material_points_m],
        vel=[(0.0, 0.0, 0.0)] * len(material_points_m),
        mass=[particle_mass] * len(material_points_m),
        radius=[0.5 * config.particle_spacing_m] * len(material_points_m),
    )
    model = builder.finalize(device=device)
    model.set_gravity(config.gravity_m_s2)
    model.mpm.young_modulus.fill_(config.young_modulus_pa)
    model.mpm.poisson_ratio.fill_(config.poisson_ratio)
    model.mpm.damping.fill_(config.damping)
    model.mpm.tensile_yield_ratio.fill_(1.0)
    return model


def _solver(
    model: Any,
    config: VolumetricMpmConfigV2,
) -> SolverImplicitMPM:  # pragma: no cover - exercised on CUDA
    return SolverImplicitMPM(
        model,
        config=SolverImplicitMPM.Config(
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            solver="cr",
            warmstart_mode="particles",
            voxel_size=config.voxel_size_m,
            grid_type="sparse",
            integration_scheme="pic",
            strain_basis="P1d",
            velocity_basis="Q1",
        ),
    )


def _simulate_material_v2(  # pragma: no cover - exercised on CUDA
    material: npt.NDArray[np.float32],
    controllers: npt.NDArray[np.float32],
    contact_map: MaterialContactMapV2,
    config: VolumetricMpmConfigV2,
    *,
    driven: bool,
    device: str,
) -> npt.NDArray[np.float32]:
    selected_device = wp.get_device(device)
    with wp.ScopedDevice(selected_device):
        model = _build_model(material, config, device=device)
        solver = _solver(model, config)
        state_0 = model.state()
        state_1 = model.state()
        contact_indices = np.ascontiguousarray(
            contact_map.material_indices,
            dtype=np.int32,
        )
        contact_device = wp.array(contact_indices, dtype=wp.int32, device=device)
        contact_rest = material[contact_indices]
        controller_rest = controllers[0]
        frame_dt = 1.0 / config.fps
        simulation_dt = frame_dt / config.substeps
        substep_coupling = 1.0 - (1.0 - config.contact_coupling_per_frame) ** (
            1.0 / config.substeps
        )
        trajectory: npt.NDArray[np.float32] = np.empty(
            (len(controllers), len(material), 3),
            dtype=np.float32,
        )
        trajectory[0] = material

        for frame_index in range(1, len(controllers)):
            previous = controllers[frame_index - 1]
            following = controllers[frame_index]
            if driven:
                controller_velocity = (following - previous) / frame_dt
            else:
                controller_velocity = np.zeros_like(previous)
            target_velocity = np.ascontiguousarray(
                contact_map.controller_weights @ controller_velocity,
                dtype=np.float32,
            )
            for substep in range(config.substeps):
                if driven:
                    alpha = _substep_endpoint_fraction(substep, config.substeps)
                    controller_position = previous + alpha * (following - previous)
                    displacement = controller_position - controller_rest
                else:
                    displacement = np.zeros_like(controller_rest)
                target_position = np.ascontiguousarray(
                    contact_rest + contact_map.controller_weights @ displacement,
                    dtype=np.float32,
                )
                wp.launch(
                    _project_compliant_contacts,
                    dim=len(contact_indices),
                    inputs=[
                        contact_device,
                        wp.array(target_position, dtype=wp.vec3, device=device),
                        wp.array(target_velocity, dtype=wp.vec3, device=device),
                        substep_coupling,
                        state_0.particle_q,
                        state_0.particle_qd,
                        state_0.mpm.particle_qd_grad,
                    ],
                    device=device,
                )
                solver.step(state_0, state_1, None, None, simulation_dt)
                state_0, state_1 = state_1, state_0
            trajectory[frame_index] = state_0.particle_q.numpy()
    return trajectory


def simulate_volumetric_mpm_v2(
    *,
    query_rest_points_m: object,
    controller_points_m: object,
    attached_query_indices: object,
    query_controller_weights: object,
    config: VolumetricMpmConfigV2,
    driven: bool,
    device: str = "cuda:0",
) -> VolumetricMpmRolloutV2:
    """Simulate one action and return a benchmark-identity displacement readout."""

    if not isinstance(config, VolumetricMpmConfigV2):
        raise TypeError("config must be VolumetricMpmConfigV2")
    if not isinstance(driven, bool):
        raise TypeError("driven must be bool")
    config.validate()
    queries = _points(query_rest_points_m, name="query_rest_points_m", minimum_count=4)
    controllers = _controllers(controller_points_m)
    material64 = regular_convex_hull_particles(
        queries,
        spacing_m=config.particle_spacing_m,
        maximum_particle_count=config.maximum_particle_count,
    )
    material = np.ascontiguousarray(material64, dtype=np.float32)
    query_map = build_material_query_map(
        queries,
        material,
        neighbour_count=config.query_neighbour_count,
        inverse_distance_power=config.query_inverse_distance_power,
    )
    contact_map = transfer_query_contacts_to_material(
        attached_query_indices,
        query_controller_weights,
        query_map,
        material_particle_count=len(material),
    )

    trajectory = _simulate_material_v2(
        material,
        controllers,
        contact_map,
        config,
        driven=driven,
        device=device,
    )
    if not np.all(np.isfinite(trajectory)):
        raise RuntimeError("Newton MPM generated non-finite volumetric particles")
    query_trajectory = read_material_displacements(
        trajectory,
        material,
        queries,
        query_map,
    )
    return VolumetricMpmRolloutV2(
        material_rest_points_m=material,
        material_trajectory_m=np.ascontiguousarray(trajectory),
        query_trajectory_m=np.ascontiguousarray(query_trajectory, dtype=np.float32),
        query_map=query_map,
        contact_map=contact_map,
    )


__all__ = [
    "VolumetricMpmConfigV2",
    "VolumetricMpmRolloutV2",
    "simulate_volumetric_mpm_v2",
]
