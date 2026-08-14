"""Optional Newton/Warp runtime for the synthetic MPM compatibility smoke."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from ._portable_contracts import content_id, write_atomic_json
from .newton_mpm_backend_v1 import (
    NEWTON_MPM_BACKEND_KIND,
    NEWTON_MPM_ENGINE_REPOSITORY,
    NEWTON_MPM_RUNTIME_SCHEMA,
    NEWTON_MPM_SCHEMA_VERSION,
    file_sha256,
    validate_newton_mpm_runtime_manifest,
)
from .physical_rollout_v1 import write_deterministic_npz
from .phystwin_online_belief import deterministic_farthest_point_ids


@dataclass(frozen=True, slots=True)
class NewtonMpmSmokeConfig:
    """Frozen parameters for a small, metrically scaled elastic beam."""

    frame_count: int = 76
    query_count: int = 64
    fps: float = 120.0
    substeps: int = 1
    voxel_size_m: float = 0.025
    density_kg_m3: float = 1000.0
    young_modulus_pa: float = 500_000.0
    poisson_ratio: float = 0.35
    damping: float = 0.002
    max_iterations: int = 50
    beam_length_m: float = 0.30
    beam_width_m: float = 0.05
    beam_height_m: float = 0.05
    action_displacement_m: float = 0.025

    def validate(self) -> None:
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least two")
        if self.query_count < 1:
            raise ValueError("query_count must be positive")
        if self.fps <= 0.0 or not np.isfinite(self.fps):
            raise ValueError("fps must be finite and positive")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        positive = {
            "voxel_size_m": self.voxel_size_m,
            "density_kg_m3": self.density_kg_m3,
            "young_modulus_pa": self.young_modulus_pa,
            "beam_length_m": self.beam_length_m,
            "beam_width_m": self.beam_width_m,
            "beam_height_m": self.beam_height_m,
            "action_displacement_m": self.action_displacement_m,
        }
        for name, value in positive.items():
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if self.damping < 0.0 or not np.isfinite(self.damping):
            raise ValueError("damping must be finite and nonnegative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


@wp.kernel
def _translate_kinematic_particles(
    indices: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    rest_positions: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_positions: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_velocities: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_velocity_gradients: wp.array(  # type: ignore[valid-type]
        dtype=wp.mat33
    ),
    displacement: wp.vec3,
    velocity: wp.vec3,
) -> None:
    local_index = wp.tid()
    particle_index = indices[local_index]
    output_positions[particle_index] = rest_positions[local_index] + displacement
    output_velocities[particle_index] = velocity
    output_velocity_gradients[particle_index] = wp.mat33(0.0)


def _build_model(
    config: NewtonMpmSmokeConfig,
    *,
    device: str,
) -> tuple[Any, np.ndarray, np.ndarray]:
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    extent = np.array(
        [config.beam_length_m, config.beam_width_m, config.beam_height_m],
        dtype=np.float64,
    )
    particles_per_cell = 2
    resolution = np.ceil(particles_per_cell * extent / config.voxel_size_m).astype(
        np.int64
    )
    cell_size = extent / (resolution + 1)
    cell_volume = float(np.prod(cell_size))
    particle_radius = float(np.cbrt(cell_volume) * 0.5)
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(float),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=int(resolution[0] + 1),
        dim_y=int(resolution[1] + 1),
        dim_z=int(resolution[2] + 1),
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=cell_volume * config.density_kg_m3,
        jitter=0.0,
        radius_mean=particle_radius,
    )
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.mpm.young_modulus.fill_(config.young_modulus_pa)
    model.mpm.poisson_ratio.fill_(config.poisson_ratio)
    model.mpm.damping.fill_(config.damping)
    model.mpm.tensile_yield_ratio.fill_(1.0)

    frame_zero = np.asarray(model.particle_q.numpy()).copy()
    half_voxel = 0.5 * config.voxel_size_m
    fixed_mask = frame_zero[:, 0] < half_voxel
    driven_mask = frame_zero[:, 0] > config.beam_length_m - half_voxel
    constrained = np.flatnonzero(np.logical_or(fixed_mask, driven_mask))
    if not len(constrained) or not np.any(driven_mask):
        raise RuntimeError("beam discretization did not create boundary particles")
    constrained_device = wp.array(constrained, dtype=wp.int32, device=device)
    model.particle_mass[constrained_device].fill_(0.0)
    return model, frame_zero, np.flatnonzero(driven_mask).astype(np.int32)


def _simulate_one(
    config: NewtonMpmSmokeConfig,
    *,
    device: str,
    driven_action: bool,
) -> np.ndarray:
    model, frame_zero, driven_indices = _build_model(config, device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver_config = SolverImplicitMPM.Config(
        max_iterations=config.max_iterations,
        tolerance=1.0e-5,
        solver="cr",
        warmstart_mode="particles",
        voxel_size=config.voxel_size_m,
        grid_type="sparse",
        integration_scheme="pic",
        strain_basis="P1d",
        velocity_basis="Q1",
    )
    solver = SolverImplicitMPM(model, config=solver_config)
    indices_device = wp.array(driven_indices, dtype=wp.int32, device=device)
    rest_device = wp.array(
        frame_zero[driven_indices],
        dtype=wp.vec3,
        device=device,
    )
    frame_dt = 1.0 / config.fps
    simulation_dt = frame_dt / config.substeps
    trajectory = np.empty(
        (config.frame_count, model.particle_count, 3),
        dtype=np.float32,
    )
    trajectory[0] = frame_zero
    duration = (config.frame_count - 1) * frame_dt
    action_velocity = config.action_displacement_m / duration

    for frame_index in range(1, config.frame_count):
        if driven_action:
            # The solver advects a kinematic particle by one simulation step.
            # Supply the position at the beginning of the interval so the
            # stored state lands on the registered frame-end displacement.
            alpha = (frame_index - 1) / (config.frame_count - 1)
            displacement = wp.vec3(0.0, 0.0, config.action_displacement_m * alpha)
            velocity = wp.vec3(0.0, 0.0, action_velocity)
            wp.launch(
                _translate_kinematic_particles,
                dim=len(driven_indices),
                inputs=[
                    indices_device,
                    rest_device,
                    state_0.particle_q,
                    state_0.particle_qd,
                    state_0.mpm.particle_qd_grad,
                    displacement,
                    velocity,
                ],
                device=device,
            )
        for _ in range(config.substeps):
            solver.step(state_0, state_1, None, None, simulation_dt)
            state_0, state_1 = state_1, state_0
        trajectory[frame_index] = state_0.particle_q.numpy()
    if not np.all(np.isfinite(trajectory)):
        raise RuntimeError("Newton MPM generated non-finite particle positions")
    return trajectory


def run_newton_mpm_smoke(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    device: str = "cuda:0",
    config: NewtonMpmSmokeConfig | None = None,
) -> dict[str, Any]:
    """Run driven and zero-action beams and seal a raw particle artifact."""

    if config is None:
        smoke = NewtonMpmSmokeConfig()
    elif isinstance(config, NewtonMpmSmokeConfig):
        smoke = config
    else:
        raise TypeError("config must be a NewtonMpmSmokeConfig")
    smoke.validate()
    raw_path = Path(raw_rollout_path)
    runtime_path = Path(runtime_manifest_path)
    if raw_path.exists() or runtime_path.exists():
        raise FileExistsError("Newton smoke output already exists")
    selected_device = wp.get_device(device)
    with wp.ScopedDevice(selected_device):
        driven = _simulate_one(smoke, device=device, driven_action=True)
        zero = _simulate_one(smoke, device=device, driven_action=False)
    if not np.array_equal(driven[0], zero[0]):
        raise RuntimeError("driven and zero-action MPM runs differ at frame zero")
    query_count = min(smoke.query_count, driven.shape[1])
    query_indices = deterministic_farthest_point_ids(
        driven[0],
        np.arange(driven.shape[1], dtype=np.int64),
        query_count,
    )
    response = np.linalg.norm(
        driven[:, query_indices] - zero[:, query_indices],
        axis=2,
    )
    maximum_response = np.max(response, axis=0)
    normalization = float(np.max(maximum_response))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("driven MPM smoke produced no action-conditioned response")
    action_support = np.asarray(maximum_response / normalization, dtype=np.float32)
    raw_arrays = {
        "driven_particle_positions_m": np.ascontiguousarray(driven),
        "zero_action_particle_positions_m": np.ascontiguousarray(zero),
        "material_query_indices": np.ascontiguousarray(query_indices),
        "action_support": np.ascontiguousarray(action_support),
    }
    write_deterministic_npz(raw_path, raw_arrays)

    simulation = {
        "scene": "kinematic-beam-bend-v1",
        "beam_extents_m": [
            smoke.beam_length_m,
            smoke.beam_width_m,
            smoke.beam_height_m,
        ],
        "action_displacement_m": [0.0, 0.0, smoke.action_displacement_m],
        "gravity_m_s2": [0.0, 0.0, 0.0],
        "density_kg_m3": smoke.density_kg_m3,
        "young_modulus_pa": smoke.young_modulus_pa,
        "poisson_ratio": smoke.poisson_ratio,
        "damping": smoke.damping,
        "voxel_size_m": smoke.voxel_size_m,
        "substeps": smoke.substeps,
        "solver": "implicit-mpm-cr",
        "max_iterations": smoke.max_iterations,
    }
    boundary = {
        "synthetic_scene": True,
        "dataset_payload_read": False,
        "future_observations_read": False,
        "outcomes_read": False,
        "known_action_used": True,
    }
    identity = {
        "schema": NEWTON_MPM_RUNTIME_SCHEMA,
        "schema_version": NEWTON_MPM_SCHEMA_VERSION,
        "backend_kind": NEWTON_MPM_BACKEND_KIND,
        "engine_repository": NEWTON_MPM_ENGINE_REPOSITORY,
        "engine_version": newton.__version__,
        "warp_version": wp.__version__,
        "python_version": platform.python_version(),
        "device": str(selected_device.alias),
        "device_name": str(selected_device.name),
        "coordinate_frame": "right-handed-z-up-world-v1",
        "position_units": "m",
        "time_units": "s",
        "frame_count": smoke.frame_count,
        "particle_count": int(driven.shape[1]),
        "query_count": int(len(query_indices)),
        "time_step_s": 1.0 / smoke.fps,
        "simulation": simulation,
        "information_boundary": boundary,
        "raw_rollout_sha256": file_sha256(raw_path),
    }
    runtime = {**identity, "runtime_id": content_id(identity)}
    write_atomic_json(runtime, runtime_path, overwrite=False)
    validate_newton_mpm_runtime_manifest(runtime, raw_rollout_path=raw_path)
    return {
        "runtime": runtime,
        "config": asdict(smoke),
        "maximum_action_response_m": normalization,
    }
