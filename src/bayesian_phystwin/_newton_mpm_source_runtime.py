"""Optional Newton runtime for the frozen one-case source-development gate."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import newton
import numpy as np
import numpy.typing as npt
import scipy
import warp as wp
from newton.solvers import SolverImplicitMPM

from ._portable_contracts import content_id, write_atomic_json
from .newton_mpm_backend_v1 import file_sha256
from .newton_mpm_source_gate_v1 import (
    GRID_MANIFEST_FILENAME,
    GRID_SCHEMA,
    IMPLEMENTATION_SOURCE_PATHS,
    SourceProtocol,
    load_source_inputs,
    load_source_protocol,
)
from .physical_rollout_v1 import (
    validate_physical_rollout_arrays,
    write_deterministic_npz,
)


@wp.kernel
def _set_kinematic_targets(  # pragma: no cover - exercised by CUDA source run
    indices: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    positions: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    velocities: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_positions: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_velocities: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    output_velocity_gradients: wp.array(  # type: ignore[valid-type]
        dtype=wp.mat33
    ),
) -> None:
    local_index = wp.tid()
    particle_index = indices[local_index]
    output_positions[particle_index] = positions[local_index]
    output_velocities[particle_index] = velocities[local_index]
    output_velocity_gradients[particle_index] = wp.mat33(0.0)


def _simulation_value(
    protocol: SourceProtocol,
    name: str,
) -> Any:
    simulation = cast(Mapping[str, Any], protocol.value["simulation"])
    return simulation[name]


def _implementation_provenance(
    protocol: SourceProtocol | None = None,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot bind the Newton source implementation") from error
    if status:
        raise RuntimeError("Newton source prediction requires a clean Git worktree")
    source_paths = (
        IMPLEMENTATION_SOURCE_PATHS
        if protocol is None
        else protocol.implementation_source_paths
    )
    source_files = {
        relative: file_sha256(repository / relative)
        for relative in sorted(source_paths)
    }
    return {
        "git_head": head,
        "git_worktree_clean": True,
        "source_files": source_files,
    }


def _simulate_source(  # pragma: no cover - exercised by CUDA source run
    protocol: SourceProtocol,
    inputs: dict[str, npt.NDArray[Any]],
    *,
    young_modulus_pa: float,
    damping: float,
    driven: bool,
    device: str,
) -> npt.NDArray[np.float32]:
    points = np.asarray(inputs["frame_zero_points_m"], dtype=np.float32)
    controllers = np.asarray(inputs["controller_points_m"], dtype=np.float32)
    attachment_indices = np.asarray(inputs["attachment_indices"], dtype=np.int32)
    attachment_weights = np.asarray(inputs["attachment_weights"], dtype=np.float32)
    radius = float(_simulation_value(protocol, "particle_radius_m"))
    density = float(_simulation_value(protocol, "density_kg_m3"))
    particle_mass = density * (2.0 * radius) ** 3

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[tuple(float(component) for component in row) for row in points],
        vel=[(0.0, 0.0, 0.0)] * len(points),
        mass=[particle_mass] * len(points),
        radius=[radius] * len(points),
    )
    model = builder.finalize(device=device)
    gravity = cast(list[float], _simulation_value(protocol, "gravity_m_s2"))
    model.set_gravity(tuple(float(value) for value in gravity))
    model.mpm.young_modulus.fill_(young_modulus_pa)
    model.mpm.poisson_ratio.fill_(float(_simulation_value(protocol, "poisson_ratio")))
    model.mpm.damping.fill_(damping)
    model.mpm.tensile_yield_ratio.fill_(1.0)
    attachment_device = wp.array(
        attachment_indices,
        dtype=wp.int32,
        device=device,
    )
    model.particle_mass[attachment_device].fill_(0.0)
    solver = SolverImplicitMPM(
        model,
        config=SolverImplicitMPM.Config(
            max_iterations=int(_simulation_value(protocol, "max_iterations")),
            tolerance=float(_simulation_value(protocol, "tolerance")),
            solver=str(_simulation_value(protocol, "solver")),
            warmstart_mode="particles",
            voxel_size=float(_simulation_value(protocol, "voxel_size_m")),
            grid_type="sparse",
            integration_scheme=str(_simulation_value(protocol, "integration_scheme")),
            strain_basis=str(_simulation_value(protocol, "strain_basis")),
            velocity_basis=str(_simulation_value(protocol, "velocity_basis")),
        ),
    )
    state_0 = model.state()
    state_1 = model.state()
    fps = float(_simulation_value(protocol, "fps"))
    substeps = int(_simulation_value(protocol, "substeps"))
    frame_dt = 1.0 / fps
    simulation_dt = frame_dt / substeps
    trajectory: npt.NDArray[np.float32] = np.empty(
        (protocol.frame_count, protocol.material_count, 3),
        dtype=np.float32,
    )
    trajectory[0] = points
    attachment_rest = points[attachment_indices]
    controller_rest = controllers[0]
    current_target = attachment_rest.copy()

    for frame_index in range(1, protocol.frame_count):
        if driven:
            controller_displacement = controllers[frame_index] - controller_rest
            next_target = attachment_rest + attachment_weights @ controller_displacement
        else:
            next_target = attachment_rest
        velocity = (next_target - current_target) / frame_dt
        wp.launch(
            _set_kinematic_targets,
            dim=len(attachment_indices),
            inputs=[
                attachment_device,
                wp.array(current_target, dtype=wp.vec3, device=device),
                wp.array(velocity, dtype=wp.vec3, device=device),
                state_0.particle_q,
                state_0.particle_qd,
                state_0.mpm.particle_qd_grad,
            ],
            device=device,
        )
        for _ in range(substeps):
            solver.step(state_0, state_1, None, None, simulation_dt)
            state_0, state_1 = state_1, state_0
        trajectory[frame_index] = state_0.particle_q.numpy()
        current_target = next_target
    if not np.all(np.isfinite(trajectory)):
        raise RuntimeError("Newton MPM generated non-finite source particles")
    return trajectory


def _candidate_directory(index: int) -> str:
    return f"candidate-{index:02d}"


def _candidate_parameters(value: object) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise ValueError("candidate parameters must be a JSON object")
    young = float(value["young_modulus_pa"])
    damping = float(value["damping"])
    return young, damping


def _physical_arrays(
    points: npt.NDArray[np.float32],
    driven: npt.NDArray[np.float32],
    zero: npt.NDArray[np.float32],
    support: npt.NDArray[np.float32],
) -> dict[str, npt.NDArray[Any]]:
    persistence = np.repeat(points[None], len(driven), axis=0)
    arrays: dict[str, npt.NDArray[Any]] = {
        "prediction_m": np.ascontiguousarray(driven),
        "persistence_m": np.ascontiguousarray(persistence),
        "driven_readout_m": np.ascontiguousarray(driven),
        "zero_action_readout_m": np.ascontiguousarray(zero),
        "action_support": np.ascontiguousarray(support),
        "frame_zero_points_m": np.ascontiguousarray(points),
    }
    validate_physical_rollout_arrays(arrays, expected_frame_count=len(driven))
    return arrays


def run_source_grid(
    *,
    protocol_path: str | Path,
    source_inputs_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run and seal all frozen MPM candidates without accepting an outcome path."""

    protocol = load_source_protocol(protocol_path)
    inputs = load_source_inputs(source_inputs_path, protocol=protocol)
    implementation = _implementation_provenance(protocol)
    source_path = Path(source_inputs_path).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected_device = wp.get_device(device)
    candidates: list[dict[str, Any]] = []
    final_predictions: list[npt.NDArray[np.float32]] = []
    grid = cast(list[dict[str, Any]], protocol.value["parameter_grid"])
    points = np.asarray(inputs["frame_zero_points_m"], dtype=np.float32)
    support = np.asarray(inputs["action_support"], dtype=np.float32)

    with wp.ScopedDevice(selected_device):
        for index, parameter_value in enumerate(grid):
            young, damping = _candidate_parameters(parameter_value)
            relative_dir = _candidate_directory(index)
            candidate_dir = output / relative_dir
            candidate_dir.mkdir()
            record: dict[str, Any] = {
                "candidate_index": index,
                "young_modulus_pa": young,
                "damping": damping,
            }
            try:
                driven = _simulate_source(
                    protocol,
                    inputs,
                    young_modulus_pa=young,
                    damping=damping,
                    driven=True,
                    device=device,
                )
                zero = _simulate_source(
                    protocol,
                    inputs,
                    young_modulus_pa=young,
                    damping=damping,
                    driven=False,
                    device=device,
                )
                replay = _simulate_source(
                    protocol,
                    inputs,
                    young_modulus_pa=young,
                    damping=damping,
                    driven=True,
                    device=device,
                )
                physical_path = candidate_dir / "physical-prediction.npz"
                write_deterministic_npz(
                    physical_path,
                    _physical_arrays(points, driven, zero, support),
                )
                replay_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                driven.astype(np.float64) - replay.astype(np.float64)
                            )
                        )
                    )
                )
                zero_drift = float(
                    np.max(
                        np.linalg.norm(
                            zero.astype(np.float64) - points.astype(np.float64)[None],
                            axis=2,
                        )
                    )
                )
                response = float(
                    np.max(
                        np.linalg.norm(
                            driven.astype(np.float64) - zero.astype(np.float64),
                            axis=2,
                        )
                    )
                )
                record.update(
                    {
                        "status": "success",
                        "physical_archive": f"{relative_dir}/physical-prediction.npz",
                        "physical_archive_sha256": file_sha256(physical_path),
                        "replay_coordinate_rmse_m": replay_rmse,
                        "maximum_zero_action_drift_m": zero_drift,
                        "maximum_action_response_m": response,
                    }
                )
                final_predictions.append(driven[-1])
            except Exception as error:  # noqa: BLE001 - retain the frozen denominator
                record.update(
                    {
                        "status": "technical_failure",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            candidates.append(record)
    if len(final_predictions) >= 2:
        final_stack = np.stack(final_predictions).astype(np.float64)
        per_particle_spread = np.sqrt(np.sum(np.var(final_stack, axis=0), axis=1))
        final_spread = float(np.median(per_particle_spread))
    else:
        final_spread = 0.0
    identity: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "source_inputs_sha256": file_sha256(source_path),
        "runtime": {
            "engine_version": newton.__version__,
            "warp_version": wp.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "python_version": platform.python_version(),
            "device": str(selected_device.alias),
            "device_name": str(selected_device.name),
        },
        "implementation": implementation,
        "information_boundary": {
            "frame_zero_geometry_read": True,
            "known_full_controller_action_read": True,
            "object_outcome_artifact_read": False,
            "target_or_held_out_artifact_read": False,
        },
        "candidates": candidates,
        "successful_candidate_count": sum(
            record["status"] == "success" for record in candidates
        ),
        "technical_failure_count": sum(
            record["status"] != "success" for record in candidates
        ),
        "final_ensemble_spread_m": final_spread,
    }
    manifest = {**identity, "grid_id": content_id(identity)}
    write_atomic_json(manifest, output / GRID_MANIFEST_FILENAME, overwrite=False)
    return manifest


__all__ = ["run_source_grid"]
