"""Outcome-blind source-grid runtime for volumetric Newton MPM v2."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import newton
import numpy as np
import numpy.typing as npt
import scipy
import warp as wp

from ._newton_mpm_source_runtime import (
    _candidate_directory,
    _candidate_parameters,
    _implementation_provenance,
    _physical_arrays,
)
from ._newton_mpm_volumetric_runtime_v2 import (
    VolumetricMpmConfigV2,
    VolumetricMpmRolloutV2,
    simulate_volumetric_mpm_v2,
)
from ._portable_contracts import content_id, write_atomic_json
from .newton_mpm_backend_v1 import file_sha256
from .newton_mpm_source_gate_v1 import (
    GRID_MANIFEST_FILENAME,
    GRID_SCHEMA,
    SourceProtocol,
    load_source_inputs,
    load_source_protocol,
)
from .physical_rollout_v1 import write_deterministic_npz


def _simulation(protocol: SourceProtocol) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], protocol.value["simulation"])


def _simulation_number(
    simulation: Mapping[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> float:
    value = simulation.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"simulation.{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"simulation.{name} must be a finite number")
    return result


def _simulation_integer(
    simulation: Mapping[str, Any],
    name: str,
    *,
    minimum: int = 1,
) -> int:
    value = simulation.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"simulation.{name} must be an integer >= {minimum}")
    return value


def _require_frozen_solver(simulation: Mapping[str, Any]) -> None:
    expected = {
        "engine": "newton-implicit-mpm-volumetric-v2",
        "solver": "cr",
        "integration_scheme": "pic",
        "strain_basis": "P1d",
        "velocity_basis": "Q1",
        "grid_type": "sparse",
        "particleization": "regular-convex-hull-v2",
        "readout": "inverse-distance-material-displacement-v2",
        "contact": "finite-mass-compliant-projection-v2",
    }
    for name, value in expected.items():
        if simulation.get(name) != value:
            raise ValueError(f"simulation.{name} differs from volumetric v2")


def _volumetric_config(
    protocol: SourceProtocol,
    *,
    young_modulus_pa: float,
    damping: float,
) -> VolumetricMpmConfigV2:
    simulation = _simulation(protocol)
    _require_frozen_solver(simulation)
    gravity = simulation.get("gravity_m_s2")
    if (
        not isinstance(gravity, list)
        or len(gravity) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in gravity
        )
    ):
        raise ValueError("simulation.gravity_m_s2 must contain three numbers")
    config = VolumetricMpmConfigV2(
        fps=_simulation_number(simulation, "fps", positive=True),
        substeps=_simulation_integer(simulation, "substeps"),
        voxel_size_m=_simulation_number(
            simulation,
            "voxel_size_m",
            positive=True,
        ),
        particle_spacing_m=_simulation_number(
            simulation,
            "particle_spacing_m",
            positive=True,
        ),
        maximum_particle_count=_simulation_integer(
            simulation,
            "maximum_particle_count",
            minimum=4,
        ),
        density_kg_m3=_simulation_number(
            simulation,
            "density_kg_m3",
            positive=True,
        ),
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=_simulation_number(simulation, "poisson_ratio"),
        damping=damping,
        contact_coupling_per_frame=_simulation_number(
            simulation,
            "contact_coupling_per_frame",
            positive=True,
        ),
        max_iterations=_simulation_integer(simulation, "max_iterations"),
        tolerance=_simulation_number(simulation, "tolerance", positive=True),
        query_neighbour_count=_simulation_integer(
            simulation,
            "query_neighbour_count",
        ),
        query_inverse_distance_power=_simulation_number(
            simulation,
            "query_inverse_distance_power",
            positive=True,
        ),
        gravity_m_s2=(
            float(gravity[0]),
            float(gravity[1]),
            float(gravity[2]),
        ),
    )
    config.validate()
    return config


def _validate_rollout_contract(
    protocol: SourceProtocol,
    rollout: VolumetricMpmRolloutV2,
) -> None:
    simulation = _simulation(protocol)
    expected_material = _simulation_integer(
        simulation,
        "expected_internal_material_particle_count",
        minimum=4,
    )
    expected_contact = _simulation_integer(
        simulation,
        "expected_transferred_contact_particle_count",
    )
    if len(rollout.material_rest_points_m) != expected_material:
        raise RuntimeError("volumetric material particle count changed")
    if len(rollout.contact_map.material_indices) != expected_contact:
        raise RuntimeError("transferred contact particle count changed")
    expected_material_shape = (
        protocol.frame_count,
        expected_material,
        3,
    )
    expected_query_shape = (
        protocol.frame_count,
        protocol.material_count,
        3,
    )
    if rollout.material_trajectory_m.shape != expected_material_shape:
        raise RuntimeError("volumetric material trajectory shape changed")
    if rollout.query_trajectory_m.shape != expected_query_shape:
        raise RuntimeError("volumetric query readout shape changed")
    if not np.all(np.isfinite(rollout.query_trajectory_m)):
        raise RuntimeError("volumetric query readout is non-finite")


def _simulate_candidate(
    protocol: SourceProtocol,
    inputs: Mapping[str, npt.NDArray[Any]],
    *,
    young_modulus_pa: float,
    damping: float,
    driven: bool,
    device: str,
) -> VolumetricMpmRolloutV2:
    rollout = simulate_volumetric_mpm_v2(
        query_rest_points_m=inputs["frame_zero_points_m"],
        controller_points_m=inputs["controller_points_m"],
        attached_query_indices=inputs["attachment_indices"],
        query_controller_weights=inputs["attachment_weights"],
        config=_volumetric_config(
            protocol,
            young_modulus_pa=young_modulus_pa,
            damping=damping,
        ),
        driven=driven,
        device=device,
    )
    _validate_rollout_contract(protocol, rollout)
    return rollout


def _same_static_maps(
    first: VolumetricMpmRolloutV2,
    second: VolumetricMpmRolloutV2,
) -> bool:
    return bool(
        np.array_equal(first.material_rest_points_m, second.material_rest_points_m)
        and np.array_equal(first.query_map.indices, second.query_map.indices)
        and np.array_equal(first.query_map.weights, second.query_map.weights)
        and np.array_equal(
            first.contact_map.material_indices,
            second.contact_map.material_indices,
        )
        and np.array_equal(
            first.contact_map.controller_weights,
            second.contact_map.controller_weights,
        )
    )


def _normalized_action_support(
    driven: npt.NDArray[np.float32],
    zero: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    response = np.linalg.norm(
        driven.astype(np.float64) - zero.astype(np.float64),
        axis=2,
    )
    maximum_response = np.max(response, axis=0)
    normalization = float(np.max(maximum_response))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("volumetric candidate produced no action response")
    return np.ascontiguousarray(maximum_response / normalization, dtype=np.float32)


def run_volumetric_source_grid(
    *,
    protocol_path: str | Path,
    source_inputs_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the frozen volumetric candidate grid without accepting outcomes."""

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
                driven = _simulate_candidate(
                    protocol,
                    inputs,
                    young_modulus_pa=young,
                    damping=damping,
                    driven=True,
                    device=device,
                )
                zero = _simulate_candidate(
                    protocol,
                    inputs,
                    young_modulus_pa=young,
                    damping=damping,
                    driven=False,
                    device=device,
                )
                replay = _simulate_candidate(
                    protocol,
                    inputs,
                    young_modulus_pa=young,
                    damping=damping,
                    driven=True,
                    device=device,
                )
                if not (
                    _same_static_maps(driven, zero)
                    and _same_static_maps(driven, replay)
                ):
                    raise RuntimeError("volumetric particle/readout maps changed")
                driven_query = driven.query_trajectory_m
                zero_query = zero.query_trajectory_m
                replay_query = replay.query_trajectory_m
                support = _normalized_action_support(driven_query, zero_query)
                physical_path = candidate_dir / "physical-prediction.npz"
                write_deterministic_npz(
                    physical_path,
                    _physical_arrays(points, driven_query, zero_query, support),
                )
                replay_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                driven_query.astype(np.float64)
                                - replay_query.astype(np.float64)
                            )
                        )
                    )
                )
                zero_drift = float(
                    np.max(
                        np.linalg.norm(
                            zero_query.astype(np.float64)
                            - points.astype(np.float64)[None],
                            axis=2,
                        )
                    )
                )
                response = float(
                    np.max(
                        np.linalg.norm(
                            driven_query.astype(np.float64)
                            - zero_query.astype(np.float64),
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
                final_predictions.append(driven_query[-1])
            except Exception as error:  # noqa: BLE001 - retain frozen denominator
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
        per_query_spread = np.sqrt(np.sum(np.var(final_stack, axis=0), axis=1))
        final_spread = float(np.median(per_query_spread))
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


__all__ = ["run_volumetric_source_grid"]
