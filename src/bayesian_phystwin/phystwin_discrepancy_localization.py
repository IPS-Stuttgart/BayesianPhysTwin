"""Information-matched localization of real PhysTwin discrepancy in Warp."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d.contracts import TwinBelief, load_contract
from causal4d.graph_temporal_discrepancy import graph_laplacian_basis
from causal4d.phystwin_backend import load_bayesian_phystwin_particles

from .dynamic_discrepancy import (
    LOCALIZATION_GRAPH_RANK,
    DynamicDiscrepancyCorrection,
    fit_dimensionless_linearized_correction,
    prefix_position_velocity_coefficients,
    scale_coefficients_to_field_limit,
    write_dynamic_discrepancy_correction,
)
from .observation_model_audit import (
    cross_view_residual_audit,
    metric_agreement_audit,
    released_observation_capability_audit,
)
from .phystwin_comparison import official_metrics_by_frame
from .phystwin_graph import PhysTwinSpringGraphConfig, build_phystwin_spring_graph
from .phystwin_residual_dynamics import _load_pickle, _sha256, _target_validity
from .phystwin_state_injection import (
    _initialize_simulator,
    _metric_summary,
    _released_self_collision_for_case,
    _state_numpy,
)
from .phystwin_structural_diagnostic import (
    _attachment_support_nodes,
    _far_graph_observation_error,
    _graph_distance,
    _horizon_summary,
    _object_rest_lengths,
    _set_simulator_arrays,
)


BASELINE = "bpt_particle_baseline"
READOUT = "graph_persistence_readout"
PREFIX_STATE = "prefix_state_position_velocity"
GENERALIZED_FORCE = "constant_generalized_force"
STRUCTURAL_CONTROL = "prefix_rest_geometry_control"
LOCALIZATION_METHODS = (
    BASELINE,
    READOUT,
    PREFIX_STATE,
    GENERALIZED_FORCE,
    STRUCTURAL_CONTROL,
)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("p,ptnc->tnc", weights, values)


def _set_particle(simulator, torch, wp, particle: np.ndarray, *, device: str) -> None:
    values = np.asarray(particle, dtype=np.float32)
    if values.shape != (2,):
        raise ValueError("localization particles must contain two spring log scales")
    with torch.no_grad():
        simulator.group_log_scale_tensor.copy_(
            torch.as_tensor(values, dtype=torch.float32, device=device)
        )
    wp.synchronize()


def _configure_rollout(
    simulator,
    torch,
    wp,
    *,
    rest_lengths_m: np.ndarray,
    controller_points_m: np.ndarray,
    external_forces_n: np.ndarray,
    device: str,
) -> None:
    _set_simulator_arrays(
        simulator,
        torch,
        wp,
        rest_lengths=rest_lengths_m,
        controller_points=controller_points_m,
        device=device,
    )
    simulator.set_external_forces(
        torch.as_tensor(
            external_forces_n,
            dtype=torch.float32,
            device=device,
        ).contiguous()
    )
    wp.synchronize()


def _rollout_state_segment(
    simulator,
    torch,
    wp,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the initial state and every subsequent official Warp frame."""

    position_tensor = torch.as_tensor(
        np.array(position_m, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    ).contiguous()
    velocity_tensor = torch.as_tensor(
        np.array(velocity_mps, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    ).contiguous()
    position_wp = wp.from_torch(position_tensor, dtype=wp.vec3, requires_grad=False)
    velocity_wp = wp.from_torch(velocity_tensor, dtype=wp.vec3, requires_grad=False)
    simulator.set_init_state(position_wp, velocity_wp)
    wp.synchronize()
    positions = [np.asarray(position_m, dtype=np.float32).copy()]
    velocities = [np.asarray(velocity_mps, dtype=np.float32).copy()]
    for frame in range(start_frame, stop_frame):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        wp.capture_launch(simulator.forward_graph)
        wp.synchronize()
        position, velocity = _state_numpy(simulator.wp_states[-1], wp)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise RuntimeError(f"non-finite Warp state at frame {frame}")
        positions.append(position)
        velocities.append(velocity)
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
        )
    return np.stack(positions), np.stack(velocities)


def _correction_response(
    simulator,
    torch,
    wp,
    *,
    endpoint_position_m: np.ndarray,
    endpoint_velocity_mps: np.ndarray,
    baseline_prefix_m: np.ndarray,
    train_end_frame: int,
    prefix_frame_count: int,
    structure_m: np.ndarray,
    graph_basis: np.ndarray,
    graph_springs: np.ndarray,
    nominal_rest_lengths_m: np.ndarray,
    num_object_springs: int,
    controller_points_m: np.ndarray,
    force_step_n: float,
    structural_step_m: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build prefix-only force and rest-geometry Warp responses."""

    object_count = len(structure_m)
    parameter_count = LOCALIZATION_GRAPH_RANK * 3
    force_response = np.empty(
        (*baseline_prefix_m.shape, parameter_count), dtype=np.float32
    )
    structural_response = np.empty_like(force_response)
    zero_force = np.zeros((object_count, 3), dtype=np.float32)
    stop_frame = train_end_frame + prefix_frame_count - 1
    for mode in range(LOCALIZATION_GRAPH_RANK):
        for coordinate in range(3):
            parameter = 3 * mode + coordinate
            force = np.zeros((object_count, 3), dtype=np.float32)
            force[:, coordinate] = force_step_n * graph_basis[:, mode]
            _configure_rollout(
                simulator,
                torch,
                wp,
                rest_lengths_m=nominal_rest_lengths_m,
                controller_points_m=controller_points_m,
                external_forces_n=force,
                device=device,
            )
            trajectory, _ = _rollout_state_segment(
                simulator,
                torch,
                wp,
                endpoint_position_m,
                endpoint_velocity_mps,
                start_frame=train_end_frame,
                stop_frame=stop_frame,
                device=device,
            )
            force_response[..., parameter] = trajectory - baseline_prefix_m

            rest_field = np.zeros_like(structure_m, dtype=np.float32)
            rest_field[:, coordinate] = structural_step_m * graph_basis[:, mode]
            perturbed_positions = structure_m + rest_field
            perturbed_lengths = _object_rest_lengths(
                perturbed_positions,
                graph_springs,
                nominal_rest_lengths_m,
                num_object_springs=num_object_springs,
            )
            _configure_rollout(
                simulator,
                torch,
                wp,
                rest_lengths_m=perturbed_lengths,
                controller_points_m=controller_points_m,
                external_forces_n=zero_force,
                device=device,
            )
            trajectory, _ = _rollout_state_segment(
                simulator,
                torch,
                wp,
                endpoint_position_m,
                endpoint_velocity_mps,
                start_frame=train_end_frame,
                stop_frame=stop_frame,
                device=device,
            )
            structural_response[..., parameter] = trajectory - baseline_prefix_m
    return force_response, structural_response


def _particle_rollouts(
    simulator,
    torch,
    wp,
    *,
    particles: np.ndarray,
    endpoint_positions_m: np.ndarray,
    endpoint_velocities_mps: np.ndarray,
    train_end_frame: int,
    frame_count: int,
    rest_lengths_m: np.ndarray,
    controller_points_m: np.ndarray,
    external_forces_n: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    velocities = []
    for particle, endpoint_position, endpoint_velocity in zip(
        particles,
        endpoint_positions_m,
        endpoint_velocities_mps,
        strict=True,
    ):
        _set_particle(simulator, torch, wp, particle, device=device)
        _configure_rollout(
            simulator,
            torch,
            wp,
            rest_lengths_m=rest_lengths_m,
            controller_points_m=controller_points_m,
            external_forces_n=external_forces_n,
            device=device,
        )
        trajectory, velocity = _rollout_state_segment(
            simulator,
            torch,
            wp,
            endpoint_position,
            endpoint_velocity,
            start_frame=train_end_frame,
            stop_frame=frame_count,
            device=device,
        )
        positions.append(trajectory)
        velocities.append(velocity)
    return np.stack(positions), np.stack(velocities)


def _prefix_state_rollouts(
    simulator,
    torch,
    wp,
    *,
    particles: np.ndarray,
    baseline_positions_m: np.ndarray,
    baseline_velocities_mps: np.ndarray,
    prefix_frame_count: int,
    heldout_start_frame: int,
    frame_count: int,
    position_field_m: np.ndarray,
    velocity_field_mps: np.ndarray,
    rest_lengths_m: np.ndarray,
    controller_points_m: np.ndarray,
    device: str,
) -> np.ndarray:
    zero_force = np.zeros_like(position_field_m)
    candidates = baseline_positions_m.copy()
    prefix_endpoint = prefix_frame_count - 1
    for particle_index, particle in enumerate(particles):
        _set_particle(simulator, torch, wp, particle, device=device)
        _configure_rollout(
            simulator,
            torch,
            wp,
            rest_lengths_m=rest_lengths_m,
            controller_points_m=controller_points_m,
            external_forces_n=zero_force,
            device=device,
        )
        continuation, _ = _rollout_state_segment(
            simulator,
            torch,
            wp,
            baseline_positions_m[particle_index, prefix_endpoint] + position_field_m,
            baseline_velocities_mps[particle_index, prefix_endpoint]
            + velocity_field_mps,
            start_frame=heldout_start_frame,
            stop_frame=frame_count,
            device=device,
        )
        candidates[particle_index, prefix_endpoint:] = continuation
    return candidates


def _coverage_summary(
    particles_m: np.ndarray,
    weights: np.ndarray,
    truth_m: np.ndarray,
    valid: np.ndarray,
    *,
    start_frame: int,
    variance_floor_m2: float,
) -> dict[str, float | int]:
    mean = _weighted_mean(particles_m, weights)
    centered = particles_m - mean[None]
    variance = np.einsum("p,ptnc->tnc", weights, np.square(centered))
    variance += variance_floor_m2
    selected = np.asarray(valid, dtype=bool).copy()
    selected[:start_frame] = False
    coordinate_selected = np.repeat(selected[:, :, None], 3, axis=2)
    residual = mean - truth_m
    z = 1.6448536269514722
    coverage = np.abs(residual) <= z * np.sqrt(variance)
    return {
        "coordinate_coverage_90": float(np.mean(coverage[coordinate_selected])),
        "coordinate_nees": float(
            np.mean(np.square(residual[coordinate_selected]) / variance[coordinate_selected])
        ),
        "mean_interval_width_m": float(
            np.mean((2.0 * z * np.sqrt(variance))[coordinate_selected])
        ),
        "valid_coordinate_count": int(np.sum(coordinate_selected)),
    }


def _field_energy(
    field: np.ndarray,
    springs: np.ndarray,
    *,
    unit: str,
) -> dict[str, Any]:
    values = np.asarray(field, dtype=float)
    edges = np.asarray(springs, dtype=np.int64)
    differences = values[edges[:, 0]] - values[edges[:, 1]]
    return {
        "unit": unit,
        "node_vector_rms": float(
            np.sqrt(np.mean(np.sum(np.square(values), axis=1)))
        ),
        "node_vector_maximum": float(
            np.max(np.linalg.norm(values, axis=1), initial=0.0)
        ),
        "edge_difference_rms": float(
            np.sqrt(np.mean(np.sum(np.square(differences), axis=1)))
        ),
    }


def _rest_strain(
    rest_positions_m: np.ndarray,
    structural_field_m: np.ndarray,
    springs: np.ndarray,
    rest_lengths_m: np.ndarray,
    *,
    num_object_springs: int,
) -> dict[str, float]:
    corrected = _object_rest_lengths(
        rest_positions_m + structural_field_m,
        springs,
        rest_lengths_m,
        num_object_springs=num_object_springs,
    )
    strain = (
        corrected[:num_object_springs] - rest_lengths_m[:num_object_springs]
    ) / rest_lengths_m[:num_object_springs]
    absolute = np.abs(strain)
    return {
        "maximum_absolute_edge_strain": float(np.max(absolute, initial=0.0)),
        "p95_absolute_edge_strain": float(np.percentile(absolute, 95.0)),
        "rms_edge_strain": float(np.sqrt(np.mean(np.square(strain)))),
    }


def _observation_audit(
    data: Mapping[str, Any],
    baseline_prefix_m: np.ndarray,
    graph_basis: np.ndarray,
    *,
    prefix_global_start: int,
    prefix_global_stop: int,
    ridge: float,
) -> dict[str, Any]:
    capability = released_observation_capability_audit(data)
    cross_view = capability["cross_view_residual_fields"]
    if not cross_view["available"]:
        return {"capabilities": capability, "cross_view": cross_view}
    observed = np.asarray(data[str(cross_view["point_key"])], dtype=float)
    valid = np.asarray(data[str(cross_view["valid_key"])], dtype=bool)
    if observed.ndim != 4:
        return {
            "capabilities": capability,
            "cross_view": {
                "available": False,
                "reason": "per-view array does not have shape (view, frame, node, 3)",
            },
        }
    audit = cross_view_residual_audit(
        observed[:, prefix_global_start:prefix_global_stop],
        valid[:, prefix_global_start:prefix_global_stop],
        baseline_prefix_m[:, : observed.shape[2]],
        graph_basis,
        ridge=ridge,
    )
    return {"capabilities": capability, "cross_view": audit}


def evaluate_phystwin_discrepancy_localization_case(
    official_repo: str | Path,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    parameter_profile_path: str | Path,
    twin_belief_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    o_plus_prefix_frames: int = 6,
    parameter_particle_count: int = 4,
    projection_ridge: float = 1e-5,
    dimensionless_ridge: float = 1e-4,
    force_step_n: float = 0.01,
    structural_step_m: float = 0.002,
    maximum_position_correction_m: float = 0.03,
    maximum_velocity_correction_mps: float = 0.50,
    maximum_force_per_node_n: float = 0.50,
    maximum_structural_correction_m: float = 0.01,
    variance_floor_m2: float = 2.5e-5,
    dt: float = 5e-5,
    num_substeps: int = 667,
    deterministic_spring_forces: bool = True,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Compare readout, state, force, and structure with one O-plus prefix."""

    if o_plus_prefix_frames != 6 or parameter_particle_count != 4:
        raise ValueError("the frozen diagnostic requires six O-plus frames and four particles")
    positive = (
        projection_ridge,
        dimensionless_ridge,
        force_step_n,
        structural_step_m,
        maximum_position_correction_m,
        maximum_velocity_correction_mps,
        maximum_force_per_node_n,
        maximum_structural_correction_m,
        variance_floor_m2,
        dt,
    )
    if any(value <= 0.0 or not np.isfinite(value) for value in positive):
        raise ValueError("localization scales and regularization must be positive")
    if num_substeps < 1 or not deterministic_spring_forces:
        raise ValueError("localization requires deterministic positive-substep Warp")

    data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_params_path)
    released = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controller = np.asarray(data["controller_points"], dtype=float)
    surface = np.asarray(data["surface_points"], dtype=float)
    interior = np.asarray(data["interior_points"], dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    frame_count, original_count, _ = observed.shape
    endpoint_frame = train_end_frame - 1
    prefix_frame_count = o_plus_prefix_frames + 1
    heldout_start_frame = train_end_frame + o_plus_prefix_frames
    prefix_global_stop = heldout_start_frame
    if not 2 <= train_end_frame < heldout_start_frame < frame_count:
        raise ValueError("training and prefix split must leave an untouched future")
    if released.shape[0] < frame_count:
        raise ValueError("released trajectory has too few frames")
    released = released[:frame_count]
    structure = np.concatenate((observed[0], surface, interior), axis=0)
    if released.shape[1:] != structure.shape:
        raise ValueError("released state and reconstructed structure disagree")
    graph = build_phystwin_spring_graph(
        structure,
        controller[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    object_springs = graph.springs[: graph.num_object_springs]
    graph_basis, graph_eigenvalues = graph_laplacian_basis(
        len(structure), object_springs, rank=LOCALIZATION_GRAPH_RANK
    )
    belief = load_contract(twin_belief_path)
    if not isinstance(belief, TwinBelief):
        raise TypeError("twin_belief_path must contain a TwinBelief")
    particles = load_bayesian_phystwin_particles(
        parameter_profile_path,
        maximum_count=parameter_particle_count,
    )
    if belief.context.case_id != Path(final_data_path).resolve().parent.name:
        raise ValueError("TwinBelief case differs from released data")
    if belief.endpoint_frame != endpoint_frame:
        raise ValueError("TwinBelief endpoint differs from train_end_frame")
    if belief.endpoint_position_m.shape != (
        parameter_particle_count,
        len(structure),
        3,
    ):
        raise ValueError("TwinBelief endpoint state differs from the PhysTwin graph")
    if not np.array_equal(belief.theta, particles.log_scales):
        raise ValueError("TwinBelief particles differ from the parameter profile")
    if not np.allclose(belief.weights, particles.weights, rtol=0.0, atol=1e-15):
        raise ValueError("TwinBelief weights differ from the parameter profile")

    valid = _target_validity(visible, motion_valid)
    self_collision = _released_self_collision_for_case(
        Path(final_data_path).resolve().parent.name
    )
    simulator, torch, wp, _ = _initialize_simulator(
        official_repo,
        data,
        optimal,
        checkpoint_path,
        graph,
        num_surface_points=original_count + len(surface),
        original_count=original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=True,
        spring_parameterization="grouped",
        device=device,
    )
    zero_force = np.zeros((len(structure), 3), dtype=np.float32)
    baseline_positions, baseline_velocities = _particle_rollouts(
        simulator,
        torch,
        wp,
        particles=particles.log_scales,
        endpoint_positions_m=belief.endpoint_position_m,
        endpoint_velocities_mps=belief.endpoint_velocity_mps,
        train_end_frame=train_end_frame,
        frame_count=frame_count,
        rest_lengths_m=graph.rest_lengths,
        controller_points_m=controller,
        external_forces_n=zero_force,
        device=device,
    )
    reference_particle = int(
        np.argmin(
            np.linalg.norm(
                particles.log_scales
                - np.einsum("p,pc->c", particles.weights, particles.log_scales)[None],
                axis=1,
            )
        )
    )
    _set_particle(
        simulator,
        torch,
        wp,
        particles.log_scales[reference_particle],
        device=device,
    )
    _configure_rollout(
        simulator,
        torch,
        wp,
        rest_lengths_m=graph.rest_lengths,
        controller_points_m=controller,
        external_forces_n=zero_force,
        device=device,
    )
    explicit_zero_positions, _ = _rollout_state_segment(
        simulator,
        torch,
        wp,
        belief.endpoint_position_m[reference_particle],
        belief.endpoint_velocity_mps[reference_particle],
        start_frame=train_end_frame,
        stop_frame=frame_count,
        device=device,
    )
    zero_force_parity = {
        "bitwise_identical": bool(
            np.array_equal(
                explicit_zero_positions,
                baseline_positions[reference_particle],
            )
        ),
        "reference_particle_index": reference_particle,
        "maximum_absolute_difference_m": float(
            np.max(
                np.abs(
                    explicit_zero_positions
                    - baseline_positions[reference_particle]
                ),
                initial=0.0,
            )
        ),
    }
    if not zero_force_parity["bitwise_identical"]:
        raise AssertionError("explicit zero force changed the official Warp rollout")

    baseline_mean = _weighted_mean(baseline_positions, particles.weights)
    truth_relative = observed[endpoint_frame:, :original_count]
    valid_relative = valid[endpoint_frame:, :original_count]
    prefix_residual = (
        truth_relative[:prefix_frame_count]
        - baseline_mean[:prefix_frame_count, :original_count]
    )
    prefix_valid = valid_relative[:prefix_frame_count]
    frame_dt_s = dt * num_substeps
    position_coefficients, velocity_coefficients, coefficient_history = (
        prefix_position_velocity_coefficients(
            prefix_residual,
            prefix_valid,
            graph_basis,
            frame_dt_s=frame_dt_s,
            ridge=projection_ridge,
        )
    )
    position_coefficients, position_limit = scale_coefficients_to_field_limit(
        graph_basis,
        position_coefficients,
        maximum_node_norm=maximum_position_correction_m,
    )
    velocity_coefficients, velocity_limit = scale_coefficients_to_field_limit(
        graph_basis,
        velocity_coefficients,
        maximum_node_norm=maximum_velocity_correction_mps,
    )

    prefix_reference = baseline_positions[
        reference_particle, :prefix_frame_count
    ]
    force_response, structural_response = _correction_response(
        simulator,
        torch,
        wp,
        endpoint_position_m=belief.endpoint_position_m[reference_particle],
        endpoint_velocity_mps=belief.endpoint_velocity_mps[reference_particle],
        baseline_prefix_m=prefix_reference,
        train_end_frame=train_end_frame,
        prefix_frame_count=prefix_frame_count,
        structure_m=structure,
        graph_basis=graph_basis,
        graph_springs=graph.springs,
        nominal_rest_lengths_m=graph.rest_lengths,
        num_object_springs=graph.num_object_springs,
        controller_points_m=controller,
        force_step_n=force_step_n,
        structural_step_m=structural_step_m,
        device=device,
    )
    force_dimensionless, force_fit = fit_dimensionless_linearized_correction(
        prefix_residual[1:],
        prefix_valid[1:],
        force_response[1:, :original_count],
        ridge=dimensionless_ridge,
    )
    structural_dimensionless, structural_fit = (
        fit_dimensionless_linearized_correction(
            prefix_residual[1:],
            prefix_valid[1:],
            structural_response[1:, :original_count],
            ridge=dimensionless_ridge,
        )
    )
    force_coefficients = force_step_n * force_dimensionless.reshape(
        LOCALIZATION_GRAPH_RANK, 3
    )
    structural_coefficients = structural_step_m * structural_dimensionless.reshape(
        LOCALIZATION_GRAPH_RANK, 3
    )
    force_coefficients, force_limit = scale_coefficients_to_field_limit(
        graph_basis,
        force_coefficients,
        maximum_node_norm=maximum_force_per_node_n,
    )
    structural_coefficients, structural_limit = scale_coefficients_to_field_limit(
        graph_basis,
        structural_coefficients,
        maximum_node_norm=maximum_structural_correction_m,
    )
    source_checksums = {
        "baseline_trajectory": _sha256(baseline_trajectory_path),
        "checkpoint": _sha256(checkpoint_path),
        "final_data": _sha256(final_data_path),
        "optimal_params": _sha256(optimal_params_path),
        "parameter_profile": _sha256(parameter_profile_path),
        "twin_belief": _sha256(twin_belief_path),
    }
    correction = DynamicDiscrepancyCorrection(
        case_id=belief.context.case_id,
        graph_basis=graph_basis,
        graph_eigenvalues=graph_eigenvalues,
        position_coefficients_m=position_coefficients,
        velocity_coefficients_mps=velocity_coefficients,
        generalized_force_coefficients_n=force_coefficients,
        structural_coefficients_m=structural_coefficients,
        prefix_frame_start=endpoint_frame,
        prefix_frame_stop=heldout_start_frame,
        frame_dt_s=frame_dt_s,
        information_boundary={
            "o_plus_prefix_frames": o_plus_prefix_frames,
            "prefix_includes_o_minus_endpoint": True,
            "future_frames_used_for_fit_or_selection": False,
            "manual_tracks_used_for_fit_or_selection": False,
            "graph_rank": LOCALIZATION_GRAPH_RANK,
            "released_case_role": "diagnostic_only_repeatedly_examined",
        },
        regularization={
            "selection": "fixed_before_all_released_case_runs",
            "projection_ridge": projection_ridge,
            "dimensionless_linear_response_ridge": dimensionless_ridge,
            "same_rule_for_force_and_structure": True,
            "holdout_based_selection": False,
        },
        source_checksums=source_checksums,
        diagnostics={
            "position_limit": position_limit,
            "velocity_limit": velocity_limit,
            "force_limit": force_limit,
            "structural_limit": structural_limit,
            "force_linearized_fit": force_fit,
            "structural_linearized_fit": structural_fit,
            "reference_particle_index": reference_particle,
        },
        metadata={
            "experiment": "phystwin_discrepancy_localization_v1",
            "constant_force_step_n": force_step_n,
            "structural_step_m": structural_step_m,
            "structural_branch_role": "information_matched_negative_control",
        },
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_record = write_dynamic_discrepancy_correction(
        output / "dynamic_discrepancy_correction", correction
    )

    position_field = correction.position_field_m()
    velocity_field = correction.velocity_field_mps()
    force_field = correction.generalized_force_field_n()
    structural_field = correction.structural_field_m()
    readout_particles = baseline_positions.copy()
    readout_particles[:, prefix_frame_count:] += position_field[None, None]
    state_particles = _prefix_state_rollouts(
        simulator,
        torch,
        wp,
        particles=particles.log_scales,
        baseline_positions_m=baseline_positions,
        baseline_velocities_mps=baseline_velocities,
        prefix_frame_count=prefix_frame_count,
        heldout_start_frame=heldout_start_frame,
        frame_count=frame_count,
        position_field_m=position_field,
        velocity_field_mps=velocity_field,
        rest_lengths_m=graph.rest_lengths,
        controller_points_m=controller,
        device=device,
    )
    force_particles, _ = _particle_rollouts(
        simulator,
        torch,
        wp,
        particles=particles.log_scales,
        endpoint_positions_m=belief.endpoint_position_m,
        endpoint_velocities_mps=belief.endpoint_velocity_mps,
        train_end_frame=train_end_frame,
        frame_count=frame_count,
        rest_lengths_m=graph.rest_lengths,
        controller_points_m=controller,
        external_forces_n=force_field,
        device=device,
    )
    corrected_rest_lengths = _object_rest_lengths(
        structure + structural_field,
        graph.springs,
        graph.rest_lengths,
        num_object_springs=graph.num_object_springs,
    )
    structural_particles, _ = _particle_rollouts(
        simulator,
        torch,
        wp,
        particles=particles.log_scales,
        endpoint_positions_m=belief.endpoint_position_m,
        endpoint_velocities_mps=belief.endpoint_velocity_mps,
        train_end_frame=train_end_frame,
        frame_count=frame_count,
        rest_lengths_m=corrected_rest_lengths,
        controller_points_m=controller,
        external_forces_n=zero_force,
        device=device,
    )
    particle_trajectories = {
        BASELINE: baseline_positions,
        READOUT: readout_particles,
        PREFIX_STATE: state_particles,
        GENERALIZED_FORCE: force_particles,
        STRUCTURAL_CONTROL: structural_particles,
    }
    global_trajectories = {}
    for method, values in particle_trajectories.items():
        trajectory = released.copy()
        trajectory[endpoint_frame:] = _weighted_mean(values, particles.weights)
        global_trajectories[method] = trajectory

    num_surface_points = original_count + len(surface)

    def metrics(trajectory: np.ndarray) -> dict[str, np.ndarray]:
        return official_metrics_by_frame(
            trajectory,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=heldout_start_frame,
            end_frame=frame_count,
        )

    baseline_metrics = metrics(global_trajectories[BASELINE])
    support_nodes = _attachment_support_nodes(graph, len(structure))
    graph_distance = _graph_distance(
        len(structure), object_springs, support_nodes
    )
    relative_heldout_start = prefix_frame_count
    energy = {
        BASELINE: {"kind": "none"},
        READOUT: {
            "kind": "observation_readout_position",
            **_field_energy(position_field, object_springs, unit="m"),
        },
        PREFIX_STATE: {
            "kind": "simulator_state_position_velocity",
            "position": _field_energy(position_field, object_springs, unit="m"),
            "velocity": _field_energy(velocity_field, object_springs, unit="m/s"),
            "kinetic_delta_proxy_j": float(
                0.5
                * np.sum(
                    graph.masses[: len(structure), None]
                    * (
                        2.0
                        * _weighted_mean(
                            baseline_velocities, particles.weights
                        )[prefix_frame_count - 1]
                        * velocity_field
                        + np.square(velocity_field)
                    )
                )
            ),
        },
        GENERALIZED_FORCE: {
            "kind": "constant_generalized_force",
            **_field_energy(force_field, object_springs, unit="N"),
        },
        STRUCTURAL_CONTROL: {
            "kind": "rest_geometry_negative_control",
            **_field_energy(structural_field, object_springs, unit="m"),
            **_rest_strain(
                structure,
                structural_field,
                graph.springs,
                graph.rest_lengths,
                num_object_springs=graph.num_object_springs,
            ),
        },
    }
    method_results = {}
    for method in LOCALIZATION_METHODS:
        candidate_metrics = metrics(global_trajectories[method])
        coverage = _coverage_summary(
            particle_trajectories[method][:, :, :original_count],
            particles.weights,
            truth_relative,
            valid_relative,
            start_frame=relative_heldout_start,
            variance_floor_m2=variance_floor_m2,
        )
        method_results[method] = {
            "future": _metric_summary(baseline_metrics, candidate_metrics),
            "horizon": _horizon_summary(baseline_metrics, candidate_metrics),
            "far_graph": _far_graph_observation_error(
                global_trajectories[method],
                observed,
                valid,
                graph_distance,
                start_frame=heldout_start_frame,
            ),
            "coverage": coverage,
            "residual_energy": energy[method],
            "metric_agreement": metric_agreement_audit(
                candidate_metrics["chamfer_distance_m"],
                candidate_metrics["track_error_m"],
            ),
        }

    observation_audit = _observation_audit(
        data,
        baseline_mean[:prefix_frame_count],
        graph_basis,
        prefix_global_start=endpoint_frame,
        prefix_global_stop=prefix_global_stop,
        ridge=projection_ridge,
    )
    archive_path = output / "localization_rollouts.npz"
    np.savez_compressed(
        archive_path,
        particle_weights=particles.weights,
        parameter_particles=particles.log_scales,
        coefficient_history=coefficient_history,
        force_response_prefix=force_response,
        structural_response_prefix=structural_response,
        **{
            f"particles__{method}": values.astype(np.float32)
            for method, values in particle_trajectories.items()
        },
        **{
            f"mean_global__{method}": values.astype(np.float32)
            for method, values in global_trajectories.items()
        },
    )
    summary = {
        "schema_version": 1,
        "experiment": "phystwin_discrepancy_localization_v1",
        "case": belief.context.case_id,
        "status": "released_case_diagnostic_not_confirmatory",
        "methods": method_results,
        "method_order": list(LOCALIZATION_METHODS),
        "information_boundary": correction.information_boundary,
        "comparison_contract": {
            "common_graph_basis": True,
            "graph_rank": LOCALIZATION_GRAPH_RANK,
            "common_o_plus_prefix": [endpoint_frame, heldout_start_frame],
            "common_heldout_continuation": [heldout_start_frame, frame_count],
            "common_physical_particle_count": parameter_particle_count,
            "common_particle_weights": particles.weights.tolist(),
            "common_controller_trajectory": True,
            "official_nonlinear_warp_rerun": True,
            "fixed_regularization_without_holdout_selection": True,
            "structural_control_interpretation": (
                "prefix-matched rest-length insertion diagnostic only; the prior "
                "equilibrium hierarchy remains rejected"
            ),
        },
        "zero_force_parity": zero_force_parity,
        "fit_diagnostics": correction.diagnostics,
        "observation_model_audit": observation_audit,
        "graph": {
            "object_vertex_count": len(structure),
            "object_spring_count": graph.num_object_springs,
            "controller_spring_count": len(graph.springs)
            - graph.num_object_springs,
            "springs_sha256": _array_sha256(graph.springs),
            "basis_sha256": _array_sha256(graph_basis),
            "basis_eigenvalues": graph_eigenvalues.tolist(),
        },
        "particles": {
            "count": parameter_particle_count,
            "weights": particles.weights.tolist(),
            "log_scales": particles.log_scales.tolist(),
            "reference_sensitivity_particle_index": reference_particle,
            "reference_sensitivity_particle_log_scales": particles.log_scales[
                reference_particle
            ].tolist(),
            "all_candidates_rerun_over_all_particles": True,
        },
        "config": {
            "train_end_frame": train_end_frame,
            "o_plus_prefix_frames": o_plus_prefix_frames,
            "projection_ridge": projection_ridge,
            "dimensionless_ridge": dimensionless_ridge,
            "force_step_n": force_step_n,
            "structural_step_m": structural_step_m,
            "maximum_position_correction_m": maximum_position_correction_m,
            "maximum_velocity_correction_mps": maximum_velocity_correction_mps,
            "maximum_force_per_node_n": maximum_force_per_node_n,
            "maximum_structural_correction_m": maximum_structural_correction_m,
            "variance_floor_m2": variance_floor_m2,
            "dt": dt,
            "num_substeps": num_substeps,
            "deterministic_spring_forces": True,
            "self_collision": self_collision,
            "device": device,
        },
        "artifacts": {
            "dynamic_discrepancy_correction": artifact_record,
            "localization_rollouts": {
                "path": str(archive_path.resolve()),
                "sha256": _sha256(archive_path),
            },
        },
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("final_data", final_data_path),
                ("baseline_trajectory", baseline_trajectory_path),
                ("optimal_params", optimal_params_path),
                ("checkpoint", checkpoint_path),
                ("parameter_profile", parameter_profile_path),
                ("twin_belief", twin_belief_path),
                ("gt_track", gt_track_path),
            )
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del simulator
    gc.collect()
    torch.cuda.empty_cache()
    return summary
