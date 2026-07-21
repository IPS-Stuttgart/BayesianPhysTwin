"""Official-Warp diagnostic for guarded action-propagated state belief updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d.contracts import TwinBelief, load_contract
from causal4d.phystwin_backend import load_bayesian_phystwin_particles

from .dynamic_discrepancy import (
    LOCALIZATION_GRAPH_RANK,
    load_dynamic_discrepancy_correction,
)
from .phystwin_comparison import official_metrics_by_frame
from .phystwin_discrepancy_localization import (
    BASELINE,
    READOUT,
    _configure_rollout,
    _particle_rollouts,
    _rollout_state_segment,
    _set_particle,
    _weighted_mean,
)
from .phystwin_graph import PhysTwinSpringGraphConfig, build_phystwin_spring_graph
from .phystwin_residual_dynamics import _load_pickle, _sha256, _target_validity
from .phystwin_state_injection import (
    _initialize_simulator,
    _metric_summary,
    _released_self_collision_for_case,
)
from .phystwin_structural_diagnostic import (
    _attachment_support_nodes,
    _far_graph_observation_error,
    _graph_distance,
    _horizon_summary,
)
from .propagated_state_belief import (
    PropagatedStateBeliefConfig,
    infer_propagated_state_belief,
)
from .propagated_state_correction import (
    PropagatedStateCorrection,
    PropagatedStateSelectionConfig,
    decode_limited_state_weights,
    modal_state_parameter_fields,
    scale_posterior_covariance_for_state_limits,
    select_propagated_state_update,
    write_propagated_state_correction,
)


RAW_PROPAGATED = "action_propagated_state_plus_bias_raw"
GUARDED_PROPAGATED = "action_propagated_state_plus_bias_guarded"
PROPAGATED_METHODS = (BASELINE, READOUT, RAW_PROPAGATED, GUARDED_PROPAGATED)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _localization_input_path(summary: Mapping[str, Any], name: str) -> Path:
    record = summary["inputs"][name]
    path = Path(str(record["path"]))
    _require(path.is_file(), f"localization input is missing: {name}")
    _require(_sha256(path) == record["sha256"], f"localization input changed: {name}")
    return path


def _state_response_rollouts(
    simulator,
    torch,
    wp,
    *,
    endpoint_position_m: np.ndarray,
    endpoint_velocity_mps: np.ndarray,
    baseline_m: np.ndarray,
    position_parameter_fields_m: np.ndarray,
    velocity_parameter_fields_mps: np.ndarray,
    start_frame: int,
    stop_frame: int,
    rest_lengths_m: np.ndarray,
    controller_points_m: np.ndarray,
    device: str,
) -> np.ndarray:
    """Evaluate one-sided official-Warp responses for state perturbations."""

    parameter_count = (
        position_parameter_fields_m.shape[2] + velocity_parameter_fields_mps.shape[2]
    )
    response = np.empty((*baseline_m.shape, parameter_count), dtype=np.float32)
    zero_force = np.zeros_like(endpoint_position_m, dtype=np.float32)
    for parameter in range(parameter_count):
        position = np.asarray(endpoint_position_m, dtype=np.float32).copy()
        velocity = np.asarray(endpoint_velocity_mps, dtype=np.float32).copy()
        if parameter < position_parameter_fields_m.shape[2]:
            position += position_parameter_fields_m[:, :, parameter]
        else:
            velocity += velocity_parameter_fields_mps[
                :, :, parameter - position_parameter_fields_m.shape[2]
            ]
        _configure_rollout(
            simulator,
            torch,
            wp,
            rest_lengths_m=rest_lengths_m,
            controller_points_m=controller_points_m,
            external_forces_n=zero_force,
            device=device,
        )
        trajectory, _ = _rollout_state_segment(
            simulator,
            torch,
            wp,
            position,
            velocity,
            start_frame=start_frame,
            stop_frame=stop_frame,
            device=device,
        )
        response[..., parameter] = trajectory - baseline_m
    return response


def _prediction_variance_m2(
    state_response_at_step_m: np.ndarray,
    graph_basis: np.ndarray,
    posterior_covariance: np.ndarray,
) -> np.ndarray:
    """Propagate the raw local Gaussian coefficient covariance to readout space."""

    response = np.asarray(state_response_at_step_m, dtype=np.float64)
    basis = np.asarray(graph_basis, dtype=np.float64)
    covariance = np.asarray(posterior_covariance, dtype=np.float64)
    state_count = response.shape[3]
    bias_count = basis.shape[1]
    _require(
        covariance.shape == (state_count + 3 * bias_count,) * 2,
        "posterior covariance shape changed",
    )
    state_covariance = covariance[:state_count, :state_count]
    variance = np.einsum(
        "tnck,kl,tncl->tnc",
        response,
        state_covariance,
        response,
        optimize=True,
    )
    for coordinate in range(3):
        bias_slice = slice(
            state_count + coordinate * bias_count,
            state_count + (coordinate + 1) * bias_count,
        )
        bias_covariance = covariance[bias_slice, bias_slice]
        bias_variance = np.einsum(
            "nb,bd,nd->n", basis, bias_covariance, basis, optimize=True
        )
        cross_covariance = covariance[:state_count, bias_slice]
        cross_variance = 2.0 * np.einsum(
            "tnk,kb,nb->tn",
            response[:, :, coordinate],
            cross_covariance,
            basis,
            optimize=True,
        )
        variance[:, :, coordinate] += bias_variance[None] + cross_variance
    return np.maximum(variance, 0.0)


def _coverage_summary(
    particles_m: np.ndarray,
    weights: np.ndarray,
    truth_m: np.ndarray,
    valid: np.ndarray,
    *,
    start_frame: int,
    variance_floor_m2: float,
    coefficient_variance_m2: np.ndarray | None = None,
) -> dict[str, Any]:
    mean = _weighted_mean(particles_m, weights)
    centered = particles_m - mean[None]
    variance = np.einsum("p,ptnc->tnc", weights, np.square(centered))
    variance += variance_floor_m2
    if coefficient_variance_m2 is not None:
        supplied = np.asarray(coefficient_variance_m2, dtype=np.float64)
        _require(supplied.shape == variance.shape, "coefficient variance shape changed")
        variance += supplied
    selected = np.asarray(valid, dtype=bool).copy()
    selected[:start_frame] = False
    coordinate_selected = np.repeat(selected[:, :, None], 3, axis=2)
    residual = mean - truth_m
    z = 1.6448536269514722
    covered = np.abs(residual) <= z * np.sqrt(variance)
    return {
        "coordinate_coverage_90": float(np.mean(covered[coordinate_selected])),
        "coordinate_nees": float(
            np.mean(
                np.square(residual[coordinate_selected]) / variance[coordinate_selected]
            )
        ),
        "mean_interval_width_m": float(
            np.mean((2.0 * z * np.sqrt(variance))[coordinate_selected])
        ),
        "valid_coordinate_count": int(np.sum(coordinate_selected)),
        "coefficient_covariance_included": coefficient_variance_m2 is not None,
        "coefficient_covariance_calibrated": False,
    }


def _global_trajectory(
    frozen_global_baseline: np.ndarray,
    relative_particles: np.ndarray,
    particle_weights: np.ndarray,
    *,
    endpoint_frame: int,
) -> np.ndarray:
    result = np.asarray(frozen_global_baseline, dtype=np.float32).copy()
    result[endpoint_frame:] = _weighted_mean(relative_particles, particle_weights)
    return result


def evaluate_guarded_propagated_state_case(
    official_repo: str | Path,
    localization_case_dir: str | Path,
    output_dir: str | Path,
    *,
    position_step_m: float = 0.005,
    velocity_step_mps: float = 0.05,
    state_weight_prior_std: float = 4.0,
    observation_std_m: float = 0.005,
    shared_bias_prior_std_m: float = 0.020,
    maximum_position_update_m: float = 0.05,
    maximum_velocity_update_mps: float = 0.25,
    fit_frame_count: int = 4,
    minimum_validation_improvement_fraction: float = 0.05,
    minimum_validation_improvement_m: float = 0.00025,
    variance_floor_m2: float = 2.5e-5,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run one exhausted-case implementation diagnostic with exact fallback."""

    localization = Path(localization_case_dir)
    localization_summary_path = localization / "summary.json"
    localization_archive_path = localization / "localization_rollouts.npz"
    localization_correction_path = localization / "dynamic_discrepancy_correction.json"
    for path in (
        localization_summary_path,
        localization_archive_path,
        localization_correction_path,
    ):
        _require(path.is_file(), f"frozen localization input is missing: {path.name}")
    localization_summary = json.loads(
        localization_summary_path.read_text(encoding="utf-8")
    )
    _require(
        localization_summary["experiment"] == "phystwin_discrepancy_localization_v1",
        "input is not the frozen localization experiment",
    )
    frozen_correction = load_dynamic_discrepancy_correction(
        localization_correction_path
    )
    case_id = str(localization_summary["case"])
    _require(frozen_correction.case_id == case_id, "frozen correction case changed")
    config = dict(localization_summary["config"])
    _require(config["o_plus_prefix_frames"] == 6, "prefix length is not frozen")
    _require(
        config["deterministic_spring_forces"] is True, "Warp path is not deterministic"
    )
    train_end_frame = int(config["train_end_frame"])
    endpoint_frame = train_end_frame - 1
    prefix_frame_count = int(config["o_plus_prefix_frames"]) + 1
    heldout_start_frame = train_end_frame + int(config["o_plus_prefix_frames"])

    final_data_path = _localization_input_path(localization_summary, "final_data")
    baseline_path = _localization_input_path(
        localization_summary, "baseline_trajectory"
    )
    optimal_path = _localization_input_path(localization_summary, "optimal_params")
    checkpoint_path = _localization_input_path(localization_summary, "checkpoint")
    parameter_profile_path = _localization_input_path(
        localization_summary, "parameter_profile"
    )
    twin_belief_path = _localization_input_path(localization_summary, "twin_belief")
    gt_track_path = _localization_input_path(localization_summary, "gt_track")
    data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_path)
    released = np.asarray(_load_pickle(baseline_path), dtype=np.float32)
    observed = np.asarray(data["object_points"], dtype=np.float64)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controller = np.asarray(data["controller_points"], dtype=np.float64)
    surface = np.asarray(data["surface_points"], dtype=np.float64)
    interior = np.asarray(data["interior_points"], dtype=np.float64)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=np.float64)
    frame_count, original_count, _ = observed.shape
    _require(heldout_start_frame < frame_count, "prefix consumes the forecast")
    structure = np.concatenate((observed[0], surface, interior), axis=0)
    _require(released.shape[1:] == structure.shape, "released state shape changed")
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
    graph_basis = frozen_correction.graph_basis
    _require(
        graph_basis.shape == (len(structure), LOCALIZATION_GRAPH_RANK),
        "frozen graph basis shape changed",
    )
    belief = load_contract(twin_belief_path)
    _require(isinstance(belief, TwinBelief), "twin belief has the wrong contract")
    particles = load_bayesian_phystwin_particles(
        parameter_profile_path,
        maximum_count=int(localization_summary["particles"]["count"]),
    )
    _require(belief.context.case_id == case_id, "twin belief case changed")
    _require(belief.endpoint_frame == endpoint_frame, "twin belief endpoint changed")
    _require(
        np.array_equal(belief.theta, particles.log_scales),
        "particle support changed",
    )

    with np.load(localization_archive_path, allow_pickle=False) as archive:
        frozen_baseline_particles = np.asarray(
            archive[f"particles__{BASELINE}"], dtype=np.float32
        )
        frozen_readout_particles = np.asarray(
            archive[f"particles__{READOUT}"], dtype=np.float32
        )
        frozen_global_baseline = np.asarray(
            archive[f"mean_global__{BASELINE}"], dtype=np.float32
        )
        frozen_global_readout = np.asarray(
            archive[f"mean_global__{READOUT}"], dtype=np.float32
        )
        archived_weights = np.asarray(archive["particle_weights"], dtype=np.float64)
        archived_particles = np.asarray(
            archive["parameter_particles"], dtype=np.float64
        )
    _require(
        np.array_equal(archived_particles, particles.log_scales)
        and np.allclose(archived_weights, particles.weights, rtol=0.0, atol=1e-15),
        "frozen localization particle mixture changed",
    )

    simulator, torch, wp, _ = _initialize_simulator(
        official_repo,
        data,
        optimal,
        checkpoint_path,
        graph,
        num_surface_points=original_count + len(surface),
        original_count=original_count,
        dt=float(config["dt"]),
        num_substeps=int(config["num_substeps"]),
        self_collision=_released_self_collision_for_case(case_id),
        deterministic_spring_forces=True,
        spring_parameterization="grouped",
        device=device,
    )
    zero_force = np.zeros((len(structure), 3), dtype=np.float32)
    fresh_baseline_particles, fresh_baseline_velocities = _particle_rollouts(
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
    replay_parity = {
        "bitwise_identical_to_frozen_localization": bool(
            np.array_equal(fresh_baseline_particles, frozen_baseline_particles)
        ),
        "maximum_absolute_difference_m": float(
            np.max(
                np.abs(fresh_baseline_particles - frozen_baseline_particles),
                initial=0.0,
            )
        ),
    }

    position_fields, velocity_fields, position_steps, velocity_steps = (
        modal_state_parameter_fields(
            graph_basis,
            position_step_m=position_step_m,
            velocity_step_mps=velocity_step_mps,
        )
    )
    reference_particle = int(
        localization_summary["particles"]["reference_sensitivity_particle_index"]
    )
    _set_particle(
        simulator,
        torch,
        wp,
        particles.log_scales[reference_particle],
        device=device,
    )
    state_response = _state_response_rollouts(
        simulator,
        torch,
        wp,
        endpoint_position_m=belief.endpoint_position_m[reference_particle],
        endpoint_velocity_mps=belief.endpoint_velocity_mps[reference_particle],
        baseline_m=fresh_baseline_particles[reference_particle],
        position_parameter_fields_m=position_fields,
        velocity_parameter_fields_mps=velocity_fields,
        start_frame=train_end_frame,
        stop_frame=frame_count,
        rest_lengths_m=graph.rest_lengths,
        controller_points_m=controller,
        device=device,
    )
    valid = _target_validity(visible, motion_valid)
    truth_relative = observed[endpoint_frame:, :original_count]
    valid_relative = valid[endpoint_frame:, :original_count]
    frozen_baseline_mean = _weighted_mean(frozen_baseline_particles, particles.weights)
    prefix_innovation = (
        truth_relative[:prefix_frame_count]
        - frozen_baseline_mean[:prefix_frame_count, :original_count]
    )
    prefix_valid = valid_relative[:prefix_frame_count]
    observation_variance = np.full(
        prefix_valid.shape, observation_std_m**2, dtype=np.float64
    )
    belief_config = PropagatedStateBeliefConfig(
        observation_std_m=observation_std_m,
        state_weight_prior_std=state_weight_prior_std,
        shared_bias_prior_std_m=shared_bias_prior_std_m,
    )
    selection_config = PropagatedStateSelectionConfig(
        fit_frame_count=fit_frame_count,
        minimum_validation_improvement_fraction=(
            minimum_validation_improvement_fraction
        ),
        minimum_validation_improvement_m=minimum_validation_improvement_m,
        projection_ridge=float(config["projection_ridge"]),
        maximum_position_update_m=maximum_position_update_m,
        maximum_velocity_update_mps=maximum_velocity_update_mps,
    )
    selection = select_propagated_state_update(
        prefix_innovation,
        prefix_valid,
        state_response[:prefix_frame_count, :original_count],
        graph_basis[:original_count],
        graph_basis,
        position_steps,
        velocity_steps,
        prior_reliability=prefix_valid.astype(np.float64),
        observation_variance_m2=observation_variance,
        belief_config=belief_config,
        selection_config=selection_config,
    )
    raw_belief = infer_propagated_state_belief(
        prefix_innovation,
        prefix_valid,
        state_response[:prefix_frame_count, :original_count],
        graph_basis[:original_count],
        prior_reliability=prefix_valid.astype(np.float64),
        observation_variance_m2=observation_variance,
        config=belief_config,
    )
    if raw_belief.accepted:
        raw_weights, position_update, velocity_update, raw_limits = (
            decode_limited_state_weights(
                raw_belief.state_weights,
                graph_basis,
                position_steps,
                velocity_steps,
                maximum_position_update_m=maximum_position_update_m,
                maximum_velocity_update_mps=maximum_velocity_update_mps,
            )
        )
        corrected_positions, _ = _particle_rollouts(
            simulator,
            torch,
            wp,
            particles=particles.log_scales,
            endpoint_positions_m=belief.endpoint_position_m + position_update[None],
            endpoint_velocities_mps=(
                belief.endpoint_velocity_mps + velocity_update[None]
            ),
            train_end_frame=train_end_frame,
            frame_count=frame_count,
            rest_lengths_m=graph.rest_lengths,
            controller_points_m=controller,
            external_forces_n=zero_force,
            device=device,
        )
        raw_particles = frozen_baseline_particles + (
            corrected_positions - fresh_baseline_particles
        )
        raw_bias = graph_basis @ raw_belief.shared_bias_coefficients_m
        raw_particles[:, prefix_frame_count:] += raw_bias[None, None]
        raw_effective_covariance = scale_posterior_covariance_for_state_limits(
            raw_belief.posterior_covariance,
            graph_rank=LOCALIZATION_GRAPH_RANK,
            position_scale=float(raw_limits["position"]["radial_scale"]),
            velocity_scale=float(raw_limits["velocity"]["radial_scale"]),
        )
        coefficient_variance = _prediction_variance_m2(
            state_response,
            graph_basis,
            raw_effective_covariance,
        )
    else:
        raw_weights = np.zeros(state_response.shape[3], dtype=np.float64)
        position_update = np.zeros((len(structure), 3), dtype=np.float64)
        velocity_update = np.zeros_like(position_update)
        raw_limits = {"fallback": True}
        raw_particles = frozen_readout_particles.copy()
        raw_effective_covariance = np.zeros(
            (state_response.shape[3] + 3 * LOCALIZATION_GRAPH_RANK,) * 2,
            dtype=np.float64,
        )
        coefficient_variance = None

    if selection.accepted and raw_belief.accepted:
        _require(
            np.allclose(selection.state_weights, raw_weights, rtol=0.0, atol=1e-12),
            "selection and full refit state weights differ",
        )
        guarded_particles = raw_particles.copy()
    else:
        guarded_particles = frozen_readout_particles.copy()
    exact_fallback = {
        "used": not selection.accepted,
        "particle_bytes_identical": bool(
            np.array_equal(guarded_particles, frozen_readout_particles)
        ),
        "global_bytes_identical": False,
        "fallback_method": READOUT,
    }

    raw_global = _global_trajectory(
        frozen_global_baseline,
        raw_particles,
        particles.weights,
        endpoint_frame=endpoint_frame,
    )
    guarded_global = (
        raw_global.copy() if selection.accepted else frozen_global_readout.copy()
    )
    exact_fallback["global_bytes_identical"] = bool(
        np.array_equal(guarded_global, frozen_global_readout)
    )
    particle_trajectories = {
        BASELINE: frozen_baseline_particles,
        READOUT: frozen_readout_particles,
        RAW_PROPAGATED: raw_particles,
        GUARDED_PROPAGATED: guarded_particles,
    }
    global_trajectories = {
        BASELINE: frozen_global_baseline,
        READOUT: frozen_global_readout,
        RAW_PROPAGATED: raw_global,
        GUARDED_PROPAGATED: guarded_global,
    }
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

    baseline_metrics = metrics(frozen_global_baseline)
    support_nodes = _attachment_support_nodes(graph, len(structure))
    graph_distance = _graph_distance(
        len(structure), graph.springs[: graph.num_object_springs], support_nodes
    )
    method_results: dict[str, Any] = {}
    for method in PROPAGATED_METHODS:
        candidate_metrics = metrics(global_trajectories[method])
        method_variance = None
        if method == RAW_PROPAGATED and coefficient_variance is not None:
            method_variance = coefficient_variance[:, :original_count]
        elif (
            method == GUARDED_PROPAGATED
            and selection.accepted
            and coefficient_variance is not None
        ):
            method_variance = coefficient_variance[:, :original_count]
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
            "coverage": _coverage_summary(
                particle_trajectories[method][:, :, :original_count],
                particles.weights,
                truth_relative,
                valid_relative,
                start_frame=prefix_frame_count,
                variance_floor_m2=variance_floor_m2,
                coefficient_variance_m2=method_variance,
            ),
        }

    frozen_bias_energy = float(
        np.sum(np.square(frozen_correction.position_coefficients_m))
    )
    raw_bias_energy = (
        float(np.sum(np.square(raw_belief.shared_bias_coefficients_m)))
        if raw_belief.accepted
        else frozen_bias_energy
    )
    shrinkage = (
        1.0 - raw_bias_energy / frozen_bias_energy if frozen_bias_energy > 0.0 else 0.0
    )
    source_checksums = {
        "localization_summary": _sha256(localization_summary_path),
        "localization_archive": _sha256(localization_archive_path),
        "localization_correction": _sha256(localization_correction_path),
        "final_data": _sha256(final_data_path),
        "checkpoint": _sha256(checkpoint_path),
        "parameter_profile": _sha256(parameter_profile_path),
        "twin_belief": _sha256(twin_belief_path),
    }
    selected_weights = selection.state_weights
    selected_bias = (
        raw_belief.shared_bias_coefficients_m
        if selection.accepted and raw_belief.accepted
        else frozen_correction.position_coefficients_m
    )
    selected_covariance = (
        raw_effective_covariance
        if selection.accepted and raw_belief.accepted
        else np.zeros(
            (len(selected_weights) + 3 * LOCALIZATION_GRAPH_RANK,) * 2,
            dtype=np.float64,
        )
    )
    artifact = PropagatedStateCorrection(
        case_id=case_id,
        graph_basis=graph_basis,
        graph_eigenvalues=frozen_correction.graph_eigenvalues,
        position_coefficient_steps_m=position_steps,
        velocity_coefficient_steps_mps=velocity_steps,
        state_weights=selected_weights,
        shared_bias_coefficients_m=selected_bias,
        posterior_covariance=selected_covariance,
        accepted_state_update=selection.accepted,
        selection_reason=selection.reason,
        prefix_frame_start=endpoint_frame,
        fit_frame_stop=endpoint_frame + fit_frame_count,
        prefix_frame_stop=heldout_start_frame,
        information_boundary={
            "o_plus_prefix_frames": 6,
            "prefix_includes_o_minus_endpoint": True,
            "forecast_frames_used_for_fit_or_selection": False,
            "manual_tracks_used_for_fit_or_selection": False,
            "released_case_role": "implementation_diagnostic_only_repeatedly_examined",
            "prospective_source_panel_promotion_claimed": False,
            "fallback": "frozen_graph_persistence_readout",
        },
        source_checksums=source_checksums,
        diagnostics={
            "selection": selection.diagnostics,
            "raw_belief": raw_belief.diagnostics,
            "raw_state_limits": raw_limits,
            "readout_correction_shrinkage_fraction": shrinkage,
            "posterior_covariance_is_raw_not_calibrated": True,
        },
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_record = write_propagated_state_correction(
        output / "propagated_state_correction", artifact
    )
    archive_path = output / "propagated_state_rollouts.npz"
    np.savez_compressed(
        archive_path,
        particle_weights=particles.weights,
        parameter_particles=particles.log_scales,
        state_response_at_step_m=state_response,
        raw_state_weights=raw_weights,
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
        "experiment": "phystwin_guarded_action_propagated_state_v1",
        "case": case_id,
        "status": "released_case_implementation_diagnostic_not_model_selection",
        "method_order": list(PROPAGATED_METHODS),
        "methods": method_results,
        "selection": {
            "accepted_state_update": selection.accepted,
            "reason": selection.reason,
            **selection.diagnostics,
        },
        "raw_belief": {
            "accepted": raw_belief.accepted,
            "reason": raw_belief.reason,
            "diagnostics": raw_belief.diagnostics,
        },
        "readout_correction_shrinkage_fraction": shrinkage,
        "replay_parity": replay_parity,
        "exact_fallback": exact_fallback,
        "config": {
            "position_step_m": position_step_m,
            "velocity_step_mps": velocity_step_mps,
            "one_sided_finite_perturbation": True,
            "state_weight_prior_std": state_weight_prior_std,
            "observation_std_m": observation_std_m,
            "shared_bias_prior_std_m": shared_bias_prior_std_m,
            "maximum_position_update_m": maximum_position_update_m,
            "maximum_velocity_update_mps": maximum_velocity_update_mps,
            "fit_frame_count": fit_frame_count,
            "minimum_validation_improvement_fraction": (
                minimum_validation_improvement_fraction
            ),
            "minimum_validation_improvement_m": minimum_validation_improvement_m,
            "variance_floor_m2": variance_floor_m2,
            "device": device,
        },
        "information_boundary": artifact.information_boundary,
        "causal_contract": {
            "state_support": "official Warp responses under recorded future action",
            "camera_only_common_mode_identifiability_claimed": False,
            "prior_reliability_uses_state_innovation": False,
            "innovation_likelihood_count": 1,
            "raw_covariance_calibrated": False,
            "future_labels_used_for_fit_or_selection": False,
            "guard_status": "development_only_not_source_calibrated",
        },
        "artifacts": {
            "correction": artifact_record,
            "rollouts": {
                "path": str(archive_path.resolve()),
                "sha256": _sha256(archive_path),
            },
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "localization_summary": localization_summary_path,
                "localization_archive": localization_archive_path,
                "localization_correction": localization_correction_path,
                "final_data": final_data_path,
                "baseline_trajectory": baseline_path,
                "optimal_params": optimal_path,
                "checkpoint": checkpoint_path,
                "parameter_profile": parameter_profile_path,
                "twin_belief": twin_belief_path,
                "gt_track": gt_track_path,
            }.items()
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def aggregate_guarded_propagated_state_cases(
    summary_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate exhausted development cases and apply the no-escalation gate."""

    _require(bool(summary_paths), "at least one case summary is required")
    cases: dict[str, Any] = {}
    method_values: dict[str, dict[str, list[float]]] = {
        method: {"chamfer_distance_m": [], "track_error_m": [], "coverage_90": []}
        for method in PROPAGATED_METHODS
    }
    input_records: list[dict[str, str]] = []
    for supplied_path in summary_paths:
        path = Path(supplied_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        _require(
            summary.get("experiment") == "phystwin_guarded_action_propagated_state_v1",
            "aggregate input is not a propagated-state diagnostic",
        )
        case = str(summary["case"])
        _require(case not in cases, f"duplicate case summary: {case}")
        _require(
            summary["status"]
            == "released_case_implementation_diagnostic_not_model_selection",
            "case status changed",
        )
        cases[case] = {
            "accepted_state_update": bool(
                summary["selection"]["accepted_state_update"]
            ),
            "selection_reason": str(summary["selection"]["reason"]),
            "prefix_validation_improvement_fraction": float(
                summary["selection"]["validation_improvement_fraction"]
            ),
            "readout_correction_shrinkage_fraction": float(
                summary["readout_correction_shrinkage_fraction"]
            ),
            "exact_fallback": dict(summary["exact_fallback"]),
        }
        for method in PROPAGATED_METHODS:
            future = summary["methods"][method]["future"]
            method_values[method]["chamfer_distance_m"].append(
                float(future["chamfer_distance_m"]["candidate_mean_m"])
            )
            method_values[method]["track_error_m"].append(
                float(future["track_error_m"]["candidate_mean_m"])
            )
            method_values[method]["coverage_90"].append(
                float(summary["methods"][method]["coverage"]["coordinate_coverage_90"])
            )
        input_records.append({"path": str(path.resolve()), "sha256": _sha256(path)})

    aggregate_methods: dict[str, Any] = {}
    for method, metrics in method_values.items():
        aggregate_methods[method] = {
            name: {
                "case_balanced_mean": float(np.mean(values)),
                "per_case": {
                    case: float(value)
                    for case, value in zip(cases, values, strict=True)
                },
            }
            for name, values in metrics.items()
        }
    readout_cd = aggregate_methods[READOUT]["chamfer_distance_m"]["case_balanced_mean"]
    readout_track = aggregate_methods[READOUT]["track_error_m"]["case_balanced_mean"]
    raw_cd = aggregate_methods[RAW_PROPAGATED]["chamfer_distance_m"][
        "case_balanced_mean"
    ]
    raw_track = aggregate_methods[RAW_PROPAGATED]["track_error_m"]["case_balanced_mean"]
    accepted_count = sum(int(case["accepted_state_update"]) for case in cases.values())
    all_rejections_are_exact = all(
        case["accepted_state_update"]
        or (
            case["exact_fallback"]["particle_bytes_identical"]
            and case["exact_fallback"]["global_bytes_identical"]
        )
        for case in cases.values()
    )
    aggregate = {
        "schema_version": 1,
        "experiment": "phystwin_guarded_action_propagated_state_v1_aggregate",
        "status": "frozen_exhausted_case_negative_diagnostic",
        "case_count": len(cases),
        "cases": cases,
        "methods": aggregate_methods,
        "comparison_vs_frozen_graph_persistence": {
            "raw_chamfer_percent_change": 100.0 * (raw_cd / readout_cd - 1.0),
            "raw_track_percent_change": 100.0 * (raw_track / readout_track - 1.0),
            "guarded_state_acceptance_count": accepted_count,
            "all_rejections_are_byte_exact_fallbacks": all_rejections_are_exact,
        },
        "decision": {
            "run_exploratory_19_case_cohort": False,
            "reason": (
                "No development case passed the forecast-blind prefix gate, and the "
                "raw state branch did not beat frozen graph persistence on both "
                "case-balanced accuracy metrics."
            ),
            "prospective_mechanism_promotion_claimed": False,
            "future_development_requirement": (
                "independent state/contact evidence or a genuinely fresh source panel"
            ),
        },
        "inputs": input_records,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return aggregate


__all__ = [
    "GUARDED_PROPAGATED",
    "PROPAGATED_METHODS",
    "RAW_PROPAGATED",
    "aggregate_guarded_propagated_state_cases",
    "evaluate_guarded_propagated_state_case",
]
