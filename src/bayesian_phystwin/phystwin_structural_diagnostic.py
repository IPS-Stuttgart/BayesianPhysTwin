"""Official-Warp diagnostic for hierarchical structural PhysTwin calibration."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.phystwin_comparison import official_metrics_by_frame
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    _load_pickle,
    _sha256,
    _target_validity,
)
from bayesian_phystwin.phystwin_state_injection import (
    _initialize_simulator,
    _metric_summary,
    _released_self_collision_for_case,
    _rollout_restart,
    _state_numpy,
)

from .structural_artifact import (
    STRUCTURAL_RANK_CANDIDATES,
    build_rigid_free_graph_basis,
    corrected_rest_geometry,
    identity_structural_twin_correction,
    write_structural_twin_correction,
)
from .structural_map import (
    STRUCTURAL_VARIANTS,
    StructuralLinearizedSession,
    StructuralMAPConfig,
    StructuralMAPResult,
    fit_hierarchical_structural_map,
    select_structural_map_result,
)
from .structural_warp import (
    apply_structural_configuration_to_simulator,
    assert_zero_configuration_parity,
    prepare_structural_warp_configuration,
    write_structural_warp_configuration,
)


RELEASED = "released_phystwin"
GRAPH_PERSISTENCE = "graph_persistence_readout"
EQUILIBRIUM_BASELINE = "equilibrium_baseline"


def _attachment_support_nodes(graph, object_count: int) -> np.ndarray:
    values = []
    for first, second in graph.springs[graph.num_object_springs :]:
        if first < object_count <= second:
            values.append(int(first))
        elif second < object_count <= first:
            values.append(int(second))
    if not values:
        raise ValueError("the PhysTwin graph has no controller attachment nodes")
    return np.asarray(sorted(set(values)), dtype=np.int64)


def _farthest_point_tracks(
    positions: np.ndarray,
    valid: np.ndarray,
    *,
    maximum_count: int,
) -> np.ndarray:
    support_count = np.sum(np.asarray(valid, dtype=bool), axis=0)
    candidates = np.flatnonzero(support_count > 0)
    if len(candidates) <= maximum_count:
        return candidates
    points = np.asarray(positions, dtype=float)[candidates]
    first = int(np.argmax(support_count[candidates]))
    selected = [first]
    minimum_squared = np.sum(np.square(points - points[first]), axis=1)
    for _ in range(1, maximum_count):
        score = minimum_squared * (
            0.5 + 0.5 * support_count[candidates] / np.max(support_count[candidates])
        )
        score[selected] = -1.0
        index = int(np.argmax(score))
        selected.append(index)
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum(np.square(points - points[index]), axis=1),
        )
    return np.sort(candidates[np.asarray(selected)])


def _set_simulator_arrays(
    simulator,
    torch,
    wp,
    *,
    rest_lengths: np.ndarray,
    controller_points: np.ndarray,
    device: str,
) -> None:
    simulator.set_rest_lengths(
        torch.as_tensor(
            rest_lengths, dtype=torch.float32, device=device
        ).contiguous()
    )
    simulator.set_controller_trajectory(
        torch.as_tensor(
            controller_points, dtype=torch.float32, device=device
        ).contiguous()
    )
    wp.synchronize()


def _full_rollout(
    simulator,
    torch,
    wp,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    controller_points: np.ndarray,
    rest_lengths: np.ndarray,
    frame_count: int,
    device: str,
) -> np.ndarray:
    _set_simulator_arrays(
        simulator,
        torch,
        wp,
        rest_lengths=rest_lengths,
        controller_points=controller_points,
        device=device,
    )
    future = _rollout_restart(
        simulator,
        torch,
        wp,
        position,
        velocity,
        start_frame=1,
        stop_frame=frame_count,
        device=device,
    )
    return np.concatenate((np.asarray(position)[None], future), axis=0)


def _settle_equilibrium(
    simulator,
    torch,
    wp,
    *,
    initial_position: np.ndarray,
    rest_lengths: np.ndarray,
    controller_anchor: np.ndarray,
    controller_shape: tuple[int, int, int],
    settle_steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    constant_controls = np.broadcast_to(
        np.asarray(controller_anchor)[None], controller_shape
    ).copy()
    _set_simulator_arrays(
        simulator,
        torch,
        wp,
        rest_lengths=rest_lengths,
        controller_points=constant_controls,
        device=device,
    )
    position = np.asarray(initial_position, dtype=float).copy()
    velocity = np.zeros_like(position)
    remaining = settle_steps
    start_position = position.copy()
    while remaining:
        count = min(remaining, controller_shape[0])
        trajectory = _rollout_restart(
            simulator,
            torch,
            wp,
            position,
            velocity,
            start_frame=0,
            stop_frame=count,
            device=device,
        )
        position = trajectory[-1]
        _, velocity = _state_numpy(simulator.wp_states[-1], wp)
        remaining -= count
    return position, velocity, {
        "settled_displacement_rms_m": float(
            np.sqrt(np.mean(np.square(position - start_position)))
        ),
        "settled_velocity_rms_mps": float(
            np.sqrt(np.mean(np.square(velocity)))
        ),
    }


def _row_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    values = np.asarray(axis, dtype=float)
    values /= np.linalg.norm(values)
    x, y, z = values
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    ).T


def _object_rest_lengths(
    positions: np.ndarray,
    springs: np.ndarray,
    released: np.ndarray,
    *,
    num_object_springs: int,
) -> np.ndarray:
    result = np.asarray(released).copy()
    edges = np.asarray(springs, dtype=np.int64)[:num_object_springs]
    result[:num_object_springs] = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1
    )
    return result


def _horizon_summary(
    baseline: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    frame_count = len(next(iter(baseline.values())))
    groups = np.array_split(np.arange(frame_count), 3)
    names = ("early", "middle", "late")
    return {
        name: {
            metric: {
                "baseline_mean_m": float(np.mean(baseline[metric][indices])),
                "candidate_mean_m": float(np.mean(candidate[metric][indices])),
                "percent_change": float(
                    100.0
                    * (
                        np.mean(candidate[metric][indices])
                        / np.mean(baseline[metric][indices])
                        - 1.0
                    )
                ),
            }
            for metric in baseline
        }
        for name, indices in zip(names, groups, strict=True)
    }


def _graph_distance(node_count: int, springs: np.ndarray, sources: np.ndarray) -> np.ndarray:
    adjacency = [[] for _ in range(node_count)]
    for first, second in springs:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    distance = np.full(node_count, np.inf)
    queue = list(map(int, sources))
    distance[queue] = 0.0
    for node in queue:
        for neighbour in adjacency[node]:
            if not np.isfinite(distance[neighbour]):
                distance[neighbour] = distance[node] + 1.0
                queue.append(neighbour)
    return distance


def _far_graph_observation_error(
    trajectory: np.ndarray,
    observed: np.ndarray,
    valid: np.ndarray,
    graph_distance: np.ndarray,
    *,
    start_frame: int,
) -> dict[str, float]:
    original_count = observed.shape[1]
    finite_distance = graph_distance[:original_count]
    threshold = float(np.quantile(finite_distance[np.isfinite(finite_distance)], 2.0 / 3.0))
    far = finite_distance >= threshold
    per_frame = []
    for frame in range(start_frame, len(observed)):
        support = valid[frame] & far
        if np.any(support):
            per_frame.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            trajectory[frame, :original_count][support]
                            - observed[frame][support],
                            axis=1,
                        )
                    )
                )
            )
    return {
        "far_graph_threshold_hops": threshold,
        "far_graph_track_count": int(np.sum(far)),
        "future_observation_error_mean_m": float(np.mean(per_frame)),
    }


def _load_graph_persistence_candidate(
    archive_path: str | Path | None,
    baseline: np.ndarray,
    *,
    train_end_frame: int,
) -> np.ndarray | None:
    if archive_path is None:
        return None
    with np.load(archive_path, allow_pickle=False) as archive:
        key = "future__output_frame_graph"
        if key not in archive.files:
            raise ValueError(f"graph-persistence archive has no {key}")
        future = np.asarray(archive[key], dtype=float)
    if future.shape != baseline[train_end_frame:].shape:
        raise ValueError("graph-persistence future shape differs from the case")
    result = baseline.copy()
    result[train_end_frame:] = future
    return result


def evaluate_phystwin_structural_case(
    official_repo: str | Path,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    gt_track_path: str | Path | None = None,
    graph_persistence_archive_path: str | Path | None = None,
    inner_validation_frames: int = 8,
    maximum_fit_tracks: int = 512,
    settle_steps: int = 30,
    basis_step: float = 0.05,
    frame_rotation_step_rad: float = np.deg2rad(0.5),
    frame_translation_step_m: float = 0.002,
    allowed_edge_strain: float = 0.10,
    dt: float = 5e-5,
    num_substeps: int = 667,
    deterministic_spring_forces: bool = True,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Fit from released O-minus only and rerun every structural mechanism in Warp."""

    if not 2 <= inner_validation_frames < train_end_frame:
        raise ValueError("inner validation must be a strict O-minus suffix")
    if maximum_fit_tracks < 16 or settle_steps < 1 or basis_step <= 0.0:
        raise ValueError("linearization controls must be positive")
    if frame_rotation_step_rad <= 0.0 or frame_translation_step_m <= 0.0:
        raise ValueError("frame sensitivity steps must be positive")
    data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_params_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controller = np.asarray(data["controller_points"], dtype=float)
    surface = np.asarray(data["surface_points"], dtype=float)
    interior = np.asarray(data["interior_points"], dtype=float)
    frame_count, original_count, _ = observed.shape
    if not train_end_frame < frame_count or baseline.shape[0] < frame_count:
        raise ValueError("training split or released trajectory is incompatible")
    baseline = baseline[:frame_count]
    structure = np.concatenate((observed[0], surface, interior), axis=0)
    if baseline.shape[1:] != structure.shape:
        raise ValueError("released object state and structure disagree")
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
    support_nodes = _attachment_support_nodes(graph, len(structure))
    basis, frequencies, basis_diagnostics = build_rigid_free_graph_basis(
        structure,
        graph.springs[: graph.num_object_springs],
        rank=16,
        support_node_indices=support_nodes,
    )
    valid = _target_validity(visible, motion_valid)
    fit_tracks = _farthest_point_tracks(
        observed[0], valid[:train_end_frame], maximum_count=maximum_fit_tracks
    )
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
        deterministic_spring_forces=deterministic_spring_forces,
        device=device,
    )

    direct = _full_rollout(
        simulator,
        torch,
        wp,
        position=structure,
        velocity=np.zeros_like(structure),
        controller_points=controller,
        rest_lengths=graph.rest_lengths,
        frame_count=frame_count,
        device=device,
    )
    identity = identity_structural_twin_correction(
        structure,
        graph.springs,
        graph.rest_lengths,
        num_object_springs=graph.num_object_springs,
        graph_basis=basis[:, :, :4],
        graph_frequencies=frequencies[:4],
        session_ids=(Path(final_data_path).resolve().parent.name,),
        support_node_indices=support_nodes,
        source_checksums={"final_data": _sha256(final_data_path)},
        allowed_edge_strain=allowed_edge_strain,
    )
    identity_configuration = prepare_structural_warp_configuration(
        identity,
        structure,
        graph.springs,
        graph.rest_lengths,
        num_object_springs=graph.num_object_springs,
        session_id=Path(final_data_path).resolve().parent.name,
        nominal_initial_position_m=structure,
        nominal_initial_velocity_mps=np.zeros_like(structure),
        controller_points_m=controller,
    )
    parity_inputs = assert_zero_configuration_parity(
        identity_configuration,
        nominal_rest_positions_m=structure,
        nominal_rest_lengths_m=graph.rest_lengths,
        nominal_initial_position_m=structure,
        nominal_initial_velocity_mps=np.zeros_like(structure),
        controller_points_m=controller,
    )
    apply_structural_configuration_to_simulator(
        simulator, torch, wp, identity_configuration, device=device
    )
    identity_rollout = _full_rollout(
        simulator,
        torch,
        wp,
        position=structure,
        velocity=np.zeros_like(structure),
        controller_points=controller,
        rest_lengths=graph.rest_lengths,
        frame_count=frame_count,
        device=device,
    )
    parity_inputs["warp_trajectory_bitwise_identical"] = bool(
        np.array_equal(direct, identity_rollout)
    )
    if not parity_inputs["warp_trajectory_bitwise_identical"]:
        raise AssertionError("zero structural correction changed the Warp rollout")

    equilibrium, equilibrium_velocity, equilibrium_diagnostics = _settle_equilibrium(
        simulator,
        torch,
        wp,
        initial_position=structure,
        rest_lengths=graph.rest_lengths,
        controller_anchor=controller[0],
        controller_shape=controller.shape,
        settle_steps=settle_steps,
        device=device,
    )
    nominal_trajectory = _full_rollout(
        simulator,
        torch,
        wp,
        position=equilibrium,
        velocity=equilibrium_velocity,
        controller_points=controller,
        rest_lengths=graph.rest_lengths,
        frame_count=frame_count,
        device=device,
    )
    selected_nominal = nominal_trajectory[:, fit_tracks]
    persistent_response = np.empty(
        (frame_count, len(fit_tracks), 3, 16), dtype=np.float32
    )
    settled_response = np.empty_like(persistent_response)
    for mode in range(16):
        perturbed_rest = structure + basis_step * basis[:, :, mode]
        perturbed_lengths = _object_rest_lengths(
            perturbed_rest,
            graph.springs,
            graph.rest_lengths,
            num_object_springs=graph.num_object_springs,
        )
        perturbed_equilibrium, perturbed_velocity, _ = _settle_equilibrium(
            simulator,
            torch,
            wp,
            initial_position=perturbed_rest,
            rest_lengths=perturbed_lengths,
            controller_anchor=controller[0],
            controller_shape=controller.shape,
            settle_steps=settle_steps,
            device=device,
        )
        trajectory = _full_rollout(
            simulator,
            torch,
            wp,
            position=perturbed_equilibrium,
            velocity=perturbed_velocity,
            controller_points=controller,
            rest_lengths=perturbed_lengths,
            frame_count=frame_count,
            device=device,
        )
        persistent_response[..., mode] = (
            (trajectory[:, fit_tracks] - selected_nominal) / basis_step
        )
        state_trajectory = _full_rollout(
            simulator,
            torch,
            wp,
            position=equilibrium + basis_step * basis[:, :, mode],
            velocity=equilibrium_velocity,
            controller_points=controller,
            rest_lengths=graph.rest_lengths,
            frame_count=frame_count,
            device=device,
        )
        settled_response[..., mode] = (
            (state_trajectory[:, fit_tracks] - selected_nominal) / basis_step
        )

    frame_response = np.empty(
        (frame_count, len(fit_tracks), 3, 6), dtype=np.float32
    )
    for axis_index, axis in enumerate(np.eye(3)):
        rotation = _row_rotation(axis, frame_rotation_step_rad)
        trajectory = _full_rollout(
            simulator,
            torch,
            wp,
            position=equilibrium @ rotation,
            velocity=equilibrium_velocity @ rotation,
            controller_points=controller @ rotation,
            rest_lengths=graph.rest_lengths,
            frame_count=frame_count,
            device=device,
        )
        frame_response[..., axis_index] = (
            trajectory[:, fit_tracks] - selected_nominal
        ) / frame_rotation_step_rad
        translation = frame_translation_step_m * axis
        trajectory = _full_rollout(
            simulator,
            torch,
            wp,
            position=equilibrium + translation,
            velocity=equilibrium_velocity,
            controller_points=controller + translation,
            rest_lengths=graph.rest_lengths,
            frame_count=frame_count,
            device=device,
        )
        frame_response[..., 3 + axis_index] = (
            trajectory[:, fit_tracks] - selected_nominal
        ) / frame_translation_step_m

    fit_mask = np.arange(train_end_frame) < train_end_frame - inner_validation_frames
    validation_mask = ~fit_mask
    results: list[StructuralMAPResult] = []
    case_name = Path(final_data_path).resolve().parent.name
    for rank in STRUCTURAL_RANK_CANDIDATES:
        session = StructuralLinearizedSession(
            session_id=case_name,
            observations_m=observed[:train_end_frame, fit_tracks],
            nominal_prediction_m=selected_nominal[:train_end_frame],
            observation_weights=valid[:train_end_frame, fit_tracks].astype(float),
            persistent_response=persistent_response[:train_end_frame, ..., :rank],
            settled_state_response=settled_response[:train_end_frame, ..., :rank],
            gravity_response=np.zeros(
                (train_end_frame, len(fit_tracks), 3, 3), dtype=np.float32
            ),
            fit_frame_mask=fit_mask,
            validation_frame_mask=validation_mask,
            frame_origin_m=np.zeros(3),
            frame_response=frame_response[:train_end_frame],
            metadata={
                "case": case_name,
                "released_data_role": "diagnostic_only_repeatedly_examined_target",
                "track_indices_sha256": _sha256_array(fit_tracks),
            },
        )
        for variant in STRUCTURAL_VARIANTS:
            results.append(
                fit_hierarchical_structural_map(
                    (session,),
                    structure,
                    graph.springs,
                    graph.rest_lengths,
                    num_object_springs=graph.num_object_springs,
                    graph_basis=basis[:, :, :rank],
                    graph_frequencies=frequencies[:rank],
                    support_node_indices=support_nodes,
                    support_model={
                        "kind": "controller_attachment_node_anchors",
                        "node_count": len(support_nodes),
                    },
                    config=StructuralMAPConfig(
                        variant=variant,
                        rank=rank,
                        persistent_prior_strength=2e-3,
                        settled_state_prior_strength=8e-3,
                        frame_rotation_prior_strength=2e-3,
                        frame_translation_prior_strength=2e-3,
                        gravity_prior_strength=3e-3,
                        edge_strain_prior_strength=2e-3,
                        huber_delta_m=0.005,
                        robust_iterations=3,
                        allowed_edge_strain=allowed_edge_strain,
                    ),
                    metadata={
                        "case": case_name,
                        "gravity_sensitivity_available": False,
                        "surface_validity_topology_available": False,
                    },
                )
            )

    selected_by_variant = {}
    for variant in STRUCTURAL_VARIANTS:
        candidates = [value for value in results if value.config.variant == variant]
        selected_by_variant[variant] = select_structural_map_result(candidates)
    selected_physical = select_structural_map_result(
        tuple(selected_by_variant.values())
    )

    trajectories = {RELEASED: baseline, EQUILIBRIUM_BASELINE: nominal_trajectory}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variant_artifacts = {}
    variant_linear_selection = {}
    for variant, result in selected_by_variant.items():
        artifact_dir = output / "variants" / variant
        artifact_record = write_structural_twin_correction(
            artifact_dir / "correction", result.correction
        )
        geometry = corrected_rest_geometry(
            result.correction,
            structure,
            graph.springs,
            graph.rest_lengths,
            num_object_springs=graph.num_object_springs,
        )
        corrected_equilibrium, corrected_velocity, settle_diagnostics = (
            _settle_equilibrium(
                simulator,
                torch,
                wp,
                initial_position=geometry.rest_positions,
                rest_lengths=geometry.rest_lengths,
                controller_anchor=controller[0],
                controller_shape=controller.shape,
                settle_steps=settle_steps,
                device=device,
            )
        )
        configuration = prepare_structural_warp_configuration(
            result.correction,
            structure,
            graph.springs,
            graph.rest_lengths,
            num_object_springs=graph.num_object_springs,
            session_id=case_name,
            nominal_initial_position_m=equilibrium,
            nominal_initial_velocity_mps=corrected_velocity,
            controller_points_m=controller,
            corrected_equilibrium_position_m=corrected_equilibrium,
        )
        configuration_record = write_structural_warp_configuration(
            artifact_dir / "warp_configuration", configuration
        )
        initial_position, initial_velocity = apply_structural_configuration_to_simulator(
            simulator,
            torch,
            wp,
            configuration,
            device=device,
        )
        trajectories[variant] = _full_rollout(
            simulator,
            torch,
            wp,
            position=initial_position,
            velocity=initial_velocity,
            controller_points=configuration.controller_points_m,
            rest_lengths=configuration.corrected_rest_lengths_m,
            frame_count=frame_count,
            device=device,
        )
        variant_artifacts[variant] = {
            "correction": artifact_record,
            "warp_configuration": configuration_record,
            "settling": settle_diagnostics,
        }
        variant_linear_selection[variant] = {
            "rank": result.config.rank,
            "validation_rmse_m": result.diagnostics["mean_validation_rmse_m"],
            "validation_standard_error_m": result.diagnostics[
                "validation_standard_error_m"
            ],
            "parameter_count": result.diagnostics["parameter_count"],
            "artifact_id": result.correction.artifact_id,
        }

    persistence = _load_graph_persistence_candidate(
        graph_persistence_archive_path,
        baseline,
        train_end_frame=train_end_frame,
    )
    if persistence is not None:
        trajectories[GRAPH_PERSISTENCE] = persistence
    gt_track = (
        None if gt_track_path is None else np.asarray(_load_pickle(gt_track_path))
    )
    if gt_track is None:
        raise ValueError("the structural diagnostic requires manual tracks")
    num_surface_points = original_count + len(surface)

    def metrics(trajectory: np.ndarray) -> dict[str, np.ndarray]:
        return official_metrics_by_frame(
            trajectory,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=train_end_frame,
            end_frame=frame_count,
        )

    released_metrics = metrics(baseline)
    object_distance = _graph_distance(
        len(structure),
        graph.springs[: graph.num_object_springs],
        support_nodes,
    )
    method_results = {}
    for method, trajectory in trajectories.items():
        candidate_metrics = metrics(trajectory)
        method_results[method] = {
            "future": _metric_summary(released_metrics, candidate_metrics),
            "horizon": _horizon_summary(released_metrics, candidate_metrics),
            "far_graph": _far_graph_observation_error(
                trajectory,
                observed,
                valid,
                object_distance,
                start_frame=train_end_frame,
            ),
        }
    np.savez_compressed(
        output / "structural_linearization.npz",
        graph_basis=basis,
        graph_frequencies=frequencies,
        support_node_indices=support_nodes,
        fit_track_indices=fit_tracks,
        nominal_selected_trajectory=selected_nominal,
        persistent_response=persistent_response,
        settled_state_response=settled_response,
        frame_response=frame_response,
        **{f"trajectory__{name}": value for name, value in trajectories.items()},
    )
    summary = {
        "schema_version": 1,
        "experiment": "hierarchical_graph_structural_calibration",
        "case": case_name,
        "status": "released_case_diagnostic_not_confirmatory",
        "config": {
            "train_end_frame": train_end_frame,
            "inner_validation_frames": inner_validation_frames,
            "maximum_fit_tracks": maximum_fit_tracks,
            "settle_steps": settle_steps,
            "basis_step": basis_step,
            "frame_rotation_step_rad": frame_rotation_step_rad,
            "frame_translation_step_m": frame_translation_step_m,
            "allowed_edge_strain": allowed_edge_strain,
            "dt": dt,
            "num_substeps": num_substeps,
            "deterministic_spring_forces": deterministic_spring_forces,
            "device": device,
        },
        "information_boundary": {
            "fit_frames": [0, train_end_frame - inner_validation_frames],
            "validation_frames": [
                train_end_frame - inner_validation_frames,
                train_end_frame,
            ],
            "holdout_frames": [train_end_frame, frame_count],
            "future_frames_used_for_fit_or_selection": False,
            "manual_tracks_used_for_fit_or_selection": False,
            "rank_candidates": list(STRUCTURAL_RANK_CANDIDATES),
            "released_target_role": "diagnostic_only_repeatedly_examined",
        },
        "basis": {
            **basis_diagnostics,
            "object_vertex_count": len(structure),
            "object_spring_count": graph.num_object_springs,
            "support_anchor_count": len(support_nodes),
            "surface_validity_topology_available": False,
        },
        "equilibrium": equilibrium_diagnostics,
        "zero_correction_parity": parity_inputs,
        "linear_selection_by_variant": variant_linear_selection,
        "selected_physical_variant": selected_physical.config.variant,
        "selected_physical_rank": selected_physical.config.rank,
        "methods": method_results,
        "artifacts": {
            "linearization_archive": str(
                (output / "structural_linearization.npz").resolve()
            ),
            "variants": variant_artifacts,
        },
        "inputs": {
            "official_repo": str(Path(official_repo).resolve()),
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "baseline_trajectory": {
                "path": str(Path(baseline_trajectory_path).resolve()),
                "sha256": _sha256(baseline_trajectory_path),
            },
            "optimal_params": {
                "path": str(Path(optimal_params_path).resolve()),
                "sha256": _sha256(optimal_params_path),
            },
            "checkpoint": {
                "path": str(Path(checkpoint_path).resolve()),
                "sha256": _sha256(checkpoint_path),
            },
            "gt_track": {
                "path": str(Path(gt_track_path).resolve()),
                "sha256": _sha256(gt_track_path),
            },
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del simulator
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    import hashlib

    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()
