"""Inject graph-regularized rest geometry into the released PhysTwin Warp model."""

from __future__ import annotations

import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from bayesian_phystwin.phystwin_additional_bayesian_confirmation import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
)
from bayesian_phystwin.phystwin_additional_confirmation import _chamfer_by_frame
from bayesian_phystwin.phystwin_bayesian_anchor import robust_random_walk_endpoint
from bayesian_phystwin.phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from bayesian_phystwin.phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_graph_discrepancy import (
    graph_discrepancy_diagnostics,
    normalized_spring_laplacian,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    _load_pickle,
    _sha256,
    _target_validity,
)
from bayesian_phystwin.phystwin_state_injection import (
    _git_commit,
    _initialize_simulator,
    _metric_summary,
    _released_self_collision_for_case,
    _rollout_restart,
    _simulator_runtime,
    estimate_endpoint_velocity_delta,
)

from causal4d.rest_geometry import (
    GraphRestGeometryCorrection,
    apply_frame_correction,
    infer_graph_rest_geometry_correction,
    reattach_controller_rest_lengths,
    rotate_vectors,
)
from causal4d.rest_geometry_transfer import (
    attach_target_controller_to_canonical_graph,
    canonical_material_graph_sha256,
    load_canonical_material_graph,
)


RELEASED = "released"
ENDPOINT_RESTART = "endpoint_restart"
OUTPUT_FRAME_GRAPH = "output_frame_graph"
FRAME_STATE = "frame_state_original_rest"
GRAPH_STATE_ORIGINAL_REST = "graph_state_original_rest"
REST_GEOMETRY_ONLY = "rest_geometry_only"
FRAME_REST_GEOMETRY = "frame_rest_geometry"
REST_GEOMETRY_REATTACHED_ONLY = "rest_geometry_reattached_only"
FRAME_REST_GEOMETRY_REATTACHED = "frame_rest_geometry_reattached"
SELECTED_FRAME_REST_GEOMETRY = "selected_frame_rest_geometry"
PRIMARY_METHOD = SELECTED_FRAME_REST_GEOMETRY
REST_GEOMETRY_METHODS = (
    RELEASED,
    ENDPOINT_RESTART,
    OUTPUT_FRAME_GRAPH,
    FRAME_STATE,
    GRAPH_STATE_ORIGINAL_REST,
    REST_GEOMETRY_ONLY,
    FRAME_REST_GEOMETRY,
    REST_GEOMETRY_REATTACHED_ONLY,
    FRAME_REST_GEOMETRY_REATTACHED,
    SELECTED_FRAME_REST_GEOMETRY,
)

CONTROLLER_REST_MODES = ("preserve", "recompute")


def _scale_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(dict.fromkeys(float(value) for value in values))
    if not result or any(not 0.0 <= value <= 1.0 for value in result):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return result


def _controller_rest_grid(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value) for value in values))
    if not result or any(value not in CONTROLLER_REST_MODES for value in result):
        raise ValueError(f"controller rest modes must lie in {CONTROLLER_REST_MODES}")
    return result


def _endpoint_posterior(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
):
    return robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=end_frame,
        process_variance=FIXED_PROCESS_STD_M**2,
        observation_variance=FIXED_OBSERVATION_STD_M**2,
        initial_variance=FIXED_INITIAL_STD_M**2,
        inlier_prior=FIXED_INLIER_PRIOR,
        outlier_variance_multiplier=FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    )


def _track_error_by_frame(
    trajectory: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
) -> np.ndarray:
    candidate = np.asarray(trajectory, dtype=float)
    target = np.asarray(observations, dtype=float)
    support = np.asarray(valid, dtype=bool)
    original_count = target.shape[1]
    if candidate.ndim != 3 or candidate.shape[2] != 3:
        raise ValueError("trajectory must have shape (T, N, 3)")
    if target.ndim != 3 or target.shape[2] != 3 or support.shape != target.shape[:2]:
        raise ValueError("observation arrays disagree")
    if not 0 <= start_frame < stop_frame <= min(len(candidate), len(target)):
        raise ValueError("track evaluation window is invalid")
    result = []
    for frame in range(start_frame, stop_frame):
        frame_support = support[frame]
        if not np.any(frame_support):
            raise ValueError(f"frame {frame} has no valid pre-holdout tracks")
        error = np.linalg.norm(
            candidate[frame, :original_count][frame_support]
            - target[frame][frame_support],
            axis=1,
        )
        result.append(float(np.mean(error)))
    return np.asarray(result)


def _set_simulator_configuration(
    simulator,
    torch,
    wp,
    *,
    rest_lengths: np.ndarray,
    controller_points: np.ndarray,
    device: str,
) -> None:
    rest_tensor = torch.as_tensor(
        rest_lengths, dtype=torch.float32, device=device
    ).contiguous()
    controller_tensor = torch.as_tensor(
        controller_points, dtype=torch.float32, device=device
    ).contiguous()
    simulator.set_rest_lengths(rest_tensor)
    simulator.set_controller_trajectory(controller_tensor)
    wp.synchronize()


def _correction_diagnostics(
    correction: GraphRestGeometryCorrection,
    object_springs: np.ndarray,
    laplacian,
) -> dict[str, object]:
    ratio = correction.rest_length_ratio
    nonrigid_norm = np.linalg.norm(correction.nonrigid_field, axis=1)
    return {
        "frame_mode": correction.frame.mode,
        "frame_rotation_deg": float(np.rad2deg(correction.frame.rotation_angle_rad)),
        "frame_translation_m": correction.frame.translation.tolist(),
        "frame_translation_norm_m": float(np.linalg.norm(correction.frame.translation)),
        "fitted_frame_point_count": correction.frame.fitted_point_count,
        "nonrigid_rms_m": float(np.sqrt(np.mean(np.square(nonrigid_norm)))),
        "nonrigid_maximum_m": float(np.max(nonrigid_norm, initial=0.0)),
        "rest_length_ratio_median": float(np.median(ratio)),
        "rest_length_ratio_minimum": float(np.min(ratio)),
        "rest_length_ratio_maximum": float(np.max(ratio)),
        "rest_length_clip_fraction": float(
            np.mean(~np.isclose(ratio, correction.unclipped_rest_length_ratio))
        ),
        "graph": graph_discrepancy_diagnostics(
            correction.nonrigid_field,
            object_springs,
            laplacian,
        ),
    }


def _make_correction(
    baseline: np.ndarray,
    graph,
    laplacian,
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    evidence_stop_frame: int,
    graph_prior_strength: float,
    frame_mode: str,
    frame_scale: float,
    rest_geometry_scale: float,
    maximum_frame_rotation_rad: float,
    maximum_frame_translation_m: float,
    maximum_nonrigid_norm_m: float,
    maximum_rest_log_ratio: float,
) -> tuple[GraphRestGeometryCorrection, object]:
    endpoint = _endpoint_posterior(
        residual,
        valid,
        end_frame=evidence_stop_frame,
    )
    object_vertex_count = baseline.shape[1]
    correction = infer_graph_rest_geometry_correction(
        baseline[evidence_stop_frame - 1],
        graph.vertices[:object_vertex_count],
        graph.springs,
        graph.rest_lengths,
        num_object_springs=graph.num_object_springs,
        endpoint_mean=endpoint.mean,
        endpoint_variance=endpoint.variance,
        observed=endpoint.update_count > 0,
        laplacian=laplacian,
        graph_prior_strength=graph_prior_strength,
        frame_mode=frame_mode,
        frame_scale=frame_scale,
        rest_geometry_scale=rest_geometry_scale,
        maximum_frame_rotation_rad=maximum_frame_rotation_rad,
        maximum_frame_translation_m=maximum_frame_translation_m,
        maximum_nonrigid_norm_m=maximum_nonrigid_norm_m,
        maximum_rest_log_ratio=maximum_rest_log_ratio,
    )
    return correction, endpoint


def _run_configured_restart(
    simulator,
    torch,
    wp,
    *,
    rest_lengths: np.ndarray,
    controller_points: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    start_frame: int,
    stop_frame: int,
    device: str,
) -> np.ndarray:
    _set_simulator_configuration(
        simulator,
        torch,
        wp,
        rest_lengths=rest_lengths,
        controller_points=controller_points,
        device=device,
    )
    return _rollout_restart(
        simulator,
        torch,
        wp,
        position,
        velocity,
        start_frame=start_frame,
        stop_frame=stop_frame,
        device=device,
    )


def evaluate_phystwin_rest_geometry_case(
    official_repo: str | Path,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    gt_track_path: str | Path | None = None,
    frame_mode: str = "se3",
    frame_scale_grid: Iterable[float] = (0.0, 0.5, 1.0),
    rest_geometry_scale_grid: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    controller_rest_mode_grid: Iterable[str] = CONTROLLER_REST_MODES,
    graph_prior_strength: float = 0.1,
    inner_validation_frames: int = 8,
    velocity_history_frames: int = 3,
    maximum_frame_rotation_rad: float = np.deg2rad(5.0),
    maximum_frame_translation_m: float = 0.02,
    maximum_nonrigid_norm_m: float = 0.01,
    maximum_rest_log_ratio: float = np.log(1.15),
    dt: float = 5e-5,
    num_substeps: int = 667,
    self_collision: bool | None = None,
    deterministic_spring_forces: bool = True,
    device: str = "cuda:0",
    canonical_material_graph_path: str | Path | None = None,
) -> dict[str, object]:
    """Select on an inner training suffix, refit, and rerun the future in Warp."""

    frame_scales = _scale_grid(frame_scale_grid, name="frame_scale_grid")
    rest_scales = _scale_grid(
        rest_geometry_scale_grid, name="rest_geometry_scale_grid"
    )
    controller_rest_modes = _controller_rest_grid(controller_rest_mode_grid)
    if graph_prior_strength <= 0.0:
        raise ValueError("graph_prior_strength must be positive")
    if inner_validation_frames < 1 or velocity_history_frames < 2:
        raise ValueError("validation and velocity histories are too short")
    if dt <= 0.0 or num_substeps < 1:
        raise ValueError("simulator time settings must be positive")
    if device != "cuda:0":
        raise ValueError(
            "the pinned official simulator selects cuda:0; use CUDA_VISIBLE_DEVICES"
        )

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
    fit_end_frame = train_end_frame - inner_validation_frames
    if not velocity_history_frames <= fit_end_frame < train_end_frame < frame_count:
        raise ValueError("training split cannot support the requested inner validation")
    if baseline.shape[0] < frame_count:
        raise ValueError("baseline trajectory has too few frames")
    baseline = baseline[:frame_count]
    structure_points = np.concatenate((observed[0], surface, interior), axis=0)
    if baseline.shape[1] != len(structure_points):
        raise ValueError("released trajectory and object state size disagree")
    if self_collision is None:
        self_collision = _released_self_collision_for_case(
            Path(final_data_path).resolve().parent.name
        )
    graph_config = PhysTwinSpringGraphConfig(
        object_radius=float(optimal["object_radius"]),
        object_max_neighbours=int(optimal["object_max_neighbours"]),
        controller_radius=float(optimal["controller_radius"]),
        controller_max_neighbours=int(optimal["controller_max_neighbours"]),
    )
    if canonical_material_graph_path is None:
        graph = build_phystwin_spring_graph(
            structure_points,
            controller[0],
            config=graph_config,
        )
    else:
        canonical_graph = load_canonical_material_graph(
            canonical_material_graph_path
        )
        if len(canonical_graph.vertices) != len(structure_points):
            raise ValueError("canonical graph and execution object size disagree")
        graph = attach_target_controller_to_canonical_graph(
            canonical_graph,
            controller[0],
            config=graph_config,
        )
    object_springs = graph.springs[: graph.num_object_springs]
    laplacian = normalized_spring_laplacian(len(structure_points), object_springs)
    residual = observed - baseline[:, :original_count]
    valid = _target_validity(visible, motion_valid)
    frame_dt = dt * num_substeps

    warnings.filterwarnings(
        "ignore",
        message=(
            "Running the tape backwards may produce incorrect gradients because "
            "recorded kernel set_control_points.*"
        ),
    )
    simulator, torch, wp, checkpoint = _initialize_simulator(
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

    selection_velocity = estimate_endpoint_velocity_delta(
        baseline[fit_end_frame - velocity_history_frames : fit_end_frame],
        frame_dt=frame_dt,
    )
    selection_records = []
    for frame_scale in frame_scales:
        for rest_scale in rest_scales:
            correction, _ = _make_correction(
                baseline,
                graph,
                laplacian,
                residual,
                valid,
                evidence_stop_frame=fit_end_frame,
                graph_prior_strength=graph_prior_strength,
                frame_mode=frame_mode,
                frame_scale=frame_scale,
                rest_geometry_scale=rest_scale,
                maximum_frame_rotation_rad=maximum_frame_rotation_rad,
                maximum_frame_translation_m=maximum_frame_translation_m,
                maximum_nonrigid_norm_m=maximum_nonrigid_norm_m,
                maximum_rest_log_ratio=maximum_rest_log_ratio,
            )
            reattached_rest_lengths, _, _ = reattach_controller_rest_lengths(
                correction.corrected_reference_vertices,
                apply_frame_correction(
                    graph.vertices[len(structure_points) :], correction.frame
                ),
                graph.springs,
                correction.corrected_rest_lengths,
                num_object_springs=graph.num_object_springs,
                maximum_log_ratio=maximum_rest_log_ratio,
            )
            corrected_controller = apply_frame_correction(controller, correction.frame)
            for controller_rest_mode in controller_rest_modes:
                candidate_rest_lengths = (
                    correction.corrected_rest_lengths
                    if controller_rest_mode == "preserve"
                    else reattached_rest_lengths
                )
                validation_future = _run_configured_restart(
                    simulator,
                    torch,
                    wp,
                    rest_lengths=candidate_rest_lengths,
                    controller_points=corrected_controller,
                    position=(
                        baseline[fit_end_frame - 1] + correction.endpoint_correction
                    ),
                    velocity=rotate_vectors(selection_velocity, correction.frame),
                    start_frame=fit_end_frame,
                    stop_frame=train_end_frame,
                    device=device,
                )
                validation_trajectory = baseline.copy()
                validation_trajectory[fit_end_frame:train_end_frame] = validation_future
                track_by_frame = _track_error_by_frame(
                    validation_trajectory,
                    observed,
                    valid,
                    start_frame=fit_end_frame,
                    stop_frame=train_end_frame,
                )
                selection_records.append(
                    {
                        "frame_scale": frame_scale,
                        "rest_geometry_scale": rest_scale,
                        "controller_rest_mode": controller_rest_mode,
                        "track_error_by_frame_m": track_by_frame.tolist(),
                        "track_error_mean_m": float(np.mean(track_by_frame)),
                    }
                )
    selection_records.sort(
        key=lambda value: (
            value["track_error_mean_m"],
            value["frame_scale"] + value["rest_geometry_scale"],
            value["rest_geometry_scale"],
            value["frame_scale"],
            value["controller_rest_mode"] != "preserve",
        )
    )
    selected = selection_records[0]
    selected_frame_scale = float(selected["frame_scale"])
    selected_rest_scale = float(selected["rest_geometry_scale"])
    selected_controller_rest_mode = str(selected["controller_rest_mode"])

    correction, endpoint = _make_correction(
        baseline,
        graph,
        laplacian,
        residual,
        valid,
        evidence_stop_frame=train_end_frame,
        graph_prior_strength=graph_prior_strength,
        frame_mode=frame_mode,
        frame_scale=selected_frame_scale,
        rest_geometry_scale=selected_rest_scale,
        maximum_frame_rotation_rad=maximum_frame_rotation_rad,
        maximum_frame_translation_m=maximum_frame_translation_m,
        maximum_nonrigid_norm_m=maximum_nonrigid_norm_m,
        maximum_rest_log_ratio=maximum_rest_log_ratio,
    )
    endpoint_index = train_end_frame - 1
    endpoint_velocity = estimate_endpoint_velocity_delta(
        baseline[train_end_frame - velocity_history_frames : train_end_frame],
        frame_dt=frame_dt,
    )
    corrected_controller = apply_frame_correction(controller, correction.frame)
    reattached_rest_lengths, controller_raw_ratio, controller_ratio = (
        reattach_controller_rest_lengths(
            correction.corrected_reference_vertices,
            apply_frame_correction(
                graph.vertices[len(structure_points) :], correction.frame
            ),
            graph.springs,
            correction.corrected_rest_lengths,
            num_object_springs=graph.num_object_springs,
            maximum_log_ratio=maximum_rest_log_ratio,
        )
    )
    frame_position = apply_frame_correction(
        baseline[endpoint_index], correction.frame
    )
    full_position = baseline[endpoint_index] + correction.endpoint_correction
    rotated_velocity = rotate_vectors(endpoint_velocity, correction.frame)

    candidates = {RELEASED: baseline.copy()}

    def add_rollout(
        method: str,
        *,
        rest_lengths: np.ndarray,
        controller_points: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        candidate = baseline.copy()
        candidate[endpoint_index] = position
        candidate[train_end_frame:] = _run_configured_restart(
            simulator,
            torch,
            wp,
            rest_lengths=rest_lengths,
            controller_points=controller_points,
            position=position,
            velocity=velocity,
            start_frame=train_end_frame,
            stop_frame=frame_count,
            device=device,
        )
        candidates[method] = candidate

    add_rollout(
        ENDPOINT_RESTART,
        rest_lengths=graph.rest_lengths,
        controller_points=controller,
        position=baseline[endpoint_index],
        velocity=endpoint_velocity,
    )
    output_candidate = baseline.copy()
    output_candidate[endpoint_index:] = apply_frame_correction(
        baseline[endpoint_index:], correction.frame
    ) + selected_rest_scale * correction.nonrigid_field[None]
    candidates[OUTPUT_FRAME_GRAPH] = output_candidate
    add_rollout(
        FRAME_STATE,
        rest_lengths=graph.rest_lengths,
        controller_points=corrected_controller,
        position=frame_position,
        velocity=rotated_velocity,
    )
    add_rollout(
        GRAPH_STATE_ORIGINAL_REST,
        rest_lengths=graph.rest_lengths,
        controller_points=corrected_controller,
        position=full_position,
        velocity=rotated_velocity,
    )
    add_rollout(
        REST_GEOMETRY_ONLY,
        rest_lengths=correction.corrected_rest_lengths,
        controller_points=controller,
        position=baseline[endpoint_index],
        velocity=endpoint_velocity,
    )
    add_rollout(
        FRAME_REST_GEOMETRY,
        rest_lengths=correction.corrected_rest_lengths,
        controller_points=corrected_controller,
        position=full_position,
        velocity=rotated_velocity,
    )
    add_rollout(
        REST_GEOMETRY_REATTACHED_ONLY,
        rest_lengths=reattached_rest_lengths,
        controller_points=controller,
        position=baseline[endpoint_index],
        velocity=endpoint_velocity,
    )
    add_rollout(
        FRAME_REST_GEOMETRY_REATTACHED,
        rest_lengths=reattached_rest_lengths,
        controller_points=corrected_controller,
        position=full_position,
        velocity=rotated_velocity,
    )
    selected_method = (
        FRAME_REST_GEOMETRY
        if selected_controller_rest_mode == "preserve"
        else FRAME_REST_GEOMETRY_REATTACHED
    )
    candidates[SELECTED_FRAME_REST_GEOMETRY] = candidates[selected_method].copy()

    gt_track = (
        None
        if gt_track_path is None
        else np.asarray(_load_pickle(gt_track_path), dtype=float)
    )
    num_surface_points = original_count + len(surface)

    def metrics_by_frame(trajectory: np.ndarray) -> dict[str, np.ndarray]:
        if gt_track is None:
            return {
                "chamfer_distance_m": _chamfer_by_frame(
                    trajectory,
                    observed,
                    visible,
                    num_surface_points=num_surface_points,
                    start_frame=train_end_frame,
                    end_frame=frame_count,
                )
            }
        return official_metrics_by_frame(
            trajectory,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=train_end_frame,
            end_frame=frame_count,
        )

    released_metrics = metrics_by_frame(baseline)
    method_results = {
        method: {"future": _metric_summary(released_metrics, metrics_by_frame(value))}
        for method, value in candidates.items()
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "rest_geometry_injection.npz"
    archive_values = {
        "nonrigid_field": correction.nonrigid_field,
        "endpoint_correction": correction.endpoint_correction,
        "frame_linear": correction.frame.linear,
        "frame_translation": correction.frame.translation,
        "canonical_reference_vertices": graph.vertices[: len(structure_points)],
        "corrected_reference_vertices": correction.corrected_reference_vertices,
        "object_springs": object_springs,
        "released_rest_lengths": graph.rest_lengths,
        "corrected_rest_lengths": correction.corrected_rest_lengths,
        "reattached_rest_lengths": reattached_rest_lengths,
        "rest_length_ratio": correction.rest_length_ratio,
        **{
            f"future__{method}": trajectory[train_end_frame:]
            for method, trajectory in candidates.items()
        },
    }
    np.savez_compressed(archive_path, **archive_values)
    updated = endpoint.update_count > 0
    summary: dict[str, object] = {
        "schema_version": 1,
        "method": "graph-regularized frame/rest-geometry correction",
        "config": {
            "train_end_frame": train_end_frame,
            "inner_validation_frames": inner_validation_frames,
            "frame_mode": frame_mode,
            "frame_scale_grid": list(frame_scales),
            "rest_geometry_scale_grid": list(rest_scales),
            "controller_rest_mode_grid": list(controller_rest_modes),
            "graph_prior_strength": graph_prior_strength,
            "velocity_history_frames": velocity_history_frames,
            "maximum_frame_rotation_rad": maximum_frame_rotation_rad,
            "maximum_frame_translation_m": maximum_frame_translation_m,
            "maximum_nonrigid_norm_m": maximum_nonrigid_norm_m,
            "maximum_rest_log_ratio": maximum_rest_log_ratio,
            "dt": dt,
            "num_substeps": num_substeps,
            "self_collision": self_collision,
            "deterministic_spring_forces": deterministic_spring_forces,
            "device": device,
            "runtime": _simulator_runtime(),
        },
        "information_boundary": {
            "correction_evidence_frames": [0, train_end_frame],
            "hyperparameter_fit_frames": [0, fit_end_frame],
            "hyperparameter_validation_frames": [fit_end_frame, train_end_frame],
            "holdout_evaluation_frames": [train_end_frame, frame_count],
            "holdout_frames_used_for_inference": False,
            "holdout_frames_used_for_hyperparameter_selection": False,
            "manual_gt_track_used_for_inference": False,
            "manual_gt_track_used_for_hyperparameter_selection": False,
        },
        "selection": {
            "criterion": "mean pseudo-track error on the inner pre-holdout suffix",
            "selected_frame_scale": selected_frame_scale,
            "selected_rest_geometry_scale": selected_rest_scale,
            "selected_controller_rest_mode": selected_controller_rest_mode,
            "candidates": selection_records,
        },
        "contract": {
            "frame": (
                "reliability-weighted bounded SE(3), applied to endpoint state, "
                "velocities, and the complete recorded controller trajectory"
            ),
            "rest_geometry": (
                "graph-smoothed nonrigid endpoint remainder transferred to the "
                "frame-zero material reference"
            ),
            "object_spring_rest_lengths": (
                "recomputed from corrected material reference and log-ratio clipped"
            ),
            "controller_spring_rest_lengths": (
                "preserve and corrected-reference reattachment are explicit "
                "ablations; the primary mode is selected pre-holdout"
            ),
            "world_gravity_and_ground": (
                "official PhysTwin world gravity and z=0 ground remain fixed; "
                "the frame update changes object/control orientation relative "
                "to those physical directions"
            ),
            "nonrigid_velocity": "zero under the tested quasi-static discrepancy model",
            "future_controls": "recorded released controller trajectory",
            "future_observations": "none",
        },
        "inputs": {
            "final_data": {"path": str(Path(final_data_path).resolve()), "sha256": _sha256(final_data_path)},
            "baseline_trajectory": {"path": str(Path(baseline_trajectory_path).resolve()), "sha256": _sha256(baseline_trajectory_path)},
            "optimal_params": {"path": str(Path(optimal_params_path).resolve()), "sha256": _sha256(optimal_params_path)},
            "checkpoint": {
                "path": str(Path(checkpoint_path).resolve()),
                "sha256": _sha256(checkpoint_path),
                "epoch": int(checkpoint.get("epoch", -1)),
            },
            "gt_track_3d": (
                None
                if gt_track_path is None
                else {"path": str(Path(gt_track_path).resolve()), "sha256": _sha256(gt_track_path)}
            ),
            "official_repo": {"path": str(Path(official_repo).resolve()), "commit": _git_commit(official_repo)},
            "canonical_material_graph": (
                None
                if canonical_material_graph_path is None
                else {
                    "path": str(Path(canonical_material_graph_path).resolve()),
                    "sha256": _sha256(canonical_material_graph_path),
                }
            ),
        },
        "graph": {
            "object_vertex_count": len(structure_points),
            "object_spring_count": graph.num_object_springs,
            "controller_spring_count": len(graph.springs) - graph.num_object_springs,
            "canonical_material_graph_sha256": canonical_material_graph_sha256(
                graph.vertices[: len(structure_points)],
                graph.springs,
                graph.rest_lengths,
                num_object_springs=graph.num_object_springs,
            ),
        },
        "endpoint_posterior": {
            "updated_track_count": int(np.sum(updated)),
            "median_std_m": float(np.median(np.sqrt(endpoint.variance[updated]))),
            "median_final_inlier_probability": float(
                np.median(endpoint.final_inlier_probability[updated])
            ),
        },
        "correction": _correction_diagnostics(correction, object_springs, laplacian),
        "controller_reattachment": {
            "rest_length_ratio_median": float(np.median(controller_ratio)),
            "rest_length_ratio_minimum": float(np.min(controller_ratio)),
            "rest_length_ratio_maximum": float(np.max(controller_ratio)),
            "rest_length_clip_fraction": float(
                np.mean(~np.isclose(controller_ratio, controller_raw_ratio))
            ),
        },
        "methods": method_results,
        "primary_method": PRIMARY_METHOD,
        "outputs": {"archive": str(archive_path.resolve())},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["summary"] = str(summary_path.resolve())
    simulator = None
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _candidate_metrics(summary: dict[str, object], method: str) -> dict[str, np.ndarray]:
    return {
        metric: np.asarray(values["candidate_by_frame_m"], dtype=float)
        for metric, values in summary["methods"][method]["future"].items()
    }


def _compact_bootstrap(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key in {"samples", "block_length", "seed", "macro", "cluster_macro"}
    }


def run_phystwin_rest_geometry_comparison(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str = "development",
    cases: Iterable[str] | None = None,
    frame_mode: str = "se3",
    frame_scale_grid: Iterable[float] = (0.0, 0.5, 1.0),
    rest_geometry_scale_grid: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    controller_rest_mode_grid: Iterable[str] = CONTROLLER_REST_MODES,
    graph_prior_strength: float = 0.1,
    inner_validation_frames: int = 8,
    velocity_history_frames: int = 3,
    maximum_frame_rotation_rad: float = np.deg2rad(5.0),
    maximum_frame_translation_m: float = 0.02,
    maximum_nonrigid_norm_m: float = 0.01,
    maximum_rest_log_ratio: float = np.log(1.15),
    dt: float = 5e-5,
    num_substeps: int = 667,
    deterministic_spring_forces: bool = True,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260712,
    force: bool = False,
    canonical_material_graph_path: str | Path | None = None,
) -> dict[str, object]:
    """Run the locked development or confirmation rest-geometry comparison."""

    root = Path(data_root)
    additional_manifest = root / "additional_evaluation_subset_manifest.json"
    main_manifest = root / "evaluation_subset_manifest.json"
    if additional_manifest.exists():
        manifest_path = additional_manifest
        is_additional = True
    elif main_manifest.exists():
        manifest_path = main_manifest
        is_additional = False
    else:
        raise FileNotFoundError("data root has no PhysTwin evaluation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = tuple(str(case) for case in manifest["selected_cases"])
    if cases is not None:
        selected = tuple(dict.fromkeys(str(case) for case in cases))
    elif is_additional or cohort == "all":
        selected = available
    elif cohort == "development":
        selected = tuple(case for case in available if case in DEVELOPMENT_CASES)
    elif cohort == "confirmation":
        selected = tuple(case for case in available if case not in DEVELOPMENT_CASES)
    else:
        raise ValueError("cohort must be all, development, or confirmation")
    if is_additional and cohort != "all" and cases is None:
        raise ValueError("additional data supports only the all cohort")
    missing = sorted(set(selected) - set(available))
    if missing or not selected:
        raise ValueError("invalid selected cases: " + ", ".join(missing))
    for case in selected:
        if not (root / case / "checkpoint.pth").is_file():
            raise FileNotFoundError(f"released checkpoint is missing for {case}")

    frame_scales = _scale_grid(frame_scale_grid, name="frame_scale_grid")
    rest_scales = _scale_grid(
        rest_geometry_scale_grid, name="rest_geometry_scale_grid"
    )
    controller_rest_modes = _controller_rest_grid(controller_rest_mode_grid)
    clusters = {case: phystwin_physical_object_cluster(case) for case in selected}
    self_collision_by_case = {
        case: _released_self_collision_for_case(case) for case in selected
    }
    runtime = _simulator_runtime()
    code_commit = _git_commit(Path(__file__).resolve().parents[2])
    specification = {
        "method": "graph-regularized frame/rest-geometry PhysTwin injection",
        "code_commit": code_commit,
        "dataset": "additional" if is_additional else "main",
        "cohort": cohort,
        "cases": list(selected),
        "frame_mode": frame_mode,
        "frame_scale_grid": list(frame_scales),
        "rest_geometry_scale_grid": list(rest_scales),
        "controller_rest_mode_grid": list(controller_rest_modes),
        "graph_prior_strength": graph_prior_strength,
        "inner_validation_frames": inner_validation_frames,
        "velocity_history_frames": velocity_history_frames,
        "maximum_frame_rotation_rad": maximum_frame_rotation_rad,
        "maximum_frame_translation_m": maximum_frame_translation_m,
        "maximum_nonrigid_norm_m": maximum_nonrigid_norm_m,
        "maximum_rest_log_ratio": maximum_rest_log_ratio,
        "dt": dt,
        "num_substeps": num_substeps,
        "deterministic_spring_forces": deterministic_spring_forces,
        "primary_method": PRIMARY_METHOD,
        "selection_boundary": (
            "fixed grid selected on an inner suffix of O-minus; endpoint correction "
            "then refit on all O-minus; no future frame enters inference or selection"
        ),
        "official_repo": str(Path(official_repo).resolve()),
        "official_commit": _git_commit(official_repo),
        "runtime": runtime,
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": bootstrap_block_length,
            "seed": bootstrap_seed,
        },
        "data_manifest": str(manifest_path.resolve()),
        "canonical_material_graph": (
            None
            if canonical_material_graph_path is None
            else {
                "path": str(Path(canonical_material_graph_path).resolve()),
                "sha256": _sha256(canonical_material_graph_path),
            }
        ),
        "status": "development" if cohort == "development" else "locked evaluation",
    }
    output = Path(output_dir)
    locked = _lock_protocol(output, specification)
    case_results = {}
    paired_vs_released = {method: {} for method in REST_GEOMETRY_METHODS}
    paired_vs_restart = {
        method: {} for method in REST_GEOMETRY_METHODS if method != ENDPOINT_RESTART
    }
    for case in selected:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_end, future_end = (int(value) for value in split["test"])
        if future_end != int(split["frame_len"]):
            raise ValueError(f"future split does not end at frame_len for {case}")
        case_output = output / "cases" / case
        summary_path = case_output / "summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            track_path = case_dir / "gt_track_3d.pkl"
            summary = evaluate_phystwin_rest_geometry_case(
                official_repo,
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_dir / "optimal_params.pkl",
                case_dir / "checkpoint.pth",
                case_output,
                train_end_frame=train_end,
                gt_track_path=track_path if track_path.exists() else None,
                frame_mode=frame_mode,
                frame_scale_grid=frame_scales,
                rest_geometry_scale_grid=rest_scales,
                controller_rest_mode_grid=controller_rest_modes,
                graph_prior_strength=graph_prior_strength,
                inner_validation_frames=inner_validation_frames,
                velocity_history_frames=velocity_history_frames,
                maximum_frame_rotation_rad=maximum_frame_rotation_rad,
                maximum_frame_translation_m=maximum_frame_translation_m,
                maximum_nonrigid_norm_m=maximum_nonrigid_norm_m,
                maximum_rest_log_ratio=maximum_rest_log_ratio,
                dt=dt,
                num_substeps=num_substeps,
                self_collision=self_collision_by_case[case],
                deterministic_spring_forces=deterministic_spring_forces,
                canonical_material_graph_path=canonical_material_graph_path,
            )
        case_results[case] = {
            "physical_object": clusters[case],
            "information_boundary": summary["information_boundary"],
            "selection": summary["selection"],
            "correction": summary["correction"],
            "methods": summary["methods"],
        }
        released_metrics = _candidate_metrics(summary, RELEASED)
        restart_metrics = _candidate_metrics(summary, ENDPOINT_RESTART)
        for method in REST_GEOMETRY_METHODS:
            method_metrics = _candidate_metrics(summary, method)
            paired_vs_released[method][case] = (released_metrics, method_metrics)
            if method != ENDPOINT_RESTART:
                paired_vs_restart[method][case] = (restart_metrics, method_metrics)

    def bootstrap(paired):
        return _compact_bootstrap(
            paired_block_bootstrap(
                paired,
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed,
                clusters=clusters,
            )
        )

    result = {
        "schema_version": 1,
        "code_commit": code_commit,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": specification["status"],
        "dataset": specification["dataset"],
        "cohort": cohort,
        "case_count": len(selected),
        "physical_object_count": len(set(clusters.values())),
        "methods": list(REST_GEOMETRY_METHODS),
        "primary_method": PRIMARY_METHOD,
        "case_results": case_results,
        "comparisons_vs_released": {
            method: bootstrap(paired) for method, paired in paired_vs_released.items()
        },
        "comparisons_vs_endpoint_restart": {
            method: bootstrap(paired) for method, paired in paired_vs_restart.items()
        },
    }
    result_path = output / "rest_geometry_comparison_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
