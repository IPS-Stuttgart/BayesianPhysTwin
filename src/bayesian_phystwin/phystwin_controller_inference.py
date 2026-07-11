"""Training-only inference of a smooth latent PhysTwin controller bias."""

from __future__ import annotations

import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .phystwin_additional_confirmation import _chamfer_by_frame
from .phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from .phystwin_controller_sensitivity import (
    _compact_bootstrap,
    _distribution,
    _future_arrays,
    _metric_summary,
    controller_hand_count,
    infer_controller_groups,
)
from .phystwin_graph import PhysTwinSpringGraphConfig, build_phystwin_spring_graph
from .phystwin_residual_dynamics import _load_pickle, _sha256
from .phystwin_state_injection import (
    _array_hash,
    _git_commit,
    _initialize_simulator,
    _released_self_collision_for_case,
    _rollout_restart,
    _simulator_runtime,
    _trajectory_error,
    estimate_endpoint_velocity_delta,
)


DEFAULT_BIAS_SCALES_M = (0.0005, 0.001, 0.002)


def persistent_group_bias_trajectory(
    controller_points: np.ndarray,
    groups: np.ndarray,
    group_bias_m: np.ndarray,
    *,
    start_frame: int,
    ramp_frames: int,
) -> np.ndarray:
    """Return a smooth ramp to a persistent translation for each hand group."""

    controls = np.asarray(controller_points, dtype=float)
    labels = np.asarray(groups, dtype=np.int32)
    bias = np.asarray(group_bias_m, dtype=float)
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError("controller_points must have shape (T, C, 3)")
    if labels.shape != (controls.shape[1],) or np.any(labels < 0):
        raise ValueError("groups must label every controller point")
    group_count = int(np.max(labels)) + 1
    if bias.shape != (group_count, 3):
        raise ValueError("group_bias_m must have shape (G, 3)")
    if not 1 <= start_frame < len(controls):
        raise ValueError("start_frame must lie inside the controller trajectory")
    if ramp_frames < 1:
        raise ValueError("ramp_frames must be positive")
    alpha = np.zeros(len(controls), dtype=float)
    alpha[start_frame:] = np.minimum(
        np.arange(1, len(controls) - start_frame + 1, dtype=float) / ramp_frames,
        1.0,
    )
    return alpha[:, None, None] * bias[labels][None]


def point_weighted_group_bias_rms(
    group_bias_m: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Measure group translations as RMS over released controller points."""

    bias = np.asarray(group_bias_m, dtype=float)
    labels = np.asarray(groups, dtype=np.int32)
    if bias.ndim != 2 or bias.shape[1] != 3:
        raise ValueError("group_bias_m must have shape (G, 3)")
    if labels.ndim != 1 or len(labels) < 1:
        raise ValueError("groups must be a nonempty vector")
    if np.any(labels < 0) or int(np.max(labels)) >= len(bias):
        raise ValueError("groups reference an unavailable bias")
    return float(np.sqrt(np.mean(np.sum(np.square(bias[labels]), axis=1))))


def scale_group_bias_direction(
    direction: np.ndarray,
    groups: np.ndarray,
    *,
    target_rms_m: float,
    maximum_group_norm_m: float,
) -> np.ndarray:
    """Scale a search direction with point-RMS and per-group proximity caps."""

    values = np.asarray(direction, dtype=float)
    if target_rms_m <= 0.0 or maximum_group_norm_m <= 0.0:
        raise ValueError("bias scale and group cap must be positive")
    current = point_weighted_group_bias_rms(values, groups)
    if current <= 1e-15:
        return np.zeros_like(values)
    result = values * (target_rms_m / current)
    maximum = float(np.max(np.linalg.norm(result, axis=1)))
    if maximum > maximum_group_norm_m:
        result *= maximum_group_norm_m / maximum
    return result


def latent_controller_objective(
    chamfer_by_frame_m: np.ndarray,
    controller_jitter_m: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    observation_sigma_m: float,
    controller_sigma_m: float,
    smoothness_sigma_m: float,
) -> dict[str, float]:
    """Compute a dimensionless data term plus proximity and smoothness priors."""

    chamfer = np.asarray(chamfer_by_frame_m, dtype=float)
    jitter = np.asarray(controller_jitter_m, dtype=float)
    if chamfer.ndim != 1 or len(chamfer) != stop_frame - start_frame:
        raise ValueError("chamfer length must match the selected frame interval")
    if jitter.ndim != 3 or jitter.shape[2] != 3 or stop_frame > len(jitter):
        raise ValueError("controller_jitter_m must have shape (T, C, 3)")
    if not 0 < start_frame < stop_frame:
        raise ValueError("selection interval must be nonempty")
    if min(observation_sigma_m, controller_sigma_m, smoothness_sigma_m) <= 0.0:
        raise ValueError("all prior scales must be positive")
    selected = jitter[start_frame:stop_frame]
    data_term = 0.5 * float(np.mean(np.square(chamfer / observation_sigma_m)))
    proximity_term = 0.5 * float(
        np.mean(np.sum(np.square(selected), axis=2)) / controller_sigma_m**2
    )
    roughness_source = jitter[max(0, start_frame - 1) : stop_frame]
    if len(roughness_source) < 3:
        smoothness_term = 0.0
        second_difference_rms_m = 0.0
    else:
        second_difference = np.diff(roughness_source, n=2, axis=0)
        second_difference_rms_m = float(
            np.sqrt(np.mean(np.sum(np.square(second_difference), axis=2)))
        )
        smoothness_term = 0.5 * float(
            np.mean(np.sum(np.square(second_difference), axis=2))
            / smoothness_sigma_m**2
        )
    return {
        "objective": data_term + proximity_term + smoothness_term,
        "data_term": data_term,
        "proximity_term": proximity_term,
        "smoothness_term": smoothness_term,
        "chamfer_mean_m": float(np.mean(chamfer)),
        "controller_vector_rms_m": float(
            np.sqrt(np.mean(np.sum(np.square(selected), axis=2)))
        ),
        "controller_second_difference_rms_m": second_difference_rms_m,
    }


def _training_chamfer(
    baseline: np.ndarray,
    rollout: np.ndarray,
    observed: np.ndarray,
    visible: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    stop_frame: int,
) -> np.ndarray:
    trajectory = baseline.copy()
    trajectory[start_frame:stop_frame] = rollout
    return _chamfer_by_frame(
        trajectory,
        observed,
        visible,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=stop_frame,
    )


def apply_latent_controller_bias(
    official_repo: str | Path,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    gt_track_path: str | Path | None = None,
    validation_fraction: float = 0.75,
    finite_difference_m: float = 0.001,
    bias_scales_m: Iterable[float] = DEFAULT_BIAS_SCALES_M,
    ramp_frames: int = 5,
    maximum_group_norm_m: float = 0.003,
    observation_sigma_m: float = 0.005,
    controller_sigma_m: float = 0.002,
    smoothness_sigma_m: float = 0.0005,
    velocity_history_frames: int = 3,
    dt: float = 5e-5,
    num_substeps: int = 667,
    self_collision: bool | None = None,
    deterministic_spring_forces: bool = True,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Infer a persistent per-hand bias from training object observations only."""

    scales = tuple(sorted(set(float(value) for value in bias_scales_m)))
    if not scales or any(value <= 0.0 for value in scales):
        raise ValueError("bias_scales_m must contain positive values")
    if not 0.5 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0.5, 1)")
    if finite_difference_m <= 0.0 or maximum_group_norm_m <= 0.0:
        raise ValueError("finite difference and group cap must be positive")
    if velocity_history_frames < 2 or ramp_frames < 1:
        raise ValueError("velocity history and ramp must be positive")
    if dt <= 0.0 or num_substeps < 1:
        raise ValueError("simulator time step settings must be positive")
    if device != "cuda:0":
        raise ValueError("use CUDA_VISIBLE_DEVICES to remap the pinned cuda:0")

    data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_params_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    controller = np.asarray(data["controller_points"], dtype=float)
    surface = np.asarray(data["surface_points"], dtype=float)
    interior = np.asarray(data["interior_points"], dtype=float)
    frame_count, original_count, _ = observed.shape
    if not velocity_history_frames <= train_end_frame < frame_count:
        raise ValueError("training endpoint does not support the velocity history")
    if baseline.shape[0] < frame_count:
        raise ValueError("baseline trajectory has too few frames")
    baseline = baseline[:frame_count]
    structure_points = np.concatenate((observed[0], surface, interior), axis=0)
    if baseline.shape[1] != len(structure_points):
        raise ValueError("released trajectory and object state size disagree")
    selection_start = max(
        velocity_history_frames,
        int(np.floor(validation_fraction * train_end_frame)),
    )
    if train_end_frame - selection_start < 4:
        raise ValueError("training selection interval must contain at least four frames")

    case_name = Path(final_data_path).resolve().parent.name
    if self_collision is None:
        self_collision = _released_self_collision_for_case(case_name)
    group_count = controller_hand_count(case_name)
    groups = infer_controller_groups(controller[0], group_count=group_count)
    graph = build_phystwin_spring_graph(
        structure_points,
        controller[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    frame_dt = dt * num_substeps
    initial_index = selection_start - 1
    estimated_velocity = estimate_endpoint_velocity_delta(
        baseline[selection_start - velocity_history_frames : selection_start],
        frame_dt=frame_dt,
    )
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

    def rollout(values: np.ndarray, stop_frame: int) -> np.ndarray:
        simulator.controller_points = torch.as_tensor(
            values, dtype=torch.float32, device=device
        ).contiguous()
        return _rollout_restart(
            simulator,
            torch,
            wp,
            baseline[initial_index],
            estimated_velocity,
            start_frame=selection_start,
            stop_frame=stop_frame,
            device=device,
        )

    num_surface_points = original_count + len(surface)
    matched_full = rollout(controller, frame_count)
    duplicate_full = rollout(controller, frame_count)
    repeatability = _trajectory_error(matched_full, duplicate_full)
    matched_training = _training_chamfer(
        baseline,
        matched_full[: train_end_frame - selection_start],
        observed,
        visible,
        num_surface_points=num_surface_points,
        start_frame=selection_start,
        stop_frame=train_end_frame,
    )
    zero_jitter = np.zeros_like(controller)
    zero_objective = latent_controller_objective(
        matched_training,
        zero_jitter,
        start_frame=selection_start,
        stop_frame=train_end_frame,
        observation_sigma_m=observation_sigma_m,
        controller_sigma_m=controller_sigma_m,
        smoothness_sigma_m=smoothness_sigma_m,
    )

    finite_differences = []
    gradient = np.zeros((group_count, 3), dtype=float)
    for group in range(group_count):
        for axis in range(3):
            means = {}
            records = {}
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                basis = np.zeros((group_count, 3), dtype=float)
                basis[group, axis] = sign * finite_difference_m
                jitter = persistent_group_bias_trajectory(
                    controller,
                    groups,
                    basis,
                    start_frame=selection_start,
                    ramp_frames=ramp_frames,
                )
                candidate_training = rollout(
                    controller + jitter, train_end_frame
                )
                chamfer = _training_chamfer(
                    baseline,
                    candidate_training,
                    observed,
                    visible,
                    num_surface_points=num_surface_points,
                    start_frame=selection_start,
                    stop_frame=train_end_frame,
                )
                means[label] = float(np.mean(chamfer))
                records[label] = {
                    "chamfer_by_frame_m": chamfer.tolist(),
                    "chamfer_mean_m": means[label],
                }
            derivative = (means["plus"] - means["minus"]) / (
                2.0 * finite_difference_m
            )
            gradient[group, axis] = derivative
            finite_differences.append(
                {
                    "group": group,
                    "axis": axis,
                    "epsilon_m": finite_difference_m,
                    "derivative_m_per_m": derivative,
                    **records,
                }
            )

    candidates = [
        {
            "candidate_id": "zero",
            "direction": "zero",
            "target_point_rms_m": 0.0,
            "group_bias_m": np.zeros((group_count, 3), dtype=float),
            "jitter": zero_jitter,
            "training_chamfer_by_frame_m": matched_training,
            "objective": zero_objective,
        }
    ]
    descent = -gradient
    if point_weighted_group_bias_rms(descent, groups) > 1e-15:
        for scale in scales:
            base_bias = scale_group_bias_direction(
                descent,
                groups,
                target_rms_m=scale,
                maximum_group_norm_m=maximum_group_norm_m,
            )
            for sign, direction_label in ((1.0, "descent"), (-1.0, "opposite")):
                bias = sign * base_bias
                jitter = persistent_group_bias_trajectory(
                    controller,
                    groups,
                    bias,
                    start_frame=selection_start,
                    ramp_frames=ramp_frames,
                )
                candidate_training = rollout(controller + jitter, train_end_frame)
                chamfer = _training_chamfer(
                    baseline,
                    candidate_training,
                    observed,
                    visible,
                    num_surface_points=num_surface_points,
                    start_frame=selection_start,
                    stop_frame=train_end_frame,
                )
                objective = latent_controller_objective(
                    chamfer,
                    jitter,
                    start_frame=selection_start,
                    stop_frame=train_end_frame,
                    observation_sigma_m=observation_sigma_m,
                    controller_sigma_m=controller_sigma_m,
                    smoothness_sigma_m=smoothness_sigma_m,
                )
                candidates.append(
                    {
                        "candidate_id": (
                            f"{direction_label}_{1000.0 * scale:.3f}mm"
                        ),
                        "direction": direction_label,
                        "target_point_rms_m": scale,
                        "group_bias_m": bias,
                        "jitter": jitter,
                        "training_chamfer_by_frame_m": chamfer,
                        "objective": objective,
                    }
                )

    selected = min(
        candidates,
        key=lambda item: (
            float(item["objective"]["objective"]),
            float(item["objective"]["controller_vector_rms_m"]),
            str(item["candidate_id"]),
        ),
    )
    selected_jitter = np.asarray(selected["jitter"], dtype=float)
    selected_full = rollout(controller + selected_jitter, frame_count)
    matched = baseline.copy()
    matched[selection_start:] = matched_full
    duplicate = baseline.copy()
    duplicate[selection_start:] = duplicate_full
    latent = baseline.copy()
    latent[selection_start:] = selected_full
    gt_track = (
        None
        if gt_track_path is None
        else np.asarray(_load_pickle(gt_track_path), dtype=float)
    )

    def future_metrics(trajectory: np.ndarray) -> dict[str, np.ndarray]:
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

    released_metrics = future_metrics(baseline)
    matched_metrics = future_metrics(matched)
    duplicate_metrics = future_metrics(duplicate)
    latent_metrics = future_metrics(latent)
    serialized_candidates = []
    for candidate in candidates:
        serialized_candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "direction": candidate["direction"],
                "target_point_rms_m": candidate["target_point_rms_m"],
                "group_bias_m": np.asarray(candidate["group_bias_m"]).tolist(),
                "training_chamfer_by_frame_m": np.asarray(
                    candidate["training_chamfer_by_frame_m"]
                ).tolist(),
                "objective": candidate["objective"],
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "latent_controller_bias.npz"
    np.savez_compressed(
        archive_path,
        controller_groups=groups,
        estimated_start_velocity=estimated_velocity,
        finite_difference_gradient=gradient,
        selected_group_bias=np.asarray(selected["group_bias_m"]),
        selected_controller_jitter=selected_jitter.astype(np.float32),
        matched_rollout=matched_full.astype(np.float32),
        selected_rollout=selected_full.astype(np.float32),
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": {
            "train_end_frame": train_end_frame,
            "validation_fraction": validation_fraction,
            "selection_start_frame": selection_start,
            "finite_difference_m": finite_difference_m,
            "bias_scales_m": list(scales),
            "ramp_frames": ramp_frames,
            "maximum_group_norm_m": maximum_group_norm_m,
            "observation_sigma_m": observation_sigma_m,
            "controller_sigma_m": controller_sigma_m,
            "smoothness_sigma_m": smoothness_sigma_m,
            "velocity_history_frames": velocity_history_frames,
            "dt": dt,
            "num_substeps": num_substeps,
            "self_collision": self_collision,
            "deterministic_spring_forces": deterministic_spring_forces,
            "device": device,
            "runtime": _simulator_runtime(),
        },
        "contract": {
            "latent_parameterization": (
                "one persistent 3D translation per inferred hand group with a "
                "linear onset ramp"
            ),
            "selection_observations": (
                "object-point Chamfer over the final training quarter only"
            ),
            "future_observations": "evaluation only, after candidate selection",
            "search": (
                "central finite-difference training-CD gradient followed by a "
                "predeclared signed line search"
            ),
            "prior": (
                "zero-centered controller proximity and temporal second-difference "
                "Gaussian penalties plus a hard per-group norm cap"
            ),
            "future_propagation": "selected persistent bias is held into the future",
            "spring_force_accumulation": (
                "fixed per-vertex incident-spring order without GPU atomics"
                if deterministic_spring_forces
                else "released per-spring GPU atomic accumulation"
            ),
        },
        "inputs": {
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
                "epoch": int(checkpoint.get("epoch", -1)),
            },
            "gt_track_3d": (
                None
                if gt_track_path is None
                else {
                    "path": str(Path(gt_track_path).resolve()),
                    "sha256": _sha256(gt_track_path),
                }
            ),
            "official_repo": {
                "path": str(Path(official_repo).resolve()),
                "commit": _git_commit(official_repo),
            },
        },
        "graph": {
            "object_vertex_count": len(structure_points),
            "object_spring_count": graph.num_object_springs,
            "controller_spring_count": len(graph.springs) - graph.num_object_springs,
            "springs_sha256": _array_hash(graph.springs),
        },
        "controller": {
            "point_count": controller.shape[1],
            "hand_group_count": group_count,
            "hand_group_sizes": [
                int(np.sum(groups == group)) for group in range(group_count)
            ],
        },
        "finite_difference_gradient": {
            "gradient_m_per_m": gradient.tolist(),
            "point_weighted_norm": point_weighted_group_bias_rms(gradient, groups),
            "axes": finite_differences,
        },
        "selection": {
            "selected_candidate_id": selected["candidate_id"],
            "selected_nonzero": selected["candidate_id"] != "zero",
            "selected_group_bias_m": np.asarray(selected["group_bias_m"]).tolist(),
            "selected_objective": selected["objective"],
            "zero_objective": zero_objective,
            "objective_improvement": (
                zero_objective["objective"] - selected["objective"]["objective"]
            ),
            "training_chamfer_percent_change": 100.0
            * (
                selected["objective"]["chamfer_mean_m"]
                / zero_objective["chamfer_mean_m"]
                - 1.0
            ),
            "candidates": serialized_candidates,
        },
        "matched_baseline": {
            "future": _metric_summary(released_metrics, matched_metrics),
            "duplicate_restart_repeatability": repeatability,
            "duplicate_restart_metric_null": _metric_summary(
                matched_metrics, duplicate_metrics
            ),
        },
        "latent_controller": {
            "future_vs_matched": _metric_summary(matched_metrics, latent_metrics),
            "future_vs_released": _metric_summary(released_metrics, latent_metrics),
            "future_object_displacement_vs_matched": _trajectory_error(
                matched_full[train_end_frame - selection_start :],
                selected_full[train_end_frame - selection_start :],
            ),
        },
        "outputs": {"archive": str(archive_path.resolve())},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["outputs"]["summary"] = str(summary_path.resolve())
    simulator.controller_points = torch.as_tensor(
        controller, dtype=torch.float32, device=device
    ).contiguous()
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def run_latent_controller_bias(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str = "development",
    cases: Iterable[str] | None = None,
    validation_fraction: float = 0.75,
    finite_difference_m: float = 0.001,
    bias_scales_m: Iterable[float] = DEFAULT_BIAS_SCALES_M,
    ramp_frames: int = 5,
    maximum_group_norm_m: float = 0.003,
    observation_sigma_m: float = 0.005,
    controller_sigma_m: float = 0.002,
    smoothness_sigma_m: float = 0.0005,
    velocity_history_frames: int = 3,
    dt: float = 5e-5,
    num_substeps: int = 667,
    deterministic_spring_forces: bool = True,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260711,
    force: bool = False,
) -> dict[str, object]:
    """Run the locked training-only latent-controller bias experiment."""

    scales = tuple(sorted(set(float(value) for value in bias_scales_m)))
    root = Path(data_root)
    manifest_path = root / "evaluation_subset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = tuple(str(case) for case in manifest["selected_cases"])
    if cases is not None:
        selected = tuple(dict.fromkeys(str(case) for case in cases))
    elif cohort == "all":
        selected = available
    elif cohort == "development":
        selected = tuple(case for case in available if case in DEVELOPMENT_CASES)
    elif cohort == "confirmation":
        selected = tuple(case for case in available if case not in DEVELOPMENT_CASES)
    else:
        raise ValueError("cohort must be all, development, or confirmation")
    missing = sorted(set(selected) - set(available))
    if missing or not selected:
        raise ValueError("invalid selected cases: " + ", ".join(missing))
    clusters = {case: phystwin_physical_object_cluster(case) for case in selected}
    self_collision_by_case = {
        case: _released_self_collision_for_case(case) for case in selected
    }
    runtime = _simulator_runtime()
    specification = {
        "method": "training-only regularized latent controller bias",
        "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "cohort": cohort,
        "cases": list(selected),
        "validation_fraction": validation_fraction,
        "finite_difference_m": finite_difference_m,
        "bias_scales_m": list(scales),
        "ramp_frames": ramp_frames,
        "maximum_group_norm_m": maximum_group_norm_m,
        "observation_sigma_m": observation_sigma_m,
        "controller_sigma_m": controller_sigma_m,
        "smoothness_sigma_m": smoothness_sigma_m,
        "velocity_history_frames": velocity_history_frames,
        "dt": dt,
        "num_substeps": num_substeps,
        "deterministic_spring_forces": deterministic_spring_forces,
        "future_labels_used_for_selection": False,
        "self_collision_by_case": self_collision_by_case,
        "official_repo": str(Path(official_repo).resolve()),
        "official_commit": _git_commit(official_repo),
        "runtime": runtime,
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": bootstrap_block_length,
            "seed": bootstrap_seed,
        },
        "data_manifest": str(manifest_path.resolve()),
        "status": "post-hoc training-only latent-controller pilot",
    }
    output = Path(output_dir)
    locked = _lock_protocol(output, specification)
    expected_base_config = {
        "validation_fraction": validation_fraction,
        "finite_difference_m": finite_difference_m,
        "bias_scales_m": list(scales),
        "ramp_frames": ramp_frames,
        "maximum_group_norm_m": maximum_group_norm_m,
        "observation_sigma_m": observation_sigma_m,
        "controller_sigma_m": controller_sigma_m,
        "smoothness_sigma_m": smoothness_sigma_m,
        "velocity_history_frames": velocity_history_frames,
        "dt": dt,
        "num_substeps": num_substeps,
        "deterministic_spring_forces": deterministic_spring_forces,
        "device": "cuda:0",
        "runtime": runtime,
    }
    case_results: dict[str, object] = {}
    restart_vs_released = {}
    latent_vs_matched = {}
    latent_vs_released = {}
    duplicate_null = {}
    for case in selected:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_end, future_end = (int(value) for value in split["test"])
        if future_end != int(split["frame_len"]):
            raise ValueError(f"future split does not end at frame_len for {case}")
        selection_start = max(
            velocity_history_frames,
            int(np.floor(validation_fraction * train_end)),
        )
        expected_config = {
            "train_end_frame": train_end,
            **expected_base_config,
            "selection_start_frame": selection_start,
            "self_collision": self_collision_by_case[case],
        }
        case_output = output / "cases" / case
        summary_path = case_output / "summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["config"] != expected_config:
                raise RuntimeError(f"cached case uses a different protocol: {case}")
        else:
            track_path = case_dir / "gt_track_3d.pkl"
            summary = apply_latent_controller_bias(
                official_repo,
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_dir / "optimal_params.pkl",
                case_dir / "checkpoint.pth",
                case_output,
                train_end_frame=train_end,
                gt_track_path=track_path if track_path.exists() else None,
                validation_fraction=validation_fraction,
                finite_difference_m=finite_difference_m,
                bias_scales_m=scales,
                ramp_frames=ramp_frames,
                maximum_group_norm_m=maximum_group_norm_m,
                observation_sigma_m=observation_sigma_m,
                controller_sigma_m=controller_sigma_m,
                smoothness_sigma_m=smoothness_sigma_m,
                velocity_history_frames=velocity_history_frames,
                dt=dt,
                num_substeps=num_substeps,
                self_collision=self_collision_by_case[case],
                deterministic_spring_forces=deterministic_spring_forces,
            )
        released, matched = _future_arrays(summary["matched_baseline"]["future"])
        restart_vs_released[case] = (released, matched)
        duplicate_baseline, duplicate_candidate = _future_arrays(
            summary["matched_baseline"]["duplicate_restart_metric_null"]
        )
        duplicate_null[case] = (duplicate_baseline, duplicate_candidate)
        matched_again, latent = _future_arrays(
            summary["latent_controller"]["future_vs_matched"]
        )
        latent_vs_matched[case] = (matched_again, latent)
        released_again, latent_again = _future_arrays(
            summary["latent_controller"]["future_vs_released"]
        )
        latent_vs_released[case] = (released_again, latent_again)
        case_results[case] = {
            "physical_object": clusters[case],
            "controller": summary["controller"],
            "finite_difference_gradient": summary["finite_difference_gradient"],
            "selection": summary["selection"],
            "matched_baseline": summary["matched_baseline"],
            "latent_controller": summary["latent_controller"],
        }

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
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": specification["status"],
        "cohort": cohort,
        "case_count": len(selected),
        "physical_object_count": len(set(clusters.values())),
        "case_results": case_results,
        "selection": {
            "nonzero_case_count": int(
                sum(case_results[case]["selection"]["selected_nonzero"] for case in selected)
            ),
            "nonzero_case_fraction": float(
                np.mean(
                    [
                        case_results[case]["selection"]["selected_nonzero"]
                        for case in selected
                    ]
                )
            ),
            "training_chamfer_percent_change": _distribution(
                case_results[case]["selection"]["training_chamfer_percent_change"]
                for case in selected
            ),
            "selected_controller_vector_rms_m": _distribution(
                case_results[case]["selection"]["selected_objective"][
                    "controller_vector_rms_m"
                ]
                for case in selected
            ),
        },
        "matched_restart_vs_released": bootstrap(restart_vs_released),
        "latent_controller_vs_matched": bootstrap(latent_vs_matched),
        "latent_controller_vs_released": bootstrap(latent_vs_released),
        "duplicate_restart_metric_null": bootstrap(duplicate_null),
        "interpretation_boundary": {
            "selection": (
                "bias direction and magnitude use training object CD plus fixed "
                "controller priors only"
            ),
            "future_labels": "evaluation only; no future oracle is computed",
            "model_scope": (
                "persistent per-hand translation is a low-dimensional first test, "
                "not a general latent trajectory smoother"
            ),
        },
    }
    result_path = output / "latent_controller_bias_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(result_path.resolve())
    return result
