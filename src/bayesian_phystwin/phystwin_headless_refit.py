"""Headless reliability-aware refitting against the official PhysTwin simulator."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
    spatial_spring_region_ids,
)
from .phystwin_profile import (
    clustered_track_log_likelihood,
    grid_parameter_posterior,
    predictive_observation_calibration,
    truncate_profile_prediction_weights,
)
from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_refit import (
    PhysTwinRefitReliabilityConfig,
    build_phystwin_track_objective,
    evaluate_phystwin_trajectory,
    evaluate_phystwin_trajectory_splits,
    phystwin_tracking_metrics,
)


@dataclass(frozen=True)
class HeadlessPhysTwinRefitConfig:
    """Simulator and optimizer settings for one matched refit variant."""

    variant: str
    train_end_frame: int
    fit_end_frame: int | None = None
    epochs: int = 0
    learning_rate: float = 1e-4
    observation_variance: float = 2.5e-5
    model_discrepancy_variance: float = 0.0
    outlier_variance_multiplier: float = 100.0
    flow_scale: float = 0.005
    boundary_scale: float = 0.03
    dt: float = 5e-5
    num_substeps: int = 667
    track_weight: float = 1.0
    acceleration_weight: float = 0.01
    optimize_collision: bool = True
    spring_parameterization: str = "dense"
    spring_region_count: int = 4
    spring_scale_weight_decay: float = 0.0
    dashpot_log_scale: float = 0.0
    drag_log_scale: float = 0.0
    selection_metric: str = "hard_valid_rmse"
    early_stopping_patience: int = 3
    profile_grid_count: int = 0
    profile_object_log_scale_half_width: float = 0.30
    profile_controller_log_scale_half_width: float = 1.00
    profile_object_prior_std: float = 0.15
    profile_controller_prior_std: float = 0.50
    profile_likelihood_temperature: float = 1.0
    profile_prediction_mass: float = 1.0
    device: str = "cuda:0"


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _spring_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "geometric_mean": float(np.exp(np.mean(np.log(values)))),
        "maximum": float(np.max(values)),
    }


def _common_objective_metrics(
    observed: np.ndarray,
    trajectory: np.ndarray,
    weights: np.ndarray,
    support: np.ndarray,
    *,
    train_end_frame: int,
    observation_variance: float,
    outlier_variance_multiplier: float,
    prior: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    residual = observed - trajectory[: len(observed), : observed.shape[1]]
    squared_norm = np.sum(np.square(residual), axis=2)
    result: dict[str, dict[str, float | int]] = {}
    for name, start, stop in (
        ("train", 1, train_end_frame),
        ("test", train_end_frame, len(observed)),
    ):
        split_weights = weights[start:stop]
        split_support = support[start:stop].astype(bool)
        weight_sum = float(np.sum(split_weights))
        weighted_rmse = (
            float(
                np.sqrt(
                    np.sum(split_weights * squared_norm[start:stop]) / weight_sum
                )
            )
            if weight_sum > 0.0
            else float("nan")
        )

        selected_q = squared_norm[start:stop][split_support]
        selected_prior = prior[start:stop][split_support]
        if len(selected_q):
            log_inlier = np.log(selected_prior) - 0.5 * selected_q / observation_variance
            log_outlier = (
                np.log1p(-selected_prior)
                - 1.5 * np.log(outlier_variance_multiplier)
                - 0.5
                * selected_q
                / (observation_variance * outlier_variance_multiplier)
            )
            zero_log_mixture = np.logaddexp(
                np.log(selected_prior),
                np.log1p(-selected_prior)
                - 1.5 * np.log(outlier_variance_multiplier),
            )
            shifted_nll = -observation_variance * (
                np.logaddexp(log_inlier, log_outlier) - zero_log_mixture
            )
            mixture_loss = float(np.mean(np.maximum(shifted_nll, 0.0)) / 3.0)
        else:
            mixture_loss = float("nan")
        result[name] = {
            "support_count": int(np.sum(split_support)),
            "effective_weight": weight_sum,
            "weighted_vector_rmse_m": weighted_rmse,
            "shifted_scaled_mixture_loss": mixture_loss,
        }
    return result


def _load_checkpoint(torch: Any, path: str | Path, device: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return checkpoint


def run_headless_phystwin_refit(
    *,
    official_repo: str | Path,
    final_data_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    cues_path: str | Path,
    output_dir: str | Path,
    config: HeadlessPhysTwinRefitConfig,
    released_trajectory_path: str | Path | None = None,
    gt_track_path: str | Path | None = None,
    profile_weights_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one refit and write its trajectory, checkpoint, history, and summary."""

    if config.epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.observation_variance <= 0.0:
        raise ValueError("observation_variance must be positive")
    if config.model_discrepancy_variance < 0.0:
        raise ValueError("model_discrepancy_variance must be nonnegative")
    if config.outlier_variance_multiplier <= 1.0:
        raise ValueError("outlier_variance_multiplier must be greater than one")
    if config.flow_scale <= 0.0 or config.boundary_scale <= 0.0:
        raise ValueError("cue scales must be positive")
    if config.num_substeps < 1 or config.dt <= 0.0:
        raise ValueError("simulator time discretization must be positive")
    if config.spring_parameterization not in {"dense", "grouped", "regional"}:
        raise ValueError(
            "spring_parameterization must be 'dense', 'grouped', or 'regional'"
        )
    if config.spring_region_count < 2:
        raise ValueError("spring_region_count must be at least two")
    if config.spring_scale_weight_decay < 0.0:
        raise ValueError("spring_scale_weight_decay must be nonnegative")
    if not np.isfinite(config.dashpot_log_scale) or not np.isfinite(
        config.drag_log_scale
    ):
        raise ValueError("damping log scales must be finite")
    if config.selection_metric not in {"hard_valid_rmse", "official_3d"}:
        raise ValueError(
            "selection_metric must be 'hard_valid_rmse' or 'official_3d'"
        )
    if config.selection_metric == "official_3d" and gt_track_path is None:
        raise ValueError("official_3d selection requires gt_track_path")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if config.profile_grid_count < 0:
        raise ValueError("profile_grid_count must be nonnegative")
    if config.profile_grid_count and config.profile_grid_count < 3:
        raise ValueError("profile_grid_count must be zero or at least three")
    if profile_weights_path is not None and not config.profile_grid_count:
        raise ValueError("profile_weights_path requires profile_grid_count")
    profile_positive_values = (
        config.profile_object_log_scale_half_width,
        config.profile_controller_log_scale_half_width,
        config.profile_object_prior_std,
        config.profile_controller_prior_std,
        config.profile_likelihood_temperature,
    )
    if any(value <= 0.0 for value in profile_positive_values):
        raise ValueError("profile widths, prior scales, and temperature must be positive")
    if not 0.0 < config.profile_prediction_mass <= 1.0:
        raise ValueError("profile_prediction_mass must be in (0, 1]")
    if config.profile_grid_count:
        if config.spring_parameterization != "grouped":
            raise ValueError("parameter profiling requires grouped springs")
        if config.epochs != 0:
            raise ValueError("parameter profiling requires epochs=0")
        if config.optimize_collision:
            raise ValueError("parameter profiling requires frozen collision parameters")
    if config.device != "cuda:0":
        raise ValueError(
            "the pinned official simulator selects cuda:0 at import; use "
            "CUDA_VISIBLE_DEVICES to remap another GPU"
        )

    try:
        import torch
        import warp as wp
    except ImportError as error:
        raise RuntimeError(
            "headless PhysTwin refits require compatible torch and warp installs"
        ) from error
    from ._phystwin_warp_backend import (
        load_official_spring_mass_module,
        make_reliability_simulator_class,
    )

    started = time.time()
    final_data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_params_path)
    if not isinstance(final_data, dict) or not isinstance(optimal, dict):
        raise ValueError("final_data and optimal_params must contain dictionaries")
    required = {
        "object_points",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
        "surface_points",
        "interior_points",
    }
    missing = required - set(final_data)
    if missing:
        raise ValueError(f"final_data is missing keys: {', '.join(sorted(missing))}")

    object_points = np.asarray(final_data["object_points"], dtype=np.float32)
    visible = np.asarray(final_data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(final_data["object_motions_valid"], dtype=bool)
    controller_points = np.asarray(final_data["controller_points"], dtype=np.float32)
    surface_points = np.asarray(final_data["surface_points"], dtype=np.float32)
    interior_points = np.asarray(final_data["interior_points"], dtype=np.float32)
    gt_track_3d = (
        None
        if gt_track_path is None
        else np.asarray(_load_pickle(gt_track_path), dtype=np.float32)
    )
    frame_count, original_count, coordinate_count = object_points.shape
    if coordinate_count != 3:
        raise ValueError("object_points must have shape (T, N, 3)")
    if not 1 < config.train_end_frame < frame_count:
        raise ValueError("train_end_frame must be between 2 and T-1")
    fit_end_frame = (
        config.train_end_frame
        if config.fit_end_frame is None
        else config.fit_end_frame
    )
    if config.fit_end_frame is not None and not 1 < fit_end_frame < config.train_end_frame:
        raise ValueError("fit_end_frame must be between 2 and train_end_frame-1")
    if config.profile_grid_count and config.fit_end_frame is None:
        raise ValueError("parameter profiling requires fit_end_frame")

    with np.load(cues_path) as archive:
        cues = {name: np.asarray(archive[name]) for name in archive.files}
    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues=cues,
        variant=config.variant,
        config=PhysTwinRefitReliabilityConfig(
            flow_scale=config.flow_scale,
            boundary_scale=config.boundary_scale,
        ),
    )

    structure_points = np.concatenate(
        (object_points[0], surface_points, interior_points),
        axis=0,
    )
    num_surface_points = original_count + len(surface_points)
    graph = build_phystwin_spring_graph(
        structure_points,
        controller_points[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    spring_group_ids = None
    if config.spring_parameterization == "regional":
        spring_group_ids = spatial_spring_region_ids(
            graph.vertices,
            graph.springs,
            num_object_springs=graph.num_object_springs,
            region_count=config.spring_region_count,
        )
    checkpoint = _load_checkpoint(torch, checkpoint_path, config.device)
    checkpoint_spring_y = torch.as_tensor(
        checkpoint["spring_Y"], dtype=torch.float32, device=config.device
    ).reshape(-1)
    if len(checkpoint_spring_y) != len(graph.springs):
        raise ValueError(
            "reconstructed graph and checkpoint disagree: "
            f"{len(graph.springs)} versus {len(checkpoint_spring_y)} springs"
        )
    if int(checkpoint["num_object_springs"]) != graph.num_object_springs:
        raise ValueError("reconstructed object spring count disagrees with checkpoint")

    runtime_cfg = SimpleNamespace(
        device=config.device,
        use_graph=True,
        data_type="real",
        collision_learn=config.optimize_collision,
        chamfer_weight=0.0,
        track_weight=config.track_weight,
        acc_weight=config.acceleration_weight,
    )
    official = load_official_spring_mass_module(
        official_repo,
        runtime_config=runtime_cfg,
    )
    simulator_class = make_reliability_simulator_class(official)
    warnings.filterwarnings(
        "ignore",
        message=(
            "Running the tape backwards may produce incorrect gradients because "
            "recorded kernel set_control_points.*"
        ),
    )

    def tensor(values: np.ndarray, dtype: Any) -> Any:
        return torch.as_tensor(values, dtype=dtype, device=config.device).contiguous()

    torch_object_points = tensor(object_points, torch.float32)
    torch_visible = tensor(visible.astype(np.int32), torch.int32)
    torch_motion_valid = tensor(motion_valid.astype(np.int32), torch.int32)
    torch_controller = tensor(controller_points, torch.float32)
    simulator = simulator_class(
        tensor(graph.vertices, torch.float32),
        tensor(graph.springs, torch.int32),
        tensor(graph.rest_lengths, torch.float32),
        tensor(graph.masses, torch.float32),
        dt=config.dt,
        num_substeps=config.num_substeps,
        spring_Y=float(optimal["global_spring_Y"]),
        collide_elas=float(optimal["collide_elas"]),
        collide_fric=float(optimal["collide_fric"]),
        dashpot_damping=float(
            optimal["dashpot_damping"] * np.exp(config.dashpot_log_scale)
        ),
        drag_damping=float(
            optimal["drag_damping"] * np.exp(config.drag_log_scale)
        ),
        collide_object_elas=float(optimal["collide_object_elas"]),
        collide_object_fric=float(optimal["collide_object_fric"]),
        collision_dist=float(optimal["collision_dist"]),
        num_object_points=len(structure_points),
        num_surface_points=num_surface_points,
        num_original_points=original_count,
        controller_points=torch_controller,
        reverse_z=True,
        spring_Y_min=0.0,
        spring_Y_max=1e5,
        gt_object_points=torch_object_points,
        gt_object_visibilities=torch_visible,
        gt_object_motions_valid=torch_motion_valid,
        self_collision=False,
        disable_backward=config.epochs == 0,
        objective=objective,
        observation_variance=(
            config.observation_variance + config.model_discrepancy_variance
        ),
        outlier_variance_multiplier=config.outlier_variance_multiplier,
        spring_parameterization=config.spring_parameterization,
        num_object_springs=graph.num_object_springs,
        spring_group_ids=spring_group_ids,
    )
    simulator.set_reference_spring_y(
        torch.log(checkpoint_spring_y).detach().clone()
    )

    def checkpoint_value(name: str) -> Any:
        return torch.as_tensor(
            checkpoint[name], dtype=torch.float32, device=config.device
        ).reshape(-1)

    simulator.set_collide(checkpoint_value("collide_elas"), checkpoint_value("collide_fric"))
    simulator.set_collide_object(
        checkpoint_value("collide_object_elas"),
        checkpoint_value("collide_object_fric"),
    )
    wp.synchronize()
    initial_spring_y = checkpoint_spring_y.detach().cpu().numpy().copy()

    def simulate_trajectory(stop_frame: int) -> np.ndarray:
        simulator.set_init_state(
            simulator.wp_init_vertices,
            simulator.wp_init_velocities,
            pure_inference=True,
        )
        frames = [
            wp.to_torch(simulator.wp_states[0].wp_x)
            .detach()
            .cpu()
            .numpy()
            .copy()
        ]
        for frame in range(1, stop_frame):
            simulator.set_controller_target(frame, pure_inference=True)
            if simulator.object_collision_flag:
                simulator.update_collision_graph()
            wp.capture_launch(simulator.forward_graph)
            wp.synchronize()
            frames.append(
                wp.to_torch(simulator.wp_states[-1].wp_x)
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v,
                pure_inference=True,
            )
        return np.stack(frames).astype(np.float32)

    def snapshot_parameters() -> dict[str, Any]:
        return {
            "spring_log_y": wp.to_torch(simulator.wp_spring_Y).detach().clone(),
            "group_log_scales": wp.to_torch(simulator.wp_group_log_scales)
            .detach()
            .clone(),
            "collide_elas": wp.to_torch(simulator.wp_collide_elas).detach().clone(),
            "collide_fric": wp.to_torch(simulator.wp_collide_fric).detach().clone(),
            "collide_object_elas": wp.to_torch(simulator.wp_collide_object_elas)
            .detach()
            .clone(),
            "collide_object_fric": wp.to_torch(simulator.wp_collide_object_fric)
            .detach()
            .clone(),
        }

    def restore_parameters(parameters: dict[str, Any]) -> None:
        if config.spring_parameterization != "dense":
            with torch.no_grad():
                simulator.group_log_scale_tensor.copy_(
                    parameters["group_log_scales"]
                )
        else:
            simulator.set_spring_Y(parameters["spring_log_y"])
        simulator.set_collide(
            parameters["collide_elas"],
            parameters["collide_fric"],
        )
        simulator.set_collide_object(
            parameters["collide_object_elas"],
            parameters["collide_object_fric"],
        )
        wp.synchronize()

    hard_objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        variant="hard",
    )

    def validation_metrics(trajectory_value: np.ndarray) -> dict[str, float]:
        mask = np.zeros((config.train_end_frame, original_count), dtype=bool)
        mask[fit_end_frame : config.train_end_frame] = hard_objective.support[
            fit_end_frame : config.train_end_frame
        ].astype(bool)
        metrics = phystwin_tracking_metrics(
            object_points[: config.train_end_frame],
            trajectory_value,
            mask,
        )
        if metrics["count"] == 0:
            raise ValueError("validation interval contains no hard-valid tracks")
        result = {
            "hard_valid_vector_rmse_m": float(metrics["vector_rmse_m"]),
        }
        if gt_track_3d is not None:
            official_metrics = evaluate_official_phystwin_interval(
                trajectory_value,
                object_points,
                visible,
                gt_track_3d,
                num_surface_points=num_surface_points,
                start_frame=fit_end_frame,
                end_frame=config.train_end_frame,
            )
            result["official_chamfer_distance_m"] = float(
                official_metrics["chamfer_distance_m"]
            )
            result["official_track_error_m"] = float(
                official_metrics["track_error_m"]
            )
        return result

    def validation_score(
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        if config.selection_metric == "hard_valid_rmse":
            return metrics["hard_valid_vector_rmse_m"]
        return 0.5 * (
            metrics["official_chamfer_distance_m"]
            / baseline["official_chamfer_distance_m"]
            + metrics["official_track_error_m"]
            / baseline["official_track_error_m"]
        )

    history: list[dict[str, float | int]] = []
    selected_epoch = config.epochs - 1
    baseline_trajectory = None
    baseline_validation_metrics = None
    best_validation_metrics = None
    best_validation_score = None
    best_parameters = None
    stale_epochs = 0
    if config.fit_end_frame is not None:
        baseline_trajectory = simulate_trajectory(frame_count)
        baseline_validation_metrics = validation_metrics(baseline_trajectory)
        best_validation_metrics = baseline_validation_metrics
        best_validation_score = validation_score(
            baseline_validation_metrics,
            baseline_validation_metrics,
        )
        best_parameters = snapshot_parameters()
        selected_epoch = -1
    if config.epochs:
        optimizer_parameter_groups: list[dict[str, object]] = []
        if config.spring_parameterization != "dense":
            optimizer_parameter_groups.append(
                {
                    "params": [wp.to_torch(simulator.wp_group_log_scales)],
                    "weight_decay": config.spring_scale_weight_decay,
                }
            )
        else:
            optimizer_parameter_groups.append(
                {"params": [wp.to_torch(simulator.wp_spring_Y)]}
            )
        if config.optimize_collision:
            optimizer_parameter_groups.append(
                {
                    "params": [
                        wp.to_torch(simulator.wp_collide_elas),
                        wp.to_torch(simulator.wp_collide_fric),
                        wp.to_torch(simulator.wp_collide_object_elas),
                        wp.to_torch(simulator.wp_collide_object_fric),
                    ],
                    "weight_decay": 0.0,
                }
            )
        optimizer = torch.optim.Adam(
            optimizer_parameter_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.99),
        )
        for epoch in range(config.epochs):
            simulator.set_init_state(
                simulator.wp_init_vertices,
                simulator.wp_init_velocities,
            )
            total_loss = 0.0
            total_track_loss = 0.0
            for frame in range(1, fit_end_frame):
                simulator.set_controller_target(frame)
                if simulator.object_collision_flag:
                    simulator.update_collision_graph()
                wp.capture_launch(simulator.graph)
                wp.synchronize()
                total_loss += float(wp.to_torch(simulator.loss).item())
                total_track_loss += float(wp.to_torch(simulator.track_loss).item())
                optimizer.step()
                simulator.tape.zero()
                simulator.clear_loss()
                simulator.set_init_state(
                    simulator.wp_states[-1].wp_x,
                    simulator.wp_states[-1].wp_v,
                )
            denominator = fit_end_frame - 1
            epoch_result: dict[str, float | int] = {
                "epoch": epoch,
                "mean_loss": total_loss / denominator,
                "mean_track_loss": total_track_loss / denominator,
            }
            if config.fit_end_frame is not None:
                current_validation_metrics = validation_metrics(
                    simulate_trajectory(config.train_end_frame)
                )
                current_validation_score = validation_score(
                    current_validation_metrics,
                    baseline_validation_metrics,
                )
                epoch_result.update(
                    {
                        f"validation_{name}": value
                        for name, value in current_validation_metrics.items()
                    }
                )
                epoch_result["validation_selection_score"] = (
                    current_validation_score
                )
                if current_validation_score < float(best_validation_score):
                    best_validation_score = current_validation_score
                    best_validation_metrics = current_validation_metrics
                    best_parameters = snapshot_parameters()
                    selected_epoch = epoch
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            history.append(epoch_result)
            if (
                config.fit_end_frame is not None
                and stale_epochs >= config.early_stopping_patience
            ):
                break

    if best_parameters is not None:
        restore_parameters(best_parameters)
    trajectory = simulate_trajectory(frame_count)

    final_spring_y = (
        torch.exp(wp.to_torch(simulator.wp_spring_Y)).detach().cpu().numpy().copy()
    )
    final_group_log_scales = (
        wp.to_torch(simulator.wp_group_log_scales)
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    final_collision = {
        name: float(wp.to_torch(getattr(simulator, f"wp_{name}")).item())
        for name in (
            "collide_elas",
            "collide_fric",
            "collide_object_elas",
            "collide_object_fric",
        )
    }
    evaluation = evaluate_phystwin_trajectory(
        object_points,
        trajectory,
        visible,
        motion_valid,
        train_end_frame=config.train_end_frame,
    )
    selection_evaluation = None
    baseline_evaluation = None
    if config.fit_end_frame is not None:
        selection_evaluation = evaluate_phystwin_trajectory_splits(
            object_points,
            trajectory,
            visible,
            motion_valid,
            splits={
                "fit": (1, fit_end_frame),
                "validation": (fit_end_frame, config.train_end_frame),
                "test": (config.train_end_frame, frame_count),
            },
        )
        baseline_evaluation = evaluate_phystwin_trajectory_splits(
            object_points,
            baseline_trajectory,
            visible,
            motion_valid,
            splits={
                "fit": (1, fit_end_frame),
                "validation": (fit_end_frame, config.train_end_frame),
                "test": (config.train_end_frame, frame_count),
            },
        )

    def official_split_evaluation(
        trajectory_value: np.ndarray,
    ) -> dict[str, dict[str, object]] | None:
        if gt_track_3d is None:
            return None
        split_intervals = (
            {
                "train": (1, config.train_end_frame),
                "test": (config.train_end_frame, frame_count),
            }
            if config.fit_end_frame is None
            else {
                "fit": (1, fit_end_frame),
                "validation": (fit_end_frame, config.train_end_frame),
                "test": (config.train_end_frame, frame_count),
            }
        )
        return {
            name: evaluate_official_phystwin_interval(
                trajectory_value,
                object_points,
                visible,
                gt_track_3d,
                num_surface_points=num_surface_points,
                start_frame=start,
                end_frame=stop,
            )
            for name, (start, stop) in split_intervals.items()
        }

    official_evaluation = official_split_evaluation(trajectory)
    baseline_official_evaluation = (
        None
        if baseline_trajectory is None
        else official_split_evaluation(baseline_trajectory)
    )
    common_objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues=cues,
        variant="cue",
        config=PhysTwinRefitReliabilityConfig(
            flow_scale=config.flow_scale,
            boundary_scale=config.boundary_scale,
        ),
    )
    common_metrics = _common_objective_metrics(
        object_points,
        trajectory,
        common_objective.weights,
        common_objective.support,
        train_end_frame=config.train_end_frame,
        observation_variance=(
            config.observation_variance + config.model_discrepancy_variance
        ),
        outlier_variance_multiplier=config.outlier_variance_multiplier,
        prior=common_objective.prior_inlier_probability,
    )

    profile_summary = None
    profile_artifact: dict[str, np.ndarray] | None = None
    if config.profile_grid_count:
        object_scale_grid = np.linspace(
            -config.profile_object_log_scale_half_width,
            config.profile_object_log_scale_half_width,
            config.profile_grid_count,
        )
        controller_scale_grid = np.linspace(
            -config.profile_controller_log_scale_half_width,
            config.profile_controller_log_scale_half_width,
            config.profile_grid_count,
        )
        external_profile = None
        if profile_weights_path is not None:
            with np.load(profile_weights_path) as archive:
                external_profile = {
                    name: np.asarray(archive[name]) for name in archive.files
                }
            required_external = {
                "object_log_scales",
                "controller_log_scales",
                "log_likelihood",
                "posterior_weights",
            }
            missing_external = required_external - set(external_profile)
            if missing_external:
                raise ValueError(
                    "profile weights are missing: "
                    + ", ".join(sorted(missing_external))
                )
            if not (
                np.array_equal(
                    external_profile["object_log_scales"], object_scale_grid
                )
                and np.array_equal(
                    external_profile["controller_log_scales"],
                    controller_scale_grid,
                )
            ):
                raise ValueError("external profile weights do not match config grids")
            log_likelihood = np.asarray(
                external_profile["log_likelihood"], dtype=float
            )
        else:
            log_likelihood = np.empty(
                (config.profile_grid_count, config.profile_grid_count),
                dtype=float,
            )
            for object_index, object_scale in enumerate(object_scale_grid):
                for controller_index, controller_scale in enumerate(
                    controller_scale_grid
                ):
                    with torch.no_grad():
                        simulator.group_log_scale_tensor.copy_(
                            torch.tensor(
                                [object_scale, controller_scale],
                                dtype=torch.float32,
                                device=config.device,
                            )
                        )
                    candidate = simulate_trajectory(fit_end_frame)[
                        :, :original_count
                    ]
                    log_likelihood[object_index, controller_index] = (
                        clustered_track_log_likelihood(
                            object_points,
                            candidate,
                            objective,
                            start_frame=1,
                            end_frame=fit_end_frame,
                            variance=(
                                config.observation_variance
                                + config.model_discrepancy_variance
                            ),
                            outlier_variance_multiplier=(
                                config.outlier_variance_multiplier
                            ),
                            temperature=config.profile_likelihood_temperature,
                        )
                    )
        posterior = grid_parameter_posterior(
            object_scale_grid,
            controller_scale_grid,
            log_likelihood,
            object_prior_std=config.profile_object_prior_std,
            controller_prior_std=config.profile_controller_prior_std,
        )
        prediction_weights = posterior.weights
        if external_profile is not None:
            prediction_weights = np.asarray(
                external_profile["posterior_weights"], dtype=float
            )
            if prediction_weights.shape != posterior.weights.shape:
                raise ValueError("external posterior_weights have the wrong shape")
            if not np.all(np.isfinite(prediction_weights)) or np.any(
                prediction_weights < 0.0
            ):
                raise ValueError("external posterior_weights must be finite and nonnegative")
            weight_sum = float(np.sum(prediction_weights))
            if weight_sum <= 0.0:
                raise ValueError("external posterior_weights must have positive mass")
            prediction_weights = prediction_weights / weight_sum
        source_prediction_weights = prediction_weights.copy()
        prediction_weights, retained_prediction_mass, prediction_particle_count = (
            truncate_profile_prediction_weights(
                prediction_weights,
                retained_mass=config.profile_prediction_mass,
            )
        )
        state_shape = trajectory.shape
        posterior_mean_accumulator = np.zeros(state_shape, dtype=np.float64)
        posterior_second_moment = np.zeros(state_shape, dtype=np.float64)
        map_flat_index = int(np.argmax(prediction_weights))
        map_trajectory = None
        flat_index = 0
        for object_index, object_scale in enumerate(object_scale_grid):
            for controller_index, controller_scale in enumerate(
                controller_scale_grid
            ):
                with torch.no_grad():
                    simulator.group_log_scale_tensor.copy_(
                        torch.tensor(
                            [object_scale, controller_scale],
                            dtype=torch.float32,
                            device=config.device,
                        )
                    )
                weight = float(prediction_weights[object_index, controller_index])
                if weight <= 0.0:
                    flat_index += 1
                    continue
                candidate = simulate_trajectory(frame_count).astype(np.float64)
                posterior_mean_accumulator += weight * candidate
                posterior_second_moment += weight * np.square(candidate)
                if flat_index == map_flat_index:
                    map_trajectory = candidate.astype(np.float32)
                flat_index += 1
        posterior_mean = posterior_mean_accumulator.astype(np.float32)
        epistemic_variance = np.maximum(
            posterior_second_moment - np.square(posterior_mean_accumulator),
            0.0,
        ).astype(np.float32)
        assert map_trajectory is not None
        posterior_evaluation = evaluate_phystwin_trajectory_splits(
            object_points,
            posterior_mean,
            visible,
            motion_valid,
            splits={
                "fit": (1, fit_end_frame),
                "validation": (fit_end_frame, config.train_end_frame),
                "test": (config.train_end_frame, frame_count),
            },
        )
        posterior_calibration: dict[str, dict[str, float | int]] = {}
        reference_calibration: dict[str, dict[str, float | int]] = {}
        for split_name, split_start, split_stop in (
            ("fit", 1, fit_end_frame),
            ("validation", fit_end_frame, config.train_end_frame),
            ("test", config.train_end_frame, frame_count),
        ):
            split_mask = np.zeros_like(visible)
            split_mask[split_start:split_stop] = hard_objective.support[
                split_start:split_stop
            ].astype(bool)
            posterior_calibration[split_name] = predictive_observation_calibration(
                object_points,
                posterior_mean[:, :original_count],
                epistemic_variance[:, :original_count],
                split_mask,
                observation_variance=config.observation_variance,
                model_discrepancy_variance=(
                    config.model_discrepancy_variance
                ),
            )
            reference_calibration[split_name] = predictive_observation_calibration(
                object_points,
                trajectory[:, :original_count],
                np.zeros_like(object_points),
                split_mask,
                observation_variance=config.observation_variance,
                model_discrepancy_variance=(
                    config.model_discrepancy_variance
                ),
            )
        profile_summary = {
            "particle_count": int(config.profile_grid_count**2),
            "state_vertex_count": int(trajectory.shape[1]),
            "fit_frame_interval": [1, fit_end_frame],
            "likelihood_variant": config.variant,
            "cluster_contract": "mean tracks within frame, sum frames",
            "posterior": posterior.summary,
            "prediction_weight_source": (
                "local_posterior"
                if profile_weights_path is None
                else str(Path(profile_weights_path).resolve())
            ),
            "prediction_effective_grid_points": float(
                1.0 / np.sum(np.square(prediction_weights))
            ),
            "prediction_requested_mass": float(config.profile_prediction_mass),
            "prediction_retained_mass": retained_prediction_mass,
            "prediction_particle_count": prediction_particle_count,
            "prediction_object_log_scale_mean": float(
                np.sum(np.sum(prediction_weights, axis=1) * object_scale_grid)
            ),
            "prediction_controller_log_scale_mean": float(
                np.sum(
                    np.sum(prediction_weights, axis=0) * controller_scale_grid
                )
            ),
            "log_likelihood_minimum": float(np.min(log_likelihood)),
            "log_likelihood_maximum": float(np.max(log_likelihood)),
            "posterior_mean_evaluation": posterior_evaluation,
            "posterior_mean_official_evaluation": official_split_evaluation(
                posterior_mean
            ),
            "posterior_predictive_calibration": posterior_calibration,
            "reference_predictive_calibration": reference_calibration,
        }
        profile_artifact = {
            "object_log_scales": object_scale_grid,
            "controller_log_scales": controller_scale_grid,
            "log_likelihood": log_likelihood,
            "log_posterior": posterior.log_posterior,
            "posterior_weights": posterior.weights,
            "source_prediction_weights": source_prediction_weights,
            "prediction_weights": prediction_weights,
            "posterior_mean_trajectory": posterior_mean,
            "epistemic_variance": epistemic_variance,
            "map_trajectory": map_trajectory,
        }
        with torch.no_grad():
            simulator.group_log_scale_tensor.zero_()

    released_evaluation = None
    released_split_evaluation = None
    released_official_evaluation = None
    released_parity = None
    released_baseline_parity = None
    selected_baseline_parity = None
    if baseline_trajectory is not None:
        selected_baseline_parity = phystwin_tracking_metrics(
            baseline_trajectory,
            trajectory,
            np.ones(baseline_trajectory.shape[:2], dtype=bool),
        )
    if released_trajectory_path is not None:
        released = np.asarray(_load_pickle(released_trajectory_path), dtype=np.float32)
        released_evaluation = evaluate_phystwin_trajectory(
            object_points,
            released,
            visible,
            motion_valid,
            train_end_frame=config.train_end_frame,
        )
        if config.fit_end_frame is not None:
            released_split_evaluation = evaluate_phystwin_trajectory_splits(
                object_points,
                released,
                visible,
                motion_valid,
                splits={
                    "fit": (1, fit_end_frame),
                    "validation": (fit_end_frame, config.train_end_frame),
                    "test": (config.train_end_frame, frame_count),
                },
            )
        released_official_evaluation = official_split_evaluation(released)
        parity_frames = min(len(released), len(trajectory))
        parity_vertices = min(released.shape[1], trajectory.shape[1])
        released_parity = phystwin_tracking_metrics(
            released[:parity_frames, :parity_vertices],
            trajectory[:parity_frames, :parity_vertices],
            np.ones((parity_frames, parity_vertices), dtype=bool),
        )
        if baseline_trajectory is not None:
            baseline_frames = min(len(released), len(baseline_trajectory))
            baseline_vertices = min(
                released.shape[1], baseline_trajectory.shape[1]
            )
            released_baseline_parity = phystwin_tracking_metrics(
                released[:baseline_frames, :baseline_vertices],
                baseline_trajectory[:baseline_frames, :baseline_vertices],
                np.ones((baseline_frames, baseline_vertices), dtype=bool),
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    profile_path = None
    if profile_artifact is not None:
        profile_path = output_path / "parameter_profile.npz"
        np.savez_compressed(profile_path, **profile_artifact)
    trajectory_path = output_path / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    baseline_trajectory_path = None
    if baseline_trajectory is not None:
        baseline_trajectory_path = output_path / "baseline_trajectory.pkl"
        with baseline_trajectory_path.open("wb") as handle:
            pickle.dump(
                baseline_trajectory,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    history_path = output_path / "history.json"
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refit_checkpoint_path = output_path / "refit_checkpoint.pt"
    torch.save(
        {
            "spring_Y": torch.as_tensor(final_spring_y),
            "collide_elas": torch.tensor([final_collision["collide_elas"]]),
            "collide_fric": torch.tensor([final_collision["collide_fric"]]),
            "collide_object_elas": torch.tensor(
                [final_collision["collide_object_elas"]]
            ),
            "collide_object_fric": torch.tensor(
                [final_collision["collide_object_fric"]]
            ),
            "num_object_springs": graph.num_object_springs,
            "variant": config.variant,
            "spring_parameterization": config.spring_parameterization,
            "group_log_scales": torch.as_tensor(final_group_log_scales),
            "spring_group_ids": (
                None
                if spring_group_ids is None
                else torch.as_tensor(spring_group_ids)
            ),
            "source_checkpoint": str(Path(checkpoint_path).resolve()),
        },
        refit_checkpoint_path,
    )

    summary = {
        "schema_version": 1,
        "config": asdict(config),
        "runtime_seconds": float(time.time() - started),
        "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "official_commit": _git_commit(official_repo),
        "runtime": {
            "torch_version": torch.__version__,
            "warp_version": wp.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(config.device),
        },
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "optimal_params": {
                "path": str(Path(optimal_params_path).resolve()),
                "sha256": _sha256(optimal_params_path),
            },
            "checkpoint": {
                "path": str(Path(checkpoint_path).resolve()),
                "sha256": _sha256(checkpoint_path),
            },
            "cues": {
                "path": str(Path(cues_path).resolve()),
                "sha256": _sha256(cues_path),
            },
            "gt_track_3d": (
                None
                if gt_track_path is None
                else {
                    "path": str(Path(gt_track_path).resolve()),
                    "sha256": _sha256(gt_track_path),
                }
            ),
            "profile_weights": (
                None
                if profile_weights_path is None
                else {
                    "path": str(Path(profile_weights_path).resolve()),
                    "sha256": _sha256(profile_weights_path),
                }
            ),
        },
        "graph": {
            "vertex_count": int(len(graph.vertices)),
            "spring_count": int(len(graph.springs)),
            "object_spring_count": int(graph.num_object_springs),
            "controller_spring_count": int(
                len(graph.springs) - graph.num_object_springs
            ),
            "springs_sha256": _array_hash(graph.springs),
            "rest_lengths_sha256": _array_hash(graph.rest_lengths),
            "spring_group_ids_sha256": (
                None
                if spring_group_ids is None
                else _array_hash(spring_group_ids)
            ),
            "spring_group_counts": (
                None
                if spring_group_ids is None
                else np.bincount(spring_group_ids).astype(int).tolist()
            ),
        },
        "parameters": {
            "initial_spring_y": _spring_summary(initial_spring_y),
            "final_spring_y": _spring_summary(final_spring_y),
            "log_spring_rms_change": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.log(final_spring_y) - np.log(initial_spring_y)
                        )
                    )
                )
            ),
            "group_log_scales": (
                {
                    "regions": [
                        float(value)
                        for value in final_group_log_scales[
                            : config.spring_region_count
                        ]
                    ],
                    "controller": float(
                        final_group_log_scales[config.spring_region_count]
                    ),
                }
                if config.spring_parameterization == "regional"
                else {
                    "object": float(final_group_log_scales[0]),
                    "controller": float(final_group_log_scales[1]),
                }
            ),
            "final_collision": final_collision,
            "fixed_dashpot_damping": float(
                optimal["dashpot_damping"] * np.exp(config.dashpot_log_scale)
            ),
            "fixed_drag_damping": float(
                optimal["drag_damping"] * np.exp(config.drag_log_scale)
            ),
        },
        "history": history,
        "selection": {
            "selected_epoch": selected_epoch,
            "metric": config.selection_metric,
            "baseline_metrics": baseline_validation_metrics,
            "best_metrics": best_validation_metrics,
            "baseline_score": (
                None
                if baseline_validation_metrics is None
                else validation_score(
                    baseline_validation_metrics,
                    baseline_validation_metrics,
                )
            ),
            "best_score": best_validation_score,
            "baseline_validation_hard_valid_vector_rmse_m": (
                None
                if baseline_validation_metrics is None
                else baseline_validation_metrics["hard_valid_vector_rmse_m"]
            ),
            "best_validation_hard_valid_vector_rmse_m": (
                None
                if best_validation_metrics is None
                else best_validation_metrics["hard_valid_vector_rmse_m"]
            ),
            "restored_best_parameters": best_parameters is not None,
        },
        "evaluation": evaluation,
        "split_evaluation": selection_evaluation,
        "official_evaluation": official_evaluation,
        "baseline_evaluation": baseline_evaluation,
        "baseline_official_evaluation": baseline_official_evaluation,
        "common_cue_evaluation": common_metrics,
        "parameter_profile": profile_summary,
        "released_evaluation": released_evaluation,
        "released_split_evaluation": released_split_evaluation,
        "released_official_evaluation": released_official_evaluation,
        "released_trajectory_parity": released_parity,
        "released_baseline_trajectory_parity": released_baseline_parity,
        "selected_baseline_trajectory_parity": selected_baseline_parity,
        "outputs": {
            "trajectory": str(trajectory_path.resolve()),
            "baseline_trajectory": (
                None
                if baseline_trajectory_path is None
                else str(baseline_trajectory_path.resolve())
            ),
            "history": str(history_path.resolve()),
            "checkpoint": str(refit_checkpoint_path.resolve()),
            "parameter_profile": (
                None if profile_path is None else str(profile_path.resolve())
            ),
        },
    }
    summary_path = output_path / "summary.json"
    summary["outputs"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
