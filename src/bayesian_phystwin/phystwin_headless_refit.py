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
)
from .phystwin_refit import (
    PhysTwinRefitReliabilityConfig,
    build_phystwin_track_objective,
    evaluate_phystwin_trajectory,
    phystwin_tracking_metrics,
)


@dataclass(frozen=True)
class HeadlessPhysTwinRefitConfig:
    """Simulator and optimizer settings for one matched refit variant."""

    variant: str
    train_end_frame: int
    epochs: int = 0
    learning_rate: float = 1e-4
    observation_variance: float = 2.5e-5
    model_discrepancy_variance: float = 0.0
    outlier_variance_multiplier: float = 100.0
    flow_scale: float = 0.005
    dt: float = 5e-5
    num_substeps: int = 667
    track_weight: float = 1.0
    acceleration_weight: float = 0.01
    optimize_collision: bool = True
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
    if config.num_substeps < 1 or config.dt <= 0.0:
        raise ValueError("simulator time discretization must be positive")
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
    frame_count, original_count, coordinate_count = object_points.shape
    if coordinate_count != 3:
        raise ValueError("object_points must have shape (T, N, 3)")
    if not 1 < config.train_end_frame < frame_count:
        raise ValueError("train_end_frame must be between 2 and T-1")

    with np.load(cues_path) as archive:
        cues = {name: np.asarray(archive[name]) for name in archive.files}
    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues=cues,
        variant=config.variant,
        config=PhysTwinRefitReliabilityConfig(flow_scale=config.flow_scale),
    )

    structure_points = np.concatenate(
        (object_points[0], surface_points, interior_points),
        axis=0,
    )
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
        dashpot_damping=float(optimal["dashpot_damping"]),
        drag_damping=float(optimal["drag_damping"]),
        collide_object_elas=float(optimal["collide_object_elas"]),
        collide_object_fric=float(optimal["collide_object_fric"]),
        collision_dist=float(optimal["collision_dist"]),
        num_object_points=len(structure_points),
        num_surface_points=original_count + len(surface_points),
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
    )
    simulator.set_spring_Y(torch.log(checkpoint_spring_y).detach().clone())

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

    history: list[dict[str, float | int]] = []
    if config.epochs:
        optimizer_parameters = [wp.to_torch(simulator.wp_spring_Y)]
        if config.optimize_collision:
            optimizer_parameters.extend(
                [
                    wp.to_torch(simulator.wp_collide_elas),
                    wp.to_torch(simulator.wp_collide_fric),
                    wp.to_torch(simulator.wp_collide_object_elas),
                    wp.to_torch(simulator.wp_collide_object_fric),
                ]
            )
        optimizer = torch.optim.Adam(
            optimizer_parameters,
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
            for frame in range(1, config.train_end_frame):
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
            denominator = config.train_end_frame - 1
            history.append(
                {
                    "epoch": epoch,
                    "mean_loss": total_loss / denominator,
                    "mean_track_loss": total_track_loss / denominator,
                }
            )

    simulator.set_init_state(
        simulator.wp_init_vertices,
        simulator.wp_init_velocities,
        pure_inference=True,
    )
    trajectory_frames = [
        wp.to_torch(simulator.wp_states[0].wp_x).detach().cpu().numpy().copy()
    ]
    for frame in range(1, frame_count):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        wp.capture_launch(simulator.forward_graph)
        wp.synchronize()
        trajectory_frames.append(
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
    trajectory = np.stack(trajectory_frames).astype(np.float32)

    final_spring_y = (
        torch.exp(wp.to_torch(simulator.wp_spring_Y)).detach().cpu().numpy().copy()
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
    common_objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues=cues,
        variant="cue",
        config=PhysTwinRefitReliabilityConfig(flow_scale=config.flow_scale),
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

    released_evaluation = None
    released_parity = None
    if released_trajectory_path is not None:
        released = np.asarray(_load_pickle(released_trajectory_path), dtype=np.float32)
        released_evaluation = evaluate_phystwin_trajectory(
            object_points,
            released,
            visible,
            motion_valid,
            train_end_frame=config.train_end_frame,
        )
        parity_frames = min(len(released), len(trajectory))
        parity_vertices = min(released.shape[1], trajectory.shape[1])
        released_parity = phystwin_tracking_metrics(
            released[:parity_frames, :parity_vertices],
            trajectory[:parity_frames, :parity_vertices],
            np.ones((parity_frames, parity_vertices), dtype=bool),
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_path / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
            "final_collision": final_collision,
            "fixed_dashpot_damping": float(optimal["dashpot_damping"]),
            "fixed_drag_damping": float(optimal["drag_damping"]),
        },
        "history": history,
        "evaluation": evaluation,
        "common_cue_evaluation": common_metrics,
        "released_evaluation": released_evaluation,
        "released_trajectory_parity": released_parity,
        "outputs": {
            "trajectory": str(trajectory_path.resolve()),
            "history": str(history_path.resolve()),
            "checkpoint": str(refit_checkpoint_path.resolve()),
        },
    }
    summary_path = output_path / "summary.json"
    summary["outputs"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
