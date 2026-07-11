"""Inject Bayesian endpoint discrepancy into the released PhysTwin state."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from .phystwin_additional_bayesian_confirmation import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
)
from .phystwin_additional_confirmation import _chamfer_by_frame
from .phystwin_bayesian_anchor import robust_random_walk_endpoint
from .phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from .phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from .phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)
from .phystwin_refit import build_phystwin_track_objective
from .phystwin_residual_dynamics import (
    _clip_residual,
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
)

ENDPOINT_RESTART = "endpoint_restart"
OUTPUT_KNN = "output_knn"
OUTPUT_GRAPH = "output_graph"
INJECT_KNN_POSITION = "inject_knn_position"
INJECT_KNN_POSITION_VELOCITY = "inject_knn_position_velocity"
INJECT_GRAPH_POSITION = "inject_graph_position"
INJECT_GRAPH_POSITION_VELOCITY = "inject_graph_position_velocity"
PRIMARY_STATE_INJECTION_METHOD = INJECT_GRAPH_POSITION_VELOCITY
STATE_INJECTION_METHODS = (
    ENDPOINT_RESTART,
    OUTPUT_KNN,
    OUTPUT_GRAPH,
    INJECT_KNN_POSITION,
    INJECT_KNN_POSITION_VELOCITY,
    INJECT_GRAPH_POSITION,
    INJECT_GRAPH_POSITION_VELOCITY,
)
DIRECT_COMPARISONS = {
    "knn_position_vs_output": (OUTPUT_KNN, INJECT_KNN_POSITION),
    "knn_position_velocity_vs_output": (
        OUTPUT_KNN,
        INJECT_KNN_POSITION_VELOCITY,
    ),
    "graph_position_vs_output": (OUTPUT_GRAPH, INJECT_GRAPH_POSITION),
    "graph_position_velocity_vs_output": (
        OUTPUT_GRAPH,
        INJECT_GRAPH_POSITION_VELOCITY,
    ),
    "knn_velocity_update_effect": (
        INJECT_KNN_POSITION,
        INJECT_KNN_POSITION_VELOCITY,
    ),
    "graph_velocity_update_effect": (
        INJECT_GRAPH_POSITION,
        INJECT_GRAPH_POSITION_VELOCITY,
    ),
    "output_graph_vs_knn": (OUTPUT_KNN, OUTPUT_GRAPH),
    "injected_position_graph_vs_knn": (
        INJECT_KNN_POSITION,
        INJECT_GRAPH_POSITION,
    ),
    "injected_position_velocity_graph_vs_knn": (
        INJECT_KNN_POSITION_VELOCITY,
        INJECT_GRAPH_POSITION_VELOCITY,
    ),
}
_SIMULATOR_CLASS_CACHE: dict[tuple[str, str], object] = {}


def _released_self_collision_for_case(case_name: str) -> bool:
    """Return the self-collision setting selected by PhysTwin's CLI."""

    return "cloth" in case_name or "package" in case_name


@lru_cache(maxsize=1)
def _simulator_runtime() -> dict[str, object]:
    """Identify the numerical runtime used for the simulator comparison."""

    try:
        import torch
        import warp as wp
    except ImportError as error:
        raise RuntimeError(
            "state injection requires compatible torch and warp installations"
        ) from error
    warp_binary = Path(wp.__file__).resolve().parent / "bin" / "warp.so"
    return {
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "warp_binary_sha256": (_sha256(warp_binary) if warp_binary.is_file() else None),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def estimate_endpoint_velocity_delta(
    state_history: np.ndarray,
    *,
    frame_dt: float,
) -> np.ndarray:
    """Estimate endpoint velocity from state history by local regression."""

    values = np.asarray(state_history, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3 or len(values) < 2:
        raise ValueError("state_history must have shape (T>=2, N, 3)")
    if frame_dt <= 0.0 or not np.isfinite(frame_dt):
        raise ValueError("frame_dt must be positive and finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("state_history must be finite")
    times = frame_dt * np.arange(len(values), dtype=float)
    centered = times - np.mean(times)
    denominator = float(np.dot(centered, centered))
    return np.tensordot(centered, values, axes=(0, 0)) / denominator


def _git_commit(path: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_checkpoint(torch, path: str | Path, device: str) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    required = {
        "spring_Y",
        "num_object_springs",
        "collide_elas",
        "collide_fric",
        "collide_object_elas",
        "collide_object_fric",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError("checkpoint is missing: " + ", ".join(sorted(missing)))
    return checkpoint


def _trajectory_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_array = np.asarray(reference, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("trajectory parity arrays must have matching shapes")
    residual = candidate_array - reference_array
    if residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("trajectory parity arrays must have shape (T, N, 3)")
    norm = np.linalg.norm(residual, axis=2)
    return {
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(residual)))),
        "vector_rmse_m": float(np.sqrt(np.mean(np.square(norm)))),
        "mean_norm_m": float(np.mean(norm)),
        "maximum_norm_m": float(np.max(norm, initial=0.0)),
    }


def _metric_summary(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, object]:
    result = {}
    for metric, baseline_values in baseline.items():
        baseline_array = np.asarray(baseline_values, dtype=float)
        candidate_array = np.asarray(candidate[metric], dtype=float)
        baseline_mean = float(np.mean(baseline_array))
        candidate_mean = float(np.mean(candidate_array))
        result[metric] = {
            "baseline_by_frame_m": baseline_array.tolist(),
            "candidate_by_frame_m": candidate_array.tolist(),
            "baseline_mean_m": baseline_mean,
            "candidate_mean_m": candidate_mean,
            "percent_change": 100.0 * (candidate_mean / baseline_mean - 1.0),
        }
    return result


def _correction_history(
    residual: np.ndarray,
    valid: np.ndarray,
    baseline: np.ndarray,
    laplacian,
    *,
    train_end_frame: int,
    history_frames: int,
    original_count: int,
    interpolation_neighbors: int,
    maximum_residual_m: float,
    graph_prior_strength: float,
) -> tuple[dict[str, np.ndarray], object]:
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, interpolation_neighbors
    )
    histories = {"knn": [], "graph": []}
    final_endpoint = None
    for frame in range(train_end_frame - history_frames, train_end_frame):
        endpoint = robust_random_walk_endpoint(
            residual,
            valid,
            end_frame=frame + 1,
            process_variance=FIXED_PROCESS_STD_M**2,
            observation_variance=FIXED_OBSERVATION_STD_M**2,
            initial_variance=FIXED_INITIAL_STD_M**2,
            inlier_prior=FIXED_INLIER_PRIOR,
            outlier_variance_multiplier=FIXED_OUTLIER_VARIANCE_MULTIPLIER,
        )
        knn = _lift_residual(
            endpoint.mean[None],
            baseline.shape[1],
            lift_indices,
            lift_weights,
            maximum_norm=maximum_residual_m,
        )[0]
        graph_posterior = graph_smoothed_discrepancy_posterior(
            endpoint.mean,
            endpoint.variance,
            endpoint.update_count > 0,
            laplacian,
            prior_strength=graph_prior_strength,
        )
        graph = _clip_residual(graph_posterior.mean[None], maximum_residual_m)[0]
        histories["knn"].append(knn)
        histories["graph"].append(graph)
        final_endpoint = endpoint
    return {
        name: np.stack(values) for name, values in histories.items()
    }, final_endpoint


def _initialize_simulator(
    official_repo: str | Path,
    data: dict[str, object],
    optimal: dict[str, object],
    checkpoint_path: str | Path,
    graph,
    *,
    num_surface_points: int,
    original_count: int,
    dt: float,
    num_substeps: int,
    self_collision: bool,
    deterministic_spring_forces: bool = False,
    device: str,
):
    try:
        import torch
        import warp as wp
    except ImportError as error:
        raise RuntimeError(
            "state injection requires compatible torch and warp installations"
        ) from error
    from ._phystwin_warp_backend import (
        load_official_spring_mass_module,
        make_reliability_simulator_class,
    )

    object_points = np.asarray(data["object_points"], dtype=np.float32)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controller_points = np.asarray(data["controller_points"], dtype=np.float32)
    checkpoint = _load_checkpoint(torch, checkpoint_path, device)
    checkpoint_spring_y = torch.as_tensor(
        checkpoint["spring_Y"], dtype=torch.float32, device=device
    ).reshape(-1)
    if len(checkpoint_spring_y) != len(graph.springs):
        raise ValueError(
            "reconstructed graph and checkpoint disagree: "
            f"{len(graph.springs)} versus {len(checkpoint_spring_y)} springs"
        )
    if int(checkpoint["num_object_springs"]) != graph.num_object_springs:
        raise ValueError("checkpoint object spring count disagrees with graph")

    cache_key = (str(Path(official_repo).resolve()), device)
    simulator_class = _SIMULATOR_CLASS_CACHE.get(cache_key)
    if simulator_class is None:
        runtime_cfg = SimpleNamespace(
            device=device,
            use_graph=True,
            data_type="real",
            collision_learn=False,
            chamfer_weight=0.0,
            track_weight=0.0,
            acc_weight=0.0,
        )
        official = load_official_spring_mass_module(
            official_repo, runtime_config=runtime_cfg
        )
        simulator_class = make_reliability_simulator_class(official)
        _SIMULATOR_CLASS_CACHE[cache_key] = simulator_class
    objective = build_phystwin_track_objective(visible, motion_valid, variant="hard")

    def tensor(values: np.ndarray, dtype):
        return torch.as_tensor(values, dtype=dtype, device=device).contiguous()

    simulator = simulator_class(
        tensor(graph.vertices, torch.float32),
        tensor(graph.springs, torch.int32),
        tensor(graph.rest_lengths, torch.float32),
        tensor(graph.masses, torch.float32),
        dt=dt,
        num_substeps=num_substeps,
        spring_Y=float(optimal["global_spring_Y"]),
        collide_elas=float(optimal["collide_elas"]),
        collide_fric=float(optimal["collide_fric"]),
        dashpot_damping=float(optimal["dashpot_damping"]),
        drag_damping=float(optimal["drag_damping"]),
        collide_object_elas=float(optimal["collide_object_elas"]),
        collide_object_fric=float(optimal["collide_object_fric"]),
        collision_dist=float(optimal["collision_dist"]),
        num_object_points=len(graph.vertices) - len(controller_points[0]),
        num_surface_points=num_surface_points,
        num_original_points=original_count,
        controller_points=tensor(controller_points, torch.float32),
        reverse_z=True,
        spring_Y_min=0.0,
        spring_Y_max=1e5,
        gt_object_points=tensor(object_points, torch.float32),
        gt_object_visibilities=tensor(visible.astype(np.int32), torch.int32),
        gt_object_motions_valid=tensor(motion_valid.astype(np.int32), torch.int32),
        self_collision=self_collision,
        disable_backward=True,
        objective=objective,
        observation_variance=FIXED_OBSERVATION_STD_M**2,
        outlier_variance_multiplier=FIXED_OUTLIER_VARIANCE_MULTIPLIER,
        spring_parameterization="dense",
        num_object_springs=graph.num_object_springs,
        deterministic_spring_forces=deterministic_spring_forces,
    )
    simulator.set_reference_spring_y(torch.log(checkpoint_spring_y))

    def checkpoint_value(name: str):
        return torch.as_tensor(
            checkpoint[name], dtype=torch.float32, device=device
        ).reshape(-1)

    simulator.set_collide(
        checkpoint_value("collide_elas"), checkpoint_value("collide_fric")
    )
    simulator.set_collide_object(
        checkpoint_value("collide_object_elas"),
        checkpoint_value("collide_object_fric"),
    )
    wp.synchronize()
    return simulator, torch, wp, checkpoint


def _state_numpy(state, wp) -> tuple[np.ndarray, np.ndarray]:
    position = wp.to_torch(state.wp_x).detach().cpu().numpy().copy()
    velocity = wp.to_torch(state.wp_v).detach().cpu().numpy().copy()
    return position, velocity


def _rollout_initial(
    simulator, wp, *, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    simulator.set_init_state(
        simulator.wp_init_vertices,
        simulator.wp_init_velocities,
    )
    wp.synchronize()
    positions = []
    velocities = []
    position, velocity = _state_numpy(simulator.wp_states[0], wp)
    positions.append(position)
    velocities.append(velocity)
    for frame in range(1, frame_count):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        wp.capture_launch(simulator.forward_graph)
        wp.synchronize()
        position, velocity = _state_numpy(simulator.wp_states[-1], wp)
        positions.append(position)
        velocities.append(velocity)
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
        )
    return np.stack(positions), np.stack(velocities)


def _rollout_restart(
    simulator,
    torch,
    wp,
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    device: str,
) -> np.ndarray:
    position_tensor = torch.as_tensor(
        position, dtype=torch.float32, device=device
    ).contiguous()
    velocity_tensor = torch.as_tensor(
        velocity, dtype=torch.float32, device=device
    ).contiguous()
    position_wp = wp.from_torch(position_tensor, dtype=wp.vec3, requires_grad=False)
    velocity_wp = wp.from_torch(velocity_tensor, dtype=wp.vec3, requires_grad=False)
    simulator.set_init_state(position_wp, velocity_wp)
    wp.synchronize()
    future = []
    for frame in range(start_frame, stop_frame):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        wp.capture_launch(simulator.forward_graph)
        wp.synchronize()
        next_position, _ = _state_numpy(simulator.wp_states[-1], wp)
        future.append(next_position)
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
        )
    return np.stack(future)


def apply_phystwin_state_injection(
    official_repo: str | Path,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    gt_track_path: str | Path | None = None,
    graph_prior_strength: float = 0.1,
    velocity_history_frames: int = 3,
    interpolation_neighbors: int = 4,
    maximum_residual_m: float = 0.01,
    dt: float = 5e-5,
    num_substeps: int = 667,
    replay_endpoint_tolerance_m: float = 0.002,
    repeatability_replays: int = 3,
    self_collision: bool | None = None,
    deterministic_spring_forces: bool = True,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Compare output correction with position/velocity simulator state resets."""

    if velocity_history_frames < 2:
        raise ValueError("velocity_history_frames must be at least two")
    if graph_prior_strength <= 0.0:
        raise ValueError("graph_prior_strength must be positive")
    if maximum_residual_m <= 0.0 or replay_endpoint_tolerance_m <= 0.0:
        raise ValueError("residual cap and replay tolerance must be positive")
    if dt <= 0.0 or num_substeps < 1:
        raise ValueError("simulator time step settings must be positive")
    if repeatability_replays < 2:
        raise ValueError("repeatability_replays must be at least two")
    if device != "cuda:0":
        raise ValueError(
            "the pinned official simulator selects cuda:0; use "
            "CUDA_VISIBLE_DEVICES to remap a GPU"
        )
    data = _load_pickle(final_data_path)
    if self_collision is None:
        case_name = Path(final_data_path).resolve().parent.name
        self_collision = _released_self_collision_for_case(case_name)
    optimal = _load_pickle(optimal_params_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
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
    object_springs = graph.springs[: graph.num_object_springs]
    laplacian = normalized_spring_laplacian(len(structure_points), object_springs)
    residual = observed - baseline[:, :original_count]
    valid = _target_validity(visible, motion_valid)
    histories, endpoint = _correction_history(
        residual,
        valid,
        baseline,
        laplacian,
        train_end_frame=train_end_frame,
        history_frames=velocity_history_frames,
        original_count=original_count,
        interpolation_neighbors=interpolation_neighbors,
        maximum_residual_m=maximum_residual_m,
        graph_prior_strength=graph_prior_strength,
    )
    frame_dt = dt * num_substeps
    velocity_delta = {
        mode: estimate_endpoint_velocity_delta(values, frame_dt=frame_dt)
        for mode, values in histories.items()
    }
    endpoint_index = train_end_frame - 1
    estimated_endpoint_velocity = estimate_endpoint_velocity_delta(
        baseline[train_end_frame - velocity_history_frames : train_end_frame],
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
    replay_positions, replay_velocities = _rollout_initial(
        simulator, wp, frame_count=frame_count
    )
    parity = {
        "all_frames": _trajectory_error(baseline, replay_positions),
        "training_frames": _trajectory_error(
            baseline[:train_end_frame], replay_positions[:train_end_frame]
        ),
        "endpoint": _trajectory_error(
            baseline[endpoint_index : endpoint_index + 1],
            replay_positions[endpoint_index : endpoint_index + 1],
        ),
        "future_frames": _trajectory_error(
            baseline[train_end_frame:], replay_positions[train_end_frame:]
        ),
    }
    parity["passed"] = bool(
        parity["endpoint"]["vector_rmse_m"] <= replay_endpoint_tolerance_m
    )
    parity["role"] = "diagnostic only; no candidate state uses replayed state"

    endpoint_restart = baseline.copy()
    endpoint_restart[train_end_frame:] = _rollout_restart(
        simulator,
        torch,
        wp,
        baseline[endpoint_index],
        estimated_endpoint_velocity,
        start_frame=train_end_frame,
        stop_frame=frame_count,
        device=device,
    )
    endpoint_restart_repeat_futures = [
        _rollout_restart(
            simulator,
            torch,
            wp,
            baseline[endpoint_index],
            estimated_endpoint_velocity,
            start_frame=train_end_frame,
            stop_frame=frame_count,
            device=device,
        )
        for _ in range(repeatability_replays - 1)
    ]
    repeatability_errors = [
        _trajectory_error(endpoint_restart[train_end_frame:], repeated)
        for repeated in endpoint_restart_repeat_futures
    ]
    restart_repeatability = {
        key: max(error[key] for error in repeatability_errors)
        for key in repeatability_errors[0]
    }
    restart_repeatability.update(
        {
            "ensemble_size": repeatability_replays,
            "bitwise_identical": bool(
                all(
                    np.array_equal(endpoint_restart[train_end_frame:], repeated)
                    for repeated in endpoint_restart_repeat_futures
                )
            ),
            "per_repeat": repeatability_errors,
        }
    )
    candidates = {ENDPOINT_RESTART: endpoint_restart}
    for mode, output_method, position_method, velocity_method in (
        (
            "knn",
            OUTPUT_KNN,
            INJECT_KNN_POSITION,
            INJECT_KNN_POSITION_VELOCITY,
        ),
        (
            "graph",
            OUTPUT_GRAPH,
            INJECT_GRAPH_POSITION,
            INJECT_GRAPH_POSITION_VELOCITY,
        ),
    ):
        correction = histories[mode][-1]
        output_candidate = baseline.copy()
        output_candidate[train_end_frame:] += correction[None]
        candidates[output_method] = output_candidate
        restart_position = baseline[endpoint_index] + correction
        for method, restart_velocity in (
            (position_method, estimated_endpoint_velocity),
            (
                velocity_method,
                estimated_endpoint_velocity + velocity_delta[mode],
            ),
        ):
            future = _rollout_restart(
                simulator,
                torch,
                wp,
                restart_position,
                restart_velocity,
                start_frame=train_end_frame,
                stop_frame=frame_count,
                device=device,
            )
            candidate = baseline.copy()
            candidate[endpoint_index] = restart_position
            candidate[train_end_frame:] = future
            candidates[method] = candidate

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

    baseline_metrics = metrics_by_frame(baseline)
    method_results = {
        method: {"future": _metric_summary(baseline_metrics, metrics_by_frame(value))}
        for method, value in candidates.items()
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "state_injection.npz"
    archive_values = {
        "correction_history__knn": histories["knn"],
        "correction_history__graph": histories["graph"],
        "velocity_delta__knn": velocity_delta["knn"],
        "velocity_delta__graph": velocity_delta["graph"],
        "estimated_endpoint_velocity": estimated_endpoint_velocity,
        "diagnostic_replay_endpoint_velocity": replay_velocities[endpoint_index],
        "diagnostic_replay_future": replay_positions[train_end_frame:],
        **{
            f"diagnostic_endpoint_restart_repeat_future__{index + 1}": repeated
            for index, repeated in enumerate(endpoint_restart_repeat_futures)
        },
    }
    for method, trajectory in candidates.items():
        archive_values[f"future__{method}"] = trajectory[train_end_frame:]
    np.savez_compressed(archive_path, **archive_values)
    updated = endpoint.update_count > 0
    summary: dict[str, object] = {
        "schema_version": 2,
        "config": {
            "train_end_frame": train_end_frame,
            "graph_prior_strength": graph_prior_strength,
            "velocity_history_frames": velocity_history_frames,
            "interpolation_neighbors": interpolation_neighbors,
            "maximum_residual_m": maximum_residual_m,
            "dt": dt,
            "num_substeps": num_substeps,
            "replay_endpoint_tolerance_m": replay_endpoint_tolerance_m,
            "repeatability_replays": repeatability_replays,
            "self_collision": self_collision,
            "deterministic_spring_forces": deterministic_spring_forces,
            "device": device,
            "runtime": _simulator_runtime(),
        },
        "contract": {
            "endpoint_position": (
                "released endpoint plus fixed robust Bayesian correction"
            ),
            "endpoint_restart": (
                "released endpoint with velocity estimated from the released "
                f"trajectory over {velocity_history_frames} frames"
            ),
            "position_only_velocity": (
                "same released-trajectory endpoint velocity estimate as the "
                "uncorrected endpoint restart"
            ),
            "position_velocity": (
                "released-trajectory endpoint velocity estimate plus "
                f"{velocity_history_frames}-frame local-linear correction velocity"
            ),
            "future_controls": "recorded released controller trajectory",
            "future_observations": "none",
            "rest_state": "original released spring rest lengths retained",
            "frame_zero_replay": (
                "diagnostic only; never supplies candidate position or velocity"
            ),
            "self_collision": (
                "released PhysTwin cloth/package case configuration"
                if self_collision
                else "released PhysTwin default configuration"
            ),
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
        "replay_parity": parity,
        "endpoint_restart_repeatability": restart_repeatability,
        "endpoint_posterior": {
            "updated_track_count": int(np.sum(updated)),
            "median_std_m": float(np.median(np.sqrt(endpoint.variance[updated]))),
            "median_final_inlier_probability": float(
                np.median(endpoint.final_inlier_probability[updated])
            ),
        },
        "endpoint_velocity_estimate": {
            "history_frames": velocity_history_frames,
            "rms_m_per_s": float(
                np.sqrt(np.mean(np.sum(np.square(estimated_endpoint_velocity), axis=1)))
            ),
            "maximum_m_per_s": float(
                np.max(np.linalg.norm(estimated_endpoint_velocity, axis=1), initial=0.0)
            ),
        },
        "state_update": {
            mode: {
                "position_correction_rms_m": float(
                    np.sqrt(np.mean(np.sum(np.square(history[-1]), axis=1)))
                ),
                "position_correction_maximum_m": float(
                    np.max(np.linalg.norm(history[-1], axis=1), initial=0.0)
                ),
                "velocity_delta_rms_m_per_s": float(
                    np.sqrt(np.mean(np.sum(np.square(velocity_delta[mode]), axis=1)))
                ),
                "velocity_delta_maximum_m_per_s": float(
                    np.max(np.linalg.norm(velocity_delta[mode], axis=1), initial=0.0)
                ),
            }
            for mode, history in histories.items()
        },
        "methods": method_results,
        "outputs": {"archive": str(archive_path.resolve())},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["summary"] = str(summary_path.resolve())
    del simulator
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _compact_bootstrap(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key in {"samples", "block_length", "seed", "macro", "cluster_macro"}
    }


def _candidate_metrics(
    summary: dict[str, object], method: str
) -> dict[str, np.ndarray]:
    return {
        metric: np.asarray(values["candidate_by_frame_m"], dtype=float)
        for metric, values in summary["methods"][method]["future"].items()
    }


def run_phystwin_state_injection_comparison(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str = "all",
    cases: Iterable[str] | None = None,
    graph_prior_strength: float = 0.1,
    velocity_history_frames: int = 3,
    interpolation_neighbors: int = 4,
    maximum_residual_m: float = 0.01,
    dt: float = 5e-5,
    num_substeps: int = 667,
    replay_endpoint_tolerance_m: float = 0.002,
    repeatability_replays: int = 3,
    deterministic_spring_forces: bool = True,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260711,
    force: bool = False,
) -> dict[str, object]:
    """Run the frozen output-correction versus state-injection comparison."""

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
    clusters = {case: phystwin_physical_object_cluster(case) for case in selected}
    self_collision_by_case = {
        case: _released_self_collision_for_case(case) for case in selected
    }
    runtime = _simulator_runtime()
    code_commit = _git_commit(Path(__file__).resolve().parents[2])
    specification = {
        "method": "PhysTwin endpoint state injection",
        "code_commit": code_commit,
        "dataset": "additional" if is_additional else "main",
        "cohort": cohort,
        "cases": list(selected),
        "fixed_filter": {
            "process_std_m": FIXED_PROCESS_STD_M,
            "observation_std_m": FIXED_OBSERVATION_STD_M,
            "initial_std_m": FIXED_INITIAL_STD_M,
            "inlier_prior": FIXED_INLIER_PRIOR,
            "outlier_variance_multiplier": FIXED_OUTLIER_VARIANCE_MULTIPLIER,
        },
        "graph_prior_strength": graph_prior_strength,
        "velocity_history_frames": velocity_history_frames,
        "velocity_estimator": (
            "local linear slope over the released state and correction posterior"
        ),
        "interpolation_neighbors": interpolation_neighbors,
        "maximum_residual_m": maximum_residual_m,
        "dt": dt,
        "num_substeps": num_substeps,
        "self_collision_rule": (
            "enabled when the case name contains 'cloth' or 'package', "
            "matching PhysTwin train_warp.py and inference_warp.py"
        ),
        "self_collision_by_case": self_collision_by_case,
        "diagnostic_replay_endpoint_tolerance_m": replay_endpoint_tolerance_m,
        "deterministic_spring_forces": deterministic_spring_forces,
        "endpoint_state_source": (
            "released endpoint position and released-trajectory velocity estimate; "
            "frame-zero replay is diagnostic only"
        ),
        "restart_repeatability_control": (
            f"{repeatability_replays} uncorrected endpoint restarts with identical "
            "state and controls"
        ),
        "future_inputs": "recorded controller positions only",
        "primary_method": PRIMARY_STATE_INJECTION_METHOD,
        "primary_selection": (
            "preselected requested full graph-smoothed position-and-velocity "
            "state update; position-only and raw-kNN variants are ablations"
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
        "status": "post-hoc frozen matched-endpoint state-injection evaluation",
    }
    output = Path(output_dir)
    locked = _lock_protocol(output, specification)
    expected_base_config = {
        "graph_prior_strength": graph_prior_strength,
        "velocity_history_frames": velocity_history_frames,
        "interpolation_neighbors": interpolation_neighbors,
        "maximum_residual_m": maximum_residual_m,
        "dt": dt,
        "num_substeps": num_substeps,
        "replay_endpoint_tolerance_m": replay_endpoint_tolerance_m,
        "repeatability_replays": repeatability_replays,
        "deterministic_spring_forces": deterministic_spring_forces,
        "device": "cuda:0",
        "runtime": runtime,
    }
    case_results = {}
    paired_vs_released = {method: {} for method in STATE_INJECTION_METHODS}
    paired_vs_restart = {
        method: {} for method in STATE_INJECTION_METHODS if method != ENDPOINT_RESTART
    }
    paired_direct = {name: {} for name in DIRECT_COMPARISONS}
    for case in selected:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_end, future_end = (int(value) for value in split["test"])
        if future_end != int(split["frame_len"]):
            raise ValueError(f"future split does not end at frame_len for {case}")
        case_output = output / "cases" / case
        summary_path = case_output / "summary.json"
        expected_config = {
            **expected_base_config,
            "self_collision": self_collision_by_case[case],
        }
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            cached = dict(summary["config"])
            cached.pop("train_end_frame")
            if cached != expected_config:
                raise RuntimeError(f"cached case uses a different protocol: {case}")
        else:
            track_path = case_dir / "gt_track_3d.pkl"
            summary = apply_phystwin_state_injection(
                official_repo,
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_dir / "optimal_params.pkl",
                case_dir / "checkpoint.pth",
                case_output,
                train_end_frame=train_end,
                gt_track_path=track_path if track_path.exists() else None,
                graph_prior_strength=graph_prior_strength,
                velocity_history_frames=velocity_history_frames,
                interpolation_neighbors=interpolation_neighbors,
                maximum_residual_m=maximum_residual_m,
                dt=dt,
                num_substeps=num_substeps,
                replay_endpoint_tolerance_m=replay_endpoint_tolerance_m,
                repeatability_replays=repeatability_replays,
                self_collision=self_collision_by_case[case],
                deterministic_spring_forces=deterministic_spring_forces,
            )
        case_results[case] = {
            "physical_object": clusters[case],
            "replay_parity": summary["replay_parity"],
            "endpoint_restart_repeatability": summary["endpoint_restart_repeatability"],
            "endpoint_velocity_estimate": summary["endpoint_velocity_estimate"],
            "state_update": summary["state_update"],
            "methods": {
                method: summary["methods"][method] for method in STATE_INJECTION_METHODS
            },
        }
        released_metrics = {
            metric: np.asarray(values["baseline_by_frame_m"], dtype=float)
            for metric, values in summary["methods"][ENDPOINT_RESTART]["future"].items()
        }
        restart_metrics = _candidate_metrics(summary, ENDPOINT_RESTART)
        method_metrics = {
            method: _candidate_metrics(summary, method)
            for method in STATE_INJECTION_METHODS
        }
        for method in STATE_INJECTION_METHODS:
            paired_vs_released[method][case] = (
                released_metrics,
                method_metrics[method],
            )
            if method != ENDPOINT_RESTART:
                paired_vs_restart[method][case] = (
                    restart_metrics,
                    method_metrics[method],
                )
        for comparison, (
            baseline_method,
            candidate_method,
        ) in DIRECT_COMPARISONS.items():
            paired_direct[comparison][case] = (
                method_metrics[baseline_method],
                method_metrics[candidate_method],
            )

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

    comparisons_vs_released = {
        method: bootstrap(paired) for method, paired in paired_vs_released.items()
    }
    comparisons_vs_restart = {
        method: bootstrap(paired) for method, paired in paired_vs_restart.items()
    }
    direct_comparisons = {
        name: bootstrap(paired) for name, paired in paired_direct.items()
    }
    result = {
        "schema_version": 2,
        "code_commit": code_commit,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": specification["status"],
        "dataset": specification["dataset"],
        "cohort": cohort,
        "case_count": len(selected),
        "physical_object_count": len(set(clusters.values())),
        "methods": list(STATE_INJECTION_METHODS),
        "primary_method": PRIMARY_STATE_INJECTION_METHOD,
        "replay_diagnostic_pass_count": sum(
            bool(value["replay_parity"]["passed"]) for value in case_results.values()
        ),
        "case_results": case_results,
        "comparisons_vs_released": comparisons_vs_released,
        "comparisons_vs_endpoint_restart": comparisons_vs_restart,
        "direct_comparisons": direct_comparisons,
    }
    result_path = output / "state_injection_comparison_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
