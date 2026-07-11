"""Matched PhysTwin sensitivity to smooth controller-trajectory error."""

from __future__ import annotations

import gc
import hashlib
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


DEFAULT_JITTER_SCALES_M = (0.001, 0.002, 0.005, 0.01)


def controller_jitter_id(scale_m: float) -> str:
    """Return a stable result identifier for a controller-jitter scale."""

    value = format(1000.0 * float(scale_m), ".12g")
    return "jitter_" + value.replace("-", "m").replace(".", "p") + "mm"


def controller_hand_count(case_name: str) -> int:
    """Infer the released one- or two-hand interaction contract."""

    return 2 if case_name.startswith("double_") or case_name == "rope_double_hand" else 1


def infer_controller_groups(
    initial_controller_points: np.ndarray,
    *,
    group_count: int,
) -> np.ndarray:
    """Partition controller points into deterministic spatial hand groups."""

    points = np.asarray(initial_controller_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < group_count:
        raise ValueError("initial_controller_points must have shape (C>=G, 3)")
    if group_count == 1:
        return np.zeros(len(points), dtype=np.int32)
    if group_count != 2:
        raise ValueError("released controller grouping supports one or two hands")
    squared = np.sum(np.square(points[:, None] - points[None]), axis=2)
    first, second = np.unravel_index(int(np.argmax(squared)), squared.shape)
    centroids = np.stack((points[first], points[second]))
    labels = np.zeros(len(points), dtype=np.int32)
    for _ in range(32):
        distances = np.sum(np.square(points[:, None] - centroids[None]), axis=2)
        updated = np.argmin(distances, axis=1).astype(np.int32)
        if np.all(updated == updated[0]):
            axis = centroids[1] - centroids[0]
            coordinate = points @ axis
            order = np.argsort(coordinate, kind="stable")
            updated[order[len(points) // 2 :]] = 1
        new_centroids = np.stack(
            [np.mean(points[updated == group], axis=0) for group in range(2)]
        )
        if np.array_equal(updated, labels) and np.allclose(new_centroids, centroids):
            labels = updated
            break
        labels = updated
        centroids = new_centroids
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("controller hand partition produced an empty group")
    first_centroid = np.mean(points[labels == 0], axis=0)
    second_centroid = np.mean(points[labels == 1], axis=0)
    difference = second_centroid - first_centroid
    dominant = int(np.argmax(np.abs(difference)))
    if difference[dominant] < 0.0:
        labels = 1 - labels
    return labels.astype(np.int32)


def smooth_group_controller_jitter(
    controller_points: np.ndarray,
    groups: np.ndarray,
    *,
    start_frame: int,
    target_rms_m: float,
    correlation_frames: float,
    seed: int,
) -> np.ndarray:
    """Generate endpoint-zero AR(1) translation error per inferred hand."""

    controls = np.asarray(controller_points, dtype=float)
    labels = np.asarray(groups, dtype=np.int32)
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError("controller_points must have shape (T, C, 3)")
    if labels.shape != (controls.shape[1],) or np.any(labels < 0):
        raise ValueError("groups must label every controller point")
    if not 1 <= start_frame < len(controls):
        raise ValueError("start_frame must leave a future interval")
    if target_rms_m <= 0.0 or correlation_frames <= 0.0:
        raise ValueError("jitter scale and correlation length must be positive")
    group_count = int(np.max(labels)) + 1
    if set(labels.tolist()) != set(range(group_count)):
        raise ValueError("controller group labels must be contiguous")
    rho = float(np.exp(-1.0 / correlation_frames))
    innovation_scale = float(np.sqrt(1.0 - rho * rho))
    rng = np.random.default_rng(seed)
    group_error = np.zeros((len(controls), group_count, 3), dtype=float)
    for frame in range(start_frame, len(controls)):
        group_error[frame] = (
            rho * group_error[frame - 1]
            + innovation_scale * rng.normal(size=(group_count, 3))
        )
    jitter = group_error[:, labels]
    current_rms = float(
        np.sqrt(np.mean(np.sum(np.square(jitter[start_frame:]), axis=2)))
    )
    if current_rms <= 1e-15:
        raise RuntimeError("controller jitter realization collapsed to zero")
    jitter *= target_rms_m / current_rms
    return jitter


def controller_jitter_diagnostics(
    controller_points: np.ndarray,
    jitter: np.ndarray,
    *,
    start_frame: int,
) -> dict[str, float]:
    """Summarize empirical controller roughness and applied perturbation."""

    controls = np.asarray(controller_points, dtype=float)
    values = np.asarray(jitter, dtype=float)
    if controls.shape != values.shape or controls.ndim != 3:
        raise ValueError("controller_points and jitter must have matching shape")
    future = values[start_frame:]
    norm = np.linalg.norm(future, axis=2)
    centered = controls - np.mean(controls, axis=1, keepdims=True)
    shape_step = np.linalg.norm(np.diff(centered, axis=0), axis=2)
    acceleration = np.linalg.norm(np.diff(controls, n=2, axis=0), axis=2)
    jitter_step = np.linalg.norm(np.diff(values[start_frame - 1 :], axis=0), axis=2)
    jitter_acceleration = np.linalg.norm(
        np.diff(values[start_frame - 1 :], n=2, axis=0), axis=2
    )
    return {
        "jitter_vector_rms_m": float(np.sqrt(np.mean(np.square(norm)))),
        "jitter_median_norm_m": float(np.median(norm)),
        "jitter_maximum_norm_m": float(np.max(norm, initial=0.0)),
        "jitter_step_median_m": float(np.median(jitter_step)),
        "jitter_acceleration_median_m": float(np.median(jitter_acceleration)),
        "released_shape_step_median_m": float(np.median(shape_step)),
        "released_shape_step_p95_m": float(np.quantile(shape_step, 0.95)),
        "released_acceleration_median_m": float(np.median(acceleration)),
        "released_acceleration_p95_m": float(np.quantile(acceleration, 0.95)),
    }


def _metric_summary(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric, baseline_raw in baseline.items():
        baseline_values = np.asarray(baseline_raw, dtype=float)
        candidate_values = np.asarray(candidate[metric], dtype=float)
        baseline_mean = float(np.mean(baseline_values))
        candidate_mean = float(np.mean(candidate_values))
        result[metric] = {
            "baseline_by_frame_m": baseline_values.tolist(),
            "candidate_by_frame_m": candidate_values.tolist(),
            "baseline_mean_m": baseline_mean,
            "candidate_mean_m": candidate_mean,
            "delta_mean_m": candidate_mean - baseline_mean,
            "percent_change": 100.0 * (candidate_mean / baseline_mean - 1.0),
        }
    return result


def _case_seed(base_seed: int, case_name: str, pair_index: int) -> int:
    digest = hashlib.sha256(case_name.encode("utf-8")).digest()
    case_offset = int.from_bytes(digest[:4], "little")
    return int((base_seed + case_offset + pair_index) % (2**32))


def apply_controller_jitter_sensitivity(
    official_repo: str | Path,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    gt_track_path: str | Path | None = None,
    jitter_scales_m: Iterable[float] = DEFAULT_JITTER_SCALES_M,
    antithetic_pair_count: int = 4,
    correlation_frames: float = 5.0,
    velocity_history_frames: int = 3,
    dt: float = 5e-5,
    num_substeps: int = 667,
    seed: int = 20260711,
    self_collision: bool | None = None,
    deterministic_spring_forces: bool = True,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Measure future sensitivity to matched smooth controller perturbations."""

    scales = tuple(sorted(set(float(value) for value in jitter_scales_m)))
    if not scales or any(value <= 0.0 for value in scales):
        raise ValueError("jitter_scales_m must contain positive values")
    if antithetic_pair_count < 1 or velocity_history_frames < 2:
        raise ValueError("pair count and velocity history are too small")
    if correlation_frames <= 0.0 or dt <= 0.0 or num_substeps < 1:
        raise ValueError("simulator and jitter settings must be positive")
    if device != "cuda:0":
        raise ValueError("use CUDA_VISIBLE_DEVICES to remap the pinned cuda:0")
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
    if not velocity_history_frames <= train_end_frame < frame_count:
        raise ValueError("training endpoint does not support the velocity history")
    if baseline.shape[0] < frame_count:
        raise ValueError("baseline trajectory has too few frames")
    baseline = baseline[:frame_count]
    structure_points = np.concatenate((observed[0], surface, interior), axis=0)
    if baseline.shape[1] != len(structure_points):
        raise ValueError("released trajectory and object state size disagree")
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
    endpoint_index = train_end_frame - 1
    estimated_velocity = estimate_endpoint_velocity_delta(
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

    def set_controls(values: np.ndarray) -> None:
        simulator.controller_points = torch.as_tensor(
            values, dtype=torch.float32, device=device
        ).contiguous()

    def rollout(values: np.ndarray) -> np.ndarray:
        set_controls(values)
        return _rollout_restart(
            simulator,
            torch,
            wp,
            baseline[endpoint_index],
            estimated_velocity,
            start_frame=train_end_frame,
            stop_frame=frame_count,
            device=device,
        )

    original_future = rollout(controller)
    duplicate_future = rollout(controller)
    repeatability = _trajectory_error(original_future, duplicate_future)
    matched = baseline.copy()
    matched[train_end_frame:] = original_future
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
    matched_metrics = metrics_by_frame(matched)
    duplicate = baseline.copy()
    duplicate[train_end_frame:] = duplicate_future
    duplicate_metrics = metrics_by_frame(duplicate)
    unit_jitters = []
    for pair_index in range(antithetic_pair_count):
        unit_jitters.append(
            smooth_group_controller_jitter(
                controller,
                groups,
                start_frame=train_end_frame,
                target_rms_m=1.0,
                correlation_frames=correlation_frames,
                seed=_case_seed(seed, case_name, pair_index),
            )
        )
    scale_results: dict[str, object] = {}
    for scale in scales:
        draws = []
        for pair_index, unit_jitter in enumerate(unit_jitters):
            for sign, sign_label in ((1.0, "plus"), (-1.0, "minus")):
                jitter = sign * scale * unit_jitter
                candidate_future = rollout(controller + jitter)
                candidate = baseline.copy()
                candidate[train_end_frame:] = candidate_future
                candidate_metrics = metrics_by_frame(candidate)
                displacement = _trajectory_error(original_future, candidate_future)
                draws.append(
                    {
                        "draw_id": f"pair_{pair_index}_{sign_label}",
                        "pair_index": pair_index,
                        "sign": int(sign),
                        "controller": controller_jitter_diagnostics(
                            controller,
                            jitter,
                            start_frame=train_end_frame,
                        ),
                        "future_object_displacement": {
                            **displacement,
                            "vector_rms_gain_per_controller_rms": (
                                displacement["vector_rmse_m"] / scale
                            ),
                        },
                        "future": _metric_summary(
                            matched_metrics, candidate_metrics
                        ),
                    }
                )
        scale_results[controller_jitter_id(scale)] = {
            "target_controller_vector_rms_m": scale,
            "draw_count": len(draws),
            "draws": draws,
        }
    set_controls(controller)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "controller_jitter.npz"
    np.savez_compressed(
        archive_path,
        controller_groups=groups,
        estimated_endpoint_velocity=estimated_velocity,
        matched_original_future=original_future,
        duplicate_original_future=duplicate_future,
        **{
            f"unit_jitter__pair_{index}": values.astype(np.float32)
            for index, values in enumerate(unit_jitters)
        },
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": {
            "train_end_frame": train_end_frame,
            "jitter_scales_m": list(scales),
            "antithetic_pair_count": antithetic_pair_count,
            "correlation_frames": correlation_frames,
            "velocity_history_frames": velocity_history_frames,
            "dt": dt,
            "num_substeps": num_substeps,
            "seed": seed,
            "self_collision": self_collision,
            "deterministic_spring_forces": deterministic_spring_forces,
            "device": device,
            "runtime": _simulator_runtime(),
        },
        "contract": {
            "endpoint_state": (
                "identical released endpoint position and local-linear released "
                "trajectory velocity for every rollout"
            ),
            "training_controls": "unmodified through the released training endpoint",
            "future_controls": (
                "recorded controls plus endpoint-zero AR(1) translation per "
                "spatially inferred hand group"
            ),
            "antithetic_design": "each random field is evaluated with both signs",
            "spring_force_accumulation": (
                "fixed per-vertex incident-spring order without GPU atomics"
                if deterministic_spring_forces
                else "released per-spring GPU atomic accumulation"
            ),
            "future_observations": "evaluation only; never used to create jitter",
            "scale_interpretation": (
                "1/2/5 mm data-scale sensitivity; 10 mm stress test, not a "
                "calibrated hand-tracker likelihood"
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
            "released_roughness": controller_jitter_diagnostics(
                controller,
                unit_jitters[0] * scales[0],
                start_frame=train_end_frame,
            ),
        },
        "matched_baseline": {
            "future": _metric_summary(released_metrics, matched_metrics),
            "duplicate_restart_repeatability": repeatability,
            "duplicate_restart_metric_null": _metric_summary(
                matched_metrics, duplicate_metrics
            ),
        },
        "scales": scale_results,
        "outputs": {"archive": str(archive_path.resolve())},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be finite and nonempty")
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "lower_quartile": float(np.quantile(array, 0.25)),
        "upper_quartile": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _future_arrays(future: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    baseline = {
        metric: np.asarray(values["baseline_by_frame_m"], dtype=float)
        for metric, values in future.items()
    }
    candidate = {
        metric: np.asarray(values["candidate_by_frame_m"], dtype=float)
        for metric, values in future.items()
    }
    return baseline, candidate


def _scale_case_sensitivity(scale_result: dict[str, object]) -> dict[str, object]:
    scale = float(scale_result["target_controller_vector_rms_m"])
    draws = scale_result["draws"]
    metrics = tuple(draws[0]["future"])
    result: dict[str, object] = {
        "future_object_displacement_gain": _distribution(
            draw["future_object_displacement"][
                "vector_rms_gain_per_controller_rms"
            ]
            for draw in draws
        )
    }
    metric_results = {}
    for metric in metrics:
        changes = np.asarray(
            [draw["future"][metric]["percent_change"] for draw in draws],
            dtype=float,
        )
        absolute_gain = np.asarray(
            [abs(draw["future"][metric]["delta_mean_m"]) / scale for draw in draws],
            dtype=float,
        )
        central_gain = []
        for pair_index in sorted({int(draw["pair_index"]) for draw in draws}):
            pair = [draw for draw in draws if int(draw["pair_index"]) == pair_index]
            plus = next(draw for draw in pair if int(draw["sign"]) == 1)
            minus = next(draw for draw in pair if int(draw["sign"]) == -1)
            central_gain.append(
                abs(
                    float(plus["future"][metric]["candidate_mean_m"])
                    - float(minus["future"][metric]["candidate_mean_m"])
                )
                / (2.0 * scale)
            )
        metric_results[metric] = {
            "percent_change": _distribution(changes),
            "probability_draw_improved": float(np.mean(changes < 0.0)),
            "best_draw_percent_change": float(np.min(changes)),
            "absolute_metric_gain_m_per_m": _distribution(absolute_gain),
            "central_metric_gain_m_per_m": _distribution(central_gain),
        }
    result["metrics"] = metric_results
    return result


def run_controller_jitter_sensitivity(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str = "development",
    cases: Iterable[str] | None = None,
    jitter_scales_m: Iterable[float] = DEFAULT_JITTER_SCALES_M,
    antithetic_pair_count: int = 4,
    correlation_frames: float = 5.0,
    velocity_history_frames: int = 3,
    dt: float = 5e-5,
    num_substeps: int = 667,
    seed: int = 20260711,
    deterministic_spring_forces: bool = True,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260711,
    force: bool = False,
) -> dict[str, object]:
    """Run the locked controller-jitter sensitivity on the main release."""

    scales = tuple(sorted(set(float(value) for value in jitter_scales_m)))
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
        "method": "matched smooth controller-jitter sensitivity",
        "code_commit": code_commit,
        "cohort": cohort,
        "cases": list(selected),
        "jitter_scales_m": list(scales),
        "antithetic_pair_count": antithetic_pair_count,
        "correlation_frames": correlation_frames,
        "controller_error_mode": "translation per inferred hand group",
        "grouping_rule": "two groups for double_* and rope_double_hand; one otherwise",
        "velocity_history_frames": velocity_history_frames,
        "dt": dt,
        "num_substeps": num_substeps,
        "seed": seed,
        "matched_endpoint_state": True,
        "deterministic_spring_forces": deterministic_spring_forces,
        "training_controls_modified": False,
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
        "status": "post-hoc controller-error sensitivity diagnostic",
    }
    output = Path(output_dir)
    locked = _lock_protocol(output, specification)
    expected_base_config = {
        "jitter_scales_m": list(scales),
        "antithetic_pair_count": antithetic_pair_count,
        "correlation_frames": correlation_frames,
        "velocity_history_frames": velocity_history_frames,
        "dt": dt,
        "num_substeps": num_substeps,
        "seed": seed,
        "deterministic_spring_forces": deterministic_spring_forces,
        "device": "cuda:0",
        "runtime": runtime,
    }
    scale_ids = tuple(controller_jitter_id(scale) for scale in scales)
    case_results: dict[str, object] = {}
    paired_restart_vs_released = {}
    paired_duplicate_null = {}
    paired_expected = {scale_id: {} for scale_id in scale_ids}
    paired_oracle = {scale_id: {} for scale_id in scale_ids}
    for case in selected:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_end, future_end = (int(value) for value in split["test"])
        if future_end != int(split["frame_len"]):
            raise ValueError(f"future split does not end at frame_len for {case}")
        case_output = output / "cases" / case
        summary_path = case_output / "summary.json"
        expected_config = {
            "train_end_frame": train_end,
            **expected_base_config,
            "self_collision": self_collision_by_case[case],
        }
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["config"] != expected_config:
                raise RuntimeError(f"cached case uses a different protocol: {case}")
        else:
            track_path = case_dir / "gt_track_3d.pkl"
            summary = apply_controller_jitter_sensitivity(
                official_repo,
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_dir / "optimal_params.pkl",
                case_dir / "checkpoint.pth",
                case_output,
                train_end_frame=train_end,
                gt_track_path=track_path if track_path.exists() else None,
                jitter_scales_m=scales,
                antithetic_pair_count=antithetic_pair_count,
                correlation_frames=correlation_frames,
                velocity_history_frames=velocity_history_frames,
                dt=dt,
                num_substeps=num_substeps,
                seed=seed,
                self_collision=self_collision_by_case[case],
                deterministic_spring_forces=deterministic_spring_forces,
            )
        baseline_metrics, restart_metrics = _future_arrays(
            summary["matched_baseline"]["future"]
        )
        paired_restart_vs_released[case] = (baseline_metrics, restart_metrics)
        duplicate_baseline, duplicate_candidate = _future_arrays(
            summary["matched_baseline"]["duplicate_restart_metric_null"]
        )
        paired_duplicate_null[case] = (duplicate_baseline, duplicate_candidate)
        case_scales = {}
        for scale_id in scale_ids:
            scale_result = summary["scales"][scale_id]
            draws = scale_result["draws"]
            expected_candidate = {
                metric: np.mean(
                    [
                        np.asarray(
                            draw["future"][metric]["candidate_by_frame_m"],
                            dtype=float,
                        )
                        for draw in draws
                    ],
                    axis=0,
                )
                for metric in restart_metrics
            }
            oracle_candidate = {}
            for metric in restart_metrics:
                best = min(
                    draws,
                    key=lambda draw: float(
                        draw["future"][metric]["candidate_mean_m"]
                    ),
                )
                oracle_candidate[metric] = np.asarray(
                    best["future"][metric]["candidate_by_frame_m"], dtype=float
                )
            paired_expected[scale_id][case] = (
                restart_metrics,
                expected_candidate,
            )
            paired_oracle[scale_id][case] = (restart_metrics, oracle_candidate)
            case_scales[scale_id] = {
                "target_controller_vector_rms_m": scale_result[
                    "target_controller_vector_rms_m"
                ],
                "draw_count": scale_result["draw_count"],
                "sensitivity": _scale_case_sensitivity(scale_result),
                "draws": draws,
            }
        case_results[case] = {
            "physical_object": clusters[case],
            "controller": summary["controller"],
            "matched_baseline": summary["matched_baseline"],
            "scales": case_scales,
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

    sensitivity_aggregate = {}
    for scale_id in scale_ids:
        metric_names = tuple(
            case_results[selected[0]]["scales"][scale_id]["sensitivity"][
                "metrics"
            ]
        )
        sensitivity_aggregate[scale_id] = {
            "target_controller_vector_rms_m": case_results[selected[0]]["scales"]
            [scale_id]["target_controller_vector_rms_m"],
            "future_object_displacement_gain": _distribution(
                case_results[case]["scales"][scale_id]["sensitivity"][
                    "future_object_displacement_gain"
                ]["median"]
                for case in selected
            ),
            "metrics": {
                metric: {
                    "probability_draw_improved_equal_case_mean": float(
                        np.mean(
                            [
                                case_results[case]["scales"][scale_id][
                                    "sensitivity"
                                ]["metrics"][metric]["probability_draw_improved"]
                                for case in selected
                            ]
                        )
                    ),
                    "probability_any_draw_improved_case": float(
                        np.mean(
                            [
                                case_results[case]["scales"][scale_id][
                                    "sensitivity"
                                ]["metrics"][metric]["best_draw_percent_change"]
                                < 0.0
                                for case in selected
                            ]
                        )
                    ),
                    "best_draw_percent_change": _distribution(
                        case_results[case]["scales"][scale_id]["sensitivity"][
                            "metrics"
                        ][metric]["best_draw_percent_change"]
                        for case in selected
                    ),
                    "absolute_metric_gain_m_per_m": _distribution(
                        case_results[case]["scales"][scale_id]["sensitivity"][
                            "metrics"
                        ][metric]["absolute_metric_gain_m_per_m"]["median"]
                        for case in selected
                    ),
                    "central_metric_gain_m_per_m": _distribution(
                        case_results[case]["scales"][scale_id]["sensitivity"][
                            "metrics"
                        ][metric]["central_metric_gain_m_per_m"]["median"]
                        for case in selected
                    ),
                }
                for metric in metric_names
            },
        }
    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": specification["status"],
        "cohort": cohort,
        "case_count": len(selected),
        "physical_object_count": len(set(clusters.values())),
        "jitter_scales_m": list(scales),
        "case_results": case_results,
        "matched_restart_vs_released": bootstrap(paired_restart_vs_released),
        "duplicate_restart_metric_null": bootstrap(paired_duplicate_null),
        "expected_random_jitter_vs_matched": {
            scale_id: bootstrap(paired) for scale_id, paired in paired_expected.items()
        },
        "future_label_oracle_vs_matched": {
            scale_id: bootstrap(paired) for scale_id, paired in paired_oracle.items()
        },
        "sensitivity": sensitivity_aggregate,
        "interpretation_boundary": {
            "expected_random_jitter": "causal sensitivity, not a correction method",
            "future_label_oracle": (
                "post-hoc upper envelope selected with future labels; never a "
                "reportable predictive method"
            ),
            "latent_control_gate": (
                "high sensitivity is necessary but not sufficient; a latent-control "
                "model must select corrections from training observations only"
            ),
        },
    }
    result_path = output / "controller_jitter_sensitivity_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(result_path.resolve())
    return result
