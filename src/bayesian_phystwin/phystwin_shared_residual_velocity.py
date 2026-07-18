"""Cross-episode residual-velocity priors with causal local adaptation."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


@dataclass(frozen=True)
class SharedResidualVelocityEpisode:
    """One released interaction and its causal split."""

    case: str
    final_data: str
    baseline_trajectory: str
    gt_track_3d: str
    fit_end_frame: int
    train_end_frame: int


@dataclass(frozen=True)
class SharedResidualVelocityConfig:
    """Cross-validation, shrinkage, and rollout settings."""

    smoothing_candidates: tuple[float, ...] = (0.25, 0.5)
    global_ridge: float = 1.0
    local_prior_strength_candidates: tuple[float, ...] = (10.0, 100.0, 1000.0)
    maximum_training_points: int = 512
    interpolation_neighbors: int = 4
    controller_kernel_fraction: float = 0.25
    maximum_velocity_multiplier: float = 2.0
    maximum_residual_m: float = 0.01
    minimum_fold_improvement: float = 0.01
    maximum_fold_metric_ratio: float = 1.02
    minimum_development_improvement: float = 0.03
    minimum_both_win_count: int = 2


@dataclass
class _LoadedEpisode:
    spec: SharedResidualVelocityEpisode
    data: dict[str, Any]
    baseline: np.ndarray
    gt_track: np.ndarray
    observed: np.ndarray
    visible: np.ndarray
    valid: np.ndarray
    controllers: np.ndarray
    residual: np.ndarray
    object_scale: float
    controller_kernel_fraction: float
    num_surface_points: int
    lift_indices: np.ndarray
    lift_weights: np.ndarray


@dataclass(frozen=True)
class _PreparedTraining:
    features: np.ndarray
    targets: np.ndarray
    endpoint_state: np.ndarray
    endpoint_velocity: np.ndarray
    velocity_cap: float


def _load_manifest(path: str | Path) -> list[SharedResidualVelocityEpisode]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError("manifest must contain an episodes list")
    episodes = [SharedResidualVelocityEpisode(**item) for item in payload["episodes"]]
    if len(episodes) < 3:
        raise ValueError("shared development requires at least three episodes")
    cases = [episode.case for episode in episodes]
    if len(set(cases)) != len(cases):
        raise ValueError("episode case names must be unique")
    return episodes


def _load_episode(spec: SharedResidualVelocityEpisode, config: SharedResidualVelocityConfig) -> _LoadedEpisode:
    data = _load_pickle(spec.final_data)
    baseline = np.asarray(_load_pickle(spec.baseline_trajectory), dtype=float)
    gt_track = np.asarray(_load_pickle(spec.gt_track_3d), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controllers = np.asarray(data["controller_points"], dtype=float)
    frame_count, original_count, _ = observed.shape
    if not 4 < spec.fit_end_frame < spec.train_end_frame < frame_count:
        raise ValueError(f"invalid causal split for {spec.case}")
    if baseline.shape[0] < frame_count or baseline.shape[1] < original_count:
        raise ValueError(f"baseline trajectory does not cover {spec.case}")
    baseline = baseline[:frame_count]
    valid = _target_validity(visible, motion_valid)
    residual = observed - baseline[:, :original_count]
    initial = baseline[0, :original_count]
    centroid = np.mean(initial, axis=0)
    object_scale = float(np.sqrt(np.mean(np.sum(np.square(initial - centroid), axis=1))))
    if not np.isfinite(object_scale) or object_scale <= 1e-8:
        raise ValueError(f"degenerate object scale for {spec.case}")
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, config.interpolation_neighbors
    )
    return _LoadedEpisode(
        spec=spec,
        data=data,
        baseline=baseline,
        gt_track=gt_track,
        observed=observed,
        visible=visible,
        valid=valid,
        controllers=controllers,
        residual=residual,
        object_scale=object_scale,
        controller_kernel_fraction=config.controller_kernel_fraction,
        num_surface_points=original_count + len(np.asarray(data["surface_points"])),
        lift_indices=lift_indices,
        lift_weights=lift_weights,
    )


def _smooth_residual(
    episode: _LoadedEpisode,
    *,
    end_frame: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    filled = _temporally_fill(episode.residual, episode.valid, end_frame)
    smoothed = filled.copy()
    for frame in range(1, end_frame):
        smoothed[frame] = (
            smoothing * filled[frame] + (1.0 - smoothing) * smoothed[frame - 1]
        )
    velocity = np.zeros_like(smoothed)
    velocity[1:] = np.diff(smoothed, axis=0)
    return smoothed, velocity


def _training_indices(
    valid: np.ndarray,
    *,
    end_frame: int,
    maximum_points: int,
) -> np.ndarray:
    supported = np.flatnonzero(np.any(valid[:end_frame], axis=0))
    if len(supported) == 0:
        raise ValueError("episode has no supported object tracks")
    if len(supported) <= maximum_points:
        return supported
    positions = np.linspace(0, len(supported) - 1, maximum_points, dtype=int)
    return supported[positions]


def _exogenous_features(
    episode: _LoadedEpisode,
    frame: int,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    if not 1 <= frame < len(episode.baseline):
        raise ValueError("exogenous feature frame must be positive and in range")
    original_count = episode.observed.shape[1]
    point_indices = (
        np.arange(original_count, dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    current = episode.baseline[frame, point_indices]
    previous = episode.baseline[frame - 1, point_indices]
    baseline_velocity = (current - previous) / episode.object_scale
    all_previous = episode.baseline[frame - 1, :original_count]
    local_position = (
        previous - np.mean(all_previous, axis=0)[None]
    ) / episode.object_scale
    controller = episode.controllers[frame]
    previous_controller = episode.controllers[frame - 1]
    differences = controller[None] - current[:, None]
    squared_distance = np.sum(np.square(differences), axis=2)
    nearest = np.argmin(squared_distance, axis=1)
    rows = np.arange(len(point_indices))
    nearest_vector = differences[rows, nearest] / episode.object_scale
    controller_velocity = (controller - previous_controller) / episode.object_scale
    nearest_velocity = controller_velocity[nearest]
    width = max(episode.object_scale * episode.object_scale, 1e-12)
    width *= 2.0 * episode.controller_kernel_fraction**2
    proximity = np.exp(-squared_distance[rows, nearest] / width)[:, None]
    weighted_vector = proximity * nearest_vector
    weighted_velocity = proximity * nearest_velocity
    global_controller_velocity = np.mean(controller_velocity, axis=0)
    object_velocity = (
        np.mean(episode.baseline[frame, :original_count], axis=0)
        - np.mean(all_previous, axis=0)
    ) / episode.object_scale
    global_controller = np.repeat(
        global_controller_velocity[None], len(point_indices), axis=0
    )
    global_object = np.repeat(object_velocity[None], len(point_indices), axis=0)
    return np.concatenate(
        (
            baseline_velocity,
            local_position,
            nearest_vector,
            nearest_velocity,
            weighted_vector,
            weighted_velocity,
            proximity,
            global_controller,
            global_object,
        ),
        axis=1,
    )


def _prepare_training(
    episode: _LoadedEpisode,
    *,
    end_frame: int,
    smoothing: float,
    maximum_points: int,
    velocity_multiplier: float,
) -> _PreparedTraining:
    smoothed, velocity = _smooth_residual(
        episode, end_frame=end_frame, smoothing=smoothing
    )
    indices = _training_indices(
        episode.valid, end_frame=end_frame, maximum_points=maximum_points
    )
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for frame in range(2, end_frame):
        feature_blocks.append(
            np.concatenate(
                (
                    smoothed[frame - 1, indices] / episode.object_scale,
                    velocity[frame - 1, indices] / episode.object_scale,
                    _exogenous_features(episode, frame, indices),
                ),
                axis=1,
            )
        )
        target_blocks.append(velocity[frame, indices] / episode.object_scale)
    normalized_velocity_norm = np.linalg.norm(
        velocity[1:, indices] / episode.object_scale, axis=2
    )
    velocity_cap = max(
        velocity_multiplier * float(np.quantile(normalized_velocity_norm, 0.99)),
        1e-6,
    )
    return _PreparedTraining(
        features=np.concatenate(feature_blocks, axis=0),
        targets=np.concatenate(target_blocks, axis=0),
        endpoint_state=_temporally_fill(
            episode.residual, episode.valid, end_frame
        )[-1]
        / episode.object_scale,
        endpoint_velocity=velocity[-1] / episode.object_scale,
        velocity_cap=velocity_cap,
    )


def _fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def _design(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (features - mean) / scale
    return np.concatenate(
        (standardized, np.ones((len(standardized), 1), dtype=float)), axis=1
    )


def _fit_global_model(
    prepared: list[_PreparedTraining],
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate([item.features for item in prepared], axis=0)
    targets = np.concatenate([item.targets for item in prepared], axis=0)
    mean, scale = _fit_scaler(features)
    design = _design(features, mean, scale)
    penalty = ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return coefficients, mean, scale


def _adapt_local_model(
    global_coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    target: _PreparedTraining,
    *,
    prior_strength: float,
) -> np.ndarray:
    design = _design(target.features, mean, scale)
    penalty = prior_strength * np.eye(design.shape[1])
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target.targets + penalty @ global_coefficients,
    )


def _rollout(
    episode: _LoadedEpisode,
    prepared: _PreparedTraining,
    coefficients: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    maximum_residual_m: float,
) -> np.ndarray:
    state = prepared.endpoint_state.copy()
    velocity = prepared.endpoint_velocity.copy()
    result = np.empty(
        (end_frame - start_frame, episode.observed.shape[1], 3), dtype=float
    )
    normalized_residual_cap = maximum_residual_m / episode.object_scale
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        features = np.concatenate(
            (state, velocity, _exogenous_features(episode, frame)), axis=1
        )
        predicted_velocity = _design(features, feature_mean, feature_scale) @ coefficients
        velocity_norm = np.linalg.norm(predicted_velocity, axis=1, keepdims=True)
        predicted_velocity *= np.minimum(
            1.0, prepared.velocity_cap / np.maximum(velocity_norm, 1e-12)
        )
        state = state + predicted_velocity
        state_norm = np.linalg.norm(state, axis=1, keepdims=True)
        state *= np.minimum(
            1.0, normalized_residual_cap / np.maximum(state_norm, 1e-12)
        )
        velocity = predicted_velocity
        result[output_index] = state * episode.object_scale
    return result


def _interval_metrics(
    episode: _LoadedEpisode,
    tracked: np.ndarray | None,
    *,
    start_frame: int,
    end_frame: int,
    config: SharedResidualVelocityConfig,
) -> tuple[dict[str, object], np.ndarray]:
    if tracked is None:
        candidate = episode.baseline
        correction = np.zeros_like(episode.baseline[start_frame:end_frame])
    else:
        correction = _lift_residual(
            tracked,
            episode.baseline.shape[1],
            episode.lift_indices,
            episode.lift_weights,
            maximum_norm=config.maximum_residual_m,
        )
        candidate = episode.baseline.copy()
        candidate[start_frame:end_frame] += correction
    metrics = evaluate_official_phystwin_interval(
        candidate,
        episode.observed,
        episode.visible,
        episode.gt_track,
        num_surface_points=episode.num_surface_points,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return metrics, correction


def _ratios(
    candidate: dict[str, object], reference: dict[str, object]
) -> dict[str, float]:
    return {
        metric: float(candidate[metric]) / float(reference[metric])
        for metric in ("chamfer_distance_m", "track_error_m")
    }


def _evaluate_fold(
    target: _LoadedEpisode,
    sources: list[_LoadedEpisode],
    *,
    smoothing: float,
    local_prior_strength: float,
    config: SharedResidualVelocityConfig,
) -> dict[str, object]:
    source_prepared = [
        _prepare_training(
            source,
            end_frame=source.spec.fit_end_frame,
            smoothing=smoothing,
            maximum_points=config.maximum_training_points,
            velocity_multiplier=config.maximum_velocity_multiplier,
        )
        for source in sources
    ]
    target_prepared = _prepare_training(
        target,
        end_frame=target.spec.fit_end_frame,
        smoothing=smoothing,
        maximum_points=config.maximum_training_points,
        velocity_multiplier=config.maximum_velocity_multiplier,
    )
    global_coefficients, mean, scale = _fit_global_model(
        source_prepared, ridge=config.global_ridge
    )
    coefficients = _adapt_local_model(
        global_coefficients,
        mean,
        scale,
        target_prepared,
        prior_strength=local_prior_strength,
    )
    dynamic = _rollout(
        target,
        target_prepared,
        coefficients,
        mean,
        scale,
        start_frame=target.spec.fit_end_frame,
        end_frame=target.spec.train_end_frame,
        maximum_residual_m=config.maximum_residual_m,
    )
    dynamic_metrics, _ = _interval_metrics(
        target,
        dynamic,
        start_frame=target.spec.fit_end_frame,
        end_frame=target.spec.train_end_frame,
        config=config,
    )
    persistence = np.repeat(
        target_prepared.endpoint_state[None] * target.object_scale,
        target.spec.train_end_frame - target.spec.fit_end_frame,
        axis=0,
    )
    persistence_metrics, _ = _interval_metrics(
        target,
        persistence,
        start_frame=target.spec.fit_end_frame,
        end_frame=target.spec.train_end_frame,
        config=config,
    )
    baseline_metrics, _ = _interval_metrics(
        target,
        None,
        start_frame=target.spec.fit_end_frame,
        end_frame=target.spec.train_end_frame,
        config=config,
    )
    ratios = _ratios(dynamic_metrics, persistence_metrics)
    accepted = (
        0.5 * sum(ratios.values())
        < 1.0 - config.minimum_fold_improvement
        and max(ratios.values()) <= config.maximum_fold_metric_ratio
    )
    return {
        "case": target.spec.case,
        "accepted": accepted,
        "ratios_relative_to_persistence": ratios,
        "baseline_official_evaluation": baseline_metrics,
        "persistence_official_evaluation": persistence_metrics,
        "dynamic_official_evaluation": dynamic_metrics,
    }


def _aggregate_folds(folds: list[dict[str, object]]) -> dict[str, object]:
    aggregate_ratios = {
        metric: float(
            np.mean(
                [fold["ratios_relative_to_persistence"][metric] for fold in folds]
            )
        )
        for metric in ("chamfer_distance_m", "track_error_m")
    }
    return {
        "aggregate_ratios_relative_to_persistence": aggregate_ratios,
        "balanced_improvement": 1.0 - 0.5 * sum(aggregate_ratios.values()),
        "both_win_count": sum(
            max(fold["ratios_relative_to_persistence"].values()) < 1.0
            for fold in folds
        ),
        "accepted_fold_count": sum(bool(fold["accepted"]) for fold in folds),
    }


def fit_shared_residual_velocity_development(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: SharedResidualVelocityConfig = SharedResidualVelocityConfig(),
) -> dict[str, object]:
    """Cross-fit a shared prior and open development futures only after its gate."""

    if not config.smoothing_candidates or any(
        not 0.0 < value <= 1.0 for value in config.smoothing_candidates
    ):
        raise ValueError("smoothing candidates must lie in (0, 1]")
    if config.global_ridge <= 0.0:
        raise ValueError("global_ridge must be positive")
    if not config.local_prior_strength_candidates or any(
        value <= 0.0 for value in config.local_prior_strength_candidates
    ):
        raise ValueError("local prior strengths must be positive")
    if config.maximum_training_points < 1:
        raise ValueError("maximum_training_points must be positive")
    if config.interpolation_neighbors < 1:
        raise ValueError("interpolation_neighbors must be positive")
    if not 0.0 < config.controller_kernel_fraction:
        raise ValueError("controller_kernel_fraction must be positive")
    if config.maximum_velocity_multiplier <= 0.0 or config.maximum_residual_m <= 0.0:
        raise ValueError("rollout caps must be positive")

    specs = _load_manifest(manifest_path)
    episodes = [_load_episode(spec, config) for spec in specs]
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, int, float, float], dict[str, object]] | None = None
    for smoothing in config.smoothing_candidates:
        for local_prior_strength in config.local_prior_strength_candidates:
            folds = [
                _evaluate_fold(
                    target,
                    [source for source in episodes if source is not target],
                    smoothing=smoothing,
                    local_prior_strength=local_prior_strength,
                    config=config,
                )
                for target in episodes
            ]
            aggregate = _aggregate_folds(folds)
            candidate = {
                "smoothing": smoothing,
                "local_prior_strength": local_prior_strength,
                "folds": folds,
                **aggregate,
            }
            candidates.append(candidate)
            ranking = (
                -float(aggregate["balanced_improvement"]),
                -int(aggregate["both_win_count"]),
                smoothing,
                -local_prior_strength,
            )
            if best is None or ranking < best[0]:
                best = (ranking, candidate)
    assert best is not None
    selected = best[1]
    aggregate_ratios = selected["aggregate_ratios_relative_to_persistence"]
    development_gate_passed = (
        float(selected["balanced_improvement"])
        >= config.minimum_development_improvement
        and int(selected["both_win_count"]) >= config.minimum_both_win_count
        and max(float(value) for value in aggregate_ratios.values()) < 1.0
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    future_results: list[dict[str, object]] = []
    if development_gate_passed:
        smoothing = float(selected["smoothing"])
        prior_strength = float(selected["local_prior_strength"])
        validation_by_case = {fold["case"]: fold for fold in selected["folds"]}
        for target in episodes:
            sources = [source for source in episodes if source is not target]
            source_prepared = [
                _prepare_training(
                    source,
                    end_frame=source.spec.train_end_frame,
                    smoothing=smoothing,
                    maximum_points=config.maximum_training_points,
                    velocity_multiplier=config.maximum_velocity_multiplier,
                )
                for source in sources
            ]
            target_prepared = _prepare_training(
                target,
                end_frame=target.spec.train_end_frame,
                smoothing=smoothing,
                maximum_points=config.maximum_training_points,
                velocity_multiplier=config.maximum_velocity_multiplier,
            )
            global_coefficients, mean, scale = _fit_global_model(
                source_prepared, ridge=config.global_ridge
            )
            coefficients = _adapt_local_model(
                global_coefficients,
                mean,
                scale,
                target_prepared,
                prior_strength=prior_strength,
            )
            use_dynamic = bool(validation_by_case[target.spec.case]["accepted"])
            future_count = len(target.observed) - target.spec.train_end_frame
            if use_dynamic:
                tracked = _rollout(
                    target,
                    target_prepared,
                    coefficients,
                    mean,
                    scale,
                    start_frame=target.spec.train_end_frame,
                    end_frame=len(target.observed),
                    maximum_residual_m=config.maximum_residual_m,
                )
                selected_method = "shared_residual_velocity"
            else:
                tracked = np.repeat(
                    target_prepared.endpoint_state[None] * target.object_scale,
                    future_count,
                    axis=0,
                )
                selected_method = "persistence"
            selected_metrics, correction = _interval_metrics(
                target,
                tracked,
                start_frame=target.spec.train_end_frame,
                end_frame=len(target.observed),
                config=config,
            )
            persistence = np.repeat(
                target_prepared.endpoint_state[None] * target.object_scale,
                future_count,
                axis=0,
            )
            persistence_metrics, _ = _interval_metrics(
                target,
                persistence,
                start_frame=target.spec.train_end_frame,
                end_frame=len(target.observed),
                config=config,
            )
            baseline_metrics, _ = _interval_metrics(
                target,
                None,
                start_frame=target.spec.train_end_frame,
                end_frame=len(target.observed),
                config=config,
            )
            trajectory = target.baseline.copy()
            trajectory[target.spec.train_end_frame :] += correction
            case_output = output / target.spec.case
            case_output.mkdir(parents=True, exist_ok=True)
            trajectory_path = case_output / "trajectory.pkl"
            with trajectory_path.open("wb") as handle:
                pickle.dump(
                    trajectory.astype(np.float32),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            model_path = case_output / "model.npz"
            np.savez_compressed(
                model_path,
                global_coefficients=global_coefficients,
                local_coefficients=coefficients,
                feature_mean=mean,
                feature_scale=scale,
            )
            future_results.append(
                {
                    "case": target.spec.case,
                    "selected_method": selected_method,
                    "baseline_official_evaluation": baseline_metrics,
                    "persistence_official_evaluation": persistence_metrics,
                    "selected_official_evaluation": selected_metrics,
                    "ratios_relative_to_persistence": _ratios(
                        selected_metrics, persistence_metrics
                    ),
                    "trajectory": str(trajectory_path.resolve()),
                    "model": str(model_path.resolve()),
                }
            )

    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "selection": "leave-one-episode-out validation suffixes",
            "global_prior": "other episodes only within each fold",
            "local_adaptation": "target prefix only",
            "future_observations_used": False,
            "future_opened_only_after_development_gate": True,
            "physical_injection_claim": False,
        },
        "manifest": {
            "path": str(Path(manifest_path).resolve()),
            "sha256": _sha256(manifest_path),
        },
        "inputs": [
            {
                "case": spec.case,
                "final_data_sha256": _sha256(spec.final_data),
                "baseline_trajectory_sha256": _sha256(spec.baseline_trajectory),
                "gt_track_3d_sha256": _sha256(spec.gt_track_3d),
            }
            for spec in specs
        ],
        "selection": {
            "development_gate_passed": development_gate_passed,
            "selected_candidate": selected,
            "candidates": candidates,
        },
        "future_metrics_opened": development_gate_passed,
        "future_results": future_results,
    }
    summary_path = output / "summary.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
