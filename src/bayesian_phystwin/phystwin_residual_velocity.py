"""Causal recurrent residual-velocity forecasting for PhysTwin trajectories."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _project_residuals,
    _sha256,
    _standardize_actions,
    _target_validity,
    _temporally_fill,
    controller_action_features,
    fit_residual_basis,
)


@dataclass(frozen=True)
class PhysTwinResidualVelocityConfig:
    """Selection and stability settings for recurrent velocity correction."""

    fit_end_frame: int
    train_end_frame: int
    rank_candidates: tuple[int, ...] = (1, 2, 4, 8)
    velocity_persistence_candidates: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.0)
    ridge_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    projection_ridge: float = 1e-6
    interpolation_neighbors: int = 4
    maximum_state_multiplier: float = 1.5
    maximum_velocity_multiplier: float = 1.5
    maximum_residual_m: float = 0.01
    minimum_validation_improvement: float = 0.0
    minimum_dynamic_improvement: float = 0.01
    maximum_metric_ratio: float = 1.02


def physical_rollout_features(object_points: np.ndarray) -> np.ndarray:
    """Summarize the known baseline rollout without using future observations."""

    points = np.asarray(object_points, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] == 0:
        raise ValueError("object_points must have shape (T, N, 3) with N > 0")
    if not np.all(np.isfinite(points)):
        raise ValueError("object_points must contain finite values")
    centroid = np.mean(points, axis=1)
    spread = np.sqrt(np.mean(np.square(points - centroid[:, None, :]), axis=(1, 2)))[
        :, None
    ]
    centroid_delta = np.zeros_like(centroid)
    spread_delta = np.zeros_like(spread)
    centroid_delta[1:] = np.diff(centroid, axis=0)
    spread_delta[1:] = np.diff(spread, axis=0)
    rms_motion = np.zeros((len(points), 1), dtype=float)
    rms_motion[1:, 0] = np.sqrt(
        np.mean(np.square(np.diff(points, axis=0)), axis=(1, 2))
    )
    return np.concatenate(
        (
            centroid - centroid[0],
            spread - spread[0],
            centroid_delta,
            spread_delta,
            rms_motion,
        ),
        axis=1,
    )


def residual_velocity_features(
    controller_points: np.ndarray,
    baseline_object_points: np.ndarray,
) -> np.ndarray:
    """Combine commanded motion and simulated-state features available at rollout."""

    controller = np.asarray(controller_points, dtype=float)
    baseline = np.asarray(baseline_object_points, dtype=float)
    if len(controller) != len(baseline):
        raise ValueError("controller and baseline trajectories must share frame count")
    controller_features = controller_action_features(controller)
    object_features = physical_rollout_features(baseline)
    controller_centroid = np.mean(controller, axis=1)
    object_centroid = np.mean(baseline, axis=1)
    relative_position = controller_centroid - object_centroid
    relative_velocity = np.zeros_like(relative_position)
    relative_velocity[1:] = np.diff(relative_position, axis=0)
    return np.concatenate(
        (
            controller_features,
            object_features,
            relative_position - relative_position[0],
            relative_velocity,
        ),
        axis=1,
    )


def _latent_scales(latent: np.ndarray, *, end_frame: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(latent[:end_frame], dtype=float)
    velocity = np.zeros_like(state)
    velocity[1:] = np.diff(state, axis=0)
    state_rms = np.sqrt(np.mean(np.square(state[1:]), axis=0))
    velocity_rms = np.sqrt(np.mean(np.square(velocity[1:]), axis=0))
    state_floor = max(float(np.max(state_rms, initial=0.0)) * 1e-3, 1e-8)
    velocity_floor = max(float(np.max(velocity_rms, initial=0.0)) * 1e-3, 1e-9)
    return np.maximum(state_rms, state_floor), np.maximum(velocity_rms, velocity_floor)


def fit_latent_residual_velocity(
    latent: np.ndarray,
    standardized_features: np.ndarray,
    *,
    end_frame: int,
    velocity_persistence: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one-step residual velocity while preserving recursive rollout state."""

    values = np.asarray(latent, dtype=float)
    features = np.asarray(standardized_features, dtype=float)
    if not 3 <= end_frame <= len(values) or len(features) < end_frame:
        raise ValueError("end_frame must contain at least three aligned frames")
    if not 0.0 <= velocity_persistence <= 1.0:
        raise ValueError("velocity_persistence must lie in [0, 1]")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    state_scale, velocity_scale = _latent_scales(values, end_frame=end_frame)
    velocity = np.zeros_like(values)
    velocity[1:] = np.diff(values, axis=0)
    normalized_state = values / state_scale
    normalized_velocity = velocity / velocity_scale
    design = np.concatenate(
        (
            normalized_state[1 : end_frame - 1],
            features[2:end_frame],
            np.ones((end_frame - 2, 1), dtype=float),
        ),
        axis=1,
    )
    target = (
        normalized_velocity[2:end_frame]
        - velocity_persistence * normalized_velocity[1 : end_frame - 1]
    )
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return coefficients, state_scale, velocity_scale


def rollout_latent_residual_velocity(
    initial_state: np.ndarray,
    initial_velocity: np.ndarray,
    standardized_features: np.ndarray,
    coefficients: np.ndarray,
    state_scale: np.ndarray,
    velocity_scale: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    velocity_persistence: float,
    state_norm_cap: float,
    velocity_norm_cap: float,
) -> np.ndarray:
    """Recursively integrate predicted velocity using the model's own states."""

    if not 0 <= start_frame <= end_frame <= len(standardized_features):
        raise ValueError("rollout interval lies outside the feature trajectory")
    previous_state = np.asarray(initial_state, dtype=float).copy()
    previous_velocity = np.asarray(initial_velocity, dtype=float).copy()
    result = np.empty((end_frame - start_frame, len(previous_state)), dtype=float)
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        design = np.concatenate(
            (
                previous_state / state_scale,
                standardized_features[frame],
                [1.0],
            )
        )
        normalized_velocity = (
            velocity_persistence * (previous_velocity / velocity_scale)
            + design @ coefficients
        )
        velocity_norm = float(np.linalg.norm(normalized_velocity))
        if velocity_norm > velocity_norm_cap:
            normalized_velocity *= velocity_norm_cap / velocity_norm
        current_velocity = normalized_velocity * velocity_scale
        current_state = previous_state + current_velocity
        state_norm = float(np.linalg.norm(current_state))
        if state_norm > state_norm_cap:
            current_state *= state_norm_cap / state_norm
            current_velocity = current_state - previous_state
        result[output_index] = current_state
        previous_state = current_state
        previous_velocity = current_velocity
    return result


def _norm_caps(
    latent: np.ndarray,
    velocity_scale: np.ndarray,
    *,
    state_multiplier: float,
    velocity_multiplier: float,
) -> tuple[float, float]:
    velocity = np.zeros_like(latent)
    velocity[1:] = np.diff(latent, axis=0)
    normalized_velocity = velocity / velocity_scale
    state_cap = max(
        state_multiplier * float(np.max(np.linalg.norm(latent, axis=1), initial=0.0)),
        1e-8,
    )
    velocity_cap = max(
        velocity_multiplier
        * float(np.max(np.linalg.norm(normalized_velocity, axis=1), initial=0.0)),
        1e-8,
    )
    return state_cap, velocity_cap


def _metric_ratios(
    metrics: dict[str, object],
    reference: dict[str, object],
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for name in ("chamfer_distance_m", "track_error_m"):
        value = float(metrics[name])
        denominator = float(reference[name])
        if denominator > 0.0:
            ratios[name] = value / denominator
        else:
            ratios[name] = 1.0 if value == 0.0 else float("inf")
    return ratios


def _relative_score(
    metrics: dict[str, object],
    reference: dict[str, object],
) -> float:
    ratios = _metric_ratios(metrics, reference)
    return 0.5 * (ratios["chamfer_distance_m"] + ratios["track_error_m"])


def fit_recurrent_residual_velocity(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinResidualVelocityConfig,
) -> dict[str, object]:
    """Select on a recursive rollout, refit, and forecast without future observations."""

    if not 3 < config.fit_end_frame < config.train_end_frame:
        raise ValueError("expected 3 < fit_end_frame < train_end_frame")
    if not config.rank_candidates or any(rank < 1 for rank in config.rank_candidates):
        raise ValueError("rank_candidates must contain positive integers")
    if not config.velocity_persistence_candidates or any(
        not 0.0 <= value <= 1.0
        for value in config.velocity_persistence_candidates
    ):
        raise ValueError("velocity persistence candidates must lie in [0, 1]")
    if not config.ridge_candidates or any(value <= 0.0 for value in config.ridge_candidates):
        raise ValueError("ridge candidates must be positive")
    if config.projection_ridge <= 0.0:
        raise ValueError("projection_ridge must be positive")
    if config.maximum_state_multiplier <= 0.0:
        raise ValueError("maximum_state_multiplier must be positive")
    if config.maximum_velocity_multiplier <= 0.0:
        raise ValueError("maximum_velocity_multiplier must be positive")
    if config.maximum_residual_m <= 0.0:
        raise ValueError("maximum_residual_m must be positive")
    if config.minimum_validation_improvement < 0.0:
        raise ValueError("minimum_validation_improvement must be nonnegative")
    if config.minimum_dynamic_improvement < 0.0:
        raise ValueError("minimum_dynamic_improvement must be nonnegative")
    if config.maximum_metric_ratio < 1.0:
        raise ValueError("maximum_metric_ratio must be at least one")

    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controllers = np.asarray(data["controller_points"], dtype=float)
    frame_count, original_count, _ = observed.shape
    if not config.train_end_frame < frame_count:
        raise ValueError("train_end_frame must be below the frame count")
    if baseline.shape[0] < frame_count or baseline.shape[1] < original_count:
        raise ValueError("baseline trajectory does not cover the observations")
    baseline = baseline[:frame_count]
    valid = _target_validity(visible, motion_valid)
    residual = observed - baseline[:, :original_count]
    features = residual_velocity_features(controllers, baseline[:, :original_count])
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, config.interpolation_neighbors
    )
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))

    def metrics_for_tracked(
        tracked: np.ndarray,
        *,
        start_frame: int,
        end_frame: int,
    ) -> tuple[dict[str, object], np.ndarray]:
        lifted = _lift_residual(
            tracked,
            baseline.shape[1],
            lift_indices,
            lift_weights,
            maximum_norm=config.maximum_residual_m,
        )
        candidate = baseline.copy()
        candidate[start_frame:end_frame] += lifted
        metrics = evaluate_official_phystwin_interval(
            candidate,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        return metrics, lifted

    baseline_validation = evaluate_official_phystwin_interval(
        baseline,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=config.fit_end_frame,
        end_frame=config.train_end_frame,
    )
    validation_count = config.train_end_frame - config.fit_end_frame
    fit_filled = _temporally_fill(residual, valid, config.fit_end_frame)
    persistence_validation_tracked = np.repeat(
        fit_filled[-1][None], validation_count, axis=0
    )
    persistence_validation, _ = metrics_for_tracked(
        persistence_validation_tracked,
        start_frame=config.fit_end_frame,
        end_frame=config.train_end_frame,
    )
    persistence_score = _relative_score(persistence_validation, baseline_validation)

    maximum_rank = min(max(config.rank_candidates), config.fit_end_frame - 1)
    fit_basis = fit_residual_basis(
        residual,
        valid,
        end_frame=config.fit_end_frame,
        maximum_rank=maximum_rank,
    )
    standardized_fit, _, _ = _standardize_actions(
        features, end_frame=config.fit_end_frame
    )
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, int, float, float], dict[str, object]] | None = None
    for rank in sorted(set(config.rank_candidates)):
        if rank > len(fit_basis):
            continue
        basis = fit_basis[:rank]
        latent = _project_residuals(
            residual[: config.fit_end_frame],
            valid[: config.fit_end_frame],
            basis,
            ridge=config.projection_ridge,
        )
        for velocity_persistence in config.velocity_persistence_candidates:
            for ridge in config.ridge_candidates:
                coefficients, state_scale, velocity_scale = (
                    fit_latent_residual_velocity(
                        latent,
                        standardized_fit,
                        end_frame=config.fit_end_frame,
                        velocity_persistence=velocity_persistence,
                        ridge=ridge,
                    )
                )
                state_cap, velocity_cap = _norm_caps(
                    latent,
                    velocity_scale,
                    state_multiplier=config.maximum_state_multiplier,
                    velocity_multiplier=config.maximum_velocity_multiplier,
                )
                predicted = rollout_latent_residual_velocity(
                    latent[-1],
                    latent[-1] - latent[-2],
                    standardized_fit,
                    coefficients,
                    state_scale,
                    velocity_scale,
                    start_frame=config.fit_end_frame,
                    end_frame=config.train_end_frame,
                    velocity_persistence=velocity_persistence,
                    state_norm_cap=state_cap,
                    velocity_norm_cap=velocity_cap,
                )
                tracked = (predicted @ basis).reshape(
                    validation_count, original_count, 3
                )
                metrics, _ = metrics_for_tracked(
                    tracked,
                    start_frame=config.fit_end_frame,
                    end_frame=config.train_end_frame,
                )
                score = _relative_score(metrics, baseline_validation)
                persistence_ratios = _metric_ratios(metrics, persistence_validation)
                candidate = {
                    "rank": rank,
                    "velocity_persistence": velocity_persistence,
                    "ridge": ridge,
                    "selection_score": score,
                    "score_relative_to_persistence": _relative_score(
                        metrics, persistence_validation
                    ),
                    "metric_ratios_relative_to_persistence": persistence_ratios,
                    "official_evaluation": metrics,
                }
                candidates.append(candidate)
                ranking = (score, rank, -ridge, -velocity_persistence)
                if best is None or ranking < best[0]:
                    best = (ranking, candidate)
    assert best is not None
    best_candidate = best[1]
    best_score = float(best_candidate["selection_score"])
    best_ratios = best_candidate["metric_ratios_relative_to_persistence"]
    dynamic_accepted = (
        best_score
        < persistence_score * (1.0 - config.minimum_dynamic_improvement)
        and max(float(value) for value in best_ratios.values())
        <= config.maximum_metric_ratio
    )
    persistence_accepted = (
        persistence_score < 1.0 - config.minimum_validation_improvement
        and max(_metric_ratios(persistence_validation, baseline_validation).values())
        <= config.maximum_metric_ratio
    )
    if dynamic_accepted:
        selected_method = "residual_velocity"
    elif persistence_accepted:
        selected_method = "persistence"
    else:
        selected_method = "baseline"

    future_count = frame_count - config.train_end_frame
    tracked_future = np.zeros((future_count, original_count, 3), dtype=float)
    model_artifact: dict[str, np.ndarray] = {
        "basis": np.empty((0, original_count * 3), dtype=float),
        "coefficients": np.empty((0, 0), dtype=float),
        "feature_mean": np.empty(0, dtype=float),
        "feature_scale": np.empty(0, dtype=float),
        "state_scale": np.empty(0, dtype=float),
        "velocity_scale": np.empty(0, dtype=float),
        "lift_indices": lift_indices,
        "lift_weights": lift_weights,
    }
    if selected_method == "persistence":
        train_filled = _temporally_fill(residual, valid, config.train_end_frame)
        tracked_future[:] = train_filled[-1]
        model_artifact["tracked_endpoint"] = train_filled[-1]
    elif selected_method == "residual_velocity":
        selected_rank = int(best_candidate["rank"])
        selected_persistence = float(best_candidate["velocity_persistence"])
        selected_ridge = float(best_candidate["ridge"])
        train_basis = fit_residual_basis(
            residual,
            valid,
            end_frame=config.train_end_frame,
            maximum_rank=selected_rank,
        )[:selected_rank]
        latent = _project_residuals(
            residual[: config.train_end_frame],
            valid[: config.train_end_frame],
            train_basis,
            ridge=config.projection_ridge,
        )
        standardized_train, feature_mean, feature_scale = _standardize_actions(
            features, end_frame=config.train_end_frame
        )
        coefficients, state_scale, velocity_scale = fit_latent_residual_velocity(
            latent,
            standardized_train,
            end_frame=config.train_end_frame,
            velocity_persistence=selected_persistence,
            ridge=selected_ridge,
        )
        state_cap, velocity_cap = _norm_caps(
            latent,
            velocity_scale,
            state_multiplier=config.maximum_state_multiplier,
            velocity_multiplier=config.maximum_velocity_multiplier,
        )
        predicted = rollout_latent_residual_velocity(
            latent[-1],
            latent[-1] - latent[-2],
            standardized_train,
            coefficients,
            state_scale,
            velocity_scale,
            start_frame=config.train_end_frame,
            end_frame=frame_count,
            velocity_persistence=selected_persistence,
            state_norm_cap=state_cap,
            velocity_norm_cap=velocity_cap,
        )
        tracked_future = (predicted @ train_basis).reshape(
            future_count, original_count, 3
        )
        model_artifact.update(
            {
                "basis": train_basis,
                "coefficients": coefficients,
                "feature_mean": feature_mean,
                "feature_scale": feature_scale,
                "state_scale": state_scale,
                "velocity_scale": velocity_scale,
            }
        )

    corrected_test, correction = metrics_for_tracked(
        tracked_future,
        start_frame=config.train_end_frame,
        end_frame=frame_count,
    )
    baseline_test = evaluate_official_phystwin_interval(
        baseline,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=config.train_end_frame,
        end_frame=frame_count,
    )
    train_filled = _temporally_fill(residual, valid, config.train_end_frame)
    persistence_test, _ = metrics_for_tracked(
        np.repeat(train_filled[-1][None], future_count, axis=0),
        start_frame=config.train_end_frame,
        end_frame=frame_count,
    )
    corrected = baseline.copy()
    corrected[config.train_end_frame :] += correction
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(corrected.astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_path = output / "residual_velocity_model.npz"
    np.savez_compressed(model_path, **model_artifact)
    correction_norm = np.linalg.norm(correction, axis=2)
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "basis_fit_interval_for_selection": [1, config.fit_end_frame],
            "selection_interval": [config.fit_end_frame, config.train_end_frame],
            "final_refit_interval": [1, config.train_end_frame],
            "future_observations_used": False,
            "future_inputs": "commanded controller trajectory and baseline PhysTwin rollout",
            "integration": "recursive residual velocity in low-rank readout space",
            "fallback": "exact baseline or temporally filled endpoint persistence",
            "physical_injection_claim": False,
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
            "gt_track_3d": {
                "path": str(Path(gt_track_path).resolve()),
                "sha256": _sha256(gt_track_path),
            },
        },
        "selection": {
            "selected_method": selected_method,
            "dynamic_accepted": dynamic_accepted,
            "persistence_accepted": persistence_accepted,
            "baseline_official_evaluation": baseline_validation,
            "persistence_official_evaluation": persistence_validation,
            "persistence_selection_score": persistence_score,
            "selected_dynamic_candidate": best_candidate,
            "candidates": candidates,
        },
        "test": {
            "baseline_official_evaluation": baseline_test,
            "persistence_official_evaluation": persistence_test,
            "corrected_official_evaluation": corrected_test,
            "selection_score_relative_to_baseline": _relative_score(
                corrected_test, baseline_test
            ),
            "selection_score_relative_to_persistence": _relative_score(
                corrected_test, persistence_test
            ),
        },
        "correction": {
            "rms_m": float(np.sqrt(np.mean(np.square(correction_norm)))),
            "maximum_m": float(np.max(correction_norm, initial=0.0)),
        },
        "outputs": {
            "trajectory": str(trajectory_path.resolve()),
            "model": str(model_path.resolve()),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
