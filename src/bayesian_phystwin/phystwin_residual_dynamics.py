"""Causal low-rank residual dynamics for released PhysTwin trajectories."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_official_evaluation import evaluate_official_phystwin_interval


@dataclass(frozen=True)
class PhysTwinResidualDynamicsConfig:
    """Selection and regularization settings for a constrained residual model."""

    fit_end_frame: int
    train_end_frame: int
    rank_candidates: tuple[int, ...] = (1, 2, 4, 8)
    persistence_candidates: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.0)
    ridge_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
    projection_ridge: float = 1e-6
    interpolation_neighbors: int = 4
    maximum_state_multiplier: float = 1.5
    maximum_residual_m: float = 0.03
    minimum_validation_improvement: float = 0.0


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_validity(visible: np.ndarray, motion_valid: np.ndarray) -> np.ndarray:
    frame_count, track_count = visible.shape
    if motion_valid.shape not in {
        (frame_count, track_count),
        (frame_count - 1, track_count),
    }:
        raise ValueError("motion_valid has an incompatible shape")
    valid = np.zeros_like(visible, dtype=bool)
    valid[0] = visible[0]
    valid[1:] = motion_valid[: frame_count - 1]
    return valid


def controller_action_features(controller_points: np.ndarray) -> np.ndarray:
    """Return compact translation, spread, and velocity control features."""

    points = np.asarray(controller_points, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] == 0:
        raise ValueError("controller_points must have shape (T, C, 3) with C > 0")
    if not np.all(np.isfinite(points)):
        raise ValueError("controller_points must contain finite values")
    centroid = np.mean(points, axis=1)
    spread = np.sqrt(np.mean(np.square(points - centroid[:, None, :]), axis=1))
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


def _temporally_fill(
    residual: np.ndarray,
    valid: np.ndarray,
    end_frame: int,
) -> np.ndarray:
    frame_indices = np.arange(end_frame)
    filled = np.zeros_like(residual[:end_frame], dtype=float)
    for track in range(residual.shape[1]):
        support = np.flatnonzero(valid[:end_frame, track])
        if len(support) == 0:
            continue
        for coordinate in range(3):
            filled[:, track, coordinate] = np.interp(
                frame_indices,
                support,
                residual[support, track, coordinate],
            )
    return filled


def fit_residual_basis(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    maximum_rank: int,
) -> np.ndarray:
    """Fit an uncentered deformation basis using only frames before ``end_frame``."""

    if not 1 < end_frame <= len(residual):
        raise ValueError("end_frame must leave at least two fitting frames")
    if maximum_rank < 1:
        raise ValueError("maximum_rank must be positive")
    filled = _temporally_fill(residual, valid, end_frame)
    matrix = filled[1:end_frame].reshape(end_frame - 1, -1)
    rank = min(maximum_rank, min(matrix.shape))
    _, _, right = np.linalg.svd(matrix, full_matrices=False)
    return right[:rank]


def _project_residuals(
    residual: np.ndarray,
    valid: np.ndarray,
    basis: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    rank = len(basis)
    basis_by_coordinate = basis.T.reshape(residual.shape[1], 3, rank)
    latent = np.zeros((len(residual), rank), dtype=float)
    regularizer = ridge * np.eye(rank)
    for frame in range(1, len(residual)):
        mask = valid[frame]
        if not np.any(mask):
            latent[frame] = latent[frame - 1]
            continue
        design = basis_by_coordinate[mask].reshape(-1, rank)
        target = residual[frame, mask].reshape(-1)
        latent[frame] = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ target,
        )
    return latent


def _standardize_actions(
    features: np.ndarray,
    *,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features[1:end_frame], axis=0)
    scale = np.std(features[1:end_frame], axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (features - mean) / scale, mean, scale


def _fit_latent_dynamics(
    latent: np.ndarray,
    standardized_actions: np.ndarray,
    *,
    end_frame: int,
    persistence: float,
    ridge: float,
) -> np.ndarray:
    design = np.concatenate(
        (
            standardized_actions[1:end_frame],
            np.ones((end_frame - 1, 1), dtype=float),
        ),
        axis=1,
    )
    target = latent[1:end_frame] - persistence * latent[: end_frame - 1]
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )


def _rollout_latent(
    initial: np.ndarray,
    standardized_actions: np.ndarray,
    dynamics: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    persistence: float,
    norm_cap: float,
) -> np.ndarray:
    result = np.empty((end_frame - start_frame, len(initial)), dtype=float)
    previous = np.asarray(initial, dtype=float).copy()
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        feature = np.concatenate((standardized_actions[frame], [1.0]))
        current = persistence * previous + feature @ dynamics
        norm = float(np.linalg.norm(current))
        if norm > norm_cap:
            current *= norm_cap / norm
        result[output_index] = current
        previous = current
    return result


def _lift_map(
    initial_vertices: np.ndarray,
    original_count: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= neighbors <= original_count:
        raise ValueError("interpolation_neighbors exceeds the original point count")
    extra = initial_vertices[original_count:]
    if len(extra) == 0:
        return (
            np.empty((0, neighbors), dtype=np.int64),
            np.empty((0, neighbors), dtype=float),
        )
    original = initial_vertices[:original_count]
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(original).query(extra, k=neighbors)
        distances = np.asarray(distances, dtype=float).reshape(len(extra), neighbors)
        indices = np.asarray(indices, dtype=np.int64).reshape(len(extra), neighbors)
    except (ImportError, OSError, ValueError):
        indices = np.empty((len(extra), neighbors), dtype=np.int64)
        distances = np.empty((len(extra), neighbors), dtype=float)
        for start in range(0, len(extra), 128):
            stop = min(start + 128, len(extra))
            squared = np.sum(
                np.square(extra[start:stop, None] - original[None, :]),
                axis=2,
            )
            local = np.argpartition(squared, neighbors - 1, axis=1)[:, :neighbors]
            indices[start:stop] = local
            distances[start:stop] = np.sqrt(
                np.take_along_axis(squared, local, axis=1)
            )
    inverse = 1.0 / np.maximum(distances, 1e-6)
    weights = inverse / np.sum(inverse, axis=1, keepdims=True)
    return indices, weights


def _clip_residual(values: np.ndarray, maximum_norm: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    scale = np.minimum(1.0, maximum_norm / np.maximum(norms, 1e-12))
    return values * scale


def _lift_residual(
    tracked_residual: np.ndarray,
    state_count: int,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    maximum_norm: float,
) -> np.ndarray:
    original_count = tracked_residual.shape[1]
    lifted = np.zeros((len(tracked_residual), state_count, 3), dtype=float)
    lifted[:, :original_count] = tracked_residual
    if state_count > original_count:
        lifted[:, original_count:] = np.sum(
            tracked_residual[:, indices] * weights[None, :, :, None],
            axis=2,
        )
    return _clip_residual(lifted, maximum_norm)


def _selection_score(
    metrics: dict[str, object],
    baseline: dict[str, object],
) -> float:
    return 0.5 * (
        float(metrics["chamfer_distance_m"])
        / float(baseline["chamfer_distance_m"])
        + float(metrics["track_error_m"]) / float(baseline["track_error_m"])
    )


def fit_action_conditioned_residual_dynamics(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinResidualDynamicsConfig,
) -> dict[str, object]:
    """Select causally, refit on training frames, and forecast residual dynamics."""

    if not 2 < config.fit_end_frame < config.train_end_frame:
        raise ValueError("expected 2 < fit_end_frame < train_end_frame")
    if not config.rank_candidates or any(rank < 1 for rank in config.rank_candidates):
        raise ValueError("rank_candidates must contain positive integers")
    if not config.persistence_candidates or any(
        not 0.0 <= value <= 1.0 for value in config.persistence_candidates
    ):
        raise ValueError("persistence candidates must lie in [0, 1]")
    if not config.ridge_candidates or any(
        value <= 0.0 for value in config.ridge_candidates
    ):
        raise ValueError("ridge candidates must be positive")
    if config.projection_ridge <= 0.0:
        raise ValueError("projection_ridge must be positive")
    if config.maximum_state_multiplier <= 0.0 or config.maximum_residual_m <= 0.0:
        raise ValueError("residual caps must be positive")

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
    features = controller_action_features(controllers)
    maximum_rank = min(max(config.rank_candidates), config.fit_end_frame - 1)
    full_basis = fit_residual_basis(
        residual,
        valid,
        end_frame=config.fit_end_frame,
        maximum_rank=maximum_rank,
    )
    lift_indices, lift_weights = _lift_map(
        baseline[0],
        original_count,
        config.interpolation_neighbors,
    )
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))
    baseline_validation = evaluate_official_phystwin_interval(
        baseline,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=config.fit_end_frame,
        end_frame=config.train_end_frame,
    )
    standardized_fit, _, _ = _standardize_actions(
        features,
        end_frame=config.fit_end_frame,
    )
    candidates: list[dict[str, object]] = []
    best: tuple[float, int, float, float, np.ndarray, np.ndarray] | None = None
    for rank in sorted(set(config.rank_candidates)):
        if rank > len(full_basis):
            continue
        basis = full_basis[:rank]
        latent = _project_residuals(
            residual[: config.fit_end_frame],
            valid[: config.fit_end_frame],
            basis,
            ridge=config.projection_ridge,
        )
        observed_norm = float(np.max(np.linalg.norm(latent, axis=1)))
        norm_cap = max(config.maximum_state_multiplier * observed_norm, 1e-8)
        for persistence in config.persistence_candidates:
            for ridge in config.ridge_candidates:
                dynamics = _fit_latent_dynamics(
                    latent,
                    standardized_fit,
                    end_frame=config.fit_end_frame,
                    persistence=persistence,
                    ridge=ridge,
                )
                predicted_latent = _rollout_latent(
                    latent[-1],
                    standardized_fit,
                    dynamics,
                    start_frame=config.fit_end_frame,
                    end_frame=config.train_end_frame,
                    persistence=persistence,
                    norm_cap=norm_cap,
                )
                tracked = (predicted_latent @ basis).reshape(
                    config.train_end_frame - config.fit_end_frame,
                    original_count,
                    3,
                )
                lifted = _lift_residual(
                    tracked,
                    baseline.shape[1],
                    lift_indices,
                    lift_weights,
                    maximum_norm=config.maximum_residual_m,
                )
                candidate = baseline.copy()
                candidate[config.fit_end_frame : config.train_end_frame] += lifted
                metrics = evaluate_official_phystwin_interval(
                    candidate,
                    observed,
                    visible,
                    gt_track,
                    num_surface_points=num_surface_points,
                    start_frame=config.fit_end_frame,
                    end_frame=config.train_end_frame,
                )
                score = _selection_score(metrics, baseline_validation)
                result = {
                    "rank": rank,
                    "persistence": persistence,
                    "ridge": ridge,
                    "selection_score": score,
                    "official_evaluation": metrics,
                }
                candidates.append(result)
                ranking = (score, rank, -ridge, -persistence)
                if best is None or ranking < best[:4]:
                    best = (score, rank, -ridge, -persistence, basis, dynamics)

    assert best is not None
    selected_score, selected_rank, negative_ridge, negative_persistence, basis, _ = best
    selected_ridge = -negative_ridge
    selected_persistence = -negative_persistence
    improvement = 1.0 - selected_score
    selected = improvement > config.minimum_validation_improvement
    corrected = baseline.copy()
    model_artifact: dict[str, np.ndarray] = {
        "basis": np.empty((0, original_count * 3)),
        "dynamics": np.empty((0, 0)),
        "action_mean": np.empty(0),
        "action_scale": np.empty(0),
        "lift_indices": lift_indices,
        "lift_weights": lift_weights,
    }
    correction = np.zeros_like(baseline[config.train_end_frame :])
    if selected:
        latent = _project_residuals(
            residual[: config.train_end_frame],
            valid[: config.train_end_frame],
            basis,
            ridge=config.projection_ridge,
        )
        standardized_train, action_mean, action_scale = _standardize_actions(
            features,
            end_frame=config.train_end_frame,
        )
        dynamics = _fit_latent_dynamics(
            latent,
            standardized_train,
            end_frame=config.train_end_frame,
            persistence=selected_persistence,
            ridge=selected_ridge,
        )
        observed_norm = float(np.max(np.linalg.norm(latent, axis=1)))
        norm_cap = max(config.maximum_state_multiplier * observed_norm, 1e-8)
        predicted_latent = _rollout_latent(
            latent[-1],
            standardized_train,
            dynamics,
            start_frame=config.train_end_frame,
            end_frame=frame_count,
            persistence=selected_persistence,
            norm_cap=norm_cap,
        )
        tracked = (predicted_latent @ basis).reshape(
            frame_count - config.train_end_frame,
            original_count,
            3,
        )
        correction = _lift_residual(
            tracked,
            baseline.shape[1],
            lift_indices,
            lift_weights,
            maximum_norm=config.maximum_residual_m,
        )
        corrected[config.train_end_frame :] += correction
        model_artifact.update(
            {
                "basis": basis,
                "dynamics": dynamics,
                "action_mean": action_mean,
                "action_scale": action_scale,
            }
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
    corrected_test = evaluate_official_phystwin_interval(
        corrected,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=config.train_end_frame,
        end_frame=frame_count,
    )
    selected_candidate = next(
        item
        for item in candidates
        if item["rank"] == selected_rank
        and item["persistence"] == selected_persistence
        and item["ridge"] == selected_ridge
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(corrected.astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_path = output / "residual_model.npz"
    np.savez_compressed(model_path, **model_artifact)
    correction_norm = np.linalg.norm(correction, axis=2)
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "basis_fit_interval": [1, config.fit_end_frame],
            "selection_interval": [config.fit_end_frame, config.train_end_frame],
            "final_dynamics_fit_interval": [1, config.train_end_frame],
            "future_inputs": "controller actions only after initialization from the final training observation",
            "state_correction": "low-rank tracked residual lifted by fixed reference-space interpolation",
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
            "accepted": selected,
            "baseline_score": 1.0,
            "best_score": selected_score,
            "relative_improvement": improvement,
            "baseline_official_evaluation": baseline_validation,
            "selected_candidate": selected_candidate,
            "candidates": candidates,
        },
        "test": {
            "baseline_official_evaluation": baseline_test,
            "corrected_official_evaluation": corrected_test,
            "selection_score_relative_to_baseline": _selection_score(
                corrected_test,
                baseline_test,
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
