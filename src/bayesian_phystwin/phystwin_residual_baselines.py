"""Fair residual-dynamics baselines for the PhysTwin confirmation protocol."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_residual_dynamics import (
    PhysTwinResidualDynamicsConfig,
    _lift_map,
    _lift_residual,
    _load_pickle,
    _project_residuals,
    _selection_score,
    _sha256,
    _standardize_actions,
    _target_validity,
    _temporally_fill,
    controller_action_features,
    fit_residual_basis,
)


BASELINE_METHODS = ("last_residual", "autonomous", "dmdc")


def _latent_norm_cap(latent: np.ndarray, multiplier: float) -> float:
    return max(multiplier * float(np.max(np.linalg.norm(latent, axis=1))), 1e-8)


def _fit_autonomous(
    latent: np.ndarray,
    *,
    end_frame: int,
    persistence: float,
) -> np.ndarray:
    return np.mean(
        latent[1:end_frame] - persistence * latent[: end_frame - 1],
        axis=0,
    )


def _rollout_autonomous(
    initial: np.ndarray,
    intercept: np.ndarray,
    *,
    frame_count: int,
    persistence: float,
    norm_cap: float,
) -> np.ndarray:
    result = np.empty((frame_count, len(initial)), dtype=float)
    previous = np.asarray(initial, dtype=float).copy()
    for index in range(frame_count):
        current = persistence * previous + intercept
        norm = float(np.linalg.norm(current))
        if norm > norm_cap:
            current *= norm_cap / norm
        result[index] = current
        previous = current
    return result


def _fit_dmdc(
    latent: np.ndarray,
    standardized_actions: np.ndarray,
    *,
    end_frame: int,
    ridge: float,
) -> np.ndarray:
    design = np.concatenate(
        (
            latent[: end_frame - 1],
            standardized_actions[1:end_frame],
            np.ones((end_frame - 1, 1), dtype=float),
        ),
        axis=1,
    )
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ latent[1:end_frame])


def _rollout_dmdc(
    initial: np.ndarray,
    standardized_actions: np.ndarray,
    coefficients: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    norm_cap: float,
) -> np.ndarray:
    result = np.empty((end_frame - start_frame, len(initial)), dtype=float)
    previous = np.asarray(initial, dtype=float).copy()
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        design = np.concatenate((previous, standardized_actions[frame], [1.0]))
        current = design @ coefficients
        norm = float(np.linalg.norm(current))
        if norm > norm_cap:
            current *= norm_cap / norm
        result[output_index] = current
        previous = current
    return result


def fit_residual_dynamics_baselines(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinResidualDynamicsConfig,
    evaluate_future: bool = True,
) -> dict[str, object]:
    """Fit last-value, autonomous, and DMDc residual comparators causally."""

    if not 2 < config.fit_end_frame < config.train_end_frame:
        raise ValueError("expected 2 < fit_end_frame < train_end_frame")
    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controllers = np.asarray(data["controller_points"], dtype=float)
    observation_frame_count, original_count, _ = observed.shape
    frame_count = baseline.shape[0]
    if observation_frame_count < config.train_end_frame:
        raise ValueError("observations do not cover the selection interval")
    if baseline.shape[1] < original_count:
        raise ValueError("baseline trajectory does not cover the observations")
    if evaluate_future and observation_frame_count < frame_count:
        raise ValueError("future evaluation requires complete observations")
    if gt_track.shape[0] < config.train_end_frame:
        raise ValueError("tracks do not cover the selection interval")
    if evaluate_future and gt_track.shape[0] < frame_count:
        raise ValueError("future evaluation requires complete tracks")
    if controllers.shape[0] < frame_count:
        raise ValueError("controller actions do not cover the prediction horizon")
    usable_observation_count = frame_count if evaluate_future else config.train_end_frame
    observed = observed[:usable_observation_count]
    observation_frame_count = len(observed)
    visible = visible[:observation_frame_count]
    motion_valid = motion_valid[: max(observation_frame_count - 1, 0)]
    gt_track = gt_track[:usable_observation_count]
    controllers = controllers[:frame_count]
    valid = _target_validity(visible, motion_valid)
    residual = observed - baseline[:observation_frame_count, :original_count]
    features = controller_action_features(controllers)
    standardized_fit, _, _ = _standardize_actions(
        features, end_frame=config.fit_end_frame
    )
    full_basis = fit_residual_basis(
        residual,
        valid,
        end_frame=config.fit_end_frame,
        maximum_rank=min(max(config.rank_candidates), config.fit_end_frame - 1),
    )
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, config.interpolation_neighbors
    )
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))
    intervals = {"validation": (config.fit_end_frame, config.train_end_frame)}
    if evaluate_future:
        intervals["test"] = (config.train_end_frame, frame_count)
    baseline_metrics = {
        name: evaluate_official_phystwin_interval(
            baseline,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=start,
            end_frame=end,
        )
        for name, (start, end) in intervals.items()
    }

    def evaluate_tracked(
        tracked: np.ndarray, start_frame: int, end_frame: int
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

    candidates: dict[str, list[dict[str, object]]] = {
        method: [] for method in BASELINE_METHODS
    }
    selected: dict[str, tuple[tuple[float, ...], dict[str, object]]] = {}
    validation_count = config.train_end_frame - config.fit_end_frame
    fit_filled = _temporally_fill(residual, valid, config.fit_end_frame)
    last_tracked = np.repeat(fit_filled[-1][None], validation_count, axis=0)
    metrics, _ = evaluate_tracked(
        last_tracked, config.fit_end_frame, config.train_end_frame
    )
    score = _selection_score(metrics, baseline_metrics["validation"])
    endpoint_candidate = {
        "selection_score": score,
        "official_evaluation": metrics,
    }
    candidates["last_residual"].append(endpoint_candidate)
    selected["last_residual"] = ((score,), endpoint_candidate)

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
        norm_cap = _latent_norm_cap(latent, config.maximum_state_multiplier)
        for persistence in config.persistence_candidates:
            intercept = _fit_autonomous(
                latent,
                end_frame=config.fit_end_frame,
                persistence=persistence,
            )
            predicted = _rollout_autonomous(
                latent[-1],
                intercept,
                frame_count=validation_count,
                persistence=persistence,
                norm_cap=norm_cap,
            )
            tracked = (predicted @ basis).reshape(validation_count, original_count, 3)
            metrics, _ = evaluate_tracked(
                tracked, config.fit_end_frame, config.train_end_frame
            )
            score = _selection_score(metrics, baseline_metrics["validation"])
            candidate = {
                "rank": rank,
                "persistence": persistence,
                "selection_score": score,
                "official_evaluation": metrics,
            }
            candidates["autonomous"].append(candidate)
            ranking = (score, rank, -persistence)
            if "autonomous" not in selected or ranking < selected["autonomous"][0]:
                selected["autonomous"] = (ranking, candidate)

        for ridge in config.ridge_candidates:
            coefficients = _fit_dmdc(
                latent,
                standardized_fit,
                end_frame=config.fit_end_frame,
                ridge=ridge,
            )
            predicted = _rollout_dmdc(
                latent[-1],
                standardized_fit,
                coefficients,
                start_frame=config.fit_end_frame,
                end_frame=config.train_end_frame,
                norm_cap=norm_cap,
            )
            tracked = (predicted @ basis).reshape(validation_count, original_count, 3)
            metrics, _ = evaluate_tracked(
                tracked, config.fit_end_frame, config.train_end_frame
            )
            score = _selection_score(metrics, baseline_metrics["validation"])
            candidate = {
                "rank": rank,
                "ridge": ridge,
                "selection_score": score,
                "official_evaluation": metrics,
            }
            candidates["dmdc"].append(candidate)
            ranking = (score, rank, -ridge)
            if "dmdc" not in selected or ranking < selected["dmdc"][0]:
                selected["dmdc"] = (ranking, candidate)

    output = Path(output_dir)
    method_summaries: dict[str, object] = {}
    future_count = frame_count - config.train_end_frame
    for method in BASELINE_METHODS:
        selected_candidate = selected[method][1]
        validation_improvement = 1.0 - float(selected_candidate["selection_score"])
        accepted = validation_improvement > config.minimum_validation_improvement
        tracked_future = np.zeros((future_count, original_count, 3), dtype=float)
        artifact: dict[str, np.ndarray] = {}
        if accepted and method == "last_residual":
            train_filled = _temporally_fill(residual, valid, config.train_end_frame)
            tracked_future = np.repeat(train_filled[-1][None], future_count, axis=0)
            artifact["tracked_endpoint"] = train_filled[-1]
        elif accepted:
            rank = int(selected_candidate["rank"])
            basis = full_basis[:rank]
            latent = _project_residuals(
                residual[: config.train_end_frame],
                valid[: config.train_end_frame],
                basis,
                ridge=config.projection_ridge,
            )
            norm_cap = _latent_norm_cap(latent, config.maximum_state_multiplier)
            artifact["basis"] = basis
            if method == "autonomous":
                persistence = float(selected_candidate["persistence"])
                intercept = _fit_autonomous(
                    latent,
                    end_frame=config.train_end_frame,
                    persistence=persistence,
                )
                predicted = _rollout_autonomous(
                    latent[-1],
                    intercept,
                    frame_count=future_count,
                    persistence=persistence,
                    norm_cap=norm_cap,
                )
                artifact["intercept"] = intercept
            else:
                standardized_train, action_mean, action_scale = _standardize_actions(
                    features, end_frame=config.train_end_frame
                )
                coefficients = _fit_dmdc(
                    latent,
                    standardized_train,
                    end_frame=config.train_end_frame,
                    ridge=float(selected_candidate["ridge"]),
                )
                predicted = _rollout_dmdc(
                    latent[-1],
                    standardized_train,
                    coefficients,
                    start_frame=config.train_end_frame,
                    end_frame=frame_count,
                    norm_cap=norm_cap,
                )
                artifact.update(
                    {
                        "coefficients": coefficients,
                        "action_mean": action_mean,
                        "action_scale": action_scale,
                    }
                )
            tracked_future = (predicted @ basis).reshape(
                future_count, original_count, 3
            )

        correction = _lift_residual(
            tracked_future,
            baseline.shape[1],
            lift_indices,
            lift_weights,
            maximum_norm=config.maximum_residual_m,
        )
        test_metrics = None
        if evaluate_future:
            test_metrics, _ = evaluate_tracked(
                tracked_future, config.train_end_frame, frame_count
            )
        corrected = baseline.copy()
        corrected[config.train_end_frame :] += correction
        method_output = output / method
        method_output.mkdir(parents=True, exist_ok=True)
        trajectory_path = method_output / "trajectory.pkl"
        with trajectory_path.open("wb") as handle:
            pickle.dump(
                corrected.astype(np.float32),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        model_path = method_output / "model.npz"
        np.savez_compressed(model_path, **artifact)
        correction_norm = np.linalg.norm(correction, axis=2)
        method_summary = {
            "selection": {
                "accepted": accepted,
                "relative_improvement": validation_improvement,
                "selected_candidate": selected_candidate,
                "candidates": candidates[method],
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
        if evaluate_future:
            assert test_metrics is not None
            method_summary["test"] = {
                "baseline_official_evaluation": baseline_metrics["test"],
                "corrected_official_evaluation": test_metrics,
                "selection_score_relative_to_baseline": _selection_score(
                    test_metrics, baseline_metrics["test"]
                ),
            }
        method_summaries[method] = method_summary

    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "basis_fit_interval": [1, config.fit_end_frame],
            "selection_interval": [config.fit_end_frame, config.train_end_frame],
            "final_fit_interval": [1, config.train_end_frame],
            "future_inputs": {
                "last_residual": "none after final training residual",
                "autonomous": "none after final training latent state",
                "dmdc": "controller actions after final training latent state",
            },
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
        "methods": method_summaries,
        "future_metrics_opened": evaluate_future,
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path.resolve())
    return summary
