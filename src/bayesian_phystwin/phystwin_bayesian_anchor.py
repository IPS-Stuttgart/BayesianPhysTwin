"""Robust Bayesian endpoint anchoring for PhysTwin simulator discrepancy."""

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
    _selection_score,
    _sha256,
    _target_validity,
)


@dataclass(frozen=True)
class BayesianResidualAnchorConfig:
    """Causal split and robust random-walk hyperparameter grid."""

    fit_end_frame: int
    train_end_frame: int
    process_std_candidates_m: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.0025, 0.005)
    observation_std_candidates_m: tuple[float, ...] = (0.001, 0.0025, 0.005)
    initial_std_m: float = 0.01
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0
    interpolation_neighbors: int = 4
    maximum_residual_m: float = 0.01
    minimum_validation_improvement: float = 0.0


@dataclass(frozen=True)
class RobustEndpointPosterior:
    """Per-track posterior at the end of an observation interval."""

    mean: np.ndarray
    variance: np.ndarray
    final_inlier_probability: np.ndarray
    update_count: np.ndarray


def robust_random_walk_endpoint(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    process_variance: float,
    observation_variance: float | np.ndarray,
    initial_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
    prior_reliability: np.ndarray | None = None,
) -> RobustEndpointPosterior:
    """Filter a 3D random-walk discrepancy with a shared robust inlier state.

    ``observation_variance`` may be a scalar or a causal ``(T, N)`` array in
    square metres. ``prior_reliability`` is evaluated before the innovation and
    only inflates that metric variance. The innovation is handled exactly once
    by the inlier/outlier mixture below.
    """

    values = np.asarray(residual, dtype=float)
    validity = np.asarray(valid, dtype=bool)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("residual must have shape (T, N, 3)")
    if validity.shape != values.shape[:2]:
        raise ValueError("valid must match the residual frame and track dimensions")
    if not 0 < end_frame <= len(values):
        raise ValueError("end_frame must lie inside the residual sequence")
    if process_variance < 0.0:
        raise ValueError("process variance must be nonnegative")
    if initial_variance <= 0.0:
        raise ValueError("initial_variance must be positive")
    if not 0.0 < inlier_prior < 1.0:
        raise ValueError("inlier_prior must lie in (0, 1)")
    if outlier_variance_multiplier <= 1.0:
        raise ValueError("outlier_variance_multiplier must exceed one")

    supplied_variance = np.asarray(observation_variance, dtype=float)
    if supplied_variance.ndim == 0:
        if not np.isfinite(supplied_variance) or supplied_variance <= 0.0:
            raise ValueError("observation variance must be finite and positive")
        observation_variance_by_frame = None
        scalar_observation_variance = float(supplied_variance)
    else:
        if supplied_variance.shape == values.shape:
            supplied_variance = np.mean(supplied_variance, axis=2)
        if supplied_variance.shape != values.shape[:2]:
            raise ValueError(
                "array observation variance must have shape (T, N) or (T, N, 3)"
            )
        if not np.all(np.isfinite(supplied_variance)) or np.any(
            supplied_variance <= 0.0
        ):
            raise ValueError("observation variance must be finite and positive")
        observation_variance_by_frame = supplied_variance
        scalar_observation_variance = 0.0

    if prior_reliability is None:
        reliability = None
    else:
        reliability = np.asarray(prior_reliability, dtype=float)
        if reliability.shape != values.shape[:2]:
            raise ValueError("prior_reliability must have shape (T, N)")
        if not np.all(np.isfinite(reliability)) or np.any(
            (reliability < 0.0) | (reliability > 1.0)
        ):
            raise ValueError("prior_reliability must lie in [0, 1]")

    track_count = values.shape[1]
    mean = np.zeros((track_count, 3), dtype=float)
    variance = np.full(track_count, initial_variance, dtype=float)
    final_probability = np.zeros(track_count, dtype=float)
    update_count = np.zeros(track_count, dtype=np.int64)
    log_prior = np.log(inlier_prior)
    log_outlier_prior = np.log1p(-inlier_prior)
    for frame in range(end_frame):
        predicted_variance = variance + process_variance
        mask = validity[frame].copy()
        if reliability is not None:
            mask &= reliability[frame] > 0.0
        variance = predicted_variance
        if not np.any(mask):
            continue
        innovation = values[frame, mask] - mean[mask]
        predicted = predicted_variance[mask]
        frame_observation_variance = (
            np.full(np.sum(mask), scalar_observation_variance, dtype=float)
            if observation_variance_by_frame is None
            else observation_variance_by_frame[frame, mask].copy()
        )
        if reliability is not None:
            frame_observation_variance /= reliability[frame, mask]
        inlier_innovation_variance = predicted + frame_observation_variance
        outlier_innovation_variance = (
            predicted + frame_observation_variance * outlier_variance_multiplier
        )
        squared_norm = np.sum(np.square(innovation), axis=1)
        log_inlier = log_prior - 0.5 * (
            3.0 * np.log(2.0 * np.pi * inlier_innovation_variance)
            + squared_norm / inlier_innovation_variance
        )
        log_outlier = log_outlier_prior - 0.5 * (
            3.0 * np.log(2.0 * np.pi * outlier_innovation_variance)
            + squared_norm / outlier_innovation_variance
        )
        probability = np.exp(log_inlier - np.logaddexp(log_inlier, log_outlier))
        inlier_gain = predicted / inlier_innovation_variance
        outlier_gain = predicted / outlier_innovation_variance
        inlier_mean = mean[mask] + inlier_gain[:, None] * innovation
        outlier_mean = mean[mask] + outlier_gain[:, None] * innovation
        updated_mean = (
            probability[:, None] * inlier_mean
            + (1.0 - probability)[:, None] * outlier_mean
        )
        inlier_variance = (1.0 - inlier_gain) * predicted
        outlier_variance = (1.0 - outlier_gain) * predicted
        inlier_spread = np.mean(np.square(inlier_mean - updated_mean), axis=1)
        outlier_spread = np.mean(np.square(outlier_mean - updated_mean), axis=1)
        updated_variance = probability * (inlier_variance + inlier_spread) + (
            1.0 - probability
        ) * (outlier_variance + outlier_spread)
        mean[mask] = updated_mean
        variance[mask] = np.maximum(updated_variance, 0.0)
        final_probability[mask] = probability
        update_count[mask] += 1
    return RobustEndpointPosterior(
        mean=mean,
        variance=variance,
        final_inlier_probability=final_probability,
        update_count=update_count,
    )


def fit_bayesian_residual_anchor(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    config: BayesianResidualAnchorConfig,
    evaluate_future: bool = True,
) -> dict[str, object]:
    """Select a robust filter on validation and hold its training posterior mean."""

    if not 2 < config.fit_end_frame < config.train_end_frame:
        raise ValueError("expected 2 < fit_end_frame < train_end_frame")
    if not config.process_std_candidates_m or any(
        value < 0.0 for value in config.process_std_candidates_m
    ):
        raise ValueError("process standard deviations must be nonnegative")
    if not config.observation_std_candidates_m or any(
        value <= 0.0 for value in config.observation_std_candidates_m
    ):
        raise ValueError("observation standard deviations must be positive")
    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
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
    usable_observation_count = (
        frame_count if evaluate_future else config.train_end_frame
    )
    observed = observed[:usable_observation_count]
    observation_frame_count = len(observed)
    visible = visible[:observation_frame_count]
    motion_valid = motion_valid[: max(observation_frame_count - 1, 0)]
    gt_track = gt_track[:usable_observation_count]
    valid = _target_validity(visible, motion_valid)
    residual = observed - baseline[:observation_frame_count, :original_count]
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, config.interpolation_neighbors
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

    def corrected_metrics(
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

    validation_count = config.train_end_frame - config.fit_end_frame
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, float, float], dict[str, object]] | None = None
    for process_std in sorted(set(config.process_std_candidates_m)):
        for observation_std in sorted(set(config.observation_std_candidates_m)):
            posterior = robust_random_walk_endpoint(
                residual,
                valid,
                end_frame=config.fit_end_frame,
                process_variance=process_std**2,
                observation_variance=observation_std**2,
                initial_variance=config.initial_std_m**2,
                inlier_prior=config.inlier_prior,
                outlier_variance_multiplier=config.outlier_variance_multiplier,
            )
            tracked = np.repeat(posterior.mean[None], validation_count, axis=0)
            metrics, _ = corrected_metrics(
                tracked, config.fit_end_frame, config.train_end_frame
            )
            score = _selection_score(metrics, baseline_validation)
            candidate = {
                "process_std_m": process_std,
                "observation_std_m": observation_std,
                "selection_score": score,
                "official_evaluation": metrics,
            }
            candidates.append(candidate)
            ranking = (score, process_std, -observation_std)
            if best is None or ranking < best[0]:
                best = (ranking, candidate)
    assert best is not None
    selected_candidate = best[1]
    validation_improvement = 1.0 - float(selected_candidate["selection_score"])
    accepted = validation_improvement > config.minimum_validation_improvement
    process_std = float(selected_candidate["process_std_m"])
    observation_std = float(selected_candidate["observation_std_m"])
    posterior = robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=config.train_end_frame,
        process_variance=process_std**2,
        observation_variance=observation_std**2,
        initial_variance=config.initial_std_m**2,
        inlier_prior=config.inlier_prior,
        outlier_variance_multiplier=config.outlier_variance_multiplier,
    )
    future_count = frame_count - config.train_end_frame
    tracked_future = np.zeros((future_count, original_count, 3), dtype=float)
    if accepted:
        tracked_future[:] = posterior.mean
    correction = _lift_residual(
        tracked_future,
        baseline.shape[1],
        lift_indices,
        lift_weights,
        maximum_norm=config.maximum_residual_m,
    )
    corrected_test = None
    baseline_test = None
    if evaluate_future:
        corrected_test, _ = corrected_metrics(
            tracked_future, config.train_end_frame, frame_count
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
    corrected = baseline.copy()
    corrected[config.train_end_frame :] += correction
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(
            corrected.astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL
        )
    posterior_path = output / "posterior.npz"
    np.savez_compressed(
        posterior_path,
        mean=posterior.mean,
        variance=posterior.variance,
        final_inlier_probability=posterior.final_inlier_probability,
        update_count=posterior.update_count,
        lift_indices=lift_indices,
        lift_weights=lift_weights,
    )
    correction_norm = np.linalg.norm(correction, axis=2)
    posterior_std = np.sqrt(posterior.variance)
    final_predictive_std = np.sqrt(posterior.variance + future_count * process_std**2)
    updated = posterior.update_count > 0
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "hyperparameter_fit_interval": [0, config.fit_end_frame],
            "selection_interval": [config.fit_end_frame, config.train_end_frame],
            "posterior_fit_interval": [0, config.train_end_frame],
            "future_mean": "final robust random-walk posterior held constant",
            "future_uncertainty": "random-walk posterior variance propagated without observations",
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
            "accepted": accepted,
            "relative_improvement": validation_improvement,
            "selected_candidate": selected_candidate,
            "candidates": candidates,
        },
        "future_metrics_opened": evaluate_future,
        "correction": {
            "rms_m": float(np.sqrt(np.mean(np.square(correction_norm)))),
            "maximum_m": float(np.max(correction_norm, initial=0.0)),
            "saturated_fraction": float(
                np.mean(correction_norm >= 0.999 * config.maximum_residual_m)
            ),
        },
        "posterior": {
            "updated_track_count": int(np.sum(updated)),
            "median_std_m": float(np.median(posterior_std[updated])),
            "upper_95_std_m": float(np.quantile(posterior_std[updated], 0.95)),
            "median_final_inlier_probability": float(
                np.median(posterior.final_inlier_probability[updated])
            ),
            "median_final_future_predictive_std_m": float(
                np.median(final_predictive_std[updated])
            ),
        },
        "outputs": {
            "trajectory": str(trajectory_path.resolve()),
            "posterior": str(posterior_path.resolve()),
        },
    }
    if evaluate_future:
        assert corrected_test is not None and baseline_test is not None
        summary["test"] = {
            "baseline_official_evaluation": baseline_test,
            "corrected_official_evaluation": corrected_test,
            "selection_score_relative_to_baseline": _selection_score(
                corrected_test, baseline_test
            ),
        }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["summary"] = str(summary_path.resolve())
    return summary
