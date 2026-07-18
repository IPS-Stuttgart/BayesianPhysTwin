"""Prefix-calibrated PGRD residual velocities for PhysTwin trajectories."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_pgrd_adapter import (
    MetricNormalizer,
    OfficialPGRDResidualPredictor,
    PGRDResidualPredictor,
    _cap_vectors,
    _inverse_distance_map,
    _metric_ratios,
    _relative_score,
    compose_dense_endpoint_with_sampled_dynamics,
    deterministic_farthest_point_sample,
)
from .phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


@dataclass(frozen=True)
class PhysTwinPGRDCalibrationConfig:
    """Small readout calibration and causal selection contract."""

    fit_end_frame: int
    train_end_frame: int
    normalized_extent_candidates: tuple[float, ...] = (0.25, 0.5, 0.75)
    yaw_candidates_degrees: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    ridge_candidates: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2)
    number_of_points: int = 512
    history_length: int = 2
    temporal_window: int = 5
    simulation_dt: float = 0.1
    model_frame_stride: int = 3
    interpolation_neighbors: int = 4
    maximum_readout_gain: float = 5.0
    maximum_residual_m: float = 0.01
    minimum_dynamic_improvement: float = 0.01
    maximum_metric_ratio: float = 1.02


def fit_calibrated_velocity_readout(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    maximum_gain: float,
) -> np.ndarray:
    """Fit and spectrally bound a three-dimensional PGRD output readout."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != x.shape or len(x) < 3:
        raise ValueError("features and targets must share shape (N, 3), N >= 3")
    if ridge <= 0.0 or maximum_gain <= 0.0:
        raise ValueError("ridge and maximum_gain must be positive")
    gram = x.T @ x / len(x) + ridge * np.eye(3)
    cross = x.T @ y / len(x)
    readout = np.linalg.solve(gram, cross)
    left, singular, right = np.linalg.svd(readout, full_matrices=False)
    singular = np.minimum(singular, maximum_gain)
    return (left * singular) @ right


def _cadence_targets(
    end_frame: int,
    *,
    history_length: int,
    frame_stride: int,
) -> list[int]:
    minimum = (history_length + 1) * frame_stride
    targets = list(range(end_frame - 1, minimum - 1, -frame_stride))
    targets.reverse()
    return targets


def collect_teacher_forced_pgrd_pairs(
    baseline_m: np.ndarray,
    observed_prefix_m: np.ndarray,
    valid_prefix: np.ndarray,
    sample_indices: np.ndarray,
    predictor: PGRDResidualPredictor,
    normalizer: MetricNormalizer,
    *,
    history_length: int,
    simulation_dt: float,
    model_frame_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect PGRD outputs and observed residual increments from prefix data."""

    baseline = np.asarray(baseline_m, dtype=float)[:, sample_indices]
    observed = np.asarray(observed_prefix_m, dtype=float)[:, sample_indices]
    valid = np.asarray(valid_prefix, dtype=bool)[:, sample_indices]
    end_frame = len(observed)
    if baseline.shape[0] < end_frame or observed.shape[1:] != baseline.shape[1:]:
        raise ValueError("teacher-forced trajectories are not aligned")
    targets = _cadence_targets(
        end_frame,
        history_length=history_length,
        frame_stride=model_frame_stride,
    )
    if not targets:
        raise ValueError("prefix is too short for PGRD calibration")
    predictor.reset()
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    residual = observed - baseline[:end_frame]
    for target_frame in targets:
        current_frame = target_frame - model_frame_stride
        history_frames = [
            current_frame - offset * model_frame_stride
            for offset in reversed(range(history_length))
        ]
        position_history = np.moveaxis(observed[history_frames], 0, 1)
        velocity_history = []
        for frame in history_frames:
            velocity_history.append(
                normalizer.velocities_to_model(
                    (observed[frame] - observed[frame - model_frame_stride])
                    / simulation_dt
                )
            )
        residual_velocity = predictor.predict(
            normalizer.positions_to_model(observed[current_frame]),
            normalizer.velocities_to_model(
                (observed[current_frame] - observed[current_frame - model_frame_stride])
                / simulation_dt
            ),
            normalizer.positions_to_model(position_history),
            np.stack(velocity_history, axis=1),
            normalizer.positions_to_model(baseline[target_frame]),
            normalizer.velocities_to_model(
                (baseline[target_frame] - baseline[current_frame]) / simulation_dt
            ),
        )
        mask = valid[current_frame] & valid[target_frame]
        if np.any(mask):
            feature_blocks.append(residual_velocity[mask] * simulation_dt)
            target_delta_m = residual[target_frame, mask] - residual[current_frame, mask]
            target_blocks.append(
                normalizer.velocities_to_model(target_delta_m / simulation_dt)
                * simulation_dt
            )
    if not feature_blocks:
        raise ValueError("prefix has no valid calibration pairs")
    return np.concatenate(feature_blocks), np.concatenate(target_blocks)


def rollout_calibrated_pgrd_correction(
    baseline_m: np.ndarray,
    observed_prefix_m: np.ndarray,
    valid_prefix: np.ndarray,
    sample_indices: np.ndarray,
    predictor: PGRDResidualPredictor,
    normalizer: MetricNormalizer,
    readout: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    history_length: int,
    simulation_dt: float,
    model_frame_stride: int,
    maximum_residual_m: float,
) -> np.ndarray:
    """Warm on the prefix and recursively integrate calibrated residual velocity."""

    baseline = np.asarray(baseline_m, dtype=float)[:, sample_indices]
    observed = np.asarray(observed_prefix_m, dtype=float)[:, sample_indices]
    if len(observed) != start_frame or end_frame > len(baseline):
        raise ValueError("prefix and rollout interval are not aligned")
    # Replaying prefix pairs fills the released transformer's causal feature window.
    collect_teacher_forced_pgrd_pairs(
        baseline_m,
        observed_prefix_m,
        valid_prefix,
        sample_indices,
        predictor,
        normalizer,
        history_length=history_length,
        simulation_dt=simulation_dt,
        model_frame_stride=model_frame_stride,
    )
    correction = observed[-1] - baseline[start_frame - 1]
    history_frames = [
        start_frame - 1 - offset * model_frame_stride
        for offset in reversed(range(history_length))
    ]
    states = [observed[frame].copy() for frame in history_frames]
    velocities = [np.zeros_like(states[0])]
    for index in range(1, len(states)):
        velocities.append(
            normalizer.velocities_to_model(
                (states[index] - states[index - 1]) / simulation_dt
            )
        )
    result = np.empty((end_frame - start_frame, len(sample_indices), 3), dtype=float)
    previous_frame = start_frame - 1
    while previous_frame < end_frame - 1:
        target_frame = min(previous_frame + model_frame_stride, len(baseline) - 1)
        residual_velocity = predictor.predict(
            normalizer.positions_to_model(states[-1]),
            velocities[-1],
            normalizer.positions_to_model(np.stack(states[-history_length:], axis=1)),
            np.stack(velocities[-history_length:], axis=1),
            normalizer.positions_to_model(baseline[target_frame]),
            normalizer.velocities_to_model(
                (baseline[target_frame] - baseline[previous_frame]) / simulation_dt
            ),
        )
        delta_model = (residual_velocity * simulation_dt) @ readout
        target_correction = correction + normalizer.displacements_to_metric(delta_model)
        target_correction = _cap_vectors(target_correction, maximum_residual_m)
        output_stop = min(target_frame, end_frame - 1)
        denominator = target_frame - previous_frame
        for frame in range(previous_frame + 1, output_stop + 1):
            fraction = (frame - previous_frame) / denominator
            result[frame - start_frame] = (
                (1.0 - fraction) * correction + fraction * target_correction
            )
        current = baseline[target_frame] + target_correction
        velocities.append(
            normalizer.velocities_to_model((current - states[-1]) / simulation_dt)
        )
        states.append(current)
        previous_frame = target_frame
        correction = target_correction
    return result


def fit_prefix_calibrated_pgrd(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinPGRDCalibrationConfig,
    pgrd_checkout: str | Path | None = None,
    pgrd_checkpoint: str | Path | None = None,
    device: str = "cuda",
    predictor: PGRDResidualPredictor | None = None,
) -> dict[str, object]:
    """Calibrate a tiny readout on fit frames and gate it on untouched validation."""

    if not (config.history_length + 2) * config.model_frame_stride < config.fit_end_frame:
        raise ValueError("fit interval is too short for causal PGRD calibration")
    if not config.fit_end_frame < config.train_end_frame:
        raise ValueError("fit_end_frame must precede train_end_frame")
    if any(value <= 0.0 for value in config.ridge_candidates):
        raise ValueError("ridge candidates must be positive")
    if predictor is None:
        if pgrd_checkout is None or pgrd_checkpoint is None:
            raise ValueError("official inference requires a checkout and checkpoint")
        predictor = OfficialPGRDResidualPredictor(
            pgrd_checkout,
            pgrd_checkpoint,
            device=device,
            history_length=config.history_length,
            temporal_window=config.temporal_window,
        )
        provenance: dict[str, object] = predictor.provenance
    else:
        provenance = {"backend": "injected test predictor"}

    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    valid = _target_validity(visible, np.asarray(data["object_motions_valid"], dtype=bool))
    frame_count, original_count, _ = observed.shape
    if not config.train_end_frame < frame_count:
        raise ValueError("train_end_frame must leave a future interval")
    baseline = baseline[:frame_count]
    if baseline.shape[1] < original_count or config.number_of_points > original_count:
        raise ValueError("baseline or sample count does not cover observed points")
    residual = observed - baseline[:, :original_count]
    sample_indices = deterministic_farthest_point_sample(
        baseline[0, :original_count], config.number_of_points
    )
    sample_lift_indices, sample_lift_weights = _inverse_distance_map(
        baseline[0, sample_indices],
        baseline[0, :original_count],
        min(config.interpolation_neighbors, config.number_of_points),
    )
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, config.interpolation_neighbors
    )
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))

    def evaluate_correction(
        original_correction: np.ndarray, start: int, end: int
    ) -> tuple[dict[str, object], np.ndarray]:
        lifted = _lift_residual(
            original_correction,
            baseline.shape[1],
            lift_indices,
            lift_weights,
            maximum_norm=config.maximum_residual_m,
        )
        trajectory = baseline.copy()
        trajectory[start:end] += lifted
        return (
            evaluate_official_phystwin_interval(
                trajectory,
                observed,
                visible,
                gt_track,
                num_surface_points=num_surface_points,
                start_frame=start,
                end_frame=end,
            ),
            lifted,
        )

    fit_filled = _temporally_fill(residual, valid, config.fit_end_frame)
    observed_fit = baseline[: config.fit_end_frame, :original_count] + fit_filled
    validation_count = config.train_end_frame - config.fit_end_frame
    persistence_validation_original = np.repeat(
        fit_filled[-1][None], validation_count, axis=0
    )
    persistence_validation, _ = evaluate_correction(
        persistence_validation_original, config.fit_end_frame, config.train_end_frame
    )
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, float, float, float], dict[str, object], np.ndarray] | None = None
    for extent in sorted(set(config.normalized_extent_candidates)):
        for yaw_degrees in sorted(set(config.yaw_candidates_degrees)):
            normalizer = MetricNormalizer.fit(
                baseline[0, sample_indices], extent, yaw_degrees=yaw_degrees
            )
            features, targets = collect_teacher_forced_pgrd_pairs(
                baseline[:, :original_count],
                observed_fit,
                valid[: config.fit_end_frame],
                sample_indices,
                predictor,
                normalizer,
                history_length=config.history_length,
                simulation_dt=config.simulation_dt,
                model_frame_stride=config.model_frame_stride,
            )
            for ridge in sorted(set(config.ridge_candidates)):
                readout = fit_calibrated_velocity_readout(
                    features,
                    targets,
                    ridge=ridge,
                    maximum_gain=config.maximum_readout_gain,
                )
                sampled = rollout_calibrated_pgrd_correction(
                    baseline[:, :original_count],
                    observed_fit,
                    valid[: config.fit_end_frame],
                    sample_indices,
                    predictor,
                    normalizer,
                    readout,
                    start_frame=config.fit_end_frame,
                    end_frame=config.train_end_frame,
                    history_length=config.history_length,
                    simulation_dt=config.simulation_dt,
                    model_frame_stride=config.model_frame_stride,
                    maximum_residual_m=config.maximum_residual_m,
                )
                original = compose_dense_endpoint_with_sampled_dynamics(
                    sampled,
                    fit_filled[-1],
                    sample_indices,
                    sample_lift_indices,
                    sample_lift_weights,
                )
                metrics, _ = evaluate_correction(
                    original, config.fit_end_frame, config.train_end_frame
                )
                ratios = _metric_ratios(metrics, persistence_validation)
                candidate = {
                    "normalized_extent": extent,
                    "yaw_degrees": yaw_degrees,
                    "ridge": ridge,
                    "calibration_pair_count": len(features),
                    "readout_singular_values": np.linalg.svd(
                        readout, compute_uv=False
                    ).tolist(),
                    "selection_score_relative_to_persistence": _relative_score(
                        metrics, persistence_validation
                    ),
                    "metric_ratios_relative_to_persistence": ratios,
                    "official_evaluation": metrics,
                }
                candidates.append(candidate)
                ranking = (
                    float(candidate["selection_score_relative_to_persistence"]),
                    ridge,
                    extent,
                    yaw_degrees,
                )
                if best is None or ranking < best[0]:
                    best = (ranking, candidate, readout)
    assert best is not None
    selected, selected_readout = best[1], best[2]
    ratios = selected["metric_ratios_relative_to_persistence"]
    accepted = (
        float(selected["selection_score_relative_to_persistence"])
        <= 1.0 - config.minimum_dynamic_improvement
        and max(float(value) for value in ratios.values()) <= config.maximum_metric_ratio
    )

    train_filled = _temporally_fill(residual, valid, config.train_end_frame)
    future_count = frame_count - config.train_end_frame
    if accepted:
        normalizer = MetricNormalizer.fit(
            baseline[0, sample_indices],
            float(selected["normalized_extent"]),
            yaw_degrees=float(selected["yaw_degrees"]),
        )
        observed_train = baseline[: config.train_end_frame, :original_count] + train_filled
        features, targets = collect_teacher_forced_pgrd_pairs(
            baseline[:, :original_count],
            observed_train,
            valid[: config.train_end_frame],
            sample_indices,
            predictor,
            normalizer,
            history_length=config.history_length,
            simulation_dt=config.simulation_dt,
            model_frame_stride=config.model_frame_stride,
        )
        selected_readout = fit_calibrated_velocity_readout(
            features,
            targets,
            ridge=float(selected["ridge"]),
            maximum_gain=config.maximum_readout_gain,
        )
        sampled_future = rollout_calibrated_pgrd_correction(
            baseline[:, :original_count],
            observed_train,
            valid[: config.train_end_frame],
            sample_indices,
            predictor,
            normalizer,
            selected_readout,
            start_frame=config.train_end_frame,
            end_frame=frame_count,
            history_length=config.history_length,
            simulation_dt=config.simulation_dt,
            model_frame_stride=config.model_frame_stride,
            maximum_residual_m=config.maximum_residual_m,
        )
        original_future = compose_dense_endpoint_with_sampled_dynamics(
            sampled_future,
            train_filled[-1],
            sample_indices,
            sample_lift_indices,
            sample_lift_weights,
        )
        method = "prefix_calibrated_pgrd"
    else:
        original_future = np.repeat(train_filled[-1][None], future_count, axis=0)
        method = "persistence"
    corrected_test, lifted_future = evaluate_correction(
        original_future, config.train_end_frame, frame_count
    )
    persistence_test, _ = evaluate_correction(
        np.repeat(train_filled[-1][None], future_count, axis=0),
        config.train_end_frame,
        frame_count,
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
    trajectory = baseline.copy()
    trajectory[config.train_end_frame :] += lifted_future
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory.astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_path = output / "calibrated_pgrd_model.npz"
    np.savez_compressed(
        model_path,
        readout=selected_readout,
        sample_indices=sample_indices,
        sample_lift_indices=sample_lift_indices,
        sample_lift_weights=sample_lift_weights,
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "fit_interval": [0, config.fit_end_frame],
            "selection_interval": [config.fit_end_frame, config.train_end_frame],
            "future_observations_used": False,
            "learned_parameters": "spectrally bounded 3x3 PGRD velocity readout",
            "fallback": "exact temporally filled endpoint persistence",
            "released_case_status": "development-only; cannot confirm transfer",
        },
        "pgrd": provenance,
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for name, path in {
                "final_data": final_data_path,
                "baseline_trajectory": baseline_trajectory_path,
                "gt_track_3d": gt_track_path,
            }.items()
        },
        "selection": {
            "selected_method": method,
            "dynamic_accepted": accepted,
            "persistence_official_evaluation": persistence_validation,
            "selected_candidate": selected,
            "candidates": candidates,
        },
        "test": {
            "future_metrics_opened": accepted,
            "baseline_official_evaluation": baseline_test,
            "persistence_official_evaluation": persistence_test,
            "corrected_official_evaluation": corrected_test,
            "selection_score_relative_to_persistence": _relative_score(
                corrected_test, persistence_test
            ),
            "metric_ratios_relative_to_persistence": _metric_ratios(
                corrected_test, persistence_test
            ),
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
