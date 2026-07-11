"""Leave-one-interaction-out hierarchical scaling of PhysTwin residual dynamics."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_comparison import (
    compare_phystwin_manifest,
    official_metrics_by_frame,
    paired_block_bootstrap,
)
from .phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from .phystwin_horizon_analysis import HORIZON_LABELS, METRICS, split_future_horizon
from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_official_evaluation import _nearest_distances
from .phystwin_residual_dynamics import (
    _fit_latent_dynamics,
    _lift_map,
    _load_pickle,
    _project_residuals,
    _rollout_latent,
    _selection_score,
    _sha256,
    _standardize_actions,
    _target_validity,
    controller_action_features,
    fit_residual_basis,
)


DynamicsKey = tuple[int, float, float]


@dataclass(frozen=True)
class HierarchicalResidualShrinkageProtocol:
    """Frozen grids for three-case outer leave-one-interaction-out selection."""

    fit_fraction: float = 0.75
    rank_candidates: tuple[int, ...] = (1, 2, 4, 8)
    persistence_candidates: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.0)
    ridge_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
    observation_std_candidates_m: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
    population_mean_candidates_m: tuple[float, ...] = (
        0.0,
        0.005,
        0.01,
        0.015,
        0.02,
        0.03,
    )
    population_std_candidates_m: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
    scale_grid_maximum_m: float = 0.04
    scale_grid_step_m: float = 0.0005
    projection_ridge: float = 1e-6
    interpolation_neighbors: int = 4
    maximum_state_multiplier: float = 1.5
    bootstrap_samples: int = 10000
    bootstrap_block_length: int = 5
    bootstrap_seed: int = 20260711
    development_cases: tuple[str, ...] = DEVELOPMENT_CASES


@dataclass(frozen=True)
class ScaleLikelihoodStatistics:
    """Frame-balanced error curve for a smoothly shrunk residual field."""

    squared_error_by_scale: np.ndarray
    frame_count: int
    raw_rms_m: float


@dataclass(frozen=True)
class SharedShrinkageSelection:
    """Shared settings selected without the held-out interaction."""

    rank: int
    persistence: float
    ridge: float
    observation_std_m: float
    population_mean_m: float
    population_std_m: float
    log_evidence: float


@dataclass
class _CaseData:
    name: str
    case_dir: Path
    observed: np.ndarray
    visible: np.ndarray
    valid: np.ndarray
    baseline: np.ndarray
    gt_track: np.ndarray
    features: np.ndarray
    residual: np.ndarray
    manual_residual: np.ndarray
    manual_valid: np.ndarray
    manual_vertex_indices: np.ndarray
    full_basis: np.ndarray
    lift_indices: np.ndarray
    lift_weights: np.ndarray
    fit_end: int
    train_end: int
    frame_count: int
    original_count: int
    state_count: int
    num_surface_points: int


def _scale_grid(protocol: HierarchicalResidualShrinkageProtocol) -> np.ndarray:
    count = int(
        round(protocol.scale_grid_maximum_m / protocol.scale_grid_step_m)
    )
    grid = np.linspace(0.0, protocol.scale_grid_maximum_m, count + 1)
    if not np.isclose(grid[1] - grid[0], protocol.scale_grid_step_m):
        raise ValueError("scale grid maximum must be divisible by its step")
    return grid


def lift_residual_unclipped(
    tracked_residual: np.ndarray,
    state_count: int,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Lift a tracked correction without pointwise clipping."""

    tracked = np.asarray(tracked_residual, dtype=float)
    if tracked.ndim != 3 or tracked.shape[2] != 3:
        raise ValueError("tracked_residual must have shape (T, N, 3)")
    original_count = tracked.shape[1]
    if state_count < original_count:
        raise ValueError("state_count cannot be below the tracked-point count")
    lifted = np.zeros((len(tracked), state_count, 3), dtype=float)
    lifted[:, :original_count] = tracked
    if state_count > original_count:
        if indices.shape != weights.shape or indices.shape[0] != state_count - original_count:
            raise ValueError("lift map does not match the extra state vertices")
        lifted[:, original_count:] = np.sum(
            tracked[:, indices] * weights[None, :, :, None],
            axis=2,
        )
    return lifted


def smooth_radial_shrinkage(values: np.ndarray, scale_m: float) -> np.ndarray:
    """Smoothly shrink vector norms toward an asymptote set by ``scale_m``."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("values must have shape (T, N, 3)")
    if scale_m < 0.0:
        raise ValueError("scale_m must be nonnegative")
    if scale_m == 0.0:
        return np.zeros_like(array)
    norms = np.linalg.norm(array, axis=2, keepdims=True)
    shrunk_norm = scale_m * np.tanh(norms / scale_m)
    return array * shrunk_norm / np.maximum(norms, 1e-12)


def frame_balanced_scale_statistics(
    target_residual: np.ndarray,
    valid: np.ndarray,
    raw_prediction: np.ndarray,
    scales_m: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
) -> ScaleLikelihoodStatistics:
    """Evaluate smooth-shrinkage errors with equal weight per validation frame."""

    target = np.asarray(target_residual, dtype=float)
    support = np.asarray(valid, dtype=bool)
    prediction = np.asarray(raw_prediction, dtype=float)
    scales = np.asarray(scales_m, dtype=float)
    if target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target_residual must have shape (T, N, 3)")
    if support.shape != target.shape[:2]:
        raise ValueError("valid must match target_residual")
    if prediction.shape != (end_frame - start_frame, target.shape[1], 3):
        raise ValueError("raw_prediction does not match the requested interval")
    if scales.ndim != 1 or len(scales) == 0 or np.any(scales < 0.0):
        raise ValueError("scales_m must be a nonempty nonnegative vector")
    squared_error = np.zeros(len(scales), dtype=float)
    used_frames = 0
    for offset, frame in enumerate(range(start_frame, end_frame)):
        mask = support[frame]
        if not np.any(mask):
            continue
        observed = target[frame, mask]
        frame_prediction = prediction[offset, mask]
        norms = np.linalg.norm(frame_prediction, axis=1)
        direction_dot_target = np.sum(frame_prediction * observed, axis=1) / np.maximum(
            norms, 1e-12
        )
        shrunk_norms = np.zeros((len(scales), len(norms)), dtype=float)
        positive = scales > 0.0
        shrunk_norms[positive] = scales[positive, None] * np.tanh(
            norms[None] / scales[positive, None]
        )
        squared_error += (
            float(np.mean(np.sum(np.square(observed), axis=1)))
            - 2.0 * np.mean(
                shrunk_norms * direction_dot_target[None], axis=1
            )
            + np.mean(np.square(shrunk_norms), axis=1)
        )
        used_frames += 1
    if used_frames == 0:
        raise ValueError("validation interval contains no supported frames")
    raw_rms = float(np.sqrt(np.mean(np.sum(np.square(prediction), axis=2))))
    return ScaleLikelihoodStatistics(
        squared_error_by_scale=squared_error,
        frame_count=used_frames,
        raw_rms_m=raw_rms,
    )


def scale_log_likelihood(
    statistics: ScaleLikelihoodStatistics,
    scales_m: np.ndarray,
    observation_std_m: float,
) -> np.ndarray:
    """Evaluate the frame-balanced Gaussian pseudo-likelihood over scale."""

    if observation_std_m <= 0.0:
        raise ValueError("observation_std_m must be positive")
    scales = np.asarray(scales_m, dtype=float)
    squared_error = np.asarray(statistics.squared_error_by_scale, dtype=float)
    if squared_error.shape != scales.shape:
        raise ValueError("statistics error curve must match scales_m")
    return (
        -0.5 * squared_error / observation_std_m**2
        - 3.0 * statistics.frame_count * math.log(observation_std_m)
    )


def positive_normal_log_prior(
    scales_m: np.ndarray,
    mean_m: float,
    std_m: float,
) -> np.ndarray:
    """Log density of a normal population distribution truncated at zero."""

    if mean_m < 0.0 or std_m <= 0.0:
        raise ValueError("positive-normal mean must be nonnegative and std positive")
    scales = np.asarray(scales_m, dtype=float)
    if np.any(scales < 0.0):
        raise ValueError("scale grid must be nonnegative")
    positive_mass = 0.5 * (1.0 + math.erf(mean_m / (std_m * math.sqrt(2.0))))
    return (
        -0.5 * np.square((scales - mean_m) / std_m)
        - math.log(std_m)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(max(positive_mass, 1e-300))
    )


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def scale_posterior(
    statistics: ScaleLikelihoodStatistics,
    scales_m: np.ndarray,
    *,
    observation_std_m: float,
    population_mean_m: float,
    population_std_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized scale weights, likelihood, and population log prior."""

    likelihood = scale_log_likelihood(statistics, scales_m, observation_std_m)
    prior = positive_normal_log_prior(
        scales_m, population_mean_m, population_std_m
    )
    log_weights = likelihood + prior
    weights = np.exp(log_weights - _logsumexp(log_weights))
    return weights, likelihood, prior


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(np.searchsorted(cumulative, probability, side="left"), len(values) - 1)
    return float(values[order[index]])


def _prepare_case(
    case_dir: Path,
    protocol: HierarchicalResidualShrinkageProtocol,
) -> _CaseData:
    split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
    train_start, train_end = (int(value) for value in split["train"])
    test_start, frame_count = (int(value) for value in split["test"])
    if train_start != 0 or test_start != train_end:
        raise ValueError(f"unsupported split for {case_dir.name}")
    fit_end = math.floor(protocol.fit_fraction * train_end)
    if not 2 < fit_end < train_end < frame_count:
        raise ValueError(f"invalid fit/validation/future split for {case_dir.name}")
    data = _load_pickle(case_dir / "final_data.pkl")
    baseline = np.asarray(_load_pickle(case_dir / "inference.pkl"), dtype=float)
    gt_track = np.asarray(_load_pickle(case_dir / "gt_track_3d.pkl"), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controllers = np.asarray(data["controller_points"], dtype=float)
    original_count = observed.shape[1]
    if baseline.shape[0] < frame_count or baseline.shape[1] < original_count:
        raise ValueError(f"baseline does not cover {case_dir.name}")
    baseline = baseline[:frame_count]
    valid = _target_validity(visible, motion_valid)
    residual = observed - baseline[:, :original_count]
    manual_initial_valid = np.isfinite(gt_track[0]).all(axis=1)
    _, manual_vertex_indices = _nearest_distances(
        baseline[0], gt_track[0, manual_initial_valid], p=2
    )
    manual_values = gt_track[:, manual_initial_valid]
    manual_valid = np.isfinite(manual_values).all(axis=2)
    manual_residual = np.zeros_like(manual_values, dtype=float)
    manual_residual[manual_valid] = (
        manual_values
        - baseline[: len(manual_values), manual_vertex_indices]
    )[manual_valid]
    maximum_rank = min(max(protocol.rank_candidates), fit_end - 1)
    full_basis = fit_residual_basis(
        residual,
        valid,
        end_frame=fit_end,
        maximum_rank=maximum_rank,
    )
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, protocol.interpolation_neighbors
    )
    return _CaseData(
        name=case_dir.name,
        case_dir=case_dir,
        observed=observed,
        visible=visible,
        valid=valid,
        baseline=baseline,
        gt_track=gt_track,
        features=controller_action_features(controllers),
        residual=residual,
        manual_residual=manual_residual,
        manual_valid=manual_valid,
        manual_vertex_indices=manual_vertex_indices,
        full_basis=full_basis,
        lift_indices=lift_indices,
        lift_weights=lift_weights,
        fit_end=fit_end,
        train_end=train_end,
        frame_count=frame_count,
        original_count=original_count,
        state_count=baseline.shape[1],
        num_surface_points=original_count + len(np.asarray(data["surface_points"])),
    )


def _validation_raw_correction(
    case: _CaseData,
    key: DynamicsKey,
    protocol: HierarchicalResidualShrinkageProtocol,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    rank, persistence, ridge = key
    if rank > len(case.full_basis):
        raise ValueError(f"rank {rank} is unavailable for {case.name}")
    basis = case.full_basis[:rank]
    latent = _project_residuals(
        case.residual[: case.fit_end],
        case.valid[: case.fit_end],
        basis,
        ridge=protocol.projection_ridge,
    )
    standardized, action_mean, action_scale = _standardize_actions(
        case.features, end_frame=case.fit_end
    )
    dynamics = _fit_latent_dynamics(
        latent,
        standardized,
        end_frame=case.fit_end,
        persistence=persistence,
        ridge=ridge,
    )
    observed_norm = float(np.max(np.linalg.norm(latent, axis=1)))
    predicted = _rollout_latent(
        latent[-1],
        standardized,
        dynamics,
        start_frame=case.fit_end,
        end_frame=case.train_end,
        persistence=persistence,
        norm_cap=max(protocol.maximum_state_multiplier * observed_norm, 1e-8),
    )
    tracked = (predicted @ basis).reshape(
        case.train_end - case.fit_end, case.original_count, 3
    )
    lifted = lift_residual_unclipped(
        tracked, case.state_count, case.lift_indices, case.lift_weights
    )
    raw_rms = float(np.sqrt(np.mean(np.sum(np.square(lifted), axis=2))))
    model = np.array(
        [rank, persistence, ridge, raw_rms], dtype=float
    )
    actions = np.concatenate([action_mean, action_scale])
    return lifted, raw_rms, model, actions


def _candidate_statistics(
    case: _CaseData,
    protocol: HierarchicalResidualShrinkageProtocol,
) -> dict[DynamicsKey, ScaleLikelihoodStatistics]:
    result = {}
    scales = _scale_grid(protocol)
    for rank in sorted(set(protocol.rank_candidates)):
        if rank > len(case.full_basis):
            continue
        for persistence in protocol.persistence_candidates:
            for ridge in protocol.ridge_candidates:
                key = (rank, persistence, ridge)
                raw, raw_rms, _, _ = _validation_raw_correction(
                    case, key, protocol
                )
                stats = frame_balanced_scale_statistics(
                    case.residual,
                    case.valid,
                    raw[:, : case.original_count],
                    scales,
                    start_frame=case.fit_end,
                    end_frame=case.train_end,
                )
                manual_stats = frame_balanced_scale_statistics(
                    case.manual_residual,
                    case.manual_valid,
                    raw[:, case.manual_vertex_indices],
                    scales,
                    start_frame=case.fit_end,
                    end_frame=case.train_end,
                )
                combined_error = 0.5 * stats.squared_error_by_scale
                combined_error += (
                    0.5
                    * manual_stats.squared_error_by_scale
                    * stats.frame_count
                    / manual_stats.frame_count
                )
                result[key] = ScaleLikelihoodStatistics(
                    squared_error_by_scale=combined_error,
                    frame_count=stats.frame_count,
                    raw_rms_m=raw_rms,
                )
    return result


def select_shared_hyperparameters(
    statistics_by_case: Mapping[str, Mapping[DynamicsKey, ScaleLikelihoodStatistics]],
    held_out_case: str,
    protocol: HierarchicalResidualShrinkageProtocol,
) -> tuple[SharedShrinkageSelection, list[dict[str, float | int]]]:
    """Select every statistical/dynamics setting using other interactions only."""

    if held_out_case not in statistics_by_case:
        raise ValueError("held_out_case is absent from the candidate statistics")
    training_cases = tuple(case for case in statistics_by_case if case != held_out_case)
    if len(training_cases) < 2:
        raise ValueError("outer leave-one-out selection requires at least two training cases")
    common_keys = set(statistics_by_case[training_cases[0]])
    for case in training_cases[1:]:
        common_keys &= set(statistics_by_case[case])
    scales = _scale_grid(protocol)
    rows: list[dict[str, float | int]] = []
    best: tuple[tuple[float, int, float, float, float, float, float], SharedShrinkageSelection] | None = None
    for rank, persistence, ridge in sorted(common_keys):
        for observation_std in protocol.observation_std_candidates_m:
            likelihoods = {
                case: scale_log_likelihood(
                    statistics_by_case[case][(rank, persistence, ridge)],
                    scales,
                    observation_std,
                )
                for case in training_cases
            }
            for population_mean in protocol.population_mean_candidates_m:
                for population_std in protocol.population_std_candidates_m:
                    prior = positive_normal_log_prior(
                        scales, population_mean, population_std
                    )
                    evidence = sum(
                        _logsumexp(likelihoods[case] + prior)
                        for case in training_cases
                    )
                    selection = SharedShrinkageSelection(
                        rank=rank,
                        persistence=persistence,
                        ridge=ridge,
                        observation_std_m=observation_std,
                        population_mean_m=population_mean,
                        population_std_m=population_std,
                        log_evidence=evidence,
                    )
                    rows.append(
                        {
                            **asdict(selection),
                        }
                    )
                    ranking = (
                        -evidence,
                        rank,
                        -ridge,
                        -persistence,
                        observation_std,
                        population_std,
                        population_mean,
                    )
                    if best is None or ranking < best[0]:
                        best = (ranking, selection)
    assert best is not None
    return best[1], rows


def _final_future_raw_correction(
    case: _CaseData,
    selection: SharedShrinkageSelection,
    protocol: HierarchicalResidualShrinkageProtocol,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    basis = case.full_basis[: selection.rank]
    latent = _project_residuals(
        case.residual[: case.train_end],
        case.valid[: case.train_end],
        basis,
        ridge=protocol.projection_ridge,
    )
    standardized, action_mean, action_scale = _standardize_actions(
        case.features, end_frame=case.train_end
    )
    dynamics = _fit_latent_dynamics(
        latent,
        standardized,
        end_frame=case.train_end,
        persistence=selection.persistence,
        ridge=selection.ridge,
    )
    observed_norm = float(np.max(np.linalg.norm(latent, axis=1)))
    predicted = _rollout_latent(
        latent[-1],
        standardized,
        dynamics,
        start_frame=case.train_end,
        end_frame=case.frame_count,
        persistence=selection.persistence,
        norm_cap=max(protocol.maximum_state_multiplier * observed_norm, 1e-8),
    )
    tracked = (predicted @ basis).reshape(
        case.frame_count - case.train_end, case.original_count, 3
    )
    lifted = lift_residual_unclipped(
        tracked, case.state_count, case.lift_indices, case.lift_weights
    )
    raw_rms = float(np.sqrt(np.mean(np.sum(np.square(lifted), axis=2))))
    return lifted, raw_rms, {
        "basis": basis,
        "dynamics": dynamics,
        "action_mean": action_mean,
        "action_scale": action_scale,
        "lift_indices": case.lift_indices,
        "lift_weights": case.lift_weights,
    }


def _compact_bootstrap(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key in {"samples", "block_length", "seed", "macro"}
    }


def run_hierarchical_residual_shrinkage(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    protocol: HierarchicalResidualShrinkageProtocol | None = None,
) -> dict[str, Any]:
    """Run three outer folds and evaluate each interaction's untouched future."""

    config = protocol or HierarchicalResidualShrinkageProtocol()
    if not 0.0 < config.fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    if not config.rank_candidates or any(rank < 1 for rank in config.rank_candidates):
        raise ValueError("rank candidates must be positive")
    if any(not 0.0 <= value <= 1.0 for value in config.persistence_candidates):
        raise ValueError("persistence candidates must lie in [0, 1]")
    positive_grids = (
        config.ridge_candidates,
        config.observation_std_candidates_m,
        config.population_std_candidates_m,
    )
    if any(not values or any(value <= 0.0 for value in values) for values in positive_grids):
        raise ValueError("ridge, observation, and population std grids must be positive")
    if config.scale_grid_maximum_m <= 0.0 or config.scale_grid_step_m <= 0.0:
        raise ValueError("scale grid bounds must be positive")
    scales = _scale_grid(config)
    root = Path(data_root)
    output = Path(output_dir)
    cases = tuple(config.development_cases)
    if len(cases) != 3:
        raise ValueError("the locked protocol requires exactly three development cases")
    specification = json.loads(
        json.dumps(
            {
                "method": "smooth radial action-residual shrinkage with a positive hierarchical magnitude scale",
                "protocol": asdict(config),
                "outer_validation": "each interaction's shared settings use the other two interactions only",
                "local_scale_data": "held-out interaction validation pseudo-tracks and manual tracks; no future labels",
                "future_inputs": "controller actions only",
                "hard_residual_cap": None,
                "cases": list(cases),
            }
        )
    )
    locked = _lock_protocol(output, specification)
    prepared = {case: _prepare_case(root / case, config) for case in cases}
    statistics = {
        case: _candidate_statistics(prepared[case], config) for case in cases
    }

    case_results: dict[str, Any] = {}
    comparison_manifest = {"schema_version": 1, "cases": []}
    paired_horizons = {horizon: {} for horizon in HORIZON_LABELS}
    for case_index, case_name in enumerate(cases):
        case = prepared[case_name]
        selection, selection_rows = select_shared_hyperparameters(
            statistics, case_name, config
        )
        case_output = output / "cases" / case_name
        case_output.mkdir(parents=True, exist_ok=True)
        grid_path = case_output / "shared_selection_grid.npz"
        np.savez_compressed(
            grid_path,
            **{
                name: np.asarray([row[name] for row in selection_rows])
                for name in selection_rows[0]
            },
        )
        key = (selection.rank, selection.persistence, selection.ridge)
        held_out_stats = statistics[case_name][key]
        posterior_weights, log_likelihood, log_prior = scale_posterior(
            held_out_stats,
            scales,
            observation_std_m=selection.observation_std_m,
            population_mean_m=selection.population_mean_m,
            population_std_m=selection.population_std_m,
        )
        scale_mean = float(np.sum(scales * posterior_weights))
        scale_std = float(
            np.sqrt(np.sum(posterior_weights * np.square(scales - scale_mean)))
        )
        validation_raw, validation_raw_rms, _, _ = _validation_raw_correction(
            case, key, config
        )
        validation_candidate = case.baseline.copy()
        validation_candidate[case.fit_end : case.train_end] += (
            smooth_radial_shrinkage(validation_raw, scale_mean)
        )
        baseline_validation = evaluate_official_phystwin_interval(
            case.baseline,
            case.observed,
            case.visible,
            case.gt_track,
            num_surface_points=case.num_surface_points,
            start_frame=case.fit_end,
            end_frame=case.train_end,
        )
        corrected_validation = evaluate_official_phystwin_interval(
            validation_candidate,
            case.observed,
            case.visible,
            case.gt_track,
            num_surface_points=case.num_surface_points,
            start_frame=case.fit_end,
            end_frame=case.train_end,
        )
        future_raw, future_raw_rms, model = _final_future_raw_correction(
            case, selection, config
        )
        correction = smooth_radial_shrinkage(future_raw, scale_mean)
        corrected = case.baseline.copy()
        corrected[case.train_end :] += correction
        trajectory_path = case_output / "trajectory.pkl"
        with trajectory_path.open("wb") as handle:
            pickle.dump(
                corrected.astype(np.float32),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        posterior_path = case_output / "scale_posterior.npz"
        np.savez_compressed(
            posterior_path,
            scale_grid_m=scales,
            weights=posterior_weights,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            **model,
        )
        baseline_test = evaluate_official_phystwin_interval(
            case.baseline,
            case.observed,
            case.visible,
            case.gt_track,
            num_surface_points=case.num_surface_points,
            start_frame=case.train_end,
            end_frame=case.frame_count,
        )
        corrected_test = evaluate_official_phystwin_interval(
            corrected,
            case.observed,
            case.visible,
            case.gt_track,
            num_surface_points=case.num_surface_points,
            start_frame=case.train_end,
            end_frame=case.frame_count,
        )
        baseline_frames = official_metrics_by_frame(
            case.baseline,
            case.observed,
            case.visible,
            case.gt_track,
            num_surface_points=case.num_surface_points,
            start_frame=case.train_end,
            end_frame=case.frame_count,
        )
        corrected_frames = official_metrics_by_frame(
            corrected,
            case.observed,
            case.visible,
            case.gt_track,
            num_surface_points=case.num_surface_points,
            start_frame=case.train_end,
            end_frame=case.frame_count,
        )
        horizon_results = {}
        for horizon, indexes in split_future_horizon(
            case.frame_count - case.train_end
        ).items():
            paired_horizons[horizon][case_name] = (
                {metric: baseline_frames[metric][indexes] for metric in METRICS},
                {metric: corrected_frames[metric][indexes] for metric in METRICS},
            )
            horizon_results[horizon] = {
                metric: {
                    "baseline_mean_m": float(np.mean(baseline_frames[metric][indexes])),
                    "corrected_mean_m": float(np.mean(corrected_frames[metric][indexes])),
                    "percent_change": 100.0
                    * (
                        float(np.mean(corrected_frames[metric][indexes]))
                        / float(np.mean(baseline_frames[metric][indexes]))
                        - 1.0
                    ),
                }
                for metric in METRICS
            }
        correction_norm = np.linalg.norm(correction, axis=2)
        top_rows = sorted(selection_rows, key=lambda row: -float(row["log_evidence"]))[:10]
        case_summary = {
            "outer_fold": {
                "held_out_interaction": case_name,
                "shared_selection_interactions": [
                    other for other in cases if other != case_name
                ],
                "selected_shared_hyperparameters": asdict(selection),
                "top_shared_candidates": top_rows,
            },
            "scale_posterior": {
                "mean_m": scale_mean,
                "std_m": scale_std,
                "q05_m": _weighted_quantile(scales, posterior_weights, 0.05),
                "q95_m": _weighted_quantile(scales, posterior_weights, 0.95),
                "validation_raw_prediction_rms_m": validation_raw_rms,
                "future_raw_prediction_rms_m": future_raw_rms,
            },
            "validation_diagnostic_not_used_for_selection": {
                "baseline": baseline_validation,
                "corrected": corrected_validation,
                "relative_score": _selection_score(
                    corrected_validation, baseline_validation
                ),
            },
            "future": {
                "baseline": baseline_test,
                "corrected": corrected_test,
                "relative_score": _selection_score(corrected_test, baseline_test),
                "horizons": horizon_results,
            },
            "correction": {
                "rms_m": float(np.sqrt(np.mean(np.square(correction_norm)))),
                "maximum_m": float(np.max(correction_norm)),
                "hard_cap_applied": False,
            },
            "inputs": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
                for name, path in {
                    "final_data": case.case_dir / "final_data.pkl",
                    "baseline_trajectory": case.case_dir / "inference.pkl",
                    "gt_track_3d": case.case_dir / "gt_track_3d.pkl",
                }.items()
            },
            "outputs": {
                "trajectory": str(trajectory_path.resolve()),
                "scale_posterior": str(posterior_path.resolve()),
                "shared_selection_grid": str(grid_path.resolve()),
            },
        }
        summary_path = case_output / "summary.json"
        summary_path.write_text(
            json.dumps(case_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        case_summary["outputs"]["summary"] = str(summary_path.resolve())
        case_results[case_name] = case_summary
        comparison_manifest["cases"].append(
            {
                "name": case_name,
                "final_data": str((case.case_dir / "final_data.pkl").resolve()),
                "gt_track_3d": str((case.case_dir / "gt_track_3d.pkl").resolve()),
                "baseline_trajectory": str((case.case_dir / "inference.pkl").resolve()),
                "candidate_trajectory": str(trajectory_path.resolve()),
                "start_frame": case.train_end,
                "end_frame": case.frame_count,
            }
        )

    manifest_path = output / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison = compare_phystwin_manifest(
        manifest_path,
        output / "comparison.json",
        samples=config.bootstrap_samples,
        block_length=config.bootstrap_block_length,
        seed=config.bootstrap_seed,
        cluster_by_phystwin_object=False,
    )
    horizon_summary = {
        horizon: _compact_bootstrap(
            paired_block_bootstrap(
                paired_horizons[horizon],
                samples=config.bootstrap_samples,
                block_length=config.bootstrap_block_length,
                seed=config.bootstrap_seed + index,
            )
        )
        for index, horizon in enumerate(HORIZON_LABELS)
    }
    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_results": case_results,
        "future_comparison": comparison,
        "future_horizons": horizon_summary,
        "contract": {
            "shared_hyperparameters": "rank, persistence, ridge, observation std, population mean, and population std",
            "selection": "outer leave-one-interaction-out; no held-out interaction enters shared selection",
            "local_scale": "posterior from equal-weight validation pseudo-track and manual-track residual channels under the held-out fold's frozen population prior",
            "correction": "raw action residual passed through smooth radial tanh shrinkage at the posterior-mean interaction scale",
            "pointwise_hard_cap": None,
            "smooth_radial_asymptote": "posterior-mean local scale; no clipping kink or flat-gradient region",
            "future_observations_or_labels_used": False,
        },
    }
    result_path = output / "hierarchical_shrinkage_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
