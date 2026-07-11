"""Split-conformal and NEES audits for Bayesian PhysTwin anchors."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .phystwin_bayesian_anchor import robust_random_walk_endpoint
from .phystwin_confirmatory import (
    DEVELOPMENT_CASES,
    _lock_protocol,
    _split_for_case,
)
from .phystwin_official_evaluation import (
    _nearest_distances,
    official_phystwin_metrics_by_frame,
)
from .phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
)


METRICS = ("chamfer_distance_m", "track_error_m")
HORIZON_LABELS = ("early", "middle", "late")
CONFORMAL_METHODS = (
    "posterior_scaled",
    "posterior_additive",
    "unscaled",
)
CHI_SQUARE_3_THRESHOLDS = {
    "80": 4.64162767608745,
    "90": 6.251388631170325,
    "95": 7.814727903251179,
}
CALIBRATION_IMPLEMENTATION_VERSION = 2


@dataclass(frozen=True)
class PhysTwinCalibrationProtocol:
    """Frozen state fit, conformal, and posterior-audit choices."""

    fit_fraction: float = 0.75
    process_std_m: float = 0.005
    observation_std_m: float = 0.001
    initial_std_m: float = 0.01
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0
    interpolation_neighbors: int = 4
    maximum_residual_m: float = 0.01
    coverage_levels: tuple[float, ...] = (0.8, 0.9)
    primary_conformal_method: str = "posterior_scaled"
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 20260711
    development_cases: tuple[str, ...] = DEVELOPMENT_CASES


def finite_sample_conformal_quantile(
    scores: np.ndarray,
    coverage: float,
) -> tuple[float, int]:
    """Return the conservative split-conformal quantile and one-based rank."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError("at least one calibration score is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be finite")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    rank = math.ceil((len(values) + 1) * coverage)
    if rank > len(values):
        return math.inf, rank
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def conformal_upper_bounds(
    calibration_target: np.ndarray,
    calibration_prediction: np.ndarray,
    future_prediction: np.ndarray,
    *,
    coverage: float,
    score: str,
) -> tuple[np.ndarray, float, int]:
    """Build nonnegative one-sided split-conformal prediction intervals."""

    target = np.asarray(calibration_target, dtype=float).reshape(-1)
    calibration = np.asarray(calibration_prediction, dtype=float).reshape(-1)
    future = np.asarray(future_prediction, dtype=float).reshape(-1)
    if target.shape != calibration.shape:
        raise ValueError("calibration targets and predictions must have equal shape")
    if not np.all(np.isfinite(target)) or np.any(target < 0.0):
        raise ValueError("calibration targets must be finite and nonnegative")
    if not np.all(np.isfinite(calibration)) or not np.all(np.isfinite(future)):
        raise ValueError("predictions must be finite")
    if score == "scaled":
        if np.any(calibration <= 0.0) or np.any(future <= 0.0):
            raise ValueError("scaled conformal predictions must be positive")
        scores = target / calibration
    elif score == "additive":
        scores = target - calibration
    else:
        raise ValueError("score must be 'scaled' or 'additive'")
    quantile, rank = finite_sample_conformal_quantile(scores, coverage)
    if not math.isfinite(quantile):
        return np.full_like(future, math.inf), quantile, rank
    upper = quantile * future if score == "scaled" else future + quantile
    return np.maximum(upper, 0.0), quantile, rank


def lift_diagonal_anchor_variance(
    original_variance: np.ndarray,
    state_count: int,
    lift_indices: np.ndarray,
    lift_weights: np.ndarray,
) -> np.ndarray:
    """Lift independent per-anchor coordinate variances to simulator vertices."""

    variance = np.asarray(original_variance, dtype=float).reshape(-1)
    indices = np.asarray(lift_indices, dtype=np.int64)
    weights = np.asarray(lift_weights, dtype=float)
    if len(variance) > state_count:
        raise ValueError("state_count is smaller than the original anchor count")
    if np.any(variance < 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("anchor variance must be finite and nonnegative")
    expected_extra = state_count - len(variance)
    if indices.shape != weights.shape or indices.shape[0] != expected_extra:
        raise ValueError("lift map does not match the requested state count")
    lifted = np.empty(state_count, dtype=float)
    lifted[: len(variance)] = variance
    if expected_extra:
        lifted[len(variance) :] = np.sum(
            np.square(weights) * variance[indices],
            axis=1,
        )
    return lifted


def _track_correspondence(
    initial_vertices: np.ndarray,
    gt_track_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    initial_mask = np.isfinite(gt_track_3d[0]).all(axis=1)
    _, indices = _nearest_distances(
        initial_vertices,
        gt_track_3d[0, initial_mask],
        p=2,
    )
    return initial_mask, indices


def _posterior_metric_scales(
    posterior_variance: np.ndarray,
    *,
    process_variance: float,
    state_count: int,
    num_surface_points: int,
    lift_indices: np.ndarray,
    lift_weights: np.ndarray,
    gt_track_3d: np.ndarray,
    track_initial_mask: np.ndarray,
    track_indices: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> dict[str, np.ndarray]:
    chamfer_scale = np.empty(end_frame - start_frame, dtype=float)
    track_scale = np.empty(end_frame - start_frame, dtype=float)
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        horizon = output_index + 1
        original = posterior_variance + horizon * process_variance
        lifted = lift_diagonal_anchor_variance(
            original,
            state_count,
            lift_indices,
            lift_weights,
        )
        radial_std = np.sqrt(3.0 * lifted)
        chamfer_scale[output_index] = np.mean(radial_std[:num_surface_points])
        current_tracks = gt_track_3d[frame, track_initial_mask]
        current_mask = np.isfinite(current_tracks).all(axis=1)
        track_scale[output_index] = (
            float(np.mean(radial_std[track_indices[current_mask]]))
            if np.any(current_mask)
            else 1e-12
        )
    return {
        "chamfer_distance_m": np.maximum(chamfer_scale, 1e-12),
        "track_error_m": np.maximum(track_scale, 1e-12),
    }


def _manual_track_nees(
    prediction: np.ndarray,
    posterior_variance: np.ndarray,
    *,
    process_variance: float,
    lift_indices: np.ndarray,
    lift_weights: np.ndarray,
    gt_track_3d: np.ndarray,
    track_initial_mask: np.ndarray,
    track_indices: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        original = posterior_variance + (output_index + 1) * process_variance
        lifted = lift_diagonal_anchor_variance(
            original,
            prediction.shape[1],
            lift_indices,
            lift_weights,
        )
        current_tracks = gt_track_3d[frame, track_initial_mask]
        current_mask = np.isfinite(current_tracks).all(axis=1)
        if not np.any(current_mask):
            continue
        selected_indices = track_indices[current_mask]
        error = prediction[frame, selected_indices] - current_tracks[current_mask]
        selected_variance = lifted[selected_indices]
        positive = selected_variance > 0.0
        if np.any(positive):
            values.append(
                np.sum(
                    np.square(error[positive]) / selected_variance[positive, None],
                    axis=1,
                )
            )
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def summarize_nees(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize 3D NEES, whose calibrated expectation is three."""

    nees = np.asarray(values, dtype=float).reshape(-1)
    if len(nees) == 0:
        return {"count": 0}
    mean_3d = float(np.mean(nees))
    result: dict[str, float | int | None] = {
        "count": int(len(nees)),
        "expected_mean_3d": 3.0,
        "mean_3d": mean_3d,
        "mean_per_coordinate": mean_3d / 3.0,
        "median_3d": float(np.median(nees)),
        "covariance_multiplier_for_mean_nees_3": mean_3d / 3.0,
    }
    for label, threshold in CHI_SQUARE_3_THRESHOLDS.items():
        result[f"ellipsoid_coverage_{label}"] = float(np.mean(nees <= threshold))
    return result


def _conformal_case_readout(
    calibration_target: np.ndarray,
    future_target: np.ndarray,
    calibration_scale: np.ndarray,
    future_scale: np.ndarray,
    coverage_levels: tuple[float, ...],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    output: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    for method in CONFORMAL_METHODS:
        method_output: dict[str, object] = {}
        if method == "posterior_scaled":
            calibration_prediction = calibration_scale
            future_prediction = future_scale
            score = "scaled"
        elif method == "posterior_additive":
            calibration_prediction = calibration_scale
            future_prediction = future_scale
            score = "additive"
        else:
            calibration_prediction = np.zeros_like(calibration_target)
            future_prediction = np.zeros_like(future_target)
            score = "additive"
        for coverage in coverage_levels:
            label = str(round(100.0 * coverage))
            upper, quantile, rank = conformal_upper_bounds(
                calibration_target,
                calibration_prediction,
                future_prediction,
                coverage=coverage,
                score=score,
            )
            finite = bool(np.all(np.isfinite(upper)))
            covered = future_target <= upper
            horizon_readout: dict[str, object] = {}
            for horizon, indices in zip(
                HORIZON_LABELS,
                np.array_split(np.arange(len(future_target)), 3),
                strict=True,
            ):
                horizon_readout[horizon] = {
                    "future_frame_count": int(len(indices)),
                    "covered_frame_count": (
                        int(np.sum(covered[indices])) if finite else None
                    ),
                    "future_coverage": (
                        float(np.mean(covered[indices])) if finite else None
                    ),
                }
            method_output[label] = {
                "nominal_coverage": coverage,
                "calibration_count": int(len(calibration_target)),
                "finite_sample_rank": rank,
                "finite_bound": finite,
                "quantile": float(quantile) if finite else None,
                "future_frame_count": int(len(future_target)),
                "covered_frame_count": int(np.sum(covered)) if finite else None,
                "future_coverage": float(np.mean(covered)) if finite else None,
                "mean_upper_bound_m": float(np.mean(upper)) if finite else None,
                "median_upper_bound_m": float(np.median(upper)) if finite else None,
                "p95_upper_bound_m": float(np.quantile(upper, 0.95))
                if finite
                else None,
                "future_by_horizon": horizon_readout,
            }
            arrays[f"{method}_{label}_upper_m"] = upper
        output[method] = method_output
    return output, arrays


def _case_bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if len(values) == 0:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = np.mean(values[indices], axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _aggregate_conformal(
    cases: tuple[str, ...],
    case_results: dict[str, dict[str, object]],
    *,
    coverage_levels: tuple[float, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for method_index, method in enumerate(CONFORMAL_METHODS):
        method_output: dict[str, object] = {}
        for metric_index, metric in enumerate(METRICS):
            metric_output: dict[str, object] = {}
            for coverage_index, coverage in enumerate(coverage_levels):
                label = str(round(100.0 * coverage))
                records = [
                    case_results[case]["conformal"][metric][method][label]
                    for case in cases
                ]
                finite = [record for record in records if record["finite_bound"]]
                case_coverages = np.asarray(
                    [record["future_coverage"] for record in finite],
                    dtype=float,
                )
                covered_frames = sum(
                    int(record["covered_frame_count"]) for record in finite
                )
                frame_count = sum(
                    int(record["future_frame_count"]) for record in finite
                )
                mean_bounds = np.asarray(
                    [record["mean_upper_bound_m"] for record in finite],
                    dtype=float,
                )
                seed = (
                    bootstrap_seed
                    + 100 * method_index
                    + 10 * metric_index
                    + coverage_index
                )
                horizon_output: dict[str, object] = {}
                for horizon in HORIZON_LABELS:
                    horizon_records = [
                        record["future_by_horizon"][horizon] for record in finite
                    ]
                    horizon_case_coverage = np.asarray(
                        [record["future_coverage"] for record in horizon_records],
                        dtype=float,
                    )
                    horizon_covered = sum(
                        int(record["covered_frame_count"]) for record in horizon_records
                    )
                    horizon_count = sum(
                        int(record["future_frame_count"]) for record in horizon_records
                    )
                    horizon_output[horizon] = {
                        "macro_case_coverage": (
                            float(np.mean(horizon_case_coverage))
                            if len(horizon_case_coverage)
                            else None
                        ),
                        "micro_frame_coverage": (
                            horizon_covered / horizon_count if horizon_count else None
                        ),
                    }
                metric_output[label] = {
                    "nominal_coverage": coverage,
                    "finite_case_count": len(finite),
                    "insufficient_calibration_case_count": len(records) - len(finite),
                    "macro_case_coverage": (
                        float(np.mean(case_coverages)) if len(finite) else None
                    ),
                    "macro_case_coverage_ci_95": _case_bootstrap_interval(
                        case_coverages,
                        samples=bootstrap_samples,
                        seed=seed,
                    ),
                    "micro_frame_coverage": (
                        covered_frames / frame_count if frame_count else None
                    ),
                    "cases_at_or_above_nominal": int(
                        np.sum(case_coverages >= coverage)
                    ),
                    "median_case_mean_upper_bound_m": (
                        float(np.median(mean_bounds)) if len(finite) else None
                    ),
                    "mean_case_mean_upper_bound_m": (
                        float(np.mean(mean_bounds)) if len(finite) else None
                    ),
                    "future_by_horizon": horizon_output,
                }
            method_output[metric] = metric_output
        output[method] = method_output
    return output


def _aggregate_nees(
    cases: tuple[str, ...],
    case_results: dict[str, dict[str, object]],
    arrays: dict[str, dict[str, np.ndarray]],
    key: str,
) -> dict[str, object]:
    available = [
        case for case in cases if key in arrays[case] and len(arrays[case][key]) > 0
    ]
    joined = (
        np.concatenate([arrays[case][key] for case in available])
        if available
        else np.empty(0, dtype=float)
    )
    case_means = np.asarray(
        [case_results[case]["nees"][key]["mean_3d"] for case in available],
        dtype=float,
    )
    result: dict[str, object] = {
        "case_count": len(available),
        "micro": summarize_nees(joined),
        "macro_mean_3d": float(np.mean(case_means)) if len(case_means) else None,
        "median_case_mean_3d": (
            float(np.median(case_means)) if len(case_means) else None
        ),
    }
    return result


def _aggregate_point_metrics(
    cases: tuple[str, ...],
    case_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for metric in METRICS:
        baseline = np.asarray(
            [
                case_results[case]["future_point_metrics"][metric]["baseline_mean_m"]
                for case in cases
            ],
            dtype=float,
        )
        anchor = np.asarray(
            [
                case_results[case]["future_point_metrics"][metric][
                    "strict_split_anchor_mean_m"
                ]
                for case in cases
            ],
            dtype=float,
        )
        percent_change = 100.0 * (anchor / baseline - 1.0)
        output[metric] = {
            "case_count": len(cases),
            "baseline_equal_case_mean_m": float(np.mean(baseline)),
            "strict_split_anchor_equal_case_mean_m": float(np.mean(anchor)),
            "median_case_percent_change": float(np.median(percent_change)),
            "mean_case_percent_change": float(np.mean(percent_change)),
            "improved_case_count": int(np.sum(percent_change < 0.0)),
        }
    return output


def _operational_anchor_nees(
    case: str,
    anchor_run: Path,
    baseline: np.ndarray,
    gt_track_3d: np.ndarray,
    *,
    train_end_frame: int,
    frame_count: int,
    track_initial_mask: np.ndarray,
    track_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    case_dir = anchor_run / "cases" / case
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    posterior = np.load(case_dir / "posterior.npz")
    if int(summary["config"]["train_end_frame"]) != train_end_frame:
        raise ValueError(f"operational anchor split mismatch in {case}")
    selected = summary["selection"]["selected_candidate"]
    process_std = float(selected["process_std_m"])
    tracked = np.repeat(
        np.asarray(posterior["mean"], dtype=float)[None],
        frame_count - train_end_frame,
        axis=0,
    )
    correction = _lift_residual(
        tracked,
        baseline.shape[1],
        posterior["lift_indices"],
        posterior["lift_weights"],
        maximum_norm=float(summary["config"]["maximum_residual_m"]),
    )
    prediction = baseline.copy()
    prediction[train_end_frame:] += correction
    values = _manual_track_nees(
        prediction,
        np.asarray(posterior["variance"], dtype=float),
        process_variance=process_std**2,
        lift_indices=posterior["lift_indices"],
        lift_weights=posterior["lift_weights"],
        gt_track_3d=gt_track_3d,
        track_initial_mask=track_initial_mask,
        track_indices=track_indices,
        start_frame=train_end_frame,
        end_frame=frame_count,
    )
    return values, {
        "accepted_on_validation": bool(summary["selection"]["accepted"]),
        "process_std_m": process_std,
        "observation_std_m": float(selected["observation_std_m"]),
        **summarize_nees(values),
    }


def run_phystwin_calibration_audit(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    anchor_run_dir: str | Path | None = None,
    protocol: PhysTwinCalibrationProtocol | None = None,
    cases: Iterable[str] | None = None,
) -> dict[str, object]:
    """Run a strict split-conformal audit and manual-track posterior NEES."""

    config = PhysTwinCalibrationProtocol() if protocol is None else protocol
    if not 0.0 < config.fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    if config.process_std_m < 0.0 or config.observation_std_m <= 0.0:
        raise ValueError("process and observation standard deviations are invalid")
    if config.primary_conformal_method not in CONFORMAL_METHODS:
        raise ValueError("unsupported primary conformal method")
    if not config.coverage_levels:
        raise ValueError("at least one conformal coverage level is required")
    if any(not 0.0 < value < 1.0 for value in config.coverage_levels):
        raise ValueError("conformal coverage levels must lie in (0, 1)")
    if len({round(100.0 * value) for value in config.coverage_levels}) != len(
        config.coverage_levels
    ):
        raise ValueError("conformal coverage labels must be unique percentages")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    root = Path(data_root)
    source_manifest_path = root / "evaluation_subset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    available = tuple(str(case) for case in source_manifest["selected_cases"])
    selected_cases = available if cases is None else tuple(dict.fromkeys(cases))
    missing = sorted(set(selected_cases) - set(available))
    if missing:
        raise ValueError("cases absent from data manifest: " + ", ".join(missing))
    development = tuple(
        case for case in selected_cases if case in config.development_cases
    )
    confirmation = tuple(
        case for case in selected_cases if case not in config.development_cases
    )
    if not confirmation:
        raise ValueError("the selected cohort contains no confirmation cases")
    case_contracts = {
        case: dict(
            zip(
                ("fit_end_frame", "train_end_frame", "frame_count"),
                _split_for_case(root / case, config.fit_fraction),
                strict=True,
            )
        )
        for case in selected_cases
    }
    anchor_run = Path(anchor_run_dir) if anchor_run_dir is not None else None
    anchor_protocol: dict[str, object] | None = None
    if anchor_run is not None:
        anchor_protocol = json.loads(
            (anchor_run / "locked_protocol.json").read_text(encoding="utf-8")
        )
    output = Path(output_dir)
    specification = {
        "method": "strict split-conformal Bayesian-anchor calibration audit",
        "implementation_version": CALIBRATION_IMPLEMENTATION_VERSION,
        "status": "post-hoc calibration audit",
        "protocol": asdict(config),
        "data_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": _sha256(source_manifest_path),
        },
        "case_contracts": case_contracts,
        "cohorts": {
            "development": list(development),
            "confirmation": list(confirmation),
        },
        "operational_anchor": (
            {
                "path": str(anchor_run.resolve()),
                "protocol_id": anchor_protocol["protocol_id"],
            }
            if anchor_run is not None and anchor_protocol is not None
            else None
        ),
        "claim_contract": {
            "conformal_guarantee": (
                "finite-sample marginal one-sided coverage if per-frame scores "
                "are exchangeable; forward-time exchangeability is an assumption"
            ),
            "state_fit_interval": "[0, fit_end_frame)",
            "calibration_interval": "[fit_end_frame, train_end_frame)",
            "future_interval": "[train_end_frame, frame_count)",
            "predictor_update_after_fit": "none",
            "validation_model_selection": "none",
            "posterior_scale": (
                "mean lifted radial standard deviation; independent-anchor "
                "diagonal covariance and random-walk propagation"
            ),
            "cap_covariance_treatment": "mean is capped; covariance is not linearized through cap",
            "nees": "manual-track state error against posterior state covariance",
        },
    }
    locked = _lock_protocol(output, specification)

    case_results: dict[str, dict[str, object]] = {}
    case_arrays: dict[str, dict[str, np.ndarray]] = {}
    for case in selected_cases:
        case_dir = root / case
        contract = case_contracts[case]
        fit_end = int(contract["fit_end_frame"])
        train_end = int(contract["train_end_frame"])
        frame_count = int(contract["frame_count"])
        data = _load_pickle(case_dir / "final_data.pkl")
        baseline = np.asarray(_load_pickle(case_dir / "inference.pkl"), dtype=float)
        gt_track_3d = np.asarray(
            _load_pickle(case_dir / "gt_track_3d.pkl"), dtype=float
        )
        observed = np.asarray(data["object_points"], dtype=float)
        visible = np.asarray(data["object_visibilities"], dtype=bool)
        motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
        if len(observed) != frame_count:
            raise ValueError(f"frame contract mismatch in {case}")
        baseline = baseline[:frame_count]
        original_count = observed.shape[1]
        valid = _target_validity(visible, motion_valid)
        residual = observed - baseline[:, :original_count]
        posterior = robust_random_walk_endpoint(
            residual,
            valid,
            end_frame=fit_end,
            process_variance=config.process_std_m**2,
            observation_variance=config.observation_std_m**2,
            initial_variance=config.initial_std_m**2,
            inlier_prior=config.inlier_prior,
            outlier_variance_multiplier=config.outlier_variance_multiplier,
        )
        lift_indices, lift_weights = _lift_map(
            baseline[0],
            original_count,
            config.interpolation_neighbors,
        )
        tracked = np.repeat(
            posterior.mean[None],
            frame_count - fit_end,
            axis=0,
        )
        correction = _lift_residual(
            tracked,
            baseline.shape[1],
            lift_indices,
            lift_weights,
            maximum_norm=config.maximum_residual_m,
        )
        prediction = baseline.copy()
        prediction[fit_end:] += correction
        num_surface_points = original_count + len(np.asarray(data["surface_points"]))
        metric_values = official_phystwin_metrics_by_frame(
            prediction,
            observed,
            visible,
            gt_track_3d,
            num_surface_points=num_surface_points,
            start_frame=fit_end,
            end_frame=frame_count,
        )
        baseline_values = official_phystwin_metrics_by_frame(
            baseline,
            observed,
            visible,
            gt_track_3d,
            num_surface_points=num_surface_points,
            start_frame=fit_end,
            end_frame=frame_count,
        )
        track_initial_mask, track_indices = _track_correspondence(
            baseline[0], gt_track_3d
        )
        metric_scales = _posterior_metric_scales(
            posterior.variance,
            process_variance=config.process_std_m**2,
            state_count=baseline.shape[1],
            num_surface_points=num_surface_points,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            gt_track_3d=gt_track_3d,
            track_initial_mask=track_initial_mask,
            track_indices=track_indices,
            start_frame=fit_end,
            end_frame=frame_count,
        )
        validation_count = train_end - fit_end
        arrays: dict[str, np.ndarray] = {
            "frame_index": np.arange(fit_end, frame_count, dtype=np.int64),
            "is_calibration": np.arange(fit_end, frame_count) < train_end,
        }
        conformal: dict[str, object] = {}
        for metric in METRICS:
            values = metric_values[metric]
            scales = metric_scales[metric]
            metric_conformal, upper_arrays = _conformal_case_readout(
                values[:validation_count],
                values[validation_count:],
                scales[:validation_count],
                scales[validation_count:],
                config.coverage_levels,
            )
            conformal[metric] = metric_conformal
            arrays[f"{metric}_value_m"] = values
            arrays[f"{metric}_baseline_m"] = baseline_values[metric]
            arrays[f"{metric}_posterior_scale_m"] = scales
            for name, value in upper_arrays.items():
                arrays[f"{metric}_{name}"] = value

        strict_validation_nees = _manual_track_nees(
            prediction,
            posterior.variance,
            process_variance=config.process_std_m**2,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            gt_track_3d=gt_track_3d,
            track_initial_mask=track_initial_mask,
            track_indices=track_indices,
            start_frame=fit_end,
            end_frame=train_end,
        )
        strict_future_nees = _manual_track_nees(
            prediction,
            posterior.variance + validation_count * config.process_std_m**2,
            process_variance=config.process_std_m**2,
            lift_indices=lift_indices,
            lift_weights=lift_weights,
            gt_track_3d=gt_track_3d,
            track_initial_mask=track_initial_mask,
            track_indices=track_indices,
            start_frame=train_end,
            end_frame=frame_count,
        )
        arrays["strict_validation_nees_3d"] = strict_validation_nees
        arrays["strict_future_nees_3d"] = strict_future_nees
        nees: dict[str, object] = {
            "strict_validation_nees_3d": summarize_nees(strict_validation_nees),
            "strict_future_nees_3d": summarize_nees(strict_future_nees),
        }
        if anchor_run is not None:
            operational_values, operational_summary = _operational_anchor_nees(
                case,
                anchor_run,
                baseline,
                gt_track_3d,
                train_end_frame=train_end,
                frame_count=frame_count,
                track_initial_mask=track_initial_mask,
                track_indices=track_indices,
            )
            arrays["operational_future_nees_3d"] = operational_values
            nees["operational_future_nees_3d"] = operational_summary
        case_output = output / "cases" / case
        case_output.mkdir(parents=True, exist_ok=True)
        arrays_path = case_output / "calibration_arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        case_result: dict[str, object] = {
            "fit_end_frame": fit_end,
            "train_end_frame": train_end,
            "frame_count": frame_count,
            "calibration_frame_count": validation_count,
            "future_frame_count": frame_count - train_end,
            "future_point_metrics": {
                metric: {
                    "baseline_mean_m": float(
                        np.mean(baseline_values[metric][validation_count:])
                    ),
                    "strict_split_anchor_mean_m": float(
                        np.mean(metric_values[metric][validation_count:])
                    ),
                }
                for metric in METRICS
            },
            "conformal": conformal,
            "nees": nees,
            "outputs": {"arrays": str(arrays_path.resolve())},
        }
        case_summary_path = case_output / "summary.json"
        case_summary_path.write_text(
            json.dumps(case_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        case_result["outputs"]["summary"] = str(case_summary_path.resolve())
        case_results[case] = case_result
        case_arrays[case] = arrays

    def cohort_readout(cohort: tuple[str, ...]) -> dict[str, object]:
        nees_output = {
            "strict_validation_nees_3d": _aggregate_nees(
                cohort,
                case_results,
                case_arrays,
                "strict_validation_nees_3d",
            ),
            "strict_future_nees_3d": _aggregate_nees(
                cohort,
                case_results,
                case_arrays,
                "strict_future_nees_3d",
            ),
        }
        if anchor_run is not None:
            nees_output["operational_future_nees_3d"] = _aggregate_nees(
                cohort,
                case_results,
                case_arrays,
                "operational_future_nees_3d",
            )
            accepted = tuple(
                case
                for case in cohort
                if case_results[case]["nees"]["operational_future_nees_3d"][
                    "accepted_on_validation"
                ]
            )
            nees_output["operational_accepted_future_nees_3d"] = _aggregate_nees(
                accepted,
                case_results,
                case_arrays,
                "operational_future_nees_3d",
            )
            zero_process = tuple(
                case
                for case in cohort
                if case_results[case]["nees"]["operational_future_nees_3d"][
                    "process_std_m"
                ]
                == 0.0
            )
            positive_process = tuple(
                case for case in cohort if case not in zero_process
            )
            nees_output["operational_zero_process_future_nees_3d"] = _aggregate_nees(
                zero_process,
                case_results,
                case_arrays,
                "operational_future_nees_3d",
            )
            nees_output["operational_positive_process_future_nees_3d"] = (
                _aggregate_nees(
                    positive_process,
                    case_results,
                    case_arrays,
                    "operational_future_nees_3d",
                )
            )
        return {
            "cases": list(cohort),
            "case_count": len(cohort),
            "future_point_metrics": _aggregate_point_metrics(cohort, case_results),
            "conformal": _aggregate_conformal(
                cohort,
                case_results,
                coverage_levels=config.coverage_levels,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
            ),
            "nees": nees_output,
        }

    result: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_conformal_method": config.primary_conformal_method,
        "claim_boundary": specification["claim_contract"],
        "case_results": case_results,
        "development": cohort_readout(development) if development else None,
        "confirmation": cohort_readout(confirmation),
    }
    result_path = output / "phystwin_calibration_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
