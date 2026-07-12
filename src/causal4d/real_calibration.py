"""Source-only affine variance calibration for real Causal4D predictions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from causal4d.contracts import PhysicalPosterior
from causal4d.physical_validation import physical_posterior_moments


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class RealCalibrationCase:
    """One independent real execution and its frozen predictive moments."""

    case_id: str
    action_id: str
    contact_region_id: str
    mean_m: np.ndarray
    variance_m2: np.ndarray
    truth_m: np.ndarray
    valid: np.ndarray
    start_frame: int
    node_group_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_m, dtype=float)
        variance = np.asarray(self.variance_m2, dtype=float)
        truth = np.asarray(self.truth_m, dtype=float)
        valid = np.asarray(self.valid, dtype=bool)
        if mean.ndim != 3 or mean.shape[2] != 3:
            raise ValueError("calibration trajectories must have shape (T, N, 3)")
        if variance.shape != mean.shape or truth.shape != mean.shape:
            raise ValueError("mean, variance, and truth must share a shape")
        if valid.shape == mean.shape:
            valid = np.all(valid, axis=2)
        if valid.shape != mean.shape[:2]:
            raise ValueError("valid must have shape (T, N) or (T, N, 3)")
        if not 0 <= self.start_frame < len(mean):
            raise ValueError("start_frame must lie inside the trajectory")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("predictive moments must be finite")
        if np.any(variance <= 0.0):
            raise ValueError("predictive variance must be positive")
        labels = self.node_group_labels
        if labels is not None and len(labels) != mean.shape[1]:
            raise ValueError("node_group_labels must identify every node")
        for value, name in (
            (self.case_id, "case_id"),
            (self.action_id, "action_id"),
            (self.contact_region_id, "contact_region_id"),
        ):
            if not value:
                raise ValueError(f"{name} must be nonempty")
        valid = valid & np.all(np.isfinite(truth), axis=2)
        valid[: self.start_frame] = False
        if not np.any(valid):
            raise ValueError("calibration case has no valid target point-frames")
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "truth_m", truth)
        object.__setattr__(self, "valid", valid)

    @property
    def coordinate_count(self) -> int:
        return int(3 * np.sum(self.valid))


@dataclass(frozen=True)
class AffineVarianceCalibration:
    """Frozen ``a * variance + b`` transformation fitted on source trials."""

    confidence_level: float
    nll_scale_a: float
    nll_floor_b_m2: float
    heldout_inflation: float
    scale_a: float
    floor_b_m2: float
    fit_case_ids: tuple[str, ...]
    calibration_case_ids: tuple[str, ...]
    calibration_trial_count: int
    minimum_calibration_trials: int
    claim_ready: bool
    calibration_unit: str = "independent execution with equal trial weight"
    dependence_warning: str = (
        "Point-frames within an execution are correlated and are not counted "
        "as independent calibration trials."
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        for value in (self.nll_scale_a, self.heldout_inflation, self.scale_a):
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError("calibration scales must be finite and positive")
        for value in (self.nll_floor_b_m2, self.floor_b_m2):
            if value < 0.0 or not np.isfinite(value):
                raise ValueError("calibration floors must be finite and nonnegative")
        if set(self.fit_case_ids) & set(self.calibration_case_ids):
            raise ValueError("fit and calibration executions must be disjoint")
        if self.calibration_trial_count != len(self.calibration_case_ids):
            raise ValueError("calibration trial count does not match case ids")
        if self.minimum_calibration_trials < 1:
            raise ValueError("minimum_calibration_trials must be positive")
        if self.claim_ready != (
            self.calibration_trial_count >= self.minimum_calibration_trials
        ):
            raise ValueError("claim_ready disagrees with the trial-count gate")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "confidence_level": self.confidence_level,
            "transform": {
                "formula": "variance_cal = a * variance_raw + b",
                "nll_scale_a": self.nll_scale_a,
                "nll_floor_b_m2": self.nll_floor_b_m2,
                "heldout_inflation": self.heldout_inflation,
                "scale_a": self.scale_a,
                "floor_b_m2": self.floor_b_m2,
            },
            "fit_case_ids": list(self.fit_case_ids),
            "calibration_case_ids": list(self.calibration_case_ids),
            "calibration_trial_count": self.calibration_trial_count,
            "minimum_calibration_trials": self.minimum_calibration_trials,
            "claim_ready": self.claim_ready,
            "calibration_unit": self.calibration_unit,
            "dependence_warning": self.dependence_warning,
        }

    @property
    def calibration_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict()).encode()).hexdigest()


def case_from_physical_posterior(
    posterior: PhysicalPosterior,
    truth_m: np.ndarray,
    valid: np.ndarray,
    *,
    start_frame: int,
    action_id: str | None = None,
    contact_region_id: str = "unregistered",
    node_group_labels: Sequence[str] | None = None,
) -> RealCalibrationCase:
    mean, variance = physical_posterior_moments(posterior)
    return RealCalibrationCase(
        case_id=posterior.context.case_id,
        action_id=action_id or posterior.context.u_cf.action_id,
        contact_region_id=contact_region_id,
        mean_m=mean,
        variance_m2=variance,
        truth_m=truth_m,
        valid=valid,
        start_frame=start_frame,
        node_group_labels=(
            None if node_group_labels is None else tuple(node_group_labels)
        ),
    )


def _mean_trial_nll(
    cases: Sequence[RealCalibrationCase],
    scale_a: float,
    floor_b_m2: float,
) -> float:
    values = []
    for case in cases:
        coordinate_valid = np.repeat(case.valid[:, :, None], 3, axis=2)
        residual = (case.mean_m - case.truth_m)[coordinate_valid]
        variance = scale_a * case.variance_m2[coordinate_valid] + floor_b_m2
        values.append(
            float(
                np.mean(
                    0.5
                    * (np.log(2.0 * np.pi * variance) + np.square(residual) / variance)
                )
            )
        )
    return float(np.mean(values))


def _fit_nll_affine(
    cases: Sequence[RealCalibrationCase],
) -> tuple[float, float, float]:
    mean_residual_variance = float(
        np.mean(
            [
                np.mean(np.square((case.mean_m - case.truth_m)[case.valid]))
                for case in cases
            ]
        )
    )
    scale_grid = np.geomspace(0.01, 100.0, 25)
    maximum_floor_std = max(np.sqrt(mean_residual_variance) * 3.0, 1e-3)
    floor_grid = np.concatenate(
        (
            [0.0],
            np.square(np.geomspace(1e-5, maximum_floor_std, 24)),
        )
    )
    best = (float("inf"), 1.0, 0.0)
    for scale in scale_grid:
        for floor in floor_grid:
            score = _mean_trial_nll(cases, float(scale), float(floor))
            candidate = (score, float(scale), float(floor))
            if candidate < best:
                best = candidate
    return best[1], best[2], best[0]


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="mergesort")
    selected_values = values[order]
    selected_weights = weights[order]
    cumulative = np.cumsum(selected_weights) / np.sum(selected_weights)
    index = min(
        int(np.searchsorted(cumulative, probability, side="left")), len(order) - 1
    )
    return float(selected_values[index])


def fit_affine_variance_calibration(
    fit_cases: Sequence[RealCalibrationCase],
    calibration_cases: Sequence[RealCalibrationCase],
    *,
    confidence_level: float = 0.90,
    minimum_calibration_trials: int = 10,
) -> tuple[AffineVarianceCalibration, dict[str, Any]]:
    """Fit on source executions and calibrate on disjoint source executions."""

    fit = tuple(fit_cases)
    calibration = tuple(calibration_cases)
    if not fit or not calibration:
        raise ValueError("fit and calibration each require source executions")
    fit_ids = tuple(case.case_id for case in fit)
    calibration_ids = tuple(case.case_id for case in calibration)
    if len(set(fit_ids)) != len(fit_ids) or len(set(calibration_ids)) != len(
        calibration_ids
    ):
        raise ValueError("source case ids must be unique within each split")
    if set(fit_ids) & set(calibration_ids):
        raise ValueError("fit and calibration source executions must be disjoint")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if minimum_calibration_trials < 1:
        raise ValueError("minimum_calibration_trials must be positive")

    nll_a, nll_b, fit_nll = _fit_nll_affine(fit)
    scores = []
    score_weights = []
    for case in calibration:
        coordinate_valid = np.repeat(case.valid[:, :, None], 3, axis=2)
        residual = np.abs((case.mean_m - case.truth_m)[coordinate_valid])
        variance = nll_a * case.variance_m2[coordinate_valid] + nll_b
        standardized = residual / np.sqrt(variance)
        scores.append(standardized)
        score_weights.append(
            np.full(len(standardized), 1.0 / len(standardized), dtype=float)
        )
    all_scores = np.concatenate(scores)
    all_weights = np.concatenate(score_weights)
    quantile = _weighted_quantile(all_scores, all_weights, confidence_level)
    gaussian_quantile = NormalDist().inv_cdf(0.5 * (1.0 + confidence_level))
    inflation = max(float(np.square(quantile / gaussian_quantile)), 1e-12)
    result = AffineVarianceCalibration(
        confidence_level=confidence_level,
        nll_scale_a=nll_a,
        nll_floor_b_m2=nll_b,
        heldout_inflation=inflation,
        scale_a=nll_a * inflation,
        floor_b_m2=nll_b * inflation,
        fit_case_ids=fit_ids,
        calibration_case_ids=calibration_ids,
        calibration_trial_count=len(calibration),
        minimum_calibration_trials=minimum_calibration_trials,
        claim_ready=len(calibration) >= minimum_calibration_trials,
    )
    diagnostics = {
        "fit_mean_trial_gaussian_nll": fit_nll,
        "calibration_equal_trial_weighted_standardized_quantile": quantile,
        "gaussian_reference_quantile": gaussian_quantile,
        "calibration_coordinate_count": int(
            sum(case.coordinate_count for case in calibration)
        ),
        "calibration_trial_count": len(calibration),
        "claim_ready": result.claim_ready,
        "blocking_reason": (
            None
            if result.claim_ready
            else (
                f"requires at least {minimum_calibration_trials} independent "
                f"calibration executions; received {len(calibration)}"
            )
        ),
    }
    return result, diagnostics


def apply_affine_variance_calibration(
    variance_m2: np.ndarray,
    calibration: AffineVarianceCalibration,
) -> np.ndarray:
    variance = np.asarray(variance_m2, dtype=float)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("raw variance must be finite and positive")
    return calibration.scale_a * variance + calibration.floor_b_m2


def _energy_score(
    mean: np.ndarray,
    variance: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int = 20260712,
    sample_count: int = 16,
) -> float:
    selected_mean = mean[valid]
    selected_std = np.sqrt(variance[valid])
    selected_truth = truth[valid]
    rng = np.random.default_rng(seed)
    first_noise = rng.standard_normal((sample_count, 1, 3))
    second_noise = rng.standard_normal((sample_count, 1, 3))
    first = selected_mean[None] + first_noise * selected_std[None]
    second = selected_mean[None] + second_noise * selected_std[None]
    return float(
        np.mean(np.linalg.norm(first - selected_truth[None], axis=2))
        - 0.5 * np.mean(np.linalg.norm(first - second, axis=2))
    )


def _metrics(
    case: RealCalibrationCase,
    variance: np.ndarray,
    selected: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, Any]:
    if not np.any(selected):
        raise ValueError("calibration metric group is empty")
    coordinate_valid = np.repeat(selected[:, :, None], 3, axis=2)
    residual = case.mean_m - case.truth_m
    selected_residual = residual[coordinate_valid]
    selected_variance = variance[coordinate_valid]
    z_score = NormalDist().inv_cdf(0.5 * (1.0 + confidence_level))
    lower = case.mean_m - z_score * np.sqrt(variance)
    upper = case.mean_m + z_score * np.sqrt(variance)
    covered = (case.truth_m >= lower) & (case.truth_m <= upper)
    calibration_curve = []
    for level in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
        quantile = NormalDist().inv_cdf(0.5 * (1.0 + level))
        curve_covered = np.abs(residual) <= quantile * np.sqrt(variance)
        calibration_curve.append(
            {
                "nominal": level,
                "empirical": float(np.mean(curve_covered[coordinate_valid])),
            }
        )
    return {
        "confidence_level": confidence_level,
        "valid_point_frames": int(np.sum(selected)),
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(selected_residual)))),
        "track_error_m": float(np.mean(np.linalg.norm(residual[selected], axis=1))),
        "coverage": float(np.mean(covered[coordinate_valid])),
        "mean_coordinate_nees": float(
            np.mean(np.square(selected_residual) / selected_variance)
        ),
        "mean_vector_nees": float(
            np.mean(np.sum(np.square(residual[selected]) / variance[selected], axis=1))
        ),
        "gaussian_nll": float(
            np.mean(
                0.5
                * (
                    np.log(2.0 * np.pi * selected_variance)
                    + np.square(selected_residual) / selected_variance
                )
            )
        ),
        "gaussian_energy_score_m": _energy_score(
            case.mean_m,
            variance,
            case.truth_m,
            selected,
        ),
        "mean_interval_width_m": float(np.mean((upper - lower)[coordinate_valid])),
        "calibration_curve": calibration_curve,
    }


def evaluate_real_prediction_case(
    case: RealCalibrationCase,
    *,
    variance_m2: np.ndarray | None = None,
    confidence_level: float = 0.90,
) -> dict[str, Any]:
    """Evaluate frozen moments by horizon and optional graph region."""

    variance = (
        case.variance_m2
        if variance_m2 is None
        else np.asarray(variance_m2, dtype=float)
    )
    if variance.shape != case.variance_m2.shape:
        raise ValueError("evaluation variance must match the calibration case")
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("evaluation variance must be finite and positive")
    groups: dict[str, np.ndarray] = {"all": case.valid.copy()}
    edges = np.linspace(case.start_frame, len(case.mean_m), 4, dtype=int)
    for index, name in enumerate(("early", "middle", "late")):
        selected = np.zeros_like(case.valid)
        selected[edges[index] : edges[index + 1]] = case.valid[
            edges[index] : edges[index + 1]
        ]
        groups[f"horizon:{name}"] = selected
    if case.node_group_labels is not None:
        labels = np.asarray(case.node_group_labels)
        for label in sorted(set(case.node_group_labels)):
            selected = case.valid.copy()
            selected[:, labels != label] = False
            if np.any(selected):
                groups[f"graph_region:{label}"] = selected
    metrics = {
        name: _metrics(
            case,
            variance,
            mask,
            confidence_level=confidence_level,
        )
        for name, mask in groups.items()
    }
    return {
        "groups": metrics,
        "worst_group_coverage": min(value["coverage"] for value in metrics.values()),
    }


def evaluate_real_calibration_case(
    case: RealCalibrationCase,
    calibration: AffineVarianceCalibration,
) -> dict[str, Any]:
    """Evaluate frozen raw and calibrated uncertainty by horizon and node group."""

    calibrated_variance = apply_affine_variance_calibration(
        case.variance_m2,
        calibration,
    )
    raw_result = evaluate_real_prediction_case(
        case,
        confidence_level=calibration.confidence_level,
    )
    calibrated_result = evaluate_real_prediction_case(
        case,
        variance_m2=calibrated_variance,
        confidence_level=calibration.confidence_level,
    )
    raw = raw_result["groups"]
    calibrated = calibrated_result["groups"]
    return {
        "case_id": case.case_id,
        "action_id": case.action_id,
        "contact_region_id": case.contact_region_id,
        "calibration_id": calibration.calibration_id,
        "calibration_claim_ready": calibration.claim_ready,
        "raw": raw,
        "calibrated": calibrated,
        "worst_group_coverage": {
            "raw": raw_result["worst_group_coverage"],
            "calibrated": calibrated_result["worst_group_coverage"],
        },
    }


def save_affine_variance_calibration(
    path: str | Path,
    calibration: AffineVarianceCalibration,
    diagnostics: dict[str, Any],
) -> None:
    payload = calibration.as_dict()
    payload["calibration_id"] = calibration.calibration_id
    payload["diagnostics"] = diagnostics
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_affine_variance_calibration(
    path: str | Path,
) -> AffineVarianceCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported affine calibration schema")
    transform = payload["transform"]
    calibration = AffineVarianceCalibration(
        confidence_level=float(payload["confidence_level"]),
        nll_scale_a=float(transform["nll_scale_a"]),
        nll_floor_b_m2=float(transform["nll_floor_b_m2"]),
        heldout_inflation=float(transform["heldout_inflation"]),
        scale_a=float(transform["scale_a"]),
        floor_b_m2=float(transform["floor_b_m2"]),
        fit_case_ids=tuple(map(str, payload["fit_case_ids"])),
        calibration_case_ids=tuple(map(str, payload["calibration_case_ids"])),
        calibration_trial_count=int(payload["calibration_trial_count"]),
        minimum_calibration_trials=int(payload["minimum_calibration_trials"]),
        claim_ready=bool(payload["claim_ready"]),
        calibration_unit=str(payload["calibration_unit"]),
        dependence_warning=str(payload["dependence_warning"]),
    )
    if payload.get("calibration_id") != calibration.calibration_id:
        raise ValueError("affine calibration checksum mismatch")
    return calibration
