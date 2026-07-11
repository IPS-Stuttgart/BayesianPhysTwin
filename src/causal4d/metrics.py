"""Intervention, parameter-recovery, and uncertainty metrics."""

from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
from typing import Any

import numpy as np

from causal4d.baselines import ParameterPosterior, PredictiveDistribution
from causal4d.simulator import PARAMETER_NAMES, PhysicalParameters


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    return float(np.interp(probability, cumulative, sorted_values))


def _weighted_crps(values: np.ndarray, weights: np.ndarray, truth: float) -> float:
    first = float(np.sum(weights * np.abs(values - truth)))
    pairwise = np.abs(values[:, None] - values[None, :])
    second = 0.5 * float(np.sum(weights[:, None] * weights[None, :] * pairwise))
    return first - second


def posterior_ambiguity(posterior: ParameterPosterior) -> dict[str, float | int]:
    """Summarise posterior spread and cross-parameter confounding."""

    weights = posterior.weights
    positive = weights > 0.0
    entropy = -float(np.sum(weights[positive] * np.log(weights[positive])))
    normalized_entropy = entropy / np.log(weights.size) if weights.size > 1 else 0.0
    mean = posterior.mean
    centered = posterior.particles - mean[None, :]
    covariance = (centered * weights[:, None]).T @ centered
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = standard_deviation[:, None] * standard_deviation[None, :]
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 1e-15,
    )
    off_diagonal = np.abs(correlation - np.diag(np.diag(correlation)))
    return {
        "posterior_effective_sample_size": posterior.effective_sample_size,
        "posterior_normalized_entropy": float(normalized_entropy),
        "posterior_max_abs_parameter_correlation": float(np.max(off_diagonal)),
        "posterior_particles_above_uniform": int(
            np.sum(weights > (1.0 / weights.size))
        ),
    }


def parameter_recovery_rows(
    posterior: ParameterPosterior,
    truth: PhysicalParameters,
    *,
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Return one recovery/calibration row per inferred physical parameter."""

    tail = 0.5 * (1.0 - confidence_level)
    truth_values = truth.as_array()
    diagnostics = posterior_ambiguity(posterior)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(PARAMETER_NAMES):
        values = posterior.particles[:, index]
        lower = _weighted_quantile(values, posterior.weights, tail)
        upper = _weighted_quantile(values, posterior.weights, 1.0 - tail)
        estimate = float(posterior.mean[index])
        true_value = float(truth_values[index])
        rows.append(
            {
                "parameter": name,
                "truth": true_value,
                "posterior_mean": estimate,
                "absolute_error": abs(estimate - true_value),
                "relative_absolute_error": abs(estimate - true_value) / true_value,
                "interval_lower": lower,
                "interval_upper": upper,
                "interval_width": upper - lower,
                "covered": bool(lower <= true_value <= upper),
                "crps": _weighted_crps(values, posterior.weights, true_value),
                **diagnostics,
            }
        )
    return rows


def _horizon_rmse(error: np.ndarray) -> tuple[float, float, float]:
    chunks = np.array_split(error, 3, axis=0)
    return tuple(float(np.sqrt(np.mean(np.square(chunk)))) for chunk in chunks)  # type: ignore[return-value]


def _direction_error_degrees(mean: np.ndarray, truth: np.ndarray) -> float:
    predicted_direction = np.mean(mean[-1] - mean[0], axis=0)
    true_direction = np.mean(truth[-1] - truth[0], axis=0)
    denominator = np.linalg.norm(predicted_direction) * np.linalg.norm(true_direction)
    if denominator <= 1e-12:
        return (
            0.0
            if np.linalg.norm(predicted_direction - true_direction) <= 1e-12
            else 180.0
        )
    cosine = float(
        np.clip(predicted_direction @ true_direction / denominator, -1.0, 1.0)
    )
    return float(np.degrees(np.arccos(cosine)))


def intervention_metrics(
    prediction: PredictiveDistribution,
    truth: np.ndarray,
    *,
    confidence_level: float,
    gross_failure_threshold_m: float,
) -> dict[str, Any]:
    """Score counterfactual accuracy and marginal predictive calibration."""

    truth = np.asarray(truth, dtype=float)
    if truth.shape != prediction.mean.shape:
        raise ValueError("truth and prediction shapes differ")
    error = prediction.mean - truth
    node_distance = np.linalg.norm(error, axis=-1)
    motion_scale = float(np.sqrt(np.mean(np.square(truth - truth[0][None, ...]))))
    early, middle, late = _horizon_rmse(error)
    standard_deviation = np.sqrt(prediction.variance)
    z_score = NormalDist().inv_cdf(0.5 * (1.0 + confidence_level))
    if prediction.interval_lower is not None and prediction.interval_upper is not None:
        lower = prediction.interval_lower
        upper = prediction.interval_upper
    else:
        lower = prediction.mean - z_score * standard_deviation
        upper = prediction.mean + z_score * standard_deviation
    coverage = float(np.mean((truth >= lower) & (truth <= upper)))
    nees = float(np.mean(np.square(error) / prediction.variance))
    gaussian_nll = float(
        np.mean(
            0.5
            * (
                np.log(2.0 * np.pi * prediction.variance)
                + np.square(error) / prediction.variance
            )
        )
    )
    fde = float(np.mean(node_distance[-1]))
    return {
        "trajectory_rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "relative_intervention_rmse": float(
            np.sqrt(np.mean(np.square(error))) / max(motion_scale, 1e-12)
        ),
        "ade_m": float(np.mean(node_distance)),
        "fde_m": fde,
        "early_rmse_m": early,
        "middle_rmse_m": middle,
        "late_rmse_m": late,
        "direction_error_deg": _direction_error_degrees(prediction.mean, truth),
        "gross_failure": bool(fde > gross_failure_threshold_m),
        "coverage": coverage,
        "coverage_error": abs(coverage - confidence_level),
        "mean_interval_width_m": float(np.mean(upper - lower)),
        "nees": nees,
        "gaussian_nll": gaussian_nll,
    }


def aggregate_interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate with equal row weight; the protocol balances objects and seeds."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["world_condition"]))].append(row)
    metrics = (
        "trajectory_rmse_m",
        "relative_intervention_rmse",
        "ade_m",
        "fde_m",
        "early_rmse_m",
        "middle_rmse_m",
        "late_rmse_m",
        "direction_error_deg",
        "gross_failure",
        "coverage",
        "coverage_error",
        "mean_interval_width_m",
        "nees",
        "gaussian_nll",
    )
    output: list[dict[str, Any]] = []
    for (method, world), selected in sorted(groups.items()):
        row: dict[str, Any] = {
            "method": method,
            "world_condition": world,
            "case_count": len(selected),
            "object_count": len({item["object"] for item in selected}),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(
                np.mean([float(item[metric]) for item in selected])
            )
        output.append(row)
    return output


def aggregate_parameters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["parameter"])].append(row)
    output: list[dict[str, Any]] = []
    for parameter, selected in sorted(groups.items()):
        output.append(
            {
                "parameter": parameter,
                "case_count": len(selected),
                "mean_absolute_error": float(
                    np.mean([item["absolute_error"] for item in selected])
                ),
                "mean_relative_absolute_error": float(
                    np.mean([item["relative_absolute_error"] for item in selected])
                ),
                "mean_crps": float(np.mean([item["crps"] for item in selected])),
                "empirical_coverage": float(
                    np.mean([float(item["covered"]) for item in selected])
                ),
                "mean_interval_width": float(
                    np.mean([item["interval_width"] for item in selected])
                ),
            }
        )
    return output
