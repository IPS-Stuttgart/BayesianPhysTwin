"""Locked, session-aware analyses for the Causal4D real protocol."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def _numeric_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _cluster_effects(
    differences: np.ndarray, cluster_ids: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    identifiers = [str(value) for value in cluster_ids]
    if len(identifiers) != len(differences) or any(not value for value in identifiers):
        raise ValueError("cluster_ids must match the observations")
    unique = list(dict.fromkeys(identifiers))
    if len(unique) < 2:
        raise ValueError("at least two independent clusters are required")
    effects = np.array(
        [
            np.mean(
                differences[
                    np.fromiter(
                        (value == identifier for value in identifiers),
                        dtype=bool,
                    )
                ]
            )
            for identifier in unique
        ],
        dtype=float,
    )
    return unique, effects


def cluster_paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    bootstrap_replicates: int = 20_000,
    seed: int = 20_260_712,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Estimate a paired candidate-minus-baseline effect by resampling sessions."""

    candidate_values = _numeric_vector(candidate, "candidate")
    baseline_values = _numeric_vector(baseline, "baseline")
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("candidate and baseline must have matching shapes")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    clusters, effects = _cluster_effects(
        candidate_values - baseline_values, cluster_ids
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(effects),
        size=(bootstrap_replicates, len(effects)),
    )
    bootstrap = np.mean(effects[samples], axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimand": "equal-session mean of paired candidate-minus-baseline effects",
        "execution_count": int(len(candidate_values)),
        "session_count": int(len(clusters)),
        "mean_difference": float(np.mean(effects)),
        "median_session_difference": float(np.median(effects)),
        "confidence_level": float(confidence_level),
        "bootstrap_interval": [float(lower), float(upper)],
        "bootstrap_replicates": int(bootstrap_replicates),
        "seed": int(seed),
        "probability_difference_below_zero": float(np.mean(bootstrap < 0.0)),
        "replication_unit": "session",
    }


def persistence_shrinkage_gate(
    nominal_correction_rms_m: Sequence[float],
    mechanism_correction_rms_m: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    bootstrap_replicates: int = 20_000,
    seed: int = 20_260_712,
) -> dict[str, Any]:
    """Test whether a physical mechanism shrinks the residual readout correction."""

    nominal = _numeric_vector(nominal_correction_rms_m, "nominal_correction_rms_m")
    mechanism = _numeric_vector(
        mechanism_correction_rms_m, "mechanism_correction_rms_m"
    )
    if nominal.shape != mechanism.shape:
        raise ValueError("correction magnitude vectors must match")
    if np.any(nominal <= 0.0) or np.any(mechanism <= 0.0):
        raise ValueError("correction RMS magnitudes must be strictly positive")
    log_ratio = np.log(mechanism / nominal)
    clusters, cluster_log_ratios = _cluster_effects(log_ratio, cluster_ids)
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(clusters),
        size=(bootstrap_replicates, len(clusters)),
    )
    bootstrap = np.mean(cluster_log_ratios[samples], axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    mean_log_ratio = float(np.mean(cluster_log_ratios))
    geometric_ratio = float(np.exp(mean_log_ratio))
    return {
        "metric": "graph_readout_correction_rms_m",
        "estimand": "equal-session mean log mechanism-to-nominal magnitude ratio",
        "execution_count": int(len(nominal)),
        "session_count": int(len(clusters)),
        "geometric_mean_ratio": geometric_ratio,
        "geometric_mean_shrinkage_fraction": float(1.0 - geometric_ratio),
        "log_ratio_interval_95": [float(lower), float(upper)],
        "ratio_interval_95": [float(np.exp(lower)), float(np.exp(upper))],
        "promotion_gate": "upper_95_percent_cluster_bootstrap_log_ratio_below_zero",
        "passed": bool(upper < 0.0),
        "bootstrap_replicates": int(bootstrap_replicates),
        "seed": int(seed),
        "replication_unit": "session",
    }


def cluster_robust_linear_regression(
    response: Sequence[float],
    features: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Fit OLS with a finite-sample-corrected session-cluster covariance."""

    y = _numeric_vector(response, "response")
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(y) or not np.all(np.isfinite(x)):
        raise ValueError("features must have finite shape (observation, feature)")
    names = [str(value) for value in feature_names]
    if len(names) != x.shape[1] or any(not value for value in names):
        raise ValueError("feature_names must name every feature")
    identifiers = [str(value) for value in cluster_ids]
    if len(identifiers) != len(y):
        raise ValueError("cluster_ids must match the observations")
    clusters = list(dict.fromkeys(identifiers))
    design = np.column_stack([np.ones(len(y)), x])
    parameter_names = ["intercept", *names]
    n, parameter_count = design.shape
    if n <= parameter_count or len(clusters) <= 1:
        raise ValueError("regression needs more observations and clusters")
    if np.linalg.matrix_rank(design) < parameter_count:
        raise ValueError("regression design is rank deficient")

    gram_inverse = np.linalg.inv(design.T @ design)
    coefficients = gram_inverse @ design.T @ y
    residual = y - design @ coefficients
    meat = np.zeros((parameter_count, parameter_count), dtype=float)
    for cluster in clusters:
        selected = np.fromiter(
            (identifier == cluster for identifier in identifiers), dtype=bool
        )
        score = design[selected].T @ residual[selected]
        meat += np.outer(score, score)
    correction = (len(clusters) / (len(clusters) - 1.0)) * (
        (n - 1.0) / (n - parameter_count)
    )
    covariance = correction * gram_inverse @ meat @ gram_inverse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0.0,
    )
    return {
        "method": "OLS with CR1 session-clustered covariance",
        "observation_count": int(n),
        "session_count": int(len(clusters)),
        "parameters": {
            name: {
                "coefficient": float(coefficient),
                "cluster_standard_error": float(standard_error),
                "t_statistic": (
                    float(t_statistic) if np.isfinite(t_statistic) else None
                ),
            }
            for name, coefficient, standard_error, t_statistic in zip(
                parameter_names,
                coefficients,
                standard_errors,
                t_statistics,
                strict=True,
            )
        },
        "claim_boundary": (
            "Mechanism-signature diagnostic on locked source data; coefficients "
            "are not independent-execution causal effects."
        ),
    }


def conformal_rank_plan(
    calibration_units: int, *, coverage: float = 0.90
) -> dict[str, Any]:
    """Return the finite-sample split-conformal order-statistic requirement."""

    if calibration_units < 1:
        raise ValueError("calibration_units must be positive")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    rank = int(math.ceil((calibration_units + 1) * coverage))
    minimum = 1
    while math.ceil((minimum + 1) * coverage) > minimum:
        minimum += 1
    return {
        "coverage": float(coverage),
        "calibration_units": int(calibration_units),
        "order_statistic_rank_one_based": rank,
        "finite_without_infinite_sentinel": bool(rank <= calibration_units),
        "minimum_calibration_units_for_finite_interval": int(minimum),
    }


def audit_base_protocol_power(protocol: dict[str, Any]) -> dict[str, Any]:
    """Expose pre-acquisition replication and calibration limits of protocol v1."""

    executions = list(protocol["executions"])
    exact_cells = Counter(
        (
            execution["contact_region_id"],
            execution["command_profile_id"],
            execution["realization_condition_id"],
        )
        for execution in executions
    )
    profiles = list(protocol["command_profiles"])
    calibration_counts = [
        len(fold["calibration_session_ids"])
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    ]
    minimum_calibration = min(calibration_counts)
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "exact_replication": {
            "maximum_same_contact_profile_condition_count": int(
                max(exact_cells.values())
            ),
            "cells_with_at_least_three_repeats": int(
                sum(count >= 3 for count in exact_cells.values())
            ),
            "passed": bool(any(count >= 3 for count in exact_cells.values())),
        },
        "signature_contrasts": {
            "distinct_outbound_durations_s": sorted(
                {float(profile["outbound_duration_s"]) for profile in profiles}
            ),
            "distinct_hold_durations_s": sorted(
                {float(profile["hold_duration_s"]) for profile in profiles}
            ),
            "independent_speed_contrast": bool(
                len({profile["outbound_duration_s"] for profile in profiles}) > 1
            ),
            "independent_hold_contrast": bool(
                len({profile["hold_duration_s"] for profile in profiles}) > 1
            ),
        },
        "calibration": {
            "minimum_independent_sessions_per_fold": int(minimum_calibration),
            "nominal_90_percent": conformal_rank_plan(
                minimum_calibration, coverage=0.90
            ),
        },
        "passed_for_first_locked_execution": False,
        "required_action": (
            "Apply the versioned pre-acquisition amendment; do not reinterpret "
            "the original 36-execution grid as adequately replicated or calibrated."
        ),
    }
