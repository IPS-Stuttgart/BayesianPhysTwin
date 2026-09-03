#!/usr/bin/env python3
"""Compare the frozen Deform360 belief with a source-residual bootstrap.

The exact successful 92-object bound-carrier dependence study is rerun first and
must reproduce its archived scientific result. The extension adds one source-only
nonparametric comparator. For each registered scalar query, source residuals are
weighted equally by source episode, centered, and rescaled to the full low-rank
arm's source-calibrated query variance. The two arms therefore share the exact
point mean and query variance; they differ only in the predictive distribution
used for query probabilities and execute-versus-fallback decisions.

This is retrospective mechanism evidence on public data. Target outcomes are used
only for final scoring, and complete physical objects are the inferential units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-bootstrap-comparator-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-bootstrap-comparator-protocol-v1"
OBJECT_COUNT = 92
PRIMARY_METRICS = ("event_brier", "decision_loss")
SECONDARY_METRICS = ("event_log_loss", "standardized_crps")
_EPS = 1e-12
_NORMAL = NormalDist()


@dataclass(frozen=True)
class WeightedEmpirical:
    values: np.ndarray
    weights: np.ndarray
    cumulative_weights: np.ndarray
    cumulative_weighted_values: np.ndarray


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_protocol(protocol: dict[str, Any], data_root: Path) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected bootstrap-comparator protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unexpected bootstrap-comparator protocol version")
    if protocol.get("status") != "frozen-before-execution":
        raise ValueError("bootstrap-comparator protocol is not frozen")
    if Path(str(protocol.get("dataset_root"))) != data_root:
        raise ValueError("dataset root changed")

    reference = protocol.get("reference_dependence_result")
    if not isinstance(reference, dict):
        raise ValueError("reference dependence-result binding is absent")
    if int(reference.get("workflow_run_id", -1)) != 33528032875:
        raise ValueError("reference workflow run changed")
    if int(reference.get("artifact_id", -1)) != 9811194776:
        raise ValueError("reference artifact changed")
    expected_result = (
        "c73659af65c2b87923f7bd668f9717afab03e449a5b3abd3a5b597ec60898fd1"
    )
    if reference.get("result_sha256") != expected_result:
        raise ValueError("reference scientific result changed")

    comparator = protocol.get("comparator")
    if not isinstance(comparator, dict):
        raise ValueError("comparator contract is absent")
    for key in (
        "source_only",
        "episode_balanced",
        "same_target_mean",
        "same_query_variance_as_full",
        "analytic_empirical_probabilities",
    ):
        if comparator.get(key) is not True:
            raise ValueError(f"required comparator property is disabled: {key}")
    if comparator.get("source_residual_semantics") != (
        "leave-one-source-episode-out-cv-residuals"
    ):
        raise ValueError("source residual semantics changed")

    statistics = protocol.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("statistics contract is absent")
    if int(statistics.get("object_bootstrap_repetitions", 0)) < 10_000:
        raise ValueError("too few object bootstrap repetitions")
    if tuple(statistics.get("co_primary_metrics", ())) != PRIMARY_METRICS:
        raise ValueError("co-primary metric roster changed")
    if tuple(statistics.get("secondary_metrics", ())) != SECONDARY_METRICS:
        raise ValueError("secondary metric roster changed")
    if float(statistics.get("familywise_confidence", math.nan)) != 0.95:
        raise ValueError("familywise confidence changed")
    if protocol.get("retrospective_target_reuse") is not True:
        raise ValueError("retrospective target reuse must be explicit")
    if protocol.get("new_measurements_collected") is not False:
        raise ValueError("new measurements are forbidden")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("protocol may not self-authorize a paper claim")


def normalized_weights(weights: np.ndarray) -> np.ndarray:
    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(vector))
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    return vector / total


def episode_balanced_weights(block_lengths: list[int]) -> np.ndarray:
    if not block_lengths or any(length <= 0 for length in block_lengths):
        raise ValueError("source episode blocks must be nonempty")
    episode_mass = 1.0 / len(block_lengths)
    return np.concatenate(
        [np.full(length, episode_mass / length) for length in block_lengths]
    )


def weighted_empirical(values: np.ndarray, weights: np.ndarray) -> WeightedEmpirical:
    vector = np.asarray(values, dtype=np.float64)
    normalized = normalized_weights(weights)
    if vector.ndim != 1 or vector.shape != normalized.shape:
        raise ValueError("values and weights must be aligned vectors")
    if not np.all(np.isfinite(vector)):
        raise ValueError("empirical values must be finite")
    order = np.argsort(vector, kind="mergesort")
    sorted_values = vector[order]
    sorted_weights = normalized[order]
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights[-1] = 1.0
    cumulative_weighted_values = np.cumsum(sorted_weights * sorted_values)
    return WeightedEmpirical(
        values=sorted_values,
        weights=sorted_weights,
        cumulative_weights=cumulative_weights,
        cumulative_weighted_values=cumulative_weighted_values,
    )


def moment_matched_distribution(
    source_values: np.ndarray,
    source_weights: np.ndarray,
    target_variance: float,
) -> tuple[WeightedEmpirical, dict[str, float]]:
    weights = normalized_weights(source_weights)
    values = np.asarray(source_values, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("source values and weights disagree")
    source_mean = float(np.sum(weights * values))
    centered = values - source_mean
    source_variance = float(np.sum(weights * centered * centered))
    if not np.isfinite(source_variance) or source_variance <= _EPS:
        raise ValueError("source query residual variance is degenerate")
    if not np.isfinite(target_variance) or target_variance <= _EPS:
        raise ValueError("target query variance must be positive")
    scale = math.sqrt(target_variance / source_variance)
    matched = centered * scale
    matched_mean = float(np.sum(weights * matched))
    matched_variance = float(np.sum(weights * matched * matched))
    return weighted_empirical(matched, weights), {
        "source_weighted_mean_before_centering": source_mean,
        "source_weighted_variance_before_scaling": source_variance,
        "applied_scale": scale,
        "matched_weighted_mean": matched_mean,
        "matched_weighted_variance": matched_variance,
        "target_variance": target_variance,
        "mean_parity_abs": abs(matched_mean),
        "variance_parity_abs": abs(matched_variance - target_variance),
    }


def empirical_cdf(
    distribution: WeightedEmpirical,
    thresholds: np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float64)
    indices = np.searchsorted(distribution.values, values, side=side)
    padded = np.concatenate(([0.0], distribution.cumulative_weights))
    return padded[indices]


def weighted_quantile(distribution: WeightedEmpirical, probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0,1]")
    index = int(
        np.searchsorted(
            distribution.cumulative_weights,
            probability,
            side="left",
        )
    )
    return float(distribution.values[min(index, len(distribution.values) - 1)])


def empirical_event_probability(
    distribution: WeightedEmpirical,
    means: np.ndarray,
    threshold: float,
    event: str,
) -> np.ndarray:
    center = np.asarray(means, dtype=np.float64)
    if event == "upper":
        return 1.0 - empirical_cdf(
            distribution,
            threshold - center,
            side="right",
        )
    if event == "absolute":
        below = empirical_cdf(
            distribution,
            -threshold - center,
            side="left",
        )
        above = 1.0 - empirical_cdf(
            distribution,
            threshold - center,
            side="right",
        )
        return below + above
    raise ValueError(f"unsupported event type: {event}")


def empirical_mean_absolute_error(
    distribution: WeightedEmpirical,
    targets: np.ndarray,
) -> np.ndarray:
    target = np.asarray(targets, dtype=np.float64)
    indices = np.searchsorted(distribution.values, target, side="right")
    padded_weight = np.concatenate(([0.0], distribution.cumulative_weights))
    padded_value = np.concatenate(
        ([0.0], distribution.cumulative_weighted_values)
    )
    left_weight = padded_weight[indices]
    left_value = padded_value[indices]
    total_value = float(distribution.cumulative_weighted_values[-1])
    return (
        target * left_weight
        - left_value
        + total_value
        - left_value
        - target * (1.0 - left_weight)
    )


def empirical_half_pair_distance(distribution: WeightedEmpirical) -> float:
    previous_weight = np.concatenate(
        ([0.0], distribution.cumulative_weights[:-1])
    )
    previous_value = np.concatenate(
        ([0.0], distribution.cumulative_weighted_values[:-1])
    )
    terms = distribution.weights * (
        distribution.values * previous_weight - previous_value
    )
    return float(np.sum(terms))


def normal_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.fromiter(
        (_NORMAL.cdf(float(value)) for value in flat),
        dtype=np.float64,
        count=len(flat),
    ).reshape(np.asarray(values).shape)


def gaussian_standardized_crps(standardized_errors: np.ndarray) -> float:
    z = np.asarray(standardized_errors, dtype=np.float64)
    cdf = normal_cdf(z)
    density = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    values = z * (2.0 * cdf - 1.0) + 2.0 * density - 1.0 / math.sqrt(
        math.pi
    )
    return float(np.mean(values))


def empirical_standardized_crps(
    distribution: WeightedEmpirical,
    target_errors: np.ndarray,
    standard_deviation: float,
) -> float:
    first = empirical_mean_absolute_error(distribution, target_errors)
    second = empirical_half_pair_distance(distribution)
    return float(np.mean(first - second) / standard_deviation)


def source_block_lengths(capture: Any) -> list[int]:
    if capture.source is None or not capture.candidates:
        raise RuntimeError("source capture is incomplete")
    first = capture.candidates[0]
    lengths = [
        len(first.cv_truths[episode.descriptor.episode_id])
        for episode in capture.source
    ]
    if capture.source_residuals is None:
        raise RuntimeError("source residual capture is absent")
    if sum(lengths) != len(capture.source_residuals):
        raise RuntimeError("source episode boundaries do not cover residuals")
    return lengths


def bootstrap_query_metrics(
    *,
    source_query_errors: np.ndarray,
    source_weights: np.ndarray,
    target_truth_query: np.ndarray,
    target_error_query: np.ndarray,
    target_variance: float,
    threshold: float,
    event: str,
    coverage_probability: float,
    fallback_cost: float,
    probability_clip: float,
) -> tuple[dict[str, float], dict[str, float]]:
    distribution, parity = moment_matched_distribution(
        source_query_errors,
        source_weights,
        target_variance,
    )
    target_mean = target_truth_query - target_error_query
    if event == "upper":
        labels = target_truth_query > threshold
    elif event == "absolute":
        labels = np.abs(target_truth_query) > threshold
    else:
        raise ValueError(f"unsupported event type: {event}")
    predicted = np.clip(
        empirical_event_probability(
            distribution,
            target_mean,
            threshold,
            event,
        ),
        probability_clip,
        1.0 - probability_clip,
    )
    labels_float = labels.astype(np.float64)
    execute = predicted <= fallback_cost
    realized_loss = np.where(execute, labels_float, fallback_cost)
    oracle_loss = np.where(labels, fallback_cost, 0.0)
    accepted = int(np.count_nonzero(execute))
    harmful = int(np.count_nonzero(execute & labels))
    absolute_distribution = weighted_empirical(
        np.abs(distribution.values),
        distribution.weights,
    )
    radius = weighted_quantile(absolute_distribution, coverage_probability)
    metrics = {
        "event_brier": float(np.mean(np.square(predicted - labels_float))),
        "event_log_loss": float(
            -np.mean(
                labels_float * np.log(predicted)
                + (1.0 - labels_float) * np.log1p(-predicted)
            )
        ),
        "decision_loss": float(np.mean(realized_loss)),
        "decision_regret": float(np.mean(realized_loss - oracle_loss)),
        "acceptance_fraction": float(np.mean(execute)),
        "harmful_accept_fraction_all": float(harmful / len(labels)),
        "harmful_accept_rate_given_accept": float(
            harmful / accepted if accepted else 0.0
        ),
        "target_90_coverage": float(
            np.mean(np.abs(target_error_query) <= radius)
        ),
        "mean_90_interval_width": float(2.0 * radius),
        "standardized_crps": empirical_standardized_crps(
            distribution,
            target_error_query,
            math.sqrt(target_variance),
        ),
    }
    return metrics, parity


def build_object_record(
    v6: Any,
    descriptors: list[Any],
    capture: Any,
    source_truth: np.ndarray,
    target_truth: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    object_id = str(descriptors[0].object_id)
    evaluation = protocol["reference_evaluation"]
    source_errors = np.asarray(capture.source_residuals, dtype=np.float64)
    target_errors = np.asarray(capture.target_errors, dtype=np.float64)
    if source_truth.shape != source_errors.shape:
        raise RuntimeError("source truth and residual shapes disagree")
    if target_truth.shape != target_errors.shape:
        raise RuntimeError("target truth and residual shapes disagree")
    block_lengths = source_block_lengths(capture)
    source_weights = episode_balanced_weights(block_lengths)
    centered_source = source_errors - source_errors.mean(axis=0, keepdims=True)
    queries: dict[str, Any] = {}
    max_mean_parity = 0.0
    max_variance_parity = 0.0

    for query_name, (weight, event) in v6.query_bank(target_truth.shape[1]).items():
        raw_variance = v6.covariance_query_variance(capture.covariance, weight)
        calibration = v6.source_query_calibration(
            centered_source,
            source_truth,
            weight,
            {name: raw_variance for name in v6.COVARIANCE_ARMS},
            event=event,
            probability=float(evaluation["coverage_probability"]),
            event_quantile=float(evaluation["event_threshold_quantile"]),
        )
        target_variance = raw_variance * float(
            calibration["shared_variance_scale"]
        )
        full = v6.query_metrics(
            centered_source_errors=centered_source,
            target_truth=target_truth,
            target_errors=target_errors,
            weight=weight,
            event=event,
            model=capture.covariance,
            calibration=calibration,
            fallback_cost=float(evaluation["fallback_cost"]),
            probability_clip=float(evaluation["probability_clip"]),
        )
        target_error_query = target_errors @ weight
        target_truth_query = target_truth @ weight
        bootstrap, parity = bootstrap_query_metrics(
            source_query_errors=source_errors @ weight,
            source_weights=source_weights,
            target_truth_query=target_truth_query,
            target_error_query=target_error_query,
            target_variance=target_variance,
            threshold=float(calibration["event_threshold"]),
            event=event,
            coverage_probability=float(evaluation["coverage_probability"]),
            fallback_cost=float(evaluation["fallback_cost"]),
            probability_clip=float(evaluation["probability_clip"]),
        )
        full_record = {
            key: float(full[key])
            for key in (
                "event_brier",
                "event_log_loss",
                "decision_loss",
                "decision_regret",
                "acceptance_fraction",
                "harmful_accept_fraction_all",
                "harmful_accept_rate_given_accept",
                "target_90_coverage",
                "mean_90_interval_width",
            )
        }
        full_record["standardized_crps"] = gaussian_standardized_crps(
            target_error_query / math.sqrt(target_variance)
        )
        queries[query_name] = {
            "event": event,
            "target_variance": target_variance,
            "event_threshold": float(calibration["event_threshold"]),
            "source_episode_count": len(block_lengths),
            "source_residual_count": len(source_errors),
            "full_low_rank": full_record,
            "source_residual_bootstrap": bootstrap,
            "moment_parity": parity,
        }
        max_mean_parity = max(max_mean_parity, parity["mean_parity_abs"])
        max_variance_parity = max(
            max_variance_parity,
            parity["variance_parity_abs"],
        )

    metric_names = (
        *PRIMARY_METRICS,
        *SECONDARY_METRICS,
        "decision_regret",
        "acceptance_fraction",
        "harmful_accept_fraction_all",
        "harmful_accept_rate_given_accept",
        "target_90_coverage",
        "mean_90_interval_width",
    )
    arm_summary = {
        arm: {
            metric: float(
                np.mean([queries[name][arm][metric] for name in queries])
            )
            for metric in metric_names
        }
        for arm in ("full_low_rank", "source_residual_bootstrap")
    }
    return {
        "object_id": object_id,
        "source_episode_count": len(block_lengths),
        "source_residual_count": len(source_errors),
        "target_window_count": len(target_errors),
        "same_target_mean": True,
        "max_query_mean_parity_abs": max_mean_parity,
        "max_query_variance_parity_abs": max_variance_parity,
        "queries": queries,
        "arm_summary": arm_summary,
    }


def stable_recovery_projection(value: dict[str, Any]) -> dict[str, Any]:
    projection = dict(value)
    for key in ("github_sha", "runner_name", "result_sha256"):
        projection.pop(key, None)
    return projection


def validate_reference_result(
    reference: dict[str, Any],
    current: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    expected = protocol["reference_dependence_result"]["result_sha256"]
    reference_unsigned = dict(reference)
    reference_supplied = reference_unsigned.pop("result_sha256", None)
    if canonical_digest(reference_unsigned) != reference_supplied:
        raise ValueError("reference result digest is invalid")
    if reference_supplied != expected:
        raise ValueError("reference result does not match frozen protocol")
    current_unsigned = dict(current)
    current_supplied = current_unsigned.pop("result_sha256", None)
    if canonical_digest(current_unsigned) != current_supplied:
        raise ValueError("recomputed dependence-result digest is invalid")
    reference_digest = canonical_digest(stable_recovery_projection(reference))
    current_digest = canonical_digest(stable_recovery_projection(current))
    if reference_digest != current_digest:
        raise RuntimeError("original dependence study did not reproduce exactly")
    return {
        "reference_result_sha256": reference_supplied,
        "reference_scientific_projection_sha256": reference_digest,
        "recomputed_scientific_projection_sha256": current_digest,
        "exact_scientific_reproduction": True,
    }


def verify_full_metrics(
    object_records: list[dict[str, Any]],
    current: dict[str, Any],
) -> None:
    by_object = {str(row["object_id"]): row for row in current["objects"]}
    if len(by_object) != len(object_records):
        raise RuntimeError("captured object roster differs from recovery result")
    metrics = (
        "event_brier",
        "event_log_loss",
        "decision_loss",
        "decision_regret",
        "acceptance_fraction",
        "harmful_accept_fraction_all",
        "harmful_accept_rate_given_accept",
        "target_90_coverage",
        "mean_90_interval_width",
    )
    for record in object_records:
        reference = by_object[record["object_id"]]
        for query_name, query in record["queries"].items():
            expected = reference["queries"][query_name]["arms"]["full_low_rank"]
            for metric in metrics:
                if query["full_low_rank"][metric] != expected[metric]:
                    raise RuntimeError(
                        f"full metric changed: {record['object_id']} "
                        f"{query_name} {metric}"
                    )


def bootstrap_interval(
    values: np.ndarray,
    repetitions: int,
    seed: int,
    confidence: float,
) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError("bootstrap input must be a nonempty vector")
    if len(vector) == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
    means = vector[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return [
        float(value)
        for value in np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    ]


def paired_summary(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    repetitions: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    differences = np.asarray(
        [
            row["arm_summary"]["full_low_rank"][metric]
            - row["arm_summary"]["source_residual_bootstrap"][metric]
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "metric": metric,
        "difference_semantics": "full_low_rank-minus-source_residual_bootstrap",
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "confidence": confidence,
        "object_bootstrap_interval": bootstrap_interval(
            differences,
            repetitions,
            seed,
            confidence,
        ),
        "object_wins": int(np.count_nonzero(differences < 0.0)),
        "object_ties": int(np.count_nonzero(differences == 0.0)),
        "object_losses": int(np.count_nonzero(differences > 0.0)),
        "worst_object_difference": float(np.max(differences)),
    }


def aggregate(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    integrity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    statistics = protocol["statistics"]
    repetitions = int(statistics["object_bootstrap_repetitions"])
    seed = int(statistics["random_seed"])
    familywise = float(statistics["familywise_confidence"])
    primary_confidence = 1.0 - (1.0 - familywise) / len(PRIMARY_METRICS)
    metric_names = (
        *PRIMARY_METRICS,
        *SECONDARY_METRICS,
        "decision_regret",
        "acceptance_fraction",
        "harmful_accept_fraction_all",
        "harmful_accept_rate_given_accept",
        "target_90_coverage",
        "mean_90_interval_width",
    )
    arm_summary = {
        arm: {
            metric: float(
                np.mean([row["arm_summary"][arm][metric] for row in rows])
            )
            for metric in metric_names
        }
        for arm in ("full_low_rank", "source_residual_bootstrap")
    }
    comparisons = {}
    for index, metric in enumerate((*PRIMARY_METRICS, *SECONDARY_METRICS)):
        confidence = primary_confidence if metric in PRIMARY_METRICS else 0.95
        comparisons[metric] = paired_summary(
            rows,
            metric,
            repetitions=repetitions,
            seed=seed + index,
            confidence=confidence,
        )
    max_mean_parity = max(row["max_query_mean_parity_abs"] for row in rows)
    max_variance_parity = max(
        row["max_query_variance_parity_abs"] for row in rows
    )
    tolerance = float(protocol["comparator"]["moment_parity_tolerance"])
    gates = {
        "exact_original_scientific_result_reproduced": bool(
            integrity["exact_scientific_reproduction"]
        ),
        "complete_92_object_roster": len(rows) == OBJECT_COUNT,
        "same_target_mean": all(row["same_target_mean"] for row in rows),
        "query_mean_parity": max_mean_parity <= tolerance,
        "query_variance_parity": max_variance_parity <= tolerance,
        "source_only_empirical_comparator": True,
    }
    bayesian_better = all(
        comparisons[metric]["object_bootstrap_interval"][1] < 0.0
        for metric in PRIMARY_METRICS
    )
    bootstrap_better = all(
        comparisons[metric]["object_bootstrap_interval"][0] > 0.0
        for metric in PRIMARY_METRICS
    )
    if bayesian_better:
        scientific_decision = "full-low-rank-better-on-both-co-primary-endpoints"
    elif bootstrap_better:
        scientific_decision = "bootstrap-better-on-both-co-primary-endpoints"
    else:
        scientific_decision = "mixed-or-not-simultaneously-distinguishable"
    summary = {
        "object_count": len(rows),
        "query_count_per_object": len(rows[0]["queries"]),
        "arm_summary": arm_summary,
        "comparisons": comparisons,
        "max_query_mean_parity_abs": max_mean_parity,
        "max_query_variance_parity_abs": max_variance_parity,
    }
    decision = {
        "integrity_gates": gates,
        "all_integrity_gates_passed": all(gates.values()),
        "scientific_decision": scientific_decision,
        "bayesian_better_on_both_co_primary_endpoints": bayesian_better,
        "bootstrap_better_on_both_co_primary_endpoints": bootstrap_better,
        "familywise_confidence": familywise,
        "bonferroni_individual_primary_confidence": primary_confidence,
        "paper_claim_authorized": False,
        "fresh_confirmation_authorized": False,
        "deployment_claim_authorized": False,
    }
    return summary, decision


def run(
    *,
    protocol_path: Path,
    recovery_runner_path: Path,
    base_runner_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    reference_result_path: Path,
    data_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    data_root = data_root.resolve(strict=True)
    validate_protocol(protocol, data_root)
    reference = read_json(reference_result_path)
    recovery = load_module(
        recovery_runner_path.resolve(strict=True),
        "deform360_bound_recovery_for_bootstrap_comparator",
    )
    object_records: list[dict[str, Any]] = []
    original_loader = recovery.load_module

    def capturing_loader(path: Path, name: str) -> Any:
        module = original_loader(path, name)
        if name != "deform360_dependence_query_v6_original":
            return module
        original_evaluate = module.evaluate_object_with_capture

        def capturing_evaluate(
            v3: Any,
            descriptors: list[Any],
            development: dict[str, Any],
            base_protocol: dict[str, Any],
            rng: np.random.Generator,
        ) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
            output = original_evaluate(
                v3,
                descriptors,
                development,
                base_protocol,
                rng,
            )
            _, capture, source_truth, target_truth = output
            object_records.append(
                build_object_record(
                    module,
                    descriptors,
                    capture,
                    source_truth,
                    target_truth,
                    protocol,
                )
            )
            return output

        module.evaluate_object_with_capture = capturing_evaluate
        return module

    recovery.load_module = capturing_loader
    try:
        _, current = recovery.run(
            base_runner_path=base_runner_path,
            protocol_path=Path(
                protocol["reference_dependence_result"]["protocol_path"]
            ),
            parent_protocol_path=parent_protocol_path,
            parent_result_path=parent_result_path,
            readiness_path=readiness_path,
            data_root=data_root,
            parent_control_root=parent_control_root,
            frozen_root=frozen_root,
        )
    finally:
        recovery.load_module = original_loader

    integrity = validate_reference_result(reference, current, protocol)
    verify_full_metrics(object_records, current)
    summary, decision = aggregate(object_records, protocol, integrity)
    if not decision["all_integrity_gates_passed"]:
        raise RuntimeError("bootstrap comparison integrity gate failed")
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root),
        "reference_dependence_result": protocol["reference_dependence_result"],
        "integrity": integrity,
        "information_boundary": {
            "retrospective_target_reuse": True,
            "exact_bound_carrier_roster_reused": True,
            "exact_frozen_point_predictor_reused": True,
            "exact_original_three_arm_result_reproduced": True,
            "bootstrap_source_residuals_only": True,
            "bootstrap_episode_balanced": True,
            "same_target_mean": True,
            "same_query_variance": True,
            "target_outcomes_used_for_comparator_construction": False,
            "unbound_numeric_payloads_opened": False,
            "new_measurements_collected": False,
        },
        "summary": summary,
        "decision": decision,
        "objects": object_records,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def make_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    lines = [
        "# Deform360 source-residual bootstrap comparator v1",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Queries per object: **{summary['query_count_per_object']}**",
        "- Original three-arm result reproduced exactly: **true**",
        "- Point mean matched: **true**",
        "- Query variance matched: **true**",
        "- Comparator: **episode-balanced out-of-episode source residuals**",
        f"- Scientific decision: **{decision['scientific_decision']}**",
        "",
        "## Object-balanced results",
        "",
        "| Arm | Brier | Decision loss | Log loss | std. CRPS | Acceptance | Harm/all |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("full_low_rank", "source_residual_bootstrap"):
        value = summary["arm_summary"][arm]
        lines.append(
            f"| `{arm}` | {value['event_brier']:.6g} | "
            f"{value['decision_loss']:.6g} | {value['event_log_loss']:.6g} | "
            f"{value['standardized_crps']:.6g} | "
            f"{value['acceptance_fraction']:.3%} | "
            f"{value['harmful_accept_fraction_all']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Paired object-level contrasts",
            "",
            "Differences are full low-rank minus residual bootstrap; negative favors the Bayesian arm.",
            "",
            "| Metric | Difference | Confidence | Object-bootstrap interval | W/T/L |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, value in summary["comparisons"].items():
        interval = value["object_bootstrap_interval"]
        lines.append(
            f"| `{metric}` | {value['mean_difference']:.6g} | "
            f"{value['confidence']:.1%} | "
            f"[{interval[0]:.6g}, {interval[1]:.6g}] | "
            f"{value['object_wins']}/{value['object_ties']}/"
            f"{value['object_losses']} |"
        )
    lines.extend(
        [
            "",
            "The co-primary intervals use a Bonferroni familywise-95% rule.",
            "This retrospective mechanism study does not authorize calibration,",
            "fresh-transfer, deployment-safety, or state-of-the-art claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_object_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "object_id",
        "source_episode_count",
        "source_residual_count",
        "target_window_count",
        "max_query_mean_parity_abs",
        "max_query_variance_parity_abs",
    ]
    for arm in ("full_low_rank", "source_residual_bootstrap"):
        for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
            fields.append(f"{arm}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {name: row[name] for name in fields[:6]}
            for arm in ("full_low_rank", "source_residual_bootstrap"):
                for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
                    output[f"{arm}_{metric}"] = row["arm_summary"][arm][metric]
            writer.writerow(output)


def self_test() -> None:
    source = np.asarray([-2.0, 0.0, 1.0, 3.0, 4.0])
    weights = episode_balanced_weights([2, 3])
    distribution, parity = moment_matched_distribution(source, weights, 4.0)
    mean = float(np.sum(distribution.weights * distribution.values))
    variance = float(
        np.sum(distribution.weights * distribution.values * distribution.values)
    )
    assert abs(mean) < 1e-12
    assert abs(variance - 4.0) < 1e-12
    assert parity["mean_parity_abs"] < 1e-12
    assert parity["variance_parity_abs"] < 1e-12
    probabilities = empirical_event_probability(
        distribution,
        np.asarray([-1.0, 0.0, 1.0]),
        0.5,
        "upper",
    )
    assert probabilities.shape == (3,)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    errors = np.asarray([-1.0, 0.0, 1.0])
    assert empirical_standardized_crps(distribution, errors, 2.0) >= 0.0
    assert gaussian_standardized_crps(errors / 2.0) >= 0.0
    print("bootstrap comparator self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--base-runner", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "protocol",
        "recovery_runner",
        "base_runner",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "reference_result",
        "data_root",
        "parent_control_root",
        "frozen_root",
        "output_json",
        "output_report",
        "output_csv",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    result = run(
        protocol_path=args.protocol,
        recovery_runner_path=args.recovery_runner,
        base_runner_path=args.base_runner,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
        reference_result_path=args.reference_result,
        data_root=args.data_root,
        parent_control_root=args.parent_control_root,
        frozen_root=args.frozen_root,
    )
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_object_csv(args.output_csv, result["objects"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(result["decision"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
