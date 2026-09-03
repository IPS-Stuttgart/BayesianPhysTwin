#!/usr/bin/env python3
"""Audit whether Deform360 query copulas add value at fixed query marginals.

The completed v6 study held the predictive field mean and every field-coordinate
marginal fixed while changing dependence. This stricter retrospective audit
intercepts the exact bound-carrier v6 reexecution, projects its full covariance
to the five already registered physical queries, and constructs three empirical
query beliefs:

* the Gaussian copula induced by the full projected covariance;
* an independently permuted copula; and
* a deterministic wrong structured copula.

For each query, all arms contain the exact same sorted residual samples. Thus the
complete empirical univariate query distribution, every single-query event
probability, and the predictive query mean are identical. Only the coupling
between queries changes. All ten unordered query pairs are scored with both
conjunction and disjunction events, using the previously source-only thresholds
and the existing execute-versus-fallback loss.

The target cohort was opened before this audit was designed. The result is a
retrospective mechanism audit, not fresh confirmation.
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
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-query-copula-audit-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-query-copula-audit-protocol-v1"
ARMS = (
    "full_query_copula",
    "independent_query_copula",
    "scrambled_query_copula",
)
METRICS = (
    "event_brier",
    "event_log_loss",
    "decision_loss",
    "decision_regret",
    "acceptance_fraction",
    "harmful_accept_fraction_all",
    "harmful_accept_rate_given_accept",
    "event_rate",
    "mean_predicted_probability",
)
_PRIMARY_METRICS = ("event_brier", "decision_loss")
_EPS = 1e-12
_NORMAL = NormalDist()


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
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_seed(seed: int, object_id: str, purpose: str) -> int:
    payload = f"{seed}\0{object_id}\0{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected copula-audit protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unexpected copula-audit protocol version")
    if protocol.get("status") != "frozen-before-first-copula-audit-execution":
        raise ValueError("copula-audit protocol is not frozen")

    lineage = protocol.get("lineage")
    evaluation = protocol.get("evaluation")
    boundary = protocol.get("information_boundary")
    interpretation = protocol.get("pre_execution_interpretation_rule")
    if not all(isinstance(value, dict) for value in (lineage, evaluation, boundary)):
        raise ValueError("protocol lineage, evaluation, or boundary is absent")
    if not isinstance(interpretation, dict):
        raise ValueError("pre-execution interpretation rule is absent")

    query_order = tuple(str(value) for value in evaluation.get("query_order", ()))
    expected_queries = (
        "total_load",
        "sensor_imbalance",
        "horizontal_balance",
        "vertical_balance",
        "center_periphery",
    )
    if query_order != expected_queries:
        raise ValueError("query order changed")
    if evaluation.get("event_bank") != (
        "all ten unordered query pairs, each scored as conjunction and disjunction"
    ):
        raise ValueError("composite event bank changed")
    if tuple(evaluation.get("arms", ())) != ARMS:
        raise ValueError("copula arm roster changed")

    sample_count = int(evaluation.get("sample_count", -1))
    if sample_count < 4096 or sample_count % 2:
        raise ValueError("sample count must be even and at least 4096")
    fallback_cost = float(evaluation.get("fallback_cost", math.nan))
    if not 0.0 < fallback_cost < 1.0:
        raise ValueError("fallback cost must be in (0,1)")
    probability_clip = float(evaluation.get("probability_clip", math.nan))
    if not 0.0 < probability_clip < 0.01:
        raise ValueError("probability clip is invalid")
    if int(evaluation.get("bootstrap_repetitions", -1)) < 1000:
        raise ValueError("too few object-bootstrap repetitions")

    permutation = tuple(
        int(value) for value in evaluation.get("scramble_permutation", ())
    )
    signs = tuple(int(value) for value in evaluation.get("scramble_signs", ()))
    if sorted(permutation) != list(range(len(query_order))):
        raise ValueError("scramble permutation is invalid")
    if any(index == value for index, value in enumerate(permutation)):
        raise ValueError("scramble permutation must be a derangement")
    if len(signs) != len(query_order) or any(value not in (-1, 1) for value in signs):
        raise ValueError("scramble signs are invalid")

    if boundary.get("retrospective_target_reuse") is not True:
        raise ValueError("retrospective reuse must be explicit")
    if boundary.get("target_open_before_protocol_design") is not True:
        raise ValueError("post-target protocol timing must be explicit")
    if boundary.get("new_measurements_collected") is not False:
        raise ValueError("new measurements are forbidden")
    if boundary.get("target_outcomes_may_select_event_pairs") is not False:
        raise ValueError("target-driven event selection is forbidden")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("the audit may not self-authorize a paper claim")


def validate_reference_result(
    result: dict[str, Any],
    result_path: Path,
    protocol: dict[str, Any],
) -> None:
    binding = protocol["lineage"]["completed_dependence_result"]
    if sha256_file(result_path) != binding["result_json_sha256"]:
        raise ValueError("completed dependence result bytes changed")
    if result.get("schema") != (
        "bayesian-phystwin/deform360-dependence-query-result-v6-"
        "bound-carrier-recovery-v1"
    ):
        raise ValueError("unexpected completed dependence result schema")
    if result.get("status") != "complete":
        raise ValueError("completed dependence result is incomplete")
    if result.get("result_sha256") != binding["embedded_result_sha256"]:
        raise ValueError("completed dependence embedded digest changed")
    if int(result["summary"]["object_count"]) != 92:
        raise ValueError("completed dependence object roster changed")


def query_covariance(model: Any, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("query weights must be a matrix")
    multiplier = float(np.asarray(model.multiplier))
    diagonal = multiplier * np.asarray(model.diagonal, dtype=np.float64)
    factor = math.sqrt(multiplier) * np.asarray(model.factor, dtype=np.float64)
    if matrix.shape[1] != len(diagonal):
        raise ValueError("query weights and covariance dimension disagree")
    result = (matrix * diagonal[None, :]) @ matrix.T
    if factor.shape[1]:
        projected = matrix @ factor
        result += projected @ projected.T
    result = 0.5 * (result + result.T)
    eigenvalues = np.linalg.eigvalsh(result)
    tolerance = 1e-10 * max(float(np.max(np.diag(result))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("projected query covariance is not positive semidefinite")
    return result


def covariance_to_correlation(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(covariance, dtype=np.float64)
    standard_deviations = np.sqrt(np.maximum(np.diag(matrix), _EPS))
    correlation = matrix / np.outer(standard_deviations, standard_deviations)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if float(np.min(eigenvalues)) < -1e-9:
        raise ValueError("query correlation is not positive semidefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    correlation = (eigenvectors * clipped) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(correlation), _EPS))
    correlation = correlation / np.outer(diagonal, diagonal)
    np.fill_diagonal(correlation, 1.0)
    return correlation, standard_deviations


def exact_standard_latents(count: int, dimension: int, seed: int) -> np.ndarray:
    if count < 2 * dimension or count % 2:
        raise ValueError("latent sample count must be even and sufficiently large")
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((count // 2, dimension))
    values = np.concatenate((half, -half), axis=0)
    moment = values.T @ values / count
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    if float(np.min(eigenvalues)) <= _EPS:
        raise ValueError("latent whitening matrix is singular")
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    values = values @ inverse_root
    if float(np.max(np.abs(values.mean(axis=0)))) > 1e-12:
        raise RuntimeError("latent antithetic mean is not zero")
    observed = values.T @ values / count
    if not np.allclose(observed, np.eye(dimension), rtol=1e-11, atol=1e-11):
        raise RuntimeError("latent whitening failed")
    return values


def normal_score_grid(count: int) -> np.ndarray:
    probabilities = (np.arange(count, dtype=np.float64) + 0.5) / count
    scores = np.fromiter(
        (_NORMAL.inv_cdf(float(probability)) for probability in probabilities),
        dtype=np.float64,
        count=count,
    )
    if not np.all(np.isfinite(scores)) or not np.all(np.diff(scores) > 0.0):
        raise RuntimeError("normal score grid is invalid")
    return scores


def gaussian_copula_samples(
    correlation: np.ndarray,
    standard_deviations: np.ndarray,
    *,
    count: int,
    seed: int,
    scores: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    correlation = np.asarray(correlation, dtype=np.float64)
    standard_deviations = np.asarray(standard_deviations, dtype=np.float64)
    dimension = len(standard_deviations)
    if correlation.shape != (dimension, dimension):
        raise ValueError("correlation and marginal dimensions disagree")
    if scores.shape != (count,):
        raise ValueError("normal score grid length changed")

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if float(np.min(eigenvalues)) < -1e-9:
        raise ValueError("correlation matrix is not positive semidefinite")
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    latent = exact_standard_latents(count, dimension, seed) @ root.T
    samples = np.empty_like(latent)
    for column in range(dimension):
        order = np.argsort(latent[:, column], kind="mergesort")
        samples[order, column] = scores * standard_deviations[column]

    empirical_correlation = np.corrcoef(samples, rowvar=False)
    correlation_error = float(np.max(np.abs(empirical_correlation - correlation)))
    variance_ratio = np.var(samples, axis=0) / np.square(standard_deviations)
    variance_error = float(np.max(np.abs(variance_ratio - 1.0)))
    return samples, correlation_error, variance_error


def independent_copula(samples: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    result = np.empty_like(values)
    identity = np.arange(len(values))
    previous: np.ndarray | None = None
    for column in range(values.shape[1]):
        permutation = rng.permutation(len(values))
        if np.array_equal(permutation, identity):
            permutation = np.roll(permutation, column + 1)
        if previous is not None and np.array_equal(permutation, previous):
            permutation = np.roll(permutation, 1)
        result[:, column] = values[permutation, column]
        previous = permutation
    return result


def scrambled_copula(
    samples: np.ndarray,
    *,
    permutation: tuple[int, ...],
    signs: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    dimension = values.shape[1]
    if sorted(permutation) != list(range(dimension)):
        raise ValueError("scramble permutation is invalid")
    if any(index == value for index, value in enumerate(permutation)):
        raise ValueError("scramble permutation must be a derangement")
    if len(signs) != dimension or any(value not in (-1, 1) for value in signs):
        raise ValueError("scramble signs are invalid")

    result = np.empty_like(values)
    for target_column, carrier_column in enumerate(permutation):
        carrier_order = np.argsort(values[:, carrier_column], kind="mergesort")
        marginal_values = np.sort(values[:, target_column], kind="mergesort")
        if signs[target_column] < 0:
            marginal_values = marginal_values[::-1]
        result[carrier_order, target_column] = marginal_values
    return result


def marginal_parity_max_abs(
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> float:
    expected = np.sort(np.asarray(reference), axis=0)
    maximum = 0.0
    for name, value in candidates.items():
        observed = np.sort(np.asarray(value), axis=0)
        if observed.shape != expected.shape:
            raise ValueError(f"sample shape changed for {name}")
        maximum = max(maximum, float(np.max(np.abs(observed - expected))))
    return maximum


def query_event(values: np.ndarray, threshold: float, event: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if event == "upper":
        return array > threshold
    if event == "absolute":
        return np.abs(array) > threshold
    raise ValueError(f"unsupported query event: {event}")


def composite_event_bank(query_names: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left in range(len(query_names)):
        for right in range(left + 1, len(query_names)):
            for operator in ("and", "or"):
                result.append(
                    {
                        "name": (
                            f"{query_names[left]}__{operator}__{query_names[right]}"
                        ),
                        "left_index": left,
                        "right_index": right,
                        "left_query": query_names[left],
                        "right_query": query_names[right],
                        "operator": operator,
                    }
                )
    return result


def probability_metrics(
    predicted_probability: np.ndarray,
    labels: np.ndarray,
    *,
    fallback_cost: float,
    probability_clip: float,
) -> dict[str, float]:
    predicted = np.clip(
        np.asarray(predicted_probability, dtype=np.float64),
        probability_clip,
        1.0 - probability_clip,
    )
    label = np.asarray(labels, dtype=bool)
    label_float = label.astype(np.float64)
    execute = predicted <= fallback_cost
    realized_loss = np.where(execute, label_float, fallback_cost)
    oracle_loss = np.where(label, fallback_cost, 0.0)
    accepted = int(np.count_nonzero(execute))
    harmful = int(np.count_nonzero(execute & label))
    return {
        "event_brier": float(np.mean(np.square(predicted - label_float))),
        "event_log_loss": float(
            -np.mean(
                label_float * np.log(predicted)
                + (1.0 - label_float) * np.log1p(-predicted)
            )
        ),
        "decision_loss": float(np.mean(realized_loss)),
        "decision_regret": float(np.mean(realized_loss - oracle_loss)),
        "acceptance_fraction": float(np.mean(execute)),
        "harmful_accept_fraction_all": float(harmful / len(label)),
        "harmful_accept_rate_given_accept": float(
            harmful / accepted if accepted else 0.0
        ),
        "event_rate": float(np.mean(label_float)),
        "mean_predicted_probability": float(np.mean(predicted)),
    }


def evaluate_copula_arms(
    *,
    target_mean: np.ndarray,
    target_truth: np.ndarray,
    residual_samples: dict[str, np.ndarray],
    thresholds: np.ndarray,
    query_events: tuple[str, ...],
    query_names: tuple[str, ...],
    fallback_cost: float,
    probability_clip: float,
) -> tuple[dict[str, Any], float]:
    mean = np.asarray(target_mean, dtype=np.float64)
    truth = np.asarray(target_truth, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if mean.shape != truth.shape:
        raise ValueError("target query mean and truth shapes disagree")
    if mean.shape[1] != len(query_names) or thresholds.shape != (len(query_names),):
        raise ValueError("query dimensions changed")

    truth_events = [
        query_event(truth[:, index], thresholds[index], query_events[index])
        for index in range(len(query_names))
    ]
    bank = composite_event_bank(query_names)
    arm_records: dict[str, Any] = {}
    single_probabilities: dict[str, np.ndarray] = {}
    for arm_name in ARMS:
        samples = np.asarray(residual_samples[arm_name], dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != len(query_names):
            raise ValueError(f"invalid residual sample matrix for {arm_name}")
        sample_events = [
            query_event(
                mean[:, index, None] + samples[None, :, index],
                thresholds[index],
                query_events[index],
            )
            for index in range(len(query_names))
        ]
        single_probabilities[arm_name] = np.stack(
            [np.mean(value, axis=1) for value in sample_events],
            axis=1,
        )

        events: dict[str, dict[str, float]] = {}
        for specification in bank:
            left = int(specification["left_index"])
            right = int(specification["right_index"])
            if specification["operator"] == "and":
                sampled_event = sample_events[left] & sample_events[right]
                label = truth_events[left] & truth_events[right]
            else:
                sampled_event = sample_events[left] | sample_events[right]
                label = truth_events[left] | truth_events[right]
            predicted = np.mean(sampled_event, axis=1)
            events[str(specification["name"])] = probability_metrics(
                predicted,
                label,
                fallback_cost=fallback_cost,
                probability_clip=probability_clip,
            )

        summary = {
            metric: float(np.mean([value[metric] for value in events.values()]))
            for metric in METRICS
        }
        operator_summary = {
            operator: {
                metric: float(
                    np.mean(
                        [
                            events[str(specification["name"])][metric]
                            for specification in bank
                            if specification["operator"] == operator
                        ]
                    )
                )
                for metric in METRICS
            }
            for operator in ("and", "or")
        }
        arm_records[arm_name] = {
            "summary": summary,
            "operator_summary": operator_summary,
            "events": events,
        }

    reference = single_probabilities[ARMS[0]]
    parity = max(
        float(np.max(np.abs(single_probabilities[arm] - reference)))
        for arm in ARMS[1:]
    )
    return arm_records, parity


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
    means = vector[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_summary(
    rows: list[dict[str, Any]],
    metric: str,
    comparator: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [
            row["arms"][ARMS[0]]["summary"][metric]
            - row["arms"][comparator]["summary"][metric]
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "metric": metric,
        "comparator": comparator,
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "object_bootstrap_95_interval": bootstrap_interval(
            differences,
            repetitions,
            seed,
        ),
        "object_wins": int(np.count_nonzero(differences < 0.0)),
        "object_ties": int(np.count_nonzero(differences == 0.0)),
        "object_losses": int(np.count_nonzero(differences > 0.0)),
        "worst_object_difference": float(np.max(differences)),
    }


def aggregate(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    reference_reproduced: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = protocol["evaluation"]
    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["bootstrap_seed"])
    arm_summary = {
        arm: {
            metric: float(
                np.mean([row["arms"][arm]["summary"][metric] for row in rows])
            )
            for metric in METRICS
        }
        for arm in ARMS
    }
    operator_summary = {
        operator: {
            arm: {
                metric: float(
                    np.mean(
                        [
                            row["arms"][arm]["operator_summary"][operator][metric]
                            for row in rows
                        ]
                    )
                )
                for metric in METRICS
            }
            for arm in ARMS
        }
        for operator in ("and", "or")
    }
    comparisons: dict[str, Any] = {}
    for comparator_index, comparator in enumerate(ARMS[1:], start=1):
        comparisons[comparator] = {
            metric: paired_summary(
                rows,
                metric,
                comparator,
                repetitions=repetitions,
                seed=seed + 1000 * comparator_index + metric_index,
            )
            for metric_index, metric in enumerate(
                ("event_brier", "decision_loss", "event_log_loss")
            )
        }

    marginal_parity = float(
        max(row["query_sample_marginal_parity_max_abs"] for row in rows)
    )
    probability_parity = float(
        max(row["single_query_event_probability_parity_max_abs"] for row in rows)
    )
    correlation_error = float(
        max(row["full_copula_correlation_approximation_max_abs"] for row in rows)
    )
    variance_error = float(
        max(
            row["normal_score_marginal_variance_relative_error_max_abs"]
            for row in rows
        )
    )
    interpretation = protocol["pre_execution_interpretation_rule"]
    require_both = bool(
        interpretation[
            "both_primary_bootstrap_upper_bounds_below_zero_against_each_control"
        ]
    )
    endpoint_support = {
        metric: all(
            comparisons[comparator][metric]["object_bootstrap_95_interval"][1] < 0.0
            for comparator in ARMS[1:]
        )
        for metric in _PRIMARY_METRICS
    }
    value_supported = (
        all(endpoint_support.values())
        if require_both
        else any(endpoint_support.values())
    )
    gates = {
        "complete_92_object_roster": len(rows) == 92,
        "completed_v6_scientific_result_reproduced": reference_reproduced,
        "predictive_query_means_identical_by_construction": True,
        "empirical_univariate_query_marginals_exactly_identical": (
            marginal_parity == 0.0
        ),
        "single_query_event_probabilities_exactly_identical": (
            probability_parity == 0.0
        ),
        "full_copula_sampling_correlation_within_tolerance": (
            correlation_error
            <= float(evaluation["maximum_correlation_approximation_error"])
        ),
        "normal_score_marginal_variance_within_tolerance": (
            variance_error <= float(evaluation["maximum_marginal_variance_error"])
        ),
        "full_brier_bootstrap_upper_bound_below_zero_against_each_control": (
            endpoint_support["event_brier"]
        ),
        "full_decision_bootstrap_upper_bound_below_zero_against_each_control": (
            endpoint_support["decision_loss"]
        ),
    }
    summary = {
        "object_count": len(rows),
        "query_count": len(evaluation["query_order"]),
        "composite_event_count": len(
            composite_event_bank(tuple(evaluation["query_order"]))
        ),
        "sample_count_per_object": int(evaluation["sample_count"]),
        "arms": list(ARMS),
        "arm_summary": arm_summary,
        "operator_summary": operator_summary,
        "comparisons": comparisons,
        "query_sample_marginal_parity_max_abs": marginal_parity,
        "single_query_event_probability_parity_max_abs": probability_parity,
        "full_copula_correlation_approximation_max_abs": correlation_error,
        "normal_score_marginal_variance_relative_error_max_abs": variance_error,
    }
    if value_supported:
        outcome = "positive-copula-value-audit"
    elif any(endpoint_support.values()):
        outcome = "mixed-copula-value-audit"
    else:
        outcome = "negative-or-inconclusive-copula-value-audit"
    decision = {
        "outcome": outcome,
        "copula_value_supported": value_supported,
        "endpoint_support": endpoint_support,
        "gates": gates,
        "retrospective_mechanism_evidence_only": True,
        "fresh_confirmation_authorized": False,
        "paper_claim_authorized": False,
    }
    return summary, decision


def scientific_projection(result: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for row in result["objects"]:
        projected = dict(row)
        projected.pop("bound_carrier_recovery", None)
        objects.append(projected)
    return {
        "protocol_id": result["protocol_id"],
        "parent_confirmation": result["parent_confirmation"],
        "bound_selection_manifest_sha256": result[
            "bound_selection_manifest_sha256"
        ],
        "summary": result["summary"],
        "decision": result["decision"],
        "objects": objects,
        "protocol": result["protocol"],
    }


def compute_object_audit(
    *,
    v6: Any,
    v3: Any,
    descriptors: list[Any],
    row: dict[str, Any],
    capture: Any,
    source_truth: np.ndarray,
    target_truth: np.ndarray,
    protocol: dict[str, Any],
    scores: np.ndarray,
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    object_id = str(descriptors[0].object_id)
    if any(str(descriptor.object_id) != object_id for descriptor in descriptors):
        raise ValueError("descriptor object identities disagree")
    target_errors = np.asarray(capture.target_errors, dtype=np.float64)
    source_errors = np.asarray(capture.source_residuals, dtype=np.float64)
    target_truth = np.asarray(target_truth, dtype=np.float64)
    predicted_mean = target_truth - target_errors
    query_names = tuple(str(value) for value in evaluation["query_order"])
    bank = v6.query_bank(target_truth.shape[1])
    if tuple(bank) != query_names:
        raise ValueError("frozen v6 query bank order changed")
    weights = np.stack([bank[name][0] for name in query_names], axis=0)
    query_events = tuple(str(bank[name][1]) for name in query_names)

    existing_arms = v6.covariance_arms(
        v3.base,
        capture.covariance,
        seed=v6.stable_seed(
            int(protocol["lineage"]["original_v6_random_seed"]),
            object_id,
            "scrambled-factor",
        ),
    )
    centered_source_errors = source_errors - source_errors.mean(axis=0, keepdims=True)
    scales: list[float] = []
    thresholds: list[float] = []
    calibrations: dict[str, Any] = {}
    for name in query_names:
        weight, event = bank[name]
        raw_variances = {
            arm_name: v6.covariance_query_variance(model, weight)
            for arm_name, model in existing_arms.items()
        }
        calibration = v6.source_query_calibration(
            centered_source_errors,
            np.asarray(source_truth, dtype=np.float64),
            weight,
            raw_variances,
            event=event,
            probability=float(evaluation["coverage_probability"]),
            event_quantile=float(evaluation["event_threshold_quantile"]),
        )
        calibrations[name] = calibration
        scales.append(float(calibration["shared_variance_scale"]))
        thresholds.append(float(calibration["event_threshold"]))

    raw_covariance = query_covariance(capture.covariance, weights)
    scale_root = np.sqrt(np.asarray(scales, dtype=np.float64))
    calibrated_covariance = (
        scale_root[:, None] * raw_covariance * scale_root[None, :]
    )
    correlation, standard_deviations = covariance_to_correlation(
        calibrated_covariance
    )
    sample_count = int(evaluation["sample_count"])
    full_samples, correlation_error, variance_error = gaussian_copula_samples(
        correlation,
        standard_deviations,
        count=sample_count,
        seed=stable_seed(
            int(evaluation["sample_seed"]),
            object_id,
            "full-query-copula",
        ),
        scores=scores,
    )
    samples = {
        ARMS[0]: full_samples,
        ARMS[1]: independent_copula(
            full_samples,
            seed=stable_seed(
                int(evaluation["sample_seed"]),
                object_id,
                "independent-query-copula",
            ),
        ),
        ARMS[2]: scrambled_copula(
            full_samples,
            permutation=tuple(
                int(value) for value in evaluation["scramble_permutation"]
            ),
            signs=tuple(int(value) for value in evaluation["scramble_signs"]),
        ),
    }
    marginal_parity = marginal_parity_max_abs(full_samples, samples)
    target_query_mean = predicted_mean @ weights.T
    target_query_truth = target_truth @ weights.T
    arm_records, probability_parity = evaluate_copula_arms(
        target_mean=target_query_mean,
        target_truth=target_query_truth,
        residual_samples=samples,
        thresholds=np.asarray(thresholds, dtype=np.float64),
        query_events=query_events,
        query_names=query_names,
        fallback_cost=float(evaluation["fallback_cost"]),
        probability_clip=float(evaluation["probability_clip"]),
    )
    return {
        "object_id": object_id,
        "target_episode_id": row["target_episode_id"],
        "target_action": row["target_action"],
        "target_action_family": row["target_action_family"],
        "window_count": int(target_truth.shape[0]),
        "query_count": len(query_names),
        "composite_event_count": len(composite_event_bank(query_names)),
        "predictive_query_mean_sha256": array_digest(target_query_mean),
        "target_query_truth_sha256": array_digest(target_query_truth),
        "query_weight_matrix_sha256": array_digest(weights),
        "calibration_sha256": canonical_digest(calibrations),
        "calibrated_query_covariance": calibrated_covariance.tolist(),
        "calibrated_query_correlation": correlation.tolist(),
        "full_copula_sample_correlation": np.corrcoef(
            full_samples, rowvar=False
        ).tolist(),
        "full_copula_correlation_approximation_max_abs": correlation_error,
        "normal_score_marginal_variance_relative_error_max_abs": variance_error,
        "query_sample_marginal_parity_max_abs": marginal_parity,
        "single_query_event_probability_parity_max_abs": probability_parity,
        "sample_sha256": {name: array_digest(value) for name, value in samples.items()},
        "arms": arm_records,
    }


def run(
    *,
    audit_protocol_path: Path,
    recovery_runner_path: Path,
    base_runner_path: Path,
    original_protocol_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    data_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
    reference_result_path: Path,
) -> dict[str, Any]:
    protocol = read_json(audit_protocol_path)
    validate_protocol(protocol)
    reference = read_json(reference_result_path)
    validate_reference_result(reference, reference_result_path, protocol)
    recovery = load_module(
        recovery_runner_path.resolve(strict=True),
        "deform360_dependence_bound_recovery_for_copula_audit",
    )

    scores = normal_score_grid(int(protocol["evaluation"]["sample_count"]))
    audit_rows: list[dict[str, Any]] = []
    original_loader: Callable[[Path, str], Any] = recovery.load_module
    resolved_base = base_runner_path.resolve(strict=True)

    def intercepting_loader(path: Path, name: str) -> Any:
        module = original_loader(path, name)
        if path.resolve() != resolved_base:
            return module
        original_evaluator = module.evaluate_object_with_capture

        def intercepted_evaluator(
            v3: Any,
            descriptors: list[Any],
            development: dict[str, Any],
            base_protocol: dict[str, Any],
            rng: np.random.Generator,
        ) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
            output = original_evaluator(
                v3,
                descriptors,
                development,
                base_protocol,
                rng,
            )
            row, capture, source_truth, target_truth = output
            audit_rows.append(
                compute_object_audit(
                    v6=module,
                    v3=v3,
                    descriptors=descriptors,
                    row=row,
                    capture=capture,
                    source_truth=source_truth,
                    target_truth=target_truth,
                    protocol=protocol,
                    scores=scores,
                )
            )
            return output

        module.evaluate_object_with_capture = intercepted_evaluator
        return module

    recovery.load_module = intercepting_loader
    try:
        _, recovered = recovery.run(
            base_runner_path=resolved_base,
            protocol_path=original_protocol_path,
            parent_protocol_path=parent_protocol_path,
            parent_result_path=parent_result_path,
            readiness_path=readiness_path,
            data_root=data_root,
            parent_control_root=parent_control_root,
            frozen_root=frozen_root,
        )
    finally:
        recovery.load_module = original_loader

    reference_projection = scientific_projection(reference)
    recovered_projection = scientific_projection(recovered)
    reference_reproduced = recovered_projection == reference_projection
    if not reference_reproduced:
        raise RuntimeError("completed v6 scientific result did not reproduce exactly")
    if [row["object_id"] for row in audit_rows] != [
        row["object_id"] for row in reference["objects"]
    ]:
        raise RuntimeError("copula-audit object order changed")

    summary, decision = aggregate(
        audit_rows,
        protocol,
        reference_reproduced=reference_reproduced,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root.resolve()),
        "lineage": protocol["lineage"],
        "reference_reproduction": {
            "exact_scientific_projection_reproduced": reference_reproduced,
            "reference_projection_sha256": canonical_digest(reference_projection),
            "recovered_projection_sha256": canonical_digest(recovered_projection),
        },
        "information_boundary": {
            "retrospective_target_reuse": True,
            "target_open_before_protocol_design": True,
            "exact_completed_v6_scientific_result_reproduced": True,
            "exact_bound_numeric_carriers_reused": True,
            "unbound_numeric_payloads_opened": False,
            "predictive_query_means_identical_across_arms": True,
            "empirical_univariate_query_marginals_identical_across_arms": True,
            "single_query_event_probabilities_identical_across_arms": True,
            "all_unordered_query_pairs_scored_without_selection": True,
            "new_measurements_collected": False,
            "fresh_confirmation": False,
        },
        "summary": summary,
        "decision": decision,
        "objects": audit_rows,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def make_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    lines = [
        "# Deform360 exact-query-marginal copula audit v1",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Existing physical queries: **{summary['query_count']}**",
        f"- Joint events per object: **{summary['composite_event_count']}**",
        f"- Copula samples per object: **{summary['sample_count_per_object']}**",
        "- Predictive query means: **identical by construction**",
        "- Empirical univariate query marginals: **exactly identical**",
        "- Single-query event probabilities: **exactly identical**",
        f"- Outcome: **{decision['outcome']}**",
        "",
        "## Object-balanced joint-event results",
        "",
        "| Arm | Brier | Log loss | Decision loss | Acceptance | Harm/all |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = summary["arm_summary"][arm]
        lines.append(
            f"| `{arm}` | {values['event_brier']:.6g} | "
            f"{values['event_log_loss']:.6g} | {values['decision_loss']:.6g} | "
            f"{values['acceptance_fraction']:.3%} | "
            f"{values['harmful_accept_fraction_all']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Full-copula paired contrasts",
            "",
            "Negative differences favor the full query copula.",
            "",
            "| Comparator | Metric | Difference | 95% object bootstrap | W/T/L |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparator, metrics in summary["comparisons"].items():
        for metric, values in metrics.items():
            interval = values["object_bootstrap_95_interval"]
            lines.append(
                f"| `{comparator}` | `{metric}` | "
                f"{values['mean_difference']:.6g} | "
                f"[{interval[0]:.6g}, {interval[1]:.6g}] | "
                f"{values['object_wins']}/{values['object_ties']}/"
                f"{values['object_losses']} |"
            )
    lines.extend(
        [
            "",
            "## Execution and interpretation gates",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
    )
    for name, passed in decision["gates"].items():
        lines.append(f"| `{name}` | {str(bool(passed)).lower()} |")
    lines.extend(
        [
            "",
            "This audit was specified after the target cohort had already been",
            "opened. It is retrospective mechanism evidence only. No target event",
            "pair was selected: every unordered pair of the five previously",
            "registered queries is scored as both a conjunction and a disjunction.",
            "",
        ]
    )
    return "\n".join(lines)


def write_object_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "object_id",
        "target_episode_id",
        "target_action_family",
        "window_count",
        "query_sample_marginal_parity_max_abs",
        "single_query_event_probability_parity_max_abs",
        "full_copula_correlation_approximation_max_abs",
    ]
    for arm in ARMS:
        for metric in ("event_brier", "decision_loss", "event_log_loss"):
            fields.append(f"{arm}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {name: row[name] for name in fields[:7]}
            for arm in ARMS:
                for metric in ("event_brier", "decision_loss", "event_log_loss"):
                    output[f"{arm}_{metric}"] = row["arms"][arm]["summary"][metric]
            writer.writerow(output)


def self_test() -> None:
    covariance = np.asarray(
        [
            [1.0, 0.75, -0.25],
            [0.75, 1.5, 0.1],
            [-0.25, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    correlation, standard_deviations = covariance_to_correlation(covariance)
    scores = normal_score_grid(4096)
    full, correlation_error, variance_error = gaussian_copula_samples(
        correlation,
        standard_deviations,
        count=4096,
        seed=71,
        scores=scores,
    )
    independent = independent_copula(full, seed=72)
    scrambled = scrambled_copula(
        full,
        permutation=(1, 2, 0),
        signs=(1, -1, 1),
    )
    samples = {
        ARMS[0]: full,
        ARMS[1]: independent,
        ARMS[2]: scrambled,
    }
    assert marginal_parity_max_abs(full, samples) == 0.0
    assert correlation_error < 0.05
    assert variance_error < 0.01
    mean = np.asarray(
        [[-0.4, 0.1, 0.3], [0.0, -0.2, 0.5], [0.5, 0.4, -0.1]],
        dtype=np.float64,
    )
    truth = np.asarray(
        [[-0.2, 0.2, 0.1], [0.3, -0.4, 0.7], [0.8, 0.5, -0.2]],
        dtype=np.float64,
    )
    records, parity = evaluate_copula_arms(
        target_mean=mean,
        target_truth=truth,
        residual_samples=samples,
        thresholds=np.asarray([0.2, 0.3, 0.25]),
        query_events=("upper", "absolute", "upper"),
        query_names=("a", "b", "c"),
        fallback_cost=0.1,
        probability_clip=1e-9,
    )
    assert parity == 0.0
    assert records[ARMS[0]]["events"] != records[ARMS[1]]["events"]
    assert len(composite_event_bank(("a", "b", "c"))) == 6
    print("Deform360 query-copula audit self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-protocol", type=Path)
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--base-runner", type=Path)
    parser.add_argument("--original-protocol", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "audit_protocol",
        "recovery_runner",
        "base_runner",
        "original_protocol",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "data_root",
        "parent_control_root",
        "frozen_root",
        "reference_result",
        "output_json",
        "output_report",
        "output_csv",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    result = run(
        audit_protocol_path=args.audit_protocol,
        recovery_runner_path=args.recovery_runner,
        base_runner_path=args.base_runner,
        original_protocol_path=args.original_protocol,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
        data_root=args.data_root,
        parent_control_root=args.parent_control_root,
        frozen_root=args.frozen_root,
        reference_result_path=args.reference_result,
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
