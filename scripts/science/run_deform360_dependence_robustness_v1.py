#!/usr/bin/env python3
"""Retrospective robustness replay for Deform360 dependence-only value.

This study reuses the exact bound 92-object cohort, frozen point predictor,
predictive means, query bank, source-only calibration, and target outcomes from
the completed Deform360 same-mean dependence-query v6 recovery. It adds four
predeclared diagnostics without selecting or refitting on target outcomes:

* a continuous dependence-strength path with fixed coordinate marginals;
* a factor-rank/energy path with discarded factor variance moved to the diagonal;
* execute-versus-fallback loss over a fixed cost grid; and
* risk ranking at fixed coverage, including coverage matched to the full model.

All comparisons remain retrospective mechanism evidence. They do not authorize
a fresh-confirmation, calibration, safety, provider-runtime, or paper claim.
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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-dependence-robustness-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-dependence-robustness-protocol-v1"
RECOVERY_REVISION = "0f9312a3cae9f854631d8c61c2893c6527014219"
ORIGINAL_V6_REVISION = "954538832106d8ded13f1101b3a2b2e855b40513"
ORIGINAL_V6_RUNNER_SHA256 = (
    "06c22fc3fe667c4f2f11eddee3dcb1b78b5465b6312136efb010611e1ebab91c"
)
REFERENCE_RUN_ID = 33528032875
REFERENCE_ARTIFACT_ID = 9811194776
REFERENCE_ARTIFACT_SHA256 = (
    "8b3bad2bc0620228ebe32027028b1666ea7772a1850f0ab45d525e30e4ced82a"
)
REFERENCE_RESULT_FILE_SHA256 = (
    "c73659af65c2b87923f7bd668f9717afab03e449a5b3abd3a5b597ec60898fd1"
)
REFERENCE_INTERNAL_RESULT_SHA256 = (
    "d430731e56ce470a5e0df8fbd3bc13dea83763beccd2df06d86cde2365d4ee36"
)
_EPS = 1e-12

CORE_METRICS = (
    "target_query_nanees",
    "target_90_coverage",
    "mean_90_interval_width",
    "query_nll",
    "event_brier",
    "event_log_loss",
    "decision_loss",
    "decision_regret",
    "acceptance_fraction",
    "harmful_accept_fraction_all",
    "harmful_accept_rate_given_accept",
)
DECISION_METRICS = (
    "decision_loss",
    "decision_regret",
    "acceptance_fraction",
    "harmful_accept_fraction_all",
    "harmful_accept_rate_given_accept",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def level_key(value: float) -> str:
    return f"{value:.6g}"


def validate_protocol(
    protocol: dict[str, Any],
    *,
    dataset_root: Path,
    base_protocol_path: Path,
    recovery_root: Path,
    original_v6_root: Path,
) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected robustness protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unexpected robustness protocol version")
    if protocol.get("status") != "frozen-before-retrospective-replay":
        raise ValueError("robustness protocol is not frozen")
    if Path(str(protocol.get("dataset_root"))) != dataset_root:
        raise ValueError("dataset root changed")

    base = protocol.get("base_study")
    if not isinstance(base, dict):
        raise ValueError("base-study binding is absent")
    expected_base = {
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_artifact_id": REFERENCE_ARTIFACT_ID,
        "reference_artifact_sha256": REFERENCE_ARTIFACT_SHA256,
        "reference_result_file_sha256": REFERENCE_RESULT_FILE_SHA256,
        "reference_internal_result_sha256": REFERENCE_INTERNAL_RESULT_SHA256,
        "recovery_revision": RECOVERY_REVISION,
        "original_v6_revision": ORIGINAL_V6_REVISION,
        "original_v6_runner_sha256": ORIGINAL_V6_RUNNER_SHA256,
    }
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            raise ValueError(f"base-study binding changed: {key}")
    if sha256_file(base_protocol_path) != str(base.get("base_protocol_file_sha256")):
        raise ValueError("base scientific protocol bytes changed")
    if git_output(recovery_root, "rev-parse", "HEAD") != RECOVERY_REVISION:
        raise ValueError("recovery control revision changed")
    if git_output(original_v6_root, "rev-parse", "HEAD") != ORIGINAL_V6_REVISION:
        raise ValueError("original v6 control revision changed")

    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation contract is absent")
    exact_vectors = {
        "dependence_strengths": [0.0, 0.25, 0.5, 0.75, 1.0],
        "rank_energy_fractions": [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
        "fallback_costs": [0.02, 0.05, 0.1, 0.2, 0.3],
        "fixed_coverages": [0.1, 0.25, 0.5, 0.75, 0.9],
    }
    for key, expected in exact_vectors.items():
        actual = [float(value) for value in evaluation.get(key, ())]
        if actual != expected:
            raise ValueError(f"registered grid changed: {key}")
    if float(evaluation.get("reference_fallback_cost", math.nan)) != 0.1:
        raise ValueError("reference fallback cost changed")
    if int(evaluation.get("bootstrap_repetitions", 0)) != 10000:
        raise ValueError("bootstrap repetitions changed")
    if int(evaluation.get("random_seed", -1)) != 260903:
        raise ValueError("random seed changed")
    if int(evaluation.get("minimum_cost_levels_for_robustness", -1)) != 4:
        raise ValueError("cost-robustness gate changed")
    if float(evaluation.get("marginal_parity_tolerance", math.nan)) != 1e-12:
        raise ValueError("marginal parity tolerance changed")
    if float(evaluation.get("reference_numeric_tolerance", math.nan)) != 1e-12:
        raise ValueError("reference numeric tolerance changed")

    boundary = protocol.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("information boundary is absent")
    required_true = (
        "retrospective_target_reuse",
        "exact_bound_object_roster_reused",
        "exact_frozen_point_predictor_reused",
        "same_predictive_mean_required",
        "coordinate_marginals_equal_required",
        "source_only_calibration_reused",
    )
    required_false = (
        "new_measurements_collected",
        "unbound_numeric_payloads_may_open",
        "target_outcomes_may_select_grids",
        "target_outcomes_may_refit_models",
        "paper_claim_authorized",
    )
    for key in required_true:
        if boundary.get(key) is not True:
            raise ValueError(f"required information boundary disabled: {key}")
    for key in required_false:
        if boundary.get(key) is not False:
            raise ValueError(f"forbidden information flow enabled: {key}")
    for key in (
        "paper_claim_authorized",
        "fresh_confirmation_authorized",
        "calibration_claim_authorized",
        "deployment_safety_claim_authorized",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"protocol may not self-authorize {key}")


def git_output(root: Path, *arguments: str) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


def marginal_variance(model: Any) -> np.ndarray:
    return np.asarray(model.multiplier, dtype=np.float64) * (
        np.asarray(model.diagonal, dtype=np.float64)
        + np.sum(np.square(np.asarray(model.factor, dtype=np.float64)), axis=1)
    )


def make_covariance_model(
    base: Any,
    reference: Any,
    *,
    diagonal: np.ndarray,
    factor: np.ndarray,
) -> Any:
    return base.CovarianceModel(
        np.asarray(reference.mean_error, dtype=np.float64).copy(),
        np.asarray(diagonal, dtype=np.float64).copy(),
        np.asarray(factor, dtype=np.float64).copy(),
        float(reference.multiplier),
        float(reference.marginal_z),
        float(reference.source_marginal_coverage),
        float(reference.source_joint_nanees),
    )


def dependence_strength_model(base: Any, covariance: Any, strength: float) -> Any:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("dependence strength must be in [0,1]")
    if strength == 1.0:
        return covariance
    factor = np.asarray(covariance.factor, dtype=np.float64)
    diagonal = np.asarray(covariance.diagonal, dtype=np.float64)
    row_energy = np.sum(np.square(factor), axis=1)
    scaled_factor = math.sqrt(strength) * factor
    compensated_diagonal = diagonal + (1.0 - strength) * row_energy
    result = make_covariance_model(
        base,
        covariance,
        diagonal=compensated_diagonal,
        factor=scaled_factor,
    )
    if not np.allclose(
        marginal_variance(result),
        marginal_variance(covariance),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("dependence-strength arm changed coordinate marginals")
    return result


def rank_energy_model(
    base: Any,
    covariance: Any,
    requested_fraction: float,
) -> tuple[Any, dict[str, float | int]]:
    if not 0.0 <= requested_fraction <= 1.0:
        raise ValueError("rank energy fraction must be in [0,1]")
    factor = np.asarray(covariance.factor, dtype=np.float64)
    diagonal = np.asarray(covariance.diagonal, dtype=np.float64)
    full_row_energy = np.sum(np.square(factor), axis=1)
    total_energy = float(np.sum(full_row_energy))
    if requested_fraction == 1.0:
        rank = int(np.linalg.matrix_rank(factor))
        return covariance, {
            "requested_energy_fraction": 1.0,
            "retained_energy_fraction": 1.0 if total_energy > _EPS else 0.0,
            "retained_rank": rank,
            "original_rank": rank,
        }
    if requested_fraction == 0.0 or total_energy <= _EPS or factor.shape[1] == 0:
        retained = np.empty((factor.shape[0], 0), dtype=np.float64)
        rank = 0
        original_rank = int(np.linalg.matrix_rank(factor))
        retained_energy = 0.0
    else:
        left, singular, _ = np.linalg.svd(factor, full_matrices=False)
        energies = np.square(singular)
        cumulative = np.cumsum(energies) / float(np.sum(energies))
        rank = int(np.searchsorted(cumulative, requested_fraction, side="left") + 1)
        retained = left[:, :rank] * singular[:rank]
        original_rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
        retained_energy = float(np.sum(np.square(retained)) / total_energy)
    retained_row_energy = np.sum(np.square(retained), axis=1)
    compensated_diagonal = diagonal + full_row_energy - retained_row_energy
    if float(np.min(compensated_diagonal)) < -1e-10:
        raise RuntimeError("rank compensation produced a negative diagonal")
    compensated_diagonal = np.maximum(compensated_diagonal, 0.0)
    result = make_covariance_model(
        base,
        covariance,
        diagonal=compensated_diagonal,
        factor=retained,
    )
    if not np.allclose(
        marginal_variance(result),
        marginal_variance(covariance),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("rank-energy arm changed coordinate marginals")
    return result, {
        "requested_energy_fraction": float(requested_fraction),
        "retained_energy_fraction": retained_energy,
        "retained_rank": rank,
        "original_rank": original_rank,
    }


def query_risk_arrays(
    v6: Any,
    *,
    target_truth: np.ndarray,
    target_errors: np.ndarray,
    weight: np.ndarray,
    event: str,
    model: Any,
    calibration: dict[str, float],
    probability_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw_variance = v6.covariance_query_variance(model, weight)
    variance = raw_variance * float(calibration["shared_variance_scale"])
    threshold = float(calibration["event_threshold"])
    target_error = target_errors @ weight
    target_truth_query = target_truth @ weight
    target_mean = target_truth_query - target_error
    if event == "upper":
        labels = target_truth_query > threshold
    elif event == "absolute":
        labels = np.abs(target_truth_query) > threshold
    else:
        raise ValueError(f"unsupported event type: {event}")
    predicted = np.clip(
        v6.event_probability(target_mean, variance, threshold, event),
        probability_clip,
        1.0 - probability_clip,
    )
    return np.asarray(predicted, dtype=np.float64), np.asarray(labels, dtype=bool)


def decision_metrics(
    predicted: np.ndarray,
    labels: np.ndarray,
    *,
    fallback_cost: float,
    accepted_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if accepted_mask is None:
        accepted_mask = predicted <= fallback_cost
    accepted_mask = np.asarray(accepted_mask, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    if accepted_mask.shape != labels.shape or predicted.shape != labels.shape:
        raise ValueError("decision arrays must have identical shapes")
    label_values = labels.astype(np.float64)
    realized = np.where(accepted_mask, label_values, fallback_cost)
    oracle = np.where(labels, fallback_cost, 0.0)
    accepted = int(np.count_nonzero(accepted_mask))
    harmful = int(np.count_nonzero(accepted_mask & labels))
    return {
        "decision_loss": float(np.mean(realized)),
        "decision_regret": float(np.mean(realized - oracle)),
        "acceptance_fraction": float(np.mean(accepted_mask)),
        "harmful_accept_fraction_all": float(harmful / len(labels)),
        "harmful_accept_rate_given_accept": float(
            harmful / accepted if accepted else 0.0
        ),
    }


def lowest_risk_mask(predicted: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(predicted, dtype=np.float64)
    if not 0 <= count <= len(values):
        raise ValueError("accepted count is outside the available rows")
    order = np.lexsort((np.arange(len(values), dtype=np.int64), values))
    result = np.zeros(len(values), dtype=bool)
    result[order[:count]] = True
    return result


def coverage_count(size: int, coverage: float) -> int:
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be in [0,1]")
    return min(size, max(0, int(math.floor(coverage * size + 0.5))))


def mean_record(
    records: list[dict[str, float]],
    metrics: tuple[str, ...],
) -> dict[str, float]:
    if not records:
        raise ValueError("cannot aggregate an empty record list")
    return {
        metric: float(np.mean([float(record[metric]) for record in records]))
        for metric in metrics
    }


def reference_projection(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "object_id",
        "source_episode_ids",
        "target_episode_id",
        "target_action",
        "target_action_family",
        "dimension",
        "window_count",
        "predictive_mean_sha256",
        "same_mean_by_construction",
        "parent_point_result_exact",
        "coordinate_marginal_parity_max_abs",
        "query_bank_sha256",
        "queries",
        "arm_summary",
        "joint_metrics",
    )
    return {key: row[key] for key in keys}


def nested_max_difference(actual: Any, expected: Any, *, path: str = "root") -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"reference keys differ at {path}")
        return max(
            (
                nested_max_difference(actual[key], expected[key], path=f"{path}.{key}")
                for key in expected
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"reference list differs at {path}")
        return max(
            (
                nested_max_difference(value, target, path=f"{path}[{index}]")
                for index, (value, target) in enumerate(
                    zip(actual, expected, strict=True)
                )
            ),
            default=0.0,
        )
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise RuntimeError(
                f"reference value differs at {path}: {actual!r} != {expected!r}"
            )
        return 0.0
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
            if actual != expected:
                raise RuntimeError(f"non-finite reference value differs at {path}")
            return 0.0
        return abs(float(actual) - float(expected))
    if actual != expected:
        raise RuntimeError(f"reference value differs at {path}")
    return 0.0


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
    means = vector[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_comparison(
    rows: list[dict[str, Any]],
    *,
    section: str,
    left: str,
    right: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [row[section][left][metric] - row[section][right][metric] for row in rows],
        dtype=np.float64,
    )
    interval = bootstrap_interval(differences, repetitions, seed)
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "object_bootstrap_95_interval": interval,
        "object_wins": int(np.count_nonzero(differences < 0.0)),
        "object_ties": int(np.count_nonzero(differences == 0.0)),
        "object_losses": int(np.count_nonzero(differences > 0.0)),
        "supported": bool(float(np.mean(differences)) < 0.0 and interval[1] < 0.0),
    }


def aggregate_grid(
    rows: list[dict[str, Any]],
    *,
    section: str,
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    labels = tuple(rows[0][section])
    if any(tuple(row[section]) != labels for row in rows):
        raise RuntimeError(f"grid labels differ across objects: {section}")
    return {
        label: {
            metric: float(np.mean([row[section][label][metric] for row in rows]))
            for metric in metrics
        }
        for label in labels
    }


def aggregate_nested_grid(
    rows: list[dict[str, Any]],
    *,
    section: str,
    metrics: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    outer_labels = tuple(rows[0][section])
    if any(tuple(row[section]) != outer_labels for row in rows):
        raise RuntimeError(f"outer grid labels differ across objects: {section}")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for outer in outer_labels:
        inner_labels = tuple(rows[0][section][outer])
        if any(tuple(row[section][outer]) != inner_labels for row in rows):
            raise RuntimeError(
                f"inner grid labels differ across objects: {section}.{outer}"
            )
        result[outer] = {
            inner: {
                metric: float(
                    np.mean([row[section][outer][inner][metric] for row in rows])
                )
                for metric in metrics
            }
            for inner in inner_labels
        }
    return result


def original_row(
    v6: Any,
    *,
    object_id: str,
    point_row: dict[str, Any],
    capture: Any,
    source_truth: np.ndarray,
    target_truth: np.ndarray,
    drift: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_errors = np.asarray(capture.target_errors, dtype=np.float64)
    source_errors = np.asarray(capture.source_residuals, dtype=np.float64)
    predicted_mean = target_truth - target_errors
    arms = v6.covariance_arms(
        v6.base,
        capture.covariance,
        seed=v6.stable_seed(
            int(evaluation["base_random_seed"]),
            object_id,
            "scrambled-factor",
        ),
    )
    reference_marginal = v6.marginal_variance(arms["full_low_rank"])
    marginal_parity = float(
        max(
            np.max(np.abs(v6.marginal_variance(model) - reference_marginal))
            for model in arms.values()
        )
    )
    bank = v6.query_bank(target_truth.shape[1])
    centered_source_errors = source_errors - source_errors.mean(axis=0, keepdims=True)
    queries: dict[str, Any] = {}
    risk_arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for query_name, (weight, event) in bank.items():
        raw_variances = {
            arm_name: v6.covariance_query_variance(model, weight)
            for arm_name, model in arms.items()
        }
        calibration = v6.source_query_calibration(
            centered_source_errors,
            source_truth,
            weight,
            raw_variances,
            event=event,
            probability=float(evaluation["coverage_probability"]),
            event_quantile=float(evaluation["event_threshold_quantile"]),
        )
        query_record = {
            "event": event,
            "weight_sha256": v6.array_digest(weight),
            "calibration": calibration,
            "arms": {},
        }
        risk_arrays[query_name] = {}
        for arm_name, model in arms.items():
            query_record["arms"][arm_name] = v6.query_metrics(
                centered_source_errors=centered_source_errors,
                target_truth=target_truth,
                target_errors=target_errors,
                weight=weight,
                event=event,
                model=model,
                calibration=calibration,
                fallback_cost=float(evaluation["reference_fallback_cost"]),
                probability_clip=float(evaluation["probability_clip"]),
            )
            risk_arrays[query_name][arm_name] = query_risk_arrays(
                v6,
                target_truth=target_truth,
                target_errors=target_errors,
                weight=weight,
                event=event,
                model=model,
                calibration=calibration,
                probability_clip=float(evaluation["probability_clip"]),
            )
        queries[query_name] = query_record

    arm_summary: dict[str, dict[str, float]] = {}
    for arm_name in v6.COVARIANCE_ARMS:
        values = [
            queries[query_name]["arms"][arm_name]
            for query_name, _ in v6.QUERY_SPECS
        ]
        arm_summary[arm_name] = mean_record(values, CORE_METRICS)
        arm_summary[arm_name]["calibration_log_error"] = float(
            np.mean(
                [
                    abs(math.log(max(value["target_query_nanees"], _EPS)))
                    for value in values
                ]
            )
        )
        arm_summary[arm_name]["coverage_absolute_error"] = float(
            np.mean(
                [
                    abs(
                        value["target_90_coverage"]
                        - float(evaluation["coverage_probability"])
                    )
                    for value in values
                ]
            )
        )
    result = {
        "object_id": object_id,
        "source_episode_ids": point_row["source_episode_ids"],
        "target_episode_id": point_row["target_episode_id"],
        "target_action": point_row["target_action"],
        "target_action_family": point_row["target_action_family"],
        "dimension": int(target_truth.shape[1]),
        "window_count": int(target_truth.shape[0]),
        "predictive_mean_sha256": v6.array_digest(predicted_mean),
        "same_mean_by_construction": True,
        "parent_point_result_exact": True,
        "coordinate_marginal_parity_max_abs": marginal_parity,
        "query_bank_sha256": v6.canonical_digest(
            {
                name: {
                    "event": event,
                    "weight_sha256": queries[name]["weight_sha256"],
                }
                for name, event in v6.QUERY_SPECS
            }
        ),
        "queries": queries,
        "arm_summary": arm_summary,
        "joint_metrics": {
            name: v6.joint_metrics(
                v6.base,
                target_errors,
                model,
                float(evaluation["coverage_probability"]),
            )
            for name, model in arms.items()
        },
        "bound_carrier_recovery": drift,
    }
    context = {
        "arms": arms,
        "bank": bank,
        "centered_source_errors": centered_source_errors,
        "source_truth": source_truth,
        "target_truth": target_truth,
        "target_errors": target_errors,
        "queries": queries,
        "risk_arrays": risk_arrays,
    }
    return result, context


def robustness_row(
    v6: Any,
    *,
    original: dict[str, Any],
    context: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    target_truth = context["target_truth"]
    target_errors = context["target_errors"]
    centered_source_errors = context["centered_source_errors"]
    source_truth = context["source_truth"]
    bank = context["bank"]
    original_arms = context["arms"]
    query_records = context["queries"]
    probability_clip = float(evaluation["probability_clip"])
    reference_cost = float(evaluation["reference_fallback_cost"])

    continuum_models = {
        level_key(float(strength)): dependence_strength_model(
            v6.base,
            original_arms["full_low_rank"],
            float(strength),
        )
        for strength in evaluation["dependence_strengths"]
    }
    rank_models: dict[str, Any] = {}
    rank_metadata: dict[str, dict[str, float | int]] = {}
    for fraction_value in evaluation["rank_energy_fractions"]:
        fraction = float(fraction_value)
        key = level_key(fraction)
        rank_models[key], rank_metadata[key] = rank_energy_model(
            v6.base,
            original_arms["full_low_rank"],
            fraction,
        )

    continuum_queries: dict[str, list[dict[str, float]]] = {
        key: [] for key in continuum_models
    }
    rank_queries: dict[str, list[dict[str, float]]] = {key: [] for key in rank_models}
    cost_queries: dict[str, dict[str, list[dict[str, float]]]] = {
        arm: {level_key(float(cost)): [] for cost in evaluation["fallback_costs"]}
        for arm in v6.COVARIANCE_ARMS
    }
    coverage_queries: dict[str, dict[str, list[dict[str, float]]]] = {
        arm: {level_key(float(value)): [] for value in evaluation["fixed_coverages"]}
        for arm in v6.COVARIANCE_ARMS
    }
    matched_queries: dict[str, list[dict[str, float]]] = {
        arm: [] for arm in v6.COVARIANCE_ARMS
    }

    for query_name, (weight, event) in bank.items():
        calibration = query_records[query_name]["calibration"]
        for key, model in continuum_models.items():
            continuum_queries[key].append(
                v6.query_metrics(
                    centered_source_errors=centered_source_errors,
                    target_truth=target_truth,
                    target_errors=target_errors,
                    weight=weight,
                    event=event,
                    model=model,
                    calibration=calibration,
                    fallback_cost=reference_cost,
                    probability_clip=probability_clip,
                )
            )
        for key, model in rank_models.items():
            rank_queries[key].append(
                v6.query_metrics(
                    centered_source_errors=centered_source_errors,
                    target_truth=target_truth,
                    target_errors=target_errors,
                    weight=weight,
                    event=event,
                    model=model,
                    calibration=calibration,
                    fallback_cost=reference_cost,
                    probability_clip=probability_clip,
                )
            )

        full_predicted, _ = context["risk_arrays"][query_name]["full_low_rank"]
        full_count = int(np.count_nonzero(full_predicted <= reference_cost))
        for arm in v6.COVARIANCE_ARMS:
            predicted, labels = context["risk_arrays"][query_name][arm]
            for cost_value in evaluation["fallback_costs"]:
                cost = float(cost_value)
                cost_queries[arm][level_key(cost)].append(
                    decision_metrics(predicted, labels, fallback_cost=cost)
                )
            for coverage_value in evaluation["fixed_coverages"]:
                coverage = float(coverage_value)
                count = coverage_count(len(predicted), coverage)
                coverage_queries[arm][level_key(coverage)].append(
                    decision_metrics(
                        predicted,
                        labels,
                        fallback_cost=reference_cost,
                        accepted_mask=lowest_risk_mask(predicted, count),
                    )
                )
            matched_queries[arm].append(
                decision_metrics(
                    predicted,
                    labels,
                    fallback_cost=reference_cost,
                    accepted_mask=lowest_risk_mask(predicted, full_count),
                )
            )

    continuum = {
        key: mean_record(records, CORE_METRICS)
        for key, records in continuum_queries.items()
    }
    rank_energy = {}
    for key, records in rank_queries.items():
        rank_energy[key] = mean_record(records, CORE_METRICS)
        rank_energy[key].update(rank_metadata[key])
    cost_sensitivity = {
        arm: {
            cost: mean_record(records, DECISION_METRICS)
            for cost, records in costs.items()
        }
        for arm, costs in cost_queries.items()
    }
    fixed_coverage = {
        arm: {
            coverage: mean_record(records, DECISION_METRICS)
            for coverage, records in levels.items()
        }
        for arm, levels in coverage_queries.items()
    }
    matched_full_reference_coverage = {
        arm: mean_record(records, DECISION_METRICS)
        for arm, records in matched_queries.items()
    }

    reference_marginal = marginal_variance(original_arms["full_low_rank"])
    diagnostic_models = list(continuum_models.values()) + list(rank_models.values())
    marginal_parity = float(
        max(
            np.max(np.abs(marginal_variance(model) - reference_marginal))
            for model in diagnostic_models
        )
    )
    return {
        "object_id": original["object_id"],
        "target_episode_id": original["target_episode_id"],
        "target_action_family": original["target_action_family"],
        "dimension": original["dimension"],
        "window_count": original["window_count"],
        "predictive_mean_sha256": original["predictive_mean_sha256"],
        "query_bank_sha256": original["query_bank_sha256"],
        "same_mean_for_all_diagnostics": True,
        "coordinate_marginal_parity_max_abs": marginal_parity,
        "continuum": continuum,
        "rank_energy": rank_energy,
        "cost_sensitivity": cost_sensitivity,
        "fixed_coverage": fixed_coverage,
        "matched_full_reference_coverage": matched_full_reference_coverage,
    }


def aggregate_robustness(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = protocol["evaluation"]
    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["random_seed"])
    continuum = aggregate_grid(rows, section="continuum", metrics=CORE_METRICS)
    rank_energy = aggregate_grid(rows, section="rank_energy", metrics=CORE_METRICS)
    for key in rank_energy:
        rank_energy[key]["mean_retained_energy_fraction"] = float(
            np.mean(
                [
                    row["rank_energy"][key]["retained_energy_fraction"]
                    for row in rows
                ]
            )
        )
        rank_energy[key]["mean_retained_rank"] = float(
            np.mean([row["rank_energy"][key]["retained_rank"] for row in rows])
        )
        rank_energy[key]["median_retained_rank"] = float(
            np.median([row["rank_energy"][key]["retained_rank"] for row in rows])
        )
        rank_energy[key]["mean_original_rank"] = float(
            np.mean([row["rank_energy"][key]["original_rank"] for row in rows])
        )
    cost_sensitivity = aggregate_nested_grid(
        rows,
        section="cost_sensitivity",
        metrics=DECISION_METRICS,
    )
    fixed_coverage = aggregate_nested_grid(
        rows,
        section="fixed_coverage",
        metrics=DECISION_METRICS,
    )
    matched = aggregate_grid(
        rows,
        section="matched_full_reference_coverage",
        metrics=DECISION_METRICS,
    )

    continuum_comparisons = {
        metric: paired_comparison(
            rows,
            section="continuum",
            left="1",
            right="0",
            metric=metric,
            repetitions=repetitions,
            seed=seed + index,
        )
        for index, metric in enumerate(("decision_loss", "event_brier", "query_nll"))
    }
    matched_comparisons = {
        comparator: paired_comparison(
            rows,
            section="matched_full_reference_coverage",
            left="full_low_rank",
            right=comparator,
            metric="decision_loss",
            repetitions=repetitions,
            seed=seed + 100 + index,
        )
        for index, comparator in enumerate(
            ("diagonal_marginal_matched", "scrambled_marginal_matched")
        )
    }

    cost_comparisons: dict[str, dict[str, Any]] = {}
    for comparator_index, comparator in enumerate(
        ("diagonal_marginal_matched", "scrambled_marginal_matched")
    ):
        cost_comparisons[comparator] = {}
        for cost_index, cost_value in enumerate(evaluation["fallback_costs"]):
            cost = level_key(float(cost_value))
            differences = np.asarray(
                [
                    row["cost_sensitivity"]["full_low_rank"][cost]["decision_loss"]
                    - row["cost_sensitivity"][comparator][cost]["decision_loss"]
                    for row in rows
                ],
                dtype=np.float64,
            )
            interval = bootstrap_interval(
                differences,
                repetitions,
                seed + 1000 + comparator_index * 100 + cost_index,
            )
            cost_comparisons[comparator][cost] = {
                "mean_difference": float(np.mean(differences)),
                "object_bootstrap_95_interval": interval,
                "object_wins": int(np.count_nonzero(differences < 0.0)),
                "object_ties": int(np.count_nonzero(differences == 0.0)),
                "object_losses": int(np.count_nonzero(differences > 0.0)),
                "supported": bool(
                    float(np.mean(differences)) < 0.0 and interval[1] < 0.0
                ),
            }

    marginal_max = float(
        max(row["coordinate_marginal_parity_max_abs"] for row in rows)
    )
    minimum_cost_levels = int(evaluation["minimum_cost_levels_for_robustness"])
    cost_support_counts = {
        comparator: sum(
            bool(record["supported"])
            for record in cost_comparisons[comparator].values()
        )
        for comparator in cost_comparisons
    }
    gates = {
        "complete_92_object_roster": len(rows) == 92,
        "same_mean_for_every_diagnostic": all(
            bool(row["same_mean_for_all_diagnostics"]) for row in rows
        ),
        "coordinate_marginals_match": (
            marginal_max <= float(evaluation["marginal_parity_tolerance"])
        ),
        "continuum_full_beats_zero_decision_loss": continuum_comparisons[
            "decision_loss"
        ]["supported"],
        "continuum_full_beats_zero_brier": continuum_comparisons["event_brier"][
            "supported"
        ],
        "matched_coverage_full_beats_diagonal": matched_comparisons[
            "diagonal_marginal_matched"
        ]["supported"],
        "matched_coverage_full_beats_scrambled": matched_comparisons[
            "scrambled_marginal_matched"
        ]["supported"],
        "cost_robust_against_diagonal": (
            cost_support_counts["diagonal_marginal_matched"] >= minimum_cost_levels
        ),
        "cost_robust_against_scrambled": (
            cost_support_counts["scrambled_marginal_matched"] >= minimum_cost_levels
        ),
    }
    decision = {
        "gates": gates,
        "dependence_robustness_supported": all(gates.values()),
        "continuum_value_supported": (
            gates["continuum_full_beats_zero_decision_loss"]
            and gates["continuum_full_beats_zero_brier"]
        ),
        "matched_coverage_value_supported": (
            gates["matched_coverage_full_beats_diagonal"]
            and gates["matched_coverage_full_beats_scrambled"]
        ),
        "cost_robust_value_supported": (
            gates["cost_robust_against_diagonal"]
            and gates["cost_robust_against_scrambled"]
        ),
        "cost_support_counts": cost_support_counts,
        "paper_claim_authorized": False,
        "fresh_confirmation_authorized": False,
        "calibration_claim_authorized": False,
        "deployment_safety_claim_authorized": False,
    }
    summary = {
        "object_count": len(rows),
        "query_count": 5,
        "coordinate_marginal_parity_max_abs": marginal_max,
        "continuum": continuum,
        "rank_energy": rank_energy,
        "cost_sensitivity": cost_sensitivity,
        "fixed_coverage": fixed_coverage,
        "matched_full_reference_coverage": matched,
        "continuum_comparisons": continuum_comparisons,
        "matched_coverage_comparisons": matched_comparisons,
        "cost_comparisons": cost_comparisons,
    }
    return summary, decision


def run(
    *,
    protocol_path: Path,
    base_protocol_path: Path,
    recovery_runner_path: Path,
    base_runner_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    reference_result_path: Path,
    data_root: Path,
    recovery_root: Path,
    original_v6_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    data_root = data_root.resolve(strict=True)
    recovery_root = recovery_root.resolve(strict=True)
    original_v6_root = original_v6_root.resolve(strict=True)
    parent_control_root = parent_control_root.resolve(strict=True)
    frozen_root = frozen_root.resolve(strict=True)
    validate_protocol(
        protocol,
        dataset_root=data_root,
        base_protocol_path=base_protocol_path,
        recovery_root=recovery_root,
        original_v6_root=original_v6_root,
    )
    if sha256_file(base_runner_path) != ORIGINAL_V6_RUNNER_SHA256:
        raise ValueError("original v6 runner bytes changed")
    if sha256_file(reference_result_path) != REFERENCE_RESULT_FILE_SHA256:
        raise ValueError("reference result file bytes changed")

    recovery = load_module(
        recovery_runner_path,
        "deform360_bound_recovery_for_robustness",
    )
    v6 = load_module(base_runner_path, "deform360_dependence_query_v6_for_robustness")
    base_protocol = v6.read_json(base_protocol_path)
    parent_protocol = v6.read_json(parent_protocol_path)
    parent_result = v6.read_json(parent_result_path)
    reference_result = v6.read_json(reference_result_path)
    if reference_result.get("result_sha256") != REFERENCE_INTERNAL_RESULT_SHA256:
        raise ValueError("reference internal result digest changed")
    unsigned_reference = dict(reference_result)
    supplied_reference_digest = unsigned_reference.pop("result_sha256")
    if v6.canonical_digest(unsigned_reference) != supplied_reference_digest:
        raise ValueError("reference result internal digest is invalid")

    v6.validate_protocol(
        base_protocol,
        parent_control_root=parent_control_root,
        parent_protocol_path=parent_protocol_path,
        data_root=data_root,
    )
    parent_by_object = v6.validate_parent_result(
        parent_result,
        base_protocol,
        parent_result_path,
    )
    parent_binding = base_protocol["parent_confirmation"]
    v5 = v6.load_module(
        parent_control_root / str(parent_binding["runner_path"]),
        "deform360_v5_parent_for_robustness",
    )
    manifest = v5.verify_readiness(
        v6.read_json(readiness_path),
        parent_protocol,
        readiness_path,
    )
    v3, development, frozen_protocol = v5.validate_frozen_method(
        frozen_root,
        parent_protocol,
    )
    v6.base = v3.base
    audit = v6.load_module(
        parent_control_root / str(parent_binding["audit_path"]),
        "deform360_v5_audit_for_robustness",
    )
    minimum = int(parent_protocol["selection"]["minimum_complete_episodes_per_object"])

    base_evaluation = dict(base_protocol["evaluation"])
    base_evaluation["base_random_seed"] = int(
        base_protocol["evaluation"]["random_seed"]
    )
    base_evaluation["reference_fallback_cost"] = float(
        base_protocol["evaluation"]["fallback_cost"]
    )
    protocol_evaluation = protocol["evaluation"]
    for key in (
        "dependence_strengths",
        "rank_energy_fractions",
        "fallback_costs",
        "fixed_coverages",
        "bootstrap_repetitions",
        "random_seed",
        "minimum_cost_levels_for_robustness",
        "marginal_parity_tolerance",
        "reference_numeric_tolerance",
    ):
        base_evaluation[key] = protocol_evaluation[key]

    reference_by_object = {
        str(row["object_id"]): row for row in reference_result.get("objects", ())
    }
    if len(reference_by_object) != 92:
        raise ValueError("reference result does not contain 92 unique objects")

    point_rng = np.random.default_rng(int(development["statistics"]["random_seed"]))
    original_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    reference_max_difference = 0.0
    bound_receipts: list[str] = []
    for index, expected in enumerate(manifest, start=1):
        object_id = str(expected["object_id"])
        print(f"[{index}/{len(manifest)}] robustness replay {object_id}", flush=True)
        parent_row = parent_by_object[object_id]
        descriptors, drift = recovery.build_bound_descriptors(
            v3=v3,
            v5=v5,
            audit=audit,
            data_root=data_root,
            expected=expected,
            parent_row=parent_row,
            minimum_episodes=minimum,
        )
        bound_receipts.append(str(drift["bound_carrier_receipt_sha256"]))
        (
            point_row,
            capture,
            source_truth,
            target_truth,
        ) = v6.evaluate_object_with_capture(
            v3,
            descriptors,
            development,
            frozen_protocol,
            point_rng,
        )
        if v6.point_projection(point_row) != v6.point_projection(parent_row):
            raise RuntimeError(
                f"exact parent point result did not reproduce: {object_id}"
            )
        original, context = original_row(
            v6,
            object_id=object_id,
            point_row=point_row,
            capture=capture,
            source_truth=source_truth,
            target_truth=target_truth,
            drift=drift,
            evaluation=base_evaluation,
        )
        reference = reference_by_object[object_id]
        difference = nested_max_difference(
            reference_projection(original),
            reference_projection(reference),
            path=f"objects.{object_id}",
        )
        reference_max_difference = max(reference_max_difference, difference)
        if difference > float(protocol_evaluation["reference_numeric_tolerance"]):
            raise RuntimeError(
                f"reference numerical projection changed for {object_id}: {difference}"
            )
        original_rows.append(original)
        robustness_rows.append(
            robustness_row(
                v6,
                original=original,
                context=context,
                protocol=protocol,
            )
        )

    original_summary, original_decision = v6.aggregate(original_rows, base_protocol)
    reference_summary_difference = nested_max_difference(
        original_summary,
        reference_result["summary"],
        path="summary",
    )
    reference_decision_difference = nested_max_difference(
        original_decision,
        reference_result["decision"],
        path="decision",
    )
    reference_max_difference = max(
        reference_max_difference,
        reference_summary_difference,
        reference_decision_difference,
    )
    if reference_max_difference > float(
        protocol_evaluation["reference_numeric_tolerance"]
    ):
        raise RuntimeError("completed v6 reference result did not reproduce")

    summary, decision = aggregate_robustness(robustness_rows, protocol)
    reference_reproduction = {
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_artifact_id": REFERENCE_ARTIFACT_ID,
        "reference_result_file_sha256": sha256_file(reference_result_path),
        "reference_internal_result_sha256": supplied_reference_digest,
        "object_count": len(original_rows),
        "original_summary_reproduced": True,
        "original_decision_reproduced": True,
        "maximum_numeric_absolute_difference": reference_max_difference,
        "numeric_tolerance": float(protocol_evaluation["reference_numeric_tolerance"]),
        "reference_projection_sha256": canonical_digest(
            [reference_projection(row) for row in original_rows]
        ),
    }
    information_boundary = {
        "retrospective_target_reuse": True,
        "exact_bound_object_roster_reused": True,
        "exact_bound_carrier_receipts_reused": True,
        "exact_frozen_point_predictor_reused": True,
        "reference_v6_result_reproduced": True,
        "same_predictive_mean_for_all_diagnostics": True,
        "coordinate_marginals_equal_for_all_diagnostics": True,
        "source_only_calibration_reused": True,
        "target_outcomes_used_to_select_grids": False,
        "target_outcomes_used_to_refit_models": False,
        "unbound_numeric_payloads_opened": False,
        "new_measurements_collected": False,
        "paper_claim_authorized": False,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root),
        "reference_reproduction": reference_reproduction,
        "bound_carrier_receipts_sha256": canonical_digest(bound_receipts),
        "information_boundary": information_boundary,
        "summary": summary,
        "decision": decision,
        "objects": robustness_rows,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def make_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    lines = [
        "# Deform360 dependence robustness replay v1",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Queries per object: **{summary['query_count']}**",
        "- Original v6 result reproduced: **true**",
        "- Predictive means fixed: **true**",
        "- Coordinate marginals fixed: **true**",
        "- Retrospective target reuse: **true**",
        "- Dependence robustness supported: "
        f"**{str(decision['dependence_robustness_supported']).lower()}**",
        "- Matched-coverage value supported: "
        f"**{str(decision['matched_coverage_value_supported']).lower()}**",
        "- Cost-robust value supported: "
        f"**{str(decision['cost_robust_value_supported']).lower()}**",
        "",
        "## Dependence-strength continuum",
        "",
        "| Strength | Decision loss | Brier | Query NLL | Acceptance |",
        "|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["continuum"].items():
        lines.append(
            f"| {key} | {values['decision_loss']:.6g} | "
            f"{values['event_brier']:.6g} | {values['query_nll']:.6g} | "
            f"{values['acceptance_fraction']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Rank/energy path",
            "",
            "| Requested energy | Mean retained energy | Median rank | "
            "Decision loss | Brier |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for key, values in summary["rank_energy"].items():
        lines.append(
            f"| {key} | {values['mean_retained_energy_fraction']:.3%} | "
            f"{values['median_retained_rank']:.1f} | {values['decision_loss']:.6g} | "
            f"{values['event_brier']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Coverage matched to the full arm at cost 0.1",
            "",
            "| Arm | Coverage | Decision loss | Harm/all | Harm/accepted |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm, values in summary["matched_full_reference_coverage"].items():
        lines.append(
            f"| `{arm}` | {values['acceptance_fraction']:.3%} | "
            f"{values['decision_loss']:.6g} | "
            f"{values['harmful_accept_fraction_all']:.3%} | "
            f"{values['harmful_accept_rate_given_accept']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Registered gates",
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
            "This is a retrospective robustness analysis on already-open outcomes. ",
            "It does not authorize a fresh-confirmation, calibration, "
            "deployment-safety, ",
            "or automatic paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for section in ("continuum", "rank_energy"):
        for level, values in result["summary"][section].items():
            rows.append(
                {
                    "section": section,
                    "arm": "",
                    "level": level,
                    "decision_loss": values["decision_loss"],
                    "event_brier": values["event_brier"],
                    "query_nll": values["query_nll"],
                    "acceptance_fraction": values["acceptance_fraction"],
                    "harmful_accept_fraction_all": values[
                        "harmful_accept_fraction_all"
                    ],
                }
            )
    for section in ("cost_sensitivity", "fixed_coverage"):
        for arm, levels in result["summary"][section].items():
            for level, values in levels.items():
                rows.append(
                    {
                        "section": section,
                        "arm": arm,
                        "level": level,
                        "decision_loss": values["decision_loss"],
                        "event_brier": "",
                        "query_nll": "",
                        "acceptance_fraction": values["acceptance_fraction"],
                        "harmful_accept_fraction_all": values[
                            "harmful_accept_fraction_all"
                        ],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class _DummyCovariance:
    mean_error: np.ndarray
    diagonal: np.ndarray
    factor: np.ndarray
    multiplier: float = 1.0
    marginal_z: float = 1.645
    source_marginal_coverage: float = 0.9
    source_joint_nanees: float = 1.0


class _DummyBase:
    CovarianceModel = _DummyCovariance


def self_test() -> None:
    covariance = _DummyCovariance(
        mean_error=np.zeros(4),
        diagonal=np.asarray([0.2, 0.3, 0.4, 0.5]),
        factor=np.asarray(
            [
                [1.0, 0.0],
                [0.7, 0.4],
                [0.0, 1.2],
                [-0.6, 0.3],
            ]
        ),
    )
    reference_marginal = marginal_variance(covariance)
    zero = dependence_strength_model(_DummyBase, covariance, 0.0)
    half = dependence_strength_model(_DummyBase, covariance, 0.5)
    full = dependence_strength_model(_DummyBase, covariance, 1.0)
    for model in (zero, half, full):
        np.testing.assert_allclose(marginal_variance(model), reference_marginal)
    assert zero.factor.shape[1] == covariance.factor.shape[1]
    np.testing.assert_allclose(full.factor, covariance.factor)

    previous_rank = -1
    for fraction in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        model, metadata = rank_energy_model(_DummyBase, covariance, fraction)
        np.testing.assert_allclose(marginal_variance(model), reference_marginal)
        assert int(metadata["retained_rank"]) >= previous_rank
        previous_rank = int(metadata["retained_rank"])

    predicted = np.asarray([0.4, 0.1, 0.1, 0.8])
    mask = lowest_risk_mask(predicted, 2)
    assert mask.tolist() == [False, True, True, False]
    assert coverage_count(30, 0.1) == 3
    metrics = decision_metrics(
        predicted,
        np.asarray([False, False, True, True]),
        fallback_cost=0.1,
        accepted_mask=mask,
    )
    assert metrics["acceptance_fraction"] == 0.5

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        write_json(path, {"ok": True})
        assert read_json(path) == {"ok": True}
    print("deform360 dependence robustness self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--base-protocol", type=Path)
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--base-runner", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--original-v6-root", type=Path)
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
        "base_protocol",
        "recovery_runner",
        "base_runner",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "reference_result",
        "data_root",
        "recovery_root",
        "original_v6_root",
        "parent_control_root",
        "frozen_root",
        "output_json",
        "output_report",
        "output_csv",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    result = run(
        protocol_path=args.protocol,
        base_protocol_path=args.base_protocol,
        recovery_runner_path=args.recovery_runner,
        base_runner_path=args.base_runner,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
        reference_result_path=args.reference_result,
        data_root=args.data_root,
        recovery_root=args.recovery_root,
        original_v6_root=args.original_v6_root,
        parent_control_root=args.parent_control_root,
        frozen_root=args.frozen_root,
    )
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_summary_csv(args.output_csv, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(result["decision"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
