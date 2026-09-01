#!/usr/bin/env python3
"""Retrospective same-mean dependence ablation on Deform360 tactile queries.

The exact frozen v3 point predictor is rerun on the previously completed
92-object Deform360 cohort. Every covariance arm receives the same predictive
mean and exactly the same coordinate marginals. The arms differ only in
cross-coordinate dependence:

* the original low-rank-plus-diagonal covariance;
* a marginal-matched diagonal covariance; and
* a marginal-matched covariance whose low-rank row directions are
  deterministically scrambled.

For each physical query, a scalar variance scale and a nominal-90% radius are
fitted from leave-one-source-episode-out residuals only. Target metrics include
query NLL, nANEES, coverage, event-probability scores, and the realized loss of
an execute-versus-fallback risk decision. This is retrospective mechanism
evidence because the target episodes were already opened by the v5 point
confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-dependence-query-result-v6"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-dependence-query-protocol-v6"
COVARIANCE_ARMS = (
    "full_low_rank",
    "diagonal_marginal_matched",
    "scrambled_marginal_matched",
)
QUERY_SPECS = (
    ("total_load", "upper"),
    ("sensor_imbalance", "absolute"),
    ("horizontal_balance", "absolute"),
    ("vertical_balance", "absolute"),
    ("center_periphery", "upper"),
)
FIELD_ROWS = 6
FIELD_COLUMNS = 16
FIELD_DIMENSION_PER_SENSOR = FIELD_ROWS * FIELD_COLUMNS
_EPS = 1e-12
_NORMAL = NormalDist()


@dataclass
class EvaluationCapture:
    source: list[Any] | None = None
    candidates: list[Any] | None = None
    weights: np.ndarray | None = None
    source_residuals: np.ndarray | None = None
    transform: Any | None = None
    covariance: Any | None = None
    target_errors: np.ndarray | None = None


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
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


def validate_protocol(
    protocol: dict[str, Any],
    *,
    parent_control_root: Path,
    parent_protocol_path: Path,
    data_root: Path,
) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected dependence-query protocol schema")
    if protocol.get("schema_version") != 6:
        raise ValueError("unexpected dependence-query protocol version")
    if protocol.get("status") != "frozen-before-v6-reexecution":
        raise ValueError("dependence-query protocol is not frozen")
    if Path(str(protocol.get("dataset_root"))) != data_root:
        raise ValueError("dataset root changed")
    parent = protocol.get("parent_confirmation")
    if not isinstance(parent, dict):
        raise ValueError("parent confirmation binding is absent")
    if git_output(parent_control_root, "rev-parse", "HEAD") != parent.get(
        "control_revision"
    ):
        raise ValueError("parent confirmation control revision changed")
    relative = parent_protocol_path.resolve().relative_to(parent_control_root.resolve())
    if str(relative) != parent.get("protocol_path"):
        raise ValueError("parent protocol path changed")
    if git_output(parent_control_root, "hash-object", str(relative)) != parent.get(
        "protocol_git_blob_sha1"
    ):
        raise ValueError("parent protocol Git blob changed")
    for path_key, blob_key in (
        ("runner_path", "runner_git_blob_sha1"),
        ("audit_path", "audit_git_blob_sha1"),
    ):
        path = Path(str(parent[path_key]))
        if git_output(parent_control_root, "hash-object", str(path)) != parent.get(
            blob_key
        ):
            raise ValueError(f"parent confirmation {path_key} changed")
    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation contract is absent")
    if tuple(evaluation.get("covariance_arms", ())) != COVARIANCE_ARMS:
        raise ValueError("covariance arm roster changed")
    registered_queries = tuple(
        (str(item.get("name")), str(item.get("event")))
        for item in evaluation.get("query_bank", ())
        if isinstance(item, dict)
    )
    if registered_queries != QUERY_SPECS:
        raise ValueError("physical query roster changed")
    if evaluation.get("same_mean_required") is not True:
        raise ValueError("same-mean requirement must be enabled")
    if evaluation.get("coordinate_marginals_equal_required") is not True:
        raise ValueError("marginal-matching requirement must be enabled")
    if evaluation.get("source_only_query_calibration") is not True:
        raise ValueError("query calibration must be source-only")
    probability = float(evaluation.get("coverage_probability", math.nan))
    quantile = float(evaluation.get("event_threshold_quantile", math.nan))
    fallback = float(evaluation.get("fallback_cost", math.nan))
    if not 0.0 < probability < 1.0:
        raise ValueError("coverage probability must be in (0,1)")
    if not 0.5 < quantile < 1.0:
        raise ValueError("event threshold quantile must be in (0.5,1)")
    if not 0.0 < fallback < 1.0:
        raise ValueError("fallback cost must be in (0,1)")
    boundary = protocol.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("information boundary is absent")
    if boundary.get("retrospective_target_reuse") is not True:
        raise ValueError("retrospective target reuse must be explicit")
    if boundary.get("point_predictor_may_change") is not False:
        raise ValueError("point predictor changes are forbidden")
    if boundary.get("target_outcomes_may_tune_protocol") is not False:
        raise ValueError("target outcomes may not tune the protocol")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("protocol may not self-authorize a paper claim")


def validate_parent_result(
    result: dict[str, Any],
    protocol: dict[str, Any],
    result_path: Path,
) -> dict[str, dict[str, Any]]:
    binding = protocol["parent_confirmation"]
    if sha256_file(result_path) != binding["result_json_sha256"]:
        raise ValueError("parent result file bytes changed")
    if result.get("schema") != (
        "bayesian-phystwin/deform360-untouched-confirmation-result-v5"
    ):
        raise ValueError("unexpected parent result schema")
    if result.get("status") != "complete":
        raise ValueError("parent point confirmation is incomplete")
    stored = result.get("result_sha256")
    unsigned = dict(result)
    unsigned.pop("result_sha256", None)
    if canonical_digest(unsigned) != stored:
        raise ValueError("parent result digest is invalid")
    if stored != binding["result_sha256"]:
        raise ValueError("parent result binding changed")
    if int(result["summary"]["object_count"]) != int(binding["object_count"]):
        raise ValueError("parent object count changed")
    rows = result.get("objects")
    if not isinstance(rows, list):
        raise ValueError("parent object rows are absent")
    by_object = {str(row["object_id"]): row for row in rows}
    if len(by_object) != int(binding["object_count"]):
        raise ValueError("parent object rows are incomplete or duplicated")
    return by_object


def stable_seed(seed: int, object_id: str, purpose: str) -> int:
    payload = f"{seed}\0{object_id}\0{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def query_bank(dimension: int) -> dict[str, tuple[np.ndarray, str]]:
    if dimension % FIELD_DIMENSION_PER_SENSOR:
        raise ValueError(
            "pooled field dimension is not a whole number of tactile grids"
        )
    sensor_count = dimension // FIELD_DIMENSION_PER_SENSOR
    if sensor_count < 2:
        raise ValueError("at least two tactile grids are required")
    weights: dict[str, tuple[np.ndarray, str]] = {}

    total = np.full(dimension, 1.0 / dimension, dtype=np.float64)
    weights["total_load"] = (total, "upper")

    sensor = np.zeros((sensor_count, FIELD_DIMENSION_PER_SENSOR), dtype=np.float64)
    sensor[0] = 1.0 / FIELD_DIMENSION_PER_SENSOR
    sensor[-1] -= 1.0 / FIELD_DIMENSION_PER_SENSOR
    weights["sensor_imbalance"] = (sensor.reshape(-1), "absolute")

    horizontal = np.zeros((sensor_count, FIELD_ROWS, FIELD_COLUMNS), dtype=np.float64)
    half = sensor_count * FIELD_ROWS * (FIELD_COLUMNS // 2)
    horizontal[:, :, : FIELD_COLUMNS // 2] = -1.0 / half
    horizontal[:, :, FIELD_COLUMNS // 2 :] = 1.0 / half
    weights["horizontal_balance"] = (horizontal.reshape(-1), "absolute")

    vertical = np.zeros((sensor_count, FIELD_ROWS, FIELD_COLUMNS), dtype=np.float64)
    half = sensor_count * (FIELD_ROWS // 2) * FIELD_COLUMNS
    vertical[:, : FIELD_ROWS // 2, :] = -1.0 / half
    vertical[:, FIELD_ROWS // 2 :, :] = 1.0 / half
    weights["vertical_balance"] = (vertical.reshape(-1), "absolute")

    center = np.zeros((sensor_count, FIELD_ROWS, FIELD_COLUMNS), dtype=np.float64)
    center_mask = np.zeros((FIELD_ROWS, FIELD_COLUMNS), dtype=bool)
    center_mask[2:4, 4:12] = True
    center_count = sensor_count * int(np.count_nonzero(center_mask))
    perimeter_count = dimension - center_count
    center[:, center_mask] = 1.0 / center_count
    center[:, ~center_mask] = -1.0 / perimeter_count
    weights["center_periphery"] = (center.reshape(-1), "upper")

    if tuple((name, event) for name, (_, event) in weights.items()) != QUERY_SPECS:
        raise RuntimeError("constructed query bank differs from registration")
    for name, (weight, _) in weights.items():
        if weight.shape != (dimension,) or not np.all(np.isfinite(weight)):
            raise RuntimeError(f"invalid query weights: {name}")
    return weights


def marginal_variance(model: Any) -> np.ndarray:
    return np.asarray(model.multiplier) * (
        np.asarray(model.diagonal)
        + np.sum(np.asarray(model.factor) * np.asarray(model.factor), axis=1)
    )


def covariance_query_variance(model: Any, weight: np.ndarray) -> float:
    diagonal = float(model.multiplier) * np.asarray(model.diagonal)
    factor = math.sqrt(float(model.multiplier)) * np.asarray(model.factor)
    value = float(np.sum(weight * weight * diagonal))
    if factor.shape[1]:
        projected = weight @ factor
        value += float(projected @ projected)
    return max(value, _EPS)


def scrambled_factor(
    factor: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    source = np.asarray(factor, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("factor must be a matrix")
    result = np.zeros_like(source)
    if source.shape[1] == 0:
        return result
    norms = np.linalg.norm(source, axis=1)
    indices = np.flatnonzero(norms > _EPS)
    if len(indices) == 0:
        return result
    directions = source[indices] / norms[indices, None]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(indices))
    if len(indices) > 1 and np.array_equal(permutation, np.arange(len(indices))):
        permutation = np.roll(permutation, 1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(indices))
    result[indices] = norms[indices, None] * directions[permutation] * signs[:, None]
    return result


def covariance_arms(base: Any, covariance: Any, *, seed: int) -> dict[str, Any]:
    factor = np.asarray(covariance.factor, dtype=np.float64)
    diagonal = np.asarray(covariance.diagonal, dtype=np.float64)
    total = diagonal + np.sum(factor * factor, axis=1)
    empty = np.empty((len(total), 0), dtype=np.float64)
    scrambled = scrambled_factor(factor, seed=seed)
    result = {
        "full_low_rank": covariance,
        "diagonal_marginal_matched": base.CovarianceModel(
            np.asarray(covariance.mean_error).copy(),
            total.copy(),
            empty,
            float(covariance.multiplier),
            float(covariance.marginal_z),
            float(covariance.source_marginal_coverage),
            float(covariance.source_joint_nanees),
        ),
        "scrambled_marginal_matched": base.CovarianceModel(
            np.asarray(covariance.mean_error).copy(),
            diagonal.copy(),
            scrambled,
            float(covariance.multiplier),
            float(covariance.marginal_z),
            float(covariance.source_marginal_coverage),
            float(covariance.source_joint_nanees),
        ),
    }
    reference = marginal_variance(covariance)
    for name, model in result.items():
        if not np.allclose(
            marginal_variance(model),
            reference,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(f"coordinate marginals changed for {name}")
    return result


def source_query_calibration(
    centered_source_errors: np.ndarray,
    source_truth: np.ndarray,
    weight: np.ndarray,
    raw_variances: dict[str, float],
    *,
    event: str,
    probability: float,
    event_quantile: float,
) -> dict[str, float]:
    """Freeze one arm-symmetric scalar calibration from source episodes only.

    The full low-rank arm is the registered primary model. Its source residuals
    determine one query scale and radius that are then reused byte-for-byte by
    both dependence-destruction controls. No control may be independently
    recalibrated, so every arm retains the same predictive mean, coordinate
    marginals, and scalar calibration; only cross-coordinate dependence differs.
    """

    if tuple(raw_variances) != COVARIANCE_ARMS:
        raise ValueError("raw query variances do not match the arm roster")
    reference_variance = float(raw_variances["full_low_rank"])
    if not np.isfinite(reference_variance) or reference_variance <= 0.0:
        raise ValueError("reference query variance must be finite and positive")
    query_error = centered_source_errors @ weight
    source_mse = float(np.mean(query_error * query_error))
    scale = max(source_mse / reference_variance, 1e-8)
    standardized = np.abs(query_error) / math.sqrt(reference_variance * scale)
    radius_multiplier = max(float(np.quantile(standardized, probability)), 1e-8)

    projected_truth = source_truth @ weight
    if event == "upper":
        threshold_values = projected_truth
        threshold = float(np.quantile(threshold_values, event_quantile))
        source_event_rate = float(np.mean(projected_truth > threshold))
    elif event == "absolute":
        threshold_values = np.abs(projected_truth)
        threshold = float(np.quantile(threshold_values, event_quantile))
        source_event_rate = float(np.mean(np.abs(projected_truth) > threshold))
    else:
        raise ValueError(f"unsupported event type: {event}")

    return {
        "reference_raw_query_variance": reference_variance,
        "shared_variance_scale": scale,
        "shared_radius_multiplier": radius_multiplier,
        "source_reference_nanees": float(source_mse / (reference_variance * scale)),
        "source_reference_coverage": float(np.mean(standardized <= radius_multiplier)),
        "event_threshold": threshold,
        "source_event_rate": source_event_rate,
    }


def gaussian_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.fromiter(
        (_NORMAL.cdf(float(value)) for value in flat),
        dtype=np.float64,
        count=len(flat),
    ).reshape(np.asarray(values).shape)


def event_probability(
    mean: np.ndarray,
    variance: float,
    threshold: float,
    event: str,
) -> np.ndarray:
    standard_deviation = math.sqrt(max(variance, _EPS))
    if event == "upper":
        return 1.0 - gaussian_cdf((threshold - mean) / standard_deviation)
    if event == "absolute":
        lower = gaussian_cdf((-threshold - mean) / standard_deviation)
        upper = 1.0 - gaussian_cdf((threshold - mean) / standard_deviation)
        return lower + upper
    raise ValueError(f"unsupported event type: {event}")


def query_metrics(
    *,
    centered_source_errors: np.ndarray,
    target_truth: np.ndarray,
    target_errors: np.ndarray,
    weight: np.ndarray,
    event: str,
    model: Any,
    calibration: dict[str, float],
    fallback_cost: float,
    probability_clip: float,
) -> dict[str, float]:
    raw_variance = covariance_query_variance(model, weight)
    scale = float(calibration["shared_variance_scale"])
    radius_multiplier = float(calibration["shared_radius_multiplier"])
    threshold = float(calibration["event_threshold"])
    variance = raw_variance * scale

    source_query_error = centered_source_errors @ weight
    source_standardized_square = source_query_error * source_query_error / variance
    source_radius = radius_multiplier * math.sqrt(variance)

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
        event_probability(target_mean, variance, threshold, event),
        probability_clip,
        1.0 - probability_clip,
    )
    labels_float = labels.astype(np.float64)
    execute = predicted <= fallback_cost
    realized_loss = np.where(execute, labels_float, fallback_cost)
    oracle_loss = np.where(labels, fallback_cost, 0.0)
    standardized_square = target_error * target_error / variance
    radius = radius_multiplier * math.sqrt(variance)
    accepted = int(np.count_nonzero(execute))
    harmful_accepted = int(np.count_nonzero(execute & labels))
    return {
        "raw_query_variance": raw_variance,
        "shared_variance_scale": scale,
        "shared_radius_multiplier": radius_multiplier,
        "source_query_nanees": float(np.mean(source_standardized_square)),
        "source_query_coverage": float(
            np.mean(np.abs(source_query_error) <= source_radius)
        ),
        "calibrated_query_variance": variance,
        "event_threshold": threshold,
        "source_event_rate": float(calibration["source_event_rate"]),
        "target_event_rate": float(np.mean(labels_float)),
        "target_query_nanees": float(np.mean(standardized_square)),
        "target_90_coverage": float(np.mean(np.abs(target_error) <= radius)),
        "mean_90_interval_width": float(2.0 * radius),
        "query_nll": float(
            np.mean(0.5 * (math.log(2.0 * math.pi * variance) + standardized_square))
        ),
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
        "harmful_accept_fraction_all": float(harmful_accepted / len(labels)),
        "harmful_accept_rate_given_accept": float(
            harmful_accepted / accepted if accepted else 0.0
        ),
    }


def joint_metrics(
    base: Any,
    errors: np.ndarray,
    model: Any,
    probability: float,
) -> dict[str, float]:
    dimension = errors.shape[1]
    normal = _NORMAL.inv_cdf(0.5 + probability / 2.0)
    chi_square = (
        dimension
        * (1.0 - 2.0 / (9.0 * dimension) + normal * math.sqrt(2.0 / (9.0 * dimension)))
        ** 3
    )
    quadratics = np.asarray(
        [base.woodbury_quadratic(error, model) for error in errors],
        dtype=np.float64,
    )
    marginal = marginal_variance(model)
    radius = float(model.marginal_z) * np.sqrt(marginal)
    logdet = base.covariance_logdet(model)
    nll = 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + quadratics) / dimension
    return {
        "joint_nanees": float(np.mean(quadratics) / dimension),
        "joint_90_ellipsoid_coverage": float(np.mean(quadratics <= chi_square)),
        "marginal_90_coverage": float(np.mean(np.abs(errors) <= radius[None, :])),
        "mean_marginal_90_width": float(2.0 * np.mean(radius)),
        "nll_per_dimension": float(np.mean(nll)),
    }


def evaluate_object_with_capture(
    v3: Any,
    descriptors: list[Any],
    development: dict[str, Any],
    base_protocol: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], EvaluationCapture, np.ndarray, np.ndarray]:
    capture = EvaluationCapture()
    original_ensemble = v3.ensemble_source_residuals
    original_fit = v3.base.fit_covariance
    original_metrics = v3.base.probabilistic_metrics
    original_transform = v3.base.build_transform

    def wrapped_ensemble(source: list[Any], candidates: list[Any], weights: np.ndarray):
        result = original_ensemble(source, candidates, weights)
        capture.source = source
        capture.candidates = candidates
        capture.weights = np.asarray(weights, dtype=np.float64).copy()
        capture.source_residuals = np.asarray(result[0], dtype=np.float64).copy()
        return result

    def wrapped_fit(*args: Any, **kwargs: Any):
        model = original_fit(*args, **kwargs)
        capture.covariance = model
        return model

    def wrapped_metrics(errors: np.ndarray, *args: Any, **kwargs: Any):
        capture.target_errors = np.asarray(errors, dtype=np.float64).copy()
        return original_metrics(errors, *args, **kwargs)

    def wrapped_transform(*args: Any, **kwargs: Any):
        value = original_transform(*args, **kwargs)
        capture.transform = value
        return value

    v3.ensemble_source_residuals = wrapped_ensemble
    v3.base.fit_covariance = wrapped_fit
    v3.base.probabilistic_metrics = wrapped_metrics
    v3.base.build_transform = wrapped_transform
    try:
        row = v3.evaluate_object(descriptors, development, base_protocol, rng)
    finally:
        v3.ensemble_source_residuals = original_ensemble
        v3.base.fit_covariance = original_fit
        v3.base.probabilistic_metrics = original_metrics
        v3.base.build_transform = original_transform

    required = (
        capture.source,
        capture.candidates,
        capture.source_residuals,
        capture.transform,
        capture.covariance,
        capture.target_errors,
    )
    if any(value is None for value in required):
        raise RuntimeError("frozen evaluator capture is incomplete")

    source = capture.source
    candidates = capture.candidates
    source_truth = np.concatenate(
        [
            np.asarray(
                candidates[0].cv_truths[episode.descriptor.episode_id],
                dtype=np.float64,
            )
            for episode in source
        ],
        axis=0,
    )
    if source_truth.shape != capture.source_residuals.shape:
        raise RuntimeError("source truth and captured residual shapes disagree")

    target_descriptor = max(descriptors, key=lambda item: item.episode_id)
    target = v3.base.load_episode(target_descriptor)
    horizon = int(development["shared_preprocessing"]["forecast_horizon_frames"])
    target_rows = v3.episode_rows(
        target,
        capture.transform,
        base_protocol,
        horizon,
    )
    target_truth = np.asarray(target_rows[4], dtype=np.float64)
    if target_truth.shape != capture.target_errors.shape:
        raise RuntimeError("target truth and captured error shapes disagree")
    return row, capture, source_truth, target_truth


def point_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_fit_id": row["source_fit_id"],
        "target_episode_id": row["target_episode_id"],
        "forecast_window_count": row["forecast_window_count"],
        "pooled_field_dimension": row["pooled_field_dimension"],
        "candidate_names": row["candidate_names"],
        "candidate_families": row["candidate_families"],
        "candidate_weights": row["candidate_weights"],
        "selected_state_ridge": row["selected_state_ridge"],
        "selected_state_kernel": row["selected_state_kernel"],
        "selected_action_ridge": row["selected_action_ridge"],
        "selected_action_kernel": row["selected_action_kernel"],
        "source_cv_active_rmse": row["source_cv_active_rmse"],
        "guard_accepts": row["guard_accepts"],
        "fallback_method": row["fallback_method"],
        "metrics": row["metrics"],
        "uncertainty": row["uncertainty"],
    }


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) == 1:
        return [float(vector[0]), float(vector[0])]
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
            row["arm_summary"]["full_low_rank"][metric]
            - row["arm_summary"][comparator][metric]
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = protocol["evaluation"]
    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["random_seed"])
    metrics = (
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
        "calibration_log_error",
        "coverage_absolute_error",
    )
    arm_summary = {
        arm: {
            metric: float(np.mean([row["arm_summary"][arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in COVARIANCE_ARMS
    }
    joint = {
        arm: {
            metric: float(np.mean([row["joint_metrics"][arm][metric] for row in rows]))
            for metric in rows[0]["joint_metrics"][arm]
        }
        for arm in COVARIANCE_ARMS
    }
    comparisons: dict[str, Any] = {}
    for comparator_index, comparator in enumerate(COVARIANCE_ARMS[1:], start=1):
        comparisons[comparator] = {
            metric: paired_summary(
                rows,
                metric,
                comparator,
                repetitions=repetitions,
                seed=seed + 1000 * comparator_index + metric_index,
            )
            for metric_index, metric in enumerate(
                ("decision_loss", "event_brier", "query_nll", "calibration_log_error")
            )
        }
    query_summary: dict[str, Any] = {}
    for query_name, _ in QUERY_SPECS:
        query_summary[query_name] = {
            arm: {
                metric: float(
                    np.mean(
                        [
                            row["queries"][query_name]["arms"][arm][metric]
                            for row in rows
                        ]
                    )
                )
                for metric in (
                    "target_query_nanees",
                    "target_90_coverage",
                    "query_nll",
                    "event_brier",
                    "decision_loss",
                    "acceptance_fraction",
                    "harmful_accept_fraction_all",
                )
            }
            for arm in COVARIANCE_ARMS
        }

    marginal_max = float(max(row["coordinate_marginal_parity_max_abs"] for row in rows))
    point_parity = all(bool(row["parent_point_result_exact"]) for row in rows)
    full = arm_summary["full_low_rank"]
    diagonal = comparisons["diagonal_marginal_matched"]
    scrambled = comparisons["scrambled_marginal_matched"]
    gates = {
        "complete_92_object_roster": len(rows) == 92,
        "exact_parent_point_result_reproduced": point_parity,
        "same_mean_for_every_covariance_arm": all(
            bool(row["same_mean_by_construction"]) for row in rows
        ),
        "coordinate_marginals_match": (
            marginal_max <= float(evaluation["marginal_parity_tolerance"])
        ),
        "full_decision_loss_better_than_diagonal": (
            diagonal["decision_loss"]["mean_difference"] < 0.0
            and diagonal["decision_loss"]["object_bootstrap_95_interval"][1] < 0.0
        ),
        "full_decision_loss_better_than_scrambled": (
            scrambled["decision_loss"]["mean_difference"] < 0.0
            and scrambled["decision_loss"]["object_bootstrap_95_interval"][1] < 0.0
        ),
        "full_brier_better_than_diagonal": (
            diagonal["event_brier"]["mean_difference"] < 0.0
            and diagonal["event_brier"]["object_bootstrap_95_interval"][1] < 0.0
        ),
        "full_brier_better_than_scrambled": (
            scrambled["event_brier"]["mean_difference"] < 0.0
            and scrambled["event_brier"]["object_bootstrap_95_interval"][1] < 0.0
        ),
        "full_query_nanees_in_registered_range": (
            float(protocol["success_gates"]["minimum_query_nanees"])
            <= full["target_query_nanees"]
            <= float(protocol["success_gates"]["maximum_query_nanees"])
        ),
        "full_query_coverage_in_registered_range": (
            float(protocol["success_gates"]["minimum_query_coverage"])
            <= full["target_90_coverage"]
            <= float(protocol["success_gates"]["maximum_query_coverage"])
        ),
    }
    decision = {
        "gates": gates,
        "superior_target_reached": all(gates.values()),
        "dependence_value_supported": all(
            gates[name]
            for name in (
                "full_decision_loss_better_than_diagonal",
                "full_decision_loss_better_than_scrambled",
                "full_brier_better_than_diagonal",
                "full_brier_better_than_scrambled",
            )
        ),
        "query_calibration_supported": (
            gates["full_query_nanees_in_registered_range"]
            and gates["full_query_coverage_in_registered_range"]
        ),
        "paper_claim_authorized": False,
        "fresh_confirmation_authorized": False,
        "strict_counterfactual_claim_authorized": False,
    }
    summary = {
        "object_count": len(rows),
        "query_count": len(QUERY_SPECS),
        "covariance_arms": list(COVARIANCE_ARMS),
        "arm_summary": arm_summary,
        "joint_field_summary": joint,
        "comparisons": comparisons,
        "query_summary": query_summary,
        "coordinate_marginal_parity_max_abs": marginal_max,
    }
    return summary, decision


def make_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    lines = [
        "# Deform360 same-mean dependence-query study v6",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Queries per object: **{summary['query_count']}**",
        "- Point predictor: **exact frozen v3**",
        "- Predictive mean across covariance arms: **identical by construction**",
        "- Coordinate marginals across covariance arms: **identical**",
        "- Dependence value supported: "
        f"**{str(decision['dependence_value_supported']).lower()}**",
        "- Query calibration supported: "
        f"**{str(decision['query_calibration_supported']).lower()}**",
        "- Superior target reached: "
        f"**{str(decision['superior_target_reached']).lower()}**",
        "",
        "## Object-balanced query results",
        "",
        "| Arm | Query nANEES | 90% coverage | NLL | Brier | Decision loss "
        "| Acceptance | Harm/all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in COVARIANCE_ARMS:
        values = summary["arm_summary"][arm]
        lines.append(
            f"| `{arm}` | {values['target_query_nanees']:.6g} | "
            f"{values['target_90_coverage']:.3%} | {values['query_nll']:.6g} | "
            f"{values['event_brier']:.6g} | {values['decision_loss']:.6g} | "
            f"{values['acceptance_fraction']:.3%} | "
            f"{values['harmful_accept_fraction_all']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Dependence-only paired contrasts",
            "",
            "Negative values favor the full low-rank covariance.",
            "",
            "| Comparator | Metric | Difference | 95% object bootstrap | W/T/L |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparator, metrics in summary["comparisons"].items():
        for metric, values in metrics.items():
            interval = values["object_bootstrap_95_interval"]
            lines.append(
                f"| `{comparator}` | `{metric}` | {values['mean_difference']:.6g} | "
                f"[{interval[0]:.6g}, {interval[1]:.6g}] | "
                f"{values['object_wins']}/{values['object_ties']}/"
                f"{values['object_losses']} |"
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
            "The target episodes were already opened by the v5 point-confirmation "
            "study.",
            "This v6 execution is therefore retrospective mechanism evidence. It "
            "isolates",
            "cross-coordinate dependence by preserving the exact predictive mean "
            "and every",
            "coordinate marginal, fitting all query scales and thresholds from source",
            "episodes only, and treating physical objects as the inferential units.",
            "",
        ]
    )
    return "\n".join(lines)


def write_object_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "object_id",
        "target_episode_id",
        "target_action_family",
        "dimension",
        "window_count",
        "parent_point_result_exact",
        "coordinate_marginal_parity_max_abs",
    ]
    for arm in COVARIANCE_ARMS:
        for metric in (
            "target_query_nanees",
            "target_90_coverage",
            "query_nll",
            "event_brier",
            "decision_loss",
            "acceptance_fraction",
            "harmful_accept_fraction_all",
        ):
            fields.append(f"{arm}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {name: row[name] for name in fields[:7]}
            for arm in COVARIANCE_ARMS:
                for metric in fields[7:]:
                    prefix = f"{arm}_"
                    if metric.startswith(prefix):
                        output[metric] = row["arm_summary"][arm][metric[len(prefix) :]]
            writer.writerow(output)


def run(
    *,
    protocol_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    data_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    parent_protocol = read_json(parent_protocol_path)
    parent_result = read_json(parent_result_path)
    data_root = data_root.resolve(strict=True)
    parent_control_root = parent_control_root.resolve(strict=True)
    frozen_root = frozen_root.resolve(strict=True)
    validate_protocol(
        protocol,
        parent_control_root=parent_control_root,
        parent_protocol_path=parent_protocol_path,
        data_root=data_root,
    )
    parent_by_object = validate_parent_result(
        parent_result, protocol, parent_result_path
    )

    parent_binding = protocol["parent_confirmation"]
    v5_path = parent_control_root / str(parent_binding["runner_path"])
    v5 = load_module(v5_path, "deform360_v5_parent_for_dependence_query")
    manifest = v5.verify_readiness(
        read_json(readiness_path),
        parent_protocol,
        readiness_path,
    )
    v3, development, base_protocol = v5.validate_frozen_method(
        frozen_root,
        parent_protocol,
    )
    audit_path = parent_control_root / str(parent_binding["audit_path"])
    audit = load_module(audit_path, "deform360_v5_audit_for_dependence_query")
    minimum = int(parent_protocol["selection"]["minimum_complete_episodes_per_object"])

    evaluation = protocol["evaluation"]
    point_rng = np.random.default_rng(int(development["statistics"]["random_seed"]))
    rows: list[dict[str, Any]] = []
    recomputed_manifest: list[dict[str, Any]] = []
    for index, expected in enumerate(manifest, start=1):
        object_id = str(expected["object_id"])
        print(
            f"[{index}/{len(manifest)}] dependence-query evaluation {object_id}",
            flush=True,
        )
        current = audit.inspect_object(data_root, object_id, minimum)
        if not current.get("eligible"):
            raise ValueError(f"object lost carrier eligibility: {object_id}")
        projection = v5.selection_projection(current)
        if projection != expected:
            raise ValueError(f"bound carrier manifest changed: {object_id}")
        recomputed_manifest.append(projection)

        descriptors = v3.base.discover_object(data_root, object_id, minimum)
        row, capture, source_truth, target_truth = evaluate_object_with_capture(
            v3,
            descriptors,
            development,
            base_protocol,
            point_rng,
        )
        parent_row = parent_by_object[object_id]
        exact_point = point_projection(row) == point_projection(parent_row)
        if not exact_point:
            raise RuntimeError(
                f"exact frozen point result did not reproduce: {object_id}"
            )

        target_errors = np.asarray(capture.target_errors, dtype=np.float64)
        source_errors = np.asarray(capture.source_residuals, dtype=np.float64)
        predicted_mean = target_truth - target_errors
        arms = covariance_arms(
            v3.base,
            capture.covariance,
            seed=stable_seed(
                int(evaluation["random_seed"]),
                object_id,
                "scrambled-factor",
            ),
        )
        reference_marginal = marginal_variance(arms["full_low_rank"])
        marginal_parity = float(
            max(
                np.max(np.abs(marginal_variance(model) - reference_marginal))
                for model in arms.values()
            )
        )
        queries: dict[str, Any] = {}
        bank = query_bank(target_truth.shape[1])
        centered_source_errors = source_errors - source_errors.mean(
            axis=0, keepdims=True
        )
        for query_name, (weight, event) in bank.items():
            raw_variances = {
                arm_name: covariance_query_variance(model, weight)
                for arm_name, model in arms.items()
            }
            calibration = source_query_calibration(
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
                "weight_sha256": array_digest(weight),
                "calibration": calibration,
                "arms": {},
            }
            for arm_name, model in arms.items():
                query_record["arms"][arm_name] = query_metrics(
                    centered_source_errors=centered_source_errors,
                    target_truth=target_truth,
                    target_errors=target_errors,
                    weight=weight,
                    event=event,
                    model=model,
                    calibration=calibration,
                    fallback_cost=float(evaluation["fallback_cost"]),
                    probability_clip=float(evaluation["probability_clip"]),
                )
            queries[query_name] = query_record

        arm_summary: dict[str, dict[str, float]] = {}
        for arm_name in COVARIANCE_ARMS:
            values = [
                queries[query_name]["arms"][arm_name] for query_name, _ in QUERY_SPECS
            ]
            arm_summary[arm_name] = {
                metric: float(np.mean([value[metric] for value in values]))
                for metric in (
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
            }
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

        result_row = {
            "object_id": object_id,
            "source_episode_ids": row["source_episode_ids"],
            "target_episode_id": row["target_episode_id"],
            "target_action": row["target_action"],
            "target_action_family": row["target_action_family"],
            "dimension": int(target_truth.shape[1]),
            "window_count": int(target_truth.shape[0]),
            "predictive_mean_sha256": array_digest(predicted_mean),
            "same_mean_by_construction": True,
            "parent_point_result_exact": exact_point,
            "coordinate_marginal_parity_max_abs": marginal_parity,
            "query_bank_sha256": canonical_digest(
                {
                    name: {
                        "event": event,
                        "weight_sha256": queries[name]["weight_sha256"],
                    }
                    for name, event in QUERY_SPECS
                }
            ),
            "queries": queries,
            "arm_summary": arm_summary,
            "joint_metrics": {
                name: joint_metrics(
                    v3.base,
                    target_errors,
                    model,
                    float(evaluation["coverage_probability"]),
                )
                for name, model in arms.items()
            },
        }
        rows.append(result_row)

    recomputed_manifest_sha = canonical_digest(recomputed_manifest)
    if (
        recomputed_manifest_sha
        != parent_protocol["readiness_binding"]["selection_manifest_sha256"]
    ):
        raise RuntimeError("recomputed manifest digest changed")
    summary, decision = aggregate(rows, protocol)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 6,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root),
        "parent_confirmation": protocol["parent_confirmation"],
        "selection_manifest_recomputed_sha256": recomputed_manifest_sha,
        "information_boundary": {
            "retrospective_target_reuse": True,
            "exact_frozen_v3_point_predictor_reused": True,
            "parent_point_result_reproduced_exactly": all(
                row["parent_point_result_exact"] for row in rows
            ),
            "same_mean_across_covariance_arms": True,
            "coordinate_marginals_matched": True,
            "query_scales_thresholds_and_radii_source_only": True,
            "target_outcomes_used_for_protocol_or_arm_selection": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "new_measurements_collected": False,
        },
        "summary": summary,
        "decision": decision,
        "objects": rows,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-control-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
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
