#!/usr/bin/env python3
"""Matched-coverage audit of the frozen Deform360 dependence study.

The parent experiment holds the predictive mean and every coordinate marginal
fixed while changing only cross-coordinate dependence. Its original deployment
rule applies one common adverse-event probability threshold, which naturally
produces different acceptance rates across covariance arms. This retrospective
audit reruns the exact bound-carrier experiment, captures the unchanged
per-window adverse-event probabilities, and compares the arms after forcing
identical acceptance counts on a predeclared coverage grid.

No predictor, query, covariance arm, source calibration, event threshold, or
recorded target is changed. Physical objects remain the bootstrap units. The
audit cannot establish fresh confirmation, calibration, or robot-control safety.
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

SCHEMA = "bayesian-phystwin/deform360-dependence-matched-coverage-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-dependence-matched-coverage-v1"
RECOVERY_RUNNER_SHA256 = (
    "6cca5314d3748304904ec97d97c7cd023956faf30a9fe48415684602a5add7ee"
)
REFERENCE_RESULT_FILE_SHA256 = (
    "c73659af65c2b87923f7bd668f9717afab03e449a5b3abd3a5b597ec60898fd1"
)
REFERENCE_RESULT_INTERNAL_SHA256 = (
    "d430731e56ce470a5e0df8fbd3bc13dea83763beccd2df06d86cde2365d4ee36"
)
EXPECTED_COVERAGE_GRID = tuple(value / 10.0 for value in range(1, 10))
PRIMARY_METRICS = (
    "normalized_selective_risk_auc",
    "mean_deployment_loss",
)
SECONDARY_METRICS = (
    "mean_selective_risk",
    "mean_harmful_accept_fraction_all",
    "mean_decision_regret",
)
_EPS = 1e-12


@dataclass(frozen=True)
class CapturedQuery:
    predicted: np.ndarray
    labels: np.ndarray
    prediction_sha256: str
    labels_sha256: str


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
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_protocol(protocol: dict[str, Any], data_root: Path) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected matched-coverage protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unexpected matched-coverage protocol version")
    if protocol.get("status") != "frozen-before-audit-execution":
        raise ValueError("matched-coverage audit is not frozen")
    if Path(str(protocol.get("dataset_root"))) != data_root:
        raise ValueError("dataset root changed")
    if tuple(protocol.get("runner_labels", ())) != (
        "self-hosted",
        "Linux",
        "X64",
        "gpuserver4090",
    ):
        raise ValueError("runner labels changed")

    base = protocol.get("base_experiment")
    if not isinstance(base, dict):
        raise ValueError("base experiment binding is absent")
    expected_base = {
        "workflow_run_id": 33528032875,
        "artifact_id": 9811194776,
        "artifact_zip_sha256": (
            "8b3bad2bc0620228ebe32027028b1666ea7772a1850f0ab45d525e30e4ced82a"
        ),
        "result_file_sha256": REFERENCE_RESULT_FILE_SHA256,
        "result_internal_sha256": REFERENCE_RESULT_INTERNAL_SHA256,
        "execution_revision": "28e8a44bfbab9e1556e0f51c53b46e91b8352481",
        "recovery_runner_sha256": RECOVERY_RUNNER_SHA256,
        "original_v6_revision": "954538832106d8ded13f1101b3a2b2e855b40513",
        "parent_control_revision": "e409527b8499a225bde8bd7f8c532a30e96548c6",
        "frozen_point_revision": "25ba91c021124569c4dcf84c66eda5ec088868e0",
    }
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            raise ValueError(f"base experiment binding changed: {key}")

    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation contract is absent")
    coverage_grid = tuple(float(value) for value in evaluation.get("coverage_grid", ()))
    if coverage_grid != EXPECTED_COVERAGE_GRID:
        raise ValueError("coverage grid changed")
    if evaluation.get("selection_order") != (
        "ascending predicted adverse-event probability with target-time index "
        "as deterministic tie break"
    ):
        raise ValueError("selection ordering changed")
    if evaluation.get("acceptance_count_rule") != (
        "floor(n_windows * nominal_coverage), lower-bounded by one"
    ):
        raise ValueError("acceptance-count rule changed")
    if evaluation.get("fallback_cost") != 0.1:
        raise ValueError("fallback cost changed")
    if evaluation.get("inferential_unit") != "physical object":
        raise ValueError("inferential unit changed")
    if evaluation.get("query_aggregation") != "equal mean over five registered queries":
        raise ValueError("query aggregation changed")
    if evaluation.get("coverage_aggregation") != (
        "equal-grid mean and normalized trapezoidal area over nominal coverage"
    ):
        raise ValueError("coverage aggregation changed")
    if evaluation.get("bootstrap_repetitions") != 10000:
        raise ValueError("bootstrap repetitions changed")
    if evaluation.get("bootstrap_seed") != 260903:
        raise ValueError("bootstrap seed changed")

    boundary = protocol.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("information boundary is absent")
    required_true = (
        "retrospective_target_reuse",
        "exact_base_experiment_rerun_required",
        "same_mean_and_coordinate_marginals_required",
        "matched_acceptance_counts_required",
    )
    required_false = (
        "new_measurements_collected",
        "point_predictor_may_change",
        "query_bank_may_change",
        "covariance_arms_may_change",
        "source_calibration_may_change",
        "target_outcomes_may_tune_coverage_grid",
    )
    for key in required_true:
        if boundary.get(key) is not True:
            raise ValueError(f"required information boundary is false: {key}")
    for key in required_false:
        if boundary.get(key) is not False:
            raise ValueError(f"forbidden information flow is enabled: {key}")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("audit protocol may not self-authorize a paper claim")


def validate_reference_result(path: Path) -> dict[str, Any]:
    if sha256_file(path) != REFERENCE_RESULT_FILE_SHA256:
        raise ValueError("reference result file bytes changed")
    result = read_json(path)
    if result.get("result_sha256") != REFERENCE_RESULT_INTERNAL_SHA256:
        raise ValueError("reference internal result digest changed")
    unsigned = dict(result)
    supplied = unsigned.pop("result_sha256")
    if canonical_digest(unsigned) != supplied:
        raise ValueError("reference result has an invalid internal digest")
    if result.get("status") != "complete":
        raise ValueError("reference result is incomplete")
    return result


def stable_base_projection(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "schema_version",
        "status",
        "protocol_id",
        "dataset_root",
        "parent_confirmation",
        "bound_selection_manifest_sha256",
        "recovery",
        "carrier_drift_summary",
        "information_boundary",
        "summary",
        "decision",
        "objects",
        "carrier_drift",
        "protocol",
    )
    return {key: result[key] for key in keys}


def capture_wrapper(v6: Any, captures: list[CapturedQuery]) -> Any:
    original = v6.query_metrics

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, float]:
        if args:
            raise TypeError("captured query_metrics requires keyword arguments")
        result = original(**kwargs)
        target_truth = np.asarray(kwargs["target_truth"], dtype=np.float64)
        target_errors = np.asarray(kwargs["target_errors"], dtype=np.float64)
        weight = np.asarray(kwargs["weight"], dtype=np.float64)
        event = str(kwargs["event"])
        calibration = kwargs["calibration"]
        fallback_cost = float(kwargs["fallback_cost"])
        probability_clip = float(kwargs["probability_clip"])
        model = kwargs["model"]

        raw_variance = v6.covariance_query_variance(model, weight)
        variance = raw_variance * float(calibration["shared_variance_scale"])
        threshold = float(calibration["event_threshold"])
        truth_query = target_truth @ weight
        target_error = target_errors @ weight
        target_mean = truth_query - target_error
        if event == "upper":
            labels = truth_query > threshold
        elif event == "absolute":
            labels = np.abs(truth_query) > threshold
        else:
            raise ValueError(f"unsupported event: {event}")
        predicted = np.clip(
            v6.event_probability(target_mean, variance, threshold, event),
            probability_clip,
            1.0 - probability_clip,
        ).astype(np.float64, copy=False)
        labels_float = labels.astype(np.float64)
        execute = predicted <= fallback_cost
        realized_loss = np.where(execute, labels_float, fallback_cost)
        checks = {
            "event_brier": float(np.mean(np.square(predicted - labels_float))),
            "decision_loss": float(np.mean(realized_loss)),
            "acceptance_fraction": float(np.mean(execute)),
            "harmful_accept_fraction_all": float(np.mean(execute & labels)),
        }
        for key, value in checks.items():
            if not math.isclose(value, float(result[key]), rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"capture changed original metric: {key}")
        captures.append(
            CapturedQuery(
                predicted=predicted.copy(),
                labels=labels.astype(np.bool_, copy=True),
                prediction_sha256=array_digest(predicted),
                labels_sha256=array_digest(labels.astype(np.uint8)),
            )
        )
        return result

    return wrapped


def rerun_with_capture(
    *,
    recovery_runner_path: Path,
    base_runner_path: Path,
    base_protocol_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    data_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> tuple[Any, dict[str, Any], list[CapturedQuery]]:
    if sha256_file(recovery_runner_path) != RECOVERY_RUNNER_SHA256:
        raise ValueError("bound-carrier recovery runner bytes changed")
    recovery = load_module(
        recovery_runner_path,
        "deform360_dependence_bound_recovery_for_matched_coverage",
    )
    original_loader = recovery.load_module
    captures: list[CapturedQuery] = []

    def patched_loader(path: Path, name: str) -> Any:
        module = original_loader(path, name)
        if path.resolve() == base_runner_path.resolve():
            module.query_metrics = capture_wrapper(module, captures)
        return module

    recovery.load_module = patched_loader
    try:
        v6, result = recovery.run(
            base_runner_path=base_runner_path,
            protocol_path=base_protocol_path,
            parent_protocol_path=parent_protocol_path,
            parent_result_path=parent_result_path,
            readiness_path=readiness_path,
            data_root=data_root,
            parent_control_root=parent_control_root,
            frozen_root=frozen_root,
        )
    finally:
        recovery.load_module = original_loader
    return v6, result, captures


def acceptance_count(window_count: int, nominal_coverage: float) -> int:
    if window_count <= 0:
        raise ValueError("window count must be positive")
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal coverage must be in (0,1)")
    return max(1, min(window_count, int(math.floor(window_count * nominal_coverage))))


def matched_curve(
    predicted: np.ndarray,
    labels: np.ndarray,
    coverage_grid: tuple[float, ...],
    fallback_cost: float,
) -> dict[str, Any]:
    scores = np.asarray(predicted, dtype=np.float64)
    outcomes = np.asarray(labels, dtype=np.bool_)
    if scores.ndim != 1 or outcomes.ndim != 1 or scores.shape != outcomes.shape:
        raise ValueError("scores and labels must be aligned vectors")
    if not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be nonempty and finite")
    indices = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((indices, scores))
    total_adverse = int(np.count_nonzero(outcomes))
    points: list[dict[str, Any]] = []
    for nominal in coverage_grid:
        accepted_count = acceptance_count(len(scores), nominal)
        accepted_indices = order[:accepted_count]
        harmful_count = int(np.count_nonzero(outcomes[accepted_indices]))
        rejected_count = len(scores) - accepted_count
        safe_count = len(scores) - total_adverse
        oracle_harmful = max(0, accepted_count - safe_count)
        deployment_loss = (
            harmful_count + fallback_cost * rejected_count
        ) / len(scores)
        oracle_loss = (
            oracle_harmful + fallback_cost * rejected_count
        ) / len(scores)
        points.append(
            {
                "nominal_coverage": float(nominal),
                "achieved_coverage": float(accepted_count / len(scores)),
                "accepted_count": accepted_count,
                "harmful_accepted_count": harmful_count,
                "selective_risk": float(harmful_count / accepted_count),
                "harmful_accept_fraction_all": float(harmful_count / len(scores)),
                "deployment_loss": float(deployment_loss),
                "oracle_loss_at_matched_coverage": float(oracle_loss),
                "decision_regret": float(deployment_loss - oracle_loss),
                "maximum_accepted_score": float(scores[accepted_indices[-1]]),
                "accepted_index_sha256": array_digest(accepted_indices),
            }
        )
    x = np.asarray([point["nominal_coverage"] for point in points], dtype=np.float64)
    if len(x) < 2 or not np.all(np.diff(x) > 0.0):
        raise ValueError("coverage grid must be strictly increasing")

    def values(name: str) -> np.ndarray:
        return np.asarray([point[name] for point in points], dtype=np.float64)

    span = float(x[-1] - x[0])
    risk = values("selective_risk")
    area = float(np.sum(0.5 * (risk[:-1] + risk[1:]) * np.diff(x)))
    normalized_auc = area / span
    return {
        "window_count": int(len(scores)),
        "adverse_event_count": total_adverse,
        "prediction_sha256": array_digest(scores),
        "labels_sha256": array_digest(outcomes.astype(np.uint8)),
        "ranking_sha256": array_digest(order),
        "normalized_selective_risk_auc": normalized_auc,
        "mean_selective_risk": float(np.mean(values("selective_risk"))),
        "mean_deployment_loss": float(np.mean(values("deployment_loss"))),
        "mean_harmful_accept_fraction_all": float(
            np.mean(values("harmful_accept_fraction_all"))
        ),
        "mean_decision_regret": float(np.mean(values("decision_regret"))),
        "curve": points,
    }


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    if len(vector) == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
    means = vector[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def mean_nested(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return float(np.mean([row["arm_summary"][arm][metric] for row in rows]))


def paired_comparison(
    rows: list[dict[str, Any]],
    *,
    comparator: str,
    metric: str,
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


def coverage_comparison(
    rows: list[dict[str, Any]],
    *,
    comparator: str,
    metric: str,
    coverage_index: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    differences: list[float] = []
    for row in rows:
        full_values = [
            row["queries"][name]["arms"]["full_low_rank"]["curve"][coverage_index][
                metric
            ]
            for name in row["query_names"]
        ]
        comparator_values = [
            row["queries"][name]["arms"][comparator]["curve"][coverage_index][metric]
            for name in row["query_names"]
        ]
        differences.append(float(np.mean(full_values) - np.mean(comparator_values)))
    vector = np.asarray(differences, dtype=np.float64)
    return {
        "metric": metric,
        "comparator": comparator,
        "mean_difference": float(np.mean(vector)),
        "object_bootstrap_95_interval": bootstrap_interval(
            vector,
            repetitions,
            seed,
        ),
        "object_wins": int(np.count_nonzero(vector < 0.0)),
        "object_ties": int(np.count_nonzero(vector == 0.0)),
        "object_losses": int(np.count_nonzero(vector > 0.0)),
    }


def build_audit(
    *,
    protocol: dict[str, Any],
    v6: Any,
    base_result: dict[str, Any],
    reference_result: dict[str, Any],
    captures: list[CapturedQuery],
) -> dict[str, Any]:
    if stable_base_projection(base_result) != stable_base_projection(reference_result):
        raise RuntimeError("rerun differs from the completed base experiment")
    expected_calls = (
        len(base_result["objects"]) * len(v6.QUERY_SPECS) * len(v6.COVARIANCE_ARMS)
    )
    if len(captures) != expected_calls:
        raise RuntimeError(
            f"captured {len(captures)} query calls, expected {expected_calls}"
        )

    evaluation = protocol["evaluation"]
    coverage_grid = tuple(float(value) for value in evaluation["coverage_grid"])
    fallback_cost = float(evaluation["fallback_cost"])
    cursor = 0
    object_rows: list[dict[str, Any]] = []
    all_labels_match = True
    all_acceptance_counts_match = True
    all_capture_metrics_match = True
    for base_object in base_result["objects"]:
        query_records: dict[str, Any] = {}
        for query_name, expected_event in v6.QUERY_SPECS:
            labels_reference: np.ndarray | None = None
            arm_records: dict[str, Any] = {}
            for arm_name in v6.COVARIANCE_ARMS:
                captured = captures[cursor]
                cursor += 1
                if len(captured.predicted) != int(base_object["window_count"]):
                    raise RuntimeError("capture window count differs from base result")
                base_query = base_object["queries"][query_name]
                if base_query["event"] != expected_event:
                    raise RuntimeError("captured query order differs from registration")
                base_metrics = base_query["arms"][arm_name]
                fixed_execute = captured.predicted <= fallback_cost
                fixed_loss = np.where(
                    fixed_execute,
                    captured.labels.astype(np.float64),
                    fallback_cost,
                )
                checks = {
                    "event_brier": float(
                        np.mean(
                            np.square(
                                captured.predicted
                                - captured.labels.astype(np.float64)
                            )
                        )
                    ),
                    "decision_loss": float(np.mean(fixed_loss)),
                    "acceptance_fraction": float(np.mean(fixed_execute)),
                }
                for metric, value in checks.items():
                    if not math.isclose(
                        value,
                        float(base_metrics[metric]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        all_capture_metrics_match = False
                if labels_reference is None:
                    labels_reference = captured.labels
                elif not np.array_equal(labels_reference, captured.labels):
                    all_labels_match = False
                arm_records[arm_name] = matched_curve(
                    captured.predicted,
                    captured.labels,
                    coverage_grid,
                    fallback_cost,
                )
            for coverage_index in range(len(coverage_grid)):
                counts = {
                    arm_records[arm]["curve"][coverage_index]["accepted_count"]
                    for arm in v6.COVARIANCE_ARMS
                }
                if len(counts) != 1:
                    all_acceptance_counts_match = False
            query_records[query_name] = {
                "event": expected_event,
                "labels_sha256": array_digest(
                    np.asarray(labels_reference, dtype=np.uint8)
                ),
                "arms": arm_records,
            }

        arm_summary: dict[str, dict[str, float]] = {}
        for arm_name in v6.COVARIANCE_ARMS:
            arm_summary[arm_name] = {
                metric: float(
                    np.mean(
                        [
                            query_records[name]["arms"][arm_name][metric]
                            for name, _ in v6.QUERY_SPECS
                        ]
                    )
                )
                for metric in PRIMARY_METRICS + SECONDARY_METRICS
            }
        object_rows.append(
            {
                "object_id": base_object["object_id"],
                "target_episode_id": base_object["target_episode_id"],
                "target_action_family": base_object["target_action_family"],
                "window_count": base_object["window_count"],
                "query_names": [name for name, _ in v6.QUERY_SPECS],
                "queries": query_records,
                "arm_summary": arm_summary,
            }
        )
    if cursor != len(captures):
        raise RuntimeError("not every captured query call was consumed")

    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["bootstrap_seed"])
    all_metrics = PRIMARY_METRICS + SECONDARY_METRICS
    arm_summary = {
        arm: {metric: mean_nested(object_rows, arm, metric) for metric in all_metrics}
        for arm in v6.COVARIANCE_ARMS
    }
    comparisons: dict[str, Any] = {}
    for comparator_index, comparator in enumerate(v6.COVARIANCE_ARMS[1:], start=1):
        comparisons[comparator] = {
            metric: paired_comparison(
                object_rows,
                comparator=comparator,
                metric=metric,
                repetitions=repetitions,
                seed=seed + comparator_index * 1000 + metric_index,
            )
            for metric_index, metric in enumerate(all_metrics)
        }

    coverage_summary: list[dict[str, Any]] = []
    for coverage_index, nominal in enumerate(coverage_grid):
        arm_values: dict[str, Any] = {}
        for arm_name in v6.COVARIANCE_ARMS:
            metric_values: dict[str, list[float]] = {
                "achieved_coverage": [],
                "selective_risk": [],
                "deployment_loss": [],
                "harmful_accept_fraction_all": [],
                "decision_regret": [],
            }
            for object_row in object_rows:
                for query_name in object_row["query_names"]:
                    point = object_row["queries"][query_name]["arms"][arm_name][
                        "curve"
                    ][coverage_index]
                    for metric in metric_values:
                        metric_values[metric].append(float(point[metric]))
            arm_values[arm_name] = {
                metric: float(np.mean(values))
                for metric, values in metric_values.items()
            }
        per_coverage_comparisons: dict[str, Any] = {}
        for comparator_index, comparator in enumerate(
            v6.COVARIANCE_ARMS[1:],
            start=1,
        ):
            per_coverage_comparisons[comparator] = {
                metric: coverage_comparison(
                    object_rows,
                    comparator=comparator,
                    metric=metric,
                    coverage_index=coverage_index,
                    repetitions=repetitions,
                    seed=(
                        seed
                        + 10000
                        + coverage_index * 100
                        + comparator_index * 10
                        + metric_index
                    ),
                )
                for metric_index, metric in enumerate(
                    ("selective_risk", "deployment_loss")
                )
            }
        coverage_summary.append(
            {
                "nominal_coverage": nominal,
                "arms": arm_values,
                "comparisons": per_coverage_comparisons,
            }
        )

    superiority_gates: dict[str, bool] = {}
    for comparator in v6.COVARIANCE_ARMS[1:]:
        for metric in PRIMARY_METRICS:
            comparison = comparisons[comparator][metric]
            superiority_gates[f"full_{metric}_better_than_{comparator}"] = (
                comparison["mean_difference"] < 0.0
                and comparison["object_bootstrap_95_interval"][1] < 0.0
            )
    gates = {
        "reference_base_result_reproduced_exactly": True,
        "complete_92_object_roster": len(object_rows) == 92,
        "captured_all_1380_query_arm_calls": len(captures) == 1380,
        "same_event_labels_across_covariance_arms": all_labels_match,
        "captured_metrics_reproduce_original_fixed_threshold_metrics": (
            all_capture_metrics_match
        ),
        "acceptance_counts_match_at_every_object_query_coverage": (
            all_acceptance_counts_match
        ),
        **superiority_gates,
    }
    decision = {
        "gates": gates,
        "matched_coverage_value_supported": all(superiority_gates.values()),
        "calibration_claim_authorized": False,
        "fresh_confirmation_authorized": False,
        "paper_claim_authorized": False,
        "robot_control_claim_authorized": False,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": base_result["dataset_root"],
        "protocol": protocol,
        "base_experiment": {
            "reference_file_sha256": REFERENCE_RESULT_FILE_SHA256,
            "reference_internal_sha256": REFERENCE_RESULT_INTERNAL_SHA256,
            "reference_stable_projection_sha256": canonical_digest(
                stable_base_projection(reference_result)
            ),
            "rerun_stable_projection_sha256": canonical_digest(
                stable_base_projection(base_result)
            ),
            "rerun_internal_sha256": base_result["result_sha256"],
            "stable_projection_exact": True,
        },
        "information_boundary": {
            "retrospective_target_reuse": True,
            "exact_bound_carrier_base_experiment_rerun": True,
            "base_point_predictions_queries_covariances_and_calibration_changed": False,
            "target_outcomes_used_to_choose_coverage_grid": False,
            "same_mean_and_coordinate_marginals_preserved": True,
            "matched_acceptance_counts_enforced": True,
            "new_measurements_collected": False,
        },
        "summary": {
            "object_count": len(object_rows),
            "query_count": len(v6.QUERY_SPECS),
            "arm_count": len(v6.COVARIANCE_ARMS),
            "captured_query_arm_calls": len(captures),
            "coverage_grid": list(coverage_grid),
            "arm_summary": arm_summary,
            "comparisons": comparisons,
            "coverage_summary": coverage_summary,
        },
        "decision": decision,
        "objects": object_rows,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def make_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    lines = [
        "# Deform360 matched-coverage dependence audit v1",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Registered queries per object: **{summary['query_count']}**",
        f"- Captured query-arm calls: **{summary['captured_query_arm_calls']}**",
        "- Predictive means: **identical across covariance arms**",
        "- Coordinate marginals: **identical across covariance arms**",
        "- Acceptance counts: **matched at every object/query/coverage point**",
        "- Base result stable projection reproduced: **true**",
        "- Matched-coverage dependence value supported: "
        f"**{str(decision['matched_coverage_value_supported']).lower()}**",
        "",
        "## Coverage-integrated results",
        "",
        "Lower is better. Each physical object receives equal weight after equal",
        "averaging over the five registered queries and the frozen coverage grid.",
        "",
        "| Arm | Selective-risk AUC | Mean deployment loss | Mean harm/all |",
        "|---|---:|---:|---:|",
    ]
    for arm, values in summary["arm_summary"].items():
        lines.append(
            f"| `{arm}` | {values['normalized_selective_risk_auc']:.6f} | "
            f"{values['mean_deployment_loss']:.6f} | "
            f"{values['mean_harmful_accept_fraction_all']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired object-bootstrap contrasts",
            "",
            "Negative differences favor full low-rank dependence.",
            "",
            "| Comparator | Metric | Difference | 95% object bootstrap | W/T/L |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparator, metrics in summary["comparisons"].items():
        for metric in PRIMARY_METRICS:
            values = metrics[metric]
            interval = values["object_bootstrap_95_interval"]
            lines.append(
                f"| `{comparator}` | `{metric}` | "
                f"{values['mean_difference']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
                f"{values['object_wins']}/{values['object_ties']}/"
                f"{values['object_losses']} |"
            )
    lines.extend(
        [
            "",
            "## Matched coverage curve",
            "",
            "| Coverage | Full risk | Diagonal risk | Scrambled risk | "
            "Full deployment | Diagonal deployment | Scrambled deployment |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["coverage_summary"]:
        arms = row["arms"]
        lines.append(
            f"| {row['nominal_coverage']:.0%} | "
            f"{arms['full_low_rank']['selective_risk']:.6f} | "
            f"{arms['diagonal_marginal_matched']['selective_risk']:.6f} | "
            f"{arms['scrambled_marginal_matched']['selective_risk']:.6f} | "
            f"{arms['full_low_rank']['deployment_loss']:.6f} | "
            f"{arms['diagonal_marginal_matched']['deployment_loss']:.6f} | "
            f"{arms['scrambled_marginal_matched']['deployment_loss']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Audit gates",
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
            "This is a retrospective matched-coverage audit on already opened",
            "Deform360 targets. It tests whether dependence improves the ranking of",
            "candidate-versus-fallback risk after removing the acceptance-rate",
            "confound. It does not establish fresh confirmation, calibrated",
            "uncertainty, robot-control safety, or an automatically authorized paper",
            "claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_curve_csv(path: Path, result: dict[str, Any]) -> None:
    fields = (
        "object_id",
        "query_name",
        "arm",
        "nominal_coverage",
        "achieved_coverage",
        "accepted_count",
        "harmful_accepted_count",
        "selective_risk",
        "harmful_accept_fraction_all",
        "deployment_loss",
        "decision_regret",
        "maximum_accepted_score",
        "prediction_sha256",
        "labels_sha256",
        "ranking_sha256",
        "accepted_index_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for object_row in result["objects"]:
            for query_name in object_row["query_names"]:
                for arm, arm_result in object_row["queries"][query_name]["arms"].items():
                    for point in arm_result["curve"]:
                        writer.writerow(
                            {
                                "object_id": object_row["object_id"],
                                "query_name": query_name,
                                "arm": arm,
                                **{
                                    key: point[key]
                                    for key in fields
                                    if key in point
                                },
                                "prediction_sha256": arm_result[
                                    "prediction_sha256"
                                ],
                                "labels_sha256": arm_result["labels_sha256"],
                                "ranking_sha256": arm_result["ranking_sha256"],
                            }
                        )


def self_test() -> None:
    grid = (0.2, 0.4, 0.6, 0.8)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.bool_)
    full = np.asarray([0.05, 0.90, 0.10, 0.80, 0.15, 0.70, 0.20, 0.60, 0.25, 0.30])
    weak = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    full_result = matched_curve(full, labels, grid, 0.1)
    weak_result = matched_curve(weak, labels, grid, 0.1)
    assert full_result["normalized_selective_risk_auc"] < weak_result[
        "normalized_selective_risk_auc"
    ]
    for full_point, weak_point in zip(
        full_result["curve"],
        weak_result["curve"],
        strict=True,
    ):
        assert full_point["accepted_count"] == weak_point["accepted_count"]
    tied = matched_curve(np.zeros(10), labels, grid, 0.1)
    first_two = np.asarray([0, 1], dtype=np.int64)
    assert tied["curve"][0]["accepted_index_sha256"] == array_digest(first_two)
    values = np.asarray([-1.0, -2.0, -3.0])
    assert bootstrap_interval(values, 100, 7) == bootstrap_interval(values, 100, 7)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        value = {"status": "self-test", "metric": full_result}
        write_json(path, value)
        assert read_json(path)["status"] == "self-test"
    print("matched-coverage audit self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-protocol", type=Path)
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--base-runner", type=Path)
    parser.add_argument("--base-protocol", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--base-rerun-output", type=Path)
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
        "base_protocol",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "reference_result",
        "data_root",
        "parent_control_root",
        "frozen_root",
        "base_rerun_output",
        "output_json",
        "output_report",
        "output_csv",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    data_root = args.data_root.resolve(strict=True)
    protocol = read_json(args.audit_protocol)
    validate_protocol(protocol, data_root)
    reference_result = validate_reference_result(args.reference_result)
    v6, base_result, captures = rerun_with_capture(
        recovery_runner_path=args.recovery_runner,
        base_runner_path=args.base_runner,
        base_protocol_path=args.base_protocol,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
        data_root=data_root,
        parent_control_root=args.parent_control_root,
        frozen_root=args.frozen_root,
    )
    write_json(args.base_rerun_output, base_result)
    result = build_audit(
        protocol=protocol,
        v6=v6,
        base_result=base_result,
        reference_result=reference_result,
        captures=captures,
    )
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_curve_csv(args.output_csv, result)
    print(json.dumps(result["summary"]["arm_summary"], indent=2, sort_keys=True))
    print(json.dumps(result["summary"]["comparisons"], indent=2, sort_keys=True))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
