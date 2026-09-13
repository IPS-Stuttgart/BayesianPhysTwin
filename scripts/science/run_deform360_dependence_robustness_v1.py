#!/usr/bin/env python3
"""Retrospective robustness audit for the frozen Deform360 dependence result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-dependence-robustness-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-dependence-robustness-protocol-v1"
RECOVERY_REVISION = "28e8a44bfbab9e1556e0f51c53b46e91b8352481"
RECOVERY_SHA256 = "6cca5314d3748304904ec97d97c7cd023956faf30a9fe48415684602a5add7ee"
REFERENCE_RUN_ID = 33528032875
REFERENCE_ARTIFACT_ID = 9811194776
REFERENCE_RESULT_SHA256 = "c73659af65c2b87923f7bd668f9717afab03e449a5b3abd3a5b597ec60898fd1"
ARMS = (
    "full_low_rank",
    "diagonal_marginal_matched",
    "scrambled_marginal_matched",
)
REGIMES = ("equal_budget_query", "heldout_query_transfer")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_protocol(value: dict[str, Any]) -> None:
    if value.get("schema") != PROTOCOL_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("unexpected robustness protocol")
    if value.get("status") != "frozen-before-trigger":
        raise ValueError("robustness protocol is not frozen")
    parent = value.get("parent", {})
    expected = {
        "recovery_revision": RECOVERY_REVISION,
        "recovery_runner_sha256": RECOVERY_SHA256,
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_artifact_id": REFERENCE_ARTIFACT_ID,
        "reference_result_sha256": REFERENCE_RESULT_SHA256,
    }
    if any(parent.get(key) != item for key, item in expected.items()):
        raise ValueError("parent evidence binding changed")
    evaluation = value.get("evaluation", {})
    if tuple(evaluation.get("covariance_arms", ())) != ARMS:
        raise ValueError("covariance arms changed")
    if tuple(evaluation.get("calibration_regimes", ())) != REGIMES:
        raise ValueError("calibration regimes changed")
    if evaluation.get("inferential_unit") != "physical object":
        raise ValueError("inferential unit changed")


def event_labels(values: np.ndarray, threshold: float, event: str) -> np.ndarray:
    if event == "upper":
        return values > threshold
    if event == "absolute":
        return np.abs(values) > threshold
    raise ValueError(f"unsupported event type: {event}")


def gaussian_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return np.fromiter(
        (_NORMAL.cdf(float(value)) for value in flat),
        dtype=float,
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


class CaptureHooks:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.arm_index = 0

    def install(self, module: Any) -> None:
        original_calibration = module.source_query_calibration
        original_metrics = module.query_metrics

        def calibration_wrapper(
            centered_source_errors: np.ndarray,
            source_truth: np.ndarray,
            weight: np.ndarray,
            raw_variances: dict[str, float],
            *,
            event: str,
            probability: float,
            event_quantile: float,
        ) -> dict[str, float]:
            result = original_calibration(
                centered_source_errors,
                source_truth,
                weight,
                raw_variances,
                event=event,
                probability=probability,
                event_quantile=event_quantile,
            )
            source_error = np.asarray(centered_source_errors) @ np.asarray(weight)
            source_truth_query = np.asarray(source_truth) @ np.asarray(weight)
            self.current = {
                "event": event,
                "threshold": float(result["event_threshold"]),
                "source_error": source_error.copy(),
                "source_mean": (source_truth_query - source_error).copy(),
                "raw_variances": {
                    name: float(raw_variances[name]) for name in ARMS
                },
                "target_error": None,
                "target_mean": None,
                "labels": None,
            }
            self.records.append(self.current)
            self.arm_index = 0
            return result

        def metrics_wrapper(**kwargs: Any) -> dict[str, float]:
            result = original_metrics(**kwargs)
            if self.current is None or self.arm_index >= len(ARMS):
                raise RuntimeError("query capture order changed")
            arm = ARMS[self.arm_index]
            self.arm_index += 1
            raw = float(module.covariance_query_variance(kwargs["model"], kwargs["weight"]))
            if not math.isclose(
                raw,
                self.current["raw_variances"][arm],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"covariance arm order changed: {arm}")
            target_error = np.asarray(kwargs["target_errors"]) @ np.asarray(
                kwargs["weight"]
            )
            target_truth = np.asarray(kwargs["target_truth"]) @ np.asarray(
                kwargs["weight"]
            )
            target_mean = target_truth - target_error
            labels = event_labels(
                target_truth,
                self.current["threshold"],
                self.current["event"],
            )
            if self.current["target_error"] is None:
                self.current["target_error"] = target_error.copy()
                self.current["target_mean"] = target_mean.copy()
                self.current["labels"] = labels.copy()
            elif not (
                np.array_equal(target_error, self.current["target_error"])
                and np.array_equal(target_mean, self.current["target_mean"])
                and np.array_equal(labels, self.current["labels"])
            ):
                raise RuntimeError("same-mean target arrays changed across arms")
            return result

        module.source_query_calibration = calibration_wrapper
        module.query_metrics = metrics_wrapper


def scientific_projection(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in result["objects"]:
        retained = dict(row)
        retained.pop("bound_carrier_recovery", None)
        rows.append(retained)
    return {
        "summary": result["summary"],
        "decision": result["decision"],
        "objects": rows,
    }


def align_records(v6: Any, result: dict[str, Any], records: list[dict[str, Any]]) -> None:
    expected = len(result["objects"]) * len(v6.QUERY_SPECS)
    if len(records) != expected:
        raise RuntimeError(f"captured {len(records)} queries, expected {expected}")
    cursor = 0
    for object_row in result["objects"]:
        for query_name, event in v6.QUERY_SPECS:
            record = records[cursor]
            cursor += 1
            if record["event"] != event:
                raise RuntimeError("query order changed")
            record["object_id"] = str(object_row["object_id"])
            record["query_name"] = str(query_name)
            if any(record[key] is None for key in ("target_error", "target_mean", "labels")):
                raise RuntimeError("target capture is incomplete")


def fit_calibration(samples: list[tuple[np.ndarray, float]]) -> dict[str, float]:
    normalized_square = np.concatenate(
        [np.square(error) / max(raw, _EPS) for error, raw in samples]
    )
    scale = max(float(np.mean(normalized_square)), 1e-8)
    standardized = np.concatenate(
        [np.abs(error) / math.sqrt(max(raw * scale, _EPS)) for error, raw in samples]
    )
    return {
        "scale": scale,
        "radius": max(float(np.quantile(standardized, 0.9)), 1e-8),
        "source_count": int(len(standardized)),
    }


def build_calibrations(
    records: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    queries = tuple(dict.fromkeys(record["query_name"] for record in records))
    result = {regime: {} for regime in REGIMES}
    for arm in ARMS:
        for query in queries:
            same_query = [
                (record["source_error"], record["raw_variances"][arm])
                for record in records
                if record["query_name"] == query
            ]
            other_queries = [
                (record["source_error"], record["raw_variances"][arm])
                for record in records
                if record["query_name"] != query
            ]
            result["equal_budget_query"][(arm, query)] = fit_calibration(same_query)
            result["heldout_query_transfer"][(arm, query)] = fit_calibration(
                other_queries
            )
    return result


def query_score(
    record: dict[str, Any],
    arm: str,
    calibration: dict[str, float],
    probability_clip: float,
) -> dict[str, Any]:
    variance = max(record["raw_variances"][arm] * calibration["scale"], _EPS)
    source_probability = np.clip(
        event_probability(
            record["source_mean"],
            variance,
            record["threshold"],
            record["event"],
        ),
        probability_clip,
        1.0 - probability_clip,
    )
    target_probability = np.clip(
        event_probability(
            record["target_mean"],
            variance,
            record["threshold"],
            record["event"],
        ),
        probability_clip,
        1.0 - probability_clip,
    )
    labels = record["labels"].astype(float)
    normalized_square = np.square(record["target_error"]) / variance
    return {
        "source_probability": source_probability,
        "target_probability": target_probability,
        "labels": record["labels"],
        "brier": float(np.mean(np.square(target_probability - labels))),
        "nll": float(
            np.mean(0.5 * (math.log(2.0 * math.pi * variance) + normalized_square))
        ),
        "coverage": float(
            np.mean(
                np.abs(record["target_error"])
                <= calibration["radius"] * math.sqrt(variance)
            )
        ),
    }


def policy_metrics(mask: np.ndarray, labels: np.ndarray, cost: float) -> dict[str, float]:
    execute = np.asarray(mask, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    realized = np.where(execute, truth.astype(float), cost)
    accepted = int(np.count_nonzero(execute))
    harmful = int(np.count_nonzero(execute & truth))
    return {
        "acceptance": float(np.mean(execute)),
        "decision_loss": float(np.mean(realized)),
        "harm_fraction_all": float(harmful / len(execute)),
        "harm_rate_given_accept": float(harmful / accepted if accepted else 0.0),
    }


def exact_coverage_mask(probability: np.ndarray, coverage: float) -> np.ndarray:
    count = min(max(int(round(coverage * len(probability))), 0), len(probability))
    order = np.argsort(probability, kind="mergesort")
    result = np.zeros(len(probability), dtype=bool)
    result[order[:count]] = True
    return result


def bootstrap(values: np.ndarray, repetitions: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    draws = np.mean(array[indices], axis=1)
    return {
        "mean_difference": float(np.mean(array)),
        "interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "object_wins": int(np.count_nonzero(array < -1e-15)),
        "object_ties": int(np.count_nonzero(np.abs(array) <= 1e-15)),
        "object_losses": int(np.count_nonzero(array > 1e-15)),
    }


def analyze(records: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    cost = float(evaluation["fallback_cost"])
    matched_coverage = float(evaluation["matched_coverage"])
    probability_clip = float(evaluation["probability_clip"])
    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["random_seed"])
    objects = tuple(dict.fromkeys(record["object_id"] for record in records))
    calibrations = build_calibrations(records)
    object_rows = []
    for object_id in objects:
        object_records = [record for record in records if record["object_id"] == object_id]
        row: dict[str, Any] = {"object_id": object_id, "regimes": {}}
        for regime in REGIMES:
            row["regimes"][regime] = {}
            for arm in ARMS:
                scores = [
                    query_score(
                        record,
                        arm,
                        calibrations[regime][(arm, record["query_name"])],
                        probability_clip,
                    )
                    for record in object_records
                ]
                source_probability = np.concatenate(
                    [score["source_probability"] for score in scores]
                )
                target_probability = np.concatenate(
                    [score["target_probability"] for score in scores]
                )
                labels = np.concatenate([score["labels"] for score in scores])
                fixed = policy_metrics(target_probability <= cost, labels, cost)
                threshold = float(np.quantile(source_probability, matched_coverage))
                source_frozen = policy_metrics(
                    target_probability <= threshold,
                    labels,
                    cost,
                )
                source_frozen["source_threshold"] = threshold
                target_rank = policy_metrics(
                    exact_coverage_mask(target_probability, matched_coverage),
                    labels,
                    cost,
                )
                row["regimes"][regime][arm] = {
                    "brier": float(np.mean([score["brier"] for score in scores])),
                    "nll": float(np.mean([score["nll"] for score in scores])),
                    "coverage_90": float(
                        np.mean([score["coverage"] for score in scores])
                    ),
                    "fixed_cost": fixed,
                    "source_frozen_matched_coverage": source_frozen,
                    "target_rank_matched_coverage": target_rank,
                }
        object_rows.append(row)

    aggregate: dict[str, Any] = {"regimes": {}, "comparisons": {}}
    for regime_index, regime in enumerate(REGIMES):
        aggregate["regimes"][regime] = {}
        aggregate["comparisons"][regime] = {}
        for arm in ARMS:
            arm_rows = [row["regimes"][regime][arm] for row in object_rows]
            aggregate["regimes"][regime][arm] = {
                "brier": float(np.mean([item["brier"] for item in arm_rows])),
                "nll": float(np.mean([item["nll"] for item in arm_rows])),
                "coverage_90": float(
                    np.mean([item["coverage_90"] for item in arm_rows])
                ),
                "fixed_cost": {
                    key: float(np.mean([item["fixed_cost"][key] for item in arm_rows]))
                    for key in (
                        "acceptance",
                        "decision_loss",
                        "harm_fraction_all",
                        "harm_rate_given_accept",
                    )
                },
                "source_frozen_matched_coverage": {
                    key: float(
                        np.mean(
                            [
                                item["source_frozen_matched_coverage"][key]
                                for item in arm_rows
                            ]
                        )
                    )
                    for key in (
                        "acceptance",
                        "decision_loss",
                        "harm_fraction_all",
                        "harm_rate_given_accept",
                    )
                },
                "target_rank_matched_coverage": {
                    key: float(
                        np.mean(
                            [
                                item["target_rank_matched_coverage"][key]
                                for item in arm_rows
                            ]
                        )
                    )
                    for key in (
                        "acceptance",
                        "decision_loss",
                        "harm_fraction_all",
                        "harm_rate_given_accept",
                    )
                },
            }
        for control_index, control in enumerate(ARMS[1:], start=1):
            comparison = {}
            metrics = {
                "brier": lambda item: item["brier"],
                "nll": lambda item: item["nll"],
                "fixed_cost_decision_loss": lambda item: item["fixed_cost"][
                    "decision_loss"
                ],
                "source_frozen_decision_loss": lambda item: item[
                    "source_frozen_matched_coverage"
                ]["decision_loss"],
                "target_rank_decision_loss": lambda item: item[
                    "target_rank_matched_coverage"
                ]["decision_loss"],
                "target_rank_harm": lambda item: item[
                    "target_rank_matched_coverage"
                ]["harm_fraction_all"],
            }
            for metric_index, (metric, getter) in enumerate(metrics.items()):
                differences = np.asarray(
                    [
                        getter(row["regimes"][regime]["full_low_rank"])
                        - getter(row["regimes"][regime][control])
                        for row in object_rows
                    ]
                )
                comparison[metric] = bootstrap(
                    differences,
                    repetitions,
                    seed + 1000 * regime_index + 100 * control_index + metric_index,
                )
            aggregate["comparisons"][regime][f"full_vs_{control}"] = comparison
    return {
        "object_count": len(objects),
        "query_count": len(records) // len(objects),
        "matched_coverage": matched_coverage,
        "fallback_cost": cost,
        "object_results": object_rows,
        "aggregate": aggregate,
    }


def make_report(result: dict[str, Any]) -> str:
    robust = result["robustness"]
    primary = robust["aggregate"]["comparisons"]["equal_budget_query"]
    lines = [
        "# Deform360 dependence robustness audit",
        "",
        f"- Parent scientific projection reproduced: **{str(result['parent_reproduction']['scientific_projection_equal']).lower()}**",
        f"- Objects: **{robust['object_count']}**",
        f"- Query families: **{robust['query_count']}**",
        f"- Matched coverage: **{100 * robust['matched_coverage']:.1f}%**",
        "- Calibration: **one source-only scale per arm and query family**",
        "- Status: **retrospective robustness evidence**",
        "",
        "Negative differences favor full low-rank dependence.",
        "",
        "| Comparator | Brier difference [95%] | Fixed-cost decision difference [95%] | Equal-coverage decision difference [95%] |",
        "|---|---:|---:|---:|",
    ]
    for key, metrics in primary.items():
        label = key.removeprefix("full_vs_")
        values = []
        for metric in ("brier", "fixed_cost_decision_loss", "target_rank_decision_loss"):
            row = metrics[metric]
            values.append(
                f"{row['mean_difference']:.6f} [{row['interval_95'][0]:.6f}, {row['interval_95'][1]:.6f}]"
            )
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "The exact-target-rank policy is descriptive only. The source-frozen",
            "matched-coverage and leave-one-query-out results are retained in",
            "`result.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.recovery_runner) != RECOVERY_SHA256:
        raise ValueError("recovery runner bytes changed")
    if sha256_file(args.reference_result) != REFERENCE_RESULT_SHA256:
        raise ValueError("reference result bytes changed")
    protocol = read_json(args.robustness_protocol)
    validate_protocol(protocol)
    reference = read_json(args.reference_result)
    recovery = load_module(args.recovery_runner, "deform360_recovery_for_robustness")
    hooks = CaptureHooks()
    original_load: Callable[..., Any] = recovery.load_module

    def wrapped_load(path: Path, name: str) -> Any:
        module = original_load(path, name)
        if name == "deform360_dependence_query_v6_original":
            hooks.install(module)
        return module

    recovery.load_module = wrapped_load
    try:
        v6, reproduced = recovery.run(
            base_runner_path=args.base_runner,
            protocol_path=args.parent_scientific_protocol,
            parent_protocol_path=args.parent_protocol,
            parent_result_path=args.parent_result,
            readiness_path=args.readiness_json,
            data_root=args.data_root,
            parent_control_root=args.parent_control_root,
            frozen_root=args.frozen_root,
        )
    finally:
        recovery.load_module = original_load
    align_records(v6, reproduced, hooks.records)
    reference_projection = scientific_projection(reference)
    reproduced_projection = scientific_projection(reproduced)
    if reproduced_projection != reference_projection:
        raise RuntimeError("parent scientific projection did not reproduce")
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(args.data_root.resolve()),
        "protocol": protocol,
        "parent_reproduction": {
            "reference_run_id": REFERENCE_RUN_ID,
            "reference_artifact_id": REFERENCE_ARTIFACT_ID,
            "reference_result_sha256": REFERENCE_RESULT_SHA256,
            "recovery_revision": RECOVERY_REVISION,
            "scientific_projection_equal": True,
            "scientific_projection_sha256": canonical_digest(reference_projection),
            "parent_point_result_reproduced_exactly": reproduced[
                "information_boundary"
            ]["parent_point_result_reproduced_exactly"],
        },
        "information_boundary": {
            "retrospective_target_reuse": True,
            "point_predictor_changed": False,
            "predictive_means_changed": False,
            "coordinate_marginals_changed": False,
            "query_bank_changed": False,
            "source_only_calibration": True,
            "target_rank_policy_is_descriptive_only": True,
            "fresh_confirmation_claimed": False,
            "paper_claim_authorized": False,
        },
        "robustness": analyze(hooks.records, protocol),
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def self_test() -> None:
    samples = [(np.asarray([1.0, -1.0]), 2.0), (np.asarray([0.5, -0.5]), 0.5)]
    calibration = fit_calibration(samples)
    assert math.isclose(calibration["scale"], 0.5)
    mask = exact_coverage_mask(np.asarray([0.4, 0.1, 0.3, 0.2]), 0.5)
    assert mask.tolist() == [False, True, False, True]
    metrics = policy_metrics(mask, np.asarray([False, True, True, False]), 0.1)
    assert math.isclose(metrics["decision_loss"], 0.3)
    sampled = bootstrap(np.asarray([-1.0, -2.0, -3.0]), 1000, 7)
    assert sampled["object_wins"] == 3
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "value.json"
        write_json(path, {"ok": True})
        assert read_json(path) == {"ok": True}
    print("deform360 dependence robustness self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--base-runner", type=Path)
    parser.add_argument("--parent-scientific-protocol", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--robustness-protocol", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "recovery_runner",
        "base_runner",
        "parent_scientific_protocol",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "data_root",
        "parent_control_root",
        "frozen_root",
        "reference_result",
        "robustness_protocol",
        "output_json",
        "output_report",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    result = run(args)
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    print(json.dumps(result["parent_reproduction"], indent=2, sort_keys=True))
    print(
        json.dumps(
            result["robustness"]["aggregate"]["comparisons"],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
