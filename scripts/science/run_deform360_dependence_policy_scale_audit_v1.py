#!/usr/bin/env python3
"""Separate Deform360 dependence probability-scale value from ranking value.

The runner repeats the exact bound-carrier v6 same-mean experiment, instruments
its unchanged query scorer, and compares the original Bayes cost decision with
two label-blind equal-coverage policies. Point predictions, coordinate
marginals, query definitions, calibration, and the original scalar results must
reproduce exactly. This is retrospective mechanism evidence only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-dependence-policy-scale-audit-v1"
AUDIT_KEY = "_policy_scale_audit_v1"
ARMS = (
    "full_low_rank",
    "diagonal_marginal_matched",
    "scrambled_marginal_matched",
)
COMPARATORS = ARMS[1:]
POLICIES = (
    "fixed_cost_original",
    "source_rate_equal_coverage",
    "full_target_quota_equal_coverage",
)
REFERENCE_RESULT_SHA256 = (
    "c73659af65c2b87923f7bd668f9717afab03e449a5b3abd3a5b597ec60898fd1"
)
_EPS = 1e-15


@dataclass
class InstrumentationState:
    source_mean: np.ndarray | None = None
    source_labels: np.ndarray | None = None
    event: str | None = None
    weight_sha256: str | None = None


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


def event_labels(values: np.ndarray, threshold: float, event: str) -> np.ndarray:
    if event == "upper":
        return np.asarray(values > threshold, dtype=np.bool_)
    if event == "absolute":
        return np.asarray(np.abs(values) > threshold, dtype=np.bool_)
    raise ValueError(f"unsupported event type: {event}")


def instrument_v6(v6: Any) -> None:
    """Append audit arrays while leaving original scalar outputs unchanged."""

    state = InstrumentationState()
    original_calibration = v6.source_query_calibration
    original_metrics = v6.query_metrics

    def wrapped_calibration(
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
        truth_query = np.asarray(source_truth) @ weight
        error_query = np.asarray(centered_source_errors) @ weight
        threshold = float(result["event_threshold"])
        state.source_mean = np.asarray(truth_query - error_query, dtype=np.float64)
        state.source_labels = event_labels(truth_query, threshold, event)
        state.event = event
        state.weight_sha256 = v6.array_digest(weight)
        return result

    def wrapped_metrics(
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
    ) -> dict[str, Any]:
        result = original_metrics(
            centered_source_errors=centered_source_errors,
            target_truth=target_truth,
            target_errors=target_errors,
            weight=weight,
            event=event,
            model=model,
            calibration=calibration,
            fallback_cost=fallback_cost,
            probability_clip=probability_clip,
        )
        if state.source_mean is None or state.source_labels is None:
            raise RuntimeError("source calibration was not captured")
        if state.event != event or state.weight_sha256 != v6.array_digest(weight):
            raise RuntimeError("source and target query contexts differ")

        variance = v6.covariance_query_variance(model, weight) * float(
            calibration["shared_variance_scale"]
        )
        threshold = float(calibration["event_threshold"])
        target_truth_query = np.asarray(target_truth) @ weight
        target_error_query = np.asarray(target_errors) @ weight
        target_mean = target_truth_query - target_error_query
        source_probability = np.clip(
            v6.event_probability(state.source_mean, variance, threshold, event),
            probability_clip,
            1.0 - probability_clip,
        )
        target_probability = np.clip(
            v6.event_probability(target_mean, variance, threshold, event),
            probability_clip,
            1.0 - probability_clip,
        )
        enriched = dict(result)
        enriched[AUDIT_KEY] = {
            "source_probability": source_probability.tolist(),
            "source_labels": state.source_labels.tolist(),
            "target_probability": target_probability.tolist(),
            "target_labels": event_labels(
                target_truth_query, threshold, event
            ).tolist(),
            "fallback_cost": float(fallback_cost),
        }
        return enriched

    v6.source_query_calibration = wrapped_calibration
    v6.query_metrics = wrapped_metrics


def scientific_projection(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for key in (
        "result_sha256",
        "github_sha",
        "runner_name",
        "carrier_drift",
        "carrier_drift_summary",
        "_reference_path",
    ):
        result.pop(key, None)
    for row in result.get("objects", []):
        if not isinstance(row, dict):
            continue
        row.pop("bound_carrier_recovery", None)
        for query in row.get("queries", {}).values():
            for arm in query.get("arms", {}).values():
                arm.pop(AUDIT_KEY, None)
    return result


def stable_lowest_mask(probability: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite vector")
    if not 0 <= count <= len(values):
        raise ValueError("selection count is invalid")
    mask = np.zeros(len(values), dtype=np.bool_)
    if count:
        order = np.lexsort((np.arange(len(values), dtype=np.int64), values))
        mask[order[:count]] = True
    return mask


def proportional_count(source_count: int, source_size: int, target_size: int) -> int:
    if source_size <= 0 or target_size <= 0:
        raise ValueError("source and target sizes must be positive")
    numerator = 2 * source_count * target_size + source_size
    return min(target_size, numerator // (2 * source_size))


def decision_loss(mask: np.ndarray, labels: np.ndarray, cost: float) -> float:
    selected = np.asarray(mask, dtype=np.bool_)
    truth = np.asarray(labels, dtype=np.bool_)
    if selected.shape != truth.shape or selected.ndim != 1 or not len(selected):
        raise ValueError("selection and label vectors must be aligned")
    return float(np.mean(np.where(selected, truth.astype(np.float64), cost)))


def rank_vector(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    order = np.lexsort((np.arange(len(values), dtype=np.int64), values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


def bootstrap(values: list[float], repetitions: int, seed: int) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.all(np.isfinite(vector)):
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(seed)
    if len(vector) == 1:
        interval = [float(vector[0]), float(vector[0])]
    else:
        indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
        interval = [
            float(value)
            for value in np.quantile(vector[indices].mean(axis=1), [0.025, 0.975])
        ]
    return {
        "mean_difference": float(np.mean(vector)),
        "median_difference": float(np.median(vector)),
        "object_bootstrap_95_interval": interval,
        "object_wins": int(np.count_nonzero(vector < 0.0)),
        "object_ties": int(np.count_nonzero(vector == 0.0)),
        "object_losses": int(np.count_nonzero(vector > 0.0)),
    }


def query_policy_record(query: dict[str, Any]) -> dict[str, Any]:
    if tuple(query["arms"]) != ARMS:
        raise ValueError("covariance-arm roster changed")
    source_probability: dict[str, np.ndarray] = {}
    target_probability: dict[str, np.ndarray] = {}
    source_labels: np.ndarray | None = None
    target_labels: np.ndarray | None = None
    cost: float | None = None
    for arm in ARMS:
        audit = query["arms"][arm][AUDIT_KEY]
        source_probability[arm] = np.asarray(
            audit["source_probability"], dtype=np.float64
        )
        target_probability[arm] = np.asarray(
            audit["target_probability"], dtype=np.float64
        )
        current_source = np.asarray(audit["source_labels"], dtype=np.bool_)
        current_target = np.asarray(audit["target_labels"], dtype=np.bool_)
        current_cost = float(audit["fallback_cost"])
        if source_labels is None:
            source_labels = current_source
            target_labels = current_target
            cost = current_cost
        elif not np.array_equal(source_labels, current_source):
            raise RuntimeError("source labels differ across arms")
        elif not np.array_equal(target_labels, current_target):
            raise RuntimeError("target labels differ across arms")
        elif cost != current_cost:
            raise RuntimeError("fallback cost differs across arms")
    if source_labels is None or target_labels is None or cost is None:
        raise RuntimeError("instrumented query is incomplete")

    full_source_count = int(
        np.count_nonzero(source_probability["full_low_rank"] <= cost)
    )
    full_target_count = int(
        np.count_nonzero(target_probability["full_low_rank"] <= cost)
    )
    source_quota = proportional_count(
        full_source_count, len(source_labels), len(target_labels)
    )
    policy_loss: dict[str, dict[str, float]] = {policy: {} for policy in POLICIES}
    policy_coverage: dict[str, dict[str, float]] = {
        policy: {} for policy in POLICIES
    }
    ranks: dict[str, list[int]] = {}
    for arm in ARMS:
        masks = {
            "fixed_cost_original": target_probability[arm] <= cost,
            "source_rate_equal_coverage": stable_lowest_mask(
                target_probability[arm], source_quota
            ),
            "full_target_quota_equal_coverage": stable_lowest_mask(
                target_probability[arm], full_target_count
            ),
        }
        for policy, mask in masks.items():
            policy_loss[policy][arm] = decision_loss(mask, target_labels, cost)
            policy_coverage[policy][arm] = float(np.mean(mask))
        ranks[arm] = rank_vector(target_probability[arm]).tolist()
    return {
        "policy_loss": policy_loss,
        "policy_coverage": policy_coverage,
        "ranks": ranks,
    }


def analyze(
    enriched: dict[str, Any],
    reference: dict[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    reference_path = Path(str(reference["_reference_path"]))
    if sha256_file(reference_path) != REFERENCE_RESULT_SHA256:
        raise ValueError("reference result bytes changed")
    projection_match = canonical_digest(scientific_projection(enriched)) == (
        canonical_digest(scientific_projection(reference))
    )
    if not projection_match:
        raise RuntimeError("instrumentation changed the frozen scalar result")

    query_names = list(enriched["summary"]["query_summary"])
    object_rows: list[dict[str, Any]] = []
    rank_matches = {comparator: 0 for comparator in COMPARATORS}
    query_group_count = 0
    for source_row in enriched["objects"]:
        records = {
            query: query_policy_record(source_row["queries"][query])
            for query in query_names
        }
        query_group_count += len(records)
        for record in records.values():
            full_rank = record["ranks"]["full_low_rank"]
            for comparator in COMPARATORS:
                if record["ranks"][comparator] == full_rank:
                    rank_matches[comparator] += 1
        policy_summary = {
            policy: {
                arm: {
                    "decision_loss": float(
                        np.mean(
                            [
                                records[q]["policy_loss"][policy][arm]
                                for q in query_names
                            ]
                        )
                    ),
                    "coverage": float(
                        np.mean(
                            [
                                records[q]["policy_coverage"][policy][arm]
                                for q in query_names
                            ]
                        )
                    ),
                }
                for arm in ARMS
            }
            for policy in POLICIES
        }
        object_rows.append(
            {
                "object_id": source_row["object_id"],
                "target_action_family": source_row["target_action_family"],
                "policy_summary": policy_summary,
            }
        )

    comparisons: dict[str, Any] = {}
    for comparator_index, comparator in enumerate(COMPARATORS):
        comparisons[comparator] = {}
        for policy_index, policy in enumerate(POLICIES):
            loss_difference = [
                row["policy_summary"][policy]["full_low_rank"]["decision_loss"]
                - row["policy_summary"][policy][comparator]["decision_loss"]
                for row in object_rows
            ]
            coverage_difference = [
                row["policy_summary"][policy]["full_low_rank"]["coverage"]
                - row["policy_summary"][policy][comparator]["coverage"]
                for row in object_rows
            ]
            comparisons[comparator][policy] = {
                "decision_loss": bootstrap(
                    loss_difference,
                    repetitions,
                    seed + 10000 * comparator_index + 100 * policy_index,
                ),
                "coverage_mean_difference": float(np.mean(coverage_difference)),
                "coverage_max_absolute_difference": float(
                    np.max(np.abs(coverage_difference))
                ),
            }

    proper_scores: dict[str, Any] = {}
    for comparator_index, comparator in enumerate(COMPARATORS):
        proper_scores[comparator] = {}
        for metric_index, metric in enumerate(
            ("event_brier", "event_log_loss", "query_nll")
        ):
            differences = [
                float(
                    np.mean(
                        [
                            row["queries"][query]["arms"]["full_low_rank"][metric]
                            - row["queries"][query]["arms"][comparator][metric]
                            for query in query_names
                        ]
                    )
                )
                for row in enriched["objects"]
            ]
            proper_scores[comparator][metric] = bootstrap(
                differences,
                repetitions,
                seed + 100000 + 1000 * comparator_index + metric_index,
            )

    leave_one_query_out: dict[str, Any] = {}
    for omitted_index, omitted in enumerate(query_names):
        retained = [query for query in query_names if query != omitted]
        leave_one_query_out[omitted] = {}
        for comparator_index, comparator in enumerate(COMPARATORS):
            differences = [
                float(
                    np.mean(
                        [
                            row["queries"][query]["arms"]["full_low_rank"][
                                "event_brier"
                            ]
                            - row["queries"][query]["arms"][comparator][
                                "event_brier"
                            ]
                            for query in retained
                        ]
                    )
                )
                for row in enriched["objects"]
            ]
            leave_one_query_out[omitted][comparator] = bootstrap(
                differences,
                repetitions,
                seed + 200000 + 1000 * omitted_index + comparator_index,
            )

    equal_coverage_exact = all(
        comparisons[comparator][policy]["coverage_max_absolute_difference"] <= _EPS
        for comparator in COMPARATORS
        for policy in POLICIES[1:]
    )
    proper_score_supported = all(
        proper_scores[comparator][metric]["object_bootstrap_95_interval"][1] < 0.0
        for comparator in COMPARATORS
        for metric in proper_scores[comparator]
    )
    leave_one_query_out_supported = all(
        leave_one_query_out[omitted][comparator]["object_bootstrap_95_interval"][1]
        < 0.0
        for omitted in query_names
        for comparator in COMPARATORS
    )
    rank_difference_observed = any(
        count != query_group_count for count in rank_matches.values()
    )
    equal_coverage_supported = all(
        comparisons[comparator][policy]["decision_loss"][
            "object_bootstrap_95_interval"
        ][1]
        < 0.0
        for comparator in COMPARATORS
        for policy in POLICIES[1:]
    )
    fixed_cost_supported = all(
        comparisons[comparator]["fixed_cost_original"]["decision_loss"][
            "object_bootstrap_95_interval"
        ][1]
        < 0.0
        for comparator in COMPARATORS
    )
    gates = {
        "reference_scalar_result_reproduced": projection_match,
        "complete_92_object_roster": len(object_rows) == 92,
        "equal_coverage_policies_are_exact": equal_coverage_exact,
        "proper_scores_favor_full_dependence": proper_score_supported,
        "brier_result_survives_each_leave_one_query_out_panel": (
            leave_one_query_out_supported
        ),
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "reference": {
            "workflow_run_id": 33528032875,
            "artifact_id": 9811194776,
            "result_file_sha256": REFERENCE_RESULT_SHA256,
        },
        "information_boundary": {
            "retrospective_target_reuse": True,
            "exact_bound_carrier_rerun": True,
            "point_predictor_changed": False,
            "covariance_arms_changed": False,
            "query_bank_changed": False,
            "target_labels_used_to_select_any_policy": False,
            "full_target_quota_uses_predictions_but_not_labels": True,
            "new_measurements_collected": False,
        },
        "object_count": len(object_rows),
        "query_group_count": query_group_count,
        "query_names": query_names,
        "rank_matches": {
            comparator: {
                "matching_query_groups": count,
                "total_query_groups": query_group_count,
                "fraction": float(count / query_group_count),
            }
            for comparator, count in rank_matches.items()
        },
        "proper_scores": proper_scores,
        "decision_policy_comparisons": comparisons,
        "leave_one_query_out_brier": leave_one_query_out,
        "gates": gates,
        "decision": {
            "probability_distribution_value_supported": proper_score_supported,
            "fixed_cost_decision_value_supported": fixed_cost_supported,
            "ranking_difference_observed": rank_difference_observed,
            "equal_coverage_decision_value_supported": equal_coverage_supported,
            "paper_claim_authorized": False,
            "deployment_claim_authorized": False,
        },
        "objects": object_rows,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# Deform360 dependence policy-scale audit v1",
        "",
        f"- Objects: **{result['object_count']}**",
        f"- Object-query groups: **{result['query_group_count']}**",
        "- Point means and coordinate marginals: **unchanged**",
        "- Probability-distribution value supported: "
        "**"
        + str(
            result["decision"]["probability_distribution_value_supported"]
        ).lower()
        + "**",
        "- Ranking difference observed: "
        f"**{str(result['decision']['ranking_difference_observed']).lower()}**",
        "- Equal-coverage decision value supported: "
        "**"
        + str(
            result["decision"]["equal_coverage_decision_value_supported"]
        ).lower()
        + "**",
        "",
        "## Proper-score contrasts",
        "",
        "Negative differences favor full low-rank dependence.",
        "",
        "| Comparator | Metric | Difference | 95% object bootstrap | W/T/L |",
        "|---|---|---:|---:|---:|",
    ]
    for comparator, metrics in result["proper_scores"].items():
        for metric, values in metrics.items():
            low, high = values["object_bootstrap_95_interval"]
            lines.append(
                f"| `{comparator}` | `{metric}` | {values['mean_difference']:.6g} | "
                f"[{low:.6g}, {high:.6g}] | {values['object_wins']}/"
                f"{values['object_ties']}/{values['object_losses']} |"
            )
    lines.extend(
        [
            "",
            "## Decision decomposition",
            "",
            "| Comparator | Policy | Loss difference | 95% object bootstrap | "
            "Coverage difference |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparator, policies in result["decision_policy_comparisons"].items():
        for policy, values in policies.items():
            loss = values["decision_loss"]
            low, high = loss["object_bootstrap_95_interval"]
            lines.append(
                f"| `{comparator}` | `{policy}` | {loss['mean_difference']:.6g} | "
                f"[{low:.6g}, {high:.6g}] | "
                f"{values['coverage_mean_difference']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Rank parity",
            "",
            "| Comparator | Matching object-query groups | Fraction |",
            "|---|---:|---:|",
        ]
    )
    for comparator, values in result["rank_matches"].items():
        lines.append(
            f"| `{comparator}` | {values['matching_query_groups']}/"
            f"{values['total_query_groups']} | {values['fraction']:.3%} |"
        )
    lines.extend(
        [
            "",
            "The two quota policies force identical coverage and use no target",
            "labels for selection. A fixed-cost gain without equal-coverage gain",
            "therefore localizes the decision benefit to posterior probability",
            "levels rather than to a different ranking of target windows.",
            "This is retrospective mechanism evidence, not deployment evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "object_id",
        "target_action_family",
        "comparator",
        "policy",
        "decision_loss_difference",
        "coverage_difference",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["objects"]:
            for comparator in COMPARATORS:
                for policy in POLICIES:
                    full = row["policy_summary"][policy]["full_low_rank"]
                    control = row["policy_summary"][policy][comparator]
                    writer.writerow(
                        {
                            "object_id": row["object_id"],
                            "target_action_family": row["target_action_family"],
                            "comparator": comparator,
                            "policy": policy,
                            "decision_loss_difference": (
                                full["decision_loss"] - control["decision_loss"]
                            ),
                            "coverage_difference": (
                                full["coverage"] - control["coverage"]
                            ),
                        }
                    )


def run_recovery(
    *,
    recovery_runner: Path,
    base_runner: Path,
    protocol: Path,
    parent_protocol: Path,
    parent_result: Path,
    readiness: Path,
    data_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> dict[str, Any]:
    recovery = load_module(recovery_runner, "dependence_policy_audit_recovery")
    original_loader: Callable[[Path, str], Any] = recovery.load_module

    def patched_loader(path: Path, name: str) -> Any:
        module = original_loader(path, name)
        if name == "deform360_dependence_query_v6_original":
            instrument_v6(module)
        return module

    recovery.load_module = patched_loader
    _, result = recovery.run(
        base_runner_path=base_runner,
        protocol_path=protocol,
        parent_protocol_path=parent_protocol,
        parent_result_path=parent_result,
        readiness_path=readiness,
        data_root=data_root,
        parent_control_root=parent_control_root,
        frozen_root=frozen_root,
    )
    return result


def self_test() -> None:
    values = np.asarray([0.4, 0.05, 0.2, 0.1])
    assert stable_lowest_mask(values, 2).tolist() == [False, True, False, True]
    assert rank_vector(values).tolist() == [3, 0, 2, 1]
    assert proportional_count(2, 4, 5) == 3
    labels = np.asarray([False, False, True, True])
    assert math.isclose(
        decision_loss(stable_lowest_mask(values, 2), labels, 0.1), 0.3
    )
    print("deform360 dependence policy-scale audit self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--base-runner", type=Path)
    parser.add_argument("--protocol", type=Path)
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
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=260903)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "recovery_runner",
        "base_runner",
        "protocol",
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
    if args.bootstrap_repetitions < 1000:
        parser.error("bootstrap repetitions must be at least 1000")

    reference = read_json(args.reference_result)
    reference["_reference_path"] = str(args.reference_result)
    enriched = run_recovery(
        recovery_runner=args.recovery_runner.resolve(strict=True),
        base_runner=args.base_runner.resolve(strict=True),
        protocol=args.protocol.resolve(strict=True),
        parent_protocol=args.parent_protocol.resolve(strict=True),
        parent_result=args.parent_result.resolve(strict=True),
        readiness=args.readiness_json.resolve(strict=True),
        data_root=args.data_root.resolve(strict=True),
        parent_control_root=args.parent_control_root.resolve(strict=True),
        frozen_root=args.frozen_root.resolve(strict=True),
    )
    result = analyze(
        enriched,
        reference,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_csv(args.output_csv, result)
    print(json.dumps(result["gates"], indent=2, sort_keys=True))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
