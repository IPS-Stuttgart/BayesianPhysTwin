"""Locked analysis for the bias-aware AllTracker ray development smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PRIMARY_ARM = "causal_selected_alltracker_ray_bias_aware_graph"
COMPARATOR_ARM = "causal_selected_dense_relative_cap"
METRICS = ("chamfer_distance_m", "track_error_m")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[dict[str, float]], metric: str) -> float:
    return sum(row[metric] for row in rows) / len(rows)


def analyze_bias_aware_ray_smoke(
    candidate: dict[str, Any],
    comparator: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Read only fields permitted by the frozen smoke protocol."""

    cases = list(protocol["cases"])
    if candidate["config"]["observation_source"] != (
        "alltracker_multiview_ray_bias_aware"
    ):
        raise ValueError("candidate observation source does not match protocol")
    if protocol["future_read"]["primary_arm"] != PRIMARY_ARM:
        raise ValueError("protocol primary arm is unexpected")
    if comparator["config"]["observation_source"] != ("cotracker3_source_depth"):
        raise ValueError("comparator observation source is unexpected")
    if set(candidate["case_results"]) != set(cases):
        raise ValueError("candidate cases do not match protocol")
    if set(comparator["case_results"]) != set(cases):
        raise ValueError("comparator cases do not match protocol")

    baseline_rows: list[dict[str, float]] = []
    candidate_rows: list[dict[str, float]] = []
    comparator_rows: list[dict[str, float]] = []
    per_case = []
    for case in cases:
        candidate_case = candidate["case_results"][case]
        comparator_case = comparator["case_results"][case]
        selector = candidate_case["causal_selection"]["selectors"][PRIMARY_ARM]
        baseline_metrics = {
            metric: float(candidate_case["baseline"][metric]) for metric in METRICS
        }
        candidate_metrics = {
            metric: float(selector["future_metrics"][metric]) for metric in METRICS
        }
        comparator_metrics = {
            metric: float(
                comparator_case["causal_selection"]["selectors"][COMPARATOR_ARM][
                    "future_metrics"
                ][metric]
            )
            for metric in METRICS
        }
        baseline_rows.append(baseline_metrics)
        candidate_rows.append(candidate_metrics)
        comparator_rows.append(comparator_metrics)
        relative = {
            metric: (candidate_metrics[metric] - baseline_metrics[metric])
            / baseline_metrics[metric]
            for metric in METRICS
        }
        both_win_or_tie = all(
            candidate_metrics[metric] <= baseline_metrics[metric] for metric in METRICS
        )
        per_case.append(
            {
                "case": case,
                "admitted": bool(selector["accepted"]),
                "fallback_applied": bool(selector["fallback_applied"]),
                "fallback_is_exact": selector["fallback_is_exact"],
                "admission": selector["admission"],
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "frozen_cotracker3_source": comparator_metrics,
                "candidate_relative_to_baseline": relative,
                "both_metric_win_or_tie": both_win_or_tie,
            }
        )

    aggregate: dict[str, Any] = {
        "baseline_mean": {metric: _mean(baseline_rows, metric) for metric in METRICS},
        "candidate_mean": {metric: _mean(candidate_rows, metric) for metric in METRICS},
        "frozen_cotracker3_source_mean": {
            metric: _mean(comparator_rows, metric) for metric in METRICS
        },
        "both_metric_win_or_tie_count": sum(
            int(row["both_metric_win_or_tie"]) for row in per_case
        ),
    }
    aggregate["candidate_percent_change_from_baseline"] = {
        metric: 100.0
        * (aggregate["candidate_mean"][metric] - aggregate["baseline_mean"][metric])
        / aggregate["baseline_mean"][metric]
        for metric in METRICS
    }
    maximum_regression = max(
        max(row["candidate_relative_to_baseline"].values()) for row in per_case
    )
    gate_config = protocol["smoke_gate"]
    gates = {
        "aggregate_future_chamfer_improves_over_raw_baseline": (
            aggregate["candidate_mean"]["chamfer_distance_m"]
            < aggregate["baseline_mean"]["chamfer_distance_m"]
        ),
        "aggregate_future_track_improves_over_raw_baseline": (
            aggregate["candidate_mean"]["track_error_m"]
            < aggregate["baseline_mean"]["track_error_m"]
        ),
        "both_metric_win_or_tie_count": (
            aggregate["both_metric_win_or_tie_count"]
            >= int(gate_config["both_metric_win_or_tie_count_at_least"])
        ),
        "maximum_case_metric_regression": (
            maximum_regression
            <= float(gate_config["maximum_allowed_case_metric_regression_fraction"])
        ),
        "aggregate_both_metrics_no_worse_than_frozen_cotracker3_source": all(
            aggregate["candidate_mean"][metric]
            <= aggregate["frozen_cotracker3_source_mean"][metric]
            for metric in METRICS
        ),
    }
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": ("post-open development smoke; not independent SOTA evidence"),
        "primary_arm": PRIMARY_ARM,
        "case_count": len(cases),
        "per_case": per_case,
        "aggregate": {
            **aggregate,
            "maximum_case_metric_regression_fraction": maximum_regression,
        },
        "gates": gates,
        "smoke_gate_passed": all(gates.values()),
        "recommendation": (
            gate_config["pass_action"]
            if all(gates.values())
            else gate_config["fail_action"]
        ),
    }


def analyze_bias_aware_ray_smoke_files(
    *,
    candidate_path: str | Path,
    comparator_path: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Load locked inputs and attach their hashes to the compact result."""

    paths = {
        "candidate": Path(candidate_path),
        "comparator": Path(comparator_path),
        "protocol": Path(protocol_path),
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    result = analyze_bias_aware_ray_smoke(
        payloads["candidate"],
        payloads["comparator"],
        payloads["protocol"],
    )
    result["inputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    return result
