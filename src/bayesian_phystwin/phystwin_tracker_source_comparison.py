"""Locked comparison of source-depth tracker observation arms."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


METRICS = ("chamfer_distance_m", "track_error_m")
PRIMARY_ARM = "causal_selected_dense_relative_cap"


def _validate_inputs(
    comparator: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    if protocol.get("protocol_id") != "phystwin-alltracker-source-depth-smoke-v1":
        raise ValueError("unexpected tracker comparison protocol")
    cases = tuple(str(case) for case in protocol["cases"])
    if tuple(candidate["case_results"]) != cases:
        raise ValueError("candidate case order differs from the locked protocol")
    if any(case not in comparator["case_results"] for case in cases):
        raise ValueError("comparator is missing a locked case")

    comparator_config = comparator["config"]
    candidate_config = candidate["config"]
    if comparator_config["observation_source"] != "cotracker3_source_depth":
        raise ValueError("comparator is not the CoTracker3 source-depth arm")
    if candidate_config["observation_source"] != "alltracker_source_depth":
        raise ValueError("candidate is not the AllTracker source-depth arm")

    method = protocol["method"]
    expected = {
        "baseline_kind": method["baseline_kind"],
        "manual_prefix_override": method["manual_prefix_override"],
        "cotracker_minimum_quality": method["minimum_quality"],
        "cotracker_maximum_cycle_error_px": method[
            "maximum_cycle_error_px"
        ],
        "prior_strengths": method["prior_strengths"],
        "maximum_residuals_m": method["maximum_residuals_m"],
        "dense_correction_scales": method["dense_correction_scales"],
        "nearest_cloud_windows": method["nearest_cloud_windows"],
        "relative_cap_quantile": method["relative_cap_quantile"],
        "relative_cap_multipliers": method["relative_cap_multipliers"],
        "temporal_gamma_candidates": method["temporal_gamma_candidates"],
        "rbf_center_counts": method["rbf_center_counts"],
        "rbf_minimum_availability_fraction": method[
            "rbf_minimum_availability_fraction"
        ],
        "planar_degrees": method["planar_degrees"],
        "planar_ridge_strength": method["planar_ridge_strength"],
        "process_std_m": method["endpoint_filter"]["process_std_m"],
        "observation_std_m": method["endpoint_filter"]["observation_std_m"],
        "initial_std_m": method["endpoint_filter"]["initial_std_m"],
        "inlier_prior": method["endpoint_filter"]["inlier_prior"],
        "outlier_variance_multiplier": method["endpoint_filter"][
            "outlier_variance_multiplier"
        ],
    }
    for key, locked in expected.items():
        for label, config in (
            ("comparator", comparator_config),
            ("candidate", candidate_config),
        ):
            if config[key] != locked:
                raise ValueError(f"{label} {key} differs from the lock")
    if candidate_config["manual_prefix_override"]:
        raise ValueError("manual prefix override must be disabled")
    for case in cases:
        for label, result in (
            ("comparator", comparator),
            ("candidate", candidate),
        ):
            candidates = result["case_results"][case]["candidates"]
            if PRIMARY_ARM not in candidates:
                raise ValueError(f"{label} lacks the locked primary arm")
    return cases


def _means(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        comparator_mean = float(
            np.mean([row["comparator"][metric] for row in rows])
        )
        candidate_mean = float(
            np.mean([row["candidate"][metric] for row in rows])
        )
        result[metric] = {
            "comparator_mean_m": comparator_mean,
            "candidate_mean_m": candidate_mean,
            "difference_m": candidate_mean - comparator_mean,
            "percent_change": 100.0
            * (candidate_mean / comparator_mean - 1.0),
        }
    return result


def analyze_tracker_source_comparison(
    comparator: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the predeclared AllTracker source-depth smoke gate."""

    cases = _validate_inputs(comparator, candidate, protocol)
    rows: list[dict[str, Any]] = []
    for case in cases:
        comparator_metrics = comparator["case_results"][case]["candidates"][
            PRIMARY_ARM
        ]
        candidate_metrics = candidate["case_results"][case]["candidates"][
            PRIMARY_ARM
        ]
        relative_change = {
            metric: float(
                candidate_metrics[metric] / comparator_metrics[metric] - 1.0
            )
            for metric in METRICS
        }
        rows.append(
            {
                "case": case,
                "comparator": {
                    metric: float(comparator_metrics[metric])
                    for metric in METRICS
                },
                "candidate": {
                    metric: float(candidate_metrics[metric])
                    for metric in METRICS
                },
                "candidate_minus_comparator_m": {
                    metric: float(
                        candidate_metrics[metric]
                        - comparator_metrics[metric]
                    )
                    for metric in METRICS
                },
                "relative_change": relative_change,
                "both_metric_win_or_tie": all(
                    relative_change[metric] <= 0.0 for metric in METRICS
                ),
            }
        )

    means = _means(rows)
    metric_wins = {
        metric: sum(
            row["relative_change"][metric] <= 0.0 for row in rows
        )
        for metric in METRICS
    }
    both_wins = sum(row["both_metric_win_or_tie"] for row in rows)
    maximum_regression = max(
        row["relative_change"][metric]
        for row in rows
        for metric in METRICS
    )
    gate_spec = protocol["smoke_gate"]
    gates = {
        "both_equal_case_future_means_improve_over_cotracker3_source": all(
            means[metric]["difference_m"] < 0.0 for metric in METRICS
        ),
        "both_metric_win_count": both_wins,
        "both_metric_win_count_gate": both_wins
        >= int(gate_spec["both_metric_win_count_at_least"]),
        "maximum_case_metric_regression_fraction": maximum_regression,
        "maximum_case_regression_gate": maximum_regression
        <= float(gate_spec["maximum_allowed_case_regression_fraction"]),
    }
    passed = bool(
        gates["both_equal_case_future_means_improve_over_cotracker3_source"]
        and gates["both_metric_win_count_gate"]
        and gates["maximum_case_regression_gate"]
    )
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": (
            "post-open development smoke; not independent SOTA evidence"
        ),
        "primary_arm": PRIMARY_ARM,
        "case_count": len(rows),
        "per_case": rows,
        "aggregate": {
            "metrics": means,
            "metric_win_or_tie_counts": metric_wins,
            "both_metric_win_or_tie_count": both_wins,
        },
        "gates": gates,
        "smoke_gate_passed": passed,
        "recommendation": (
            gate_spec["pass_action"] if passed else gate_spec["fail_action"]
        ),
    }
