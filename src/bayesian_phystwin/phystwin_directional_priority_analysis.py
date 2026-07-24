"""Locked analysis for the exploratory CoTracker3 directional-priority arm."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .phystwin_comparison import phystwin_physical_object_cluster
from .phystwin_multiview_priority_analysis import _cluster_interval

METRICS = ("chamfer_distance_m", "track_error_m")
PRIMARY_ARM = "causal_selected_dense_relative_cap"
FIXED_GRAPH_ARM = "combined_graph__lambda_0p0001__cap_060mm"


def _validate_inputs(
    source: Mapping[str, Any],
    hard: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    cases = tuple(protocol["cohort"]["cases"])
    for name, result in (
        ("source", source),
        ("hard", hard),
        ("candidate", candidate),
    ):
        if tuple(result["case_results"]) != cases:
            raise ValueError(f"{name} case order differs from the locked protocol")
    source_config = source["config"]
    hard_config = hard["config"]
    candidate_config = candidate["config"]
    if source_config["observation_source"] != "cotracker3_source_depth":
        raise ValueError("source result is not the source-depth comparator")
    if hard_config["observation_source"] != "cotracker3_multiview_priority":
        raise ValueError("hard result is not the hard multiview-priority arm")
    if (
        candidate_config["observation_source"]
        != "cotracker3_multiview_directional_priority"
    ):
        raise ValueError("candidate is not the directional-priority arm")

    method = protocol["method"]
    checks = {
        "multiview_priority_minimum_availability_fraction": (
            method["priority_threshold"]
        ),
        "cotracker_minimum_quality": method["minimum_view_quality"],
        "cotracker_maximum_cycle_error_px": (
            method["maximum_forward_backward_error_px"]
        ),
        "cotracker_maximum_reprojection_error_px": (
            method["maximum_reprojection_error_px"]
        ),
        "cotracker_minimum_camera_count": method["minimum_camera_count"],
    }
    for key, expected in checks.items():
        if not np.isclose(float(candidate_config[key]), float(expected)):
            raise ValueError(f"candidate {key} differs from the locked protocol")
    if int(candidate_config["multiview_tangent_neighbor_count"]) != 16:
        raise ValueError("candidate tangent neighbor count differs from lock")
    shared_keys = (
        "baseline_kind",
        "manual_prefix_override",
        "prior_strengths",
        "maximum_residuals_m",
        "dense_correction_scales",
        "relative_cap_quantile",
        "relative_cap_multipliers",
        "process_std_m",
        "observation_std_m",
        "initial_std_m",
        "inlier_prior",
        "outlier_variance_multiplier",
    )
    for key in shared_keys:
        if source_config[key] != candidate_config[key]:
            raise ValueError(f"source and candidate differ on shared setting {key}")
        if hard_config[key] != candidate_config[key]:
            raise ValueError(f"hard and candidate differ on shared setting {key}")
    if source_config["manual_prefix_override"]:
        raise ValueError("manual prefix override must be disabled")
    return cases


def _mean_comparison(
    rows: list[dict[str, Any]],
    *,
    comparator: str,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        comparator_mean = float(
            np.mean([row[comparator][metric] for row in rows])
        )
        candidate_mean = float(
            np.mean([row["candidate"][metric] for row in rows])
        )
        result[metric] = {
            f"{comparator}_mean_m": comparator_mean,
            "candidate_mean_m": candidate_mean,
            "difference_m": candidate_mean - comparator_mean,
            "percent_change": 100.0
            * (candidate_mean / comparator_mean - 1.0),
        }
    return result


def analyze_directional_priority_results(
    source: Mapping[str, Any],
    hard: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    bootstrap_draws: int = 100_000,
    bootstrap_seed: int = 20260724,
) -> dict[str, Any]:
    """Analyze the one locked directional endpoint arm."""

    cases = _validate_inputs(source, hard, candidate, protocol)
    rows: list[dict[str, Any]] = []
    source_differences = {metric: {} for metric in METRICS}
    hard_differences = {metric: {} for metric in METRICS}
    for case in cases:
        source_case = source["case_results"][case]
        hard_case = hard["case_results"][case]
        candidate_case = candidate["case_results"][case]
        source_metrics = source_case["candidates"][PRIMARY_ARM]
        hard_metrics = hard_case["candidates"][PRIMARY_ARM]
        candidate_metrics = candidate_case["candidates"][PRIMARY_ARM]
        source_fixed = source_case["candidates"][FIXED_GRAPH_ARM]
        candidate_fixed = candidate_case["candidates"][FIXED_GRAPH_ARM]
        candidate_minus_source = {
            metric: float(candidate_metrics[metric] - source_metrics[metric])
            for metric in METRICS
        }
        candidate_minus_hard = {
            metric: float(candidate_metrics[metric] - hard_metrics[metric])
            for metric in METRICS
        }
        fixed_difference = {
            metric: float(candidate_fixed[metric] - source_fixed[metric])
            for metric in METRICS
        }
        for metric in METRICS:
            source_differences[metric][case] = candidate_minus_source[metric]
            hard_differences[metric][case] = candidate_minus_hard[metric]
        routing = candidate_case["cotracker_depth_lift"]
        rows.append(
            {
                "case": case,
                "physical_object": phystwin_physical_object_cluster(case),
                "source": {
                    metric: float(source_metrics[metric]) for metric in METRICS
                },
                "hard": {
                    metric: float(hard_metrics[metric]) for metric in METRICS
                },
                "candidate": {
                    metric: float(candidate_metrics[metric]) for metric in METRICS
                },
                "candidate_minus_source_m": candidate_minus_source,
                "candidate_minus_hard_m": candidate_minus_hard,
                "both_metric_source_win_or_tie": all(
                    candidate_minus_source[metric] <= 0.0 for metric in METRICS
                ),
                "both_metric_hard_win_or_tie": all(
                    candidate_minus_hard[metric] <= 0.0 for metric in METRICS
                ),
                "fixed_graph_diagnostic": {
                    "source": {
                        metric: float(source_fixed[metric]) for metric in METRICS
                    },
                    "candidate": {
                        metric: float(candidate_fixed[metric])
                        for metric in METRICS
                    },
                    "candidate_minus_source_m": fixed_difference,
                },
                "priority_identity_count": int(
                    routing["priority_identity_count"]
                ),
                "priority_identity_fraction": float(
                    routing["priority_identity_fraction"]
                ),
                "source_update_count": int(routing["source_update_count"]),
                "multiview_tangent_update_count": int(
                    routing["multiview_tangent_update_count"]
                ),
                "multiview_tangent_updates_without_source_count": int(
                    routing["multiview_tangent_updates_without_source_count"]
                ),
            }
        )

    source_means = _mean_comparison(rows, comparator="source")
    hard_means = _mean_comparison(rows, comparator="hard")
    source_intervals = {
        metric: _cluster_interval(
            source_differences[metric],
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        for metric in METRICS
    }
    hard_intervals = {
        metric: _cluster_interval(
            hard_differences[metric],
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        for metric in METRICS
    }
    source_metric_wins = {
        metric: sum(
            row["candidate_minus_source_m"][metric] <= 0.0 for row in rows
        )
        for metric in METRICS
    }
    hard_metric_wins = {
        metric: sum(
            row["candidate_minus_hard_m"][metric] <= 0.0 for row in rows
        )
        for metric in METRICS
    }
    source_both_wins = sum(
        row["both_metric_source_win_or_tie"] for row in rows
    )
    hard_both_wins = sum(row["both_metric_hard_win_or_tie"] for row in rows)

    fixed_graph_metrics: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        source_mean = float(
            np.mean(
                [
                    row["fixed_graph_diagnostic"]["source"][metric]
                    for row in rows
                ]
            )
        )
        candidate_mean = float(
            np.mean(
                [
                    row["fixed_graph_diagnostic"]["candidate"][metric]
                    for row in rows
                ]
            )
        )
        fixed_graph_metrics[metric] = {
            "source_mean_m": source_mean,
            "candidate_mean_m": candidate_mean,
            "difference_m": candidate_mean - source_mean,
            "percent_change": 100.0 * (candidate_mean / source_mean - 1.0),
        }

    gate_spec = protocol["transfer_gate_for_fresh_evaluation"]
    gates = {
        "both_equal_case_future_means_improve_over_source": all(
            source_means[metric]["difference_m"] < 0.0 for metric in METRICS
        ),
        "both_physical_object_cluster_95_percent_upper_bounds_below_zero": all(
            source_intervals[metric]["upper_95_m"] < 0.0 for metric in METRICS
        ),
        "two_metric_win_or_tie_count": source_both_wins,
        "two_metric_win_or_tie_gate": source_both_wins
        >= int(gate_spec["two_metric_win_or_tie_count_at_least"]),
    }
    passed = all(
        value
        for key, value in gates.items()
        if key != "two_metric_win_or_tie_count"
    )
    return {
        "schema_version": 1,
        "status": "post-open exploratory transfer; not independent SOTA evidence",
        "primary_arm": PRIMARY_ARM,
        "case_count": len(rows),
        "per_case": rows,
        "aggregate": {
            "versus_source": {
                "metrics": source_means,
                "metric_win_or_tie_counts": source_metric_wins,
                "both_metric_win_or_tie_count": source_both_wins,
                "object_cluster_bootstrap": source_intervals,
            },
            "versus_hard_priority": {
                "metrics": hard_means,
                "metric_win_or_tie_counts": hard_metric_wins,
                "both_metric_win_or_tie_count": hard_both_wins,
                "object_cluster_bootstrap": hard_intervals,
            },
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
            "fixed_graph_60mm_versus_source": {
                "arm": FIXED_GRAPH_ARM,
                "metrics": fixed_graph_metrics,
            },
        },
        "gates": gates,
        "fresh_evaluation_justified": passed,
        "recommendation": (
            "Freeze and preregister a genuinely fresh-object evaluation."
            if passed
            else "Do not advance this fixed method to fresh-object evaluation."
        ),
    }
