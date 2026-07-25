"""Locked analysis for the exploratory CoTracker3 multiview-priority arm."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .phystwin_comparison import phystwin_physical_object_cluster

METRICS = ("chamfer_distance_m", "track_error_m")
PRIMARY_ARM = "causal_selected_dense_relative_cap"
FIXED_GRAPH_ARM = "combined_graph__lambda_0p0001__cap_060mm"


def _cluster_interval(
    differences: Mapping[str, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    if draws < 1:
        raise ValueError("draws must be positive")
    grouped: dict[str, list[float]] = {}
    for case, difference in differences.items():
        grouped.setdefault(phystwin_physical_object_cluster(case), []).append(
            float(difference)
        )
    cluster_means = np.asarray(
        [np.mean(values) for values in grouped.values()],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    samples = np.mean(
        rng.choice(
            cluster_means,
            size=(draws, len(cluster_means)),
            replace=True,
        ),
        axis=1,
    )
    return {
        "cluster_count": len(cluster_means),
        "mean_difference_m": float(np.mean(cluster_means)),
        "lower_95_m": float(np.quantile(samples, 0.025)),
        "upper_95_m": float(np.quantile(samples, 0.975)),
        "probability_improved": float(np.mean(samples < 0.0)),
    }


def _validate_inputs(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    cases = tuple(protocol["cohort"]["cases"])
    if tuple(source["case_results"]) != cases:
        raise ValueError("source case order differs from the locked protocol")
    if tuple(candidate["case_results"]) != cases:
        raise ValueError("candidate case order differs from the locked protocol")
    source_config = source["config"]
    candidate_config = candidate["config"]
    if source_config["observation_source"] != "cotracker3_source_depth":
        raise ValueError("source result is not the source-depth comparator")
    if candidate_config["observation_source"] != "cotracker3_multiview_priority":
        raise ValueError("candidate result is not the multiview-priority arm")
    method = protocol["method"]
    checks = {
        "multiview_priority_minimum_availability_fraction": (
            method["minimum_identity_prefix_availability_fraction"]
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
    if source_config["manual_prefix_override"]:
        raise ValueError("manual prefix override must be disabled")
    return cases


def analyze_multiview_priority_results(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    bootstrap_draws: int = 100_000,
    bootstrap_seed: int = 20260724,
) -> dict[str, Any]:
    """Compare the one locked primary arm without selecting on future scores."""

    cases = _validate_inputs(source, candidate, protocol)
    rows: list[dict[str, Any]] = []
    differences = {metric: {} for metric in METRICS}
    for case in cases:
        source_case = source["case_results"][case]
        candidate_case = candidate["case_results"][case]
        source_metrics = source_case["candidates"][PRIMARY_ARM]
        candidate_metrics = candidate_case["candidates"][PRIMARY_ARM]
        source_fixed = source_case["candidates"][FIXED_GRAPH_ARM]
        candidate_fixed = candidate_case["candidates"][FIXED_GRAPH_ARM]
        metric_difference = {
            metric: float(candidate_metrics[metric] - source_metrics[metric])
            for metric in METRICS
        }
        fixed_difference = {
            metric: float(candidate_fixed[metric] - source_fixed[metric])
            for metric in METRICS
        }
        for metric in METRICS:
            differences[metric][case] = metric_difference[metric]
        priority = candidate_case["cotracker_depth_lift"]
        rows.append(
            {
                "case": case,
                "physical_object": phystwin_physical_object_cluster(case),
                "source": {
                    metric: float(source_metrics[metric]) for metric in METRICS
                },
                "candidate": {
                    metric: float(candidate_metrics[metric]) for metric in METRICS
                },
                "candidate_minus_source_m": metric_difference,
                "candidate_percent_change": {
                    metric: 100.0
                    * (
                        float(candidate_metrics[metric])
                        / float(source_metrics[metric])
                        - 1.0
                    )
                    for metric in METRICS
                },
                "both_metric_win_or_tie": all(
                    metric_difference[metric] <= 0.0 for metric in METRICS
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
                    priority["priority_identity_count"]
                ),
                "priority_identity_fraction": float(
                    priority["priority_identity_fraction"]
                ),
            }
        )

    means: dict[str, dict[str, float]] = {}
    intervals: dict[str, dict[str, float | int]] = {}
    metric_win_counts: dict[str, int] = {}
    for metric in METRICS:
        source_mean = float(np.mean([row["source"][metric] for row in rows]))
        candidate_mean = float(
            np.mean([row["candidate"][metric] for row in rows])
        )
        means[metric] = {
            "source_mean_m": source_mean,
            "candidate_mean_m": candidate_mean,
            "difference_m": candidate_mean - source_mean,
            "percent_change": 100.0 * (candidate_mean / source_mean - 1.0),
        }
        intervals[metric] = _cluster_interval(
            differences[metric],
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        metric_win_counts[metric] = sum(
            row["candidate_minus_source_m"][metric] <= 0.0 for row in rows
        )

    both_win_count = sum(row["both_metric_win_or_tie"] for row in rows)
    fixed_graph_metrics: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        fixed_source_mean = float(
            np.mean(
                [
                    row["fixed_graph_diagnostic"]["source"][metric]
                    for row in rows
                ]
            )
        )
        fixed_candidate_mean = float(
            np.mean(
                [
                    row["fixed_graph_diagnostic"]["candidate"][metric]
                    for row in rows
                ]
            )
        )
        fixed_graph_metrics[metric] = {
            "source_mean_m": fixed_source_mean,
            "candidate_mean_m": fixed_candidate_mean,
            "difference_m": fixed_candidate_mean - fixed_source_mean,
            "percent_change": 100.0
            * (fixed_candidate_mean / fixed_source_mean - 1.0),
        }
    fixed_graph_both_win_count = sum(
        all(
            row["fixed_graph_diagnostic"]["candidate_minus_source_m"][metric]
            <= 0.0
            for metric in METRICS
        )
        for row in rows
    )
    gate_spec = protocol["transfer_gate_for_fresh_evaluation"]
    gates = {
        "both_equal_case_future_means_improve": all(
            means[metric]["difference_m"] < 0.0 for metric in METRICS
        ),
        "both_physical_object_cluster_95_percent_upper_bounds_below_zero": all(
            intervals[metric]["upper_95_m"] < 0.0 for metric in METRICS
        ),
        "two_metric_win_or_tie_count": both_win_count,
        "two_metric_win_or_tie_gate": both_win_count
        >= int(gate_spec["two_metric_win_or_tie_count_at_least"]),
    }
    passed = all(
        value
        for key, value in gates.items()
        if key != "two_metric_win_or_tie_count"
    )
    return {
        "schema_version": 1,
        "status": (
            "post-open exploratory transfer; not independent SOTA evidence"
        ),
        "primary_arm": PRIMARY_ARM,
        "case_count": len(rows),
        "per_case": rows,
        "aggregate": {
            "metrics": means,
            "metric_win_or_tie_counts": metric_win_counts,
            "both_metric_win_or_tie_count": both_win_count,
            "object_cluster_bootstrap": intervals,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
            "fixed_graph_60mm_diagnostic": {
                "arm": FIXED_GRAPH_ARM,
                "metrics": fixed_graph_metrics,
                "both_metric_win_or_tie_count": fixed_graph_both_win_count,
            },
        },
        "gates": gates,
        "fresh_evaluation_justified": passed,
        "recommendation": (
            "Freeze the method and preregister a genuinely fresh-object evaluation."
            if passed
            else "Do not advance this fixed method to a fresh-object evaluation."
        ),
    }
