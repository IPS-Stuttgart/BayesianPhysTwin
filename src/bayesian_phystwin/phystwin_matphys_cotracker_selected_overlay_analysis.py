"""Locked analysis for the MatPhys/CoTracker selected-overlay source arm."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, SupportsFloat, cast

import numpy as np

METRICS: Final = ("chamfer_distance_m", "track_error_m")
PRIMARY_CANDIDATE: Final = "causal_selected_dense_relative_cap_temporal"
FIXED_COMPARATOR: Final = "dense_raw_manual_override__scale_0p75__cap_040mm"
CASE_COUNT: Final = 22


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_metric(value: object, *, label: str) -> float:
    result = float(cast(SupportsFloat, value))
    _require(np.isfinite(result) and result >= 0.0, f"invalid {label}")
    return result


def _summarize_candidate(
    cases: Mapping[str, Any],
    *,
    candidate_name: str,
) -> dict[str, Any]:
    baseline_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    candidate_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    joint_wins = 0
    exact_ties = 0
    worst_regression_percent = 0.0
    for case_name in sorted(cases):
        case = cases[case_name]
        _require(isinstance(case, Mapping), f"malformed case: {case_name}")
        baseline = case.get("baseline")
        candidates = case.get("candidates")
        _require(isinstance(baseline, Mapping), "baseline metrics are missing")
        _require(isinstance(candidates, Mapping), "candidate metrics are missing")
        _require(
            candidate_name in candidates,
            f"locked candidate is missing: {candidate_name}",
        )
        candidate = candidates[candidate_name]
        _require(isinstance(candidate, Mapping), "locked candidate is malformed")

        comparisons = []
        for metric in METRICS:
            reference = _finite_metric(
                baseline.get(metric),
                label=f"{case_name} baseline {metric}",
            )
            value = _finite_metric(
                candidate.get(metric),
                label=f"{case_name} candidate {metric}",
            )
            _require(reference > 0.0, "baseline metric must be positive")
            baseline_values[metric].append(reference)
            candidate_values[metric].append(value)
            comparisons.append(value < reference)
            regression = 100.0 * (value / reference - 1.0)
            worst_regression_percent = max(worst_regression_percent, regression)
        joint_wins += int(all(comparisons))
        exact_ties += int(
            all(
                candidate_values[metric][-1] == baseline_values[metric][-1]
                for metric in METRICS
            )
        )

    baseline_mean = {
        metric: float(np.mean(baseline_values[metric])) for metric in METRICS
    }
    candidate_mean = {
        metric: float(np.mean(candidate_values[metric])) for metric in METRICS
    }
    improvement = {
        metric: 100.0 * (1.0 - candidate_mean[metric] / baseline_mean[metric])
        for metric in METRICS
    }
    return {
        "source_candidate_name": candidate_name,
        "baseline_equal_case_mean": baseline_mean,
        "candidate_equal_case_mean": candidate_mean,
        "relative_improvement_percent": improvement,
        "joint_case_wins": joint_wins,
        "exact_case_ties": exact_ties,
        "maximum_case_metric_regression_percent": worst_regression_percent,
        "below_published_8mm_15mm_operating_point": bool(
            candidate_mean["chamfer_distance_m"] < 0.008
            and candidate_mean["track_error_m"] < 0.015
        ),
    }


def analyze_matphys_cotracker_selected_overlay_report(
    report: Mapping[str, Any],
    *,
    primary_candidate: str = PRIMARY_CANDIDATE,
    fixed_comparator: str = FIXED_COMPARATOR,
    minimum_relative_improvement_percent: float = 5.0,
    minimum_joint_case_wins: int = 16,
    maximum_case_metric_regression_percent: float = 10.0,
) -> dict[str, Any]:
    """Expose only the two frozen automatic arms from the broad source report."""

    config_value = report.get("config")
    boundary_value = report.get("information_boundary")
    cases_value = report.get("case_results")
    _require(isinstance(config_value, Mapping), "source configuration is missing")
    _require(isinstance(boundary_value, Mapping), "information boundary is missing")
    _require(isinstance(cases_value, Mapping), "source cases are missing")
    config = cast(Mapping[str, Any], config_value)
    boundary = cast(Mapping[str, Any], boundary_value)
    cases = cast(Mapping[str, Any], cases_value)
    _require(len(cases) == CASE_COUNT, "source report must contain exactly 22 cases")

    _require(
        config.get("baseline_kind") == "selected_overlay_sequential",
        "unexpected physical baseline",
    )
    _require(
        config.get("observation_source") == "cotracker3_source_depth",
        "unexpected observation source",
    )
    _require(config.get("manual_prefix_override") is False, "manual prefix is enabled")
    _require(
        config.get("manual_observed_track_count") is None,
        "manual identity split is enabled",
    )
    _require(
        boundary.get("future_inputs_used_for_prediction") is False,
        "future observations entered prediction",
    )
    _require(
        boundary.get("manual_prefix_role")
        == "disabled; manual tracks are evaluation-only",
        "manual-track information boundary changed",
    )

    expected_sequences = {
        "dense_correction_scales": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "maximum_residuals_m": [0.04, 0.06, 0.1, 0.15, 0.2, 0.4],
        "prior_strengths": [0.0001],
        "relative_cap_multipliers": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        "temporal_gamma_candidates": [0.0, 0.25, 0.5, 1.0],
    }
    for name, expected in expected_sequences.items():
        _require(config.get(name) == expected, f"frozen {name} changed")
    _require(config.get("cotracker_minimum_quality") == 0.5, "quality gate changed")
    _require(
        config.get("cotracker_maximum_cycle_error_px") == 5.0,
        "cycle-error gate changed",
    )

    arms = {
        "primary_causal_temporal": _summarize_candidate(
            cases,
            candidate_name=primary_candidate,
        ),
        "fixed_scale_control": _summarize_candidate(
            cases,
            candidate_name=fixed_comparator,
        ),
    }
    primary = arms["primary_causal_temporal"]
    gates = {
        "chamfer_improvement_at_least_threshold": (
            primary["relative_improvement_percent"]["chamfer_distance_m"]
            >= minimum_relative_improvement_percent
        ),
        "track_improvement_at_least_threshold": (
            primary["relative_improvement_percent"]["track_error_m"]
            >= minimum_relative_improvement_percent
        ),
        "joint_case_wins_at_least_threshold": (
            primary["joint_case_wins"] >= minimum_joint_case_wins
        ),
        "maximum_case_metric_regression_within_limit": (
            primary["maximum_case_metric_regression_percent"]
            <= maximum_case_metric_regression_percent
        ),
        "below_published_8mm_15mm_operating_point": primary[
            "below_published_8mm_15mm_operating_point"
        ],
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "status": "post-open automatic MatPhys/CoTracker composition source diagnostic",
        "case_count": CASE_COUNT,
        "primary_candidate": primary_candidate,
        "fixed_comparator": fixed_comparator,
        "minimum_relative_improvement_percent": minimum_relative_improvement_percent,
        "minimum_joint_case_wins": minimum_joint_case_wins,
        "maximum_case_metric_regression_percent": (
            maximum_case_metric_regression_percent
        ),
        "arms": arms,
        "gates": gates,
        "gate_passed": passed,
        "decision": (
            "advance-to-fresh-public-object-source-panel"
            if passed
            else "stop-selected-overlay-cotracker-composition"
        ),
        "claim_boundary": (
            "All 22 PhysTwin outcomes were previously opened. CoTracker uses only "
            "prefix RGB-D evidence and manual tracks are evaluation-only. This is "
            "development evidence, not independent transfer or a state-of-the-art claim."
        ),
    }


__all__ = [
    "FIXED_COMPARATOR",
    "PRIMARY_CANDIDATE",
    "analyze_matphys_cotracker_selected_overlay_report",
]
