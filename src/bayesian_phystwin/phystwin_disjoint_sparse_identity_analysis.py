"""Locked analysis for disjoint sparse-identity source diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


PRIMARY_CANDIDATE = "causal_selected_dense_relative_cap_temporal"
METRICS = ("chamfer_distance_m", "track_error_m")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _analyze_report(
    report: Mapping[str, Any],
    *,
    observed_count: int,
    candidate_name: str,
) -> dict[str, Any]:
    config = report.get("config", {})
    _require(
        config.get("baseline_kind") == "raw_matphys_replay",
        "unexpected physical baseline",
    )
    _require(
        config.get("observation_source") == "final_data",
        "unexpected dense observation source",
    )
    _require(config.get("manual_prefix_override") is True, "manual prefix is disabled")
    _require(
        config.get("manual_observed_track_count") == observed_count,
        "manual observed count does not match the declared arm",
    )
    cases = report.get("case_results", {})
    _require(len(cases) == 22, "the disjoint source report must contain 22 cases")

    baseline_values = {metric: [] for metric in METRICS}
    candidate_values = {metric: [] for metric in METRICS}
    joint_wins = 0
    hidden_counts: list[int] = []
    future_support: list[float] = []
    trackless_frames = 0
    for case in sorted(cases):
        result = cases[case]
        _require(candidate_name in result["candidates"], "primary candidate is missing")
        baseline = result["baseline"]
        candidate = result["candidates"][candidate_name]
        for metric in METRICS:
            baseline_values[metric].append(float(baseline[metric]))
            candidate_values[metric].append(float(candidate[metric]))
        joint_wins += int(
            all(
                float(candidate[metric]) < float(baseline[metric]) for metric in METRICS
            )
        )
        split = result.get("manual_identity_split")
        support = result.get("manual_identity_support")
        _require(isinstance(split, dict), "manual identity split is missing")
        _require(isinstance(support, dict), "manual identity support is missing")
        _require(
            len(split.get("observed_indices", [])) == observed_count,
            "observed identity count changed",
        )
        hidden_counts.append(len(split.get("hidden_indices", [])))
        future_support.append(float(support["hidden_future_frame_fraction"]))
        trackless_frames += int(support["trackless_future_frame_count"])

    baseline_mean = {
        metric: float(np.mean(baseline_values[metric])) for metric in METRICS
    }
    candidate_mean = {
        metric: float(np.mean(candidate_values[metric])) for metric in METRICS
    }
    improvements = {
        metric: 100.0 * (1.0 - candidate_mean[metric] / baseline_mean[metric])
        for metric in METRICS
    }
    return {
        "observed_identity_count": observed_count,
        "hidden_identity_count_minimum": int(min(hidden_counts)),
        "hidden_identity_count_maximum": int(max(hidden_counts)),
        "minimum_hidden_future_frame_support": float(min(future_support)),
        "trackless_future_frame_count": int(trackless_frames),
        "joint_case_wins": int(joint_wins),
        "baseline": baseline_mean,
        "candidate": candidate_mean,
        "relative_improvement_percent": improvements,
        "below_published_8mm_15mm_operating_point": bool(
            candidate_mean["chamfer_distance_m"] < 0.008
            and candidate_mean["track_error_m"] < 0.015
        ),
    }


def analyze_disjoint_sparse_identity_reports(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    primary_observed_count: int = 4,
    candidate_name: str = PRIMARY_CANDIDATE,
    minimum_relative_improvement_percent: float = 5.0,
    minimum_joint_case_wins: int = 16,
) -> dict[str, Any]:
    """Analyze only the predeclared arm across a sparse-sensor count ladder."""

    _require(primary_observed_count in reports, "primary observed-count arm is missing")
    _require(
        minimum_relative_improvement_percent >= 0.0,
        "minimum improvement must be nonnegative",
    )
    _require(
        1 <= minimum_joint_case_wins <= 22,
        "minimum joint wins must lie in [1, 22]",
    )
    arms = {
        str(count): _analyze_report(
            report,
            observed_count=count,
            candidate_name=candidate_name,
        )
        for count, report in sorted(reports.items())
    }
    primary = arms[str(primary_observed_count)]
    gates = {
        "zero_trackless_future_frames": (primary["trackless_future_frame_count"] == 0),
        "minimum_hidden_future_frame_support_is_one": (
            primary["minimum_hidden_future_frame_support"] == 1.0
        ),
        "chamfer_improvement_at_least_threshold": (
            primary["relative_improvement_percent"]["chamfer_distance_m"]
            >= minimum_relative_improvement_percent
        ),
        "hidden_track_improvement_at_least_threshold": (
            primary["relative_improvement_percent"]["track_error_m"]
            >= minimum_relative_improvement_percent
        ),
        "joint_case_wins_at_least_threshold": (
            primary["joint_case_wins"] >= minimum_joint_case_wins
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "status": "post-open disjoint sparse-identity source diagnostic",
        "primary_candidate": candidate_name,
        "primary_observed_identity_count": primary_observed_count,
        "minimum_relative_improvement_percent": (minimum_relative_improvement_percent),
        "minimum_joint_case_wins": minimum_joint_case_wins,
        "arms": arms,
        "gates": gates,
        "gate_passed": passed,
        "decision": (
            "advance-to-registered-noise-and-dropout-source-gate"
            if passed
            else "stop-disjoint-sparse-identity-route"
        ),
        "claim_boundary": (
            "The cohort is outcome-open and manual prefix tracks emulate sparse "
            "material sensors. Assimilated identities are excluded from every future "
            "track score. This is source mechanism evidence, not confirmation or an "
            "open-loop state-of-the-art claim."
        ),
    }
