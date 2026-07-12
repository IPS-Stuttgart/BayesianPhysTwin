"""Equal-case aggregation for released structural-calibration diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _load_summary(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _candidate_mean(
    summary: Mapping[str, Any], method: str, metric: str, *, horizon: str | None = None
) -> float:
    record = summary["methods"][method]
    if horizon is None:
        return float(record["future"][metric]["candidate_mean_m"])
    return float(record["horizon"][horizon][metric]["candidate_mean_m"])


def _baseline_mean(
    summary: Mapping[str, Any], metric: str, *, horizon: str | None = None
) -> float:
    record = summary["methods"]["released_phystwin"]
    if horizon is None:
        return float(record["future"][metric]["baseline_mean_m"])
    return float(record["horizon"][horizon][metric]["baseline_mean_m"])


def _equal_case_percent_change(
    summaries: Sequence[Mapping[str, Any]],
    method: str,
    metric: str,
    *,
    horizon: str | None = None,
) -> float:
    ratios = [
        _candidate_mean(summary, method, metric, horizon=horizon)
        / _baseline_mean(summary, metric, horizon=horizon)
        for summary in summaries
    ]
    return float(100.0 * (np.mean(ratios) - 1.0))


def _paired_method_ratio(
    summaries: Sequence[Mapping[str, Any]],
    numerator_methods: Sequence[str],
    denominator_method: str,
    metric: str,
    *,
    horizon: str | None = None,
) -> float:
    ratios = [
        _candidate_mean(summary, numerator, metric, horizon=horizon)
        / _candidate_mean(summary, denominator_method, metric, horizon=horizon)
        for summary, numerator in zip(summaries, numerator_methods, strict=True)
    ]
    return float(np.mean(ratios))


def aggregate_structural_diagnostics(
    values: Sequence[str | Path | Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate released cases without promoting them to confirmatory evidence."""

    summaries = tuple(_load_summary(value) for value in values)
    if not summaries:
        raise ValueError("at least one structural diagnostic is required")
    if any(
        summary.get("schema_version") != 1
        or summary.get("experiment")
        != "hierarchical_graph_structural_calibration"
        for summary in summaries
    ):
        raise ValueError("unsupported structural diagnostic summary")
    cases = [str(summary["case"]) for summary in summaries]
    if len(set(cases)) != len(cases):
        raise ValueError("structural diagnostic cases must be unique")
    required_methods = {
        "released_phystwin",
        "graph_persistence_readout",
        "baseline",
        "frame_only",
        "initial_state_only",
        "rest_geometry_only",
        "rest_state",
        "hierarchical",
    }
    if any(required_methods - set(summary["methods"]) for summary in summaries):
        raise ValueError("structural diagnostic ladder is incomplete")
    methods = sorted(required_methods)
    metrics = ("chamfer_distance_m", "track_error_m")
    aggregate = {}
    for method in methods:
        aggregate[method] = {
            "future_equal_case_percent_change": {
                metric: _equal_case_percent_change(summaries, method, metric)
                for metric in metrics
            },
            "late_equal_case_percent_change": {
                metric: _equal_case_percent_change(
                    summaries, method, metric, horizon="late"
                )
                for metric in metrics
            },
            "far_graph_observation_error_mean_m": float(
                np.mean(
                    [
                        summary["methods"][method]["far_graph"][
                            "future_observation_error_mean_m"
                        ]
                        for summary in summaries
                    ]
                )
            ),
        }
    selected_methods = [
        str(summary["selected_physical_variant"]) for summary in summaries
    ]
    selected_track_ratio = _paired_method_ratio(
        summaries,
        selected_methods,
        "graph_persistence_readout",
        "track_error_m",
    )
    selected_late_track_ratio = _paired_method_ratio(
        summaries,
        selected_methods,
        "graph_persistence_readout",
        "track_error_m",
        horizon="late",
    )
    selected_far_ratio = float(
        np.mean(
            [
                summary["methods"][selected]["far_graph"][
                    "future_observation_error_mean_m"
                ]
                / summary["methods"]["graph_persistence_readout"]["far_graph"][
                    "future_observation_error_mean_m"
                ]
                for summary, selected in zip(
                    summaries, selected_methods, strict=True
                )
            ]
        )
    )
    acceptance = {
        "cross_action_track_vs_graph_persistence": {
            "maximum_mean_error_ratio": 0.95,
            "observed_mean_error_ratio": selected_track_ratio,
            "passed": selected_track_ratio <= 0.95,
        },
        "late_track_vs_graph_persistence": {
            "maximum_mean_error_ratio": 0.95,
            "observed_mean_error_ratio": selected_late_track_ratio,
            "passed": selected_late_track_ratio <= 0.95,
        },
        "far_graph_vs_graph_persistence": {
            "maximum_mean_error_ratio": 0.95,
            "observed_mean_error_ratio": selected_far_ratio,
            "passed": selected_far_ratio <= 0.95,
        },
        "coverage_gate": {
            "passed": False,
            "reason": "released deterministic diagnostics do not identify predictive coverage",
        },
    }
    return {
        "schema_version": 1,
        "aggregate": "hierarchical_structural_released_diagnostics",
        "status": "diagnostic_not_confirmatory",
        "case_count": len(summaries),
        "cases": cases,
        "selected_physical_variant_counts": dict(Counter(selected_methods)),
        "selected_physical_ranks": {
            summary["case"]: int(summary["selected_physical_rank"])
            for summary in summaries
        },
        "methods": aggregate,
        "acceptance_gates": acceptance,
        "structural_candidate_accepted": all(
            value["passed"] for value in acceptance.values()
        ),
        "information_boundary": {
            "case_rank_and_mechanism_selection": "O-minus only",
            "future_metrics_used_for_selection": False,
            "released_cases_may_lock_confirmatory_configuration": False,
            "individual_counterfactual_claimed": False,
        },
    }


def write_structural_diagnostic_aggregate(
    output_path: str | Path,
    values: Sequence[str | Path | Mapping[str, Any]],
) -> dict[str, Any]:
    result = aggregate_structural_diagnostics(values)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
