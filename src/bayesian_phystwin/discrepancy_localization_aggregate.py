"""Equal-case aggregation for released discrepancy-localization diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .phystwin_discrepancy_localization import (
    BASELINE,
    GENERALIZED_FORCE,
    LOCALIZATION_METHODS,
    PREFIX_STATE,
    READOUT,
    STRUCTURAL_CONTROL,
)


def _metric(summary: dict[str, Any], method: str, metric: str) -> float:
    return float(summary["methods"][method]["future"][metric]["candidate_mean_m"])


def _late_metric(summary: dict[str, Any], method: str, metric: str) -> float:
    return float(
        summary["methods"][method]["horizon"]["late"][metric]["candidate_mean_m"]
    )


def _mean_ratio(
    summaries: Sequence[dict[str, Any]],
    numerator: str,
    denominator: str,
    metric: str,
    *,
    late: bool = False,
) -> float:
    getter = _late_metric if late else _metric
    return float(
        np.mean(
            [
                getter(summary, numerator, metric)
                / getter(summary, denominator, metric)
                for summary in summaries
            ]
        )
    )


def aggregate_discrepancy_localization(
    summary_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate diagnostic cases without treating them as confirmation data."""

    if len(summary_paths) < 2:
        raise ValueError("localization aggregation requires at least two cases")
    summaries = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in summary_paths
    ]
    cases = [str(summary["case"]) for summary in summaries]
    if len(set(cases)) != len(cases):
        raise ValueError("localization cases must be unique")
    for summary in summaries:
        if summary.get("experiment") != "phystwin_discrepancy_localization_v1":
            raise ValueError("input is not a discrepancy-localization diagnostic")
        if tuple(summary["method_order"]) != LOCALIZATION_METHODS:
            raise ValueError("localization method ladder differs across cases")
        contract = summary["comparison_contract"]
        if (
            contract.get("graph_rank") != 4
            or contract.get("common_physical_particle_count") != 4
            or not contract.get("official_nonlinear_warp_rerun")
        ):
            raise ValueError("localization comparison contract is not frozen")
        if not summary["zero_force_parity"]["bitwise_identical"]:
            raise ValueError("a case failed zero-force parity")

    methods: dict[str, Any] = {}
    for method in LOCALIZATION_METHODS:
        methods[method] = {
            "future_equal_case_mean_m": {
                metric: float(
                    np.mean([_metric(summary, method, metric) for summary in summaries])
                )
                for metric in ("chamfer_distance_m", "track_error_m")
            },
            "future_equal_case_percent_change_vs_baseline": {
                metric: float(
                    100.0
                    * (
                        np.mean(
                            [
                                _metric(summary, method, metric)
                                / _metric(summary, BASELINE, metric)
                                for summary in summaries
                            ]
                        )
                        - 1.0
                    )
                )
                for metric in ("chamfer_distance_m", "track_error_m")
            },
            "late_equal_case_percent_change_vs_baseline": {
                metric: float(
                    100.0
                    * (
                        np.mean(
                            [
                                _late_metric(summary, method, metric)
                                / _late_metric(summary, BASELINE, metric)
                                for summary in summaries
                            ]
                        )
                        - 1.0
                    )
                )
                for metric in ("chamfer_distance_m", "track_error_m")
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
            "coordinate_coverage_90_mean": float(
                np.mean(
                    [
                        summary["methods"][method]["coverage"][
                            "coordinate_coverage_90"
                        ]
                        for summary in summaries
                    ]
                )
            ),
        }

    force_ratios = {
        "track_vs_readout": _mean_ratio(
            summaries, GENERALIZED_FORCE, READOUT, "track_error_m"
        ),
        "chamfer_vs_readout": _mean_ratio(
            summaries, GENERALIZED_FORCE, READOUT, "chamfer_distance_m"
        ),
        "late_track_vs_readout": _mean_ratio(
            summaries,
            GENERALIZED_FORCE,
            READOUT,
            "track_error_m",
            late=True,
        ),
        "far_graph_vs_readout": float(
            np.mean(
                [
                    summary["methods"][GENERALIZED_FORCE]["far_graph"][
                        "future_observation_error_mean_m"
                    ]
                    / summary["methods"][READOUT]["far_graph"][
                        "future_observation_error_mean_m"
                    ]
                    for summary in summaries
                ]
            )
        ),
    }
    state_ratios = {
        "track_vs_readout": _mean_ratio(
            summaries, PREFIX_STATE, READOUT, "track_error_m"
        ),
        "chamfer_vs_readout": _mean_ratio(
            summaries, PREFIX_STATE, READOUT, "chamfer_distance_m"
        ),
        "late_track_vs_readout": _mean_ratio(
            summaries, PREFIX_STATE, READOUT, "track_error_m", late=True
        ),
    }
    force_gates = {
        "track_better_than_readout": force_ratios["track_vs_readout"] < 1.0,
        "chamfer_not_degraded_vs_readout": force_ratios[
            "chamfer_vs_readout"
        ] <= 1.0,
        "late_track_better_than_readout": force_ratios[
            "late_track_vs_readout"
        ] < 1.0,
        "far_graph_better_than_readout": force_ratios[
            "far_graph_vs_readout"
        ] < 1.0,
        "no_force_limit_hit": all(
            not summary["fit_diagnostics"]["force_limit"]["limit_applied"]
            for summary in summaries
        ),
    }
    force_supported = all(force_gates.values())
    state_matches_readout = bool(
        state_ratios["track_vs_readout"] <= 1.02
        and state_ratios["chamfer_vs_readout"] <= 1.02
        and state_ratios["late_track_vs_readout"] <= 1.02
    )
    best_track_method = min(
        LOCALIZATION_METHODS,
        key=lambda method: methods[method]["future_equal_case_mean_m"][
            "track_error_m"
        ],
    )
    cross_view_available = all(
        summary["observation_model_audit"]["cross_view"].get("status")
        == "available"
        for summary in summaries
    )
    observation_supported = False
    if cross_view_available:
        cross_view_ratio = float(
            np.mean(
                [
                    summary["observation_model_audit"]["cross_view"][
                        "mean_cross_view_error_ratio"
                    ]
                    for summary in summaries
                ]
            )
        )
        observation_supported = best_track_method == READOUT and cross_view_ratio >= 1.0
    else:
        cross_view_ratio = None

    if force_supported:
        conclusion = "generalized_force_location_supported_diagnostically"
    elif state_matches_readout:
        conclusion = "prefix_state_location_supported_diagnostically"
    elif observation_supported:
        conclusion = "observation_readout_bias_supported_diagnostically"
    elif best_track_method == READOUT:
        conclusion = "readout_is_best_but_physical_vs_observation_location_unresolved"
    elif best_track_method == STRUCTURAL_CONTROL:
        conclusion = "matched_structural_control_wins_but_conflicts_with_prior_rejection"
    else:
        conclusion = "discrepancy_location_inconclusive"

    result = {
        "schema_version": 1,
        "aggregate": "phystwin_discrepancy_localization_v1",
        "status": "diagnostic_not_confirmatory",
        "case_count": len(cases),
        "cases": cases,
        "methods": methods,
        "comparisons": {
            "constant_force_vs_readout": force_ratios,
            "prefix_state_vs_readout": state_ratios,
        },
        "acceptance_gates": {
            "constant_force": force_gates,
            "constant_force_supported": force_supported,
            "prefix_state_matches_readout": state_matches_readout,
            "cross_view_available_in_every_case": cross_view_available,
            "cross_view_error_ratio": cross_view_ratio,
            "observation_bias_supported": observation_supported,
        },
        "best_equal_case_track_method": best_track_method,
        "localization_conclusion": conclusion,
        "claim_boundary": {
            "released_cases_repeatedly_examined": True,
            "may_select_confirmatory_physical_mechanism": False,
            "structural_object_persistent_hypothesis_reopened": False,
            "multi_action_protocol_required_for_confirmation": True,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
