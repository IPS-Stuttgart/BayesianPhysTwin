import copy

import pytest

from bayesian_phystwin.phystwin_disjoint_sparse_identity_analysis import (
    PRIMARY_CANDIDATE,
    analyze_disjoint_sparse_identity_reports,
)


def _report(count: int, *, candidate_scale: float = 0.8) -> dict:
    cases = {}
    for index in range(22):
        cases[f"case-{index:02d}"] = {
            "baseline": {
                "chamfer_distance_m": 0.010,
                "track_error_m": 0.020,
            },
            "candidates": {
                PRIMARY_CANDIDATE: {
                    "chamfer_distance_m": 0.010 * candidate_scale,
                    "track_error_m": 0.020 * candidate_scale,
                }
            },
            "manual_identity_split": {
                "observed_indices": list(range(count)),
                "hidden_indices": list(range(count, 9)),
            },
            "manual_identity_support": {
                "hidden_future_frame_fraction": 1.0,
                "trackless_future_frame_count": 0,
            },
        }
    return {
        "config": {
            "baseline_kind": "raw_matphys_replay",
            "observation_source": "final_data",
            "manual_prefix_override": True,
            "manual_observed_track_count": count,
        },
        "case_results": cases,
    }


def test_analysis_passes_only_the_predeclared_disjoint_arm() -> None:
    result = analyze_disjoint_sparse_identity_reports(
        {1: _report(1), 2: _report(2), 4: _report(4)},
        primary_observed_count=4,
    )

    assert result["gate_passed"] is True
    assert result["decision"] == "advance-to-registered-noise-and-dropout-source-gate"
    assert result["arms"]["4"]["joint_case_wins"] == 22
    assert result["arms"]["4"]["hidden_identity_count_minimum"] == 5


def test_analysis_fails_when_hidden_future_has_trackless_frames() -> None:
    report = _report(4)
    report["case_results"]["case-00"]["manual_identity_support"][
        "trackless_future_frame_count"
    ] = 1
    report["case_results"]["case-00"]["manual_identity_support"][
        "hidden_future_frame_fraction"
    ] = 0.9

    result = analyze_disjoint_sparse_identity_reports({4: report})

    assert result["gate_passed"] is False
    assert result["gates"]["zero_trackless_future_frames"] is False


def test_analysis_rejects_a_mismatched_observed_count() -> None:
    report = _report(4)
    report["config"]["manual_observed_track_count"] = 2

    with pytest.raises(ValueError, match="does not match"):
        analyze_disjoint_sparse_identity_reports({4: report})


def test_nonprimary_candidates_cannot_change_the_result() -> None:
    report = _report(4)
    altered = copy.deepcopy(report)
    for case in altered["case_results"].values():
        case["candidates"]["future_oracle"] = {
            "chamfer_distance_m": 0.0,
            "track_error_m": 0.0,
        }

    first = analyze_disjoint_sparse_identity_reports({4: report})
    second = analyze_disjoint_sparse_identity_reports({4: altered})

    assert first == second
