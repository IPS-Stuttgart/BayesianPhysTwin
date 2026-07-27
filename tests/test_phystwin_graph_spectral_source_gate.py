from __future__ import annotations

import json

import pytest

from bayesian_phystwin.phystwin_graph_spectral_source_gate import (
    GRAPH_SPECTRAL_SOURCE_CONTRACT,
    GraphSpectralSourceConfig,
    _candidate_passes,
    _candidate_summary,
    _load_protocol,
)


def _protocol() -> dict[str, object]:
    return {
        "contract": GRAPH_SPECTRAL_SOURCE_CONTRACT,
        "source_cases": ["a", "b", "c"],
        "target_cases": ["target"],
        "source_folds": [
            {"held_out_cases": ["a", "b"]},
            {"held_out_cases": ["c"]},
        ],
        "model": {},
    }


def test_protocol_requires_disjoint_complete_fold_coverage(tmp_path) -> None:
    payload = _protocol()
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, config = _load_protocol(path)
    assert loaded["source_cases"] == ["a", "b", "c"]
    assert config == GraphSpectralSourceConfig()

    payload["target_cases"] = ["a"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        _load_protocol(path)


def test_candidate_summary_uses_whole_fold_joint_wins() -> None:
    results = [
        {
            "case": "a",
            "ratios_relative_to_persistence": {
                "chamfer_distance_m": 0.9,
                "track_error_m": 0.9,
            },
        },
        {
            "case": "b",
            "ratios_relative_to_persistence": {
                "chamfer_distance_m": 0.95,
                "track_error_m": 1.01,
            },
        },
        {
            "case": "c",
            "ratios_relative_to_persistence": {
                "chamfer_distance_m": 0.8,
                "track_error_m": 0.8,
            },
        },
    ]
    folds = [
        {"held_out_cases": ["a", "b"]},
        {"held_out_cases": ["c"]},
    ]
    summary = _candidate_summary(results, folds)
    assert summary["both_win_fold_count"] == 1
    assert summary["maximum_case_metric_ratio"] == pytest.approx(1.01)


def test_gate_requires_improvement_transfer_and_case_safety() -> None:
    config = GraphSpectralSourceConfig(
        minimum_balanced_improvement=0.03,
        minimum_both_win_folds=2,
        maximum_case_metric_ratio=1.05,
    )
    candidate = {
        "balanced_improvement": 0.04,
        "both_win_fold_count": 2,
        "maximum_case_metric_ratio": 1.04,
        "aggregate_ratios_relative_to_persistence": {
            "chamfer_distance_m": 0.95,
            "track_error_m": 0.97,
        },
    }
    assert _candidate_passes(candidate, config)
    candidate["maximum_case_metric_ratio"] = 1.06
    assert not _candidate_passes(candidate, config)
