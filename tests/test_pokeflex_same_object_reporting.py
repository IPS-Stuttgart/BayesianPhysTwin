"""Tests for the bounded PokeFlex same-object paper reporting path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin import pokeflex_same_object_reporting as reporting

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULT_PATH = (
    _REPOSITORY_ROOT
    / "results"
    / "sota"
    / "pokeflex_independent_depth_regret_guard_prospective_v1"
    / "prospective_evaluation.json"
)


def _synthetic_result() -> dict[str, object]:
    decisions = [
        {
            "frame_id": "SeenA_T7:f00001",
            "take_id": "SeenA_T7",
            "object": "SeenA",
            "take": "T7",
            "target_frame": 1,
            "baseline_error_mm": 2.0,
            "selected_error_mm": 1.0,
            "hidden_regret_mm": -1.0,
            "candidate_upper_regret_mm": -0.3,
            "selector_adjusted_upper_regret_mm": -0.2,
            "selected_arm": "candidate_a",
            "accepted": True,
        },
        {
            "frame_id": "SeenA_T8:f00001",
            "take_id": "SeenA_T8",
            "object": "SeenA",
            "take": "T8",
            "target_frame": 1,
            "baseline_error_mm": 2.0,
            "selected_error_mm": 2.0,
            "hidden_regret_mm": 0.0,
            "candidate_upper_regret_mm": 0.2,
            "selector_adjusted_upper_regret_mm": 0.3,
            "selected_arm": "released_checkpoint",
            "accepted": False,
        },
        {
            "frame_id": "SeenB_T7:f00001",
            "take_id": "SeenB_T7",
            "object": "SeenB",
            "take": "T7",
            "target_frame": 1,
            "baseline_error_mm": 2.0,
            "selected_error_mm": 2.0,
            "hidden_regret_mm": 0.0,
            "candidate_upper_regret_mm": 0.1,
            "selector_adjusted_upper_regret_mm": 0.2,
            "selected_arm": "released_checkpoint",
            "accepted": False,
        },
    ]
    takes = [
        {
            "take_id": "SeenA_T7",
            "object": "SeenA",
            "target_frame_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 1.0,
            "relative_improvement": 0.5,
        },
        {
            "take_id": "SeenA_T8",
            "object": "SeenA",
            "target_frame_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 2.0,
            "relative_improvement": 0.0,
        },
        {
            "take_id": "SeenB_T7",
            "object": "SeenB",
            "target_frame_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 2.0,
            "relative_improvement": 0.0,
        },
    ]
    objects = [
        {
            "object": "SeenA",
            "take_count": 2,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 1.5,
            "relative_improvement": 0.25,
        },
        {
            "object": "SeenB",
            "take_count": 1,
            "baseline_mean_CD_UL1_mm": 2.0,
            "selected_mean_CD_UL1_mm": 2.0,
            "relative_improvement": 0.0,
        },
    ]
    return {
        "artifact_kind": reporting.EXPECTED_ARTIFACT_KIND,
        "claim_status": reporting.EXPECTED_CLAIM_STATUS,
        "gate_passed": True,
        "aggregation": "equal frames, takes, then objects",
        "take_ids": ["SeenA_T7", "SeenA_T8", "SeenB_T7"],
        "take_count": 3,
        "object_count": 2,
        "object_wins": 2,
        "object_losses": 0,
        "baseline_object_mean_CD_UL1_mm": 2.0,
        "selected_object_mean_CD_UL1_mm": 1.75,
        "object_balanced_relative_improvement": 0.125,
        "accepted_frame_count": 1,
        "accepted_frame_wins": 1,
        "accepted_frame_losses": 0,
        "exact_fallback_frame_count": 2,
        "takes": takes,
        "objects": objects,
        "decisions": decisions,
        "deployment_artifact": {
            "candidate_certificate": {},
            "selector_correction_bound": {},
        },
    }


def test_committed_same_object_result_retains_bounded_claim() -> None:
    result = reporting.load_json_object(_RESULT_PATH)
    summary = reporting.validate_bounded_result(result)

    assert summary["take_count"] == 3
    assert summary["object_count"] == 2
    assert summary["frame_count"] == 241
    assert summary["accepted_frame_count"] == 87
    assert summary["exact_fallback_frame_count"] == 154
    assert summary["accepted_frame_wins"] == 77
    assert summary["accepted_frame_losses"] == 10
    assert summary["object_wins"] == 2
    assert summary["object_losses"] == 0
    assert summary["object_balanced_relative_improvement"] == pytest.approx(
        0.030601394040032672
    )
    assert "independent-object generalization" in summary["excluded_claims"]


def test_post_outcome_diagnostic_reconstructs_frozen_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_result()
    frames = [
        {
            "frame_id": value["frame_id"],
            "take_id": value["take_id"],
            "object": value["object"],
            "take": value["take"],
            "target_frame": value["target_frame"],
            "baseline_error_mm": value["baseline_error_mm"],
        }
        for value in result["decisions"]
    ]
    rows = [
        {
            **frames[0],
            "candidate": "candidate_a",
            "features": np.asarray([-0.3]),
            "candidate_error_mm": 1.0,
        },
        {
            **frames[1],
            "candidate": "candidate_b",
            "features": np.asarray([0.2]),
            "candidate_error_mm": 3.0,
        },
        {
            **frames[2],
            "candidate": "candidate_c",
            "features": np.asarray([0.1]),
            "candidate_error_mm": 1.9,
        },
    ]
    certificate = SimpleNamespace(
        minimum_improvement=0.0,
        nominal_coverage=0.9,
        finite_sample_coverage=0.92,
        upper_regret=lambda features: float(features[0]),
    )
    bound = SimpleNamespace(
        upper_regret_m=0.1,
        nominal_coverage=0.9,
        finite_sample_coverage=0.91,
    )
    monkeypatch.setattr(
        reporting,
        "extract_pokeflex_regret_guard_rows",
        lambda payloads: (rows, frames),
    )
    monkeypatch.setattr(reporting, "_certificate_from_dict", lambda value: certificate)
    monkeypatch.setattr(reporting, "_bound_from_dict", lambda value: bound)

    diagnostic = reporting.build_candidate_diagnostics([{}], result)
    summary = diagnostic["candidate_diagnostic"]
    assert summary["candidate_supported_frame_count"] == 3
    assert summary["accepted_frame_count"] == 1
    assert summary["safe_accepted_frame_count"] == 1
    assert summary["harmful_accepted_frame_count"] == 0
    assert summary["harmful_candidate_fallback_count"] == 1
    assert summary["beneficial_candidate_fallback_count"] == 1
    assert diagnostic["analysis_role"].startswith("post-outcome visualization")


def test_calibration_figure_renders_from_diagnostic(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = (
        _REPOSITORY_ROOT
        / "scripts"
        / "paper"
        / "make_pokeflex_same_object_figure.py"
    )
    spec = importlib.util.spec_from_file_location("pokeflex_same_object_figure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    diagnostic = {
        "bounded_result": reporting.validate_bounded_result(_synthetic_result()),
        "candidate_diagnostic": {
            "adjusted_upper_bound_coverage": 2.0 / 3.0,
            "accepted_harmful_fraction": 0.0,
        },
        "rows": [
            {
                "candidate_supported": True,
                "accepted": True,
                "candidate_regret_mm": -1.0,
                "selector_adjusted_upper_regret_mm": -0.2,
            },
            {
                "candidate_supported": True,
                "accepted": False,
                "candidate_regret_mm": 1.0,
                "selector_adjusted_upper_regret_mm": 0.3,
            },
            {
                "candidate_supported": True,
                "accepted": False,
                "candidate_regret_mm": -0.1,
                "selector_adjusted_upper_regret_mm": 0.2,
            },
        ],
    }
    png_path = tmp_path / "figure.png"
    pdf_path = tmp_path / "figure.pdf"
    module.render_figure(diagnostic, png_path, pdf_path)
    assert png_path.stat().st_size > 10_000
    assert pdf_path.stat().st_size > 5_000
