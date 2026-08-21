from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sota" / "matphys_residual_overlay_audit_v1.json"
RESULT_PATH = (
    ROOT
    / "results"
    / "sota"
    / "diagnostics"
    / "matphys_residual_overlay_audit_v1"
    / "result.json"
)


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_matphys_overlay_result_is_bound_to_reviewed_inputs_and_code() -> None:
    config = _load(CONFIG_PATH)
    evidence = cast(dict[str, Any], config["evidence"])
    implementation = cast(dict[str, Any], config["implementation"])
    inputs = cast(dict[str, Any], config["frozen_inputs"])

    assert _sha256(RESULT_PATH) == evidence["sha256"]
    assert RESULT_PATH.stat().st_size == evidence["bytes"]
    assert _sha256(ROOT / implementation["analyzer_module"]["path"]) == (
        implementation["analyzer_module"]["sha256"]
    )
    assert _sha256(ROOT / implementation["runner"]["path"]) == (
        implementation["runner"]["sha256"]
    )
    assert inputs["family_selection"]["sha256"] == (
        "5eedb6cb5a747b856c0af696c5029038a8022f00828f43295f201578a4494890"
    )
    assert inputs["future_summary"]["sha256"] == (
        "6560317dbaebaf99b46328e526febf4a276d6183163284bc56b0d473dfa5b9d9"
    )
    assert inputs["target_or_held_v8_access"] is False


def test_nonzero_matphys_subset_records_positive_transfer_without_point_novelty() -> None:
    result = _load(RESULT_PATH)
    primary = cast(dict[str, Any], result["primary_nonzero_matphys_subset"])
    means = cast(dict[str, dict[str, float]], primary["equal_case_mean"])
    comparisons = cast(dict[str, dict[str, Any]], primary["comparisons_vs_backbone"])

    assert result["selection_changed"] is False
    assert primary["case_count"] == 8
    assert means["backbone"] == pytest.approx(
        {
            "chamfer_distance_m": 0.010689126111670433,
            "track_error_m": 0.01860111069228318,
        }
    )
    assert means["bayesian_anchor"] == pytest.approx(
        {
            "chamfer_distance_m": 0.009597970683599945,
            "track_error_m": 0.016322317729476257,
        }
    )
    assert means["last_residual"] == pytest.approx(
        {
            "chamfer_distance_m": 0.009469460776861125,
            "track_error_m": 0.016215567062346496,
        }
    )
    bayesian = comparisons["bayesian_anchor"]
    assert bayesian["percent_change_vs_backbone"] == pytest.approx(
        {
            "chamfer_distance_m": -10.208088263447102,
            "track_error_m": -12.250843514157994,
        }
    )
    assert bayesian["case_wins"] == {
        "chamfer_distance_m": 7,
        "track_error_m": 7,
    }
    assert means["last_residual"]["chamfer_distance_m"] < means["bayesian_anchor"][
        "chamfer_distance_m"
    ]
    assert means["last_residual"]["track_error_m"] < means["bayesian_anchor"][
        "track_error_m"
    ]


def test_full_stack_is_kept_separate_from_direct_matphys_subset() -> None:
    result = _load(RESULT_PATH)
    secondary = cast(dict[str, Any], result["secondary_full_fallback_stack"])
    family_counts = cast(dict[str, int], result["selected_family_counts"])

    assert secondary["case_count"] == 22
    assert family_counts == {
        "alpha_0000": 14,
        "alpha_0250": 1,
        "alpha_0500": 1,
        "alpha_0750": 1,
        "alpha_1000": 5,
    }
    assert "not an independent MatPhys reproduction" in result["claim_boundary"]
