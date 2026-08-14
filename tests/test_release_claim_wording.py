from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUPPORT = ROOT / "SUPPORT.md"
CLAIM = ROOT / "docs" / "phystwin_release_claim_v1.md"
COVARIANCE_PROTOCOL = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_covariance_only_independent_validation_v1.json"
)
FRESH_PROTOCOL = (
    ROOT / "protocols" / "locks" / "deform360_official_hub_fresh_object_session_v6.json"
)


def _text(path: Path) -> str:
    """Normalize Markdown wrapping without changing scientific wording."""

    return " ".join(path.read_text(encoding="utf-8").split())


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_release_surfaces_retain_matched_comparator_and_raw_covariance_boundary() -> (
    None
):
    for path in (README, SUPPORT, CLAIM):
        text = _text(path)
        assert "last_residual" in text or "last-residual" in text
        assert "raw posterior covariance" in text.casefold()
        assert "independent" in text

    claim = _text(CLAIM)
    assert "0.019156 m" in claim
    assert "0.019205 m" in claim
    assert "unique best deterministic predictor" in claim
    assert "calibrated raw posterior covariance" in claim


def test_release_surfaces_bind_covariance_only_effect_and_width_cost() -> None:
    for path in (README, SUPPORT, CLAIM):
        text = _text(path)
        assert "-9.136" in text
        assert "[-13.961, -4.312]" in text
        assert "0.01645 m" in text
        assert "0.05094 m" in text
        assert "3.10×" in text

    readme = _text(README)
    assert "exact `last_residual` point-prediction object" in readme
    assert "17/22" in readme
    assert "0.706" in readme
    assert "0.910" in readme


def test_release_surfaces_keep_deform360_protocols_distinct() -> None:
    covariance = _json(COVARIANCE_PROTOCOL)
    fresh = _json(FRESH_PROTOCOL)

    assert covariance["cohort"]["development_object_session_count"] == 10
    assert covariance["cohort"]["target_object_session_count"] == 12
    assert (
        covariance["prediction_barrier"]["source_prediction_seal_count_required"] == 100
    )
    assert fresh["guard_calibration"]["outer_folds"] == 10
    assert fresh["fresh_selection"]["object_count"] == 16
    assert fresh["evaluation"]["target_unit_count"] == 16

    obsolete = (
        "separate registered deform360 v6 route now freezes exactly ten opened "
        "source object-sessions and twelve disjoint confirmation object-sessions"
    )
    for path in (README, SUPPORT, CLAIM):
        text = _text(path)
        assert "313/324" in text
        assert "11" in text
        assert "100 sealed" in text
        assert "deform360_covariance_only_independent_validation_v1" in text
        assert "twelve" in text
        assert "fresh-object-session v6/v6.1" in text
        assert "sixteen" in text
        assert obsolete not in text.casefold()

    claim = _text(CLAIM)
    assert "Separate covariance-only independent-validation route" in claim
    assert "Distinct Deform360 fresh-object-session v6/v6.1 route" in claim
    assert "A source-negative result is complete evidence" in claim
    assert "No donor, scale, endpoint" in claim


def test_release_surfaces_report_terminal_v61_source_status() -> None:
    for path in (README, SUPPORT, CLAIM):
        normalized = _text(path).replace("source-gate", "source gate")
        assert "endpoint-processing technical failure" in normalized
        assert "source gate" in normalized
        assert "fresh-target" in normalized
        assert "replacement" in normalized.casefold()
        assert "continuation" in normalized.casefold()
        assert "v6.1 retirement record" in normalized

    claim = _text(CLAIM).replace("source-gate", "source gate")
    assert (
        "terminal receipt forbids replacement, retry, and source continuation" in claim
    )
    assert "No fresh-target, confirmation, or held-v8 payload was opened" in claim


def test_release_claim_forbids_overstatement() -> None:
    claim = _text(CLAIM)
    forbidden_boundaries = (
        "unique deterministic winner",
        "calibrated raw posterior covariance",
        "independent-object transfer",
        "deployment safety",
        "overall state of the art",
    )
    for boundary in forbidden_boundaries:
        assert boundary in claim
