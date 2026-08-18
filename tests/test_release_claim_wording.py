from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUPPORT = ROOT / "SUPPORT.md"
CLAIM = ROOT / "docs" / "phystwin_release_claim_v1.md"
SNAPSHOT = ROOT / "evidence" / "public_claim_snapshot_v1.json"
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


def _claims() -> dict[str, dict[str, Any]]:
    payload = _json(SNAPSHOT)
    claims = payload["claims"]
    assert isinstance(claims, list)
    result: dict[str, dict[str, Any]] = {}
    for claim in claims:
        assert isinstance(claim, dict)
        claim_id = claim["id"]
        assert isinstance(claim_id, str)
        assert claim_id not in result
        result[claim_id] = claim
    return result


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

    claims = _claims()
    comparison = claims["unique_deterministic_winner"]
    assert comparison["status"] == "not_confirmed"
    assert comparison["metrics"]["last_residual_track_error_m"] == 0.019156
    raw = claims["raw_covariance_calibration"]
    assert raw["status"] == "refuted"
    assert raw["metrics"]["operational_mean_nees_3d"] == 1355.05


def test_release_surfaces_bind_covariance_only_effect_and_width_cost() -> None:
    for path in (SUPPORT, CLAIM):
        text = _text(path)
        assert "-9.136" in text
        assert "[-13.961, -4.312]" in text
        assert "0.01645 m" in text
        assert "0.05094 m" in text
        assert "3.10×" in text

    readme = _text(README)
    assert "Retrospective covariance-only proper-score value established?" in readme
    assert "Yes, with width cost" in readme
    assert "3.10x" in readme
    assert "evidence/public_claim_snapshot_v1.json" in readme

    metrics = _claims()["retrospective_covariance_only_value"]["metrics"]
    assert metrics["exact_mean_preserved_case_count"] == 22
    assert metrics["case_count"] == 22
    assert metrics["gaussian_nll_change"] == -9.136
    assert metrics["simultaneous_95_percent_ci"] == [-13.961, -4.312]
    assert metrics["object_session_wins"] == 17
    assert metrics["baseline_marginal_90_percent_coverage"] == 0.706
    assert metrics["candidate_marginal_90_percent_coverage"] == 0.91
    assert metrics["baseline_mean_full_interval_width_m"] == 0.01645
    assert metrics["candidate_mean_full_interval_width_m"] == 0.05094
    assert metrics["mean_full_interval_width_ratio"] == 3.1


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
        assert obsolete not in _text(path).casefold()

    for path in (SUPPORT, CLAIM):
        text = _text(path)
        assert "313/324" in text
        assert "11" in text
        assert "100 sealed" in text
        assert "deform360_covariance_only_independent_validation_v1" in text
        assert "twelve" in text
        assert "fresh-object-session v6/v6.1" in text
        assert "sixteen" in text

    readme = _text(README)
    assert "Fresh independent covariance-only confirmation established?" in readme
    assert "100 sealed source prediction records" in readme
    assert "twelve disjoint confirmation object-sessions remain closed" in readme
    assert "Fresh-object-session v6/v6.1 transfer established?" in readme
    assert "Terminal, no claim" in readme

    claims = _claims()
    covariance_claim = claims["covariance_only_independent_validation"]
    covariance_metrics = covariance_claim["metrics"]
    assert covariance_claim["status"] == "not_established"
    assert (
        covariance_metrics["protocol_id"]
        == "deform360_covariance_only_independent_validation_v1"
    )
    assert covariance_metrics["development_object_session_count"] == 10
    assert covariance_metrics["source_prediction_seal_count_required"] == 100
    assert covariance_metrics["confirmation_object_session_count"] == 12
    assert covariance_metrics["confirmation_payload_opened"] is False

    fresh_claim = claims["fresh_object_session_v61"]
    fresh_metrics = fresh_claim["metrics"]
    assert fresh_claim["status"] == "terminal_without_claim"
    assert fresh_metrics["source_prediction_success_count"] == 313
    assert fresh_metrics["source_prediction_expected_count"] == 324
    assert fresh_metrics["retained_support_negative_count"] == 11
    assert fresh_metrics["fresh_target_object_session_count"] == 16

    claim = _text(CLAIM)
    assert "Separate covariance-only independent-validation route" in claim
    assert "Distinct Deform360 fresh-object-session v6/v6.1 route" in claim
    assert "A source-negative result is complete evidence" in claim
    assert "No donor, scale, endpoint" in claim


def test_release_surfaces_report_terminal_v61_source_status() -> None:
    for path in (SUPPORT, CLAIM):
        normalized = _text(path).replace("source-gate", "source gate")
        assert "endpoint-processing technical failure" in normalized
        assert "source gate" in normalized
        assert "fresh-target" in normalized
        assert "replacement" in normalized.casefold()
        assert "continuation" in normalized.casefold()
        assert "v6.1 retirement record" in normalized

    readme = _text(README).replace("source-gate", "source gate")
    assert "endpoint-processing technical failure" in readme
    assert "source gate" in readme
    assert "fresh-target" in readme
    assert "replacement" in readme.casefold()
    assert "continuation" in readme.casefold()

    fresh = _claims()["fresh_object_session_v61"]
    metrics = fresh["metrics"]
    assert fresh["status"] == "terminal_without_claim"
    assert metrics["failure_stage"] == "endpoint-processing technical failure"
    assert metrics["source_gate_evaluated"] is False
    assert metrics["replacement_retry_source_continuation_forbidden"] is True
    assert metrics["fresh_target_payload_opened"] is False
    assert metrics["confirmation_payload_opened"] is False
    assert metrics["held_v8_payload_opened"] is False
    assert metrics["authorized_fresh_object_claim"] is False

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
