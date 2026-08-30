from __future__ import annotations

from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_policy_certificate_source_v1/summary.json"
)


def test_retained_result_is_content_bound_and_stops_before_futures() -> None:
    value = dict(load_strict_json_object(SUMMARY, label="policy certificate result"))
    identity = value.pop("artifact_id")
    assert identity == content_id(value)
    assert value["status"] == "retained_pre_future_gate_failure"
    assert value["pre_future"]["accepted_worlds"] == 12
    assert value["pre_future"]["fallback_worlds"] == 276
    assert value["pre_future"]["pre_future_gate_passed"] is False
    assert value["ordinary_calibration_futures"] == 96
    assert value["ordinary_evaluation_prefix_worlds"] == 288
    assert value["ordinary_evaluation_futures"] == 0
    assert value["prospective_coverage_claim"] is False
    assert value["matched_comparison_scored"] is False
    assert value["retry_authorized"] is False
    assert value["replacement_authorized"] is False
    assert value["protected_data_read"] is False


def test_result_prose_preserves_the_negative_claim_boundary() -> None:
    text = (
        ROOT / "docs/dlolab_slingshot_policy_certificate_source_v1_result.md"
    ).read_text()
    for statement in (
        "Retained negative at the pre-future gate",
        "Evaluation futures generated or read: 0",
        "there is no prospective coverage",
        "unopened evaluation futures must not be used",
        "retroactively rescue this method",
        "Any successor requires",
        "new calibration/evaluation roster",
    ):
        assert statement in text
