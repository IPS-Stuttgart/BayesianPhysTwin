from __future__ import annotations

from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v2/summary.json"
)


def test_terminal_result_is_content_bound_and_unscored() -> None:
    value = dict(load_strict_json_object(SUMMARY, label="policy certificate result"))
    identity = value.pop("artifact_id")
    assert identity == content_id(value)
    assert value["status"] == "retained_evaluation_native_qa_failure"
    assert value["pre_future"]["accepted_worlds"] == 34
    assert value["pre_future"]["fallback_worlds"] == 254
    assert value["pre_future"]["pre_future_gate_passed"] is True
    assert value["ordinary_calibration_futures"] == 128
    assert value["ordinary_evaluation_prefix_worlds"] == 288
    assert value["evaluation_future_claims"] == 288
    assert value["ordinary_evaluation_futures"] == 286
    assert value["technical_failures"] == 2
    assert value["technical_failure_world_indices"] == [46, 261]
    assert value["terminal"]["partial_scoring_performed"] is False
    assert value["prospective_policy_value_claim"] is False
    assert value["prospective_coverage_claim"] is False
    assert value["matched_comparison_scored"] is False
    assert value["complete_288_world_denominator_scored"] is False
    assert value["retry_authorized"] is False
    assert value["replacement_authorized"] is False
    assert value["protected_data_read"] is False


def test_result_prose_preserves_failure_and_positive_prefix_boundaries() -> None:
    text = " ".join(
        (ROOT / "docs/dlolab_slingshot_policy_certificate_source_v2_result.md")
        .read_text()
        .split()
    )
    for statement in (
        "Retained technical negative at the complete-denominator native-QA gate",
        "34 accepted, 254 exact fallbacks",
        "Ordinary evaluation future seals: 286/288",
        "no partial score",
        "It does not establish that the 34 accepted decisions improved value",
        "must not be scored as a subset",
        "two failed worlds must not be retried or replaced",
        "new disjoint roster",
    ):
        assert statement in text


def test_failure_verifier_names_the_complete_denominator_contract() -> None:
    text = (
        ROOT
        / "scripts/remote/verify_dlolab_slingshot_policy_certificate_source_v2_failure.py"
    ).read_text()
    for expression in (
        "FAILED_WORLD_INDICES = (46, 261)",
        "all_indices = set(range(288))",
        'or (raw_root / "result.json").exists()',
        'task_failures[label].get("retry_authorized") is not False',
        'task_failures[label].get("replacement_authorized") is not False',
    ):
        assert expression in text
