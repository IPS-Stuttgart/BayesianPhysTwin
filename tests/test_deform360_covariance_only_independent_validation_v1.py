from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "protocols/locks/deform360_covariance_only_independent_validation_v1.json"
)
SELECTION = (
    ROOT / "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
SUMMARY = ROOT / "results/science/full22_covariance_only_hybrid_v1/summary.json"
COMPOSITION = ROOT / "src/bayesian_phystwin/covariance_only_hybrid.py"
ANALYSIS = ROOT / "src/bayesian_phystwin/covariance_only_hybrid_analysis.py"
DOCUMENT = ROOT / "docs/deform360_covariance_only_independent_validation_v1.md"
INVENTORY_DOCUMENT = ROOT / "docs/deform360_covariance_source_input_inventory_v1.md"
WORKFLOW = (
    ROOT / ".github/workflows/deform360-covariance-only-independent-validation-v1.yml"
)
INVENTORY_SOURCE = (
    ROOT / "src/bayesian_phystwin/deform360_covariance_source_inventory_v1.py"
)
PRODUCER_SOURCE = (
    ROOT / "src/bayesian_phystwin/deform360_covariance_source_producer_v1.py"
)
PRODUCER_SCRIPT = (
    ROOT / "scripts/science/run_deform360_covariance_source_producer_v1.py"
)
EXPECTED_PROTOCOL_ID = (
    "0f13d7a1f1610588ca9e7119f94814c99940fb31050419de16fa9cae06f683cc"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _content_id(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "protocol_id"}
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def test_protocol_is_content_addressed_and_target_closed() -> None:
    protocol = _load(PROTOCOL)

    assert protocol["schema"] == (
        "bayesian-phystwin.deform360-covariance-only-independent-validation"
    )
    assert protocol["schema_version"] == 1
    assert protocol["status"] == "target-closed-preregistered"
    assert protocol["protocol_id"] == EXPECTED_PROTOCOL_ID == _content_id(protocol)

    boundary = protocol["information_boundary"]
    assert boundary == {
        "causal4d_intervention_claim_authorized": False,
        "claim_authorized_before_target_result": False,
        "confirmation_side_retuning_allowed": False,
        "development_outcomes_already_open": True,
        "promotion_authorized_before_target_result": False,
        "replacement_allowed": False,
        "target_outcomes_used": False,
        "target_payloads_opened": False,
        "target_roster_selected_before_covariance_only_candidate": True,
    }


def test_frozen_candidate_matches_sealed_development_evidence() -> None:
    protocol = _load(PROTOCOL)
    summary = _load(SUMMARY)
    candidate = protocol["frozen_candidate"]

    assert candidate["reference_mean"] == "last_residual"
    assert candidate["reference_predictor_id"] == "last_residual"
    assert candidate["covariance_donor_id"] == "independent_endpoint_v1"
    assert candidate["early_middle_late_covariance_scales"] == [8.0, 16.0, 16.0]
    assert candidate["observation_std_m"] == 0.005
    assert candidate["covariance_eigenvalue_floor_m2"] == 1e-12
    assert candidate["point_prediction_change_allowed"] is False
    assert candidate["donor_substitution_allowed"] is False
    assert candidate["target_scale_retuning_allowed"] is False
    assert candidate["target_horizon_redefinition_allowed"] is False

    assert summary["full_source_fit_for_separate_fresh_study"] == {
        "covariance_donor": "independent_endpoint_v1",
        "early_middle_late_scales": [8.0, 16.0, 16.0],
        "mean_predictor": "last_residual",
    }
    assert summary["mean_identity"] == {
        "chamfer_distance_difference_m": 0.0,
        "exact_identity_case_count": 22,
        "reference_predictor": "last_residual",
        "track_error_difference_m": 0.0,
    }
    assert summary["claim_authorized"] is False
    assert summary["selection_authorized"] is False
    assert summary["promotion_authorized"] is False

    evidence = protocol["development_evidence"]
    assert evidence["implementation_merge_revision"] == (
        "247bc8b85ec425f4272ba867e3a7c878a7e05d56"
    )
    assert evidence["report_id"] == summary["report_id"]
    assert (
        evidence["development_effect_mean_gaussian_nll"]
        == (summary["primary_effect"]["mean_gaussian_nll_difference"])
    )
    assert (
        evidence["development_effect_simultaneous_95_ci"]
        == (summary["primary_effect"]["simultaneous_95_ci"])
    )


def test_implementation_and_selection_bytes_are_frozen() -> None:
    protocol = _load(PROTOCOL)
    identity = protocol["implementation_identity"]
    cohort = protocol["cohort"]

    assert (
        _git_blob_sha1(COMPOSITION)
        == (identity["covariance_composition_git_blob_sha1"])
    )
    assert _git_blob_sha1(ANALYSIS) == identity["analysis_git_blob_sha1"]
    assert _git_blob_sha1(SUMMARY) == identity["development_summary_git_blob_sha1"]
    assert _git_blob_sha1(SELECTION) == cohort["selection_git_blob_sha1"]
    assert identity["exact_distribution_identity_required_before_target_opening"]
    assert identity["runtime_and_source_artifacts_content_addressed"]


def test_cohort_is_exactly_twelve_object_disjoint_sessions() -> None:
    protocol = _load(PROTOCOL)
    selection = _load(SELECTION)["selection"]
    source = selection["calibration"]
    target = selection["confirmation"]
    cohort = protocol["cohort"]

    source_ids = {row["object_id"] for row in source}
    target_ids = {row["object_id"] for row in target}
    assert len(source_ids) == cohort["development_object_session_count"] == 10
    assert len(target_ids) == cohort["target_object_session_count"] == 12
    assert source_ids.isdisjoint(target_ids)
    assert {
        stratum: sum(row["stratum"] == stratum for row in target)
        for stratum in ("sheet", "volumetric")
    } == {"sheet": 6, "volumetric": 6}
    assert cohort["statistical_unit"] == "complete-physical-object-session"
    assert cohort["source_and_target_disjoint_by_object_identity"] is True
    assert cohort["replacement_allowed"] is False
    assert cohort["target_informed_exclusion_allowed"] is False
    assert cohort["failed_or_unsupported_units_retained_in_denominator"] is True
    assert cohort["new_robot_acquisition_required"] is False


def test_prediction_barrier_and_primary_test_are_fixed() -> None:
    protocol = _load(PROTOCOL)
    barrier = protocol["prediction_barrier"]
    hypothesis = protocol["primary_hypothesis"]
    inference = protocol["inference"]

    assert barrier["source_physical_manifest_count_required"] == 10
    assert barrier["source_prediction_seal_count_required"] == 100
    assert barrier["complete_target_prefix_prediction_count_required"] == 12
    assert barrier["all_target_predictions_sealed_before_any_target_future_opening"]
    assert barrier["target_prefix_frame_range_half_open"] == [0, 58]
    assert barrier["target_future_frame_range_half_open"] == [58, 76]
    assert barrier["target_unscored_buffer_frame_range_half_open"] == [76, 81]
    assert barrier["target_payload_opening_count"] == 1
    assert barrier["future_frames_used_for_prediction"] is False
    assert barrier["target_outcomes_used_for_prediction_or_admission"] is False
    assert barrier["source_candidate_or_scale_retuning_allowed"] is False
    assert barrier["human_selection_allowed"] is False

    assert hypothesis["candidate"] == (
        "C1_last_residual_plus_frozen_bayesian_covariance"
    )
    assert hypothesis["comparator"] == "B1_last_residual"
    assert hypothesis["confirmatory_contrast_count"] == 1
    assert hypothesis["direction"] == "lower-is-better"
    assert hypothesis["point_noninferiority"] == "exact-identity-by-construction"
    assert hypothesis["positive_rule"] == (
        "two-sided-95-percent-object-clustered-bootstrap-upper-bound-below-zero"
    )
    assert inference["bootstrap_replicates"] == 100000
    assert inference["bootstrap_seed"] == 20260812
    assert inference["confidence"] == 0.95
    assert inference["resampling_unit"] == "physical-object-session"
    assert inference["within_unit_rows_do_not_increase_sample_size"] is True
    assert inference["missing_unit_imputation"] == "exact-zero-effect-fallback-tie"


def test_secondary_analyses_cannot_select_or_rescue() -> None:
    protocol = _load(PROTOCOL)
    secondary = protocol["predeclared_secondary_analyses"]
    decision = protocol["claim_decision"]

    assert secondary["horizon_family"]["aggregations"] == [
        "early",
        "middle",
        "late",
    ]
    assert secondary["horizon_family"]["selection_role"] == "none"
    assert secondary["observation_noise_sensitivity"]["observation_std_m"] == [
        0.0025,
        0.005,
        0.01,
    ]
    assert secondary["observation_noise_sensitivity"]["selection_role"] == "none"
    calibration = secondary["calibration_and_sharpness"]
    assert calibration["marginal_coverage_levels"] == [0.5, 0.9, 0.95]
    assert calibration["calibration_qualification_interval"] == [0.8, 0.98]
    assert (
        calibration["maximum_width_ratio_for_calibration_sharpness_qualification"]
        == 4.0
    )
    heterogeneity = secondary["heterogeneity"]
    assert heterogeneity["strata"] == ["sheet", "volumetric"]
    assert heterogeneity["minimum_better_or_tied_units_overall"] == 8
    assert heterogeneity["minimum_better_or_tied_units_per_stratum"] == 4
    assert heterogeneity["action_family_restriction_allowed"] is False

    assert decision["target_side_rescue_or_retuning_allowed"] is False
    assert decision["negative_or_inconclusive_result_is_complete"] is True
    assert decision["deployment_authorized"] is False
    assert decision["physical_state_identification_claimed"] is False
    assert decision["state_of_the_art_claimed"] is False


def test_contract_workflow_is_authenticated_source_only_and_data_closed() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    document = DOCUMENT.read_text(encoding="utf-8")
    inventory_document = INVENTORY_DOCUMENT.read_text(encoding="utf-8")

    assert isinstance(workflow, dict)
    assert set(workflow["on"]) == {
        "pull_request",
        "push",
        "issue_comment",
    }
    assert workflow["on"]["issue_comment"] == {"types": ["created"]}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {
        "contracts",
        "source-input-inventory",
        "source-prediction-barrier",
    }

    contracts = workflow["jobs"]["contracts"]
    assert contracts["if"] == "github.event_name != 'issue_comment'"
    assert contracts["runs-on"] == "ubuntu-latest"

    inventory = workflow["jobs"]["source-input-inventory"]
    assert inventory["runs-on"] == [
        "self-hosted",
        "Linux",
        "X64",
        "host-workstation2",
    ]
    assert inventory["permissions"] == {
        "contents": "read",
        "issues": "write",
    }
    condition = str(inventory["if"])
    for required in (
        "github.event_name == 'issue_comment'",
        "github.event.issue.number == 775",
        "github.event.issue.pull_request == null",
        "github.actor == 'FlorianPfaff'",
        "github.event.comment.user.login == 'FlorianPfaff'",
        "github.event.comment.body == '/bpt-inventory-covariance-source-v1'",
    ):
        assert required in condition

    for required in (
        "SOURCE_INVENTORY_COMMAND: /bpt-inventory-covariance-source-v1",
        "SOURCE_PRODUCER_COMMAND: /bpt-produce-covariance-source-v1",
        "AUTHORIZED_RUNNER_NAME: workstation2",
        (
            "SOURCE_ROOT: /mnt/lexar4tb/datasets/"
            "deform360_official_hub_visuotactile_v1/calibration-source"
        ),
        (
            "PROCESSED_ROOT: /mnt/lexar4tb/datasets/"
            "deform360_official_hub_visuotactile_v1/calibration-processed"
        ),
        "FORBIDDEN_CONFIRMATION_ROOT:",
        'test "$RUNNER_NAME" = "$AUTHORIZED_RUNNER_NAME"',
        "run_deform360_covariance_source_producer_v1.py",
        "validate-inventory",
        "validate-panel",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "repos/${GITHUB_REPOSITORY}/issues/775/comments",
    ):
        assert required in workflow_text
    for forbidden in (
        "workflow_dispatch:",
        "contents: write",
        "git push",
    ):
        assert forbidden not in workflow_text

    producer = workflow["jobs"]["source-prediction-barrier"]
    assert producer["runs-on"] == [
        "self-hosted",
        "Linux",
        "X64",
        "host-workstation2",
    ]
    assert producer["permissions"] == {
        "actions": "read",
        "contents": "read",
        "issues": "write",
    }
    producer_condition = str(producer["if"])
    for required in (
        "github.event.issue.number == 775",
        "github.actor == 'FlorianPfaff'",
        "github.event.comment.body == '/bpt-produce-covariance-source-v1'",
    ):
        assert required in producer_condition
    producer_steps = "\n".join(str(step.get("run", "")) for step in producer["steps"])
    for required in (
        'mkdir "$claim"',
        "execute \\",
        '--upstream-run-root "$RESOLVED_UPSTREAM_ROOT"',
        '--forbidden-confirmation-root "$RESOLVED_FORBIDDEN_ROOT"',
        "'.workflow_run.id'",
        "'.workflow_run.head_sha'",
        "source_suffix_scoring_authorized",
        "confirmation_prediction_authorized",
        "prediction_record_count",
    ):
        assert required in producer_steps
    for forbidden in (
        "evaluate_source_gate",
        "source suffix scoring",
        "confirmation-outcome",
    ):
        assert forbidden not in producer_steps.lower()

    inventory_source = INVENTORY_SOURCE.read_text(encoding="utf-8")
    producer_source = PRODUCER_SOURCE.read_text(encoding="utf-8")
    producer_script = PRODUCER_SCRIPT.read_text(encoding="utf-8")
    assert "selection calibration roster changed" in inventory_source
    assert "roster = tuple(" in inventory_source
    assert "SOURCE_ROSTER" in inventory_source
    assert "for object_id, episode, stratum in SOURCE_ROSTER" in producer_source
    assert 'source_suffix_scoring_authorized": False' in producer_source
    assert "def _execute(" in producer_script

    assert "twelve separately selected confirmation object-sessions" in document
    assert "A negative or inconclusive result is complete" in document
    assert "/bpt-inventory-covariance-source-v1" in inventory_document
    assert "/bpt-produce-covariance-source-v1" in inventory_document
    assert "never enters the confirmation root" in inventory_document
