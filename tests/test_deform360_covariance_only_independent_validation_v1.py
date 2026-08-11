from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "protocols/locks/deform360_covariance_only_independent_validation_v1.json"
)
SELECTION_PATH = (
    ROOT
    / "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
SUMMARY_PATH = (
    ROOT / "results/science/full22_covariance_only_hybrid_v1/summary.json"
)
COMPOSITION_PATH = ROOT / "src/bayesian_phystwin/covariance_only_hybrid.py"
ANALYSIS_PATH = ROOT / "src/bayesian_phystwin/covariance_only_hybrid_analysis.py"
DOCUMENT_PATH = (
    ROOT / "docs/deform360_covariance_only_independent_validation_v1.md"
)
WORKFLOW_PATH = (
    ROOT
    / ".github/workflows/deform360-covariance-only-independent-validation-v1.yml"
)
EXPECTED_PROTOCOL_ID = "58dfe75b82270cf7bb7e33ebaee8c51b13baabc6ac317e412e25c60cbbd2b79d"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _content_id(value: dict[str, Any]) -> str:
    identity = {key: item for key, item in value.items() if key != "protocol_id"}
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def test_protocol_is_content_addressed_and_target_closed() -> None:
    protocol = _load(PROTOCOL_PATH)

    assert protocol["schema"] == (
        "bayesian-phystwin.deform360-covariance-only-independent-validation"
    )
    assert protocol["schema_version"] == 1
    assert protocol["protocol_name"] == (
        "deform360-covariance-only-independent-validation-v1"
    )
    assert protocol["status"] == "target-closed-preregistered"
    assert protocol["protocol_id"] == EXPECTED_PROTOCOL_ID
    assert protocol["protocol_id"] == _content_id(protocol)

    boundary = protocol["information_boundary"]
    assert boundary["development_outcomes_already_open"] is True
    assert boundary["target_roster_selected_before_covariance_only_candidate"] is True
    assert boundary["target_payloads_opened"] is False
    assert boundary["target_outcomes_used"] is False
    assert boundary["confirmation_side_retuning_allowed"] is False
    assert boundary["replacement_allowed"] is False
    assert boundary["claim_authorized_before_target_result"] is False
    assert boundary["promotion_authorized_before_target_result"] is False


def test_frozen_candidate_matches_the_sealed_development_result() -> None:
    protocol = _load(PROTOCOL_PATH)
    summary = _load(SUMMARY_PATH)
    candidate = protocol["frozen_candidate"]

    assert candidate == {
        "covariance_donor_id": "independent_endpoint_v1",
        "covariance_eigenvalue_floor_m2": 1e-12,
        "donor_substitution_allowed": False,
        "early_middle_late_covariance_scales": [8.0, 16.0, 16.0],
        "horizon_bins": ["early", "middle", "late"],
        "marginal_coverage_z": 1.6448536269514722,
        "mean_identity_requirement": (
            "same-caller-owned-float64-c-contiguous-array-object"
        ),
        "nominal_marginal_coverage": 0.9,
        "observation_std_m": 0.005,
        "point_prediction_change_allowed": False,
        "reference_mean": "last_residual",
        "reference_predictor_id": "last_residual",
        "target_horizon_redefinition_allowed": False,
        "target_observation_noise_selection_allowed": False,
        "target_scale_retuning_allowed": False,
        "unsupported_or_rejected_behavior": (
            "exact-last_residual-reference-and-fallback"
        ),
    }

    frozen = summary["full_source_fit_for_separate_fresh_study"]
    assert frozen == {
        "covariance_donor": candidate["covariance_donor_id"],
        "early_middle_late_scales": (
            candidate["early_middle_late_covariance_scales"]
        ),
        "mean_predictor": candidate["reference_mean"],
    }
    assert summary["mean_identity"]["exact_identity_case_count"] == 22
    assert summary["mean_identity"]["track_error_difference_m"] == 0.0
    assert summary["mean_identity"]["chamfer_distance_difference_m"] == 0.0
    assert summary["claim_authorized"] is False
    assert summary["selection_authorized"] is False
    assert summary["promotion_authorized"] is False


def test_exact_source_files_and_development_evidence_are_bound() -> None:
    identity = _load(PROTOCOL_PATH)["implementation_identity"]

    assert _git_blob_sha1(COMPOSITION_PATH) == (
        identity["covariance_composition_git_blob_sha1"]
    )
    assert _git_blob_sha1(ANALYSIS_PATH) == identity["analysis_git_blob_sha1"]
    assert _git_blob_sha1(SUMMARY_PATH) == (
        identity["development_summary_git_blob_sha1"]
    )
    assert identity["exact_distribution_identity_required_before_target_opening"]
    assert identity["runtime_and_source_artifacts_content_addressed"]


def test_cohorts_are_exact_disjoint_complete_object_sessions() -> None:
    protocol = _load(PROTOCOL_PATH)
    selection = _load(SELECTION_PATH)
    cohort = protocol["cohort"]

    assert _git_blob_sha1(SELECTION_PATH) == cohort["selection_git_blob_sha1"]
    assert cohort["selection_path"] == SELECTION_PATH.relative_to(ROOT).as_posix()

    selected = selection["selection"]
    calibration = selected["calibration"]
    confirmation = selected["confirmation"]
    assert len(calibration) == cohort["development_object_session_count"] == 10
    assert len(confirmation) == cohort["target_object_session_count"] == 12

    calibration_ids = {row["object_id"] for row in calibration}
    confirmation_ids = {row["object_id"] for row in confirmation}
    assert len(calibration_ids) == 10
    assert len(confirmation_ids) == 12
    assert calibration_ids.isdisjoint(confirmation_ids)
    assert {row["stratum"] for row in confirmation} == {"sheet", "volumetric"}
    assert {
        stratum: sum(row["stratum"] == stratum for row in confirmation)
        for stratum in ("sheet", "volumetric")
    } == cohort["target_stratum_counts"] == {"sheet": 6, "volumetric": 6}

    assert cohort["statistical_unit"] == "complete-physical-object-session"
    assert cohort["source_and_target_disjoint_by_object_identity"] is True
    assert cohort["replacement_allowed"] is False
    assert cohort["target_informed_exclusion_allowed"] is False
    assert cohort["failed_or_unsupported_units_retained_in_denominator"] is True
    assert cohort["new_robot_acquisition_required"] is False


def test_prediction_barrier_precedes_the_single_target_opening() -> None:
    barrier = _load(PROTOCOL_PATH)["prediction_barrier"]

    assert barrier["source_readiness_receipt_required"] is True
    assert barrier["source_physical_manifest_count_required"] == 10
    assert barrier["source_prediction_seal_count_required"] == 100
    assert barrier["complete_target_prefix_prediction_count_required"] == 12
    assert barrier["all_target_predictions_sealed_before_any_target_future_opening"]
    assert barrier["target_prefix_frame_range_half_open"] == [0, 58]
    assert barrier["target_future_frame_range_half_open"] == [58, 76]
    assert barrier["target_unscored_buffer_frame_range_half_open"] == [76, 81]
    assert barrier["future_frames_used_for_prediction"] is False
    assert barrier["target_outcomes_used_for_prediction_or_admission"] is False
    assert barrier["target_payload_opening_count"] == 1
    assert barrier["source_candidate_or_scale_retuning_allowed"] is False
    assert barrier["human_selection_allowed"] is False


def test_primary_hypothesis_has_one_object_session_level_contrast() -> None:
    protocol = _load(PROTOCOL_PATH)
    hypothesis = protocol["primary_hypothesis"]
    inference = protocol["inference"]

    assert hypothesis == {
        "across_unit_aggregation": "equal-physical-object-session-mean",
        "candidate": "C1_last_residual_plus_frozen_bayesian_covariance",
        "comparator": "B1_last_residual",
        "confirmatory_contrast_count": 1,
        "difference": "candidate-minus-comparator",
        "direction": "lower-is-better",
        "endpoint": "gaussian-negative-log-predictive-density-common-5mm",
        "point_noninferiority": "exact-identity-by-construction",
        "positive_rule": (
            "two-sided-95-percent-object-clustered-bootstrap-upper-bound-below-zero"
        ),
        "within_unit_aggregation": (
            "equal-mean-over-early-middle-late-object-session-horizon-scores"
        ),
    }
    assert inference["bootstrap_replicates"] == 100000
    assert inference["bootstrap_seed"] == 20260812
    assert inference["confidence"] == 0.95
    assert inference["resampling_unit"] == "physical-object-session"
    assert inference["within_unit_rows_do_not_increase_sample_size"] is True
    assert inference["exact_paired_sign_test_reported"] is True
    assert inference["missing_unit_imputation"] == "exact-zero-effect-fallback-tie"


def test_secondary_analyses_are_fixed_and_nonselective() -> None:
    protocol = _load(PROTOCOL_PATH)
    secondary = protocol["predeclared_secondary_analyses"]
    decision = protocol["claim_decision"]

    assert secondary["horizon_family"] == {
        "aggregations": ["early", "middle", "late"],
        "interval": "max-t-simultaneous-95-percent-object-session-bootstrap",
        "selection_role": "none",
    }
    assert secondary["observation_noise_sensitivity"] == {
        "interval": "max-t-simultaneous-95-percent-object-session-bootstrap",
        "observation_std_m": [0.0025, 0.005, 0.01],
        "primary_value": 0.005,
        "selection_role": "none",
    }
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

    assert decision["primary_covariance_value_claim_requires_primary_positive_rule"]
    assert decision["target_side_rescue_or_retuning_allowed"] is False
    assert decision["negative_or_inconclusive_result_is_complete"] is True
    assert decision["deployment_authorized"] is False
    assert decision["physical_state_identification_claimed"] is False
    assert decision["state_of_the_art_claimed"] is False


def test_contract_workflow_is_data_closed() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    document = DOCUMENT_PATH.read_text(encoding="utf-8")

    assert isinstance(workflow, dict)
    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"contracts"}
    assert workflow["jobs"]["contracts"]["runs-on"] == "ubuntu-latest"
    assert "workflow_dispatch:" not in workflow_text
    assert "self-hosted" not in workflow_text
    assert "contents: write" not in workflow_text
    assert "issues: write" not in workflow_text
    assert "git push" not in workflow_text
    assert "/mnt/" not in workflow_text
    assert "actions/upload-artifact" not in workflow_text

    assert "twelve separately selected confirmation object-sessions" in document
    assert "A negative or inconclusive result is complete" in document
