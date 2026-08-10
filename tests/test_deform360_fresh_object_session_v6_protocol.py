from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_fresh_object_session_v6.json"
)
V5_POLICY_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_joint_sparse_prospective_v5.json"
)
V5_SELECTION_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
DOCUMENT_PATH = ROOT / "docs/deform360_fresh_object_session_v6.md"
WORKFLOW_PATH = (
    ROOT / ".github/workflows/deform360-fresh-object-session-v6-contracts.yml"
)
MANIFEST_PATH = ROOT / "MANIFEST.in"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _content_id(value: dict[str, Any], *, excluded: str) -> str:
    identity = {key: item for key, item in value.items() if key != excluded}
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_v6_is_content_addressed_and_does_not_rewrite_v5() -> None:
    policy = _load_json(POLICY_PATH)
    v5 = _load_json(V5_POLICY_PATH)
    selection = _load_json(V5_SELECTION_PATH)

    assert policy["schema"] == (
        "bayesian-phystwin.deform360-fresh-object-session-prospective-policy"
    )
    assert policy["schema_version"] == 6
    assert policy["policy_id"] == _content_id(policy, excluded="policy_id")
    assert policy["protocol_id"] == (
        "deform360-official-hub-fresh-object-session-v6"
    )
    assert policy["semantics"] == (
        "prospective-fresh-object-session-source-selected-challenger-v6"
    )

    predecessor = policy["predecessor_boundary"]
    assert predecessor["v5_policy_path"] == V5_POLICY_PATH.relative_to(ROOT).as_posix()
    assert predecessor["v5_policy_id"] == v5["policy_id"]
    assert predecessor["v5_selection_path"] == (
        V5_SELECTION_PATH.relative_to(ROOT).as_posix()
    )
    assert predecessor["v5_selection_artifact_sha256"] == (
        selection["selection_artifact_sha256"]
    )
    assert predecessor[
        "v5_must_have_immutable_terminal_record_before_v6_target_selection"
    ]
    assert predecessor[
        "v6_source_selection_completed_before_v5_terminal_outcome_read"
    ]
    assert predecessor[
        "v6_target_execution_if_source_passes_independent_of_v5_result_sign"
    ]
    assert predecessor["v5_method_threshold_or_outcome_used_for_v6_design"] is False
    assert predecessor["v5_confirmation_units_excluded_from_v6"] is True
    assert predecessor["v5_result_reinterpreted_or_rewritten"] is False
    assert predecessor["v6_positive_result_relabels_v5_negative"] is False


def test_cross_version_reporting_prevents_a_v5_rescue_claim() -> None:
    reporting = _load_json(POLICY_PATH)["cross_version_reporting"]

    assert reporting == {
        "cross_version_candidate_threshold_or_covariance_retuning_allowed": False,
        "cross_version_replication_claim_requires_v5_and_v6_positive": True,
        "v5_and_v6_terminal_records_must_be_reported_together": True,
        (
            "v6_execution_after_source_authorization_is_not_optional_by_"
            "v5_result_sign"
        ): True,
        "v6_positive_result_does_not_override_or_relabel_v5": True,
        "v6_standalone_claim_scope": (
            "the-sixteen-fresh-v6-object-session-units-only"
        ),
    }


def test_fresh_selection_is_disjoint_metadata_only_and_no_replacement() -> None:
    selection = _load_json(POLICY_PATH)["fresh_selection"]

    assert selection["selection_unit"] == "unique-physical-object-session"
    assert selection["object_count"] == 16
    assert selection["stratum_counts"] == {"sheet": 8, "volumetric": 8}
    assert selection["one_session_per_object"] is True
    assert selection["object_identity_must_be_previously_untouched"] is True
    assert selection["episode_identity_must_be_previously_untouched"] is True
    assert selection["metadata_only_before_selection_lock"] is True
    assert selection["replacement_allowed"] is False
    assert "all-v5-confirmation-object-identities" in selection["required_exclusions"]
    assert "all-v5-development-object-identities" in selection["required_exclusions"]
    assert selection["selection_lock_required_before_payload_access"] is True
    assert selection["insufficient_eligible_units_action"] == (
        "publish-terminal-support-negative-without-payload-access"
    )
    assert set(selection["forbidden_prelock_inputs"]) >= {
        "camera-pixels",
        "tactile-arrays",
        "prediction-residuals",
        "future-frames",
        "target-outcomes",
    }


def test_source_tournament_advances_at_most_one_frozen_challenger() -> None:
    tournament = _load_json(POLICY_PATH)["source_tournament"]
    challengers = tournament["challenger_candidates"]

    assert tournament["source_unit_count"] == 10
    assert tournament["source_stratum_counts"] == {"sheet": 5, "volumetric": 5}
    assert tournament["reference_candidate_id"] == "B1_last_causal_residual"
    assert tournament["baseline_candidate_id"] == "B0_physical_fallback"
    assert tournament["maximum_advanced_challengers"] == 1
    assert [row["candidate_id"] for row in challengers] == [
        "D1_dynamic_endpoint_model_average_v2",
        "VT1_joint_sparse_visuotactile_guarded_v5",
    ]
    assert [row["complexity_rank"] for row in challengers] == [1, 2]
    assert tournament["candidate_predictions_sealed_before_source_suffix_scoring"]
    assert tournament["outer_folds"] == 10
    assert tournament["inner_training_units_per_fold"] == 9
    assert tournament["final_winner_must_match_at_least_outer_folds"] == 8
    assert tournament["minimum_nonregressing_held_out_units"] == 8
    assert tournament["minimum_nonregressing_held_out_units_per_stratum"] == 4
    assert tournament["reference_retained_action"] == (
        "publish-terminal-source-negative-without-v6-target-access"
    )
    assert tournament["human_candidate_selection_allowed"] is False
    assert tournament["target_outcomes_used"] is False


def test_guard_covariance_and_interval_calibration_are_source_only() -> None:
    policy = _load_json(POLICY_PATH)
    guard = policy["guard_calibration"]
    calibration = policy["covariance_calibration"]

    assert guard["strategy"] == "nested-source-only-tie-preserving-risk-coverage"
    assert guard["candidate_specific_thresholds_fitted_inside_each_outer_fold"]
    assert guard[
        "candidate_prediction_and_risk_score_sealed_before_suffix_scoring"
    ]
    assert guard["risk_score_semantics"] == (
        "lower-is-safer-inclusive-threshold-v1"
    )
    assert guard["risk_threshold_tie_policy"] == (
        "accept-complete-tied-score-blocks"
    )
    assert guard["minimum_accepted_source_units"] == 8
    assert guard["minimum_accepted_source_units_per_stratum"] == 4
    assert guard["confirmation_adaptation"] == "forbidden"
    assert guard["exact_point_and_interval_fallback_for_every_rejection"]

    assert calibration["required_raw_methods"] == [
        "working-irls",
        "observed-information",
        "group-sandwich",
    ]
    assert calibration["method_complexity_order"] == [
        "working-irls",
        "observed-information",
        "group-sandwich",
    ]
    assert calibration["exact-prior-fallback-role"] == "rejected-inference-only"
    assert calibration["all_available_methods_or_explicit_unavailability_required"]
    assert calibration["method_selection_nested_inside_candidate_outer_folds"]
    assert calibration["selected_method_must_match_at_least_outer_folds"] == 8
    assert calibration["variance_scale_fitted_source_only"] is True
    assert calibration["interval_calibration"] == "group-clustered-split-conformal"
    assert calibration["target_nominal_coverage"] == 0.90
    assert calibration["minimum_source_object_balanced_coverage"] == 0.80
    assert calibration["maximum_source_object_balanced_coverage"] == 0.98
    assert calibration["maximum_mean_full_interval_width_ratio_vs_reference"] == 1.25
    assert calibration["proper_score_nonregression_vs_reference_required"] is True
    assert calibration["reference_interval_required"] is True
    assert calibration["confirmation_adaptation"] == "forbidden"
    assert calibration["raw_covariance_claimed_calibrated"] is False


def test_target_decision_uses_units_and_three_familywise_contrasts() -> None:
    policy = _load_json(POLICY_PATH)
    evaluation = policy["evaluation"]
    decision = policy["positive_decision"]

    assert evaluation["statistical_unit"] == "physical-object-session"
    assert evaluation["target_unit_count"] == 16
    assert evaluation["target_stratum_counts"] == {"sheet": 8, "volumetric": 8}
    assert evaluation["all_target_units_in_denominator"] is True
    assert evaluation["within_unit_rows_do_not_increase_sample_size"] is True
    assert evaluation["observation_causal_frame_range_half_open"] == [0, 58]
    assert evaluation["future_evaluation_frame_range_half_open"] == [58, 76]
    assert evaluation["unscored_terminal_buffer_frame_range_half_open"] == [76, 81]
    assert evaluation["endpoint_view_reservation"]["views_per_unit"] == 2
    assert evaluation["endpoint_view_reservation"]["metadata_only_selection"] is True
    assert (
        evaluation["endpoint_view_reservation"][
            "reserved_views_contribute_likelihood"
        ]
        is False
    )

    roles = {row["method_id"]: row["role"] for row in evaluation["methods"]}
    assert roles == {
        "B0_physical_fallback": "baseline-and-exact-fallback",
        "B1_last_causal_residual": "registered-reference",
        "C1_source_selected_challenger_guarded": "primary-candidate",
        "C2_source_selected_challenger_unguarded": "safety-diagnostic",
    }
    assert decision["primary_method_id"] == "C1_source_selected_challenger_guarded"
    assert decision["minimum_relative_improvement_vs_physical_fallback"] == 0.10
    assert decision["minimum_relative_improvement_vs_last_causal_residual"] == 0.05
    assert (
        decision[
            "minimum_relative_gaussian_nll_improvement_vs_last_causal_residual"
        ]
        == 0.02
    )
    assert decision["minimum_improved_target_units_vs_each_primary_comparator"] == 13
    assert (
        decision[
            "minimum_improved_units_per_stratum_vs_each_primary_comparator"
        ]
        == 6
    )
    assert decision["minimum_accepted_target_units"] == 12
    assert decision["minimum_accepted_units_per_stratum"] == 6
    assert decision["maximum_harmful_accepted_units"] == 0
    assert decision["require_exact_point_and_interval_fallback_for_every_rejection"]

    contrasts = [
        {
            "metric": "primary-loss",
            "comparator": "B0_physical_fallback",
        },
        {
            "metric": "primary-loss",
            "comparator": "B1_last_causal_residual",
        },
        {
            "metric": "gaussian-nll",
            "comparator": "B1_last_causal_residual",
        },
    ]
    assert decision[
        "require_familywise_bootstrap_upper_bound_below_zero_for"
    ] == contrasts

    probability = sum(math.comb(16, count) for count in range(13, 17)) / 2**16
    assert math.isclose(probability, 697 / 65536)
    assert probability < 0.025

    bootstrap = evaluation["bootstrap"]
    assert bootstrap["replicates"] == 20000
    assert bootstrap["seed"] == 20260811
    assert bootstrap["familywise_confidence"] == 0.95
    assert bootstrap["confirmatory_contrast_count"] == 3
    assert math.isclose(
        bootstrap["per_confirmatory_contrast_confidence"],
        59 / 60,
    )
    assert bootstrap["confirmatory_contrasts"] == contrasts


def test_information_order_stops_before_target_when_source_fails() -> None:
    policy = _load_json(POLICY_PATH)
    boundary = policy["information_boundary"]
    stages = policy["stage_order"]

    assert boundary == {
        "v6_design_frozen_before_v5_terminal_outcome_use": True,
        "v6_source_selection_completed_before_v5_terminal_outcome_read": True,
        "v5_confirmation_payloads_used": False,
        "v5_confirmation_outcomes_used": False,
        "v5_result_sign_used_for_v6_go_no_go": False,
        "v6_target_payloads_opened_before_source_authorization": False,
        "v6_target_outcomes_used_for_candidate_selection": False,
        "v6_target_outcomes_used_for_covariance_or_guard_calibration": False,
        "future_frames_used_for_candidate_prediction_or_risk_score": False,
        "human_selection_allowed": False,
        "human_approval_required": False,
        "new_measurements_required": False,
        "replacement_allowed": False,
        "confirmation_side_retuning_allowed": False,
        "cross_version_retuning_allowed": False,
        "causal4d_evaluation_before_v6_decision": False,
    }
    source_selection = stages.index(
        "run-nested-source-only-candidate-covariance-and-guard-selection"
    )
    v5_terminal = stages.index("complete-or-close-v5-under-its-own-immutable-lock")
    assert source_selection < v5_terminal

    source_stop = stages.index(
        "terminate-without-target-access-if-no-stable-challenger"
    )
    target_selection = stages.index(
        "publish-metadata-only-disjoint-v6-object-session-selection-lock"
    )
    assert source_stop < target_selection

    continuation = stages.index(
        "apply-precommitted-v6-continuation-regardless-of-v5-result-sign"
    )
    assert v5_terminal < continuation < target_selection

    execution_freeze = stages.index(
        "freeze-exact-v6-execution-revisions-and-machine-authorization"
    )
    target_opening = stages.index(
        "open-sixteen-fresh-object-session-payloads-once"
    )
    assert execution_freeze < target_opening
    assert stages[-1] == "consider-causal4d-only-after-v6-positive-result"


def test_contract_workflow_and_distribution_are_data_closed() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    manifest = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    normalized_document = " ".join(document.split())

    assert isinstance(workflow, dict)
    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"contracts"}
    assert workflow["jobs"]["contracts"]["runs-on"] == "ubuntu-latest"
    assert "workflow_dispatch:" not in text
    assert "runs-on: self-hosted" not in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "/mnt/" not in text
    assert "actions/upload-artifact" not in text

    assert "include docs/deform360_fresh_object_session_v6.md" in manifest
    assert (
        "include protocols/locks/deform360_official_hub_fresh_object_session_v6.json"
        in manifest
    )
    assert "It does not edit" in document
    assert "may advance at most one challenger" in document
    assert "requires no new recording" in normalized_document
    assert (
        "No second challenger can be promoted after target opening"
        in normalized_document
    )
    assert (
        "A positive v6 result does not override or relabel a negative v5 result"
        in normalized_document
    )
