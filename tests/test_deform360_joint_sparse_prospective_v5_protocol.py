from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_joint_sparse_prospective_v5.json"
)
V4_POLICY_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_joint_sparse_observability_v4.json"
)
SELECTION_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
DOCUMENT_PATH = ROOT / "docs/deform360_joint_sparse_prospective_v5.md"
WORKFLOW_PATH = (
    ROOT / ".github/workflows/deform360-joint-sparse-prospective-v5-contracts.yml"
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


def test_policy_identity_and_frozen_cohort_bindings() -> None:
    policy = _load_json(POLICY_PATH)
    selection = _load_json(SELECTION_PATH)
    v4_policy = _load_json(V4_POLICY_PATH)

    assert policy["schema"] == (
        "bayesian-phystwin.deform360-joint-sparse-prospective-policy"
    )
    assert policy["schema_version"] == 5
    assert policy["semantics"] == (
        "prospective-object-level-joint-sparse-visuotactile-benefit-v5"
    )
    assert policy["policy_id"] == _content_id(policy, excluded="policy_id")

    selection_lock = policy["selection_lock"]
    assert selection_lock["path"] == SELECTION_PATH.relative_to(ROOT).as_posix()
    assert selection_lock["selection_sha256"] == selection["selection_sha256"]
    assert (
        selection_lock["selection_artifact_sha256"]
        == selection["selection_artifact_sha256"]
    )
    assert selection_lock["development_object_count"] == 10
    assert selection_lock["confirmation_object_count"] == 12
    assert selection_lock["development_stratum_counts"] == {
        "sheet": 5,
        "volumetric": 5,
    }
    assert selection_lock["confirmation_stratum_counts"] == {
        "sheet": 6,
        "volumetric": 6,
    }

    prerequisites = policy["prerequisites"]
    assert (
        prerequisites["v4_policy_path"] == V4_POLICY_PATH.relative_to(ROOT).as_posix()
    )
    assert prerequisites["v4_policy_id"] == v4_policy["policy_id"]
    assert prerequisites["required_v4_status"] == "development-design-supported"
    assert prerequisites["minimum_v4_supported_objects"] == 8
    assert prerequisites["minimum_v4_supported_objects_per_stratum"] == 4


def test_partial_factors_are_admitted_at_object_query_level() -> None:
    policy = _load_json(POLICY_PATH)
    admission = policy["factor_admission"]
    evaluation = policy["evaluation"]
    boundary = policy["information_boundary"]

    assert admission["admission_unit"] == "physical-object-query"
    assert admission["all_registered_cameras_retained_in_provenance"] is True
    assert admission["unsupported_partial_factor_action"] == (
        "retain-with-zero-likelihood"
    )
    assert admission["technical_failure_action"] == (
        "retain-and-deploy-exact-physical-fallback"
    )
    assert admission["replacement_allowed"] is False
    assert "minimum_supported_stream_fraction" not in admission

    assert admission["minimum_distinct_cameras"] == 2
    assert admission["minimum_distinct_causal_windows"] == 2
    assert admission["minimum_distinct_spatial_clusters"] == 8
    assert admission["require_full_registered_query_rank"] is True
    assert admission["maximum_single_camera_information_fraction"] == 0.85
    assert admission["minimum_leave_one_camera_rank_fraction"] == 0.75
    assert admission["minimum_leave_one_window_rank_fraction"] == 0.75

    assert evaluation["statistical_unit"] == "physical-object"
    assert evaluation["within_object_rows_do_not_increase_sample_size"] is True
    assert evaluation["all_confirmation_objects_in_denominator"] is True
    assert evaluation["rejected_or_unsupported_candidate_uses_exact_fallback"] is True
    assert evaluation["registered_action_frame_range_half_open"] == [0, 81]
    assert evaluation["observation_causal_frame_range_half_open"] == [0, 58]
    assert evaluation["future_evaluation_frame_range_half_open"] == [58, 76]
    assert evaluation["unscored_terminal_buffer_frame_range_half_open"] == [
        76,
        81,
    ]
    assert evaluation["official_geometry_processing_revision"] == (
        "d8522a4403b766aeb387510c04e89032a56fdf35"
    )
    endpoint_views = evaluation["endpoint_view_reservation"]
    assert endpoint_views["views_per_object"] == 2
    assert endpoint_views["reserved_views_contribute_likelihood"] is False
    assert endpoint_views["same_rule_source_and_confirmation"] is True
    assert endpoint_views["pixel_values_or_outcomes_used_for_selection"] is False
    assert (
        evaluation["primary_endpoint"]["observation_views_excluded_from_target_views"]
        is True
    )

    assert boundary == {
        "adaptive_confirmation_payloads_used": False,
        "causal4d_evaluation_before_v5_decision": False,
        "confirmation_payloads_opened_before_protocol_freeze": False,
        "confirmation_side_retuning_allowed": False,
        "development_objects_previously_opened": True,
        "future_frames_used_for_source_calibration": False,
        "human_selection_allowed": False,
        "replacement_allowed": False,
        "target_outcomes_used_for_method_or_threshold_selection": False,
    }


def test_primary_decision_is_object_level_and_difficult_to_win_by_fallback() -> None:
    policy = _load_json(POLICY_PATH)
    decision = policy["positive_decision"]
    methods = {row["method_id"]: row["role"] for row in policy["methods"]}

    assert methods == {
        "B0_physical_fallback": "baseline",
        "B1_last_causal_residual": "registered-reference",
        "V1_joint_sparse_visual_guarded": "mechanism-comparator",
        "T1_contact_anchor_only": "mechanism-comparator",
        "VT1_joint_sparse_visuotactile_guarded": "primary-candidate",
        "VT2_joint_sparse_visuotactile_unguarded": "safety-diagnostic",
        "VT3_joint_sparse_visuotactile_anchor_bias": "bias-diagnostic",
    }
    assert decision["primary_method_id"] == "VT1_joint_sparse_visuotactile_guarded"
    assert decision["minimum_relative_improvement_vs_physical_fallback"] == 0.1
    assert decision["minimum_relative_improvement_vs_last_causal_residual"] == 0.05
    assert decision["minimum_contact_increment_over_visual_only"] == 0.02
    assert decision["minimum_accepted_confirmation_objects"] == 10
    assert decision["minimum_accepted_objects_per_stratum"] == 5
    assert (
        decision["minimum_improved_confirmation_objects_vs_each_primary_comparator"]
        == 10
    )
    assert (
        decision["minimum_improved_objects_per_stratum_vs_each_primary_comparator"] == 5
    )
    assert decision["maximum_harmful_accepted_objects"] == 0
    assert decision["harmful_update_relative_margin"] == 0.02
    assert decision["maximum_stratum_mean_regression"] == 0.02
    assert decision["require_all_checks"] is True
    assert decision["require_exact_fallback_for_every_rejection"] is True
    assert decision["require_paired_bootstrap_upper_bound_below_zero_vs"] == [
        "B0_physical_fallback",
        "B1_last_causal_residual",
    ]

    one_sided_sign_probability = sum(
        math.comb(12, improved_count) for improved_count in range(10, 13)
    ) / (2**12)
    assert math.isclose(one_sided_sign_probability, 79 / 4096)
    assert one_sided_sign_probability < 0.025

    bootstrap = policy["evaluation"]["bootstrap"]
    assert bootstrap == {
        "confidence": 0.95,
        "contract": "bayesian-phystwin-group-clustered-paired-bootstrap-v1",
        "replicates": 10000,
        "seed": 20260810,
    }


def test_stage_order_freezes_source_before_one_time_confirmation() -> None:
    policy = _load_json(POLICY_PATH)
    stages = policy["stage_order"]
    calibration = policy["source_calibration"]
    implementation = policy["frozen_implementation"]

    assert stages == [
        "run-v4-structural-development-on-ten-opened-objects",
        "freeze-v5-policy-and-exact-software-revisions",
        "fit-and-cross-validate-source-only-calibration-on-ten-development-objects",
        "seal-source-calibration-and-confirmation-opening-authorization",
        "open-twelve-confirmation-payloads-once",
        "evaluate-all-seven-methods-with-all-objects-in-denominator",
        "publish-positive-or-negative-object-level-result",
        "consider-causal4d-only-after-v5-positive-result",
    ]
    assert calibration["strategy"] == (
        "nested-leave-one-object-out-cross-fit-then-full-source-refit"
    )
    assert calibration["confirmation_adaptation"] == "forbidden"
    assert calibration["equal_object_weighting"] is True
    assert calibration["minimum_passing_objects"] == 8
    assert calibration["minimum_passing_objects_per_stratum"] == 4
    assert calibration["risk_score_semantics"] == (
        "lower-is-safer-inclusive-threshold-v1"
    )
    assert calibration["risk_threshold_tie_policy"] == (
        "accept-complete-tied-score-blocks"
    )
    assert calibration["population_harm_risk_certificate_claimed"] is False
    assert implementation["causal4d_primary_evaluation"] is False
    assert (
        implementation["source_and_confirmation_use_identical_execution_revisions"]
        is True
    )
    assert (
        implementation["exact_execution_revisions_frozen_before_source_residual_fit"]
        is True
    )


def test_contract_workflow_is_hosted_read_only_and_data_closed() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(workflow, dict)
    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"contracts"}
    assert workflow["jobs"]["contracts"]["runs-on"] == "ubuntu-latest"

    assert "workflow_dispatch:" not in text
    assert "runs-on: self-hosted" not in text
    assert "contents: write" not in text
    assert "persist-credentials: true" not in text
    assert "git push" not in text
    assert "/mnt/lexar4tb" not in text
    assert "DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT" not in text
    assert "actions/upload-artifact" not in text
    assert "confirmation payload" not in text.lower()
    assert "target outcome" not in text.lower()


def test_documentation_and_source_distribution_include_the_design() -> None:
    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    manifest = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()

    assert "A camera can be only partially informative" in document
    assert (
        "every rejected or unsupported candidate returns the unchanged physical"
        in document.lower()
    )
    assert "The physical object is the sole independent statistical unit" in document
    assert "Causal4D is deliberately not part of the primary experiment" in document
    assert "open-twelve-confirmation-payloads-once" not in document
    assert "include docs/deform360_joint_sparse_prospective_v5.md" in manifest
    assert (
        "include "
        "protocols/locks/deform360_official_hub_joint_sparse_prospective_v5.json"
        in manifest
    )
