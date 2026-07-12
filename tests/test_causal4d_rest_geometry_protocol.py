from copy import deepcopy

import pytest

from causal4d.real_protocol import (
    build_same_object_real_protocol,
    scaffold_dataset,
)
from causal4d.rest_geometry_protocol import (
    audit_rest_geometry_dataset_readiness,
    build_rest_geometry_analysis_plan,
    validate_rest_geometry_fold_result,
)


def test_analysis_plan_preserves_all_locked_fold_roles() -> None:
    protocol = build_same_object_real_protocol()

    plan = build_rest_geometry_analysis_plan(protocol)

    assert plan["fold_count"] == 12
    assert plan["execution_count"] == 36
    assert plan["information_boundary"][
        "target_actions_may_select_hyperparameters"
    ] is False
    target_ids = [
        execution_id
        for fold in plan["folds"]
        for execution_id in fold["target_execution_ids"]
    ]
    assert sorted(target_ids) == sorted(
        execution["execution_id"] for execution in protocol["executions"]
    )


def test_scaffold_fails_closed_until_real_manifests_exist(tmp_path) -> None:
    protocol = build_same_object_real_protocol()
    scaffold_dataset(protocol, tmp_path)

    readiness = audit_rest_geometry_dataset_readiness(tmp_path)

    assert readiness["ready_for_confirmatory_analysis"] is False
    assert readiness["status"] == "awaiting_acquisition"
    assert readiness["complete_manifest_count"] == 0
    assert readiness["template_manifest_count"] == 36
    assert readiness["missing_manifest_count"] == 0
    assert readiness["missing_top_level_artifacts"] == [
        "object_registration.json",
        "slip_pilot.json",
    ]


def _valid_fold_result(protocol, fold):
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "fold_id": fold["fold_id"],
        "hyperparameter_fit_execution_ids": fold["fit_execution_ids"],
        "calibration_execution_ids": fold["calibration_execution_ids"],
        "target_execution_ids": fold["target_execution_ids"],
        "information_boundary": {
            "target_holdout_frames_used_for_inference": False,
            "target_outcomes_used_for_hyperparameter_selection": False,
            "manual_gt_track_used_for_hyperparameter_selection": False,
            "individual_counterfactual_ground_truth_claimed": False,
        },
        "target_records": [
            {
                "execution_id": execution_id,
                "correction_evidence": "pre_holdout_only",
                "shared_hyperparameters_frozen_before_target": True,
            }
            for execution_id in fold["target_execution_ids"]
        ],
    }


def test_fold_result_validator_rejects_target_selected_hyperparameters() -> None:
    protocol = build_same_object_real_protocol()
    fold = protocol["splits"]["cross_action_contact_calibration_folds"][0]
    result = _valid_fold_result(protocol, fold)
    assert validate_rest_geometry_fold_result(protocol, result)["passed"] is True
    leaked = deepcopy(result)
    leaked["information_boundary"][
        "target_outcomes_used_for_hyperparameter_selection"
    ] = True

    with pytest.raises(ValueError, match="information boundary"):
        validate_rest_geometry_fold_result(protocol, leaked)
