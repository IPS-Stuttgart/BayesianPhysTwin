import json
import math
from copy import deepcopy

import pytest

from causal4d.real_protocol import build_same_object_real_protocol, write_protocol
from causal4d.rest_geometry_cross_action import (
    build_rest_geometry_candidate_evidence,
    build_candidate_evidence_from_case_summaries,
    build_rest_geometry_protocol_result,
    build_rest_geometry_transfer_result_record,
    build_rest_geometry_transfer_plan,
    candidate_evidence_sha256,
    select_all_rest_geometry_fold_locks,
    select_rest_geometry_fold_lock,
    transfer_plan_sha256,
    transfer_result_record_sha256,
    validate_rest_geometry_candidate_evidence,
    validate_rest_geometry_transfer_plan,
    validate_rest_geometry_protocol_result,
    write_rest_geometry_cross_action_selection,
    write_rest_geometry_protocol_result,
)


def _hyperparameters(
    *,
    frame_mode="none",
    frame_scale=0.0,
    rest_scale=0.0,
    controller_mode="preserve",
):
    return {
        "frame_mode": frame_mode,
        "frame_scale": frame_scale,
        "rest_geometry_scale": rest_scale,
        "controller_rest_mode": controller_mode,
        "graph_prior_strength": 0.1,
        "rest_length_ratio_bound": 1.15,
    }


def _candidate(hyperparameters, error):
    return {
        "hyperparameters": hyperparameters,
        "validation_track_error_by_frame_m": [error, error, error],
        "validation_track_error_mean_m": error,
    }


def _evidence(protocol, execution_id, index):
    complex_error = 0.60 if index % 2 == 0 else 1.05
    return build_rest_geometry_candidate_evidence(
        protocol,
        execution_id,
        [
            _candidate(_hyperparameters(), 1.0),
            _candidate(_hyperparameters(rest_scale=0.25), 0.85),
            _candidate(
                _hyperparameters(
                    frame_mode="se3",
                    frame_scale=1.0,
                    rest_scale=1.0,
                    controller_mode="recompute",
                ),
                complex_error,
            ),
        ],
        fixed_config_sha256="a" * 64,
        source_artifacts_sha256=f"{index + 1:064x}",
    )


def _all_evidence(protocol):
    return {
        execution["execution_id"]: _evidence(
            protocol, execution["execution_id"], index
        )
        for index, execution in enumerate(protocol["executions"])
    }


def test_candidate_evidence_rejects_a_rehashed_holdout_boundary_change() -> None:
    protocol = build_same_object_real_protocol()
    evidence = _evidence(protocol, protocol["executions"][0]["execution_id"], 0)
    leaked = deepcopy(evidence)
    leaked["information_boundary"]["holdout_frames_used"] = True
    leaked["evidence_sha256"] = candidate_evidence_sha256(leaked)

    with pytest.raises(ValueError, match="holdout boundary"):
        validate_rest_geometry_candidate_evidence(protocol, leaked)


def _case_summary(frame_mode):
    candidates = [
        {
            "frame_scale": 0.0,
            "rest_geometry_scale": 0.0,
            "controller_rest_mode": "preserve",
            "track_error_by_frame_m": [0.01, 0.012],
            "track_error_mean_m": 0.011,
        },
        {
            "frame_scale": 1.0,
            "rest_geometry_scale": 0.5,
            "controller_rest_mode": "recompute",
            "track_error_by_frame_m": [0.008, 0.01],
            "track_error_mean_m": 0.009,
        },
    ]
    return {
        "config": {
            "frame_mode": frame_mode,
            "graph_prior_strength": 0.1,
            "inner_validation_frames": 2,
            "velocity_history_frames": 3,
            "maximum_frame_rotation_rad": 0.1,
            "maximum_frame_translation_m": 0.02,
            "maximum_nonrigid_norm_m": 0.01,
            "maximum_rest_log_ratio": math.log(1.15),
            "dt": 5e-5,
            "num_substeps": 667,
            "self_collision": False,
            "deterministic_spring_forces": True,
        },
        "information_boundary": {
            "holdout_frames_used_for_inference": False,
            "holdout_frames_used_for_hyperparameter_selection": False,
            "manual_gt_track_used_for_hyperparameter_selection": False,
        },
        "selection": {"candidates": candidates},
        "inputs": {
            "final_data": {"sha256": "1" * 64},
            "baseline_trajectory": {"sha256": "2" * 64},
            "optimal_params": {"sha256": "3" * 64},
            "checkpoint": {"sha256": "4" * 64},
            "gt_track_3d": {"sha256": "5" * 64},
            "official_repo": {"commit": "deadbeef"},
        },
    }


def test_case_summary_adapter_emits_only_pre_holdout_candidate_evidence() -> None:
    protocol = build_same_object_real_protocol()
    execution_id = protocol["executions"][0]["execution_id"]

    evidence = build_candidate_evidence_from_case_summaries(
        protocol,
        execution_id,
        [_case_summary("none"), _case_summary("se3")],
    )

    assert validate_rest_geometry_candidate_evidence(protocol, evidence)[
        "candidate_count"
    ] == 4
    assert evidence["information_boundary"]["holdout_frames_used"] is False


def test_fold_selection_uses_exact_fit_ids_and_one_standard_error_rule() -> None:
    protocol = build_same_object_real_protocol()
    fold = protocol["splits"]["cross_action_contact_calibration_folds"][0]
    all_evidence = _all_evidence(protocol)
    fit_evidence = {
        execution_id: all_evidence[execution_id]
        for execution_id in fold["fit_execution_ids"]
    }

    lock = select_rest_geometry_fold_lock(
        protocol,
        fold["fold_id"],
        fit_evidence,
    )

    assert lock["selected_hyperparameters"]["rest_geometry_scale"] == 0.25
    assert lock["selected_hyperparameters"]["frame_scale"] == 0.0
    assert lock["information_boundary"]["target_execution_evidence_used"] is False
    leaked_input = {**fit_evidence, fold["target_execution_ids"][0]: all_evidence[fold["target_execution_ids"][0]]}
    with pytest.raises(ValueError, match="exactly its fit executions"):
        select_rest_geometry_fold_lock(protocol, fold["fold_id"], leaked_input)


def test_all_fold_locks_build_exact_factual_and_transfer_tracks() -> None:
    protocol = build_same_object_real_protocol()
    locks = select_all_rest_geometry_fold_locks(protocol, _all_evidence(protocol))

    plan = build_rest_geometry_transfer_plan(protocol, locks)
    validation = validate_rest_geometry_transfer_plan(protocol, locks, plan)

    assert len(locks) == 12
    assert validation == {
        "record_count": 66,
        "factual_count": 36,
        "same_grasp_count": 18,
        "new_contact_count": 12,
        "passed": True,
    }
    assert all(
        not record["target_response_prefix_allowed"]
        for record in plan["same_grasp_intervention_prediction"]
        + plan["new_contact_intervention_prediction"]
    )


def test_transfer_plan_rejects_rehashed_source_substitution() -> None:
    protocol = build_same_object_real_protocol()
    locks = select_all_rest_geometry_fold_locks(protocol, _all_evidence(protocol))
    plan = build_rest_geometry_transfer_plan(protocol, locks)
    changed = deepcopy(plan)
    changed["same_grasp_intervention_prediction"][0]["source_execution_id"] = (
        protocol["executions"][-1]["execution_id"]
    )
    changed["plan_sha256"] = transfer_plan_sha256(changed)

    with pytest.raises(ValueError, match="source/target policy"):
        validate_rest_geometry_transfer_plan(protocol, locks, changed)


_METHOD_FACTORS = {
    "released": 1.0,
    "endpoint_restart": 0.99,
    "output_frame_graph": 0.80,
    "frame_state_original_rest": 0.98,
    "graph_state_original_rest": 1.01,
    "rest_geometry_only": 0.94,
    "frame_rest_geometry": 0.92,
    "frame_rest_geometry_reattached": 0.93,
    "selected_frame_rest_geometry": 0.90,
}


def _result_metrics():
    return {
        method: {
            "future_chamfer_distance_m": [0.01 * factor] * 3,
            "future_track_error_m": [0.02 * factor] * 3,
        }
        for method, factor in _METHOD_FACTORS.items()
    }


def _result_records(plan):
    plan_records = (
        plan["factual_continuation"]
        + plan["same_grasp_intervention_prediction"]
        + plan["new_contact_intervention_prediction"]
    )
    return [
        build_rest_geometry_transfer_result_record(
            plan_record,
            _result_metrics(),
            canonical_material_graph_sha256="b" * 64,
            source_correction_sha256=f"{1000 + index:064x}",
            target_rollout_bundle_sha256=f"{2000 + index:064x}",
        )
        for index, plan_record in enumerate(plan_records)
    ]


def test_protocol_result_recomputes_all_three_transfer_aggregates() -> None:
    protocol = build_same_object_real_protocol()
    locks = select_all_rest_geometry_fold_locks(protocol, _all_evidence(protocol))
    plan = build_rest_geometry_transfer_plan(protocol, locks)

    result = build_rest_geometry_protocol_result(
        protocol,
        locks,
        plan,
        _result_records(plan),
    )

    assert validate_rest_geometry_protocol_result(
        protocol, locks, plan, result
    )["passed"] is True
    for track in result["aggregate"].values():
        assert track["selected_frame_rest_geometry"][
            "future_track_error_m"
        ]["equal_record_macro_percent_change"] == pytest.approx(-10.0)


def test_protocol_result_rejects_rehashed_target_response_leakage() -> None:
    protocol = build_same_object_real_protocol()
    locks = select_all_rest_geometry_fold_locks(protocol, _all_evidence(protocol))
    plan = build_rest_geometry_transfer_plan(protocol, locks)
    records = _result_records(plan)
    transfer_record = next(
        record
        for record in records
        if record["evaluation_track"] == "same_grasp_intervention_prediction"
    )
    transfer_record["target_response_prefix_allowed"] = True
    transfer_record["record_sha256"] = transfer_result_record_sha256(transfer_record)

    with pytest.raises(ValueError, match="locked plan record"):
        build_rest_geometry_protocol_result(protocol, locks, plan, records)


def test_protocol_result_rejects_mixed_material_graphs() -> None:
    protocol = build_same_object_real_protocol()
    locks = select_all_rest_geometry_fold_locks(protocol, _all_evidence(protocol))
    plan = build_rest_geometry_transfer_plan(protocol, locks)
    records = _result_records(plan)
    records[-1]["canonical_material_graph_sha256"] = "c" * 64
    records[-1]["record_sha256"] = transfer_result_record_sha256(records[-1])

    with pytest.raises(ValueError, match="one canonical graph"):
        build_rest_geometry_protocol_result(protocol, locks, plan, records)


def test_file_pipeline_writes_locks_plan_and_final_result(tmp_path) -> None:
    protocol = build_same_object_real_protocol()
    protocol_path = write_protocol(tmp_path / "protocol.json", protocol)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    for execution_id, evidence in _all_evidence(protocol).items():
        (evidence_root / f"{execution_id}.json").write_text(
            json.dumps(evidence), encoding="utf-8"
        )
    selection_root = tmp_path / "selection"

    selection = write_rest_geometry_cross_action_selection(
        protocol_path,
        evidence_root,
        selection_root,
    )
    plan = json.loads(
        (selection_root / "rest_geometry_transfer_plan.json").read_text(
            encoding="utf-8"
        )
    )
    records_root = tmp_path / "records"
    records_root.mkdir()
    for index, record in enumerate(_result_records(plan)):
        (records_root / f"record-{index:02d}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    output = tmp_path / "result.json"

    aggregate = write_rest_geometry_protocol_result(
        protocol_path,
        selection_root,
        records_root,
        output,
    )

    assert selection["fold_lock_count"] == 12
    assert selection["transfer_record_count"] == 66
    assert aggregate["record_count"] == 66
    assert output.is_file()
