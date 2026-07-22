from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_force_depth_regret_guard import (
    FEATURE_NAMES,
    FORCE_FIELD_LOCK,
    FROZEN_CANDIDATE_RUNNER_SHA256,
    evaluate_pokeflex_force_depth_cross_object,
    extract_pokeflex_force_depth_rows,
)
from bayesian_phystwin.pokeflex_independent_depth_protocol import (
    POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
)


def _artifact(object_index: int, take_index: int) -> dict[str, object]:
    take_id = f"Object{object_index}_T{take_index}"
    candidates = (
        "checkpoint_force_action_plane_local_state_relative_0.4_residual_scale_0.125",
        "checkpoint_force_action_plane_local_state_relative_0.4_residual_scale_0.25",
    )
    targets = []
    updates = []
    for target_frame in (7, 8):
        baseline = 10.0 + 0.1 * object_index + 0.01 * take_index
        targets.append(
            {
                "target_frame": target_frame,
                "released_checkpoint_CD_UL1_mm": baseline,
                candidates[0]: baseline - 1.0,
                candidates[1]: baseline + 0.5,
                "independent_anchor_regret": {
                    candidates[0]: {
                        "per_sensor_mm": [-0.8, -0.7],
                    },
                    candidates[1]: {
                        "per_sensor_mm": [0.4, 0.5],
                    },
                },
            }
        )
        updates.append(
            {
                "target_frame": target_frame,
                "rms_update_m": 0.004,
                "correction_to_prior_motion_ratio": 1.5,
                "prior_motion_rms_m": 0.003,
                "correction_prior_motion_cosine": 0.5,
                "previous_correction_cosine": 0.7,
                "force_y": 20.0,
                "force_y_delta": 1.0,
                "median_robust_weight": 0.9,
                "downweighted_fraction": 0.1,
                "assignment_variance_m2_mean": 0.002**2,
            }
        )
    return {
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "future_observation_used": False,
        "take": {"id": take_id},
        "force_depth_regret_development": {
            "candidate_runner_sha256": FROZEN_CANDIDATE_RUNNER_SHA256,
            "candidate_fields": list(FORCE_FIELD_LOCK),
            "candidate_scales": [0.0, 0.125, 0.25, 0.5, 1.0],
            "measured_force_and_tool_motion_used": True,
            "d405_evidence": "frame f-1 only",
            "prediction_target": "frame f",
            "future_observation_used": False,
            "target_objects_opened": False,
        },
        "independent_depth_anchor": {
            "protocol_sha256": POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
            "median_residual_mm": [2.0, 3.0],
        },
        "updates": updates,
        "targets": targets,
    }


def _cohort() -> list[dict[str, object]]:
    return [
        _artifact(object_index, take_index)
        for object_index in range(5)
        for take_index in range(1, 4)
    ]


def test_hidden_outcome_does_not_change_force_depth_features() -> None:
    first = _artifact(0, 1)
    second = deepcopy(first)
    candidate = (
        "checkpoint_force_action_plane_local_state_relative_0.4_residual_scale_0.125"
    )
    second["targets"][0][candidate] = 1000.0

    first_rows, _ = extract_pokeflex_force_depth_rows([first])
    second_rows, _ = extract_pokeflex_force_depth_rows([second])

    np.testing.assert_array_equal(first_rows[0]["features"], second_rows[0]["features"])
    assert len(first_rows[0]["features"]) == len(FEATURE_NAMES)
    assert first_rows[0]["regret_mm"] != second_rows[0]["regret_mm"]


def test_nested_cross_object_guard_selects_beneficial_force_arm() -> None:
    result = evaluate_pokeflex_force_depth_cross_object(_cohort())

    assert result["cross_object"]["gate_passed"]
    assert result["cross_object"]["object_wins"] == 5
    assert result["cross_object"]["accepted_frame_losses"] == 0
    assert result["cross_object"]["object_balanced_relative_improvement"] > 0.09
    assert (
        result["selector_controls"]["predicted_mean"][
            "object_balanced_relative_improvement"
        ]
        > 0.09
    )
    assert (
        result["candidate_bank_oracle"]["object_balanced_relative_improvement"] > 0.09
    )
    assert result["cross_fitting"].startswith("outer leave-one-object-out")
    assert result["fixed_arm_controls"]["maximin"]["gate_passed"]
    assert result["fixed_arm_controls"]["maximin"]["object_wins"] == 5


def test_outer_held_object_cannot_change_its_fold_calibration() -> None:
    first = _cohort()
    second = deepcopy(first)
    candidate = (
        "checkpoint_force_action_plane_local_state_relative_0.4_residual_scale_0.125"
    )
    for artifact in second:
        if artifact["take"]["id"].startswith("Object4_"):
            for target in artifact["targets"]:
                target[candidate] += 1000.0

    first_result = evaluate_pokeflex_force_depth_cross_object(first)
    second_result = evaluate_pokeflex_force_depth_cross_object(second)

    assert (
        first_result["fold_candidate_certificates"]["Object4"]
        == second_result["fold_candidate_certificates"]["Object4"]
    )
    assert (
        first_result["fold_selector_bounds"]["Object4"]
        == second_result["fold_selector_bounds"]["Object4"]
    )


def test_unsupported_force_depth_features_fall_back_exactly() -> None:
    cohort = _cohort()
    for artifact in cohort:
        if artifact["take"]["id"].startswith("Object4_"):
            for update in artifact["updates"]:
                update["force_y"] = 1_000_000.0
    result = evaluate_pokeflex_force_depth_cross_object(cohort)
    held = [row for row in result["decisions"] if row["object"] == "Object4"]

    assert held
    assert all(not row["accepted"] for row in held)
    assert all(row["selected_error_mm"] == row["baseline_error_mm"] for row in held)


def test_future_input_duplicate_take_and_runner_drift_are_rejected() -> None:
    future = _artifact(0, 1)
    future["future_observation_used"] = True
    with pytest.raises(ValueError, match="future"):
        extract_pokeflex_force_depth_rows([future])

    duplicate = _artifact(0, 1)
    with pytest.raises(ValueError, match="duplicate"):
        extract_pokeflex_force_depth_rows([duplicate, deepcopy(duplicate)])

    drifted = _artifact(0, 1)
    drifted["force_depth_regret_development"]["candidate_runner_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        extract_pokeflex_force_depth_rows([drifted])
