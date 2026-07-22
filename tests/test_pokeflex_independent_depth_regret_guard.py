from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_independent_depth_regret_guard import (
    FEATURE_NAMES,
    evaluate_pokeflex_regret_guard_cross_object,
    evaluate_pokeflex_regret_guard_prospective,
    extract_pokeflex_regret_guard_rows,
)
from bayesian_phystwin.pokeflex_independent_depth_protocol import (
    POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
)


def _artifact(object_index: int, take_index: int) -> dict[str, object]:
    take_id = f"Object{object_index}_T{take_index}"
    candidates = (
        "checkpoint_action_local_state_relative_0.4_residual_scale_0.125",
        "checkpoint_action_local_state_relative_0.4_residual_scale_0.25",
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


def test_hidden_outcome_does_not_change_candidate_features() -> None:
    first = _artifact(0, 1)
    second = deepcopy(first)
    candidate = "checkpoint_action_local_state_relative_0.4_residual_scale_0.125"
    second["targets"][0][candidate] = 1000.0

    first_rows, _ = extract_pokeflex_regret_guard_rows([first])
    second_rows, _ = extract_pokeflex_regret_guard_rows([second])

    np.testing.assert_array_equal(first_rows[0]["features"], second_rows[0]["features"])
    assert len(first_rows[0]["features"]) == len(FEATURE_NAMES)
    assert first_rows[0]["regret_mm"] != second_rows[0]["regret_mm"]


def test_cross_object_guard_selects_supported_beneficial_arm() -> None:
    cohort = _cohort()
    result = evaluate_pokeflex_regret_guard_cross_object(cohort)

    assert result["cross_object"]["gate_passed"]
    assert result["cross_object"]["object_wins"] == 5
    assert result["cross_object"]["accepted_frame_losses"] == 0
    assert result["cross_object"]["object_balanced_relative_improvement"] > 0.09
    assert len(
        result["deployment_artifact"]["selector_correction_bound"][
            "group_scores_mm"
        ]
    ) == 15

    prospective = evaluate_pokeflex_regret_guard_prospective(
        cohort[:3],
        result,
        expected_take_ids=["Object0_T1", "Object0_T2", "Object0_T3"],
    )
    assert prospective["object_balanced_relative_improvement"] > 0.09
    assert prospective["accepted_frame_losses"] == 0


def test_future_input_and_duplicate_take_are_rejected() -> None:
    future = _artifact(0, 1)
    future["future_observation_used"] = True
    with pytest.raises(ValueError, match="future"):
        extract_pokeflex_regret_guard_rows([future])

    duplicate = _artifact(0, 1)
    with pytest.raises(ValueError, match="duplicate"):
        extract_pokeflex_regret_guard_rows([duplicate, deepcopy(duplicate)])
