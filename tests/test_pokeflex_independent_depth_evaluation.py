import copy

import pytest

from bayesian_phystwin.pokeflex_independent_depth_evaluation import (
    BASELINE_ARM,
    evaluate_independent_depth_artifacts,
    evaluate_independent_depth_take,
    evaluate_locked_independent_depth_source_validation,
)


CANDIDATE_A = "checkpoint_action_local_state_relative_0.7_residual_scale_1"
CANDIDATE_B = "checkpoint_action_local_state_relative_0.4_residual_scale_0.5"


def _target(
    frame: int,
    baseline: float,
    candidate_a: float,
    candidate_b: float,
    *,
    predicted_a: float | None = None,
    predicted_b: float | None = None,
) -> dict[str, object]:
    regrets = {}
    if predicted_a is not None and predicted_b is not None:
        regrets = {
            CANDIDATE_A: {
                "evaluated_prediction_frame": frame - 1,
                "per_sensor_mm": [predicted_a - 0.1, predicted_a],
                "mean_mm": predicted_a - 0.05,
                "covariance_intersection_upper_mm": predicted_a,
            },
            CANDIDATE_B: {
                "evaluated_prediction_frame": frame - 1,
                "per_sensor_mm": [predicted_b, predicted_b - 0.2],
                "mean_mm": predicted_b - 0.1,
                "covariance_intersection_upper_mm": predicted_b,
            },
        }
    return {
        "target_frame": frame,
        "released_checkpoint_CD_UL1_mm": baseline,
        CANDIDATE_A: candidate_a,
        CANDIDATE_B: candidate_b,
        "independent_anchor_regret": regrets,
    }


def _artifact(take_id: str = "FoamDice_T3") -> dict[str, object]:
    return {
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "future_observation_used": False,
        "take": {"id": take_id},
        "independent_depth_anchor": {
            "future_observation_used": False,
            "median_residual_mm": [2.0, 3.0],
        },
        "targets": [
            _target(6, 5.0, 4.0, 5.5),
            _target(7, 4.5, 4.0, 5.0, predicted_a=-0.8, predicted_b=0.4),
            _target(8, 4.0, 4.3, 3.5, predicted_a=0.2, predicted_b=0.6),
        ],
    }


def test_competence_aligns_anchor_to_preceding_hidden_prediction() -> None:
    result = evaluate_independent_depth_take(_artifact())
    rows = result["competence"]["rows"]
    row = next(value for value in rows if value["candidate"] == CANDIDATE_A)

    assert row["evidence_frame"] == 7
    assert row["evaluated_prediction_frame"] == 6
    assert row["hidden_regret_mm"] == pytest.approx(-1.0)
    assert result["competence"]["sign_agreement"] == 0.75


def test_one_frame_lag_selector_uses_current_evidence_for_current_target() -> None:
    result = evaluate_independent_depth_take(_artifact())
    rows = result["selector"]["rows"]

    assert rows[0]["selected_arm"] == BASELINE_ARM
    assert rows[1]["selected_arm"] == CANDIDATE_A
    assert rows[1]["selected_CD_UL1_mm"] == 4.0
    assert rows[2]["selected_arm"] == BASELINE_ARM
    assert result["selector"]["wins"] == 1
    assert result["selector"]["losses"] == 0
    assert result["selector"]["fallback_ties"] == 2


def test_selector_margin_preserves_exact_fallback() -> None:
    result = evaluate_independent_depth_take(
        _artifact(), minimum_anchor_improvement_mm=0.9
    )

    assert all(
        row["selected_arm"] == BASELINE_ARM for row in result["selector"]["rows"]
    )
    assert result["selector"]["selected_mean_CD_UL1_mm"] == pytest.approx(
        result["selector"]["baseline_mean_CD_UL1_mm"]
    )


def test_failed_sensor_is_excluded_without_consulting_outcome() -> None:
    payload = _artifact()
    payload["independent_depth_anchor"]["median_residual_mm"] = [20.0, 2.0]
    evidence = payload["targets"][1]["independent_anchor_regret"][CANDIDATE_A]
    evidence["per_sensor_mm"] = [-2.0, 0.3]

    result = evaluate_independent_depth_take(payload)

    assert result["sensor_quality"]["eligible_sensor_indices"] == [1]
    assert result["selector"]["rows"][1]["selected_arm"] == BASELINE_ARM


def test_no_calibrated_sensor_returns_exact_fallback() -> None:
    payload = _artifact()
    payload["independent_depth_anchor"]["median_residual_mm"] = [20.0, 30.0]

    result = evaluate_independent_depth_take(payload)

    assert result["competence"]["candidate_frame_pair_count"] == 0
    assert all(
        row["selected_arm"] == BASELINE_ARM for row in result["selector"]["rows"]
    )


def test_invalid_future_anchor_alignment_is_rejected() -> None:
    payload = _artifact()
    payload["targets"][1]["independent_anchor_regret"][CANDIDATE_A][
        "evaluated_prediction_frame"
    ] = 7

    with pytest.raises(ValueError, match="preceding prediction"):
        evaluate_independent_depth_take(payload)


def test_object_balanced_aggregate_does_not_weight_longer_take_more() -> None:
    first = _artifact("FoamDice_T3")
    second = copy.deepcopy(_artifact("MemoryFoam_T3"))
    for target in second["targets"]:
        target["released_checkpoint_CD_UL1_mm"] *= 2.0
        target[CANDIDATE_A] *= 2.0
        target[CANDIDATE_B] *= 2.0

    result = evaluate_independent_depth_artifacts([first, second])

    expected_baseline = (4.5 + 9.0) / 2.0
    expected_selected = ((5.0 + 4.0 + 4.0) / 3.0 + (10.0 + 8.0 + 8.0) / 3.0) / 2.0
    assert result["object_balanced_selector"][
        "baseline_mean_CD_UL1_mm"
    ] == pytest.approx(expected_baseline)
    assert result["object_balanced_selector"][
        "selected_mean_CD_UL1_mm"
    ] == pytest.approx(expected_selected)


def test_duplicate_take_artifact_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate take"):
        evaluate_independent_depth_artifacts([_artifact(), _artifact()])


def test_locked_source_validation_rejects_missing_take_inventory() -> None:
    protocol = {
        "artifact_kind": "PokeFlexIndependentDepthSourceValidationProtocol",
        "protocol_sha256": "a" * 64,
        "evidence_boundary": {
            "development_objects": ["FoamDice"],
            "source_validation_takes": ["T1", "T4"],
        },
        "method_lock": {
            "static_template_support_radius_mm": 15.0,
            "minimum_anchor_improvement_mm": 0.0,
            "maximum_calibration_median_residual_mm": 10.0,
        },
        "source_validation": {},
    }

    with pytest.raises(ValueError, match="inventory"):
        evaluate_locked_independent_depth_source_validation(
            [_artifact("FoamDice_T1")], protocol
        )
