import copy

import pytest

from bayesian_phystwin.pokeflex_dual_sensor_regret_guard import (
    evaluate_pokeflex_dual_sensor_consensus,
)


CANDIDATE = "checkpoint_action_local_state_relative_0.55_residual_scale_0.5"


def _payload(
    *,
    d405_regret: list[float],
    kinect_regret: list[float],
    candidate_error: float = 8.0,
) -> dict[str, object]:
    return {
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "future_observation_used": False,
        "online_observation_regret_recorded": True,
        "take": {"id": "FoamDice_T1"},
        "targets": [
            {
                "target_frame": 7,
                "released_checkpoint_CD_UL1_mm": 10.0,
                CANDIDATE: candidate_error,
                "independent_anchor_regret": {
                    CANDIDATE: {
                        "evaluated_prediction_frame": 6,
                        "per_sensor_mm": d405_regret,
                        "covariance_intersection_upper_mm": max(d405_regret),
                    }
                },
                "online_observation_regret": {
                    CANDIDATE: {
                        "per_view_mm": kinect_regret,
                        "covariance_intersection_upper_mm": max(kinect_regret),
                    }
                },
            }
        ],
    }


def test_consensus_accepts_only_when_both_sensor_families_improve() -> None:
    accepted = evaluate_pokeflex_dual_sensor_consensus(
        [_payload(d405_regret=[-2.0, -1.0], kinect_regret=[-0.5, -0.25])]
    )
    rejected = evaluate_pokeflex_dual_sensor_consensus(
        [_payload(d405_regret=[-2.0, -1.0], kinect_regret=[-0.5, 0.25])]
    )

    assert accepted["accepted_frame_count"] == 1
    assert accepted["selected_object_mean_CD_UL1_mm"] == 8.0
    assert accepted["decisions"][0]["consensus_upper_regret_mm"] == -0.25
    assert rejected["accepted_frame_count"] == 0
    assert rejected["selected_object_mean_CD_UL1_mm"] == 10.0
    assert rejected["decisions"][0]["selected_arm"] == "released_checkpoint"


def test_duplicating_correlated_observation_does_not_increase_confidence() -> None:
    original = evaluate_pokeflex_dual_sensor_consensus(
        [_payload(d405_regret=[-2.0, -1.0], kinect_regret=[-0.5, -0.25])]
    )
    duplicated = evaluate_pokeflex_dual_sensor_consensus(
        [
            _payload(
                d405_regret=[-2.0, -1.0, -1.0],
                kinect_regret=[-0.5, -0.25, -0.25],
            )
        ]
    )

    assert duplicated["decisions"][0]["consensus_upper_regret_mm"] == (
        original["decisions"][0]["consensus_upper_regret_mm"]
    )


def test_empty_delayed_evidence_returns_exact_baseline() -> None:
    payload = _payload(d405_regret=[-1.0], kinect_regret=[-1.0])
    payload["targets"][0]["independent_anchor_regret"] = {}
    payload["targets"][0]["online_observation_regret"] = {}

    result = evaluate_pokeflex_dual_sensor_consensus([payload])

    assert result["accepted_frame_count"] == 0
    assert result["exact_fallback_frame_count"] == 1
    assert result["decisions"][0]["selected_error_mm"] == (
        result["decisions"][0]["baseline_error_mm"]
    )


def test_rejects_future_observation_or_wrong_take_inventory() -> None:
    payload = _payload(d405_regret=[-1.0], kinect_regret=[-1.0])
    future = copy.deepcopy(payload)
    future["future_observation_used"] = True

    with pytest.raises(ValueError, match="future input"):
        evaluate_pokeflex_dual_sensor_consensus([future])
    with pytest.raises(ValueError, match="take inventory"):
        evaluate_pokeflex_dual_sensor_consensus(
            [payload], expected_take_ids=["FoamDice_T2"]
        )
