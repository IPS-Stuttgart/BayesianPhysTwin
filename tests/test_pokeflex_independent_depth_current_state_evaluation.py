import pytest

from bayesian_phystwin.pokeflex_independent_depth_current_state_evaluation import (
    BASELINE_ARM,
    evaluate_current_state_artifacts,
    evaluate_current_state_take,
)


CANDIDATE = "checkpoint_action_local_state_relative_0.7_residual_scale_1"


def _artifact(take_id: str = "FoamDice_T1") -> dict[str, object]:
    return {
        "artifact_kind": "PokeFlexIndependentDepthCurrentStateDiagnostic",
        "future_observation_used": False,
        "target_mesh_used_by_diagnostic_runner": False,
        "take": {"id": take_id},
        "independent_depth_anchor": {"median_residual_mm": [2.0, 3.0]},
        "targets": [
            {
                "source_frame": 6,
                "target_frame": 7,
                "released_checkpoint_CD_UL1_mm": 5.0,
                CANDIDATE: 4.0,
                "current_state_anchor_regret": {
                    CANDIDATE: {
                        "evidence_frame": 6,
                        "target_prediction_frame": 7,
                        "per_sensor_mm": [-0.8, -0.6],
                    }
                },
            },
            {
                "source_frame": 7,
                "target_frame": 8,
                "released_checkpoint_CD_UL1_mm": 4.0,
                CANDIDATE: 4.5,
                "current_state_anchor_regret": {
                    CANDIDATE: {
                        "evidence_frame": 7,
                        "target_prediction_frame": 8,
                        "per_sensor_mm": [0.2, 0.1],
                    }
                },
            },
        ],
    }


def test_same_time_selector_uses_source_frame_evidence_for_next_target() -> None:
    result = evaluate_current_state_take(_artifact())

    assert result["selector"]["rows"][0]["selected_arm"] == CANDIDATE
    assert result["selector"]["rows"][1]["selected_arm"] == BASELINE_ARM
    assert result["selector"]["relative_improvement"] == pytest.approx(1.0 / 9.0)


def test_same_time_selector_excludes_failed_sensor() -> None:
    payload = _artifact()
    payload["independent_depth_anchor"]["median_residual_mm"] = [20.0, 2.0]
    payload["targets"][0]["current_state_anchor_regret"][CANDIDATE][
        "per_sensor_mm"
    ] = [-2.0, 0.3]

    result = evaluate_current_state_take(payload)

    assert result["selector"]["rows"][0]["selected_arm"] == BASELINE_ARM


def test_current_state_aggregate_balances_objects() -> None:
    first = _artifact("FoamDice_T1")
    second = _artifact("MemoryFoam_T1")

    result = evaluate_current_state_artifacts([first, second])

    assert result["object_count"] == 2
    assert result["object_balanced_selector"]["object_wins"] == 2
