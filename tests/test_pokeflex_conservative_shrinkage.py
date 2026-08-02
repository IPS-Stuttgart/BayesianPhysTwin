from __future__ import annotations

from copy import deepcopy

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage import (
    evaluate_pokeflex_conservative_shrinkage_source,
    load_pokeflex_conservative_shrinkage_protocol,
)

WEAK = "checkpoint_action_local_state_relative_0.4_residual_scale_0.125"
WIDE = "checkpoint_action_local_state_relative_0.7_residual_scale_0.125"
STRONG = "checkpoint_action_local_state_relative_0.55_residual_scale_0.25"


def _artifact(object_index: int, take_index: int) -> dict[str, object]:
    baseline = 10.0 + object_index
    weak_gain = 0.2 + 0.05 * object_index
    targets = []
    updates = []
    for frame, supported in ((7, True), (8, False)):
        targets.append(
            {
                "target_frame": frame,
                "released_checkpoint_CD_UL1_mm": baseline,
                WEAK: baseline - weak_gain if supported else baseline,
                WIDE: baseline - weak_gain * 1.1 if supported else baseline,
                STRONG: (
                    baseline - 0.8
                    if supported and object_index != 1
                    else baseline + 0.4
                    if supported
                    else baseline
                ),
            }
        )
        updates.append(
            {
                "target_frame": frame,
                "accepted": supported,
                "action_supported": supported,
            }
        )
    return {
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "future_observation_used": False,
        "take": {"id": f"Object{object_index}_T{take_index}"},
        "targets": targets,
        "updates": updates,
    }


def _cohort() -> list[dict[str, object]]:
    return [_artifact(object_index, 1) for object_index in range(3)]


def test_smallest_safe_arm_is_stable_and_preserves_fallback() -> None:
    result = evaluate_pokeflex_conservative_shrinkage_source(_cohort())

    assert result["source_gate_passed"]
    assert result["selected_arm"] == WEAK
    assert result["cross_fitted"]["stable_selection"]
    assert result["cross_fitted"]["held_object_win_count"] == 3
    assert result["fallback"]["unsupported_frame_count"] == 3
    assert result["fallback"]["selected_arm_metric_mismatch_count"] == 0


def test_future_input_and_fallback_mismatch_fail_closed() -> None:
    future = _cohort()
    future[0]["future_observation_used"] = True
    with pytest.raises(ValueError, match="future"):
        evaluate_pokeflex_conservative_shrinkage_source(future)

    mismatch = _cohort()
    mismatch[0]["targets"][1][WEAK] -= 0.1
    result = evaluate_pokeflex_conservative_shrinkage_source(mismatch)
    assert not result["source_gate_passed"]
    assert result["fallback"]["selected_arm_metric_mismatch_count"] == 1


def test_take_inventory_and_candidate_bank_are_immutable() -> None:
    cohort = _cohort()
    expected = [row["take"]["id"] for row in cohort]
    result = evaluate_pokeflex_conservative_shrinkage_source(
        cohort,
        expected_take_ids=expected,
    )
    assert result["take_count"] == 3

    changed = deepcopy(cohort)
    del changed[0]["targets"][0][WEAK]
    with pytest.raises(ValueError, match="candidate"):
        evaluate_pokeflex_conservative_shrinkage_source(changed)

    with pytest.raises(ValueError, match="inventory"):
        evaluate_pokeflex_conservative_shrinkage_source(
            cohort,
            expected_take_ids=[*expected, "Object9_T1"],
        )


def test_canonical_protocol_is_valid() -> None:
    protocol = load_pokeflex_conservative_shrinkage_protocol(
        "configs/sota/pokeflex_conservative_shrinkage_source_v1.json"
    )
    assert protocol["protocol_sha256"] == (
        "73b69d3efae27d5afe511bc795c3e270546722e410aaca698db5afcc90ed23e9"
    )
    assert len(protocol["opened_source_take_ids"]) == 27
