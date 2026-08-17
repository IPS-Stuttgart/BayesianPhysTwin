import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    ACTION_ROBUST_OFFICIAL13_PUBLIC_MULTIPLIERS,
    OFFICIAL13_GLOBAL_FRAME_BALANCED_CD_UL1_MM,
    OFFICIAL13_PUBLIC_TARGET_TAKE_IDS,
    OFFICIAL18_MISSING_PUBLIC_TAKE_IDS,
    TARGET_PROTOCOL_ACTION_ROBUST_OFFICIAL13_PUBLIC_V1,
    TARGET_PROTOCOL_ACTION_ROBUST_OFFICIAL13_PUBLIC_V1_SHA256,
    evaluate_target_metrics,
    load_pokeflex_shrinkage_target_protocol,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "sota" / "pokeflex_action_robust_official13_public_v1.json"
FROZEN_FILE_SHA256 = "194d1d15c04660110e21d9c802b48d321cbc1b6025921e5c4de59aa0d442dcba"


def _per_take(
    *,
    candidate_offset: float = -0.1,
    global_offset: float = 0.0,
) -> list[dict[str, object]]:
    rows = []
    global_candidate = OFFICIAL13_GLOBAL_FRAME_BALANCED_CD_UL1_MM + global_offset
    for take_id in OFFICIAL13_PUBLIC_TARGET_TAKE_IDS:
        baseline = global_candidate + 0.2
        candidate = global_candidate + candidate_offset
        rows.append(
            {
                "take_id": take_id,
                "baseline_mean_CD_UL1_mm": baseline,
                "global_candidate_mean_CD_UL1_mm": global_candidate,
                "candidate_mean_CD_UL1_mm": candidate,
                "scored_frame_count": 1,
                "frames": [
                    {
                        "baseline_CD_UL1_mm": baseline,
                        "global_candidate_CD_UL1_mm": global_candidate,
                        "candidate_CD_UL1_mm": candidate,
                        "candidate_jaccard": None,
                    }
                ],
            }
        )
    return rows


def test_frozen_action_robust_official13_protocol_is_exact() -> None:
    protocol = load_pokeflex_shrinkage_target_protocol(FROZEN)

    assert protocol["protocol_id"] == (
        TARGET_PROTOCOL_ACTION_ROBUST_OFFICIAL13_PUBLIC_V1
    )
    assert protocol["protocol_sha256"] == (
        TARGET_PROTOCOL_ACTION_ROBUST_OFFICIAL13_PUBLIC_V1_SHA256
    )
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_FILE_SHA256
    assert tuple(protocol["target_cohort"]["take_ids"]) == (
        OFFICIAL13_PUBLIC_TARGET_TAKE_IDS
    )
    assert tuple(protocol["target_cohort"]["prospective_take_ids"]) == ()
    assert tuple(protocol["target_cohort"]["previously_opened_take_ids"]) == (
        OFFICIAL13_PUBLIC_TARGET_TAKE_IDS
    )
    assert tuple(protocol["target_cohort"]["missing_official_take_ids"]) == (
        OFFICIAL18_MISSING_PUBLIC_TAKE_IDS
    )
    assert (
        protocol["method"]["action_robust_scale_calibration"]["multipliers"]
        == ACTION_ROBUST_OFFICIAL13_PUBLIC_MULTIPLIERS
    )
    assert (
        protocol["gates"]["public_subset_numeric_reference"][
            "direct_full18_comparison_authorized"
        ]
        is False
    )


def test_action_robust_official13_gate_requires_both_references() -> None:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))

    passed = evaluate_target_metrics(_per_take(), protocol)
    tied = evaluate_target_metrics(_per_take(candidate_offset=0.0), protocol)
    reproduction_drift = evaluate_target_metrics(
        _per_take(global_offset=2e-9),
        protocol,
    )

    assert passed["checkpoint_pairing"]["passed"] is True
    assert passed["global_scale_advancement"]["passed"] is True
    assert passed["global_scale_reproduction_passed"] is True
    assert passed["public_subset_numeric_reference_passed"] is True
    assert passed["all_target_gates_passed"] is True
    assert passed["published_direct_comparison_authorized"] is False
    assert tied["global_scale_advancement"]["passed"] is False
    assert tied["all_target_gates_passed"] is False
    assert reproduction_drift["global_scale_reproduction_passed"] is False
    assert reproduction_drift["all_target_gates_passed"] is False


@pytest.mark.parametrize("object_name", ["MemoryFoam", "3dPrintedBunny", "FoamDice"])
def test_action_robust_official13_global_fallback_is_locked(object_name: str) -> None:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    changed = deepcopy(protocol)
    changed["method"]["action_robust_scale_calibration"]["multipliers"][object_name] = (
        2.0
    )
    changed["method"]["effective_scale_by_object"][object_name] = 0.25
    changed["protocol_sha256"] = target_protocol_sha256(changed)

    with pytest.raises(ValueError, match="multiplier map"):
        validate_pokeflex_shrinkage_target_protocol(
            changed,
            bind_action_robust_digest=False,
        )


def test_action_robust_official13_cannot_gain_prospective_status() -> None:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    changed = deepcopy(protocol)
    changed["target_cohort"]["prospective_take_ids"] = [
        OFFICIAL13_PUBLIC_TARGET_TAKE_IDS[0]
    ]
    changed["protocol_sha256"] = target_protocol_sha256(changed)

    with pytest.raises(ValueError, match="gained prospective status"):
        validate_pokeflex_shrinkage_target_protocol(
            changed,
            bind_action_robust_digest=False,
        )
