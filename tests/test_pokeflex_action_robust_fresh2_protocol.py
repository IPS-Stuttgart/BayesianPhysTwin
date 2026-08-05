import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    ACTION_ROBUST_FRESH2_PUBLIC_TARGET_TAKE_IDS,
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH2_V5,
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH2_V5_SHA256,
    evaluate_target_metrics,
    load_pokeflex_shrinkage_target_protocol,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    ROOT / "configs" / "sota" / "pokeflex_action_robust_shrinkage_fresh2_v5.json"
)
FROZEN_FILE_SHA256 = "bb2b59aea13ca6c6e271295e4c0d6b703345d29929dda79c019157669273dc35"


def _per_take(candidate_offset: float = -0.1) -> list[dict[str, object]]:
    rows = []
    for index, take_id in enumerate(ACTION_ROBUST_FRESH2_PUBLIC_TARGET_TAKE_IDS):
        baseline = 6.0 + index
        global_candidate = baseline - 0.1
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


def test_frozen_final_two_protocol_is_exact() -> None:
    protocol = load_pokeflex_shrinkage_target_protocol(FROZEN)

    assert protocol["protocol_id"] == TARGET_PROTOCOL_ACTION_ROBUST_FRESH2_V5
    assert protocol["protocol_sha256"] == (
        TARGET_PROTOCOL_ACTION_ROBUST_FRESH2_V5_SHA256
    )
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_FILE_SHA256
    assert tuple(protocol["target_cohort"]["take_ids"]) == (
        ACTION_ROBUST_FRESH2_PUBLIC_TARGET_TAKE_IDS
    )
    assert protocol["method"]["action_robust_scale_calibration"]["multipliers"][
        "Pillow"
    ] == 2.0
    assert protocol["method"]["action_robust_scale_calibration"]["multipliers"][
        "PlushDice"
    ] == 4.0


def test_final_two_gate_requires_both_objects_to_beat_both_references() -> None:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    passed = evaluate_target_metrics(_per_take(), protocol)
    tied = evaluate_target_metrics(_per_take(candidate_offset=0.0), protocol)

    assert passed["checkpoint_pairing"]["win_count"] == 2
    assert passed["global_scale_advancement"]["win_count"] == 2
    assert passed["all_target_gates_passed"] is True
    assert tied["global_scale_advancement"]["passed"] is False
    assert tied["all_target_gates_passed"] is False


def test_final_two_protocol_rejects_freshness_or_extension_mutation() -> None:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    changed = deepcopy(protocol)
    changed["freshness_audit"]["selected_zip_sha256"]["Pillow_T4"] = "0" * 64
    changed["protocol_sha256"] = target_protocol_sha256(changed)
    with pytest.raises(ValueError, match="archive bytes"):
        validate_pokeflex_shrinkage_target_protocol(
            changed,
            bind_action_robust_digest=False,
        )

    changed = deepcopy(protocol)
    changed["source_gate"]["all18_source_extension"][
        "parent_rows_used_by_target_unchanged"
    ] = False
    changed["protocol_sha256"] = target_protocol_sha256(changed)
    with pytest.raises(ValueError, match="target scale rows"):
        validate_pokeflex_shrinkage_target_protocol(
            changed,
            bind_action_robust_digest=False,
        )
