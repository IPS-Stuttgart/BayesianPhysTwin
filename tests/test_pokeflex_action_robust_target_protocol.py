import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    ACTION_ROBUST_FRESH6_PUBLIC_TARGET_TAKE_IDS,
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3_SHA256,
    evaluate_target_metrics,
    load_pokeflex_shrinkage_target_protocol,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "sota" / "pokeflex_action_robust_shrinkage_fresh6_v3.json"
FROZEN_FILE_SHA256 = "173434fe5916c57dd4e8809f098152b096c6cf09e7efdf37190b488ee5cc7263"


def _per_take(candidate_offset: float = -0.1) -> list[dict[str, object]]:
    rows = []
    for index, take_id in enumerate(ACTION_ROBUST_FRESH6_PUBLIC_TARGET_TAKE_IDS):
        baseline = 6.0 + index * 0.1
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


def test_frozen_action_robust_protocol_is_exact() -> None:
    protocol = load_pokeflex_shrinkage_target_protocol(FROZEN)

    assert protocol["protocol_id"] == TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3
    assert protocol["protocol_sha256"] == (
        TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3_SHA256
    )
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_FILE_SHA256
    assert tuple(protocol["target_cohort"]["take_ids"]) == (
        ACTION_ROBUST_FRESH6_PUBLIC_TARGET_TAKE_IDS
    )
    assert (
        protocol["method"]["action_robust_scale_calibration"]["multipliers"][
            "3dPrintedPyramid"
        ]
        == 1.0
    )


def test_action_robust_metric_gate_requires_advancement_over_global() -> None:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    passed = evaluate_target_metrics(_per_take(), protocol)
    tied = evaluate_target_metrics(_per_take(candidate_offset=0.0), protocol)

    assert passed["checkpoint_pairing"]["passed"] is True
    assert passed["global_scale_advancement"]["passed"] is True
    assert passed["all_target_gates_passed"] is True
    assert tied["global_scale_advancement"]["passed"] is False
    assert tied["all_target_gates_passed"] is False


def test_action_robust_protocol_rejects_multiplier_change() -> None:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    changed = deepcopy(protocol)
    changed["method"]["action_robust_scale_calibration"]["multipliers"]["Pillow"] = 4.0
    changed["protocol_sha256"] = target_protocol_sha256(changed)
    with pytest.raises(ValueError, match="multiplier map"):
        validate_pokeflex_shrinkage_target_protocol(
            changed,
            bind_action_robust_digest=False,
        )
