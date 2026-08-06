import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_action_robust_scale import (
    ACTION_ROBUST_SCALE_FILE_SHA256,
    ACTION_ROBUST_SCALE_SHA256,
    action_robust_control_summary,
    build_action_robust_scale_calibration,
    calibration_sha256,
    select_action_robust_multiplier,
    validate_action_robust_scale_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "sota" / "pokeflex_action_robust_scale_v3.json"


def _row(
    take_id: str,
    scores: dict[float, float],
    *,
    support: int = 20,
) -> dict[str, object]:
    return {
        "take_id": take_id,
        "supported_frame_count": support,
        "mean_CD_UL1_mm_by_multiplier": {
            str(value): scores.get(value, 11.0)
            for value in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
        },
    }


def test_repeated_action_selector_keeps_consistent_gain() -> None:
    rows = [
        _row("Object_T1", {1.0: 10.0, 2.0: 9.5}),
        _row("Object_T2", {1.0: 12.0, 2.0: 11.6}),
    ]

    selected = select_action_robust_multiplier(rows)

    assert selected["multiplier"] == 2.0
    assert selected["minimum_source_relative_improvement"] > 0.0
    assert selected["selection_reason"] == "repeated-action-maximin"


def test_repeated_action_selector_rejects_conflicting_gain() -> None:
    rows = [
        _row("Object_T1", {0.5: 9.0, 1.0: 10.0, 4.0: 11.0}),
        _row("Object_T2", {0.5: 11.0, 1.0: 10.0, 4.0: 9.0}),
    ]

    selected = select_action_robust_multiplier(rows)

    assert selected["multiplier"] == 1.0
    assert selected["selection_reason"] == "global-lower-envelope-fallback"


def test_missing_repeated_action_support_uses_global_scale() -> None:
    rows = [
        _row("Object_T1", {1.0: 10.0, 2.0: 9.0}),
        _row("Object_T2", {1.0: 10.0, 2.0: 9.0}, support=0),
    ]

    selected = select_action_robust_multiplier(rows)

    assert selected["multiplier"] == 1.0
    assert selected["selection_reason"] == "insufficient-repeated-action-support"


def test_production_selector_controls_detect_signal_and_reject_placebo() -> None:
    controls = action_robust_control_summary()

    assert controls["passed"] is True
    assert controls["positive_detection_count"] == 12
    assert controls["placebo_deviation_count"] == 0


def test_builder_and_validator_bind_both_source_actions() -> None:
    objects = [f"Object{index}" for index in range(12)]
    first_rows = [_row(f"{name}_T1", {1.0: 10.0, 2.0: 9.8}) for name in objects]
    second_rows = [_row(f"{name}_T2", {1.0: 10.0, 2.0: 9.9}) for name in objects]
    first = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexFresh12PostopenScaleHeadroomAudit",
        "audit_sha256": (
            "78179996296b5ed47692e3ee716308c4525deeb71ce2881442331b5643b4bf94"
        ),
        "status": "post-open diagnostic; not prospective evidence",
        "takes": first_rows,
    }
    second = {
        "schema_version": 1,
        "artifact_kind": (
            "PokeFlexInstanceFresh12PostopenGlobalScaleHeadroomCompactAudit"
        ),
        "audit_sha256": (
            "08bc71efec3c8c99bb469efada2a82048b978a0f5fdde3a149c47b60ac395587"
        ),
        "status": "post-open development diagnostic; not prospective evidence",
        "takes": second_rows,
    }

    calibration = build_action_robust_scale_calibration(first, second)
    validation = validate_action_robust_scale_calibration(
        calibration,
        bind_registered_digest=False,
    )

    assert validation["passed"] is True
    assert set(validation["multipliers"].values()) == {2.0}
    assert calibration["source_gate"]["source_action_regression_count"] == 0
    assert calibration["calibration_sha256"] == calibration_sha256(calibration)

    changed = deepcopy(calibration)
    changed["objects"]["Object0"]["multiplier"] = 4.0
    with pytest.raises(ValueError, match="checksum"):
        validate_action_robust_scale_calibration(
            changed,
            bind_registered_digest=False,
        )


def test_frozen_action_robust_calibration_is_exact() -> None:
    payload = json.loads(FROZEN.read_text(encoding="utf-8"))
    validation = validate_action_robust_scale_calibration(payload)

    assert payload["calibration_sha256"] == ACTION_ROBUST_SCALE_SHA256
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == (
        ACTION_ROBUST_SCALE_FILE_SHA256
    )
    assert validation["multipliers"] == {
        "3dPrintedCylinder": 3.0,
        "3dPrintedPizza": 0.5,
        "3dPrintedPyramid": 1.0,
        "Beanbag": 4.0,
        "FoamCylinder": 3.0,
        "FoamHalfSphere": 2.0,
        "Pillow": 2.0,
        "PlushDice": 4.0,
        "PlushMoon": 4.0,
        "PlushTurtle": 4.0,
        "PlushVolleyball": 1.0,
        "Sponge": 1.5,
    }
    assert payload["source_gate"] == {
        "adjusted_object_count": 10,
        "controls_passed": True,
        "mean_source_action_relative_improvement": 0.012887797451845465,
        "minimum_source_action_relative_improvement": 0.0,
        "passed": True,
        "source_action_count": 24,
        "source_action_regression_count": 0,
        "source_object_count": 12,
    }
