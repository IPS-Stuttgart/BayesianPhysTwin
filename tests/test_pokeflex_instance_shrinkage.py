import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_instance_shrinkage import (
    INSTANCE_SCALE_CALIBRATION_FILE_SHA256,
    INSTANCE_SCALE_CALIBRATION_SHA256,
    build_instance_scale_calibration,
    calibration_sha256,
    select_source_multiplier,
    validate_instance_scale_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUDIT = (
    ROOT / "results" / "sota" / "pokeflex_fresh12_scale_headroom_v1" / "audit.json"
)
CALIBRATION = ROOT / "configs" / "sota" / "pokeflex_instance_scale_calibration_v2.json"


def test_source_multiplier_uses_exact_default_without_support() -> None:
    row = {
        "supported_frame_count": 0,
        "mean_CD_UL1_mm_by_multiplier": {
            "0.5": 1.0,
            "1.0": 1.0,
            "1.5": 1.0,
            "2.0": 1.0,
        },
    }

    assert select_source_multiplier(row) == (1.0, "no-supported-source-update")


def test_source_multiplier_rejects_an_unregistered_bank() -> None:
    with pytest.raises(ValueError, match="bank changed"):
        select_source_multiplier(
            {"supported_frame_count": 1},
            candidate_multipliers=(0.5, 1.0),
        )


def test_instance_calibration_is_deterministic_and_bounded() -> None:
    source = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))

    first = build_instance_scale_calibration(source)
    second = build_instance_scale_calibration(source)
    validation = validate_instance_scale_calibration(first)

    assert first == second
    assert validation["passed"] is True
    assert first["calibration_sha256"] == INSTANCE_SCALE_CALIBRATION_SHA256
    assert validation["multipliers"] == {
        "3dPrintedCylinder": 2.0,
        "3dPrintedPizza": 0.5,
        "3dPrintedPyramid": 0.5,
        "Beanbag": 2.0,
        "FoamCylinder": 2.0,
        "FoamHalfSphere": 2.0,
        "Pillow": 2.0,
        "PlushDice": 2.0,
        "PlushMoon": 2.0,
        "PlushTurtle": 2.0,
        "PlushVolleyball": 1.0,
        "Sponge": 1.5,
    }


def test_frozen_calibration_bytes_match_the_builder() -> None:
    source = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    frozen = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    assert hashlib.sha256(CALIBRATION.read_bytes()).hexdigest() == (
        INSTANCE_SCALE_CALIBRATION_FILE_SHA256
    )
    assert frozen == build_instance_scale_calibration(source)


def test_calibration_rejects_future_target_access() -> None:
    source = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    calibration = build_instance_scale_calibration(source)
    calibration["future_take_outcomes_opened"] = True
    calibration["calibration_sha256"] = calibration_sha256(calibration)

    with pytest.raises(ValueError, match="future target access"):
        validate_instance_scale_calibration(calibration)
