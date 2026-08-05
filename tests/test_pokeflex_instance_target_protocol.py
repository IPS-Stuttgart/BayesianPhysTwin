import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    INSTANCE_FRESH12_PUBLIC_TARGET_TAKE_IDS,
    TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
    TARGET_PROTOCOL_INSTANCE_FRESH12_V2_SHA256,
    evaluate_target_metrics,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT / "scripts" / "development" / "build_pokeflex_instance_target_protocol.py"
)
BASE = (
    ROOT / "configs" / "sota" / "pokeflex_conservative_shrinkage_fresh12_public_v1.json"
)
FRESHNESS = (
    ROOT / "configs" / "sota" / "pokeflex_instance_fresh12_exclusion_audit_v2.json"
)
CALIBRATION = ROOT / "configs" / "sota" / "pokeflex_instance_scale_calibration_v2.json"
FROZEN = ROOT / "configs" / "sota" / "pokeflex_instance_shrinkage_fresh12_v2.json"
FROZEN_FILE_SHA256 = "0ba63c8d435c781fe52237be42fd4e0debef56ddb863b028e5b4b3fa8a518cf1"


def _builder_module():
    spec = importlib.util.spec_from_file_location(
        "instance_target_builder", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build():
    module = _builder_module()
    return module.build_protocol(
        json.loads(BASE.read_text(encoding="utf-8")),
        json.loads(FRESHNESS.read_text(encoding="utf-8")),
        json.loads(CALIBRATION.read_text(encoding="utf-8")),
        freshness_path=FRESHNESS.relative_to(ROOT),
        calibration_path=CALIBRATION.relative_to(ROOT),
        locked_at_utc="2026-08-04T18:19:58Z",
    )


def test_instance_target_protocol_is_deterministic_and_complete() -> None:
    first = _build()
    second = _build()

    assert first == second
    assert first["protocol_id"] == TARGET_PROTOCOL_INSTANCE_FRESH12_V2
    assert tuple(first["target_cohort"]["take_ids"]) == (
        INSTANCE_FRESH12_PUBLIC_TARGET_TAKE_IDS
    )
    assert validate_pokeflex_shrinkage_target_protocol(first)["passed"] is True


def test_frozen_instance_target_protocol_matches_builder_bytes() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    assert frozen == _build()
    assert frozen["protocol_sha256"] == TARGET_PROTOCOL_INSTANCE_FRESH12_V2_SHA256
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_FILE_SHA256


def test_instance_target_protocol_rejects_multiplier_mutation() -> None:
    protocol = _build()
    protocol["method"]["instance_scale_calibration"]["multipliers"][
        "3dPrintedPizza"
    ] = 2.0

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_pokeflex_shrinkage_target_protocol(protocol)


def test_instance_target_protocol_semantically_binds_effective_scales() -> None:
    protocol = _build()
    protocol["method"]["effective_scale_by_object"]["3dPrintedPizza"] = 0.25
    protocol["protocol_sha256"] = target_protocol_sha256(protocol)

    with pytest.raises(ValueError, match="effective scale map changed"):
        validate_pokeflex_shrinkage_target_protocol(protocol)


def _target_rows(
    *,
    baseline: float,
    global_candidate: float,
    instance_candidate: float,
):
    return [
        {
            "take_id": take_id,
            "scored_frame_count": 1,
            "baseline_mean_CD_UL1_mm": baseline,
            "global_candidate_mean_CD_UL1_mm": global_candidate,
            "candidate_mean_CD_UL1_mm": instance_candidate,
            "frames": [
                {
                    "baseline_CD_UL1_mm": baseline,
                    "global_candidate_CD_UL1_mm": global_candidate,
                    "candidate_CD_UL1_mm": instance_candidate,
                    "candidate_jaccard": None,
                }
            ],
        }
        for take_id in INSTANCE_FRESH12_PUBLIC_TARGET_TAKE_IDS
    ]


def test_instance_target_reports_all_three_paired_arms() -> None:
    result = evaluate_target_metrics(
        _target_rows(
            baseline=5.0,
            global_candidate=4.9,
            instance_candidate=4.8,
        ),
        _build(),
    )

    assert result["global_scale_checkpoint_pairing"]["passed"] is True
    assert result["checkpoint_pairing"]["passed"] is True
    assert result["global_scale_advancement"]["passed"] is True
    assert result["all_target_gates_passed"] is True


def test_instance_target_rejects_nonfinite_global_control() -> None:
    rows = _target_rows(
        baseline=5.0,
        global_candidate=4.9,
        instance_candidate=4.8,
    )
    rows[0]["frames"][0]["global_candidate_CD_UL1_mm"] = float("nan")

    with pytest.raises(ValueError, match="global frames are non-finite"):
        evaluate_target_metrics(rows, _build())
