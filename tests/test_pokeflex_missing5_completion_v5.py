import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_missing5_completion_v5 import (
    PARENT_PROTOCOL_FILE_SHA256,
    SOURCE_PROTOCOL_FILE_SHA256,
    SOURCE_RESULT_FILE_SHA256,
    TARGET_MULTIPLIERS,
    build_completion_protocol,
    protocol_sha256,
    validate_completion_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs" / "sota" / "pokeflex_action_robust_official18_v4.json"
SOURCE_PROTOCOL = ROOT / "configs" / "sota" / "pokeflex_missing5_scale_source_v5.json"
SOURCE_RESULT = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_missing5_scale_source_v5"
    / "source_result.json"
)
FROZEN_PROTOCOL = (
    ROOT / "configs" / "sota" / "pokeflex_missing5_scale_completion_v5.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build() -> dict[str, object]:
    return build_completion_protocol(
        _load(PARENT),
        _load(SOURCE_PROTOCOL),
        _load(SOURCE_RESULT),
        locked_at_utc="2026-08-06T08:00:00Z",
        parent_protocol_file_sha256=PARENT_PROTOCOL_FILE_SHA256,
        source_protocol_file_sha256=SOURCE_PROTOCOL_FILE_SHA256,
        source_result_file_sha256=SOURCE_RESULT_FILE_SHA256,
    )


def test_builder_freezes_only_the_five_unavailable_target_scales() -> None:
    protocol = _build()
    validation = validate_completion_protocol(
        protocol,
        bind_registered_digest=False,
    )

    assert validation["passed"] is True
    assert validation["target_multipliers"] == {
        "Pillow_T8": 1.0,
        "3dPrintedCylinder_T7": 2.0,
        "3dPrintedHeart_T14": 1.5,
        "Sponge_T10": 1.0,
        "3dPrintedPizza_T13": 1.0,
    }
    assert protocol["method"]["target_effective_scales"] == {
        "Pillow_T8": 0.125,
        "3dPrintedCylinder_T7": 0.25,
        "3dPrintedHeart_T14": 0.1875,
        "Sponge_T10": 0.125,
        "3dPrintedPizza_T13": 0.125,
    }
    assert protocol["parent_v4"]["unchanged"] is True
    assert protocol["target_cohort"]["target_outcomes_opened_at_lock"] is False


def test_registered_completion_protocol_is_exact() -> None:
    protocol = _load(FROZEN_PROTOCOL)

    validation = validate_completion_protocol(protocol)

    assert validation["protocol_sha256"] == (
        "11d6eb1ff115f0021e1ab9ad959b0dfd614ca455e5f54d1dd05c99e9b916c7de"
    )
    assert hashlib.sha256(FROZEN_PROTOCOL.read_bytes()).hexdigest() == (
        "960eb903634c621b0ea2244a2039cb92da974f9a196d54c77622cdd40f2ab271"
    )
    assert protocol["target_cohort"]["target_outcomes_opened_at_lock"] is False
    assert protocol["held_v8_accessed"] is False


def test_resigned_target_scale_change_is_rejected() -> None:
    protocol = _build()
    changed = deepcopy(protocol)
    changed["method"]["target_multipliers"]["Pillow_T8"] = 2.0
    changed["method"]["target_effective_scales"]["Pillow_T8"] = 0.25
    changed["protocol_sha256"] = protocol_sha256(changed)

    with pytest.raises(ValueError, match="multiplier"):
        validate_completion_protocol(changed, bind_registered_digest=False)


def test_resigned_gate_weakening_is_rejected() -> None:
    protocol = _build()
    changed = deepcopy(protocol)
    changed["gates"]["prospective_v5_vs_v4"][
        "minimum_per_object_relative_improvement"
    ] = -0.01
    changed["protocol_sha256"] = protocol_sha256(changed)

    with pytest.raises(ValueError, match="per-object"):
        validate_completion_protocol(changed, bind_registered_digest=False)


def test_builder_rejects_failed_or_modified_source_gate() -> None:
    source_result = _load(SOURCE_RESULT)
    source_result["source_gate"]["passed"] = False
    from bayesian_phystwin.pokeflex_missing5_scale import result_sha256

    source_result["result_sha256"] = result_sha256(source_result)

    with pytest.raises(ValueError, match="registered source result|did not pass"):
        build_completion_protocol(
            _load(PARENT),
            _load(SOURCE_PROTOCOL),
            source_result,
            locked_at_utc="2026-08-06T08:00:00Z",
            parent_protocol_file_sha256=PARENT_PROTOCOL_FILE_SHA256,
            source_protocol_file_sha256=SOURCE_PROTOCOL_FILE_SHA256,
            source_result_file_sha256=SOURCE_RESULT_FILE_SHA256,
        )


def test_target_map_constant_matches_expected_objects() -> None:
    assert set(TARGET_MULTIPLIERS) == {
        "Pillow_T8",
        "3dPrintedCylinder_T7",
        "3dPrintedHeart_T14",
        "Sponge_T10",
        "3dPrintedPizza_T13",
    }
