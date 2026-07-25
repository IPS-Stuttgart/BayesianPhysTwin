import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_robot_fusion_protocol import (
    load_pokeflex_robot_fusion_source_protocol,
    validate_pokeflex_robot_fusion_source_protocol,
)


PROTOCOL = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "pokeflex_robot_fusion_source_v1.json"
)


def test_robot_fusion_source_protocol_is_locked() -> None:
    result = load_pokeflex_robot_fusion_source_protocol(PROTOCOL)

    assert result["passed"] is True
    assert result["development_objects"] == (
        "FoamDice",
        "MemoryFoam",
        "PlushOctopus",
        "3dPrintedHeart",
        "ToiletPaperRoll",
    )
    assert result["source_takes"] == ("T1", "T4", "T5", "T6")


def test_robot_fusion_protocol_rejects_target_opening() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["evidence_boundary"]["target_objects_remain_sealed"] = False

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_pokeflex_robot_fusion_source_protocol(payload)


def test_robot_fusion_runner_has_no_target_object_inventory() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_pokeflex_robot_fusion_source.py"
    ).read_text(encoding="utf-8")

    assert "target_objects_opened\": False" in source
    assert "3dPrintedCylinder" not in source
    assert "PlushTurtle" not in source
    assert "target geometry is loaded only after" in source.lower()
