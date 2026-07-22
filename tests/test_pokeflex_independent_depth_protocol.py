import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_independent_depth_protocol import (
    load_pokeflex_independent_depth_protocol,
    validate_pokeflex_independent_depth_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "pokeflex_independent_depth_development_v1.json"
)


def test_independent_depth_protocol_keeps_every_fresh_boundary_sealed() -> None:
    result = load_pokeflex_independent_depth_protocol(PROTOCOL)
    boundary = result["payload"]["evidence_boundary"]

    assert result["prospective_development_validation_take"] == "T2"
    assert boundary["calibration_objects_remain_sealed"] is True
    assert boundary["target_objects_remain_sealed"] is True


def test_independent_depth_protocol_forbids_target_frame_anchor() -> None:
    result = load_pokeflex_independent_depth_protocol(PROTOCOL)
    causal = result["payload"]["causal_input_contract"]

    assert causal["realsense_history"] == "f-5 through f-1 only"
    assert causal["frame_f_kinect_or_realsense_allowed_before_prediction"] is False


def test_independent_depth_protocol_rejects_relaxed_false_safe_gate() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["source_competence_gates"]["maximum_false_safe_rate"] = 0.25

    with pytest.raises(ValueError, match="checksum mismatch|false-safe"):
        validate_pokeflex_independent_depth_protocol(payload)
