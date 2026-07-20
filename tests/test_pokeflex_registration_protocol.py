import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_registration_protocol import (
    POKEFLEX_OBJECTS,
    load_pokeflex_registration_protocol,
    validate_pokeflex_registration_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "pokeflex_bayesian_registration_v1.json"
)


def test_pokeflex_registration_protocol_is_object_disjoint_and_exhaustive() -> None:
    result = load_pokeflex_registration_protocol(PROTOCOL)

    partitions = (
        set(result["development_objects"]),
        set(result["calibration_objects"]),
        set(result["target_objects"]),
        set(result["excluded_objects"]),
    )
    assert set().union(*partitions) == set(POKEFLEX_OBJECTS)
    for index, first in enumerate(partitions):
        for second in partitions[index + 1 :]:
            assert not first & second


def test_pokeflex_registration_protocol_locks_causal_history() -> None:
    result = load_pokeflex_registration_protocol(PROTOCOL)
    inputs = result["payload"]["causal_input_contract"]

    assert inputs["allowed_observation_frames"] == "f-5 through f-1 only"
    assert inputs["forbidden_observation_frames"] == "f and all later frames"
    assert inputs["synthetic_point_clouds_from_target_mesh_allowed"] is False


def test_pokeflex_registration_protocol_rejects_target_leakage() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["causal_input_contract"][
        "target_take_deformed_mesh_allowed_before_final_scoring"
    ] = True

    with pytest.raises(ValueError, match="checksum mismatch|causal input"):
        validate_pokeflex_registration_protocol(payload)


def test_pokeflex_registration_protocol_rejects_overlap() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["cohort"]["target_objects"].append("FoamDice")

    with pytest.raises(ValueError, match="checksum mismatch|overlap"):
        validate_pokeflex_registration_protocol(payload)
