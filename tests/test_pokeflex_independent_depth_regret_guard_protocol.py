import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_independent_depth_regret_guard_protocol import (
    pokeflex_regret_guard_prospective_sha256,
    validate_pokeflex_regret_guard_prospective_protocol,
)


PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "pokeflex_independent_depth_regret_guard_prospective_v1.json"
)


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _resign(payload: dict[str, object]) -> None:
    payload["protocol_sha256"] = pokeflex_regret_guard_prospective_sha256(payload)


def test_prospective_protocol_matches_canonical_lock() -> None:
    result = validate_pokeflex_regret_guard_prospective_protocol(_payload())

    assert result["passed"] is True
    assert result["take_ids"] == (
        "FoamDice_T7",
        "FoamDice_T8",
        "PlushOctopus_T7",
    )


def test_prospective_protocol_rejects_take_replacement() -> None:
    payload = copy.deepcopy(_payload())
    payload["prospective_cohort"]["take_ids"][-1] = "PlushOctopus_T8"
    _resign(payload)

    with pytest.raises(ValueError, match="canonical lock"):
        validate_pokeflex_regret_guard_prospective_protocol(payload)


def test_prospective_protocol_rejects_weaker_gate() -> None:
    payload = copy.deepcopy(_payload())
    payload["evaluation"]["minimum_object_wins"] = 1
    _resign(payload)

    with pytest.raises(ValueError, match="canonical lock"):
        validate_pokeflex_regret_guard_prospective_protocol(payload)
