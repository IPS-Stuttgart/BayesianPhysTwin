import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_independent_depth_source_validation_protocol import (
    pokeflex_independent_depth_source_validation_sha256,
    validate_pokeflex_independent_depth_source_validation_protocol,
)


PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "pokeflex_independent_depth_source_validation_v2.json"
)


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _resign(payload: dict[str, object]) -> None:
    payload["protocol_sha256"] = pokeflex_independent_depth_source_validation_sha256(
        payload
    )


def test_source_validation_protocol_matches_canonical_lock() -> None:
    result = validate_pokeflex_independent_depth_source_validation_protocol(
        _payload()
    )

    assert result["passed"] is True
    assert result["source_validation_takes"] == ("T1", "T4", "T5", "T6")


def test_source_validation_protocol_rejects_radius_change() -> None:
    payload = copy.deepcopy(_payload())
    payload["method_lock"]["static_template_support_radius_mm"] = 20.0
    _resign(payload)

    with pytest.raises(ValueError, match="canonical lock"):
        validate_pokeflex_independent_depth_source_validation_protocol(payload)


def test_source_validation_protocol_rejects_t2_access_weakening() -> None:
    payload = copy.deepcopy(_payload())
    payload["source_validation"]["all_required_before_T2_access"] = False
    _resign(payload)

    with pytest.raises(ValueError, match="canonical lock"):
        validate_pokeflex_independent_depth_source_validation_protocol(payload)
