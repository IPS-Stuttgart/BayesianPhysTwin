import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_independent_depth_regret_guard_calibration_protocol import (
    pokeflex_regret_guard_calibration_sha256,
    validate_pokeflex_regret_guard_calibration_protocol,
)


PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "pokeflex_independent_depth_regret_guard_calibration_v1.json"
)


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _resign(payload: dict[str, object]) -> None:
    payload["protocol_sha256"] = pokeflex_regret_guard_calibration_sha256(payload)


def test_calibration_protocol_matches_canonical_lock() -> None:
    result = validate_pokeflex_regret_guard_calibration_protocol(_payload())

    assert result["passed"] is True
    assert result["take_ids"] == (
        "3dPrintedPyramid_T2",
        "Beanbag_T2",
        "FoamCylinder_T2",
        "PlushMoon_T2",
    )


def test_calibration_protocol_rejects_refit() -> None:
    payload = copy.deepcopy(_payload())
    payload["frozen_deployment"]["no_refit_or_recalibration"] = False
    _resign(payload)

    with pytest.raises(ValueError, match="canonical lock"):
        validate_pokeflex_regret_guard_calibration_protocol(payload)


def test_calibration_protocol_rejects_target_opening() -> None:
    payload = copy.deepcopy(_payload())
    payload["calibration_cohort"]["target_objects_remain_sealed"] = False
    _resign(payload)

    with pytest.raises(ValueError, match="canonical lock"):
        validate_pokeflex_regret_guard_calibration_protocol(payload)
