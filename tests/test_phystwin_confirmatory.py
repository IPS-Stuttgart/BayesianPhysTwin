import json
from pathlib import Path

import pytest

from bayesian_phystwin.phystwin_confirmatory import (
    PhysTwinConfirmatoryProtocol,
    _lock_protocol,
    _split_for_case,
)


def test_split_uses_first_three_quarters_of_released_training(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "split.json").write_text(
        json.dumps({"frame_len": 85, "train": [0, 59], "test": [59, 85]})
    )

    assert _split_for_case(case, 0.75) == (44, 59, 85)


def test_protocol_lock_is_idempotent_and_rejects_changes(tmp_path: Path):
    specification = {"protocol": {"maximum_residual_m": 0.01}}

    first = _lock_protocol(tmp_path, specification)
    second = _lock_protocol(tmp_path, specification)

    assert first == second
    with pytest.raises(RuntimeError, match="different locked protocol"):
        _lock_protocol(tmp_path, {"protocol": {"maximum_residual_m": 0.03}})


def test_confirmatory_defaults_keep_development_cases_explicit():
    protocol = PhysTwinConfirmatoryProtocol()

    assert protocol.maximum_residual_m == 0.01
    assert protocol.development_cases == (
        "single_lift_sloth",
        "double_lift_sloth",
        "double_stretch_sloth",
    )
