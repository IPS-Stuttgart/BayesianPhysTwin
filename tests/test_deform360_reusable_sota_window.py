from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_reusable_sota_window import (
    authorize_development_fit_window,
    load_reusable_sota_window,
    reusable_sota_window_sha256,
    select_reusable_sota_action_window,
    validate_reusable_sota_window,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/causal4d_public/deform360_reusable_sota_v1.json"
ADDENDUM = ROOT / "configs/causal4d_public/deform360_reusable_sota_window_v1.json"


def _payload() -> dict:
    return json.loads(ADDENDUM.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> None:
    payload["config_sha256"] = reusable_sota_window_sha256(payload)


def test_canonical_reusable_sota_window_loads() -> None:
    payload = load_reusable_sota_window(ADDENDUM)
    result = validate_reusable_sota_window(payload)
    assert result["passed"] is True
    assert result["window_length_frames"] == 81
    assert result["processed_frame_count"] == 76


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("held_window_selection_may_read_object_geometry", True),
        ("held_window_selection_may_read_object_tracks", True),
        ("held_window_selection_may_read_tactile", True),
        ("held_future_outcomes_sealed_until_prediction_hash", False),
    ),
)
def test_window_addendum_rejects_held_leakage(field: str, value: bool) -> None:
    payload = copy.deepcopy(_payload())
    payload["config"]["information_boundary"][field] = value
    _rehash(payload)
    with pytest.raises(ValueError, match="information boundary"):
        validate_reusable_sota_window(payload)


def test_window_authorization_rejects_held_and_confirmatory_cases() -> None:
    parent = load_reusable_sota_config(PARENT)
    addendum = load_reusable_sota_window(ADDENDUM)
    accepted = authorize_development_fit_window(
        parent, addendum, object_id="004-rubber-band", episode_id=1
    )
    assert accepted["passed"] is True
    with pytest.raises(ValueError, match="fit episode"):
        authorize_development_fit_window(
            parent, addendum, object_id="004-rubber-band", episode_id=0
        )
    with pytest.raises(ValueError, match="development object"):
        authorize_development_fit_window(
            parent, addendum, object_id="068-nylon-rope", episode_id=1
        )


def test_action_window_uses_closed_motion_and_earliest_tie() -> None:
    addendum = load_reusable_sota_window(ADDENDUM)
    frame_count = 220
    actions = np.zeros((frame_count, 1, 5, 3), dtype=np.float64)
    openings = np.ones((frame_count, 1), dtype=np.float64)
    openings[20:101] = 0.0
    actions[20:101, 0, :, 0] = np.linspace(0.0, 0.08, 81)[:, None]
    # An identical later path must lose the declared earliest-start tie break.
    openings[122:203] = 0.0
    actions[122:203, 0, :, 0] = np.linspace(0.0, 0.08, 81)[:, None]
    selected = select_reusable_sota_action_window(actions, openings, addendum)
    assert selected["selected_raw_frame_range_half_open"] == [20, 101]
    assert selected["object_geometry_read"] is False
    assert selected["tactile_read"] is False
