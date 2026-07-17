from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
    reusable_sota_config_sha256,
    validate_reusable_sota_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/causal4d_public/deform360_reusable_sota_v1.json"


def _payload() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> None:
    payload["config_sha256"] = reusable_sota_config_sha256(payload)


def test_canonical_reusable_sota_protocol_loads() -> None:
    loaded = load_reusable_sota_config(CONFIG_PATH)
    result = validate_reusable_sota_config(loaded)
    assert result["passed"] is True
    assert result["development_object_count"] == 12
    assert result["confirmatory_object_count"] == 6
    assert result["confirmatory_held_episode_count"] == 24


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("confirmatory_future_object_frames_allowed_before_prediction_seal", True),
        ("confirmatory_future_particle_tracks_allowed_before_prediction_seal", True),
        ("confirmatory_future_point_clouds_allowed_before_prediction_seal", True),
        ("confirmatory_tactile_allowed_before_prediction_seal", True),
        ("held_predictions_must_be_checksummed_before_outcome_reveal", False),
    ),
)
def test_reusable_sota_protocol_rejects_information_leakage(
    field: str, value: bool
) -> None:
    payload = copy.deepcopy(_payload())
    payload["config"]["information_boundary"][field] = value
    _rehash(payload)
    with pytest.raises(ValueError, match="information boundary"):
        validate_reusable_sota_config(payload)


def test_reusable_sota_protocol_rejects_panel_substitution() -> None:
    payload = copy.deepcopy(_payload())
    payload["config"]["confirmatory_objects"]["3d"][0] = "091-posthoc-object"
    _rehash(payload)
    with pytest.raises(ValueError, match="confirmatory panel"):
        validate_reusable_sota_config(payload)


def test_reusable_sota_protocol_rejects_zero_shot_overclaim() -> None:
    payload = copy.deepcopy(_payload())
    payload["config"]["claim"]["zero_shot_multi_object_claim"] = True
    _rehash(payload)
    with pytest.raises(ValueError, match="claim boundary"):
        validate_reusable_sota_config(payload)
