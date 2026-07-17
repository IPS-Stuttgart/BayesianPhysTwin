from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_replication_controls import (
    ContactTransitionModel,
    predict_causal_contact_transition,
)
from causal4d_public.deform360_reusable_contact_transition import (
    contact_transition_config_sha256,
    load_contact_transition_addendum,
    validate_contact_transition_addendum,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/causal4d_public/deform360_reusable_contact_transition_v1.json"
)


def _payload() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> None:
    payload["config_sha256"] = contact_transition_config_sha256(payload)


def test_canonical_contact_transition_addendum_loads() -> None:
    payload = load_contact_transition_addendum(CONFIG_PATH)
    result = validate_contact_transition_addendum(payload)
    assert result["passed"] is True
    assert result["development_held_episode_count"] == 48
    assert result["confirmatory_access_authorized"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("held_future_object_geometry_before_seal", True),
        ("held_future_tactile_before_seal", True),
        ("confirmatory_data_access_before_method_freeze", True),
    ),
)
def test_contact_transition_addendum_rejects_leakage(
    field: str, value: bool
) -> None:
    payload = copy.deepcopy(_payload())
    payload["config"]["information_boundary"][field] = value
    _rehash(payload)
    with pytest.raises(ValueError, match="information boundary"):
        validate_contact_transition_addendum(payload)


def test_contact_transition_addendum_rejects_parent_rewrite() -> None:
    payload = copy.deepcopy(_payload())
    payload["config"]["parent_protocol"]["modified"] = True
    _rehash(payload)
    with pytest.raises(ValueError, match="parent lock"):
        validate_contact_transition_addendum(payload)


def test_zero_shot_transition_infers_initial_state_from_onset_hazard() -> None:
    model = ContactTransitionModel(
        feature_names=(
            "gripper_openness_m",
            "gripper_to_predicted_object_proximity_m",
            "relative_closing_speed_m_s",
        ),
        feature_mean=(0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0),
        onset_coefficients=(4.0, 0.0, 0.0, 0.0),
        release_coefficients=(-10.0, 0.0, 0.0, 0.0),
        ridge_strength=1.0,
        transition_threshold=0.5,
    )
    openings = np.zeros((4, 1), dtype=np.float64)
    controllers = np.zeros((4, 1, 3), dtype=np.float64)
    objects = np.ones((4, 2, 3), dtype=np.float64)
    probability, state = predict_causal_contact_transition(
        model,
        openings,
        controllers,
        objects,
        dt_seconds=1.0 / 30.0,
        initial_contact_state=None,
    )
    assert probability[0, 0] > 0.95
    assert state[:, 0].tolist() == [True, True, True, True]


def test_explicit_prefix_state_preserves_legacy_frame_zero_behavior() -> None:
    model = ContactTransitionModel(
        feature_names=("a", "b", "c"),
        feature_mean=(0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0),
        onset_coefficients=(10.0, 0.0, 0.0, 0.0),
        release_coefficients=(-10.0, 0.0, 0.0, 0.0),
        ridge_strength=1.0,
        transition_threshold=0.5,
    )
    _, state = predict_causal_contact_transition(
        model,
        np.zeros((3, 1)),
        np.zeros((3, 1, 3)),
        np.ones((3, 1, 3)),
        dt_seconds=1.0,
        initial_contact_state=np.array([False]),
    )
    assert state[0, 0] == np.False_
    assert state[1, 0] == np.True_
