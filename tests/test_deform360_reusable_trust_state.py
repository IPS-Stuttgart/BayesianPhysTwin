import json
from pathlib import Path

import pytest

from causal4d_public.deform360_reusable_trust_state import (
    load_reusable_trust_state_addendum,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/causal4d_public"
PARENT = CONFIG / "deform360_reusable_trust_fresh_v1.json"
PHYSICS = CONFIG / "deform360_reusable_trust_physics_addendum_v1.json"
EXECUTION = CONFIG / "deform360_reusable_trust_execution_v1.json"
MASK = CONFIG / "deform360_reusable_trust_mask_addendum_v5.json"
STATE = CONFIG / "deform360_reusable_trust_state_addendum_v1.json"


def test_state_addendum_locks_rigid_rest_preserving_source_policy() -> None:
    protocol = load_reusable_trust_state_addendum(
        PARENT,
        PHYSICS,
        EXECUTION,
        MASK,
        STATE,
    )

    assert protocol["state_addendum"]["state_policy"]["mode"] == (
        "rigid-rest-preserving"
    )
    assert protocol["state_addendum"]["source_gate"]["required_episode_ids"] == [
        1,
        3,
        4,
        6,
        7,
        9,
    ]


def test_state_addendum_rejects_outcome_informed_policy(tmp_path: Path) -> None:
    payload = json.loads(STATE.read_text(encoding="utf-8"))
    payload["lock_timing"]["source_future_object_outcomes_inspected"] = True
    changed = tmp_path / "state.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="outcome access"):
        load_reusable_trust_state_addendum(
            PARENT,
            PHYSICS,
            EXECUTION,
            MASK,
            changed,
        )


def test_state_addendum_rejects_rest_length_changes(tmp_path: Path) -> None:
    payload = json.loads(STATE.read_text(encoding="utf-8"))
    payload["state_policy"]["object_rest_lengths_changed"] = True
    changed = tmp_path / "state.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="state semantics"):
        load_reusable_trust_state_addendum(
            PARENT,
            PHYSICS,
            EXECUTION,
            MASK,
            changed,
        )
