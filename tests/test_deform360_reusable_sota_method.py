from __future__ import annotations

import copy
from pathlib import Path

import pytest

from causal4d_public.deform360_reusable_sota_method import (
    load_reusable_sota_method,
    reusable_sota_physical_candidates,
    validate_reusable_sota_method,
)


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "configs/causal4d_public/deform360_reusable_sota_method_v1.json"


def test_reusable_sota_method_locks_the_eighteen_candidate_grid() -> None:
    method = load_reusable_sota_method(METHOD)
    candidates = reusable_sota_physical_candidates(method)
    assert len(candidates) == 18
    assert candidates[0]["label"] == "y10000-drag1-dash50"
    assert candidates[-1]["label"] == "y50000-drag10-dash100"
    assert method["config"]["prediction_bank"]["held_future_object_or_tactile_used"] is False


def test_reusable_sota_method_rejects_grid_or_boundary_changes() -> None:
    method = load_reusable_sota_method(METHOD)
    changed = copy.deepcopy(method)
    changed["config"]["physical_grid"]["init_spring_y"][0] = 9000.0
    with pytest.raises(ValueError, match="checksum"):
        validate_reusable_sota_method(changed)

    changed = copy.deepcopy(method)
    changed["config"]["prediction_bank"]["held_future_object_or_tactile_used"] = True
    with pytest.raises(ValueError, match="checksum"):
        validate_reusable_sota_method(changed)
