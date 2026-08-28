from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_mechanism import (
    assess,
    change_projectile_coupling,
    protocol,
)


class Geom:
    def __init__(self, index, coupled=True):
        self.idx = index
        self._needs_coup = coupled

    @property
    def needs_coup(self):
        return self._needs_coup


def test_only_projectile_nonrigid_coupling_changes():
    geoms = [Geom(0), Geom(1), Geom(2, False)]
    env = SimpleNamespace(
        sphere=SimpleNamespace(geoms=[geoms[1]]),
        scene=SimpleNamespace(
            sim=SimpleNamespace(rigid_solver=SimpleNamespace(geoms=geoms))
        ),
    )
    assert change_projectile_coupling(env, False)["before"] == [True, True, False]
    record = change_projectile_coupling(env, True)
    assert record["after"] == [True, False, False]
    assert not record["rigid_collision_filters_changed"]
    with pytest.raises(ValueError, match="contract"):
        change_projectile_coupling(env, True)


def _row():
    row = {
        name: np.zeros((900, 1, 3))
        for name in ("sphere_pos_m", "cube_pos_m", "gripper_pos_m")
    }
    row["rod_pos_m"] = np.zeros((900, 1, 12, 3))
    row["cube_pos_m"][:, 0, 1] = 0.23
    row["cube_pos_m"][-1, 0, 1] = 0.28
    row["sphere_pos_m"][-1, 0, 1] = 0.1
    row["controls"] = np.zeros((1, 3, 6))
    return row


def test_contact_removal_must_remove_progress_without_changing_reference():
    row = _row()
    removed = deepcopy(row)
    removed["cube_pos_m"][-1, 0, 1] = 0.23
    removed["sphere_pos_m"][-1, 0, 1] = 0
    result = assess([deepcopy(row), deepcopy(row), removed], row)
    assert result["mechanism_audit_passed"]
    assert not result["parent_full_memory_gate_passed"]
    assert not result["state_restart_authorized"]
    assert not assess([row, row, row], row)["mechanism_audit_passed"]
    changed = deepcopy(row)
    changed["rod_pos_m"][400, 0, 2, 1] = 2e-6
    assert not assess([changed, row, removed], row)["mechanism_audit_passed"]
    with pytest.raises(ValueError, match="three"):
        assess([row, removed], row)


def test_protocol_does_not_reclassify_failed_hidden_state_replay():
    p = protocol()
    assert p["native_evaluations"] == 3
    assert p["full_memory_gate_from_parent"] == "retained_failed_not_relaxed"
    assert not p["uncertainty_method_evaluation_authorized"]


def test_no_native_progress_is_a_finite_failed_result():
    row = _row()
    row["sphere_pos_m"][:] = 0
    row["cube_pos_m"][:, 0, 1] = 0.23
    result = assess([row, row, row], row)
    assert not result["mechanism_audit_passed"]
    assert result["contact_disabled_progress_ratios"] == {
        "cube_progress_m": None,
        "sphere_progress_m": None,
    }
