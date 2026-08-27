"""Controller competence is distinct from optimizer gain or Bayesian evidence."""

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_benchmark import RIGID_FIELDS
from bayesian_phystwin_experiments.dlolab_native import STATE_FIELDS
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import (
    final_checks,
    protocol,
    task_metrics,
    worker_environment,
)


def _row():
    row = {
        name: np.zeros((900, 1, 3))
        for name in ("sphere_pos_m", "cube_pos_m", "gripper_pos_m")
    }
    row["rod_pos_m"] = np.zeros((900, 1, 12, 3))
    row["cube_pos_m"][:, 0, 1] = 0.23
    row["cube_pos_m"][-1, 0, 1] += 0.05
    row["sphere_pos_m"][-1, 0, 1] = 0.05
    row["controls"] = np.zeros((1, 3, 6))
    row.update({f"memory_RigidSolverState.{k}": np.zeros((1, 2)) for k in RIGID_FIELDS})
    row.update(
        {
            f"memory_RODSolverState.{k}": np.zeros(
                (1, 2), dtype=bool if k == "fixed" else float
            )
            for k in STATE_FIELDS
        }
    )
    return row


def test_optimizer_budget_is_fixed_and_not_a_bayesian_method():
    p = protocol()
    assert p["population"] * p["generations"] == p["evaluation_count"] == 64
    assert p["additional_selected_isolated_replay_count"] == 1
    assert (
        not p["method_evaluation_authorized"] and not p["published_controller_parity"]
    )


def test_task_progress_and_all_memory_must_survive_isolated_replay():
    row = _row()
    result = final_checks(row, deepcopy(row), 6.900000095367432)
    assert result["controller_competence_passed"]
    assert not result["bayesian_gain"]
    changed = deepcopy(row)
    changed["memory_RODSolverState.vel"][0, 0] = 1e-5
    assert not final_checks(row, changed, 6.9)["controller_competence_passed"]
    changed = deepcopy(row)
    changed["rod_pos_m"][400, 0, 3, 1] = 2e-6
    assert not final_checks(row, changed, 6.9)["checks"]["replay_positions"]


def test_good_replay_without_progress_is_not_task_competence():
    row = _row()
    row["cube_pos_m"][-1, 0, 1] = 0.231
    assert task_metrics(row)["cube_progress_m"] == pytest.approx(0.001)
    assert not final_checks(row, deepcopy(row), 6.9)["controller_competence_passed"]


def test_selected_replay_cannot_change_control_identity_or_drop_memory():
    row = _row()
    changed = deepcopy(row)
    changed["controls"][0, 0, 0] = 0.01
    with pytest.raises(ValueError, match="identity"):
        final_checks(row, changed, 6.9)
    row.pop("memory_RODSolverState.vel")
    with pytest.raises(ValueError, match="complete"):
        final_checks(row, deepcopy(row), 6.9)


def test_worker_uses_qualified_environment_after_upstream_import_mutation(monkeypatch):
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "LD_LIBRARY_PATH": "/qualified/osmesa",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    monkeypatch.setenv("LD_LIBRARY_PATH", "/qualified/osmesa:/upstream/ParticleMesher")
    monkeypatch.setenv("PYTHONPATH", "src")
    child = worker_environment({"environment": environment})
    assert all(child[key] == value for key, value in environment.items())
    assert child["PYTHONPATH"] == "src"
    import os

    assert os.environ["LD_LIBRARY_PATH"].endswith("/upstream/ParticleMesher")
    with pytest.raises(ValueError, match="complete"):
        worker_environment({"environment": {"LD_LIBRARY_PATH": "/qualified/osmesa"}})
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ValueError, match="CPU"):
        worker_environment({"environment": environment})
