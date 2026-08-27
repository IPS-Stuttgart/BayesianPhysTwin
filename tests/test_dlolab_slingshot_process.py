"""Pure process-isolation contracts; no native simulation."""

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_benchmark import (
    RIGID_FIELDS,
    slingshot_actions,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_benchmark import protocol as old_protocol
from bayesian_phystwin_experiments.dlolab_native import STATE_FIELDS
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    protocol,
    qualify,
    run_native,
)


def _arrays():
    row = {
        "rod_pos_m": np.zeros((900, 1, 12, 3)),
        "sphere_pos_m": np.zeros((900, 1, 3)),
        "cube_pos_m": np.zeros((900, 1, 3)),
        "gripper_pos_m": np.zeros((900, 1, 3)),
    }
    row["gripper_pos_m"][100:, :, 1] = 0.02
    row["rod_pos_m"][100:, :, 6, 1] = 0.02
    row.update({f"memory_RigidSolverState.{k}": np.ones((1, 2)) for k in RIGID_FIELDS})
    row.update(
        {
            f"memory_RODSolverState.{k}": np.ones(
                (1, 2), dtype=bool if k == "fixed" else float
            )
            for k in STATE_FIELDS
        }
    )
    rows = [deepcopy(row) for _ in range(3)]
    for index, item in enumerate(rows):
        item["controls"] = slingshot_actions()[[0, 1, 1][index]][None]
    return rows


def test_process_protocol_preserves_science_and_failure():
    prior = old_protocol()
    new = protocol()
    for key, value in prior.items():
        if key != "schema":
            assert new[key] == value
    assert new["reset_contract"] == "new_python_process_per_rollout"
    assert new["automatic_method_evaluation_authorized"] is False
    assert len(new["retained_parent_result_id"]) == 64


def test_native_bundle_verifies_every_boolean_and_float_byte(tmp_path):
    arrays = _arrays()[0]
    manifest = write_native_bundle(tmp_path, arrays)
    restored = load_native_bundle(tmp_path, manifest)
    for name in arrays:
        assert restored[name].dtype == arrays[name].dtype
        assert restored[name].tobytes() == arrays[name].tobytes()
    changed = deepcopy(manifest)
    changed["arrays"]["memory_RODSolverState.fixed"] = "0" * 64
    with pytest.raises(ValueError, match="changed native array"):
        load_native_bundle(tmp_path, changed)
    changed = deepcopy(manifest)
    changed["file"] = "../arrays.npz"
    with pytest.raises(ValueError, match="manifest"):
        load_native_bundle(tmp_path, changed)
    with (tmp_path / "arrays.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="bundle changed"):
        load_native_bundle(tmp_path, manifest)


def test_qualification_rederives_all_checks_from_complete_arrays():
    arrays = _arrays()
    result = qualify(arrays)
    assert result["native_qualification_passed"]
    assert result["memory_replay"]["byte_identical"]
    assert result["method_evaluation_authorized"] is False
    arrays[2]["memory_RODSolverState.vel"][:] = 1e-8
    assert not qualify(arrays)["native_qualification_passed"]
    arrays = _arrays()
    arrays[2]["rod_pos_m"][600, 0, 6, 1] += 2e-6
    assert not qualify(arrays)["checks"]["position_replay_within_1um"]
    arrays = _arrays()
    arrays[2]["rod_pos_m"][600, 0, 0, 1] += 1e-6
    assert not qualify(arrays)["checks"]["fixed_endpoints_unchanged"]
    with pytest.raises(ValueError, match="all three"):
        qualify(arrays[:2])


def test_no_missing_memory_or_mutated_action_can_qualify():
    arrays = _arrays()
    for row in arrays:
        row.pop("memory_RODSolverState.vel")
    with pytest.raises(ValueError, match="complete native memory"):
        qualify(arrays)
    arrays = _arrays()
    arrays[1]["controls"][0, 0, 0] = 0.001
    with pytest.raises(ValueError, match="action changed"):
        qualify(arrays)


def test_invalid_native_control_fails_before_native_import(tmp_path):
    with pytest.raises(ValueError, match="native control"):
        run_native(tmp_path, tmp_path, np.zeros((3, 6)))
