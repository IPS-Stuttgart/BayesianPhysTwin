"""Pure contracts, without importing or running the native task."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_benchmark import (
    RIGID_FIELDS,
    fixed_endpoint_error,
    memory_comparison,
    native_memory,
    protocol,
    slingshot_actions,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_native import STATE_FIELDS


def test_actions_preserve_official_three_stage_control_contract():
    actions = slingshot_actions()
    assert actions.shape == (2, 3, 6)
    assert actions.dtype == np.float64
    assert np.count_nonzero(actions[0]) == 0
    assert np.count_nonzero(actions[1]) == 3
    np.testing.assert_array_equal(actions[1, :, 1], [-0.04] * 3)
    assert np.max(np.linalg.norm(actions[:, :, :3], axis=-1)) < 0.1
    assert protocol()["automatic_method_evaluation_authorized"] is False


def test_exactly_two_native_solvers_with_all_memory_are_required():
    rigid = type("RigidSolverState", (), {})()
    rod = type("RODSolverState", (), {})()
    for value, fields in ((rigid, RIGID_FIELDS), (rod, STATE_FIELDS)):
        for name in fields:
            setattr(value, name, np.ones((1, 2)))
    arrays = native_memory(SimpleNamespace(solvers_state=[None, rigid, rod]))
    assert len(arrays) == 23
    assert memory_comparison(arrays, arrays)["byte_identical"] is True
    with pytest.raises(ValueError, match="exactly"):
        native_memory(SimpleNamespace(solvers_state=[rod]))
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["RODSolverState.pos"] += 0.001
    assert not memory_comparison(arrays, changed)["within_tolerance"]
    assert not memory_comparison(arrays, changed)["byte_identical"]
    changed["RODSolverState.pos"] = np.ones((1, 3))
    with pytest.raises(ValueError, match="layout"):
        memory_comparison(arrays, changed)


def test_native_arrays_are_copied_and_nonfinite_memory_is_rejected():
    rigid = type("RigidSolverState", (), {})()
    rod = type("RODSolverState", (), {})()
    for value, fields in ((rigid, RIGID_FIELDS), (rod, STATE_FIELDS)):
        for name in fields:
            setattr(value, name, np.ones((1, 2)))
    state = SimpleNamespace(solvers_state=[rigid, rod])
    arrays = native_memory(state)
    rod.pos[0, 0] = 2
    assert arrays["RODSolverState.pos"][0, 0] == 1
    rigid.qpos[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        native_memory(state)


def test_fixed_endpoint_check_does_not_mix_identity_or_world_axes():
    trace = np.zeros((3, 1, 12, 3))
    trace[:, :, :, 0] = np.arange(12) * 0.02
    changed = trace.copy()
    changed[:, :, 6] += 1
    assert fixed_endpoint_error([trace, changed]) == 0.0
    changed[-1, 0, 11, 2] = 0.001
    assert fixed_endpoint_error([trace, changed]) == 0.001
    with pytest.raises(ValueError):
        fixed_endpoint_error([trace[0]])


def test_boolean_native_memory_is_compared_without_arithmetic_subtraction():
    left = {"fixed": np.array([[True, False]]), "pos": np.zeros((1, 2, 3))}
    right = {k: v.copy() for k, v in left.items()}
    same = memory_comparison(left, right)
    assert same["byte_identical"] and same["within_tolerance"]
    assert same["maximum_absolute_difference"] == 0
    right["fixed"][0, 1] = True
    changed = memory_comparison(left, right)
    assert not changed["byte_identical"] and not changed["within_tolerance"]
    assert changed["maximum_absolute_difference"] == 1


def test_native_bundle_preserves_boolean_dtype_and_is_write_once(tmp_path):
    arrays = {"fixed": np.array([[True, False]]), "pos": np.zeros((1, 2, 3))}
    manifest = write_native_bundle(tmp_path, arrays)
    assert set(manifest["arrays"]) == set(arrays)
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as data:
        for key, value in arrays.items():
            assert data[key].dtype == value.dtype
            assert data[key].tobytes() == value.tobytes()
    with pytest.raises(FileExistsError):
        write_native_bundle(tmp_path, arrays)
    with pytest.raises(ValueError):
        write_native_bundle(tmp_path, {"unsafe": np.array([object()])})


def test_reporting_failure_keeps_all_completed_native_artifacts(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/remote/qualify_dlolab_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "native_benchmark_reporter_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent_protocol = protocol()
    parent_protocol.pop("reporter_repair_only")
    parent_protocol.pop("seal_each_rollout_before_analysis")
    parent_protocol["schema"] = "dlolab-native-slingshot-qualification-v1"
    packages = {
        p: "synthetic"
        for p in (
            "pin",
            "pin-pink",
            "qpsolvers",
            "proxsuite",
            "quadprog",
            "mushroom-rl",
            "omegaconf",
        )
    }
    parent = {
        "artifact_id": "f" * 64,
        "attempt_id": "a" * 64,
        "protected_data_read": False,
        "method_evaluation_authorized": False,
    }
    parent_attempt = {
        "artifact_id": "a" * 64,
        "protocol": parent_protocol,
        "native_source": {},
        "runtime": {"benchmark_packages": packages},
    }
    monkeypatch.setattr(module, "clean_revision", lambda _: "b" * 40)
    monkeypatch.setattr(module, "source_identity", lambda *_: {})
    monkeypatch.setattr(module, "runtime_identity", dict)
    monkeypatch.setattr(
        module,
        "file_digest",
        lambda _: "aecc4225c9e8c06998d4e339df28a53d32ca12ae26fde8a42f9bd34680819db3",
    )
    monkeypatch.setattr(
        module,
        "read_record",
        lambda p: parent if p.name == "failure.json" else parent_attempt,
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "synthetic")
    gs = SimpleNamespace(cpu="cpu", _initialized=False)
    gs.init = lambda **_: setattr(gs, "_initialized", True)
    gs.destroy = lambda: setattr(gs, "_initialized", False)
    actual_import = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: gs if name == "genesis" else actual_import(name),
    )

    class NativeEnvironment:
        def __init__(self, config):
            self.scene = SimpleNamespace(step=lambda: None, get_state=lambda: None)
            self.qpos_seq = np.zeros((1, 30, 9))

        def init_cmaes_env(self, **_):
            pass

        def eval_traj(self, _):
            for _ in range(900):
                self.scene.step()
            return {"cum_reward": np.zeros(1), "forward_time": 0.0}

    fake_package = ModuleType("envs")
    fake_env = ModuleType("envs.env_slingshot")
    fake_env.Train_Env_Slingshot = NativeEnvironment
    monkeypatch.setitem(sys.modules, "envs", fake_package)
    monkeypatch.setitem(sys.modules, "envs.env_slingshot", fake_env)
    monkeypatch.setattr(
        module,
        "observe",
        lambda _: {
            "rod_pos_m": np.zeros((1, 12, 3)),
            "sphere_pos_m": np.zeros((1, 3)),
            "cube_pos_m": np.zeros((1, 3)),
            "gripper_pos_m": np.zeros((1, 3)),
        },
    )
    monkeypatch.setattr(
        module, "native_memory", lambda _: {"fixed": np.array([True, False])}
    )

    def failing_report(*_):
        raise ValueError("injected reporting failure")

    monkeypatch.setattr(module, "memory_comparison", failing_report)
    output = tmp_path / "qualification"
    with pytest.raises(ValueError, match="injected reporting"):
        module.run(output, tmp_path / "assets", tmp_path / "parent/failure.json")
    from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

    failure = read_record(output / "failure.json")
    assert failure["terminal_stage"] == "replay-analysis"
    assert len(failure["completed_rollout_seals"]) == 3
    assert (
        read_record(output / "generation.json")["rollout_seals"]
        == failure["completed_rollout_seals"]
    )
    for index in range(3):
        seal = read_record(output / f"run-{index}/seal.json")
        assert seal["run_index"] == index
        with np.load(output / f"run-{index}/arrays.npz", allow_pickle=False) as data:
            assert data[f"run_{index}_memory_fixed"].dtype == np.bool_
    assert not gs._initialized
