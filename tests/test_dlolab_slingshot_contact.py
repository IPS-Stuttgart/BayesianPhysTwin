from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_contact import (
    COUPLINGS,
    POSITION_FIELDS,
    RUN_ORDER,
    contact_adapter,
    geometry_binding,
    information_value,
    nominal_replay,
    protocol,
    task,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_contact_runner", ROOT / "scripts/remote/run_dlolab_slingshot_contact.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def environment(coupling=0.9):
    values = np.asarray([0.02, coupling, coupling], dtype=np.float64)
    geoms = [
        SimpleNamespace(idx=i, coup_friction=float(v)) for i, v in enumerate(values)
    ]
    return SimpleNamespace(
        franka1=SimpleNamespace(geoms=geoms[1:]),
        scene=SimpleNamespace(
            sim=SimpleNamespace(
                rigid_solver=SimpleNamespace(
                    geoms=geoms,
                    geoms_info=SimpleNamespace(
                        coup_friction=SimpleNamespace(to_numpy=lambda: values.copy())
                    ),
                )
            )
        ),
    )


@pytest.mark.parametrize("coupling", COUPLINGS)
def test_adapter_changes_only_one_material_and_binds_before_action(coupling):
    class Material:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Environment:
        def init_cmaes_env(self, n_steps_sub=10):
            assert n_steps_sub == 10
            return "native-initialized"

    original_material = Material.__init__
    original_initialize = Environment.init_cmaes_env
    with contact_adapter(Material, Environment, coupling) as captured:
        sphere = Material(needs_coup=True, coup_friction=0.02, rho=200)
        franka = Material(needs_coup=True, coup_friction=0.9)
        assert sphere.values == {"needs_coup": True, "coup_friction": 0.02, "rho": 200}
        assert franka.values == {"needs_coup": True, "coup_friction": coupling}
        assert Environment.init_cmaes_env(environment(coupling)) == "native-initialized"
    assert Material.__init__ is original_material
    assert Environment.init_cmaes_env is original_initialize
    assert captured["modified_material_count"] == 1
    assert captured["geometry"]["robot_geometry_indices"] == [1, 2]
    assert captured["geometry"]["robot_coupling_values"] == [coupling, coupling]
    assert captured["geometry"]["nonrobot_geometry"] == [{"index": 0, "coupling": 0.02}]
    assert captured["geometry"]["verified_before_native_action"]


def test_adapter_restores_classes_on_changed_or_duplicate_constructor():
    class Material:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Environment:
        def init_cmaes_env(self):
            pass

    original = Material.__init__
    with pytest.raises(ValueError, match="constructor changed"):
        with contact_adapter(Material, Environment, 0.3):
            Material(needs_coup=True, coup_friction=0.9, rho=1)
    assert Material.__init__ is original
    with pytest.raises(ValueError, match="more than one"):
        with contact_adapter(Material, Environment, 0.3):
            Material(needs_coup=True, coup_friction=0.9)
            Material(needs_coup=True, coup_friction=0.9)
    assert Material.__init__ is original
    with pytest.raises(ValueError, match="incomplete"):
        with contact_adapter(Material, Environment, 0.3):
            pass
    assert Material.__init__ is original


def test_solver_mismatch_cannot_be_reported_as_realized_contact():
    with pytest.raises(ValueError, match="did not reach"):
        geometry_binding(environment(0.9), 0.3)
    value = environment(0.3)
    value.franka1.geoms = []
    with pytest.raises(ValueError, match="layout"):
        geometry_binding(value, 0.3)
    value = environment(0.3)
    value.scene.sim.rigid_solver.geoms_info.coup_friction.to_numpy = lambda: np.full(
        3, np.nan
    )
    with pytest.raises(ValueError, match="layout"):
        geometry_binding(value, 0.3)


@pytest.mark.parametrize("index", [-1, 3, True, 0.0])
def test_unregistered_contact_task_refused(index):
    with pytest.raises(ValueError, match="unregistered"):
        task(index)


def test_fixed_protocol_preserves_controls_and_source_boundary():
    value = protocol()
    assert RUN_ORDER == (2, 0, 1)
    assert value["source_world_count"] == 3
    assert value["observation_frames"] == [139, 219, 299]
    assert value["branch_frame"] == 299
    assert value["native_steps"] == 900
    assert value[
        "coefficient_is_native_tangential_coupling_not_measured_coulomb_friction"
    ]
    for index in range(3):
        spec = task(index)
        assert spec["franka_coupling"] == COUPLINGS[index]
        assert (
            spec["worlds"]
            == [{"x_offset_m": 0.0, "bending_E": 100000.0, "stretching_K": 800000.0}]
            * 8
        )
    assert not any(
        value[k]
        for k in (
            "new_recordings",
            "gpu_work",
            "protected_data_read",
            "retry_authorized",
            "calibration_or_evaluation_worlds_read",
            "method_evaluation_authorized",
        )
    )


def test_nominal_identity_requires_positions_commands_and_exact_rewards():
    reference = {name: np.zeros((900, 8, 3)) for name in POSITION_FIELDS}
    reference["controls"] = np.zeros((8, 3, 6))
    value = {k: v.copy() for k, v in reference.items()}
    assert nominal_replay(value, reference, [7.0] * 8, [7.0] * 8)["passed"]
    value["rod_pos_m"][500, 0, 0] = 2e-6
    assert not nominal_replay(value, reference, [7.0] * 8, [7.0] * 8)["passed"]
    value["rod_pos_m"][500, 0, 0] = 0
    assert not nominal_replay(value, reference, [7.0] * 8, [7.001] * 8)["passed"]
    value["controls"][0, 1, 0] = 0.01
    with pytest.raises(ValueError, match="matched nominal"):
        nominal_replay(value, reference, [7.0] * 8, [7.0] * 8)


def test_uninformative_prefix_adds_no_decision_value():
    prefix = np.zeros((3, 3, 4, 3))
    rewards = np.zeros((3, 7))
    rewards[0, 0], rewards[1, 1], rewards[2, 2] = 1, 2, 3
    result = information_value(prefix, rewards)
    assert result["best_blind_action"] == 2
    assert (
        abs(result["arms"]["bias_aware_posterior_mean"]["gain_over_best_blind"]) < 1e-12
    )
    assert not result["source_information_value_passed"]
    assert result["integration_only_not_independent_control_performance"]


def test_information_and_bayesian_value_positive_control():
    prefix = np.zeros((3, 3, 4, 3))
    prefix[2, 2, 3, 0] = 0.1
    rewards = np.full((3, 7), 6.9)
    rewards[0, 0] += 1
    rewards[:2, 1] += 0.8
    rewards[2, 2] += 1
    result = information_value(prefix, rewards)
    assert result["source_information_value_passed"]
    assert result["posterior_gain_over_map"] > 0.19
    assert result["arms"]["bias_aware_posterior_mean"]["gain_over_best_blind"] > 0.3
    for arm in result["arms"].values():
        assert abs(sum(arm["action_probability"]) - 1) < 1e-12
    shifted = information_value(prefix + 1, rewards)
    assert (
        shifted["source_information_value_passed"]
        == result["source_information_value_passed"]
    )
    np.testing.assert_allclose(
        shifted["mahalanobis_prefix_distances"],
        result["mahalanobis_prefix_distances"],
        atol=1e-10,
    )


@pytest.mark.parametrize("kind", ["prefix_shape", "reward_shape", "nonfinite"])
def test_incomplete_or_nonfinite_information_bank_refused(kind):
    prefix = np.zeros((3, 3, 4, 3))
    rewards = np.zeros((3, 7))
    if kind == "prefix_shape":
        prefix = prefix[:2]
    elif kind == "reward_shape":
        rewards = rewards[:, :6]
    else:
        prefix[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        information_value(prefix, rewards)


def test_wrong_root_rejected_before_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "read_record", lambda *args: pytest.fail("source accessed")
    )
    with pytest.raises(ValueError, match="write-once"):
        runner.validate(tmp_path)


def test_contact_worker_cannot_bypass_nominal_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate", lambda *args: ({}, {}, {}, []))
    monkeypatch.setattr(
        runner,
        "require_identity",
        lambda *args: (_ for _ in ()).throw(ValueError("identity gate")),
    )
    monkeypatch.setattr(
        runner, "run_contact_world", lambda *args: pytest.fail("native execution")
    )
    with pytest.raises(ValueError, match="identity gate"):
        runner.worker(tmp_path, 0)
    assert list(tmp_path.iterdir()) == []


def test_flipped_identity_boolean_does_not_authorize_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "read_record", lambda *args: {"contact_worlds_authorized": True}
    )
    monkeypatch.setattr(
        runner, "identity_result", lambda *args: {"contact_worlds_authorized": False}
    )
    with pytest.raises(ValueError, match="rederived"):
        runner.require_identity(tmp_path, {}, {}, {}, [])


def test_identity_failure_keeps_one_complete_attempt_without_retry(
    tmp_path, monkeypatch
):
    output = tmp_path / "contact"
    old = {"source_sha256": {}, "controls": np.zeros((8, 3, 6)).tolist()}
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "clean_revision", lambda *args: "test-commit")
    monkeypatch.setattr(runner, "file_digest", lambda *args: "test-sha")
    monkeypatch.setattr(runner, "source", lambda: (old, {}, []))
    launched = []
    monkeypatch.setattr(runner, "launch", lambda *args: launched.append(args[-1]))
    monkeypatch.setattr(
        runner, "identity_result", lambda *args: {"contact_worlds_authorized": False}
    )
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == [2]
    assert result["native_worlds_completed"] == 1
    assert result["source_gate_passed"] is False
    with pytest.raises(FileExistsError):
        runner.run(output)
    assert launched == [2]


def test_retained_failure_refuses_worker_before_native_initialization(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    write_record(tmp_path / "failure.json", {"stage": "native"})
    monkeypatch.setattr(
        runner, "run_contact_world", lambda *args: pytest.fail("native execution")
    )
    with pytest.raises(ValueError, match="terminal retained"):
        runner.worker(tmp_path, 2)


def test_complete_driver_preserves_denominator_and_prefix_only_likelihood(
    tmp_path, monkeypatch
):
    output = tmp_path / "contact"
    old = {"source_sha256": {}, "controls": np.zeros((8, 3, 6)).tolist()}
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "clean_revision", lambda *args: "test-commit")
    monkeypatch.setattr(runner, "file_digest", lambda *args: "test-sha")
    monkeypatch.setattr(runner, "source", lambda: (old, {}, []))
    monkeypatch.setattr(
        runner,
        "identity_result",
        lambda *args: {"contact_worlds_authorized": True},
    )
    launched = []
    monkeypatch.setattr(runner, "launch", lambda *args: launched.append(args[-1]))

    def load(output, lock, old, index):
        rod = np.full((900, 8, 12, 3), index / 1000)
        sphere = np.full((900, 8, 3), index / 1000)
        rod[300:] = sphere[300:] = 10000 + index
        data = {"rod_pos_m": rod, "sphere_pos_m": sphere}
        seal = {
            "artifact_id": f"case-{index}",
            "native": {"contact_realization": {"geometry": {"nonrobot_geometry": []}}},
        }
        qa = {
            "qa_passed": True,
            "metrics": [{"native_reward": 7 + index / 10}] * 8,
        }
        return seal, data, qa

    def score(history, rewards):
        assert history.shape == (3, 3, 4, 3)
        assert rewards.shape == (3, 7)
        np.testing.assert_array_equal(history[:, 0, 0, 0], np.arange(3) / 1000)
        assert history.max() == 0.002
        return {"source_information_value_passed": False}

    monkeypatch.setattr(runner, "load_task", load)
    monkeypatch.setattr(runner, "information_value", score)
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == [2, 0, 1]
    assert result["native_worlds_completed"] == 3
    assert not result["source_gate_passed"]
    assert not result["method_evaluation_authorized"]
    bank = read_record(output / "source-bank/seal.json")
    assert bank["source_seal_ids"] == [f"case-{i}" for i in range(3)]
