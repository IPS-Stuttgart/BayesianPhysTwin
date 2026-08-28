from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_grip import (
    CARTESIAN_INDICES,
    FORCE_FRAMES,
    FORCES_N,
    controls,
    force_at,
    grip_adapter,
    protocol,
    reference_checks,
    validate_force_record,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_grip_runner", ROOT / "scripts/remote/run_dlolab_slingshot_grip.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def source_commands():
    value = json.loads(
        (
            ROOT / "results/source/dlolab_slingshot_belief_control_v1/lock.json"
        ).read_text()
    )
    return np.asarray(value["controls"], dtype=np.float64)


class Robot:
    def __init__(self):
        self.force = np.zeros((8, 2))
        self.positions = []

    def get_dofs_force_range(self, fingers):
        assert fingers == [7, 8]
        return np.full(2, -30.0), np.full(2, 30.0)

    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None):
        assert dofs_idx_local == [7, 8] and envs_idx is None
        self.force = np.asarray(force).copy()

    def get_dofs_control_force(self, fingers):
        assert fingers == [7, 8]
        return self.force.copy()

    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None):
        self.positions.append((np.asarray(position).copy(), dofs_idx_local, envs_idx))


class Scene:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        return self.steps


class Environment:
    def __init__(self):
        self.scene = Scene()
        self.franka1 = Robot()
        self.c1 = SimpleNamespace(fingers_dof=[7, 8])

    def init_cmaes_env(self, n_steps_sub=10):
        assert n_steps_sub == 10
        return "initialized"


def execute_fake_schedule():
    with grip_adapter(Environment) as record:
        env = Environment()
        assert env.init_cmaes_env() == "initialized"
        for frame in range(900):
            if frame in FORCE_FRAMES:
                env.franka1.control_dofs_force(
                    np.full((8, 2), -1.0 if frame == 0 else -3.0), [7, 8]
                )
            if frame == 150:
                env.franka1.control_dofs_position(np.full((8, 7), 0.5), list(range(7)))
            if frame == 700:
                env.franka1.control_dofs_position(
                    np.full((8, 2), 0.08), [7, 8], list(range(8))
                )
            assert env.scene.step() == frame + 1
    return record, env


def test_complete_schedule_is_causal_bounded_native_and_restores_methods():
    original = Environment.init_cmaes_env
    record, env = execute_fake_schedule()
    assert Environment.init_cmaes_env is original
    assert env.scene.step.__func__ is Scene.step
    assert env.franka1.control_dofs_force.__func__ is Robot.control_dofs_force
    assert len(record["force_calls"]) == 31 and record["native_steps"] == 900
    for call in record["force_calls"]:
        frame = call["native_step"]
        np.testing.assert_array_equal(call["command_N"], force_at(frame))
        np.testing.assert_array_equal(call["solver_control_force_N"], force_at(frame))
        if frame < 300:
            assert np.unique(call["command_N"]).item() == (-1 if frame == 0 else -3)
    assert np.min(record["force_calls"][-1]["command_N"]) == -24
    assert len(record["release_calls"]) == 1
    assert len(env.franka1.positions) == 2
    validate_force_record(record)


@pytest.mark.parametrize(
    "mutation", ["early", "late", "missing", "clipped", "release", "limits"]
)
def test_force_record_rejects_causal_and_native_contract_violations(mutation):
    record, _ = execute_fake_schedule()
    record = copy.deepcopy(record)
    if mutation == "early":
        record["force_calls"][1]["command_N"] = force_at(300).tolist()
    elif mutation == "late":
        record["force_calls"][11]["native_step"] = 301
    elif mutation == "missing":
        record["force_calls"].pop()
    elif mutation == "clipped":
        record["force_calls"][-1]["solver_control_force_N"][0][0] = -2.0
    elif mutation == "release":
        record["release_calls"][0]["native_step"] = 701
    else:
        record["lower_force_limit_N"][0] = -24.0
    with pytest.raises(ValueError):
        validate_force_record(record)


def test_force_actuator_mismatch_stops_before_stepping_and_restores(monkeypatch):
    original = Environment.init_cmaes_env
    monkeypatch.setattr(Robot, "get_dofs_control_force", lambda *args: np.zeros((8, 2)))
    with pytest.raises(ValueError, match="clipped or changed"):
        with grip_adapter(Environment):
            env = Environment()
            env.init_cmaes_env()
            env.franka1.control_dofs_force(np.full((8, 2), -1.0), [7, 8])
    assert Environment.init_cmaes_env is original
    assert env.scene.steps == 0
    assert env.franka1.control_dofs_force.__func__ is Robot.control_dofs_force


@pytest.mark.parametrize("frame", [-1, 1, 99, 299, 700, True])
def test_unregistered_force_frame_refused(frame):
    with pytest.raises(ValueError, match="unregistered"):
        force_at(frame)


def test_cartesian_controls_are_exact_existing_actions_with_unchanged_prefix():
    original = source_commands()
    before = original.tobytes()
    result = controls(original)
    np.testing.assert_array_equal(result, original[list(CARTESIAN_INDICES)])
    np.testing.assert_array_equal(result[5], result[7])
    assert all(np.array_equal(row[0], original[5, 0]) for row in result)
    assert original.tobytes() == before
    assert FORCES_N[5] == FORCES_N[7] == -3
    for wrong in (
        original[:7],
        original.astype(np.float32),
        np.full_like(original, np.nan),
    ):
        with pytest.raises(ValueError):
            controls(wrong)


def test_reference_check_catches_fallback_and_prebranch_changes():
    source = source_commands()
    reference = {
        "rod_pos_m": np.zeros((900, 8, 12, 3)),
        "sphere_pos_m": np.zeros((900, 8, 3)),
        "cube_pos_m": np.zeros((900, 8, 3)),
        "gripper_pos_m": np.zeros((900, 8, 3)),
        "controls": source,
    }
    candidate = {k: v.copy() for k, v in reference.items()}
    candidate["controls"] = controls(source)
    assert reference_checks(candidate, reference, [7.0] * 8, [7.0] * 8)["passed"]
    candidate["rod_pos_m"][299, 0, 0, 0] = 2e-6
    assert not reference_checks(candidate, reference, [7.0] * 8, [7.0] * 8)["passed"]
    candidate["rod_pos_m"][299, 0, 0, 0] = 0
    candidate["cube_pos_m"][400, 5, 1] = 2e-6
    assert not reference_checks(candidate, reference, [7.0] * 8, [7.0] * 8)["passed"]


def test_protocol_has_new_action_not_new_physics_or_observation():
    value = protocol()
    assert value["only_new_control_is_post_prefix_grip_force"]
    assert value["force_branch_native_step"] == 300
    assert value["observation_frames"][-1] == 299
    assert value["release_native_step"] == 700
    assert value["source_world_count"] == 3
    assert value["source_noise_integration_reused_not_independent_confirmation"]
    assert not any(
        value[k]
        for k in (
            "gpu_work",
            "new_recordings",
            "protected_data_read",
            "method_evaluation_authorized",
            "retry_authorized",
        )
    )


def test_root_and_earlier_gate_fail_before_native_execution(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="write-once"):
        runner.validate(tmp_path)
    monkeypatch.setattr(runner, "validate", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(
        runner,
        "require_previous",
        lambda *args: (_ for _ in ()).throw(ValueError("earlier gate")),
    )
    monkeypatch.setattr(
        runner, "run_grip_world", lambda *args: pytest.fail("native execution")
    )
    with pytest.raises(ValueError, match="earlier gate"):
        runner.worker(tmp_path, 0)
    assert list(tmp_path.iterdir()) == []


def test_forged_gate_boolean_does_not_override_recomputation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "read_record", lambda *args: {"passed": True})
    monkeypatch.setattr(runner, "gate", lambda *args: {"passed": False})
    with pytest.raises(ValueError, match="rederived"):
        runner.require_previous(tmp_path, {}, {}, {}, 0)


def test_nominal_failure_retains_denominator_and_stops_without_retry(
    tmp_path, monkeypatch
):
    output = tmp_path / "grip"
    old = {"controls": source_commands().tolist()}
    references = {i: ({"artifact_id": f"reference-{i}"}, {}) for i in range(3)}
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "clean_revision", lambda *args: "test-head")
    monkeypatch.setattr(runner, "file_digest", lambda *args: "test-sha")
    monkeypatch.setattr(
        runner, "source", lambda: (old, {"source_sha256": {}}, references)
    )
    monkeypatch.setattr(runner, "gate", lambda *args: {"passed": False})
    launched = []
    monkeypatch.setattr(runner, "launch", lambda *args: launched.append(args[-1]))
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == [2]
    assert result["native_worlds_completed"] == 1
    assert result["source_gate_passed"] is False
    with pytest.raises(FileExistsError):
        runner.run(output)
    assert launched == [2]


def test_full_driver_uses_all_worlds_and_only_prefix_features(tmp_path, monkeypatch):
    output = tmp_path / "grip"
    old = {"controls": source_commands().tolist()}
    references = {i: ({"artifact_id": f"reference-{i}"}, {}) for i in range(3)}
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "clean_revision", lambda *args: "test-head")
    monkeypatch.setattr(runner, "file_digest", lambda *args: "test-sha")
    monkeypatch.setattr(
        runner, "source", lambda: (old, {"source_sha256": {}}, references)
    )
    monkeypatch.setattr(runner, "gate", lambda *args: {"passed": True})
    launched = []
    monkeypatch.setattr(runner, "launch", lambda *args: launched.append(args[-1]))

    def load(output, lock, old, references, index):
        rod = np.full((900, 8, 12, 3), index / 1000)
        sphere = np.full((900, 8, 3), index / 1000)
        rod[300:] = sphere[300:] = 10000
        return (
            {"artifact_id": f"source-{index}"},
            {"rod_pos_m": rod, "sphere_pos_m": sphere},
            {"native_qa": {"metrics": [{"native_reward": 7.0}] * 8}},
        )

    def score(prefix, reward):
        assert prefix.shape == (3, 3, 4, 3) and prefix.max() == 0.002
        assert reward.shape == (3, 7)
        return {"source_information_value_passed": False}

    monkeypatch.setattr(runner, "load_task", load)
    monkeypatch.setattr(runner, "information_value", score)
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == [2, 0, 1]
    assert result["native_worlds_completed"] == 3
    assert not result["source_gate_passed"]
    assert not result["method_evaluation_authorized"]
