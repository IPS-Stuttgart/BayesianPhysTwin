from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_controls import (
    native_reward_from_trace,
)
from bayesian_phystwin_experiments.dlolab_slingshot_late import (
    controls,
    information_value,
    native_checks,
    observations,
    prior,
    protocol,
    repeat_checks,
    task,
)

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module("late_runner", ROOT / "scripts/remote/run_dlolab_slingshot_late.py")
batch_fixture = module("late_batch", ROOT / "tests/test_dlolab_slingshot_batch.py")
grip_fixture = module("late_grip", ROOT / "tests/test_dlolab_slingshot_grip.py")


def fixture(index=0):
    source = grip_fixture.source_commands()
    reference = batch_fixture._bank()
    reference["controls"] = source
    data = copy.deepcopy(reference)
    data["controls"] = controls(source)
    world = task(index)["world"]
    rewards = [
        native_reward_from_trace(data["cube_pos_m"][:, i : i + 1]) for i in range(8)
    ]
    native = {
        "native_cumulative_reward": rewards,
        "world_realization": {
            "bending": [[world["bending_E"]] * 8],
            "stretching": [[world["stretching_K"]] * 8],
            "sphere_initial_position_m": [[0.12, 0.06, 0.2]] * 8,
            "cube_initial_position_m": [[0.12, 0.23, 0.22]] * 8,
        },
    }
    return source, reference, data, native


def test_frozen_later_branch_and_complete_source_accounting():
    p = protocol()
    assert p["branch_native_step"] == 500 and p["observation_frames"] == [139, 319, 499]
    assert p["native_batches"] == 11 and p["native_trajectories"] == 88
    assert [task(i)["source_world_index"] for i in range(11)] == [
        4,
        4,
        4,
        0,
        1,
        2,
        3,
        5,
        6,
        7,
        8,
    ]
    assert prior().shape == (9,) and prior().sum() == 1 and prior()[4] == 0.25
    assert p["numeric_pair_margin"] == 0.0005
    assert p["minimum_adjusted_information_gain"] == 0.005
    assert not any(
        p[k]
        for k in (
            "gpu_work",
            "protected_data_read",
            "retry_authorized",
            "method_evaluation_authorized",
            "new_recordings",
            "earlier_failed_studies_reopened",
        )
    )


@pytest.mark.parametrize("index", [-1, 11, True, 3.0, "3"])
def test_unregistered_native_task_refused(index):
    with pytest.raises(ValueError, match="unregistered"):
        task(index)


def test_new_actions_only_change_last_macro_and_preserve_references():
    original = grip_fixture.source_commands()
    before = original.tobytes()
    value = controls(original)
    assert original.tobytes() == before
    np.testing.assert_array_equal(value[:, :2], np.repeat(original[5:6, :2], 8, axis=0))
    np.testing.assert_array_equal(value[0], original[5])
    np.testing.assert_array_equal(value[5], original[6])
    assert value[0].tobytes() == value[7].tobytes()
    assert len({row.tobytes() for row in value}) == 7
    assert np.max(np.linalg.norm(value[:, :, :3], axis=-1)) <= 0.1 + 1e-12
    assert np.max(np.abs(value[:, :, 3:])) <= 1
    np.testing.assert_array_equal(value[1, 2, :3], np.zeros(3))
    np.testing.assert_array_equal(value[1, 2, 3:], original[5, 2, 3:])


@pytest.mark.parametrize("change", ["shape", "dtype", "nonfinite", "prefix", "limit"])
def test_bad_controls_fail_before_native_import(change):
    value = grip_fixture.source_commands()
    if change == "shape":
        value = value[:7]
    elif change == "dtype":
        value = value.astype(np.float32)
    elif change == "nonfinite":
        value[0, 0, 0] = np.nan
    elif change == "prefix":
        value[6, 1, 0] += 0.01
    else:
        value[5, 2, 0] = 0.5
    with pytest.raises(ValueError):
        controls(value)


def test_observation_reader_has_hard_500_frame_and_identity_contract():
    rod = np.arange(500 * 8 * 12 * 3, dtype=float).reshape(500, 8, 12, 3)
    sphere = np.arange(500 * 8 * 3, dtype=float).reshape(500, 8, 3)
    result = observations({"rod_pos_m": rod, "sphere_pos_m": sphere})
    assert result.shape == (8, 3, 4, 3)
    np.testing.assert_array_equal(result[2, 1, :3], rod[319, 2, [3, 6, 8]])
    np.testing.assert_array_equal(result[4, 2, 3], sphere[499, 4])
    with pytest.raises(ValueError, match="500-frame"):
        observations({"rod_pos_m": rod[:499], "sphere_pos_m": sphere[:499]})
    with pytest.raises(ValueError, match="only registered"):
        observations({"rod_pos_m": rod, "sphere_pos_m": sphere, "cube_pos_m": sphere})


def check(data, native, source, reference):
    return native_checks(
        data,
        native,
        source,
        reference,
        [
            native_reward_from_trace(reference["cube_pos_m"][:, i : i + 1])
            for i in range(8)
        ],
        task(0)["world"],
    )


def test_native_arithmetic_and_frozen_reference_contract():
    source, reference, data, native = fixture()
    assert check(data, native, source, reference)["passed"]
    data["sphere_pos_m"][600, 5, 0] += 0.0005
    assert check(data, native, source, reference)["passed"]
    data["sphere_pos_m"][600, 5, 0] += 0.001
    assert not check(data, native, source, reference)["checks"][
        "retained_reference_positions"
    ]


@pytest.mark.parametrize(
    "change",
    [
        "prefix",
        "fixed",
        "duplicate",
        "reward",
        "material",
        "placement",
        "controls",
        "nan",
    ],
)
def test_native_corruption_is_not_admitted(change):
    source, reference, data, native = fixture()
    if change == "prefix":
        data["sphere_pos_m"][499, 3, 0] = 0.00001
    elif change == "fixed":
        data["rod_pos_m"][899, 4, 0, 0] = 1e-6
    elif change == "duplicate":
        data["sphere_pos_m"][800, 7, 0] = 0.002
    elif change == "reward":
        native["native_cumulative_reward"][1] += 0.01
    elif change == "material":
        native["world_realization"]["bending"][0][0] = 1
    elif change == "placement":
        native["world_realization"]["sphere_initial_position_m"] = [[0, 0, 0]] * 8
    elif change == "controls":
        data["controls"][0, 0, 0] = 0.5
    else:
        data["rod_vel_m_s"][100, 0, 2, 1] = np.nan
    if change in ("prefix", "fixed", "duplicate"):
        assert not check(data, native, source, reference)["passed"]
    else:
        with pytest.raises(ValueError):
            check(data, native, source, reference)


def test_new_context_repeats_include_every_candidate_not_only_fallback():
    _, _, data, native = fixture()
    rows = [copy.deepcopy(data) for _ in range(3)]
    rewards = np.asarray([native["native_cumulative_reward"]] * 3)
    assert repeat_checks(rows, rewards)["passed"]
    rewards[1, 3] += 0.0003
    assert not repeat_checks(rows, rewards)["passed"]
    rewards[1, 3] -= 0.0003
    rows[2]["sphere_pos_m"][899, 2, 0] = 0.002
    assert not repeat_checks(rows, rewards)["passed"]


def test_no_information_does_not_invent_value_and_all_primary_gates_apply():
    h = np.zeros((9, 3, 4, 3))
    rewards = np.full((9, 7), 7.0)
    rewards[:, 2] += 0.1
    result = information_value(h, rewards, rewards)
    assert result["best_fixed_action"] == 2
    assert result["arms"]["bias_aware_posterior_mean"][
        "expected_native_reward"
    ] == pytest.approx(7.1)
    assert result["adjusted_information_gain"] == pytest.approx(-0.0005)
    assert not result["source_information_value_passed"]
    assert result["integration_only_not_independent_control_performance"]


def test_informative_positive_control_does_not_claim_gain_over_equivalent_map():
    h = np.zeros((9, 3, 4, 3))
    h[:, 2, 0, 0] = np.arange(9) * 0.1
    reward = np.full((9, 7), 7.0)
    reward[np.arange(9), np.arange(9) % 7] += 0.5
    result = information_value(h, reward, reward)
    assert result["adjusted_information_gain"] > 0.1
    assert result["posterior_gain_over_map"] == pytest.approx(0)
    assert not result["source_information_value_passed"]
    for arm in result["arms"].values():
        assert sum(arm["action_probability"]) == pytest.approx(1)


def test_future_worker_cannot_bypass_a_failed_preceding_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "gate", lambda *args: {"qa": {"passed": False}})
    write_record(tmp_path / "admission-00.json", {"qa": {"passed": True}})
    with pytest.raises(ValueError, match="rederive"):
        runner.require_previous(tmp_path, {}, {}, [], 1)


def test_remaining_worlds_cannot_bypass_repeat_gate(tmp_path, monkeypatch):
    computed = {"qa": {"passed": True}}
    monkeypatch.setattr(runner, "gate", lambda *args: computed)
    for i in range(3):
        write_record(tmp_path / f"admission-{i:02d}.json", computed)
    monkeypatch.setattr(
        runner, "repeat_gate", lambda *args: {"numerics": {"passed": False}}
    )
    write_record(tmp_path / "repeat-gate.json", {"numerics": {"passed": True}})
    with pytest.raises(ValueError, match="numerical qualification"):
        runner.require_previous(tmp_path, {}, {}, [], 3)


@pytest.mark.parametrize("terminal", ["failure.json", "result.json"])
def test_terminal_output_and_other_roots_cannot_initialize(
    tmp_path, monkeypatch, terminal
):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    write_record(tmp_path / terminal, {"terminal": True})
    with pytest.raises(ValueError, match="terminal"):
        runner.validate(tmp_path)
    with pytest.raises(ValueError, match="registered"):
        runner.validate(tmp_path / "another")


@pytest.mark.parametrize(
    "failed_stage,expected_calls",
    [("none", 11), ("prefix", 0), ("native", 1), ("repeat", 3), ("runtime", 1)],
)
def test_orchestration_is_bounded_and_retains_every_failure(
    tmp_path, monkeypatch, failed_stage, expected_calls
):
    output = tmp_path / "new"
    source, reference, data, native = fixture()
    old = {"controls": source.tolist(), "source_sha256": {}}
    references = [
        (str(i), reference, native["native_cumulative_reward"]) for i in range(9)
    ]
    calls = []
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "clean_revision", lambda root: "a" * 40)
    monkeypatch.setattr(runner, "source", lambda: (old, references))
    monkeypatch.setattr(
        runner,
        "source_information",
        lambda refs: {"source_bank_authorized": failed_stage != "prefix"},
    )

    def launch(*args):
        calls.append(args[-1])
        if failed_stage == "runtime":
            raise RuntimeError("retained synthetic runtime failure")

    monkeypatch.setattr(runner, "launch", launch)

    def load(*args):
        index = args[-1]
        seal = {"artifact_id": str(index), "task": task(index), "native": native}
        return seal, data, {"passed": failed_stage != "native"}

    monkeypatch.setattr(runner, "load_task", load)
    monkeypatch.setattr(
        runner,
        "repeat_gate",
        lambda *args: {"numerics": {"passed": failed_stage != "repeat"}},
    )
    if failed_stage == "runtime":
        with pytest.raises(RuntimeError, match="retained"):
            runner.run(output)
        assert read_record(output / "failure.json")["worker_invocations_attempted"] == 1
    else:
        runner.run(output)
        result = read_record(output / "result.json")
        assert result["completed_batches"] == expected_calls
        assert result["unrun_batches"] == 11 - expected_calls
        assert not result["source_gate_passed"]
        assert (output / "source-bank").exists() == (failed_stage == "none")
    assert calls == list(range(expected_calls))
    with pytest.raises(FileExistsError):
        runner.run(output)
