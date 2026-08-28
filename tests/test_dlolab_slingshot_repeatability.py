from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments import dlolab_slingshot_grip as grip
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_repeatability import (
    LAYOUTS,
    controls,
    forces,
    protocol,
    run_repeat,
    summarize,
    task,
    validate_force_record,
)

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


runner = module(
    "repeat_runner", ROOT / "scripts/remote/run_dlolab_slingshot_repeatability.py"
)
fixtures = module("grip_fixtures", ROOT / "tests/test_dlolab_slingshot_grip.py")


def test_exact_fifteen_task_roster_and_layout_multisets():
    tasks = [task(i) for i in range(15)]
    assert len({row["name"] for row in tasks}) == 15
    assert [row["contact_index"] for row in tasks] == [2] * 5 + [0] * 5 + [1] * 5
    for start in (0, 5, 10):
        assert [row["layout"] for row in tasks[start : start + 5]] == ["a"] * 3 + [
            "b"
        ] * 2
    assert sorted(LAYOUTS["a"]) == sorted(LAYOUTS["b"]) == [2, 2, 5, 5, 5, 5, 6, 6]
    p = protocol()
    assert p["native_batch_count"] == 15 and p["native_trajectory_count"] == 120
    assert (
        p["observed_reward_span_budget"] == p["minimum_scientific_reward_gain"] * 0.05
    )
    assert p["duplicate_and_cross_process_differences_are_outcomes_not_admission"]
    assert not any(
        p[k]
        for k in (
            "new_recovery_actions_run",
            "new_controller_evaluation_authorized",
            "gpu_work",
            "protected_data_read",
            "new_recordings",
            "robot_execution",
            "retry_authorized",
        )
    )


@pytest.mark.parametrize("index", [-1, 15, True, 2.0, "2"])
def test_unregistered_task_is_rejected(index):
    with pytest.raises(ValueError, match="unregistered"):
        task(index)


def test_exact_known_commands_and_forces_with_no_input_mutation():
    source = fixtures.source_commands()
    before = source.tobytes()
    previous = grip.controls(source)
    for index in range(15):
        value = controls(source, index)
        np.testing.assert_array_equal(
            value, previous[task(index)["grip_source_indices"]]
        )
        np.testing.assert_array_equal(value[5], value[7])
        assert np.linalg.norm(value[:, :, :3], axis=-1).max() <= 0.1 + 1e-12
        assert forces(index) == tuple(
            -3.0 if p == 5 else -24.0 for p in task(index)["grip_source_indices"]
        )
    assert source.tobytes() == before


@pytest.mark.parametrize("index", [0, 3, 5, 8, 10, 14])
def test_native_force_context_and_world_index_are_bound(index, tmp_path, monkeypatch):
    original = grip.FORCES_N

    def fake(*args):
        assert args[-1] == task(index)["contact_index"]
        assert grip.FORCES_N == forces(index)
        record, _ = fixtures.execute_fake_schedule()
        return {}, {"grip_schedule": record}

    monkeypatch.setattr(grip, "run_grip_world", fake)
    _, native = run_repeat(
        tmp_path, tmp_path, controls(fixtures.source_commands(), index), index
    )
    assert grip.FORCES_N == original
    validate_force_record(native["grip_schedule"], index)
    other_layout = 3 if task(index)["layout"] == "a" else 0
    with pytest.raises(ValueError, match="force schedule"):
        validate_force_record(native["grip_schedule"], other_layout)
    assert grip.FORCES_N == original


def records():
    output = []
    for index in range(15):
        spec = task(index)
        shift = (index % 5) * 1e-5 + (3e-5 if spec["layout"] == "b" else 0.0)
        arrays = {
            name: np.full(
                (900, 8, 12, 3) if name == "rod_pos_m" else (900, 8, 3), 0.2 + shift
            )
            for name in ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
        }
        reward = [
            ({2: 7.1, 6: 7.2, 5: 7.0}[p] + shift) for p in spec["grip_source_indices"]
        ]
        output.append(
            {
                "task": spec,
                "arrays": arrays,
                "reward": reward,
                "measurement_admitted": True,
            }
        )
    return output


def test_shared_numerical_error_cancels_in_paired_contrast_not_absolute_reward():
    result = summarize(records())
    assert result["observed_numerical_budget_passed"]
    assert result["maximum_reward_span"] == pytest.approx(7e-5)
    assert result["maximum_coordinate_span_m"] == pytest.approx(7e-5)
    assert result["maximum_paired_regret_span"] < 1e-12
    for world in result["worlds"]:
        assert world["batch_count"] == 5
        assert [p["trajectory_count"] for p in world["policies"]] == [10, 10, 20]
        for policy in world["policies"]:
            assert policy["within_batch_reward_span"] == 0
            assert policy["within_batch_coordinate_span_m"] == 0
            assert policy["same_layout_same_slot_process_reward_span"] == pytest.approx(
                2e-5
            )
            assert policy["layout_a_minus_b_mean_reward"] == pytest.approx(-5.5e-5)
        covariance = np.asarray(world["descriptive_reward_covariance"])
        assert covariance[0, 0] > 0
        np.testing.assert_allclose(
            covariance, np.full((3, 3), covariance[0, 0]), rtol=1e-9
        )
        np.testing.assert_allclose(
            world["descriptive_paired_regret_covariance"], 0, atol=1e-20
        )
    assert not result["population_repeatability_bound_established"]
    assert not result["controller_value_or_posterior_computed"]
    assert not result["earlier_failed_path_study_reopened"]
    assert not result["new_controller_evaluation_authorized"]


def test_reward_and_regret_budget_failures_are_preserved():
    values = records()
    for row in values:
        for slot, p in enumerate(row["task"]["grip_source_indices"]):
            if p == 2:
                row["reward"][slot] += (row["task"]["index"] % 5) * 0.0002
    result = summarize(values)
    assert not result["observed_numerical_budget_passed"]
    assert not result["checks"]["observed_reward_span_within_0_00025"]
    assert not result["checks"]["observed_regret_span_within_0_0005"]
    assert result["checks"]["observed_coordinate_span_within_1mm"]


def test_within_batch_position_difference_is_reported_not_dropped():
    values = records()
    values[0]["arrays"]["cube_pos_m"][800, 0, 1] += 0.002
    result = summarize(values)
    assert not result["checks"]["observed_coordinate_span_within_1mm"]
    world = result["worlds"][2]
    assert world["policies"][0]["within_batch_coordinate_span_m"] == pytest.approx(
        0.002
    )
    assert world["policies"][0]["trajectory_count"] == 10


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "order", "unadmitted", "nonfinite", "shape"]
)
def test_incomplete_or_invalid_denominator_is_not_scored(kind):
    values = records()
    if kind == "missing":
        values.pop()
    elif kind == "duplicate":
        values[-1] = copy.deepcopy(values[0])
    elif kind == "order":
        values.reverse()
    elif kind == "unadmitted":
        values[0]["measurement_admitted"] = False
    elif kind == "nonfinite":
        values[0]["reward"][0] = np.nan
    else:
        values[0]["arrays"]["rod_pos_m"] = values[0]["arrays"]["rod_pos_m"][:899]
    with pytest.raises(ValueError):
        summarize(values)


def test_alternate_root_and_forged_admission_stop_before_native(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="write-once"):
        runner.validate(tmp_path)
    monkeypatch.setattr(runner, "validate", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(
        runner, "read_record", lambda *args: {"measurement_admitted": True}
    )
    monkeypatch.setattr(runner, "gate", lambda *args: {"measurement_admitted": False})
    monkeypatch.setattr(
        runner, "run_repeat", lambda *args: pytest.fail("native execution")
    )
    with pytest.raises(ValueError, match="rederive"):
        runner.worker(tmp_path, 1)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fixed", [True, False])
def test_replay_difference_is_recorded_without_censoring_measurement(
    fixed, tmp_path, monkeypatch
):
    spec = task(0)
    realization = {"world_realization": {}, "contact_realization": {}}
    seal = {
        "lock_id": "lock",
        "claim_id": "claim",
        "task": spec,
        "native": {**realization, "grip_schedule": {}},
        "bundle": {},
    }
    claim = {"artifact_id": "claim", "lock_id": "lock", "task": spec}
    monkeypatch.setattr(
        runner, "read_record", lambda path: seal if path.name == "seal.json" else claim
    )
    monkeypatch.setattr(runner, "load_native_bundle", lambda *args: {})
    monkeypatch.setattr(runner, "validate_force_record", lambda *args: None)
    qa = {
        "checks": {
            "fixed_endpoints": fixed,
            "duplicate_positions": False,
            "duplicate_rewards": False,
        },
        "qa_passed": False,
    }
    monkeypatch.setattr(runner, "native_qa", lambda *args: qa)
    _, _, admission = runner.load_task(
        tmp_path,
        {"artifact_id": "lock"},
        {"controls": fixtures.source_commands().tolist()},
        {2: ({"native": realization}, {})},
        0,
    )
    assert admission["measurement_admitted"] is fixed
    assert admission["legacy_native_qa"] is qa
    assert admission["duplicate_checks_not_used_to_censor_replay_variation"]


def mock_driver(tmp_path, monkeypatch):
    output = tmp_path / "audit"
    old = {"controls": fixtures.source_commands().tolist()}
    refs = {i: ({"artifact_id": f"ref-{i}"}, {}) for i in range(3)}
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(
        runner,
        "source",
        lambda: (
            old,
            {"source_sha256": {}, "artifact_id": "grip-lock"},
            {"artifact_id": "grip-result"},
            refs,
        ),
    )
    monkeypatch.setattr(runner, "file_digest", lambda *args: "source-sha")
    monkeypatch.setattr(runner, "clean_revision", lambda *args: "test-head")
    launched = []
    monkeypatch.setattr(runner, "launch", lambda *args: launched.append(args[-1]))
    return output, launched


def test_physical_admission_failure_retains_planned_and_unrun_counts(
    tmp_path, monkeypatch
):
    output, launched = mock_driver(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "gate", lambda *args: {"measurement_admitted": False})
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == [0]
    assert result["planned_batches"] == 15 and result["completed_batches"] == 1
    assert result["admitted_batches"] == 0 and result["unrun_batches"] == 14
    with pytest.raises(FileExistsError):
        runner.run(output)
    assert launched == [0]


def test_complete_driver_never_authorizes_controller_or_reopens_path(
    tmp_path, monkeypatch
):
    output, launched = mock_driver(tmp_path, monkeypatch)
    values = records()
    monkeypatch.setattr(runner, "gate", lambda *args: {"measurement_admitted": True})

    def load(*args):
        index = args[-1]
        return (
            {
                "artifact_id": f"seal-{index}",
                "native": {"native_cumulative_reward": values[index]["reward"]},
            },
            values[index]["arrays"],
            {"measurement_admitted": True},
        )

    monkeypatch.setattr(runner, "load_task", load)
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == list(range(15))
    assert result["observed_numerical_budget_passed"]
    assert result["completed_batches"] == result["admitted_batches"] == 15
    assert result["unrun_batches"] == 0 and result["native_trajectories"] == 120
    assert not result["new_controller_evaluation_authorized"]
    assert not result["metrics"]["earlier_failed_path_study_reopened"]


def test_native_exception_is_retained_without_retry(tmp_path, monkeypatch):
    output, _ = mock_driver(tmp_path, monkeypatch)

    def fail(*args):
        raise RuntimeError("mock native exit")

    monkeypatch.setattr(runner, "launch", fail)
    with pytest.raises(RuntimeError, match="mock native"):
        runner.run(output)
    failure = read_record(output / "failure.json")
    assert (
        failure["completed_batches"] == 0
        and failure["worker_invocations_attempted"] == 1
    )
    assert failure["unrun_batches"] == 14 and not failure["retry_authorized"]
