from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bayesian_phystwin_experiments import dlolab_slingshot_grip as grip
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_path import (
    FORCES_N,
    RETAINED_GRIP_INDICES,
    VERTICAL_DETOURS_M,
    compare_previous_policy,
    controls,
    protocol,
    reference_checks,
    run_path_world,
    validate_force_record,
)

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "contact_path_runner", ROOT / "scripts/remote/run_dlolab_slingshot_path.py"
)
fixtures = load_module("grip_fixtures", ROOT / "tests/test_dlolab_slingshot_grip.py")


def test_paths_are_causal_bounded_deterministic_and_keep_strong_controls():
    source = fixtures.source_commands()
    before = source.tobytes()
    previous = grip.controls(source)
    candidate = controls(source)
    np.testing.assert_array_equal(candidate, controls(source))
    np.testing.assert_array_equal(candidate[:, 0], previous[:, 0])
    np.testing.assert_array_equal(
        candidate[:5, :, 3:], np.repeat(previous[2:3, :, 3:], 5, axis=0)
    )
    assert np.linalg.norm(candidate[:, :, :3], axis=-1).max() <= 0.1 + 1e-12
    for index, detour in enumerate(VERTICAL_DETOURS_M):
        assert candidate[index, 1, 2] == previous[2, 1, 2] + detour
        assert candidate[index, 2, 2] == previous[2, 2, 2] - detour
        for macro in (1, 2):
            before_xy, after_xy = previous[2, macro, :2], candidate[index, macro, :2]
            np.testing.assert_allclose(
                after_xy / before_xy, np.repeat(after_xy[0] / before_xy[0], 2)
            )
            assert np.linalg.norm(after_xy) <= np.linalg.norm(before_xy) + 1e-12
    for current, old in RETAINED_GRIP_INDICES.items():
        np.testing.assert_array_equal(candidate[current], previous[old])
    assert source.tobytes() == before and not np.shares_memory(source, candidate)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "prefix", "limit"])
def test_invalid_source_bank_is_not_silently_repaired(kind):
    source = fixtures.source_commands()
    if kind == "shape":
        source = source[:7]
    elif kind == "dtype":
        source = source.astype(np.float32)
    elif kind == "nan":
        source[6, 1, 0] = np.nan
    elif kind == "prefix":
        source[0, 0, 0] += 0.001
    else:
        source[6, 1, 2] = 0.2
    with pytest.raises(ValueError):
        controls(source)


def test_real_force_adapter_is_reused_with_restored_context(monkeypatch, tmp_path):
    original = grip.FORCES_N

    def fake_run(*args):
        assert grip.FORCES_N == FORCES_N
        record, _ = fixtures.execute_fake_schedule()
        return {}, {"grip_schedule": record}

    monkeypatch.setattr(grip, "run_grip_world", fake_run)
    _, report = run_path_world(
        tmp_path, tmp_path, controls(fixtures.source_commands()), 2
    )
    assert grip.FORCES_N == original
    record = report["grip_schedule"]
    validate_force_record(record)
    assert grip.FORCES_N == original
    for call in record["force_calls"]:
        expected = np.full((8, 2), -1.0 if call["native_step"] == 0 else -3.0)
        if call["native_step"] >= 300:
            expected = np.repeat(np.asarray(FORCES_N)[:, None], 2, axis=1)
        np.testing.assert_array_equal(call["command_N"], expected)
    record["force_calls"][-1]["solver_control_force_N"][0][0] = -12
    with pytest.raises(ValueError, match="force schedule"):
        validate_force_record(record)
    assert grip.FORCES_N == original


def test_force_context_restores_after_native_failure(monkeypatch, tmp_path):
    original = grip.FORCES_N

    def fail(*args):
        raise RuntimeError("mock native failure")

    monkeypatch.setattr(grip, "run_grip_world", fail)
    with pytest.raises(RuntimeError, match="mock"):
        run_path_world(tmp_path, tmp_path, controls(fixtures.source_commands()), 2)
    assert grip.FORCES_N == original


def references():
    source = fixtures.source_commands()
    contact = {
        "rod_pos_m": np.zeros((900, 8, 12, 3)),
        "sphere_pos_m": np.zeros((900, 8, 3)),
        "cube_pos_m": np.zeros((900, 8, 3)),
        "gripper_pos_m": np.zeros((900, 8, 3)),
        "controls": source,
    }
    previous = {k: v.copy() for k, v in contact.items()}
    previous["controls"] = grip.controls(source)
    candidate = {k: v.copy() for k, v in contact.items()}
    candidate["controls"] = controls(source)
    return candidate, contact, previous


def test_reference_check_keeps_all_strong_choices_and_prefix():
    candidate, contact, previous = references()
    result = reference_checks(
        candidate, contact, previous, [7.0] * 8, [7.0] * 8, [7.0] * 8
    )
    assert result["passed"] and result["retained_grip_error_m"] == 0


@pytest.mark.parametrize(
    "kind", ["position", "reward", "prefix", "fallback", "command"]
)
def test_replay_defect_prevents_admission(kind):
    candidate, contact, previous = references()
    rewards = [7.0] * 8
    if kind == "position":
        candidate["sphere_pos_m"][500, 4, 0] = 2e-6
    elif kind == "reward":
        rewards[6] = 7.1
    elif kind == "prefix":
        candidate["rod_pos_m"][299, 0, 0, 0] = 2e-6
    elif kind == "fallback":
        candidate["cube_pos_m"][500, 7, 0] = 2e-6
    else:
        candidate["controls"][4, 1, 0] += 0.001
        with pytest.raises(ValueError, match="retained strong"):
            reference_checks(
                candidate, contact, previous, rewards, [7.0] * 8, [7.0] * 8
            )
        return
    assert not reference_checks(
        candidate, contact, previous, rewards, [7.0] * 8, [7.0] * 8
    )["passed"]


def test_old_force_schedule_cannot_masquerade_as_new_one():
    record, _ = fixtures.execute_fake_schedule()
    with pytest.raises(ValueError, match="force schedule"):
        validate_force_record(record)
    with patch.object(grip, "FORCES_N", FORCES_N):
        record, _ = fixtures.execute_fake_schedule()
    validate_force_record(record)


def test_previous_policy_is_a_required_control_not_a_replaced_comparator():
    old = {
        "metrics": {
            "arms": {"bias_aware_posterior_mean": {"expected_native_reward": 7.0}}
        }
    }
    metric = {
        "arms": {"bias_aware_posterior_mean": {"expected_native_reward": 7.001}},
        "checks": {"old_gate": True},
    }
    result = compare_previous_policy(metric, old)
    assert not result["source_information_value_passed"]
    metric["arms"]["bias_aware_posterior_mean"]["expected_native_reward"] = 7.003
    assert compare_previous_policy(metric, old)["source_information_value_passed"]
    metric["checks"]["old_gate"] = False
    assert not compare_previous_policy(metric, old)["source_information_value_passed"]


def test_protocol_does_not_relax_inference_or_native_boundaries():
    value, original = protocol(), grip.protocol()
    for key in (
        "independent_noise_sd_m",
        "shared_bias_sd_m",
        "observation_frames",
        "observed_nodes",
        "integration_draws_per_world",
        "integration_seed",
        "minimum_information_gain",
        "minimum_relative_excess_information_gain",
        "minimum_gain_over_map",
        "require_not_worse_than_ignored_bias",
        "release_native_step",
    ):
        assert value[key] == original[key]
    assert value["retained_grip_action_pairs"] == [[4, 2], [5, 5], [6, 6], [7, 7]]
    assert value["final_cartesian_endpoint_not_claimed_identical"]
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


def test_alternate_root_or_forged_gate_cannot_reach_native(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="write-once"):
        runner.validate(tmp_path)
    monkeypatch.setattr(runner, "validate", lambda *args: ({}, {}, {}, {}, {}))
    monkeypatch.setattr(runner, "read_record", lambda *args: {"passed": True})
    monkeypatch.setattr(runner, "gate", lambda *args: {"passed": False})
    monkeypatch.setattr(
        runner, "run_path_world", lambda *args: pytest.fail("native ran")
    )
    with pytest.raises(ValueError, match="rederived"):
        runner.worker(tmp_path, 0)
    assert list(tmp_path.iterdir()) == []


def mock_source_driver(tmp_path, monkeypatch):
    output = tmp_path / "paths"
    old = {"controls": fixtures.source_commands().tolist()}
    refs = {i: ({"artifact_id": f"reference-{i}"}, {}) for i in range(3)}
    previous = {
        "metrics": {
            "arms": {"bias_aware_posterior_mean": {"expected_native_reward": 7.0}}
        }
    }
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "clean_revision", lambda *args: "test-head")
    monkeypatch.setattr(runner, "file_digest", lambda *args: "test-sha")
    monkeypatch.setattr(
        runner, "source", lambda: (old, {}, {"source_sha256": {}}, previous, refs)
    )
    launched = []
    monkeypatch.setattr(runner, "launch", lambda *args: launched.append(args[-1]))
    return output, launched


def test_native_gate_failure_is_retained_without_retry_or_later_worlds(
    tmp_path, monkeypatch
):
    output, launched = mock_source_driver(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "gate", lambda *args: {"passed": False})
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == [2] and result["native_worlds_completed"] == 1
    assert not result["source_gate_passed"]
    with pytest.raises(FileExistsError):
        runner.run(output)
    assert launched == [2]


def test_full_driver_uses_causal_prefix_and_all_three_worlds(tmp_path, monkeypatch):
    output, launched = mock_source_driver(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "gate", lambda *args: {"passed": True})

    def load(*args):
        index = args[-1]
        rod, sphere = (
            np.full((900, 8, 12, 3), index / 1000),
            np.full((900, 8, 3), index / 1000),
        )
        rod[300:] = sphere[300:] = 10000
        return (
            {"artifact_id": f"source-{index}"},
            {"rod_pos_m": rod, "sphere_pos_m": sphere},
            {"native_qa": {"metrics": [{"native_reward": 7.0}] * 8}},
        )

    def score(prefix, reward):
        assert prefix.shape == (3, 3, 4, 3) and prefix.max() == 0.002
        assert reward.shape == (3, 7)
        return {
            "arms": {"bias_aware_posterior_mean": {"expected_native_reward": 7.001}},
            "checks": {"information_gain_at_least_0_005": False},
        }

    monkeypatch.setattr(runner, "load_task", load)
    monkeypatch.setattr(runner, "information_value", score)
    runner.run(output)
    result = read_record(output / "result.json")
    assert launched == [2, 0, 1]
    assert result["native_worlds_completed"] == 3
    assert not result["source_gate_passed"]
    assert not result["method_evaluation_authorized"]


def test_worker_exception_retains_terminal_failure(tmp_path, monkeypatch):
    output, _ = mock_source_driver(tmp_path, monkeypatch)

    def fail(*args):
        raise RuntimeError("mock worker exit")

    monkeypatch.setattr(runner, "launch", fail)
    with pytest.raises(RuntimeError, match="mock worker"):
        runner.run(output)
    failure = read_record(output / "failure.json")
    assert not failure["retry_authorized"]
    assert failure["stage"] == "native-contact-path-worlds"
    assert not (output / "result.json").exists()
