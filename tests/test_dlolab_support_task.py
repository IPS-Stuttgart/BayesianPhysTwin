"""Source-task contracts; no empirical native or protected data access."""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_native import DloLabRuntime
from bayesian_phystwin_experiments.dlolab_support_task import (
    NativeSupportRuntime,
    SupportTaskConfig,
    action_commands,
    contact_clearance,
    qualification_protocol,
    segment_distance,
    sensitivity_gate,
    source_task_losses,
    source_worlds,
    support_positions,
    task_actions,
    task_goals,
)


def test_native_api_is_reused_without_rewriting_dynamics_or_state():
    for method in ("capture", "restore", "step", "rollout", "positions", "close"):
        assert getattr(NativeSupportRuntime, method) is getattr(DloLabRuntime, method)
    config = SupportTaskConfig()
    bending, x = source_worlds()
    assert bending.shape == x.shape == (6,)
    assert task_actions().shape == (12, 3)
    assert task_goals().shape == (9, 3)
    assert config.identity != dataclasses.replace(config, support_height_m=0.5).identity
    assert qualification_protocol()["method_comparison"] is False


def test_support_geometry_and_zero_hold_are_exact():
    config = SupportTaskConfig()
    support = support_positions(config, source_worlds()[1])
    np.testing.assert_allclose(support[:, :, 2], 0.48)
    np.testing.assert_allclose(np.linalg.norm(np.diff(support, axis=1), axis=-1), 0.06)
    initial = np.zeros((6, 25, 3))
    initial[:, :, 0] = np.arange(25) * 0.025
    initial[:, :, 2] = 0.6
    hold = action_commands(config, initial, 0)
    np.testing.assert_array_equal(hold, np.broadcast_to(initial[:, :2], hold.shape))
    motion = action_commands(config, initial, 11)
    np.testing.assert_array_equal(motion[0], initial[:, :2])
    np.testing.assert_allclose(motion[-1], initial[:, :2] + task_actions()[11])
    with pytest.raises(ValueError):
        support_positions(config, np.array([0.01]))


@pytest.mark.parametrize(
    "a0,a1,b0,b1,expected",
    [
        ([0, 0, 0], [1, 0, 0], [0.5, -1, 0], [0.5, 1, 0], 0.0),
        ([0, 0, 0], [1, 0, 0], [0.5, -1, 2], [0.5, 1, 2], 2.0),
        ([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], 1.0),
        ([0, 0, 0], [1, 0, 0], [0.5, 1, 0], [2, 1, 0], 1.0),
        ([0, 0, 0], [1, 0, 0], [2, 1, 0], [2, 2, 0], 2**0.5),
    ],
)
def test_segment_distance_matches_interior_boundary_and_parallel_cases(
    a0, a1, b0, b1, expected
):
    arrays = [np.asarray(x, dtype=float) for x in (a0, a1, b0, b1)]
    np.testing.assert_allclose(segment_distance(*arrays), expected, atol=1e-14)
    np.testing.assert_allclose(
        segment_distance(arrays[2], arrays[3], arrays[0], arrays[1]),
        expected,
        atol=1e-14,
    )


def test_contact_clearance_detects_native_capsule_surface():
    config = SupportTaskConfig()
    support = support_positions(config, np.array([0.20]))
    trajectory = np.zeros((2, 1, 25, 3))
    trajectory[:, :, :, 0] = np.arange(25) * 0.025
    trajectory[:, :, :, 2] = (
        config.support_height_m + config.support_radius_m + config.rod.segment_radius_m
    )
    np.testing.assert_allclose(
        contact_clearance(trajectory, support, config), 0.0, atol=1e-14
    )


def test_loss_alignment_and_tail_mean():
    config = SupportTaskConfig()
    futures = np.zeros((12, config.horizon_steps, 6, 25, 3))
    futures[:, -50:, :, -1] = task_goals()[0]
    futures[:, :-50] = 1000
    losses = source_task_losses(futures, config)
    np.testing.assert_allclose(losses[0, :, 0], 0)
    np.testing.assert_allclose(
        losses[0, :, 11], config.effort_weight * np.sum(task_actions()[11] ** 2)
    )
    with pytest.raises(ValueError, match="complete"):
        source_task_losses(futures[:11], config)


def test_sensitivity_requires_value_not_merely_different_action_labels():
    losses = np.ones((9, 6, 12))
    losses[:, :, 1] = 0.5
    assert not sensitivity_gate(losses)["decision_sensitivity_passed"]
    for world in range(6):
        losses[:3, world, 2 + world % 2] = 0.1
    result = sensitivity_gate(losses)
    assert result["passing_goals"] == 3
    assert result["decision_sensitivity_passed"]
    assert result["automatic_evaluation_authorization"] is False
    losses = np.ones((9, 6, 12))
    for world in range(6):
        losses[:, world, 2 + world % 2] -= 1e-9
    assert not sensitivity_gate(losses)["decision_sensitivity_passed"]
    with pytest.raises(ValueError):
        sensitivity_gate(losses[:8])


@pytest.mark.parametrize("checks", [{"contact": False}, {"contact": True}, {}])
def test_failed_native_contract_prevents_any_decision_array_read(
    tmp_path, monkeypatch, checks
):
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/remote/qualify_dlolab_support_decisions.py"
    )
    spec = importlib.util.spec_from_file_location("support_source_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = {"artifact_id": "a" * 64}
    monkeypatch.setattr(module, "validate_lock", lambda _: lock)
    monkeypatch.setattr(
        module,
        "read_record",
        lambda _: {
            "schema": "dlolab-support-native-seal-v1",
            "lock_id": lock["artifact_id"],
            "native_qualification_passed": True,
            "protected_data_read": False,
            "method_comparison": False,
            "method_evaluation_authorized": False,
            "checks": checks,
        },
    )
    reads = []
    monkeypatch.setattr(module, "load_bundle", lambda *_: reads.append(True))
    with pytest.raises(ValueError, match="contract failed"):
        module.analyze(tmp_path)
    assert reads == []
