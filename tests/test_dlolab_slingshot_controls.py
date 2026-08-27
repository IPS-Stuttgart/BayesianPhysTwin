"""Fixed task-control bank and native reward arithmetic, without simulation."""

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_controls import (
    action_bank,
    candidate_metrics,
    native_reward_from_trace,
    protocol,
    summarize,
)


def test_action_bank_is_bounded_deterministic_and_development_only():
    actions, names = action_bank()
    assert actions.shape == (24, 3, 6)
    assert len(set(names)) == len(actions)
    assert actions.tobytes() == action_bank()[0].tobytes()
    assert np.max(np.linalg.norm(actions[:, :, :3], axis=-1)) <= 0.1 + 1e-15
    assert np.max(np.abs(actions[:, :, 3:])) <= 1
    assert np.all(actions[:, :, 1] <= 0)
    assert not np.any(actions[0])
    assert not protocol()["method_evaluation_authorized"]
    assert not protocol()["adaptive_tuning"]


def test_native_reward_uses_postrelease_final_frame_and_float32_accumulation():
    cube = np.zeros((900, 1, 3))
    cube[:, 0, 1] = 0.23
    assert native_reward_from_trace(cube) == 6.900000095367432
    cube[699, 0, 1] = 4
    assert native_reward_from_trace(cube) == 6.900000095367432
    cube[899, 0, 1] = 0.25
    expected = np.float32(0)
    for _ in range(29):
        expected += np.float32(0.23)
    expected += np.float32(0.25)
    assert native_reward_from_trace(cube) == float(expected)
    with pytest.raises(ValueError):
        native_reward_from_trace(cube[:-1])


def _candidate(index):
    cube = np.zeros((900, 1, 3))
    cube[:, 0, 1] = 0.23
    sphere = np.zeros_like(cube)
    sphere[:, 0, 1] = 0.06
    if index == 1:
        cube[-1, 0, 1] += 0.05
        sphere[-1, 0, 1] += 0.05
    arrays = {
        "cube_pos_m": cube,
        "sphere_pos_m": sphere,
        "gripper_pos_m": np.zeros_like(cube),
        "controls": action_bank()[0][index][None],
    }
    return arrays


def test_candidate_reward_and_registered_action_cannot_be_changed():
    arrays = _candidate(1)
    reward = native_reward_from_trace(arrays["cube_pos_m"])
    row = candidate_metrics(arrays, 1, reward)
    assert row["cube_forward_progress_m"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="reward"):
        candidate_metrics(arrays, 1, reward + 0.001)
    arrays["controls"] = action_bank()[0][2][None]
    with pytest.raises(ValueError, match="commands"):
        candidate_metrics(arrays, 1, reward)


def test_full_denominator_and_failure_accounting_are_required():
    rows = [
        candidate_metrics(
            _candidate(i), i, native_reward_from_trace(_candidate(i)["cube_pos_m"])
        )
        for i in range(24)
    ]
    result = summarize(rows, [])
    assert result["task_competence_passed"]
    assert result["capable_candidate_count"] == 1
    assert result["best_capable_candidate"]["index"] == 1
    assert not result["bayesian_gain"] and not result["published_controller_parity"]
    with pytest.raises(ValueError, match="denominator"):
        summarize(rows[:-1], [])
    failed = summarize(rows[:-1], [23])
    assert not failed["task_competence_passed"]
    assert failed["retained_failure_count"] == 1
    with pytest.raises(ValueError, match="denominator"):
        summarize(rows, [23])


def test_direct_gripper_cube_proximity_cannot_qualify_a_candidate():
    rows = [
        candidate_metrics(
            _candidate(i), i, native_reward_from_trace(_candidate(i)["cube_pos_m"])
        )
        for i in range(24)
    ]
    rows[1]["minimum_gripper_cube_separation_m"] = 0.04
    assert not summarize(rows, [])["task_competence_passed"]
