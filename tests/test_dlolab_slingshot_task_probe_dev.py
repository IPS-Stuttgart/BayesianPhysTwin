from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_task_probe_dev import (
    CANDIDATE_NAMES,
    conditional_prior,
    evaluate_candidates,
    frontloaded_controls,
    new_probe_task,
    protocol,
)


def _controls() -> np.ndarray:
    value = np.zeros((8, 3, 6), dtype=np.float64)
    value[:, 1, :3] = np.asarray([0.05, -0.05, 0.0])
    value[7] = value[5]
    return value


def test_frontloaded_controls_preserve_command_sum_and_limits() -> None:
    original = _controls()
    for fraction in (0.6, 0.7):
        result = frontloaded_controls(original, fraction)
        np.testing.assert_allclose(
            result[:, 0, :3] + result[:, 1, :3],
            original[:, 0, :3] + original[:, 1, :3],
            rtol=0,
            atol=1e-15,
        )
        assert np.max(np.linalg.norm(result[:, :, :3], axis=-1)) <= 0.1


def test_frontloaded_controls_reject_unregistered_fraction() -> None:
    with pytest.raises(ValueError, match="registered"):
        frontloaded_controls(_controls(), 0.65)


def test_tasks_cover_the_nine_world_slice_without_replacement() -> None:
    first = new_probe_task(0, 0)
    second = new_probe_task(0, 1)
    assert first["world_indices"] + second["world_indices"] == list(range(9, 18))
    assert len(second["world_indices"]) == 1


def test_conditional_prior_is_positive_and_normalized() -> None:
    prior = conditional_prior()
    assert prior.shape == (9,)
    assert np.all(prior > 0)
    assert prior.sum() == pytest.approx(1.0)


def test_task_value_can_prefer_a_new_probe() -> None:
    histories = np.zeros((len(CANDIDATE_NAMES), 9, 3, 4, 3), dtype=np.float64)
    # New probe 70 uniquely identifies each world along one coordinate.
    histories[3, :, 0, 0, 0] = np.arange(9) * 0.02
    rewards = np.zeros((9, 7), dtype=np.float64)
    rewards[:, 0] = 1.0
    for world in range(9):
        rewards[world, 1 + world % 6] = 3.0
    result = evaluate_candidates(
        histories,
        rewards,
        np.full(9, 1 / 9),
        draws=32,
        seed=7,
    )
    assert result["task_aware_probe_name"] == "frontload_70_new"
    assert result["generic_mi_probe_name"] == "frontload_70_new"
    assert result["value_feasibility_passed"] is True


def test_protocol_is_development_only_and_target_closed() -> None:
    value = protocol()
    assert value["future_protocol_automatically_authorized"] is False
    assert value["truth_future_generated"] is False
    assert value["protected_data_read"] is False
    assert value["retry_authorized"] is False
