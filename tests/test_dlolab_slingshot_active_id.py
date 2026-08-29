from __future__ import annotations

import json

import numpy as np

from bayesian_phystwin_experiments.dlolab_slingshot_active_id import (
    expected_value_screen,
    particle_task,
    protocol,
)


def test_particle_tasks_cover_only_the_two_new_slices() -> None:
    indices = []
    for group in range(2):
        for batch in range(2):
            indices.extend(particle_task(group, batch)["world_indices"])
    assert indices == list(range(9)) + list(range(18, 27))


def test_full_particle_screen_can_identify_active_value() -> None:
    histories = np.zeros((2, 27, 3, 4, 3), dtype=np.float64)
    histories[1, :, 0, 0, 0] = np.arange(27) * 0.02
    rewards = np.zeros((27, 7), dtype=np.float64)
    rewards[:, 0] = 1.0
    for world in range(27):
        rewards[world, 1 + world % 6] = 3.0
    result = expected_value_screen(histories, rewards, draws=16, seed=12)
    assert result["candidates"][1]["expected_bayes_reward"] > result["candidates"][0][
        "expected_bayes_reward"
    ]
    assert result["particle_value_gate_passed"] is True
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_protocol_stops_before_continuous_truth() -> None:
    value = protocol()
    assert value["continuous_truth_protocol_automatically_authorized"] is False
    assert value["truth_probe_generated"] is False
    assert value["truth_future_generated"] is False
    assert value["retry_authorized"] is False
