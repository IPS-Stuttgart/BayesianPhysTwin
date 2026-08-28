from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_value import (
    action_bank,
    decision_value,
    protocol,
    worlds,
)


def test_bank_preserves_exact_incumbent_duplicate_and_shared_prefix():
    control = np.arange(18, dtype=np.float64).reshape(1, 3, 6) / 200
    bank = action_bank(control)
    assert bank[0].tobytes() == bank[7].tobytes() == control[0].tobytes()
    assert all(row[:1].tobytes() == control[0, :1].tobytes() for row in bank)
    assert np.linalg.norm(bank[1:7, 1:, :3], axis=-1).max() <= 0.1
    assert len({row.tobytes() for row in bank}) == 7
    assert len(worlds()) == 9 and protocol()["native_evaluations"] == 72
    with pytest.raises(ValueError, match="control"):
        action_bank(control.astype(np.float32))


def _rows():
    rows = []
    for index, world in enumerate(worlds()):
        scores = np.full(8, 7.0)
        scores[1 + index % 3] = 7.1
        rows.append(
            {
                "world": world,
                "world_qa_passed": True,
                "metrics": [{"native_reward": score} for score in scores],
            }
        )
    return rows


def test_oracle_screen_requires_value_over_best_blind_not_only_zero():
    rows = _rows()
    result = decision_value(rows)
    assert result["source_decision_value_passed"]
    assert not result["bayesian_gain"]
    for row in rows:
        row["metrics"][4]["native_reward"] = 8.0
    assert not decision_value(rows)["source_decision_value_passed"]
    assert decision_value(rows)["best_world_blind_action"] == 4


def test_numeric_floor_incomplete_worlds_and_qa_cannot_be_ignored():
    rows = _rows()
    rows[2]["world_qa_passed"] = False
    assert not decision_value(rows)["source_decision_value_passed"]
    with pytest.raises(ValueError, match="nine"):
        decision_value(rows[:-1])
    swapped = deepcopy(rows)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="nine"):
        decision_value(swapped)
    rows = _rows()
    for row in rows:
        for metric in row["metrics"]:
            metric["native_reward"] = 7 + (metric["native_reward"] - 7) * 0.001
    result = decision_value(rows)
    assert result["numeric_margin_adjusted_oracle_gain"] < 0
    assert not result["source_decision_value_passed"]
