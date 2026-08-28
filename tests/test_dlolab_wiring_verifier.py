import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_wiring_source import (
    information_value,
    native_reward,
    protocol,
)

SPEC = importlib.util.spec_from_file_location(
    "wiring_check",
    Path(__file__).resolve().parents[1] / "scripts/verify_dlolab_wiring_source.py",
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_second_reward_formula_matches_native_formula():
    rng = np.random.default_rng(40)
    points = rng.normal(0.1, 0.03, (4, 8, 30, 3))
    target = rng.normal(0.1, 0.04, (30, 3))
    assert np.allclose(
        checker.reward(points, target),
        native_reward(points, target),
        rtol=0,
        atol=1e-14,
    )


def test_sherman_morrison_matches_whitened_belief_value():
    rng = np.random.default_rng(44)
    prefix = rng.normal(0.1, 0.001, (9, 3, 5, 3))
    returns = rng.uniform(0.2, 0.9, (9, 7))
    second = checker.belief_value(prefix, returns, protocol())
    first = information_value(prefix, returns)
    assert checker.compare(second, first) < 1e-10


def test_changed_decision_boolean_is_rejected():
    with pytest.raises(ValueError, match="value differs"):
        checker.compare({"passed": True}, {"passed": False})


def test_changed_canonical_record_is_rejected(tmp_path):
    path = tmp_path / "record.json"
    path.write_text(
        '{"value": 2, "artifact_id": "' + checker.canonical({"value": 1}) + '"}'
    )
    with pytest.raises(ValueError, match="digest differs"):
        checker.read(path)


def test_array_identity_binds_dtype_and_order():
    from bayesian_phystwin_experiments.deform_state_restart import array_digest

    values = np.arange(90, dtype=np.float64).reshape(30, 3)
    assert checker.array_id(values) == array_digest(values)
    assert checker.array_id(values.astype(np.float32)) != checker.array_id(values)
