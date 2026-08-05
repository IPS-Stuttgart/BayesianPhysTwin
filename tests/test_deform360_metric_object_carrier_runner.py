from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "remote"
    / "run_deform360_metric_object_carrier_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("_metric_object_carrier_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_npz_writer_is_byte_deterministic(tmp_path: Path) -> None:
    arrays = {
        "z": np.arange(12, dtype=np.float64).reshape(4, 3),
        "a": np.asarray([True, False, True]),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    MODULE._deterministic_npz(first, arrays)
    MODULE._deterministic_npz(second, dict(reversed(tuple(arrays.items()))))
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as stored:
        assert np.array_equal(stored["z"], arrays["z"])
        assert np.array_equal(stored["a"], arrays["a"])
