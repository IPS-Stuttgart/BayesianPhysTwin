from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_active_bayes_v2 import (
    continuous_worlds,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_active_bayes_v2_runner",
    ROOT / "scripts/remote/run_dlolab_slingshot_active_bayes_v2.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_alternate_root_is_rejected_before_artifact_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "read_record",
        lambda path: pytest.fail("artifact read before root rejection"),
    )
    with pytest.raises(ValueError, match="registered active-Bayes root"):
        runner._validate(tmp_path)


def test_future_requires_recomputed_barrier_before_claim(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_validate",
        lambda output: ({"artifact_id": "lock"}, {}, {}, np.zeros((8, 3, 6))),
    )
    monkeypatch.setattr(
        runner,
        "_require_barrier",
        lambda *args: (_ for _ in ()).throw(ValueError("missing decision barrier")),
    )
    monkeypatch.setattr(
        runner,
        "run_registered_worlds",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="decision barrier"):
        runner._worker(tmp_path, "future", 0, None)
    assert list(tmp_path.iterdir()) == []


def test_future_loader_rederives_barrier_before_reading_future(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_require_barrier",
        lambda *args: (_ for _ in ()).throw(ValueError("missing decision barrier")),
    )
    monkeypatch.setattr(
        runner,
        "read_record",
        lambda path: pytest.fail("future artifact read before barrier"),
    )
    with pytest.raises(ValueError, match="decision barrier"):
        runner._load_future(tmp_path, {}, {}, np.zeros((8, 3, 6)), 0)


def test_barrier_is_rederived_not_trusted(tmp_path, monkeypatch) -> None:
    expected = {
        "schema": "dlolab-slingshot-active-bayes-decision-barrier-v2",
        "lock_id": "lock",
        "decision_seal_id": "decision",
        "pre_future": {"pre_future_gate_passed": True},
        "future_read": False,
        "future_generated": False,
    }
    recorded = {**expected, "future_generated": True}
    monkeypatch.setattr(runner, "read_record", lambda path: recorded)
    monkeypatch.setattr(runner, "_barrier_contents", lambda *args: expected)
    with pytest.raises(ValueError, match="barrier changed"):
        runner._require_barrier(tmp_path, {}, {}, np.zeros((8, 3, 6)))


def test_world_realization_is_exactly_rederived() -> None:
    worlds = continuous_worlds()[:8]
    native = {
        "world_realization": {
            "bending": [[row["bending_E"] for row in worlds]],
            "stretching": [[row["stretching_K"] for row in worlds]],
            "sphere_initial_position_m": [
                [0.12 + row["x_offset_m"], 0.06, 0.2] for row in worlds
            ],
            "cube_initial_position_m": [
                [0.12 + row["x_offset_m"], 0.23, 0.22] for row in worlds
            ],
        }
    }
    assert runner._world_realization_matches(native, worlds)
    native["world_realization"]["bending"][0][0] += 1
    assert not runner._world_realization_matches(native, worlds)
