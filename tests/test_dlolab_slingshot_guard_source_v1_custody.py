from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_belief import sample_worlds

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_guard_source_v1_runner",
    ROOT / "scripts/remote/run_dlolab_slingshot_guard_source_v1.py",
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
    with pytest.raises(ValueError, match="registered guard root"):
        runner._validate(tmp_path)


def test_parent_reward_loader_requires_recomputed_barrier_before_read(
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
        lambda path: pytest.fail("parent outcome read before barrier"),
    )
    with pytest.raises(ValueError, match="decision barrier"):
        runner._load_parent_rewards(
            tmp_path,
            {},
            {},
            np.zeros((8, 3, 6)),
            sample_worlds("calibration"),
        )


def test_barrier_is_rederived_not_trusted(tmp_path, monkeypatch) -> None:
    expected = {
        "schema": "dlolab-slingshot-guard-decision-barrier-v1",
        "lock_id": "lock",
        "decision_seal_id": "decision",
        "pre_outcome": {"pre_outcome_gate_passed": True},
        "source_outcome_read": False,
    }
    recorded = {**expected, "source_outcome_read": True}
    monkeypatch.setattr(runner, "read_record", lambda path: recorded)
    monkeypatch.setattr(runner, "_barrier_contents", lambda *args: expected)
    with pytest.raises(ValueError, match="barrier changed"):
        runner._require_barrier(
            tmp_path,
            {},
            {},
            np.zeros((8, 3, 6)),
            sample_worlds("calibration"),
        )


def test_prefix_task_pads_only_native_slots() -> None:
    first = runner.prefix_task(0)
    last = runner.prefix_task(2)
    assert first["world_indices"] == list(range(8))
    assert first["native_world_indices"] == list(range(8))
    assert last["world_indices"] == [16, 17, 18]
    assert last["native_world_indices"] == [16, 17, 18, 18, 18, 18, 18, 18]


def test_world_realization_is_exactly_rederived() -> None:
    worlds = sample_worlds("calibration")[:8]
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
    native["world_realization"]["stretching"][0][0] += 1
    assert not runner._world_realization_matches(native, worlds)
