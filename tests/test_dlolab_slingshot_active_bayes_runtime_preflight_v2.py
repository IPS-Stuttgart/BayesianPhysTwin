from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_active_bayes_runtime_preflight_v2",
    ROOT
    / "scripts/remote/run_dlolab_slingshot_active_bayes_runtime_preflight_v2.py",
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_protocol_is_prefix_only_and_preserves_v1() -> None:
    value = preflight.protocol()
    assert value["v1_retried"] is False
    assert value["v1_scientific_result"] is False
    assert value["prefix_steps"] == 300
    assert value["future_simulated"] is False
    assert value["reward_scored"] is False
    assert value["study_attempt_consumed"] is False
    assert value["retry_authorized"] is False


def test_world_realization_requires_exact_registered_values() -> None:
    world = preflight.particle_worlds()[0]
    native = {
        "world_realization": {
            "bending": [[world["bending_E"]] * 8],
            "stretching": [[world["stretching_K"]] * 8],
            "sphere_initial_position_m": [
                [0.12 + world["x_offset_m"], 0.06, 0.2]
            ]
            * 8,
            "cube_initial_position_m": [
                [0.12 + world["x_offset_m"], 0.23, 0.22]
            ]
            * 8,
        }
    }
    assert preflight._world_realization(native, world)
    native["world_realization"]["bending"][0][0] += 1
    assert not preflight._world_realization(native, world)


def test_alternate_root_rejected_before_source_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "_source",
        lambda: pytest.fail("source read before root rejection"),
    )
    with pytest.raises(ValueError, match="fresh registered runtime preflight"):
        preflight.run(tmp_path)
