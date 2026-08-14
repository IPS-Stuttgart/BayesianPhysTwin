from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.cli.external_physics_backend import (
    _parameterization,
    _producer_artifacts,
)
from bayesian_phystwin.external_physics_backend_v1 import (
    build_external_physics_runtime_manifest,
    validate_external_physics_runtime_manifest,
)
from bayesian_phystwin.physics_backend_registry_v1 import BUILTIN_BACKEND_PROFILES


def _write_raw(path: Path) -> None:
    frame_zero = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    driven = np.repeat(frame_zero[None], 3, axis=0)
    zero = driven.copy()
    driven[:, 1, 2] += np.linspace(0.0, 0.02, 3)
    np.savez(
        path,
        driven_entity_positions_m=driven,
        zero_action_entity_positions_m=zero,
        query_entity_indices=np.array([0, 1], dtype=np.int64),
        action_support=np.array([0.0, 1.0], dtype=np.float64),
    )


def _runtime(raw: Path) -> dict[str, Any]:
    return build_external_physics_runtime_manifest(
        raw_rollout_path=raw,
        profile=BUILTIN_BACKEND_PROFILES[0],
        engine_revision="a" * 40,
        engine_version="test-engine-1",
        producer_repository="IPS-Stuttgart/BayesianPhysTwin",
        producer_revision="b" * 40,
        coordinate_frame="right-handed-z-up-world-v1",
        time_step_s=1.0 / 120.0,
        topology_sha256="c" * 64,
        material_model="neo-hookean",
        observation_end_frame_exclusive=1,
    )


def test_cli_helper_rejects_malformed_and_duplicate_artifacts() -> None:
    digest = "d" * 64
    assert _producer_artifacts([f"scene.json={digest}"]) == {
        "scene.json": digest
    }
    with pytest.raises(ValueError, match="PATH=SHA256"):
        _producer_artifacts(["scene.json"])
    with pytest.raises(ValueError, match="duplicate producer artifact"):
        _producer_artifacts(
            [f"scene.json={digest}", f"scene.json={digest}"]
        )
    assert _parameterization(None) == {}


def test_runtime_validator_covers_optional_raw_check_and_scalar_guards(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.npz"
    _write_raw(raw)
    runtime = _runtime(raw)
    assert validate_external_physics_runtime_manifest(runtime) == runtime

    cases: list[tuple[tuple[str, ...], object, str]] = [
        (("frame_count",), 0, "positive integer"),
        (
            ("information_boundary", "observation_end_frame_exclusive"),
            -1,
            "nonnegative integer",
        ),
        (("time_step_s",), True, "finite positive number"),
        (("time_step_s",), 0.0, "finite positive number"),
        (("backend_profile",), [], "JSON object"),
    ]
    for path, value, message in cases:
        candidate = json.loads(json.dumps(runtime))
        target: Any = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            validate_external_physics_runtime_manifest(candidate)
