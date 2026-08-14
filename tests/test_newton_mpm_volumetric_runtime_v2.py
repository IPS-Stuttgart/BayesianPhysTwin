from __future__ import annotations

import importlib
import sys
import types
from dataclasses import replace
from types import ModuleType
from typing import Any

import numpy as np
import pytest


def _runtime(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    warp = types.ModuleType("warp")
    warp.kernel = lambda function: function
    newton = types.ModuleType("newton")
    newton.ModelBuilder = object
    solvers = types.ModuleType("newton.solvers")
    solvers.SolverImplicitMPM = object
    newton.solvers = solvers
    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setitem(sys.modules, "newton", newton)
    monkeypatch.setitem(sys.modules, "newton.solvers", solvers)
    sys.modules.pop(
        "bayesian_phystwin._newton_mpm_volumetric_runtime_v2",
        None,
    )
    return importlib.import_module(
        "bayesian_phystwin._newton_mpm_volumetric_runtime_v2"
    )


def _cube() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 0.0, "fps"),
        ("voxel_size_m", np.nan, "voxel_size_m"),
        ("particle_spacing_m", -1.0, "particle_spacing_m"),
        ("density_kg_m3", 0.0, "density_kg_m3"),
        ("young_modulus_pa", np.inf, "young_modulus_pa"),
        ("tolerance", 0.0, "tolerance"),
        ("substeps", 0, "substeps"),
        ("substeps", True, "substeps"),
        ("maximum_particle_count", 3, "maximum_particle_count"),
        ("max_iterations", 0, "max_iterations"),
        ("query_neighbour_count", 0, "query_neighbour_count"),
        ("query_inverse_distance_power", 0.0, "query_inverse_distance_power"),
        ("poisson_ratio", 0.5, "poisson_ratio"),
        ("damping", -0.1, "damping"),
        ("contact_coupling_per_frame", 0.0, "contact_coupling_per_frame"),
        ("contact_coupling_per_frame", 1.1, "contact_coupling_per_frame"),
        ("gravity_m_s2", (0.0, np.nan, 0.0), "gravity_m_s2"),
    ],
)
def test_volumetric_runtime_config_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    runtime = _runtime(monkeypatch)
    config = replace(runtime.VolumetricMpmConfigV2(), **{field: value})
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_volumetric_runtime_orchestration_preserves_query_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)
    queries = _cube()
    controllers = np.zeros((3, 1, 3), dtype=np.float64)
    calls: list[tuple[bool, str]] = []

    def simulate(
        material: np.ndarray,
        controller_values: np.ndarray,
        contact_map: object,
        config: object,
        *,
        driven: bool,
        device: str,
    ) -> np.ndarray:
        del contact_map, config
        calls.append((driven, device))
        translation = (
            np.arange(len(controller_values))[:, None, None]
            * np.asarray([0.01, -0.02, 0.03])[None, None]
        )
        return np.asarray(material[None] + translation, dtype=np.float32)

    monkeypatch.setattr(runtime, "_simulate_material_v2", simulate)
    result = runtime.simulate_volumetric_mpm_v2(
        query_rest_points_m=queries,
        controller_points_m=controllers,
        attached_query_indices=np.asarray([0, 1], dtype=np.int64),
        query_controller_weights=np.ones((2, 1)),
        config=runtime.VolumetricMpmConfigV2(
            particle_spacing_m=0.5,
            voxel_size_m=1.0,
        ),
        driven=True,
        device="fake:0",
    )

    assert calls == [(True, "fake:0")]
    assert np.array_equal(result.query_trajectory_m[0], queries.astype(np.float32))
    assert np.allclose(
        result.query_trajectory_m[2],
        queries + np.asarray([0.02, -0.04, 0.06]),
    )
    assert result.material_trajectory_m.shape[0] == 3
    assert len(result.contact_map.material_indices) > 0


def test_volumetric_runtime_rejects_bad_invocations_and_nonfinite_solver_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)
    queries = _cube()
    controllers = np.zeros((2, 1, 3), dtype=np.float64)
    config = runtime.VolumetricMpmConfigV2(
        particle_spacing_m=0.5,
        voxel_size_m=1.0,
    )
    common = {
        "query_rest_points_m": queries,
        "controller_points_m": controllers,
        "attached_query_indices": np.asarray([0], dtype=np.int64),
        "query_controller_weights": np.ones((1, 1)),
        "device": "fake:0",
    }
    with pytest.raises(TypeError, match="VolumetricMpmConfigV2"):
        runtime.simulate_volumetric_mpm_v2(
            **common,
            config=object(),
            driven=True,
        )
    with pytest.raises(TypeError, match="driven"):
        runtime.simulate_volumetric_mpm_v2(
            **common,
            config=config,
            driven=1,
        )

    monkeypatch.setattr(
        runtime,
        "_simulate_material_v2",
        lambda material, controllers, *args, **kwargs: np.full(
            (len(controllers), len(material), 3),
            np.nan,
            dtype=np.float32,
        ),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        runtime.simulate_volumetric_mpm_v2(
            **common,
            config=config,
            driven=True,
        )


@pytest.mark.parametrize(
    ("queries", "controllers", "message"),
    [
        (np.zeros((3, 3)), np.zeros((2, 1, 3)), r"N>=4"),
        (np.full((4, 3), "bad"), np.zeros((2, 1, 3)), "finite numeric"),
        (_cube(), np.zeros((1, 1, 3)), r"T>=2"),
        (_cube(), np.zeros((2, 0, 3)), "controller points"),
        (_cube(), np.full((2, 1, 3), np.nan), "finite"),
    ],
)
def test_volumetric_runtime_validates_query_and_controller_shapes(
    monkeypatch: pytest.MonkeyPatch,
    queries: np.ndarray,
    controllers: np.ndarray,
    message: str,
) -> None:
    runtime = _runtime(monkeypatch)
    with pytest.raises(ValueError, match=message):
        runtime.simulate_volumetric_mpm_v2(
            query_rest_points_m=queries,
            controller_points_m=controllers,
            attached_query_indices=np.asarray([0], dtype=np.int64),
            query_controller_weights=np.ones((1, 1)),
            config=runtime.VolumetricMpmConfigV2(
                particle_spacing_m=0.5,
                voxel_size_m=1.0,
            ),
            driven=True,
            device="fake:0",
        )
