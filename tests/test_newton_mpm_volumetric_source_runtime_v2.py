from __future__ import annotations

import importlib
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.cli import newton_mpm_backend as newton_cli
from bayesian_phystwin.newton_mpm_source_gate_v1 import (
    SourceProtocol,
    load_source_protocol,
)
from bayesian_phystwin.newton_mpm_volumetric_bridge_v2 import (
    MaterialContactMapV2,
    MaterialQueryMapV2,
)


def _import_runtime(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module_names = (
        "bayesian_phystwin._newton_mpm_source_runtime",
        "bayesian_phystwin._newton_mpm_volumetric_runtime_v2",
        "bayesian_phystwin._newton_mpm_volumetric_source_runtime_v2",
    )
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    newton = types.ModuleType("newton")
    newton.__version__ = "1.5.0"  # type: ignore[attr-defined]
    solvers = types.ModuleType("newton.solvers")

    class FakeSolverImplicitMpm:
        pass

    solvers.SolverImplicitMPM = FakeSolverImplicitMpm  # type: ignore[attr-defined]
    newton.solvers = solvers  # type: ignore[attr-defined]
    warp = types.ModuleType("warp")
    warp.__version__ = "1.16.0"  # type: ignore[attr-defined]
    warp.kernel = lambda function: function  # type: ignore[attr-defined]
    warp.get_device = lambda device: SimpleNamespace(  # type: ignore[attr-defined]
        alias=device,
        name="synthetic-device",
    )
    warp.ScopedDevice = lambda _device: nullcontext()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "newton", newton)
    monkeypatch.setitem(sys.modules, "newton.solvers", solvers)
    monkeypatch.setitem(sys.modules, "warp", warp)
    return importlib.import_module(module_names[-1])


def _protocol(*, expected_material_count: int = 8) -> SourceProtocol:
    density = 4 * 1000.0 * (2.0 * 0.003) ** 3 / (expected_material_count * 0.01**3)
    value: dict[str, Any] = {
        "protocol_id": "volumetric-source-test-v2",
        "geometry": {
            "frame_count": 5,
            "material_particle_count": 4,
            "observed_identity_count": 2,
        },
        "simulation": {
            "engine": "newton-implicit-mpm-volumetric-v2",
            "engine_version": "1.5.0",
            "warp_version": "1.16.0",
            "numpy_version": np.__version__,
            "scipy_version": "1.18.0",
            "fps": 30.0,
            "substeps": 4,
            "voxel_size_m": 0.02,
            "particle_spacing_m": 0.01,
            "maximum_particle_count": 100,
            "density_kg_m3": density,
            "poisson_ratio": 0.35,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "max_iterations": 50,
            "tolerance": 1.0e-5,
            "solver": "cr",
            "integration_scheme": "pic",
            "strain_basis": "P1d",
            "velocity_basis": "Q1",
            "grid_type": "sparse",
            "particleization": "regular-convex-hull-v2",
            "readout": "inverse-distance-material-displacement-v2",
            "contact": "finite-mass-compliant-projection-v2",
            "mass_normalization": "preserve-reference-direct-particle-total-mass-v2",
            "particle_radius_rule": "half-particle-spacing-v2",
            "reference_query_particle_radius_m": 0.003,
            "reference_query_density_kg_m3": 1000.0,
            "contact_coupling_per_frame": 0.35,
            "query_neighbour_count": 8,
            "query_inverse_distance_power": 2.0,
            "expected_internal_material_particle_count": expected_material_count,
            "expected_transferred_contact_particle_count": 2,
            "expected_query_map_maximum_distance_m": 0.0,
            "query_map_distance_tolerance_m": 1.0e-12,
        },
        "parameter_grid": [
            {"young_modulus_pa": 25_000.0, "damping": 0.002},
            {"young_modulus_pa": 100_000.0, "damping": 0.02},
        ],
        "implementation_source_paths": [
            "src/bayesian_phystwin/_newton_mpm_source_runtime.py",
            "src/bayesian_phystwin/_newton_mpm_volumetric_source_runtime_v2.py",
            "src/bayesian_phystwin/_newton_mpm_volumetric_runtime_v2.py",
            "src/bayesian_phystwin/newton_mpm_volumetric_bridge_v2.py",
            "src/bayesian_phystwin/newton_mpm_source_gate_v1.py",
            "src/bayesian_phystwin/cli/newton_mpm_backend.py",
        ],
    }
    return SourceProtocol(
        path=Path("protocol.json"),
        value=value,
        sha256="a" * 64,
    )


def _inputs() -> dict[str, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ],
        dtype=np.float32,
    )
    controllers = np.zeros((5, 2, 3), dtype=np.float32)
    controllers[:, :, 0] = np.arange(5, dtype=np.float32)[:, None] * 0.001
    return {
        "frame_zero_points_m": points,
        "controller_points_m": controllers,
        "attachment_indices": np.array([0, 1], dtype=np.int32),
        "attachment_weights": np.eye(2, dtype=np.float32),
        "action_support": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }


def _fake_rollout(
    runtime: types.ModuleType,
    *,
    config: object,
    driven: bool,
    material_count: int = 8,
) -> object:
    points = _inputs()["frame_zero_points_m"]
    material = np.zeros((material_count, 3), dtype=np.float32)
    material[:, 0] = np.arange(material_count, dtype=np.float32) * 0.001
    material_trajectory = np.repeat(material[None], 5, axis=0)
    query_trajectory = np.repeat(points[None], 5, axis=0)
    if driven:
        scale = np.float32(config.young_modulus_pa / 100_000_000.0)
        query_trajectory[:, :, 0] += scale * np.arange(5, dtype=np.float32)[:, None]
    query_map = MaterialQueryMapV2(
        indices=np.arange(4, dtype=np.int64)[:, None],
        weights=np.ones((4, 1), dtype=np.float64),
        maximum_distance_m=0.0,
    )
    contact_map = MaterialContactMapV2(
        material_indices=np.array([0, 1], dtype=np.int64),
        controller_weights=np.eye(2, dtype=np.float64),
    )
    return runtime.VolumetricMpmRolloutV2(
        material_rest_points_m=material,
        material_trajectory_m=material_trajectory,
        query_trajectory_m=query_trajectory,
        query_map=query_map,
        contact_map=contact_map,
    )


def test_volumetric_config_binds_the_new_hypothesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _import_runtime(monkeypatch)
    protocol = _protocol()

    config = runtime._volumetric_config(
        protocol,
        young_modulus_pa=25_000.0,
        damping=0.002,
    )

    assert config.contact_coupling_per_frame == 0.35
    assert config.particle_spacing_m == 0.01
    assert config.young_modulus_pa == 25_000.0
    changed = dict(protocol.value)
    changed["simulation"] = dict(protocol.value["simulation"], solver="cg")
    with pytest.raises(ValueError, match="simulation.solver"):
        runtime._volumetric_config(
            SourceProtocol(protocol.path, changed, protocol.sha256),
            young_modulus_pa=25_000.0,
            damping=0.002,
        )
    changed_density = dict(protocol.value)
    changed_density["simulation"] = dict(
        protocol.value["simulation"],
        density_kg_m3=1000.0,
    )
    with pytest.raises(ValueError, match="total-mass normalization"):
        runtime._volumetric_config(
            SourceProtocol(protocol.path, changed_density, protocol.sha256),
            young_modulus_pa=25_000.0,
            damping=0.002,
        )


def test_volumetric_source_grid_seals_query_readouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _import_runtime(monkeypatch)
    protocol = _protocol()
    inputs = _inputs()
    source_path = tmp_path / "source-inputs.npz"
    source_path.write_bytes(b"source-inputs")
    monkeypatch.setattr(runtime, "load_source_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        runtime,
        "load_source_inputs",
        lambda _path, *, protocol: inputs,
    )
    monkeypatch.setattr(
        runtime,
        "_implementation_provenance",
        lambda _protocol: {
            "git_head": "b" * 40,
            "git_worktree_clean": True,
            "source_files": {
                path: "c" * 64 for path in protocol.implementation_source_paths
            },
        },
    )

    def fake_simulate(**kwargs: object) -> object:
        return _fake_rollout(
            runtime,
            config=kwargs["config"],
            driven=bool(kwargs["driven"]),
        )

    monkeypatch.setattr(runtime, "simulate_volumetric_mpm_v2", fake_simulate)
    output = tmp_path / "grid"
    manifest = runtime.run_volumetric_source_grid(
        protocol_path=tmp_path / "protocol.json",
        source_inputs_path=source_path,
        output_dir=output,
        device="cuda:test",
    )

    assert manifest["successful_candidate_count"] == 2
    assert manifest["technical_failure_count"] == 0
    assert manifest["final_ensemble_spread_m"] > 0.0
    assert manifest["information_boundary"]["object_outcome_artifact_read"] is False
    with np.load(output / "candidate-00" / "physical-prediction.npz") as stored:
        assert stored["prediction_m"].shape == (5, 4, 3)
        np.testing.assert_allclose(
            stored["action_support"],
            np.ones(4, dtype=np.float32),
            atol=2.0e-7,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            stored["frame_zero_points_m"], inputs["frame_zero_points_m"]
        )
    assert (
        json.loads((output / "newton-grid.json").read_text())["grid_id"]
        == manifest["grid_id"]
    )


def test_particle_count_mismatch_is_a_retained_technical_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _import_runtime(monkeypatch)
    protocol = _protocol(expected_material_count=9)
    source_path = tmp_path / "source-inputs.npz"
    source_path.write_bytes(b"source-inputs")
    monkeypatch.setattr(runtime, "load_source_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        runtime,
        "load_source_inputs",
        lambda _path, *, protocol: _inputs(),
    )
    monkeypatch.setattr(
        runtime,
        "_implementation_provenance",
        lambda _protocol: {"git_head": "b" * 40},
    )
    monkeypatch.setattr(
        runtime,
        "simulate_volumetric_mpm_v2",
        lambda **kwargs: _fake_rollout(
            runtime,
            config=kwargs["config"],
            driven=bool(kwargs["driven"]),
            material_count=8,
        ),
    )

    manifest = runtime.run_volumetric_source_grid(
        protocol_path=tmp_path / "protocol.json",
        source_inputs_path=source_path,
        output_dir=tmp_path / "grid",
        device="cuda:test",
    )

    assert manifest["successful_candidate_count"] == 0
    assert manifest["technical_failure_count"] == 2
    assert all(
        record["error_message"] == "volumetric material particle count changed"
        for record in manifest["candidates"]
    )


def test_volumetric_action_support_rejects_no_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _import_runtime(monkeypatch)
    trajectory = np.zeros((3, 4, 3), dtype=np.float32)

    with pytest.raises(RuntimeError, match="no action response"):
        runtime._normalized_action_support(trajectory, trajectory.copy())


def test_volumetric_source_cli_delegates_to_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    module_name = "bayesian_phystwin._newton_mpm_volumetric_source_runtime_v2"
    runtime = types.ModuleType(module_name)

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "sealed"}

    runtime.run_volumetric_source_grid = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, runtime)
    args = SimpleNamespace(
        protocol=Path("protocol.json"),
        source_inputs=Path("inputs.npz"),
        output_dir=Path("output"),
        device="cuda:0",
    )

    assert newton_cli._run_volumetric_source_grid(args) == {"status": "sealed"}
    assert calls == [
        {
            "protocol_path": Path("protocol.json"),
            "source_inputs_path": Path("inputs.npz"),
            "output_dir": Path("output"),
            "device": "cuda:0",
        }
    ]


def test_volumetric_source_command_is_registered() -> None:
    args = newton_cli.build_parser().parse_args(
        [
            "source-run-volumetric-grid",
            "protocol.json",
            "source-inputs.npz",
            "grid",
            "--device",
            "cuda:1",
        ]
    )

    assert args.command == "source-run-volumetric-grid"
    assert args.device == "cuda:1"


def test_frozen_volumetric_source_protocol_is_valid() -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = load_source_protocol(
        repository
        / "configs/sota/newton_mpm_double_stretch_zebra_volumetric_source_v2.json"
    )
    simulation = protocol.value["simulation"]

    assert (
        protocol.protocol_id == "newton-mpm-double-stretch-zebra-volumetric-source-v2"
    )
    assert protocol.fit_range == (138, 167)
    assert protocol.validation_range == (167, 177)
    assert protocol.future_range == (177, 198)
    assert simulation["expected_internal_material_particle_count"] == 5620
    assert simulation["expected_transferred_contact_particle_count"] == 25
    assert protocol.value["external_components"]["prob4d"] == "unused"


def test_volumetric_source_cli_reports_missing_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin._newton_mpm_volumetric_source_runtime_v2",
        None,
    )
    args = SimpleNamespace(
        protocol=Path("protocol.json"),
        source_inputs=Path("inputs.npz"),
        output_dir=Path("output"),
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match=r"install bayesian-phystwin\[mpm\]"):
        newton_cli._run_volumetric_source_grid(args)
