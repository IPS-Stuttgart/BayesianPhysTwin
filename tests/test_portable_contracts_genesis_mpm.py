from __future__ import annotations

import builtins
import importlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    load_physical_archive as load_deform360_physical_archive,
)
from bayesian_phystwin.genesis_mpm_backend_v1 import (
    ARTIFACT_FILENAME,
    GENESIS_MPM_BACKEND_KIND,
    GENESIS_MPM_ENGINE_REPOSITORY,
    GENESIS_MPM_RUNTIME_SCHEMA,
    PHYSICAL_ARCHIVE_FILENAME,
    file_sha256,
    load_genesis_particle_rollout,
    materialize_genesis_mpm_backend,
    validate_genesis_mpm_backend,
    validate_genesis_mpm_runtime_manifest,
)
from bayesian_phystwin.physical_rollout_v1 import (
    load_physical_rollout_archive,
    write_deterministic_npz,
)


def _raw_arrays(frame_count: int = 76) -> dict[str, np.ndarray]:
    frame_zero = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.15, 0.00, 0.00],
            [0.20, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    zero = np.repeat(frame_zero[None], frame_count, axis=0)
    driven = zero.copy()
    ramp = np.linspace(0.0, 0.02, frame_count, dtype=np.float32)
    driven[:, :, 2] += (
        ramp[:, None] * np.linspace(0.0, 1.0, len(frame_zero), dtype=np.float32)[None]
    )
    driven[0] = frame_zero
    return {
        "driven_particle_positions_m": driven,
        "zero_action_particle_positions_m": zero,
        "material_query_indices": np.array([0, 4, 2], dtype=np.int64),
        "action_support": np.array([0.0, 1.0, 0.5], dtype=np.float32),
    }


def _raw_archive(path: Path, arrays: dict[str, np.ndarray] | None = None) -> Path:
    write_deterministic_npz(path, arrays or _raw_arrays())
    return path


def _runtime_manifest(path: Path, raw_path: Path) -> Path:
    raw = _raw_arrays()
    driven = raw["driven_particle_positions_m"]
    zero = raw["zero_action_particle_positions_m"]
    maximum_response = float(np.max(np.linalg.norm(driven - zero, axis=2)))
    maximum_step = float(np.max(np.linalg.norm(np.diff(driven, axis=0), axis=2)))
    identity = {
        "schema": GENESIS_MPM_RUNTIME_SCHEMA,
        "schema_version": 1,
        "backend_kind": GENESIS_MPM_BACKEND_KIND,
        "engine_repository": GENESIS_MPM_ENGINE_REPOSITORY,
        "engine_version": "1.2.2",
        "torch_version": "2.4.0+cu121",
        "python_version": "3.12.3",
        "device": "gpu",
        "device_name": "synthetic-test-device",
        "coordinate_frame": "right-handed-z-up-world-v1",
        "position_units": "m",
        "time_units": "s",
        "frame_count": raw["driven_particle_positions_m"].shape[0],
        "particle_count": raw["driven_particle_positions_m"].shape[1],
        "query_count": len(raw["material_query_indices"]),
        "time_step_s": 1.0 / 120.0,
        "simulation": {
            "scene": "compliant-gripper-beam-bend-v1",
            "beam_extents_m": [0.3, 0.05, 0.05],
            "action_displacement_m": [0.0, 0.0, 0.025],
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "density_kg_m3": 1000.0,
            "young_modulus_pa": 50000.0,
            "poisson_ratio": 0.3,
            "elastic_model": "corotation",
            "grid_density": 64,
            "substeps": 8,
            "attachment_stiffness": 500.0,
            "solver": "genesis-mpm",
        },
        "diagnostics": {
            "maximum_action_response_m": maximum_response,
            "maximum_particle_step_m": maximum_step,
            "response_to_action_ratio": maximum_response / 0.025,
            "stability_cap_ratio": 3.0,
            "stability_gate_passed": True,
        },
        "implementation": {
            "repository": "IPS-Stuttgart/BayesianPhysTwin",
            "revision": "1" * 40,
            "source_files_sha256": {
                "src/bayesian_phystwin/_genesis_mpm_runtime.py": "2" * 64,
                "src/bayesian_phystwin/genesis_mpm_backend_v1.py": "3" * 64,
                "src/bayesian_phystwin/physical_rollout_v1.py": "4" * 64,
            },
        },
        "information_boundary": {
            "synthetic_scene": True,
            "dataset_payload_read": False,
            "future_observations_read": False,
            "outcomes_read": False,
            "known_action_used": True,
        },
        "raw_rollout_sha256": file_sha256(raw_path),
    }
    runtime = {**identity, "runtime_id": content_id(identity)}
    write_atomic_json(runtime, path, overwrite=False)
    return path


def test_genesis_materialization_preserves_identity_and_consumer_contract(
    tmp_path: Path,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    output = tmp_path / "output"

    artifact = materialize_genesis_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )

    assert artifact == validate_genesis_mpm_backend(output)
    assert artifact["mapping"]["material_identity_preserved"] is True
    physical = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    expected = _raw_arrays()["driven_particle_positions_m"][:, [0, 4, 2]]
    np.testing.assert_array_equal(physical["prediction_m"], expected)
    np.testing.assert_array_equal(physical["driven_readout_m"], expected)
    np.testing.assert_array_equal(
        physical["persistence_m"], np.repeat(expected[0][None], 76, axis=0)
    )
    existing = load_deform360_physical_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    np.testing.assert_array_equal(existing["prediction_m"], expected)


def test_genesis_materialization_is_byte_deterministic(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    first = materialize_genesis_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=tmp_path / "first",
    )
    second = materialize_genesis_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=tmp_path / "second",
    )
    assert first["artifact_id"] == second["artifact_id"]
    assert (tmp_path / "first" / PHYSICAL_ARCHIVE_FILENAME).read_bytes() == (
        tmp_path / "second" / PHYSICAL_ARCHIVE_FILENAME
    ).read_bytes()
    assert (tmp_path / "first" / ARTIFACT_FILENAME).read_bytes() == (
        tmp_path / "second" / ARTIFACT_FILENAME
    ).read_bytes()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda arrays: arrays.__setitem__(
                "material_query_indices", np.array([1, 1], dtype=np.int64)
            ),
            "must be unique",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "material_query_indices", np.array([1, 9], dtype=np.int64)
            ),
            "exceeds particle count",
        ),
        (
            lambda arrays: arrays["zero_action_particle_positions_m"].__setitem__(
                (0, 0, 0), 0.001
            ),
            "differ at frame zero",
        ),
    ],
)
def test_genesis_raw_rollout_rejects_invalid_identity_contracts(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    arrays = _raw_arrays()
    mutate(arrays)
    query_count = len(arrays["material_query_indices"])
    arrays["action_support"] = np.ones(query_count, dtype=np.float32)
    path = _raw_archive(tmp_path / "raw.npz", arrays)
    with pytest.raises(ValueError, match=message):
        load_genesis_particle_rollout(path)


def test_genesis_runtime_rejects_changed_units_and_identity(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert (
        validate_genesis_mpm_runtime_manifest(runtime)["runtime_id"]
        == runtime["runtime_id"]
    )
    runtime["position_units"] = "mm"
    with pytest.raises(ValueError, match="must be metres"):
        validate_genesis_mpm_runtime_manifest(runtime)

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["diagnostics"]["maximum_action_response_m"] *= 0.5
    runtime["diagnostics"]["response_to_action_ratio"] *= 0.5
    identity = {key: value for key, value in runtime.items() if key != "runtime_id"}
    runtime["runtime_id"] = content_id(identity)
    with pytest.raises(ValueError, match="response diagnostic changed"):
        validate_genesis_mpm_runtime_manifest(runtime, raw_rollout_path=raw_path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("simulation", "simulation must be a JSON object"),
        ("diagnostics", "diagnostics must be a JSON object"),
        ("implementation", "implementation must be a JSON object"),
        (
            "information_boundary",
            "information_boundary must be a JSON object",
        ),
    ],
)
def test_genesis_runtime_rejects_non_object_sections(
    tmp_path: Path, field: str, message: str
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[field] = []
    with pytest.raises(ValueError, match=message):
        validate_genesis_mpm_runtime_manifest(runtime)


def test_genesis_runtime_rejects_invalid_source_hash_roster(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["implementation"]["source_files_sha256"] = []
    with pytest.raises(ValueError, match="source_files_sha256 must be a JSON object"):
        validate_genesis_mpm_runtime_manifest(runtime)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_count", True, "positive integer"),
        ("time_step_s", True, "finite positive number"),
        ("time_step_s", np.nan, "finite positive number"),
    ],
)
def test_genesis_runtime_rejects_invalid_scalars(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[field] = value
    with pytest.raises(ValueError, match=message):
        validate_genesis_mpm_runtime_manifest(runtime)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "three finite numbers"),
        ([0.3, "bad", 0.05], "three finite numbers"),
        ([0.3, np.nan, 0.05], "three finite numbers"),
    ],
)
def test_genesis_runtime_rejects_invalid_vectors(
    tmp_path: Path, value: list[Any], message: str
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["simulation"]["beam_extents_m"] = value
    with pytest.raises(ValueError, match=message):
        validate_genesis_mpm_runtime_manifest(runtime)


def test_genesis_bundle_detects_mutated_provenance(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    output = tmp_path / "output"
    materialize_genesis_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )
    provenance = output / "provenance" / "genesis-particle-rollout.npz"
    provenance.write_bytes(provenance.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte count changed"):
        validate_genesis_mpm_backend(output)


@pytest.mark.parametrize(
    ("field", "nested", "message"),
    [
        ("inputs", None, "inputs must be a JSON object"),
        ("inputs", "raw_rollout", "raw_rollout must be a JSON object"),
        ("mapping", None, "mapping must be a JSON object"),
    ],
)
def test_genesis_bundle_rejects_non_object_records(
    tmp_path: Path, field: str, nested: str | None, message: str
) -> None:
    raw_path = _raw_archive(tmp_path / f"{field}-raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / f"{field}-runtime.json", raw_path
    )
    output = tmp_path / f"{field}-{nested or 'root'}"
    materialize_genesis_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )
    artifact_path = output / ARTIFACT_FILENAME
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if nested is None:
        artifact[field] = []
    else:
        artifact[field][nested] = []
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_genesis_mpm_backend(output)


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    genesis = types.ModuleType("genesis")
    genesis.__version__ = "1.2.2"
    genesis.gpu = "gpu"
    genesis.cpu = "cpu"

    torch = types.ModuleType("torch")
    torch.__version__ = "2.4.0+cu121"
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: f"fake-gpu-{index}",
    )
    monkeypatch.setitem(sys.modules, "genesis", genesis)
    monkeypatch.setitem(sys.modules, "torch", torch)
    sys.modules.pop("bayesian_phystwin._genesis_mpm_runtime", None)


def _runtime_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _install_fake_runtime(monkeypatch)
    return importlib.import_module("bayesian_phystwin._genesis_mpm_runtime")


def _trajectories(frame_count: int = 4) -> tuple[np.ndarray, np.ndarray]:
    frame_zero = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    zero = np.repeat(frame_zero[None], frame_count, axis=0)
    driven = zero.copy()
    driven[:, -1, 2] = np.linspace(0.0, 0.02, frame_count, dtype=np.float32)
    return driven, zero


def test_genesis_optional_runtime_materializes_valid_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime_module(monkeypatch)
    driven, zero = _trajectories()
    driven[:, 1, 2] = np.linspace(0.0, 0.01, len(driven), dtype=np.float32)
    outputs = iter((driven, zero))
    monkeypatch.setattr(runtime, "_simulate_one", lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(
        runtime,
        "deterministic_farthest_point_ids",
        lambda *args, **kwargs: np.array([0, 1], dtype=np.int64),
    )
    monkeypatch.setattr(
        runtime,
        "_implementation_record",
        lambda: {
            "repository": "IPS-Stuttgart/BayesianPhysTwin",
            "revision": "1" * 40,
            "source_files_sha256": {
                "src/bayesian_phystwin/_genesis_mpm_runtime.py": "2" * 64,
                "src/bayesian_phystwin/genesis_mpm_backend_v1.py": "3" * 64,
                "src/bayesian_phystwin/physical_rollout_v1.py": "4" * 64,
            },
        },
    )
    config = runtime.GenesisMpmSmokeConfig(frame_count=4, query_count=2)
    raw_path = tmp_path / "raw.npz"
    manifest_path = tmp_path / "runtime.json"
    result = runtime.run_genesis_mpm_smoke(
        raw_rollout_path=raw_path,
        runtime_manifest_path=manifest_path,
        config=config,
    )
    assert result["maximum_action_response_m"] == pytest.approx(0.02)
    assert result["runtime"]["device_name"] == "fake-gpu-0"
    validate_genesis_mpm_runtime_manifest(result["runtime"], raw_rollout_path=raw_path)


def test_genesis_runtime_binds_implementation_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_module(monkeypatch)
    record = runtime._implementation_record()
    assert record["repository"] == "IPS-Stuttgart/BayesianPhysTwin"
    assert len(record["revision"]) == 40
    assert set(record["source_files_sha256"]) == set(runtime._IMPLEMENTATION_PATHS)

    monkeypatch.setenv("BPT_IMPLEMENTATION_REVISION", "not-a-revision")
    with pytest.raises(RuntimeError, match="not a lowercase Git SHA-1"):
        runtime._implementation_record()

    monkeypatch.setenv("BPT_IMPLEMENTATION_REVISION", "1" * 40)
    monkeypatch.setattr(runtime, "_IMPLEMENTATION_PATHS", ("missing.py",))
    with pytest.raises(RuntimeError, match="source file is unavailable"):
        runtime._implementation_record()


def test_genesis_runtime_normalizes_particle_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_module(monkeypatch)

    class TensorLike:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def detach(self) -> TensorLike:
            return self

        def cpu(self) -> TensorLike:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    class Entity:
        def __init__(self, value: Any) -> None:
            self.value = value

        def get_particles_pos(self) -> Any:
            return self.value

    positions = np.arange(12, dtype=np.float64).reshape(1, 4, 3)
    normalized = runtime._positions_numpy(Entity(TensorLike(positions)))
    assert normalized.shape == (4, 3)
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    np.testing.assert_array_equal(
        runtime._positions_numpy(Entity(positions[0])), positions[0]
    )
    with pytest.raises(RuntimeError, match="unexpected particle-position shape"):
        runtime._positions_numpy(Entity(np.zeros((4, 2), dtype=np.float32)))
    assert runtime._backend_value("cpu") == "cpu"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_count", 1, "at least two"),
        ("query_count", 0, "positive"),
        ("fps", np.nan, "finite and positive"),
        ("substeps", 0, "positive"),
        ("grid_density", 4, "at least eight"),
        ("young_modulus_pa", 0.0, "finite and positive"),
        ("poisson_ratio", 0.5, "poisson_ratio"),
        ("elastic_model", "unknown", "elastic_model"),
        ("seed", True, "seed"),
    ],
)
def test_genesis_runtime_config_validation(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    runtime = _runtime_module(monkeypatch)
    config = replace(runtime.GenesisMpmSmokeConfig(), **{field: value})
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_genesis_runtime_rejects_bad_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime_module(monkeypatch)
    with pytest.raises(TypeError, match="GenesisMpmSmokeConfig"):
        runtime.run_genesis_mpm_smoke(
            raw_rollout_path=tmp_path / "raw.npz",
            runtime_manifest_path=tmp_path / "runtime.json",
            config=object(),
        )
    with pytest.raises(ValueError, match="gpu or cpu"):
        runtime.run_genesis_mpm_smoke(
            raw_rollout_path=tmp_path / "raw.npz",
            runtime_manifest_path=tmp_path / "runtime.json",
            backend="metal",
        )
    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"exists")
    with pytest.raises(FileExistsError, match="already exists"):
        runtime.run_genesis_mpm_smoke(
            raw_rollout_path=existing,
            runtime_manifest_path=tmp_path / "runtime.json",
        )

    config = runtime.GenesisMpmSmokeConfig(frame_count=4, query_count=2)
    driven, zero = _trajectories()
    zero[0, 0, 0] = 1.0
    outputs = iter((driven, zero))
    monkeypatch.setattr(runtime, "_simulate_one", lambda *args, **kwargs: next(outputs))
    with pytest.raises(RuntimeError, match="frame zero"):
        runtime.run_genesis_mpm_smoke(
            raw_rollout_path=tmp_path / "mismatch.npz",
            runtime_manifest_path=tmp_path / "mismatch.json",
            config=config,
        )

    driven, zero = _trajectories()
    driven[:, -1, 2] = np.linspace(0.0, 0.04, 4, dtype=np.float32)
    outputs = iter((driven, zero))
    monkeypatch.setattr(runtime, "_simulate_one", lambda *args, **kwargs: next(outputs))
    with pytest.raises(RuntimeError, match="stability cap"):
        runtime.run_genesis_mpm_smoke(
            raw_rollout_path=tmp_path / "unstable.npz",
            runtime_manifest_path=tmp_path / "unstable.json",
            config=config,
        )

    outputs = iter((zero.copy(), zero.copy()))
    monkeypatch.setattr(runtime, "_simulate_one", lambda *args, **kwargs: next(outputs))
    with pytest.raises(RuntimeError, match="no action-conditioned response"):
        runtime.run_genesis_mpm_smoke(
            raw_rollout_path=tmp_path / "no-response.npz",
            runtime_manifest_path=tmp_path / "no-response.json",
            config=config,
        )


def test_genesis_cli_dispatches_commands_and_missing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bayesian_phystwin.cli.genesis_mpm_backend as cli

    calls: list[tuple[str, Any]] = []

    def materialize(**kwargs: Any) -> dict[str, object]:
        calls.append(("materialize", kwargs))
        return {"kind": "materialized"}

    def validate(path: Path) -> dict[str, object]:
        calls.append(("validate", path))
        return {"kind": "validated"}

    monkeypatch.setattr(cli, "materialize_genesis_mpm_backend", materialize)
    monkeypatch.setattr(cli, "validate_genesis_mpm_backend", validate)
    assert (
        cli.main(
            [
                "materialize",
                str(tmp_path / "raw.npz"),
                str(tmp_path / "runtime.json"),
                str(tmp_path / "bundle"),
            ]
        )
        == 0
    )
    assert cli.main(["validate", str(tmp_path / "bundle")]) == 0
    assert {name for name, _ in calls} == {"materialize", "validate"}
    assert '"kind": "materialized"' in capsys.readouterr().out

    runtime_module = _runtime_module(monkeypatch)
    smoke_calls: list[dict[str, Any]] = []

    def run_smoke(**kwargs: Any) -> dict[str, object]:
        smoke_calls.append(kwargs)
        return {"kind": "smoke"}

    monkeypatch.setattr(runtime_module, "run_genesis_mpm_smoke", run_smoke)
    assert (
        cli.main(
            [
                "smoke",
                str(tmp_path / "smoke-bundle"),
                "--backend",
                "cpu",
                "--frames",
                "5",
                "--queries",
                "3",
                "--fps",
                "60",
                "--substeps",
                "4",
                "--grid-density",
                "32",
                "--attachment-stiffness",
                "250",
                "--action-displacement-m",
                "0.02",
            ]
        )
        == 0
    )
    assert len(smoke_calls) == 1
    assert smoke_calls[0]["backend"] == "cpu"
    config = smoke_calls[0]["config"]
    assert config.frame_count == 5
    assert config.action_displacement_m == pytest.approx(0.02)

    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "bayesian_phystwin._genesis_mpm_runtime":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("bayesian_phystwin._genesis_mpm_runtime", None)
    args = cli.build_parser().parse_args(["smoke", str(tmp_path / "smoke")])
    with pytest.raises(RuntimeError, match="optional"):
        cli._smoke(args)
