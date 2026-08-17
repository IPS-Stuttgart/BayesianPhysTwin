from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.cli.lagrangian_backend import main as cli_main
from bayesian_phystwin.lagrangian_backend_v1 import (
    ARTIFACT_FILENAME,
    GENESIS_MPM_PROFILE,
    GENESIS_WORLD_REPOSITORY,
    JAX_FEM_PROFILE,
    JAX_FEM_REPOSITORY,
    LAGRANGIAN_RUNTIME_SCHEMA,
    PHYSICAL_ARCHIVE_FILENAME,
    RAW_ARCHIVE_FILENAME,
    describe_lagrangian_backend_profiles,
    file_sha256,
    load_lagrangian_rollout,
    materialize_lagrangian_backend,
    validate_lagrangian_backend,
    validate_lagrangian_runtime_manifest,
)
from bayesian_phystwin.physical_rollout_v1 import (
    load_physical_rollout_archive,
    write_deterministic_npz,
)

JAX_REVISION = "82c6993c16704e38611f9cb91a5b70f1c690daee"
GENESIS_REVISION = "06a5f2518c254f7ef2cc8757a7f84ed96eb68232"
DEFAULT_FLOAT_DTYPE = np.dtype("float32")


def _raw_arrays(
    dtype: np.dtype[np.floating] = DEFAULT_FLOAT_DTYPE,
) -> dict[str, np.ndarray]:
    frame_zero = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.15, 0.00, 0.00],
            [0.20, 0.00, 0.00],
        ],
        dtype=dtype,
    )
    zero = np.repeat(frame_zero[None], 8, axis=0)
    driven = zero.copy()
    ramp = np.linspace(0.0, 0.02, len(driven), dtype=dtype)
    shape = np.linspace(0.0, 1.0, len(frame_zero), dtype=dtype)
    driven[:, :, 2] += ramp[:, None] * shape[None]
    driven[0] = frame_zero
    return {
        "driven_point_positions_m": driven,
        "zero_action_point_positions_m": zero,
        "material_query_indices": np.array([0, 4, 2], dtype=np.int64),
        "action_support": np.array([0.0, 1.0, 0.5], dtype=dtype),
    }


def _raw_archive(path: Path, arrays: dict[str, np.ndarray] | None = None) -> Path:
    write_deterministic_npz(path, arrays or _raw_arrays())
    return path


def _profile_fields(profile: str) -> tuple[str, str, str, str, str, dict[str, object]]:
    if profile == JAX_FEM_PROFILE:
        return (
            JAX_FEM_REPOSITORY,
            JAX_REVISION,
            "0+82c6993",
            "mesh-node",
            "differentiable-fem",
            {
                "element_type": "HEX8",
                "constitutive_model": "neo-hookean",
                "nonlinear_solver": "newton",
                "differentiation_mode": "jax-autodiff",
                "precision": "float32",
            },
        )
    return (
        GENESIS_WORLD_REPOSITORY,
        GENESIS_REVISION,
        "1.3.3",
        "material-particle",
        "explicit-mpm",
        {
            "solver": "mpm",
            "material_model": "elastic",
            "compute_backend": "cpu",
            "particle_size_m": 0.02,
            "substeps": 10,
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "differentiable": True,
            "precision": "float32",
        },
    )


def _runtime_manifest(path: Path, raw_path: Path, profile: str) -> Path:
    (
        repository,
        revision,
        version,
        identity_kind,
        solver_family,
        metadata,
    ) = _profile_fields(profile)
    step_axis, step_units, step_size = (
        ("load-step", "1", 1.0 / 7.0)
        if profile == JAX_FEM_PROFILE
        else ("time", "s", 0.004)
    )
    identity = {
        "schema": LAGRANGIAN_RUNTIME_SCHEMA,
        "schema_version": 1,
        "backend_profile": profile,
        "engine_repository": repository,
        "engine_revision": revision,
        "engine_version": version,
        "python_version": "3.12.13",
        "device": "cpu",
        "coordinate_frame": "right-handed-z-up-world-v1",
        "position_units": "m",
        "step_axis": step_axis,
        "step_units": step_units,
        "step_size": step_size,
        "frame_count": 8,
        "point_count": 5,
        "query_count": 3,
        "identity_kind": identity_kind,
        "solver_family": solver_family,
        "backend_metadata": metadata,
        "source_artifacts": {
            "scene/config.json": "a" * 64,
            "scene/mesh.vtk": "b" * 64,
        },
        "information_boundary": {
            "source_kind": "synthetic",
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


@pytest.mark.parametrize("profile", [JAX_FEM_PROFILE, GENESIS_MPM_PROFILE])
def test_profiles_materialize_to_the_shared_physical_contract(
    tmp_path: Path, profile: str
) -> None:
    raw_path = _raw_archive(tmp_path / f"{profile}.npz")
    runtime_path = _runtime_manifest(tmp_path / f"{profile}.json", raw_path, profile)
    output = tmp_path / f"out-{profile}"

    artifact = materialize_lagrangian_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )

    assert artifact == validate_lagrangian_backend(output)
    assert artifact["backend_profile"] == profile
    assert artifact["mapping"]["material_identity_preserved"] is True
    physical = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    expected = _raw_arrays()["driven_point_positions_m"][:, [0, 4, 2]]
    np.testing.assert_array_equal(physical["prediction_m"], expected)
    np.testing.assert_array_equal(physical["driven_readout_m"], expected)
    np.testing.assert_array_equal(
        physical["persistence_m"], np.repeat(expected[0][None], 8, axis=0)
    )


def test_profile_registry_exposes_strain_and_dynamics_candidates() -> None:
    profiles = {
        item["backend_profile"]: item for item in describe_lagrangian_backend_profiles()
    }
    assert set(profiles) == {JAX_FEM_PROFILE, GENESIS_MPM_PROFILE}
    assert profiles[JAX_FEM_PROFILE]["identity_kind"] == "mesh-node"
    assert profiles[JAX_FEM_PROFILE]["solver_family"] == "differentiable-fem"
    assert profiles[GENESIS_MPM_PROFILE]["identity_kind"] == "material-particle"
    assert profiles[GENESIS_MPM_PROFILE]["solver_family"] == "explicit-mpm"


def test_materialization_is_byte_deterministic(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, JAX_FEM_PROFILE
    )
    first = materialize_lagrangian_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=tmp_path / "first",
    )
    second = materialize_lagrangian_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=tmp_path / "second",
    )
    assert first["artifact_id"] == second["artifact_id"]
    for relative in (
        ARTIFACT_FILENAME,
        PHYSICAL_ARCHIVE_FILENAME,
        f"provenance/{RAW_ARCHIVE_FILENAME}",
    ):
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()


def test_rejects_unknown_or_mismatched_profile(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, JAX_FEM_PROFILE
    )
    runtime = json.loads(runtime_path.read_text())

    runtime["backend_profile"] = "unknown-v1"
    with pytest.raises(ValueError, match="unknown backend_profile"):
        validate_lagrangian_runtime_manifest(runtime)

    runtime = json.loads(runtime_path.read_text())
    runtime["engine_repository"] = GENESIS_WORLD_REPOSITORY
    with pytest.raises(ValueError, match="does not match backend profile"):
        validate_lagrangian_runtime_manifest(runtime)


def test_rejects_duplicate_queries_frame_zero_drift_and_unsupported_dtype(
    tmp_path: Path,
) -> None:
    duplicate = _raw_arrays()
    duplicate["material_query_indices"] = np.array([1, 1], dtype=np.int64)
    duplicate["action_support"] = np.ones(2, dtype=np.float32)
    duplicate_path = _raw_archive(tmp_path / "duplicate.npz", duplicate)
    with pytest.raises(ValueError, match="must be unique"):
        load_lagrangian_rollout(duplicate_path)

    drift = _raw_arrays()
    drift["zero_action_point_positions_m"] = drift[
        "zero_action_point_positions_m"
    ].copy()
    drift["zero_action_point_positions_m"][0, 0, 0] += 0.001
    drift_path = _raw_archive(tmp_path / "drift.npz", drift)
    with pytest.raises(ValueError, match="differ at frame zero"):
        load_lagrangian_rollout(drift_path)

    half = _raw_arrays(np.dtype("float16"))
    half_path = _raw_archive(tmp_path / "half.npz", half)
    with pytest.raises(ValueError, match="float32 or float64"):
        load_lagrangian_rollout(half_path)


def test_rejects_future_information_and_precision_mismatch(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, JAX_FEM_PROFILE
    )
    runtime = json.loads(runtime_path.read_text())
    runtime["information_boundary"]["outcomes_read"] = True
    with pytest.raises(ValueError, match="outcomes must remain closed"):
        validate_lagrangian_runtime_manifest(runtime)

    runtime = json.loads(runtime_path.read_text())
    runtime["backend_metadata"]["precision"] = "float64"
    identity = {key: value for key, value in runtime.items() if key != "runtime_id"}
    runtime["runtime_id"] = content_id(identity)
    changed = tmp_path / "precision.json"
    write_atomic_json(runtime, changed, overwrite=False)
    with pytest.raises(ValueError, match="precision differs"):
        materialize_lagrangian_backend(
            raw_rollout_path=raw_path,
            runtime_manifest_path=changed,
            output_dir=tmp_path / "out",
        )


def test_bundle_detects_mutated_provenance_and_checksum(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, GENESIS_MPM_PROFILE
    )
    output = tmp_path / "out"
    materialize_lagrangian_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )
    provenance = output / "provenance" / RAW_ARCHIVE_FILENAME
    provenance.write_bytes(provenance.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte count changed"):
        validate_lagrangian_backend(output)


def test_runtime_manifest_source_only_and_validation_edges(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, GENESIS_MPM_PROFILE
    )
    runtime = json.loads(runtime_path.read_text())
    runtime["information_boundary"] = {
        "source_kind": "source-only",
        "dataset_payload_read": True,
        "future_observations_read": False,
        "outcomes_read": False,
        "known_action_used": True,
    }
    identity = {key: value for key, value in runtime.items() if key != "runtime_id"}
    runtime["runtime_id"] = content_id(identity)
    assert (
        validate_lagrangian_runtime_manifest(runtime)["runtime_id"]
        == runtime["runtime_id"]
    )

    for field, replacement, message in (
        ("frame_count", True, "frame_count must be a positive integer"),
        ("step_size", True, "step_size must be a finite positive number"),
        ("step_size", float("inf"), "step_size must be a finite positive number"),
        ("backend_metadata", None, "backend_metadata must be a JSON object"),
    ):
        changed = json.loads(json.dumps(runtime))
        changed[field] = replacement
        with pytest.raises(ValueError, match=message):
            validate_lagrangian_runtime_manifest(changed)

    for gravity in (
        [0.0, -9.81],
        [0.0, True, -9.81],
        [0.0, float("nan"), -9.81],
    ):
        changed = json.loads(json.dumps(runtime))
        changed["backend_metadata"]["gravity_m_s2"] = gravity
        with pytest.raises(ValueError, match="must contain three finite numbers"):
            validate_lagrangian_runtime_manifest(changed)


def test_rejects_unreadable_rollout_and_changed_checksum(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.npz"
    invalid.write_text("not an npz", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load raw Lagrangian rollout"):
        load_lagrangian_rollout(invalid)

    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, JAX_FEM_PROFILE
    )
    output = tmp_path / "out"
    materialize_lagrangian_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )
    checksum = output / "SHA256SUMS"
    checksum.write_text(checksum.read_text(encoding="utf-8") + "changed\n")
    with pytest.raises(ValueError, match="checksum manifest changed"):
        validate_lagrangian_backend(output)


def test_cli_lists_profiles_and_validates_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_main(["profiles"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["schema"] == "bayesian-phystwin.material-backend-registry"
    assert [item["profile_id"] for item in listed["profiles"]] == [
        "jax-fem-quasistatic-v1",
        "warp-fem-v1",
        "sofa-fem-v1",
        "genesis-mpm-v1",
        "position-based-dynamics-v1",
        "physx-fem-v1",
        "mujoco-flex-v1",
        "drake-fem-v1",
        "fenicsx-fem-v1",
        "pyelastica-cosserat-rod-v1",
    ]
    genesis = next(
        item for item in listed["profiles"] if item["profile_id"] == "genesis-mpm-v1"
    )
    assert {variant["producer_profile_id"] for variant in genesis["variants"]} == {
        "genesis-mpm-v1",
        "genesis-world-mpm-v1",
    }

    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, JAX_FEM_PROFILE
    )
    output = tmp_path / "out"
    assert cli_main(["materialize", str(raw_path), str(runtime_path), str(output)]) == 0
    capsys.readouterr()
    assert cli_main(["validate", str(output)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["backend_profile"] == JAX_FEM_PROFILE
