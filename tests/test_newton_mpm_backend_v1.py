from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    load_physical_archive as load_deform360_physical_archive,
)
from bayesian_phystwin.newton_mpm_backend_v1 import (
    ARTIFACT_FILENAME,
    NEWTON_MPM_BACKEND_KIND,
    NEWTON_MPM_ENGINE_REPOSITORY,
    NEWTON_MPM_RUNTIME_SCHEMA,
    PHYSICAL_ARCHIVE_FILENAME,
    file_sha256,
    load_newton_particle_rollout,
    materialize_newton_mpm_backend,
    validate_newton_mpm_backend,
)
from bayesian_phystwin.physical_rollout_v1 import (
    PHYSICAL_ROLLOUT_ARRAY_NAMES,
    load_physical_rollout_archive,
    write_deterministic_npz,
)


def _raw_arrays() -> dict[str, np.ndarray]:
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
    zero = np.repeat(frame_zero[None], 76, axis=0)
    driven = zero.copy()
    ramp = np.linspace(0.0, 0.02, len(driven), dtype=np.float32)
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
    identity = {
        "schema": NEWTON_MPM_RUNTIME_SCHEMA,
        "schema_version": 1,
        "backend_kind": NEWTON_MPM_BACKEND_KIND,
        "engine_repository": NEWTON_MPM_ENGINE_REPOSITORY,
        "engine_version": "1.5.0",
        "warp_version": "1.16.0",
        "python_version": "3.12.13",
        "device": "cuda:0",
        "device_name": "synthetic-test-device",
        "coordinate_frame": "right-handed-z-up-world-v1",
        "position_units": "m",
        "time_units": "s",
        "frame_count": raw["driven_particle_positions_m"].shape[0],
        "particle_count": raw["driven_particle_positions_m"].shape[1],
        "query_count": len(raw["material_query_indices"]),
        "time_step_s": 1.0 / 120.0,
        "simulation": {
            "scene": "kinematic-beam-bend-v1",
            "beam_extents_m": [0.3, 0.05, 0.05],
            "action_displacement_m": [0.0, 0.0, 0.025],
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "density_kg_m3": 1000.0,
            "young_modulus_pa": 500000.0,
            "poisson_ratio": 0.35,
            "damping": 0.002,
            "voxel_size_m": 0.025,
            "substeps": 1,
            "solver": "implicit-mpm-cr",
            "max_iterations": 50,
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


def test_materializes_generic_physical_rollout_and_preserves_query_identity(
    tmp_path: Path,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    output = tmp_path / "output"

    artifact = materialize_newton_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )

    assert artifact == validate_newton_mpm_backend(output)
    assert artifact["mapping"]["material_identity_preserved"] is True
    physical = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    assert set(physical) == PHYSICAL_ROLLOUT_ARRAY_NAMES
    expected = _raw_arrays()["driven_particle_positions_m"][:, [0, 4, 2]]
    np.testing.assert_array_equal(physical["prediction_m"], expected)
    np.testing.assert_array_equal(physical["driven_readout_m"], expected)
    np.testing.assert_array_equal(
        physical["persistence_m"],
        np.repeat(expected[0][None], 76, axis=0),
    )
    # The existing Deform360 consumer accepts the exact same six-array archive.
    existing = load_deform360_physical_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    np.testing.assert_array_equal(existing["prediction_m"], expected)


def test_materialization_is_byte_deterministic(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)

    first = materialize_newton_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=tmp_path / "first",
    )
    second = materialize_newton_mpm_backend(
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


def test_rejects_duplicate_or_out_of_range_material_queries(tmp_path: Path) -> None:
    for name, indices, message in (
        ("duplicate", [1, 1], "must be unique"),
        ("outside", [1, 9], "exceeds particle count"),
    ):
        arrays = _raw_arrays()
        arrays["material_query_indices"] = np.array(indices, dtype=np.int64)
        arrays["action_support"] = np.ones(len(indices), dtype=np.float32)
        path = _raw_archive(tmp_path / f"{name}.npz", arrays)
        with pytest.raises(ValueError, match=message):
            load_newton_particle_rollout(path)


def test_rejects_frame_zero_drift_and_changed_units(tmp_path: Path) -> None:
    arrays = _raw_arrays()
    arrays["zero_action_particle_positions_m"] = arrays[
        "zero_action_particle_positions_m"
    ].copy()
    arrays["zero_action_particle_positions_m"][0, 0, 0] += 0.001
    drifted = _raw_archive(tmp_path / "drifted.npz", arrays)
    with pytest.raises(ValueError, match="differ at frame zero"):
        load_newton_particle_rollout(drifted)

    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["position_units"] = "mm"
    changed = tmp_path / "changed-runtime.json"
    changed.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="must be metres"):
        materialize_newton_mpm_backend(
            raw_rollout_path=raw_path,
            runtime_manifest_path=changed,
            output_dir=tmp_path / "output",
        )


def test_bundle_detects_mutated_particle_provenance(tmp_path: Path) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path)
    output = tmp_path / "output"
    materialize_newton_mpm_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )

    provenance = output / "provenance" / "newton-particle-rollout.npz"
    provenance.write_bytes(provenance.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte count changed"):
        validate_newton_mpm_backend(output)
