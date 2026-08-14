from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.cli.external_physics_backend import main
from bayesian_phystwin.external_physics_backend_v1 import (
    ARTIFACT_FILENAME,
    EXTERNAL_PHYSICS_RAW_ARRAY_NAMES,
    PHYSICAL_ARCHIVE_FILENAME,
    array_sha256,
    build_external_physics_runtime_manifest,
    load_external_entity_rollout,
    materialize_external_physics_backend,
    physical_rollout_from_external_entities,
    validate_external_physics_backend,
    validate_external_physics_runtime_manifest,
    write_external_physics_runtime_manifest,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive
from bayesian_phystwin.physics_backend_registry_v1 import BUILTIN_BACKEND_PROFILES


def _raw_arrays() -> dict[str, np.ndarray[Any, Any]]:
    frame_zero = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    driven = np.repeat(frame_zero[None], 5, axis=0)
    zero = driven.copy()
    driven[:, 2:, 2] += np.linspace(0.0, 0.04, 5)[:, None]
    return {
        "driven_entity_positions_m": driven,
        "zero_action_entity_positions_m": zero,
        "query_entity_indices": np.array([0, 2, 3], dtype=np.int64),
        "action_support": np.array([0.0, 0.8, 1.0], dtype=np.float64),
    }


def _write_raw(
    path: Path,
    arrays: dict[str, np.ndarray[Any, Any]] | None = None,
) -> None:
    np.savez(path, **(arrays or _raw_arrays()))


def _runtime_kwargs(raw: Path) -> dict[str, object]:
    return {
        "raw_rollout_path": raw,
        "profile": BUILTIN_BACKEND_PROFILES[0],
        "engine_revision": "a" * 40,
        "engine_version": "test-engine-1",
        "producer_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "producer_revision": "b" * 40,
        "coordinate_frame": "right-handed-z-up-world-v1",
        "time_step_s": 1.0 / 120.0,
        "topology_sha256": "c" * 64,
        "material_model": "neo-hookean",
        "observation_end_frame_exclusive": 2,
        "parameterization": {
            "density_kg_m3": 1000.0,
            "young_modulus_pa": 100000.0,
        },
        "producer_artifacts": {"configs/genesis-scene.json": "d" * 64},
    }


def _write_runtime(raw: Path, path: Path) -> dict[str, Any]:
    return write_external_physics_runtime_manifest(
        output_path=path,
        **_runtime_kwargs(raw),
    )


def test_runtime_manifest_binds_profile_identity_and_causal_boundary(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.npz"
    _write_raw(raw)
    runtime = build_external_physics_runtime_manifest(**_runtime_kwargs(raw))
    assert runtime["backend_profile"]["profile_id"] == "genesis-mpm-v1"
    assert runtime["frame_count"] == 5
    assert runtime["entity_count"] == 4
    assert runtime["query_count"] == 3
    assert runtime["entity_identity_sha256"] == array_sha256(
        _raw_arrays()["driven_entity_positions_m"][0]
    )
    assert runtime["information_boundary"] == {
        "observation_end_frame_exclusive": 2,
        "future_observations_used": False,
        "outcomes_used_for_selection": False,
        "target_outcomes_used": False,
        "known_action_used": True,
    }
    assert validate_external_physics_runtime_manifest(
        runtime,
        raw_rollout_path=raw,
    ) == runtime


def test_end_to_end_bundle_rederives_portable_rollout(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    runtime_path = tmp_path / "runtime.json"
    output = tmp_path / "bundle"
    _write_raw(raw)
    runtime = _write_runtime(raw, runtime_path)

    artifact = materialize_external_physics_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )
    assert artifact["runtime_id"] == runtime["runtime_id"]
    assert artifact["backend_profile"]["profile_id"] == "genesis-mpm-v1"
    assert artifact["mapping"]["persistent_entity_identity_preserved"] is True
    assert validate_external_physics_backend(output) == artifact

    _, arrays = load_external_entity_rollout(raw)
    expected = physical_rollout_from_external_entities(arrays)
    actual = load_physical_rollout_archive(output / PHYSICAL_ARCHIVE_FILENAME)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_bundle_is_deterministic_for_the_same_inputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    runtime_path = tmp_path / "runtime.json"
    _write_raw(raw)
    _write_runtime(raw, runtime_path)
    first = materialize_external_physics_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=runtime_path,
        output_dir=tmp_path / "first",
    )
    second = materialize_external_physics_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=runtime_path,
        output_dir=tmp_path / "second",
    )
    assert first["artifact_id"] == second["artifact_id"]
    assert (
        (tmp_path / "first" / PHYSICAL_ARCHIVE_FILENAME).read_bytes()
        == (tmp_path / "second" / PHYSICAL_ARCHIVE_FILENAME).read_bytes()
    )


def test_tampered_output_and_extra_files_fail_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    runtime_path = tmp_path / "runtime.json"
    output = tmp_path / "bundle"
    _write_raw(raw)
    _write_runtime(raw, runtime_path)
    materialize_external_physics_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )
    (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file roster"):
        validate_external_physics_backend(output)
    (output / "unexpected.txt").unlink()
    physical = output / PHYSICAL_ARCHIVE_FILENAME
    physical.write_bytes(physical.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="byte count|SHA-256"):
        validate_external_physics_backend(output)


def test_information_boundary_rejects_future_or_target_use(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    _write_raw(raw)
    runtime = build_external_physics_runtime_manifest(**_runtime_kwargs(raw))
    for field in (
        "future_observations_used",
        "outcomes_used_for_selection",
        "target_outcomes_used",
    ):
        bad = json.loads(json.dumps(runtime))
        bad["information_boundary"][field] = True
        identity = {key: value for key, value in bad.items() if key != "runtime_id"}
        bad["runtime_id"] = content_id(identity)
        with pytest.raises(ValueError, match="forbidden"):
            validate_external_physics_runtime_manifest(bad)


def test_runtime_rejects_wrong_raw_digest_and_entity_order(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    _write_raw(raw)
    runtime = build_external_physics_runtime_manifest(**_runtime_kwargs(raw))
    changed = _raw_arrays()
    changed["driven_entity_positions_m"][0, 0, 0] = 0.01
    changed["zero_action_entity_positions_m"][0, 0, 0] = 0.01
    other = tmp_path / "other.npz"
    _write_raw(other, changed)
    with pytest.raises(ValueError, match="raw rollout SHA-256"):
        validate_external_physics_runtime_manifest(runtime, raw_rollout_path=other)


def test_raw_rollout_rejects_identity_and_query_contract_violations(
    tmp_path: Path,
) -> None:
    cases = []
    mismatch = _raw_arrays()
    mismatch["zero_action_entity_positions_m"][0, 0, 0] = 0.1
    cases.append((mismatch, "differ at frame zero"))
    duplicate = _raw_arrays()
    duplicate["query_entity_indices"] = np.array([0, 0, 3], dtype=np.int64)
    cases.append((duplicate, "must be unique"))
    invalid_support = _raw_arrays()
    invalid_support["action_support"][1] = 2.0
    cases.append((invalid_support, "action_support is invalid"))
    missing = _raw_arrays()
    del missing[next(iter(EXTERNAL_PHYSICS_RAW_ARRAY_NAMES))]
    cases.append((missing, "cannot load external entity rollout"))

    for index, (arrays, message) in enumerate(cases):
        path = tmp_path / f"invalid-{index}.npz"
        _write_raw(path, arrays)
        with pytest.raises(ValueError, match=message):
            load_external_entity_rollout(path)


def test_runtime_writer_is_no_clobber_by_default(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    runtime_path = tmp_path / "runtime.json"
    _write_raw(raw)
    _write_runtime(raw, runtime_path)
    with pytest.raises(FileExistsError):
        _write_runtime(raw, runtime_path)


def test_cli_lists_profiles_and_materializes(tmp_path: Path, capsys: Any) -> None:
    assert main(["profiles"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["profile_id"] == "genesis-mpm-v1"

    raw = tmp_path / "raw.npz"
    runtime_path = tmp_path / "runtime.json"
    output = tmp_path / "bundle"
    parameterization = tmp_path / "parameters.json"
    _write_raw(raw)
    parameterization.write_text('{"young_modulus_pa": 100000.0}\n')
    assert (
        main(
            [
                "runtime",
                "genesis-mpm-v1",
                str(raw),
                str(runtime_path),
                "--engine-revision",
                "a" * 40,
                "--engine-version",
                "test-engine-1",
                "--producer-repository",
                "IPS-Stuttgart/BayesianPhysTwin",
                "--producer-revision",
                "b" * 40,
                "--coordinate-frame",
                "right-handed-z-up-world-v1",
                "--time-step-s",
                str(1.0 / 120.0),
                "--topology-sha256",
                "c" * 64,
                "--material-model",
                "neo-hookean",
                "--observation-end-frame-exclusive",
                "2",
                "--parameterization-json",
                str(parameterization),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["materialize", str(raw), str(runtime_path), str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["backend_profile"]["profile_id"] == "genesis-mpm-v1"
    assert (output / ARTIFACT_FILENAME).is_file()
