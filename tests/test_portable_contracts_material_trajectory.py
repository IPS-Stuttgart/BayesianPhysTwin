from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.cli.material_trajectory_backend import main
from bayesian_phystwin.material_trajectory_backend_v1 import (
    ARTIFACT_FILENAME,
    MATERIAL_BACKEND_PROFILES,
    MATERIAL_TRAJECTORY_RUNTIME_SCHEMA,
    PHYSICAL_ARCHIVE_FILENAME,
    RAW_ARCHIVE_FILENAME,
    RUNTIME_FILENAME,
    file_sha256,
    get_material_backend_profile,
    load_material_trajectory_rollout,
    material_backend_profile_records,
    materialize_material_trajectory_backend,
    validate_material_runtime_manifest,
    validate_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import (
    PHYSICAL_ROLLOUT_ARRAY_NAMES,
    load_physical_rollout_archive,
    write_deterministic_npz,
)

BACKEND_KINDS = tuple(sorted(MATERIAL_BACKEND_PROFILES))


def _raw_arrays() -> dict[str, np.ndarray]:
    frame_zero = np.stack(
        [
            np.linspace(0.0, 0.20, 5, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
        ],
        axis=1,
    )
    zero = np.repeat(frame_zero[None], 8, axis=0)
    driven = zero.copy()
    ramp = np.linspace(0.0, 0.025, len(driven), dtype=np.float32)
    spatial = np.linspace(0.0, 1.0, len(frame_zero), dtype=np.float32)
    driven[:, :, 2] += ramp[:, None] * spatial[None]
    driven[0] = frame_zero
    return {
        "driven_material_positions_m": driven,
        "zero_action_material_positions_m": zero,
        "material_query_indices": np.array([4, 0, 2], dtype=np.int64),
        "action_support": np.array([1.0, 0.0, 0.5], dtype=np.float32),
    }


def _raw_archive(path: Path, arrays: dict[str, np.ndarray] | None = None) -> Path:
    write_deterministic_npz(path, arrays or _raw_arrays())
    return path


def _runtime_payload(raw_path: Path, backend_kind: str) -> dict[str, Any]:
    raw = _raw_arrays()
    profile = get_material_backend_profile(backend_kind)
    identity: dict[str, Any] = {
        "schema": MATERIAL_TRAJECTORY_RUNTIME_SCHEMA,
        "schema_version": 1,
        "backend_kind": backend_kind,
        "engine_repository": profile.engine_repository,
        "engine_revision": "a" * 40,
        "engine_version": "test-release",
        "producer_version": "exporter-v1",
        "python_version": "3.12.0",
        "device": "cpu",
        "device_name": "synthetic-test-device",
        "coordinate_frame": "right-handed-z-up-world-v1",
        "position_units": "m",
        "time_units": "s",
        "frame_count": raw["driven_material_positions_m"].shape[0],
        "state_count": raw["driven_material_positions_m"].shape[1],
        "query_count": len(raw["material_query_indices"]),
        "time_step_s": 1.0 / 120.0,
        "solver_family": profile.solver_family,
        "identity_kind": profile.identity_kind,
        "simulation": {
            "scene_id": "beam-bend-source-v1",
            "model_kind": "deformable-solid",
            "constitutive_model": "neo-hookean",
            "integrator": "implicit-euler",
            "solver": "profile-native-solver",
            "substeps": 2,
            "engine_parameters": {
                "density_kg_m3": 1000.0,
                "young_modulus_pa": 500000.0,
                "poisson_ratio": 0.35,
            },
        },
        "information_boundary": {
            "future_observations_read": False,
            "future_outcomes_read": False,
            "known_action_used": True,
            "action_support_uses_observation_residuals": False,
            "material_query_indices_fixed_at_frame_zero": True,
        },
        "raw_rollout_sha256": file_sha256(raw_path),
    }
    return {**identity, "runtime_id": content_id(identity)}


def _runtime_manifest(path: Path, raw_path: Path, backend_kind: str) -> Path:
    write_atomic_json(_runtime_payload(raw_path, backend_kind), path, overwrite=False)
    return path


def _rehash(payload: dict[str, Any], identity_field: str) -> None:
    identity = {key: value for key, value in payload.items() if key != identity_field}
    payload[identity_field] = content_id(identity)


def test_profile_catalog_is_sorted_and_complete() -> None:
    records = material_backend_profile_records()
    assert tuple(record["backend_kind"] for record in records) == BACKEND_KINDS
    assert BACKEND_KINDS == (
        "genesis-mpm-v1",
        "mujoco-flex-v1",
        "sofa-fem-v1",
    )
    with pytest.raises(ValueError, match="unsupported material backend kind"):
        get_material_backend_profile("unknown-v1")


@pytest.mark.parametrize("backend_kind", BACKEND_KINDS)
def test_profiles_materialize_deterministically_into_one_physical_contract(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path, backend_kind)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_artifact = materialize_material_trajectory_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=first,
    )
    second_artifact = materialize_material_trajectory_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=second,
    )

    assert first_artifact == validate_material_trajectory_backend(first)
    assert first_artifact["artifact_id"] == second_artifact["artifact_id"]
    assert (
        first_artifact["profile"]
        == get_material_backend_profile(backend_kind).to_dict()
    )
    physical = load_physical_rollout_archive(first / PHYSICAL_ARCHIVE_FILENAME)
    assert set(physical) == PHYSICAL_ROLLOUT_ARRAY_NAMES
    expected = _raw_arrays()["driven_material_positions_m"][:, [4, 0, 2]]
    np.testing.assert_array_equal(physical["prediction_m"], expected)
    np.testing.assert_array_equal(physical["driven_readout_m"], expected)
    for relative in (
        ARTIFACT_FILENAME,
        PHYSICAL_ARCHIVE_FILENAME,
        f"provenance/{RAW_ARCHIVE_FILENAME}",
        f"provenance/{RUNTIME_FILENAME}",
        "SHA256SUMS",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("engine_repository", "google-deepmind/mujoco", "repository does not match"),
        ("solver_family", "material-point-method", "solver family does not match"),
        ("identity_kind", "flex-vertex-index", "identity kind does not match"),
        ("coordinate_frame", "right-handed-y-up-world-v1", "canonical"),
        ("position_units", "mm", "units must be metres"),
    ),
)
def test_runtime_profile_fields_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    payload = _runtime_payload(raw_path, "sofa-fem-v1")
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_material_runtime_manifest(payload)


def test_valid_runtime_without_source_path_and_invalid_scalar_contracts(
    tmp_path: Path,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    valid = _runtime_payload(raw_path, "sofa-fem-v1")
    assert (
        validate_material_runtime_manifest(valid)["runtime_id"] == valid["runtime_id"]
    )

    invalid_cases = (
        ("simulation", [], "JSON object"),
        ("frame_count", 0, "positive integer"),
        ("time_step_s", "fast", "finite positive number"),
        ("time_step_s", float("inf"), "finite positive number"),
    )
    for field, value, message in invalid_cases:
        payload = _runtime_payload(raw_path, "sofa-fem-v1")
        payload[field] = value
        with pytest.raises(ValueError, match=message):
            validate_material_runtime_manifest(payload)

    with pytest.raises(ValueError, match="ordinary non-symlink file"):
        validate_material_runtime_manifest(valid, raw_rollout_path=tmp_path)


def test_runtime_rejects_nonfinite_parameters_and_future_information(
    tmp_path: Path,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    nonfinite = _runtime_payload(raw_path, "genesis-mpm-v1")
    nonfinite["simulation"]["engine_parameters"]["young_modulus_pa"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_material_runtime_manifest(nonfinite)

    changed = _runtime_payload(raw_path, "genesis-mpm-v1")
    changed["information_boundary"]["future_outcomes_read"] = True
    with pytest.raises(ValueError, match="information boundary changed"):
        validate_material_runtime_manifest(changed)


@pytest.mark.parametrize("case", ("duplicate", "outside", "support", "frame-zero"))
def test_material_identity_arrays_fail_closed(tmp_path: Path, case: str) -> None:
    arrays = _raw_arrays()
    message = ""
    if case == "duplicate":
        arrays["material_query_indices"] = np.array([1, 1], dtype=np.int64)
        arrays["action_support"] = np.ones(2, dtype=np.float32)
        message = "must be unique"
    elif case == "outside":
        arrays["material_query_indices"] = np.array([1, 9], dtype=np.int64)
        arrays["action_support"] = np.ones(2, dtype=np.float32)
        message = "exceeds state count"
    elif case == "support":
        arrays["action_support"] = np.array([1.1, 0.0, 0.5], dtype=np.float32)
        message = "action_support is invalid"
    else:
        arrays["zero_action_material_positions_m"] = arrays[
            "zero_action_material_positions_m"
        ].copy()
        arrays["zero_action_material_positions_m"][0, 0, 0] += 0.001
        message = "differ at frame zero"
    path = _raw_archive(tmp_path / f"{case}.npz", arrays)
    with pytest.raises(ValueError, match=message):
        load_material_trajectory_rollout(path)


def test_bundle_detects_count_provenance_roster_and_profile_tampering(
    tmp_path: Path,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(
        tmp_path / "runtime.json", raw_path, "mujoco-flex-v1"
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["state_count"] += 1
    _rehash(runtime, "runtime_id")
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime state count differs"):
        materialize_material_trajectory_backend(
            raw_rollout_path=raw_path,
            runtime_manifest_path=runtime_path,
            output_dir=tmp_path / "count",
        )

    runtime_path.unlink()
    _runtime_manifest(runtime_path, raw_path, "mujoco-flex-v1")
    output = tmp_path / "output"
    materialize_material_trajectory_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output,
    )
    provenance = output / "provenance" / RAW_ARCHIVE_FILENAME
    provenance.write_bytes(provenance.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte count changed"):
        validate_material_trajectory_backend(output)

    output2 = tmp_path / "output2"
    materialize_material_trajectory_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output2,
    )
    (output2 / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file roster changed"):
        validate_material_trajectory_backend(output2)

    output3 = tmp_path / "output3"
    materialize_material_trajectory_backend(
        raw_rollout_path=raw_path,
        runtime_manifest_path=runtime_path,
        output_dir=output3,
    )
    artifact_path = output3 / ARTIFACT_FILENAME
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["profile"]["identity_kind"] = "changed"
    _rehash(artifact, "artifact_id")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="backend profile changed"):
        validate_material_trajectory_backend(output3)


def test_cli_profiles_are_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profiles"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "bayesian-phystwin.material-backend-registry"
    assert [record["profile_id"] for record in payload["profiles"]] == [
        "jax-fem-quasistatic-v1",
        "sofa-fem-v1",
        "genesis-mpm-v1",
        "mujoco-flex-v1",
    ]


def test_cli_materialize_validate_and_module_entrypoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = _raw_archive(tmp_path / "raw.npz")
    runtime_path = _runtime_manifest(tmp_path / "runtime.json", raw_path, "sofa-fem-v1")
    output = tmp_path / "output"

    assert main(["materialize", str(raw_path), str(runtime_path), str(output)]) == 0
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["backend_kind"] == "sofa-fem-v1"

    assert main(["validate", str(output)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["artifact_id"] == materialized["artifact_id"]

    module_path = Path(sys.modules[main.__module__].__file__ or "")
    monkeypatch.setattr(sys, "argv", ["material_trajectory_backend", "profiles"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(module_path), run_name="__main__")
    assert exit_info.value.code == 0
    module_payload = json.loads(capsys.readouterr().out)
    assert module_payload["schema"] == "bayesian-phystwin.material-backend-registry"
    assert len(module_payload["profiles"]) == 4
