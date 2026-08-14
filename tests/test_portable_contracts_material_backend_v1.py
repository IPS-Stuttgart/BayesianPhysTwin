from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import bayesian_phystwin.material_backend_v1 as backend
from bayesian_phystwin.cli import lagrangian_backend, material_trajectory_backend


def _runtime(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_registry_consolidates_duplicate_genesis_profiles() -> None:
    record = backend.describe_material_backend_profiles()
    profiles = record["profiles"]
    assert isinstance(profiles, list)
    assert [item["profile_id"] for item in profiles] == [
        "jax-fem-quasistatic-v1",
        "sofa-fem-v1",
        "genesis-mpm-v1",
        "mujoco-flex-v1",
    ]

    canonical = backend.resolve_material_backend_profile("genesis-mpm-v1")
    legacy = backend.resolve_material_backend_profile("genesis-world-mpm-v1")
    assert canonical.profile_id == legacy.profile_id == "genesis-mpm-v1"
    assert canonical.transport == "material-trajectory-v1"
    assert not canonical.legacy_alias
    assert legacy.transport == "lagrangian-export-v1"
    assert legacy.legacy_alias


def test_materialize_dispatches_from_runtime_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def lagrangian(**kwargs: Any) -> dict[str, str]:
        calls.append(("lagrangian", kwargs))
        return {"artifact": "lagrangian"}

    def material(**kwargs: Any) -> dict[str, str]:
        calls.append(("material", kwargs))
        return {"artifact": "material"}

    monkeypatch.setattr(backend, "materialize_lagrangian_backend", lagrangian)
    monkeypatch.setattr(
        backend,
        "materialize_material_trajectory_backend",
        material,
    )
    raw = tmp_path / "raw.npz"
    raw.write_bytes(b"not-read-by-dispatch")

    legacy_runtime = _runtime(
        tmp_path,
        {"backend_profile": "genesis-world-mpm-v1"},
    )
    assert backend.materialize_material_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=legacy_runtime,
        output_dir=tmp_path / "legacy-output",
        profile_id="genesis-mpm-v1",
    ) == {"artifact": "lagrangian"}
    assert calls[-1][0] == "lagrangian"

    canonical_runtime = _runtime(
        tmp_path,
        {"backend_kind": "genesis-mpm-v1"},
    )
    assert backend.materialize_material_backend(
        raw_rollout_path=raw,
        runtime_manifest_path=canonical_runtime,
        output_dir=tmp_path / "canonical-output",
    ) == {"artifact": "material"}
    assert calls[-1][0] == "material"


def test_materialize_rejects_profile_and_transport_mismatches(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.npz"
    raw.write_bytes(b"unused")
    runtime = _runtime(tmp_path, {"backend_kind": "sofa-fem-v1"})
    with pytest.raises(ValueError, match="does not match"):
        backend.materialize_material_backend(
            raw_rollout_path=raw,
            runtime_manifest_path=runtime,
            output_dir=tmp_path / "output",
            profile_id="jax-fem-quasistatic-v1",
        )

    invalid = _runtime(
        tmp_path,
        {"backend_profile": "genesis-mpm-v1"},
    )
    with pytest.raises(ValueError, match="transport schema disagree"):
        backend.materialize_material_backend(
            raw_rollout_path=raw,
            runtime_manifest_path=invalid,
            output_dir=tmp_path / "invalid-output",
        )


def test_validate_auto_detects_one_artifact_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        backend,
        "validate_lagrangian_backend",
        lambda root: {"validated": "lagrangian", "root": str(root)},
    )
    monkeypatch.setattr(
        backend,
        "validate_material_trajectory_backend",
        lambda root: {"validated": "material", "root": str(root)},
    )

    lagrangian_root = tmp_path / "lagrangian"
    lagrangian_root.mkdir()
    (lagrangian_root / backend.LAGRANGIAN_ARTIFACT_FILENAME).touch()
    assert backend.validate_material_backend(lagrangian_root)["validated"] == (
        "lagrangian"
    )

    material_root = tmp_path / "material"
    material_root.mkdir()
    (material_root / backend.MATERIAL_TRAJECTORY_ARTIFACT_FILENAME).touch()
    assert backend.validate_material_backend(material_root)["validated"] == "material"

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    (ambiguous / backend.LAGRANGIAN_ARTIFACT_FILENAME).touch()
    (ambiguous / backend.MATERIAL_TRAJECTORY_ARTIFACT_FILENAME).touch()
    with pytest.raises(ValueError, match="exactly one"):
        backend.validate_material_backend(ambiguous)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        backend.validate_material_backend(empty)


def test_legacy_cli_module_routes_to_canonical_cli() -> None:
    assert material_trajectory_backend.main is lagrangian_backend.main
    assert material_trajectory_backend.build_parser is lagrangian_backend.build_parser
    args = lagrangian_backend.build_parser().parse_args(["profiles"])
    assert args.command == "profiles"
