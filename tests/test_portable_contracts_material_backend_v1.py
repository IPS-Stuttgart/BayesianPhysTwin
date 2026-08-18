from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import bayesian_phystwin.material_backend_v1 as backend
from bayesian_phystwin.cli import lagrangian_backend, material_trajectory_backend
from bayesian_phystwin.material_trajectory_backend_v1 import (
    MATERIAL_BACKEND_PROFILES as MATERIAL_TRAJECTORY_PROFILES,
)


def _runtime(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _variant(
    producer_profile_id: str = "test-profile-v1",
    *,
    transport: backend.BackendTransportV1 = "material-trajectory-v1",
    legacy: bool = False,
) -> backend.MaterialBackendVariantV1:
    return backend.MaterialBackendVariantV1(
        producer_profile_id=producer_profile_id,
        transport=transport,
        legacy=legacy,
    )


def _spec(
    profile_id: str = "test-family-v1",
    *,
    priority: int = 1,
    maturity: backend.BackendMaturityV1 = "supported",
    variants: tuple[backend.MaterialBackendVariantV1, ...] | None = None,
) -> backend.MaterialBackendSpecV1:
    return backend.MaterialBackendSpecV1(
        profile_id=profile_id,
        engine_repository="example/engine",
        solver_family="test-solver",
        identity_kind="test-node",
        priority=priority,
        maturity=maturity,
        variants=(_variant(),) if variants is None else variants,
    )


def test_registry_consolidates_duplicate_genesis_profiles() -> None:
    record = backend.describe_material_backend_profiles()
    profiles = record["profiles"]
    assert isinstance(profiles, list)
    assert [item["profile_id"] for item in profiles] == [
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

    canonical = backend.resolve_material_backend_profile("genesis-mpm-v1")
    legacy = backend.resolve_material_backend_profile("genesis-world-mpm-v1")
    assert canonical.profile_id == legacy.profile_id == "genesis-mpm-v1"
    assert canonical.producer_profile_id == "genesis-mpm-v1"
    assert legacy.producer_profile_id == "genesis-world-mpm-v1"
    assert canonical.transport == "material-trajectory-v1"
    assert not canonical.legacy_alias
    assert legacy.transport == "lagrangian-export-v1"
    assert legacy.legacy_alias
    assert legacy.to_record()["selected_variant"]["legacy"] is True

    drake = backend.resolve_material_backend_profile("drake-fem-v1")
    assert drake.spec.engine_repository == "RobotLocomotion/drake"
    assert drake.spec.solver_family == "finite-element-method"
    assert drake.spec.identity_kind == "deformable-body-vertex-index"
    assert drake.spec.maturity == "experimental"
    assert drake.transport == "material-trajectory-v1"


def test_material_transport_profiles_match_canonical_registry() -> None:
    material_specs = {
        spec.profile_id: spec
        for spec in backend.MATERIAL_BACKEND_SPECS.values()
        if any(
            variant.transport == "material-trajectory-v1" and not variant.legacy
            for variant in spec.variants
        )
    }
    assert set(material_specs) == set(MATERIAL_TRAJECTORY_PROFILES)
    for profile_id, profile in MATERIAL_TRAJECTORY_PROFILES.items():
        spec = material_specs[profile_id]
        assert spec.engine_repository == profile.engine_repository
        assert spec.solver_family == profile.solver_family
        assert spec.identity_kind == profile.identity_kind


@pytest.mark.parametrize(
    ("producer_profile_id", "transport", "legacy", "error", "message"),
    [
        (
            "",
            "material-trajectory-v1",
            False,
            ValueError,
            "canonical string",
        ),
        (
            " padded ",
            "material-trajectory-v1",
            False,
            ValueError,
            "canonical string",
        ),
        (
            cast(str, 7),
            "material-trajectory-v1",
            False,
            ValueError,
            "canonical string",
        ),
        (
            "profile-v1",
            cast(backend.BackendTransportV1, "unknown-transport"),
            False,
            ValueError,
            "unsupported backend transport",
        ),
        (
            "profile-v1",
            "material-trajectory-v1",
            cast(bool, 1),
            TypeError,
            "genuine bool",
        ),
    ],
)
def test_backend_variant_validation_fails_closed(
    producer_profile_id: str,
    transport: backend.BackendTransportV1,
    legacy: bool,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        backend.MaterialBackendVariantV1(
            producer_profile_id=producer_profile_id,
            transport=transport,
            legacy=legacy,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: backend.MaterialBackendSpecV1(
                profile_id="",
                engine_repository="example/engine",
                solver_family="solver",
                identity_kind="node",
                priority=1,
                maturity="supported",
                variants=(_variant(),),
            ),
            "profile_id must be a canonical string",
        ),
        (
            lambda: _spec(priority=0),
            "priority must be a positive integer",
        ),
        (
            lambda: _spec(maturity=cast(backend.BackendMaturityV1, "unknown")),
            "unsupported backend maturity",
        ),
        (
            lambda: _spec(variants=()),
            "at least one backend variant",
        ),
        (
            lambda: _spec(variants=(_variant(), _variant())),
            "producer profile IDs must be unique",
        ),
        (
            lambda: _spec(variants=(_variant(legacy=True),)),
            "requires a non-legacy variant",
        ),
    ],
)
def test_backend_spec_validation_fails_closed(
    factory: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_producer_index_rejects_duplicates_across_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _variant("shared-profile-v1")
    first = _spec("first-family-v1", variants=(shared,))
    second = _spec("second-family-v1", variants=(shared,))
    monkeypatch.setattr(
        backend,
        "MATERIAL_BACKEND_SPECS",
        {first.profile_id: first, second.profile_id: second},
    )
    with pytest.raises(RuntimeError, match="duplicate producer profile"):
        backend._producer_index()


@pytest.mark.parametrize("profile_id", ["", cast(str, 3)])
def test_resolve_rejects_noncanonical_profile_ids(profile_id: str) -> None:
    with pytest.raises(ValueError, match="nonempty string"):
        backend.resolve_material_backend_profile(profile_id)


def test_resolve_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown material backend profile"):
        backend.resolve_material_backend_profile("unknown-profile-v1")


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


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "backend_profile": "jax-fem-quasistatic-v1",
            "backend_kind": "sofa-fem-v1",
        },
    ],
)
def test_runtime_selection_requires_exactly_one_transport_key(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    runtime = _runtime(tmp_path, payload)
    with pytest.raises(ValueError, match="exactly one"):
        backend._runtime_selection(runtime)


def test_runtime_selection_requires_string_profile(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, {"backend_kind": 3})
    with pytest.raises(ValueError, match="must be a string"):
        backend._runtime_selection(runtime)


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


def test_legacy_cli_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_name = "bayesian_phystwin.cli.material_trajectory_backend"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["material_trajectory_backend", "profiles"],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "bayesian-phystwin.material-backend-registry"
