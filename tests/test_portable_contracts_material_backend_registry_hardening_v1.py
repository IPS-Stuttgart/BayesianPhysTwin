from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import bayesian_phystwin.material_backend_v1 as backend


def _runtime(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _variant(
    producer_profile_id: str,
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
    profile_id: str,
    *,
    priority: int = 1,
    variants: tuple[backend.MaterialBackendVariantV1, ...],
) -> backend.MaterialBackendSpecV1:
    return backend.MaterialBackendSpecV1(
        profile_id=profile_id,
        engine_repository="example/engine",
        solver_family="test-solver",
        identity_kind="test-node",
        priority=priority,
        maturity="supported",
        variants=variants,
    )


def test_registry_requires_exactly_one_nonlegacy_default_variant() -> None:
    with pytest.raises(ValueError, match="exactly one non-legacy default"):
        _spec(
            "ambiguous-family-v1",
            variants=(
                _variant("ambiguous-a-v1"),
                _variant("ambiguous-b-v1"),
            ),
        )


def test_registry_priority_ties_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second = _spec(
        "z-family-v1",
        variants=(_variant("z-family-v1"),),
    )
    first = _spec(
        "a-family-v1",
        variants=(_variant("a-family-v1"),),
    )
    monkeypatch.setattr(
        backend,
        "MATERIAL_BACKEND_SPECS",
        {second.profile_id: second, first.profile_id: first},
    )
    record = backend.describe_material_backend_profiles()
    profiles = record["profiles"]
    assert isinstance(profiles, list)
    assert [item["profile_id"] for item in profiles] == [
        "a-family-v1",
        "z-family-v1",
    ]


def test_transport_specific_profile_assertion_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def materialize(**kwargs: Any) -> dict[str, object]:
        nonlocal called
        called = True
        return {"unexpected": kwargs}

    monkeypatch.setattr(
        backend,
        "materialize_material_trajectory_backend",
        materialize,
    )
    raw = tmp_path / "raw.npz"
    raw.write_bytes(b"not-read")
    runtime = _runtime(tmp_path, {"backend_kind": "genesis-mpm-v1"})

    with pytest.raises(ValueError, match="requested producer profile"):
        backend.materialize_material_backend(
            raw_rollout_path=raw,
            runtime_manifest_path=runtime,
            output_dir=tmp_path / "output",
            profile_id="genesis-world-mpm-v1",
        )
    assert not called
