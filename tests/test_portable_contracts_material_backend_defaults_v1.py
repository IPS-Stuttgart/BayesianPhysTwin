from __future__ import annotations

from types import MappingProxyType
from typing import cast

import pytest

import bayesian_phystwin.material_backend_v1 as backend


def _variant(
    producer_profile_id: str,
    *,
    transport: backend.BackendTransportV1 = "material-trajectory-v1",
    legacy: bool = False,
    default: bool | None = None,
) -> backend.MaterialBackendVariantV1:
    return backend.MaterialBackendVariantV1(
        producer_profile_id=producer_profile_id,
        transport=transport,
        legacy=legacy,
        default=default,
    )


def _spec(
    profile_id: str,
    variants: tuple[backend.MaterialBackendVariantV1, ...],
) -> backend.MaterialBackendSpecV1:
    return backend.MaterialBackendSpecV1(
        profile_id=profile_id,
        engine_repository="example/engine",
        solver_family="test-solver",
        identity_kind="test-node",
        priority=1,
        maturity="supported",
        variants=variants,
    )


def test_variant_default_normalization_preserves_existing_callers() -> None:
    current = _variant("current-v1")
    retained = _variant("legacy-v1", legacy=True)
    secondary = _variant("secondary-v1", default=False)

    assert current.default is True
    assert retained.default is False
    assert secondary.default is False
    assert current.to_record()["default"] is True
    assert retained.to_record()["default"] is False

    with pytest.raises(TypeError, match="default must be a genuine bool"):
        _variant("invalid-v1", default=cast(bool, 1))
    with pytest.raises(ValueError, match="legacy backend variant cannot be"):
        _variant("invalid-legacy-v1", legacy=True, default=True)


def test_explicit_default_allows_multiple_current_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secondary = _variant(
        "test-family-secondary-v1",
        transport="lagrangian-export-v1",
        default=False,
    )
    selected = _variant(
        "test-family-primary-v1",
        transport="material-trajectory-v1",
        default=True,
    )
    spec = _spec("test-family-v1", (secondary, selected))

    monkeypatch.setattr(
        backend,
        "MATERIAL_BACKEND_SPECS",
        MappingProxyType({spec.profile_id: spec}),
    )
    monkeypatch.setattr(
        backend,
        "_PRODUCER_INDEX",
        MappingProxyType(backend._producer_index()),
    )

    canonical = backend.resolve_material_backend_profile("test-family-v1")
    exact_secondary = backend.resolve_material_backend_profile(
        "test-family-secondary-v1"
    )
    assert canonical.producer_profile_id == "test-family-primary-v1"
    assert canonical.transport == "material-trajectory-v1"
    assert canonical.is_default
    assert exact_secondary.transport == "lagrangian-export-v1"
    assert not exact_secondary.legacy_alias
    assert not exact_secondary.is_default

    with pytest.raises(ValueError, match="producer profile does not match"):
        backend._assert_requested_profile(
            requested_profile_id="test-family-secondary-v1",
            resolved_runtime=canonical,
        )


@pytest.mark.parametrize(
    "variants",
    [
        (
            _variant("first-v1", default=False),
            _variant("second-v1", default=False),
        ),
        (
            _variant("first-v1", default=True),
            _variant("second-v1", default=True),
        ),
    ],
)
def test_backend_family_requires_exactly_one_explicit_default(
    variants: tuple[backend.MaterialBackendVariantV1, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly one non-legacy default"):
        _spec("ambiguous-family-v1", variants)


def test_nondefault_variant_cannot_shadow_its_canonical_family() -> None:
    with pytest.raises(ValueError, match="may only identify the default"):
        _spec(
            "canonical-family-v1",
            (
                _variant("primary-producer-v1", default=True),
                _variant("canonical-family-v1", default=False),
            ),
        )


def test_producer_index_rejects_registry_key_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec("canonical-family-v1", (_variant("canonical-family-v1"),))
    monkeypatch.setattr(
        backend,
        "MATERIAL_BACKEND_SPECS",
        {"wrong-registry-key-v1": spec},
    )
    with pytest.raises(RuntimeError, match="registry key must equal"):
        backend._producer_index()


def test_producer_index_rejects_canonical_namespace_shadowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _spec("first-family-v1", (_variant("first-producer-v1"),))
    second = _spec("second-family-v1", (_variant("first-family-v1"),))
    monkeypatch.setattr(
        backend,
        "MATERIAL_BACKEND_SPECS",
        {
            first.profile_id: first,
            second.profile_id: second,
        },
    )
    with pytest.raises(RuntimeError, match="collides with another canonical"):
        backend._producer_index()


def test_registry_description_exposes_one_default_per_family() -> None:
    record = backend.describe_material_backend_profiles()
    assert record["schema_version"] == 2
    profiles = record["profiles"]
    assert isinstance(profiles, list)
    for profile in profiles:
        variants = profile["variants"]
        defaults = [variant for variant in variants if variant["default"]]
        assert len(defaults) == 1
        assert defaults[0]["legacy"] is False
