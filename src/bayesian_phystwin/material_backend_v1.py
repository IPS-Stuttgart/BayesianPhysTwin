"""Canonical registry and dispatcher for external deformable backends.

Transport-specific version-1 facades remain available for immutable artifacts.
New integrations should select a canonical backend family through this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from ._portable_contracts import load_strict_json_object
from .lagrangian_backend_v1 import (
    ARTIFACT_FILENAME as LAGRANGIAN_ARTIFACT_FILENAME,
)
from .lagrangian_backend_v1 import (
    materialize_lagrangian_backend,
    validate_lagrangian_backend,
)
from .material_trajectory_backend_v1 import (
    ARTIFACT_FILENAME as MATERIAL_TRAJECTORY_ARTIFACT_FILENAME,
)
from .material_trajectory_backend_v1 import (
    materialize_material_trajectory_backend,
    validate_material_trajectory_backend,
)

BackendTransportV1 = Literal["lagrangian-export-v1", "material-trajectory-v1"]
BackendMaturityV1 = Literal["preferred", "supported", "experimental"]


@dataclass(frozen=True, slots=True)
class MaterialBackendVariantV1:
    """One transport-specific producer profile for a canonical backend family."""

    producer_profile_id: str
    transport: BackendTransportV1
    legacy: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.producer_profile_id) is not str
            or not self.producer_profile_id
            or self.producer_profile_id.strip() != self.producer_profile_id
        ):
            raise ValueError("producer_profile_id must be a canonical string")
        if self.transport not in {
            "lagrangian-export-v1",
            "material-trajectory-v1",
        }:
            raise ValueError("unsupported backend transport")
        if type(self.legacy) is not bool:
            raise TypeError("legacy must be a genuine bool")

    def to_record(self) -> dict[str, object]:
        return {
            "producer_profile_id": self.producer_profile_id,
            "transport": self.transport,
            "legacy": self.legacy,
        }


@dataclass(frozen=True, slots=True)
class MaterialBackendSpecV1:
    """Canonical backend family independent of its artifact transport."""

    profile_id: str
    engine_repository: str
    solver_family: str
    identity_kind: str
    priority: int
    maturity: BackendMaturityV1
    variants: tuple[MaterialBackendVariantV1, ...]

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "engine_repository",
            "solver_family",
            "identity_kind",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{name} must be a canonical string")
        if type(self.priority) is not int or self.priority < 1:
            raise ValueError("priority must be a positive integer")
        if self.maturity not in {"preferred", "supported", "experimental"}:
            raise ValueError("unsupported backend maturity")
        if not self.variants:
            raise ValueError("at least one backend variant is required")
        producer_ids = tuple(item.producer_profile_id for item in self.variants)
        if len(set(producer_ids)) != len(producer_ids):
            raise ValueError("backend producer profile IDs must be unique")
        default_count = sum(not item.legacy for item in self.variants)
        if default_count != 1:
            raise ValueError(
                "a backend family requires exactly one non-legacy default variant"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "engine_repository": self.engine_repository,
            "solver_family": self.solver_family,
            "identity_kind": self.identity_kind,
            "priority": self.priority,
            "maturity": self.maturity,
            "variants": [item.to_record() for item in self.variants],
        }


MATERIAL_BACKEND_SPECS: Final[Mapping[str, MaterialBackendSpecV1]] = MappingProxyType(
    {
        "jax-fem-quasistatic-v1": MaterialBackendSpecV1(
            profile_id="jax-fem-quasistatic-v1",
            engine_repository="deepmodeling/jax-fem",
            solver_family="differentiable-fem",
            identity_kind="mesh-node",
            priority=1,
            maturity="preferred",
            variants=(
                MaterialBackendVariantV1(
                    producer_profile_id="jax-fem-quasistatic-v1",
                    transport="lagrangian-export-v1",
                ),
            ),
        ),
        "sofa-fem-v1": MaterialBackendSpecV1(
            profile_id="sofa-fem-v1",
            engine_repository="sofa-framework/sofa",
            solver_family="finite-element-method",
            identity_kind="mechanical-node-index",
            priority=2,
            maturity="supported",
            variants=(
                MaterialBackendVariantV1(
                    producer_profile_id="sofa-fem-v1",
                    transport="material-trajectory-v1",
                ),
            ),
        ),
        "genesis-mpm-v1": MaterialBackendSpecV1(
            profile_id="genesis-mpm-v1",
            engine_repository="Genesis-Embodied-AI/genesis-world",
            solver_family="material-point-method",
            identity_kind="material-particle-index",
            priority=3,
            maturity="supported",
            variants=(
                MaterialBackendVariantV1(
                    producer_profile_id="genesis-mpm-v1",
                    transport="material-trajectory-v1",
                ),
                MaterialBackendVariantV1(
                    producer_profile_id="genesis-world-mpm-v1",
                    transport="lagrangian-export-v1",
                    legacy=True,
                ),
            ),
        ),
        "mujoco-flex-v1": MaterialBackendSpecV1(
            profile_id="mujoco-flex-v1",
            engine_repository="google-deepmind/mujoco",
            solver_family="mujoco-flex",
            identity_kind="flex-vertex-index",
            priority=4,
            maturity="experimental",
            variants=(
                MaterialBackendVariantV1(
                    producer_profile_id="mujoco-flex-v1",
                    transport="material-trajectory-v1",
                ),
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedMaterialBackendV1:
    """Canonical family plus the exact producer transport used by an artifact."""

    spec: MaterialBackendSpecV1
    variant: MaterialBackendVariantV1

    @property
    def profile_id(self) -> str:
        return self.spec.profile_id

    @property
    def producer_profile_id(self) -> str:
        return self.variant.producer_profile_id

    @property
    def transport(self) -> BackendTransportV1:
        return self.variant.transport

    @property
    def legacy_alias(self) -> bool:
        return self.variant.legacy

    def to_record(self) -> dict[str, object]:
        return {
            "canonical_profile": self.spec.to_record(),
            "selected_variant": self.variant.to_record(),
        }


def _producer_index() -> dict[str, ResolvedMaterialBackendV1]:
    result: dict[str, ResolvedMaterialBackendV1] = {}
    for spec in MATERIAL_BACKEND_SPECS.values():
        for variant in spec.variants:
            if variant.producer_profile_id in result:
                raise RuntimeError("duplicate producer profile in backend registry")
            result[variant.producer_profile_id] = ResolvedMaterialBackendV1(
                spec=spec,
                variant=variant,
            )
    return result


_PRODUCER_INDEX: Final = MappingProxyType(_producer_index())


def describe_material_backend_profiles() -> dict[str, object]:
    """Return the canonical, priority-ordered external-backend registry."""

    profiles = sorted(
        MATERIAL_BACKEND_SPECS.values(),
        key=lambda item: (item.priority, item.profile_id),
    )
    return {
        "schema": "bayesian-phystwin.material-backend-registry",
        "schema_version": 1,
        "profiles": [item.to_record() for item in profiles],
        "extension_rule": (
            "Add one canonical family and one transport variant; do not create a "
            "parallel public artifact family for an already registered engine."
        ),
    }


def resolve_material_backend_profile(profile_id: str) -> ResolvedMaterialBackendV1:
    """Resolve either a canonical family ID or an exact producer-profile ID."""

    if type(profile_id) is not str or not profile_id:
        raise ValueError("profile_id must be a nonempty string")
    spec = MATERIAL_BACKEND_SPECS.get(profile_id)
    if spec is not None:
        variant = next(item for item in spec.variants if not item.legacy)
        return ResolvedMaterialBackendV1(spec=spec, variant=variant)
    resolved = _PRODUCER_INDEX.get(profile_id)
    if resolved is None:
        raise ValueError(f"unknown material backend profile: {profile_id}")
    return resolved


def _runtime_selection(runtime_manifest_path: str | Path) -> ResolvedMaterialBackendV1:
    runtime = load_strict_json_object(
        runtime_manifest_path,
        label="material backend runtime manifest",
    )
    backend_profile = runtime.get("backend_profile")
    backend_kind = runtime.get("backend_kind")
    if (backend_profile is None) == (backend_kind is None):
        raise ValueError(
            "runtime manifest must declare exactly one of backend_profile or "
            "backend_kind"
        )
    selected = backend_profile if backend_profile is not None else backend_kind
    if type(selected) is not str:
        raise ValueError("runtime backend profile must be a string")
    resolved = resolve_material_backend_profile(selected)
    expected_transport: BackendTransportV1 = (
        "lagrangian-export-v1"
        if backend_profile is not None
        else "material-trajectory-v1"
    )
    if resolved.transport != expected_transport:
        raise ValueError("runtime profile and transport schema disagree")
    return resolved


def _assert_requested_profile(
    *,
    requested_profile_id: str,
    resolved_runtime: ResolvedMaterialBackendV1,
) -> None:
    """Check a canonical-family assertion or an exact legacy producer assertion."""

    requested = resolve_material_backend_profile(requested_profile_id)
    if requested.profile_id != resolved_runtime.profile_id:
        raise ValueError(
            "requested canonical backend does not match the runtime manifest"
        )
    # Canonical family IDs intentionally admit their registered transport variants.
    # A transport-specific alias is an exact producer assertion and must not
    # silently accept another transport from the same family.
    if (
        requested_profile_id != requested.profile_id
        and requested.producer_profile_id != resolved_runtime.producer_profile_id
    ):
        raise ValueError(
            "requested producer profile does not match the runtime manifest"
        )


def materialize_material_backend(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    output_dir: str | Path,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch one runtime manifest through its registered artifact transport."""

    resolved = _runtime_selection(runtime_manifest_path)
    if profile_id is not None:
        _assert_requested_profile(
            requested_profile_id=profile_id,
            resolved_runtime=resolved,
        )
    if resolved.transport == "lagrangian-export-v1":
        return materialize_lagrangian_backend(
            raw_rollout_path=raw_rollout_path,
            runtime_manifest_path=runtime_manifest_path,
            output_dir=output_dir,
        )
    return materialize_material_trajectory_backend(
        raw_rollout_path=raw_rollout_path,
        runtime_manifest_path=runtime_manifest_path,
        output_dir=output_dir,
    )


def validate_material_backend(output_dir: str | Path) -> dict[str, Any]:
    """Auto-detect and validate exactly one registered backend artifact family."""

    root = Path(output_dir)
    lagrangian = root / LAGRANGIAN_ARTIFACT_FILENAME
    material = root / MATERIAL_TRAJECTORY_ARTIFACT_FILENAME
    if lagrangian.exists() == material.exists():
        raise ValueError(
            "backend bundle must contain exactly one recognized artifact manifest"
        )
    if lagrangian.exists():
        return validate_lagrangian_backend(root)
    return validate_material_trajectory_backend(root)


__all__ = [
    "BackendMaturityV1",
    "BackendTransportV1",
    "LAGRANGIAN_ARTIFACT_FILENAME",
    "MATERIAL_BACKEND_SPECS",
    "MATERIAL_TRAJECTORY_ARTIFACT_FILENAME",
    "MaterialBackendSpecV1",
    "MaterialBackendVariantV1",
    "ResolvedMaterialBackendV1",
    "describe_material_backend_profiles",
    "materialize_material_backend",
    "resolve_material_backend_profile",
    "validate_material_backend",
]
