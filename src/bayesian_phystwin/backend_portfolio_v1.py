"""Machine-readable evidence-first budget for external physical backends.

The material backend registry records implementation compatibility. This module
records the independent evidence stage and constrains active qualification work
so adapter breadth cannot be mistaken for scientific progress.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from .material_backend_v1 import MATERIAL_BACKEND_SPECS

BackendEvidenceStageV1 = Literal[
    "registered-adapter",
    "native-smoke-passed",
    "source-physics-qualified",
    "source-value-qualified",
    "fresh-object-qualified",
    "downstream-causal-qualified",
]

BACKEND_PORTFOLIO_SCHEMA: Final = "bayesian-phystwin.backend-portfolio"
BACKEND_PORTFOLIO_VERSION: Final = 1
MAX_ACTIVE_QUALIFICATION_CANDIDATES: Final = 2
ACTIVE_QUALIFICATION_CANDIDATES: Final = (
    "jax-fem-quasistatic-v1",
    "genesis-mpm-v1",
)

# This is the admitted family roster when the evidence-first freeze started.
# A new family cannot enter the canonical registry while the freeze is active.
_ADMISSION_FREEZE_BASELINE: Final = frozenset(
    {
        "jax-fem-quasistatic-v1",
        "warp-fem-v1",
        "sofa-fem-v1",
        "genesis-mpm-v1",
        "position-based-dynamics-v1",
        "physx-fem-v1",
        "mujoco-flex-v1",
        "drake-fem-v1",
    }
)

_EVIDENCE_STAGE_BY_PROFILE: Final[Mapping[str, BackendEvidenceStageV1]] = (
    MappingProxyType(
        {
            "jax-fem-quasistatic-v1": "native-smoke-passed",
            "warp-fem-v1": "registered-adapter",
            "sofa-fem-v1": "registered-adapter",
            "genesis-mpm-v1": "source-physics-qualified",
            "position-based-dynamics-v1": "registered-adapter",
            "physx-fem-v1": "registered-adapter",
            "mujoco-flex-v1": "registered-adapter",
            "drake-fem-v1": "registered-adapter",
        }
    )
)

_STAGE_ORDER: Final[Mapping[BackendEvidenceStageV1, int]] = MappingProxyType(
    {
        "registered-adapter": 0,
        "native-smoke-passed": 1,
        "source-physics-qualified": 2,
        "source-value-qualified": 3,
        "fresh-object-qualified": 4,
        "downstream-causal-qualified": 5,
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def backend_evidence_stage(profile_id: str) -> BackendEvidenceStageV1:
    """Return the evidence stage independently of implementation maturity."""

    if type(profile_id) is not str or not profile_id:
        raise ValueError("profile_id must be a nonempty string")
    stage = _EVIDENCE_STAGE_BY_PROFILE.get(profile_id)
    if stage is None:
        raise ValueError(f"unknown backend evidence profile: {profile_id}")
    return stage


def validate_backend_portfolio() -> dict[str, object]:
    """Validate the frozen family roster and active qualification budget."""

    registry_ids = frozenset(MATERIAL_BACKEND_SPECS)
    stage_ids = frozenset(_EVIDENCE_STAGE_BY_PROFILE)
    _require(
        stage_ids == registry_ids,
        "every canonical backend family requires exactly one evidence stage",
    )

    active = tuple(ACTIVE_QUALIFICATION_CANDIDATES)
    _require(
        len(active) == len(set(active)),
        "active backend qualification candidates must be unique",
    )
    _require(
        len(active) <= MAX_ACTIVE_QUALIFICATION_CANDIDATES,
        "active backend qualification budget exceeded",
    )
    _require(
        set(active).issubset(registry_ids),
        "active backend qualification candidate is not registered",
    )
    for profile_id in active:
        stage = backend_evidence_stage(profile_id)
        _require(
            _STAGE_ORDER[stage] >= _STAGE_ORDER["native-smoke-passed"],
            "an active qualification candidate must pass its native smoke first",
        )
        _require(
            _STAGE_ORDER[stage] < _STAGE_ORDER["source-value-qualified"],
            "a source-value-qualified backend must leave the active source funnel",
        )

    source_value_profiles = tuple(
        sorted(
            profile_id
            for profile_id, stage in _EVIDENCE_STAGE_BY_PROFILE.items()
            if _STAGE_ORDER[stage] >= _STAGE_ORDER["source-value-qualified"]
        )
    )
    admission_frozen = not source_value_profiles
    if admission_frozen:
        _require(
            registry_ids == _ADMISSION_FREEZE_BASELINE,
            "new backend family admitted while evidence-first freeze is active",
        )

    profiles = []
    for profile_id, spec in sorted(
        MATERIAL_BACKEND_SPECS.items(),
        key=lambda item: (item[1].priority, item[0]),
    ):
        stage = backend_evidence_stage(profile_id)
        profiles.append(
            {
                "profile_id": profile_id,
                "implementation_maturity": spec.maturity,
                "evidence_stage": stage,
                "active_qualification_candidate": profile_id in active,
                "recommendation_authorized": (
                    _STAGE_ORDER[stage] >= _STAGE_ORDER["source-value-qualified"]
                ),
            }
        )

    return {
        "schema": BACKEND_PORTFOLIO_SCHEMA,
        "schema_version": BACKEND_PORTFOLIO_VERSION,
        "admission_frozen": admission_frozen,
        "new_family_admission_allowed": not admission_frozen,
        "maximum_active_qualification_candidates": (
            MAX_ACTIVE_QUALIFICATION_CANDIDATES
        ),
        "active_qualification_candidates": list(active),
        "source_value_qualified_profiles": list(source_value_profiles),
        "profiles": profiles,
        "policy_document": "docs/backend_admission_policy_v1.md",
    }


def describe_backend_portfolio() -> dict[str, object]:
    """Return the validated evidence-stage and work-in-progress snapshot."""

    return validate_backend_portfolio()


__all__ = [
    "ACTIVE_QUALIFICATION_CANDIDATES",
    "BACKEND_PORTFOLIO_SCHEMA",
    "BACKEND_PORTFOLIO_VERSION",
    "BackendEvidenceStageV1",
    "MAX_ACTIVE_QUALIFICATION_CANDIDATES",
    "backend_evidence_stage",
    "describe_backend_portfolio",
    "validate_backend_portfolio",
]
