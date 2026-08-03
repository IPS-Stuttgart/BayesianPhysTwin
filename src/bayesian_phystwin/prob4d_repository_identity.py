"""Canonical and frozen repository identities for Prob4D artifacts."""

from __future__ import annotations

PROB4D_SOURCE_REPOSITORY = "IPS-Stuttgart/Prob4D"
PROB4D_LEGACY_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROB4D_SOURCE_REPOSITORIES = frozenset(
    {
        PROB4D_SOURCE_REPOSITORY,
        PROB4D_LEGACY_SOURCE_REPOSITORY,
    }
)


def is_prob4d_source_repository(value: object) -> bool:
    """Return whether ``value`` is a supported canonical or frozen identity."""

    return isinstance(value, str) and value in PROB4D_SOURCE_REPOSITORIES


def prob4d_source_repository_is_legacy(value: object) -> bool:
    """Return whether ``value`` names the frozen pre-transfer repository."""

    return value == PROB4D_LEGACY_SOURCE_REPOSITORY


__all__ = [
    "PROB4D_LEGACY_SOURCE_REPOSITORY",
    "PROB4D_SOURCE_REPOSITORIES",
    "PROB4D_SOURCE_REPOSITORY",
    "is_prob4d_source_repository",
    "prob4d_source_repository_is_legacy",
]
