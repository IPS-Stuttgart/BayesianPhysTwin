"""Shared metadata helpers for versioned cross-repository provider APIs."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution, version


def installed_distribution_version(distribution_name: str, *, fallback: str) -> str:
    """Return the installed distribution version or a source-tree fallback."""

    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return fallback


def installed_distribution_revision(distribution_name: str) -> str | None:
    """Return the VCS revision recorded by a direct-URL installation, if present."""

    try:
        direct_url = distribution(distribution_name).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    commit_id = payload.get("vcs_info", {}).get("commit_id")
    return str(commit_id) if commit_id else None


__all__ = [
    "installed_distribution_revision",
    "installed_distribution_version",
]
