"""Exact repository and runtime provenance for Bayesian-PhysTwin runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

RepositoryRole = Literal[
    "primary",
    "upstream",
    "observation",
    "downstream",
    "paper",
    "environment",
    "dependency",
]
_VALID_REPOSITORY_ROLES = frozenset(
    {
        "primary",
        "upstream",
        "observation",
        "downstream",
        "paper",
        "environment",
        "dependency",
    }
)


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return cast(
            dict[str, Any],
            json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False)),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def validate_revision(value: str, *, name: str = "revision") -> str:
    """Require an exact lowercase 40-character Git commit."""

    normalized = str(value).lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be an exact 40-character Git revision")
    return normalized


def normalize_github_repository(remote_url: str) -> str:
    """Normalize a GitHub HTTPS/SSH remote to ``owner/repository``."""

    value = str(remote_url).strip()
    if not value:
        raise ValueError("Git remote URL must be nonempty")
    if value.startswith("git@github.com:"):
        path = value.split(":", 1)[1]
    elif "github.com/" in value:
        path = value.split("github.com/", 1)[1]
    else:
        raise ValueError("only github.com repository remotes are supported")
    path = path.removesuffix(".git").strip("/")
    parts = path.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError("GitHub remote must identify exactly owner/repository")
    return "/".join(parts)


@dataclass(frozen=True)
class RepositoryState:
    """Exact state of one repository participating in a result."""

    repository: str
    revision: str
    dirty: bool
    role: RepositoryRole

    def __post_init__(self) -> None:
        repository = str(self.repository).strip()
        role = str(self.role)
        parts = repository.split("/")
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError("repository must use owner/name form")
        if role not in _VALID_REPOSITORY_ROLES:
            raise ValueError("unknown repository role")
        if not isinstance(self.dirty, bool):
            raise ValueError("dirty must be boolean")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(
            self,
            "revision",
            validate_revision(self.revision, name=f"{repository} revision"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "dirty": self.dirty,
            "role": self.role,
        }


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect Git repository at {root}") from error
    return completed.stdout.strip()


def discover_git_repository_state(
    root: str | Path,
    *,
    repository: str | None = None,
    role: RepositoryRole = "primary",
) -> RepositoryState:
    """Read the exact revision and dirty state from a local Git checkout."""

    checkout = Path(root).resolve()
    revision = _git_output(checkout, "rev-parse", "HEAD")
    status = _git_output(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    resolved_repository = repository
    if resolved_repository is None:
        remote = _git_output(checkout, "config", "--get", "remote.origin.url")
        resolved_repository = normalize_github_repository(remote)
    return RepositoryState(
        repository=resolved_repository,
        revision=revision,
        dirty=bool(status),
        role=role,
    )


def default_runtime_environment(
    *,
    overrides: Mapping[str, Any] | None = None,
    environment_variables: Sequence[str] = (),
) -> dict[str, Any]:
    """Return portable runtime metadata without collecting arbitrary secrets."""

    selected_environment: dict[str, str] = {}
    for name in sorted(set(map(str, environment_variables))):
        if not name:
            raise ValueError("environment variable names must be nonempty")
        if name in os.environ:
            selected_environment[name] = os.environ[name]
    result: dict[str, Any] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byte_order": sys.byteorder,
        "selected_environment": selected_environment,
    }
    if overrides is not None:
        result.update(_json_mapping(overrides, name="runtime overrides"))
    return _json_mapping(result, name="runtime environment")


__all__ = [
    "RepositoryRole",
    "RepositoryState",
    "default_runtime_environment",
    "discover_git_repository_state",
    "normalize_github_repository",
    "validate_revision",
]
