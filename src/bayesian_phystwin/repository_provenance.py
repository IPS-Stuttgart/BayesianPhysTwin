"""Exact repository and runtime provenance for Bayesian-PhysTwin runs."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import SplitResult, urlsplit

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
_GITHUB_OWNER = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$"
)
_GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_GITHUB_SCP_REMOTE = re.compile(r"^git@github\.com:(?P<path>[^?#]+)$")


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

    if type(value) is not str or value != value.strip() or value != value.lower():
        raise ValueError(f"{name} must be an exact lowercase Git revision")
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be an exact lowercase Git revision")
    return value


def _canonical_repository(value: object, *, name: str = "repository") -> str:
    if type(value) is not str or value != value.strip():
        raise ValueError(f"{name} must use canonical owner/name form")
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError(f"{name} must use canonical owner/name form")
    owner, repository = parts
    if _GITHUB_OWNER.fullmatch(owner) is None:
        raise ValueError(f"{name} contains an invalid GitHub owner")
    if (
        _GITHUB_REPOSITORY.fullmatch(repository) is None
        or repository in {".", ".."}
    ):
        raise ValueError(f"{name} contains an invalid GitHub repository name")
    return value


def _remote_path_from_url(parsed: SplitResult) -> str:
    if parsed.query or parsed.fragment:
        raise ValueError("Git remote URL must not contain a query or fragment")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Git remote URL contains an invalid port") from error

    hostname = parsed.hostname.lower() if parsed.hostname is not None else None
    if parsed.scheme == "https":
        if (
            hostname != "github.com"
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
        ):
            raise ValueError("HTTPS Git remote must use exactly github.com")
    elif parsed.scheme == "ssh":
        if (
            hostname != "github.com"
            or parsed.username != "git"
            or parsed.password is not None
            or port not in {None, 22}
        ):
            raise ValueError("SSH Git remote must use git@github.com")
    else:
        raise ValueError("only GitHub HTTPS and SSH repository remotes are supported")
    return parsed.path.removeprefix("/")


def normalize_github_repository(remote_url: str) -> str:
    """Normalize a GitHub HTTPS/SSH remote to canonical ``owner/repository``."""

    if type(remote_url) is not str or not remote_url or remote_url != remote_url.strip():
        raise ValueError("Git remote URL must be a canonical nonempty string")

    scp_match = _GITHUB_SCP_REMOTE.fullmatch(remote_url)
    if scp_match is not None:
        path = scp_match.group("path")
    else:
        path = _remote_path_from_url(urlsplit(remote_url))

    repository = path.removesuffix(".git").strip("/")
    if repository != path.removesuffix(".git"):
        raise ValueError("Git remote path must not contain surrounding slashes")
    return _canonical_repository(repository, name="GitHub remote")


@dataclass(frozen=True)
class RepositoryState:
    """Exact state of one repository participating in a result."""

    repository: str
    revision: str
    dirty: bool
    role: RepositoryRole

    def __post_init__(self) -> None:
        repository = _canonical_repository(self.repository)
        if type(self.role) is not str or self.role not in _VALID_REPOSITORY_ROLES:
            raise ValueError("unknown repository role")
        if type(self.dirty) is not bool:
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
