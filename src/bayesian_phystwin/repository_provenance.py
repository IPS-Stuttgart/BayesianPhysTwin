"""Exact repository and runtime provenance for Bayesian-PhysTwin runs."""

from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
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
_ENVIRONMENT_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUNTIME_BASE_FIELDS = frozenset(
    {
        "python_implementation",
        "python_version",
        "python_compiler",
        "operating_system",
        "machine",
        "processor",
        "byte_order",
        "selected_environment",
    }
)


def _strict_json_value(
    value: object,
    *,
    name: str,
    path: str,
    active_containers: set[int],
) -> Any:
    """Return a detached JSON value without coercing keys or scalar subclasses."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number at {path}")
        return value

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{name} contains a circular mapping at {path}")
        active_containers.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(
                        f"{name} requires genuine string keys at {path}; "
                        f"received {type(key).__name__}"
                    )
                result[key] = _strict_json_value(
                    item,
                    name=name,
                    path=f"{path}.{key}",
                    active_containers=active_containers,
                )
            return {key: result[key] for key in sorted(result)}
        finally:
            active_containers.remove(identity)

    if type(value) in {list, tuple}:
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{name} contains a circular sequence at {path}")
        active_containers.add(identity)
        try:
            return [
                _strict_json_value(
                    item,
                    name=name,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)

    raise ValueError(
        f"{name} contains a non-JSON value at {path}: {type(value).__name__}"
    )


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    result = _strict_json_value(
        value,
        name=name,
        path="$",
        active_containers=set(),
    )
    if not isinstance(result, dict):
        raise AssertionError("mapping validation did not return a dictionary")
    return result


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


def _environment_variable_names(values: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        if type(value) is not str or _ENVIRONMENT_VARIABLE.fullmatch(value) is None:
            raise ValueError(
                "environment variable names must be canonical identifiers"
            )
        names.append(value)
    return tuple(sorted(set(names)))


def default_runtime_environment(
    *,
    overrides: Mapping[str, Any] | None = None,
    environment_variables: Sequence[str] = (),
) -> dict[str, Any]:
    """Return portable runtime metadata without collecting arbitrary secrets."""

    selected_environment: dict[str, str] = {}
    for name in _environment_variable_names(environment_variables):
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
        additional = _json_mapping(overrides, name="runtime overrides")
        collisions = sorted(_RUNTIME_BASE_FIELDS & additional.keys())
        if collisions:
            raise ValueError(
                "runtime overrides cannot replace inferred fields: "
                + ", ".join(collisions)
            )
        result.update(additional)
    return _json_mapping(result, name="runtime environment")


__all__ = [
    "RepositoryRole",
    "RepositoryState",
    "default_runtime_environment",
    "discover_git_repository_state",
    "normalize_github_repository",
    "validate_revision",
]
