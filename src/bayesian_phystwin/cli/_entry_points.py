"""Read and protect the frozen legacy console-script surface."""

from __future__ import annotations

import ast
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Final

LEGACY_SURFACE_VERSION: Final = 1
ROOT_TARGET: Final = "bayesian_phystwin.cli.main:main"
_FROZEN_LEGACY_SURFACE_SHA256: Final = (
    "aff6ff00b08d90efb1dd279aa5a1a81c27b5edf39eb30f650a02f0df8b1c4ecc"
)


def _source_project_scripts() -> dict[str, str] | None:
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file():
            continue
        scripts: dict[str, str] = {}
        in_scripts = False
        for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line == "[project.scripts]":
                in_scripts = True
                continue
            if in_scripts and line.startswith("["):
                break
            if in_scripts and line and not line.startswith("#"):
                name, value = line.split("=", maxsplit=1)
                target = ast.literal_eval(value.strip())
                if not isinstance(target, str):
                    raise RuntimeError(f"non-string script target for {name.strip()}")
                scripts[name.strip()] = target
        return scripts
    return None


def _project_scripts() -> dict[str, str]:
    source_scripts = _source_project_scripts()
    if source_scripts is not None:
        return source_scripts
    try:
        package = distribution("bayesian-phystwin")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "cannot locate installed metadata or source pyproject.toml"
        ) from exc
    return {
        entry.name: entry.value
        for entry in package.entry_points
        if entry.group == "console_scripts"
    }


def _digest(entries: dict[str, str]) -> str:
    canonical = "\n".join(f"{name}={entries[name]}" for name in sorted(entries))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_frozen_legacy_entry_points() -> dict[str, str]:
    """Return the existing aliases after detecting additions, removals, or drift."""

    scripts = _project_scripts()
    if scripts.get("bpt") != ROOT_TARGET:
        raise RuntimeError("the grouped bpt entry point changed unexpectedly")
    legacy = {
        name: target
        for name, target in scripts.items()
        if name.startswith("bpt-")
    }
    if _digest(legacy) != _FROZEN_LEGACY_SURFACE_SHA256:
        raise RuntimeError(
            "the frozen bpt-* entry-point surface changed; use the grouped registry "
            f"or intentionally revise legacy surface v{LEGACY_SURFACE_VERSION}"
        )
    return legacy
