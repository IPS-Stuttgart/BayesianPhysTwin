#!/usr/bin/env python3
"""Validate the versioned BayesianPhysTwin root export snapshot."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "api/root-public-api-v0.4.json"
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"
SCHEMA: Final = "bayesian-phystwin.root-public-api-snapshot"
SCHEMA_VERSION: Final = 1
POLICY: Final = "exact-legacy-root-export-surface"
_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "package",
        "compatibility_line",
        "policy",
        "symbols",
    }
)
_PROJECT_SECTION = re.compile(r"^\s*\[project\]\s*(?:#.*)?$")
_SECTION = re.compile(r"^\s*\[[^]]+\]\s*(?:#.*)?$")
_PROJECT_VERSION = re.compile(r"^\s*version\s*=\s*(['\"])([^'\"]+)\1\s*(?:#.*)?$")


class PublicApiError(ValueError):
    """Raised when the root public API snapshot or implementation drifts."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicApiError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicApiError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise PublicApiError(f"{name} must be a nonempty literal string")
    return value


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and strictly validate a root API snapshot."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicApiError("cannot read root API manifest") from error
    manifest = _mapping(value, name="root API manifest")
    if set(manifest) != _FIELDS:
        raise PublicApiError("root API manifest fields changed")
    if manifest["schema"] != SCHEMA or manifest["schema_version"] != SCHEMA_VERSION:
        raise PublicApiError("root API manifest contract changed")
    if manifest["policy"] != POLICY:
        raise PublicApiError("root API policy changed")

    package = _literal(manifest["package"], name="package")
    compatibility_line = _literal(
        manifest["compatibility_line"], name="compatibility line"
    )
    symbols = [
        _literal(symbol, name=f"symbols[{index}]")
        for index, symbol in enumerate(_sequence(manifest["symbols"], name="symbols"))
    ]
    if not symbols:
        raise PublicApiError("root API snapshot is empty")
    if len(symbols) != len(set(symbols)):
        raise PublicApiError("root API snapshot contains duplicate symbols")
    if any(not symbol.isidentifier() or symbol.startswith("_") for symbol in symbols):
        raise PublicApiError("root API snapshot contains an invalid public identifier")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "package": package,
        "compatibility_line": compatibility_line,
        "policy": POLICY,
        "symbols": symbols,
    }


def project_version(path: str | Path = DEFAULT_PYPROJECT) -> str:
    """Read one literal project version without requiring Python 3.11 tomllib."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PublicApiError("cannot read project metadata") from error

    in_project = False
    versions: list[str] = []
    for line in lines:
        if _PROJECT_SECTION.fullmatch(line):
            in_project = True
            continue
        if _SECTION.fullmatch(line):
            in_project = False
            continue
        if not in_project:
            continue
        match = _PROJECT_VERSION.fullmatch(line)
        if match is not None:
            versions.append(match.group(2))
    if len(versions) != 1:
        raise PublicApiError("project metadata must contain one literal version")
    return _literal(versions[0], name="project version")


def validate_public_api(
    manifest: Mapping[str, Any],
    *,
    module: ModuleType | None = None,
    version: str | None = None,
) -> dict[str, object]:
    """Compare one imported package with its exact compatibility-line snapshot."""

    package = _literal(manifest.get("package"), name="package")
    compatibility_line = _literal(
        manifest.get("compatibility_line"), name="compatibility line"
    )
    expected = [
        _literal(symbol, name=f"symbols[{index}]")
        for index, symbol in enumerate(
            _sequence(manifest.get("symbols"), name="symbols")
        )
    ]
    if len(expected) != len(set(expected)):
        raise PublicApiError("root API snapshot contains duplicate symbols")

    imported = module if module is not None else importlib.import_module(package)
    if imported.__name__ != package:
        raise PublicApiError("imported module identity differs from the manifest")
    raw_actual = getattr(imported, "__all__", None)
    actual = [
        _literal(symbol, name=f"__all__[{index}]")
        for index, symbol in enumerate(_sequence(raw_actual, name="package __all__"))
    ]
    if len(actual) != len(set(actual)):
        raise PublicApiError("package __all__ contains duplicate symbols")
    missing_attributes = [symbol for symbol in actual if not hasattr(imported, symbol)]
    if missing_attributes:
        raise PublicApiError(
            "package __all__ references missing attributes: "
            + ", ".join(missing_attributes)
        )
    if actual != expected:
        added = sorted(set(actual) - set(expected))
        removed = sorted(set(expected) - set(actual))
        if added or removed:
            raise PublicApiError(
                f"root API set changed; added={added!r}, removed={removed!r}"
            )
        raise PublicApiError("root API order changed")

    exact_version = version if version is not None else project_version()
    parts = exact_version.split(".")
    if len(parts) < 2 or ".".join(parts[:2]) != compatibility_line:
        raise PublicApiError(
            "project version is outside the manifest compatibility line"
        )
    return {
        "package": package,
        "project_version": exact_version,
        "compatibility_line": compatibility_line,
        "policy": manifest["policy"],
        "symbol_count": len(actual),
        "status": "matched",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--pyproject", default=str(DEFAULT_PYPROJECT))
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
        report = validate_public_api(
            manifest,
            version=project_version(arguments.pyproject),
        )
    except PublicApiError as error:
        print(f"public API validation failed: {error}")
        return 2
    if arguments.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "root public API matched: "
            f"{report['symbol_count']} symbols on {report['compatibility_line']}.x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
