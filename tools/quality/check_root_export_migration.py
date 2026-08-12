#!/usr/bin/env python3
"""Validate the legacy-root to owning-module migration contract."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIGRATION_MANIFEST = REPOSITORY_ROOT / "api/root-export-migration-v1.json"
DEFAULT_ROOT_MANIFEST = REPOSITORY_ROOT / "api/root-public-api-v0.4.json"

MIGRATION_SCHEMA = "bayesian-phystwin.root-export-migration"
MIGRATION_SCHEMA_VERSION = 1
MIGRATION_POLICY = "lazy-legacy-root-to-owning-module"
SOURCE_PACKAGE = "bayesian_phystwin"
SOURCE_COMPATIBILITY_LINE = "0.4"
TARGET_COMPATIBILITY_LINE = "0.5"
ROOT_API_SNAPSHOT = "api/root-public-api-v0.4.json"
ROOT_SCHEMA = "bayesian-phystwin.root-public-api-snapshot"
ROOT_SCHEMA_VERSION = 1
ROOT_POLICY = "exact-legacy-root-export-surface"
_MIGRATION_FIELDS = {
    "schema",
    "schema_version",
    "source_package",
    "source_compatibility_line",
    "target_compatibility_line",
    "policy",
    "root_api_snapshot",
    "owners",
}
_ROOT_FIELDS = {
    "schema",
    "schema_version",
    "package",
    "compatibility_line",
    "policy",
    "symbols",
}
_OWNER_FIELDS = {"module", "symbols"}


class RootExportMigrationError(ValueError):
    """Raised when the root-export migration contract is inconsistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RootExportMigrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except OSError as error:
        raise RootExportMigrationError(f"cannot read {name} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RootExportMigrationError(f"invalid {name} JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise RootExportMigrationError(f"{name} root must be a JSON object")
    return payload


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise RootExportMigrationError(f"{name} must be nonempty canonical text")
    if any(character in value for character in "\x00\r\n"):
        raise RootExportMigrationError(f"{name} must be single-line text")
    return value


def _compatibility_line(value: object, *, name: str) -> str:
    result = _canonical_text(value, name=name)
    pieces = result.split(".")
    if len(pieces) != 2 or any(not piece.isdigit() for piece in pieces):
        raise RootExportMigrationError(f"{name} must be a major.minor line")
    return result


def _module_name(value: object, *, source_package: str, index: int) -> str:
    module = _canonical_text(value, name=f"owners[{index}].module")
    prefix = f"{source_package}."
    if not module.startswith(prefix):
        raise RootExportMigrationError(
            f"owners[{index}].module must be below {source_package}"
        )
    if any(not part.isidentifier() for part in module.split(".")):
        raise RootExportMigrationError(
            f"owners[{index}].module must be a canonical Python module"
        )
    return module


def _owner_symbols(
    value: object,
    *,
    owner_index: int,
    seen_symbols: set[str],
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RootExportMigrationError(
            f"owners[{owner_index}].symbols must be a JSON array"
        )
    if not value:
        raise RootExportMigrationError(
            f"owners[{owner_index}].symbols must not be empty"
        )

    result: list[str] = []
    for symbol_index, raw_symbol in enumerate(value):
        symbol = _canonical_text(
            raw_symbol,
            name=f"owners[{owner_index}].symbols[{symbol_index}]",
        )
        if not symbol.isidentifier():
            raise RootExportMigrationError(
                f"owners[{owner_index}].symbols[{symbol_index}] "
                "must be a Python identifier"
            )
        if symbol in seen_symbols:
            raise RootExportMigrationError(f"duplicate export symbol: {symbol}")
        seen_symbols.add(symbol)
        result.append(symbol)
    return result


def load_migration_manifest(path: Path = DEFAULT_MIGRATION_MANIFEST) -> dict[str, Any]:
    """Load and structurally validate the migration manifest."""

    payload = _load_json(path, name="root-export migration manifest")
    if set(payload) != _MIGRATION_FIELDS:
        missing = sorted(_MIGRATION_FIELDS - set(payload))
        unknown = sorted(set(payload) - _MIGRATION_FIELDS)
        raise RootExportMigrationError(
            f"migration manifest fields changed; missing={missing}, unknown={unknown}"
        )
    if payload["schema"] != MIGRATION_SCHEMA:
        raise RootExportMigrationError("root-export migration schema changed")
    if payload["schema_version"] != MIGRATION_SCHEMA_VERSION:
        raise RootExportMigrationError("root-export migration schema version changed")
    if payload["policy"] != MIGRATION_POLICY:
        raise RootExportMigrationError("root-export migration policy changed")
    if payload["root_api_snapshot"] != ROOT_API_SNAPSHOT:
        raise RootExportMigrationError("root API snapshot binding changed")

    source_package = _canonical_text(
        payload["source_package"],
        name="source_package",
    )
    if source_package != SOURCE_PACKAGE:
        raise RootExportMigrationError("root-export source package changed")
    source_line = _compatibility_line(
        payload["source_compatibility_line"],
        name="source_compatibility_line",
    )
    if source_line != SOURCE_COMPATIBILITY_LINE:
        raise RootExportMigrationError("root-export source compatibility line changed")
    target_line = _compatibility_line(
        payload["target_compatibility_line"],
        name="target_compatibility_line",
    )
    if target_line != TARGET_COMPATIBILITY_LINE:
        raise RootExportMigrationError("root-export target compatibility line changed")

    raw_owners = payload["owners"]
    if isinstance(raw_owners, (str, bytes)) or not isinstance(raw_owners, Sequence):
        raise RootExportMigrationError("owners must be a JSON array")
    if not raw_owners:
        raise RootExportMigrationError("owners must not be empty")

    owners: list[dict[str, object]] = []
    seen_modules: set[str] = set()
    seen_symbols: set[str] = set()
    for index, raw_owner in enumerate(raw_owners):
        if not isinstance(raw_owner, Mapping) or set(raw_owner) != _OWNER_FIELDS:
            raise RootExportMigrationError(
                f"owners[{index}] must contain exactly module and symbols"
            )
        module = _module_name(
            raw_owner["module"],
            source_package=source_package,
            index=index,
        )
        if module in seen_modules:
            raise RootExportMigrationError(f"duplicate owner module: {module}")
        seen_modules.add(module)
        owners.append(
            {
                "module": module,
                "symbols": _owner_symbols(
                    raw_owner["symbols"],
                    owner_index=index,
                    seen_symbols=seen_symbols,
                ),
            }
        )

    return {
        "schema": MIGRATION_SCHEMA,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "source_package": source_package,
        "source_compatibility_line": source_line,
        "target_compatibility_line": target_line,
        "policy": MIGRATION_POLICY,
        "root_api_snapshot": ROOT_API_SNAPSHOT,
        "owners": owners,
    }


def _root_symbols(path: Path) -> tuple[str, ...]:
    payload = _load_json(path, name="root public API manifest")
    if set(payload) != _ROOT_FIELDS:
        missing = sorted(_ROOT_FIELDS - set(payload))
        unknown = sorted(set(payload) - _ROOT_FIELDS)
        raise RootExportMigrationError(
            f"root public API fields changed; missing={missing}, unknown={unknown}"
        )
    if (
        payload["schema"] != ROOT_SCHEMA
        or payload["schema_version"] != ROOT_SCHEMA_VERSION
    ):
        raise RootExportMigrationError("root public API schema changed")
    if payload["package"] != SOURCE_PACKAGE:
        raise RootExportMigrationError("root public API package changed")
    if payload["compatibility_line"] != SOURCE_COMPATIBILITY_LINE:
        raise RootExportMigrationError("root public API compatibility line changed")
    if payload["policy"] != ROOT_POLICY:
        raise RootExportMigrationError("root public API policy changed")

    symbols = payload["symbols"]
    if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
        raise RootExportMigrationError(
            "root public API manifest symbols must be a JSON array"
        )
    result = tuple(
        _canonical_text(value, name=f"root symbols[{index}]")
        for index, value in enumerate(symbols)
    )
    if len(set(result)) != len(result):
        raise RootExportMigrationError("root public API manifest contains duplicates")
    return result


def _manifest_mapping(migration: Mapping[str, Any]) -> dict[str, str]:
    raw_owners = migration["owners"]
    if not isinstance(raw_owners, list):
        raise RootExportMigrationError("validated owners changed type")

    result: dict[str, str] = {}
    for owner in raw_owners:
        if not isinstance(owner, Mapping):
            raise RootExportMigrationError("validated owner changed type")
        module = str(owner["module"])
        symbols = owner["symbols"]
        if not isinstance(symbols, list):
            raise RootExportMigrationError("validated owner symbols changed type")
        for symbol in symbols:
            result[str(symbol)] = module
    return result


def _runtime_mapping(
    package: ModuleType,
    *,
    source_package: str,
) -> dict[str, str]:
    raw_mapping = getattr(package, "_ROOT_EXPORT_MODULES", None)
    if not isinstance(raw_mapping, Mapping):
        raise RootExportMigrationError(
            "package does not expose the internal lazy-root mapping"
        )

    result: dict[str, str] = {}
    for raw_symbol, raw_module in raw_mapping.items():
        symbol = _canonical_text(raw_symbol, name="runtime export symbol")
        module = _canonical_text(raw_module, name=f"runtime module for {symbol}")
        if module.startswith(f"{source_package}."):
            full_module = module
        else:
            full_module = f"{source_package}.{module}"
        result[symbol] = full_module
    return result


def validate_root_export_migration(
    migration_path: Path = DEFAULT_MIGRATION_MANIFEST,
    *,
    root_manifest_path: Path = DEFAULT_ROOT_MANIFEST,
    package: ModuleType | None = None,
    resolve: bool = True,
) -> dict[str, object]:
    """Validate snapshot coverage, runtime ownership, and object identity."""

    migration = load_migration_manifest(migration_path)
    expected_symbols = _root_symbols(root_manifest_path)
    expected_mapping = _manifest_mapping(migration)

    missing = sorted(set(expected_symbols) - set(expected_mapping))
    extra = sorted(set(expected_mapping) - set(expected_symbols))
    if missing or extra:
        raise RootExportMigrationError(
            "migration symbols differ from the frozen root API snapshot; "
            f"missing={missing}, extra={extra}"
        )

    source_package = str(migration["source_package"])
    runtime_package = (
        importlib.import_module(source_package) if package is None else package
    )
    if runtime_package.__name__ != source_package:
        raise RootExportMigrationError("runtime package identity changed")

    runtime_mapping = _runtime_mapping(
        runtime_package,
        source_package=source_package,
    )
    if runtime_mapping != expected_mapping:
        missing = sorted(set(expected_mapping) - set(runtime_mapping))
        extra = sorted(set(runtime_mapping) - set(expected_mapping))
        changed = sorted(
            symbol
            for symbol in set(expected_mapping) & set(runtime_mapping)
            if expected_mapping[symbol] != runtime_mapping[symbol]
        )
        raise RootExportMigrationError(
            "runtime migration mapping differs from the manifest; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    if tuple(getattr(runtime_package, "__all__", ())) != expected_symbols:
        raise RootExportMigrationError(
            "runtime root export order differs from the frozen snapshot"
        )

    if resolve:
        for symbol in expected_symbols:
            module_name = expected_mapping[symbol]
            owner = importlib.import_module(module_name)
            try:
                owner_value = getattr(owner, symbol)
                root_value = getattr(runtime_package, symbol)
            except AttributeError as error:
                raise RootExportMigrationError(
                    f"migration target does not resolve: {module_name}.{symbol}"
                ) from error
            if root_value is not owner_value:
                raise RootExportMigrationError(
                    f"lazy root export identity differs for {symbol}"
                )

    return {
        "source_package": source_package,
        "source_compatibility_line": migration["source_compatibility_line"],
        "target_compatibility_line": migration["target_compatibility_line"],
        "policy": migration["policy"],
        "owner_count": len(migration["owners"]),
        "symbol_count": len(expected_symbols),
        "resolved": resolve,
        "status": "matched",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MIGRATION_MANIFEST,
        help="Root-export migration manifest.",
    )
    parser.add_argument(
        "--root-manifest",
        type=Path,
        default=DEFAULT_ROOT_MANIFEST,
        help="Frozen historical root API snapshot.",
    )
    parser.add_argument(
        "--no-resolve",
        action="store_true",
        help="Validate mappings without importing every owning module.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_root_export_migration(
            args.manifest,
            root_manifest_path=args.root_manifest,
            resolve=not args.no_resolve,
        )
    except RootExportMigrationError as error:
        print(f"root-export migration error: {error}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
