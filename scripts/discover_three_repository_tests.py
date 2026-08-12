#!/usr/bin/env python3
"""Stage repository-owned three-repository tests for isolated wheel validation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

INVENTORY_SCHEMA = "bayesian-phystwin.three-repository-test-inventory.v1"
_OWNER_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_TEST_PATTERN = "test_three_repository_*.py"


@dataclass(frozen=True)
class SourceSpec:
    owner: str
    root: Path


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _validate_owner(owner: str) -> None:
    if _OWNER_PATTERN.fullmatch(owner) is None:
        _fail(
            "source owner must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores"
        )


def _parse_source(value: str) -> SourceSpec:
    owner, separator, raw_root = value.partition("=")
    if not separator or not owner or not raw_root:
        _fail("--source must have the form OWNER=REPOSITORY_ROOT")
    _validate_owner(owner)
    return SourceSpec(owner=owner, root=Path(raw_root))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalize_sources(
    sources: Sequence[SourceSpec],
    *,
    output_paths: Sequence[Path],
) -> tuple[SourceSpec, ...]:
    if not sources:
        _fail("at least one --source is required")

    normalized: list[SourceSpec] = []
    for source in sources:
        _validate_owner(source.owner)
        root = source.root.expanduser().resolve(strict=True)
        if not root.is_dir():
            _fail(f"source repository root is not a directory: {root}")
        normalized.append(SourceSpec(owner=source.owner, root=root))

    owners = [source.owner for source in normalized]
    if len(owners) != len(set(owners)):
        _fail("source owners must be unique")

    roots = [source.root for source in normalized]
    if len(roots) != len(set(roots)):
        _fail("source repository roots must be unique")

    for source in normalized:
        for output_path in output_paths:
            if _is_relative_to(output_path, source.root):
                _fail(
                    f"output path must not be inside source repository {source.owner}: "
                    f"{output_path}"
                )

    return tuple(sorted(normalized, key=lambda source: source.owner))


def _relative_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            _fail(f"integration test tree contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            _fail(f"integration test tree contains a non-regular entry: {path}")
        files.append(path.relative_to(root))
    return tuple(files)


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def stage_integration_tests(
    sources: Sequence[SourceSpec],
    *,
    output_root: Path,
    path_list: Path,
    inventory_path: Path,
) -> dict[str, object]:
    output_root = output_root.expanduser().resolve()
    path_list = path_list.expanduser().resolve()
    inventory_path = inventory_path.expanduser().resolve()
    if path_list == inventory_path:
        _fail("path list and inventory must use different paths")
    sources = _normalize_sources(
        sources,
        output_paths=(output_root, path_list, inventory_path),
    )

    if output_root.exists():
        if not output_root.is_dir():
            _fail(f"output root exists and is not a directory: {output_root}")
        if any(output_root.iterdir()):
            _fail(f"output root must be empty: {output_root}")
    if path_list.exists():
        _fail(f"path list already exists: {path_list}")
    if inventory_path.exists():
        _fail(f"inventory already exists: {inventory_path}")

    discovered: list[tuple[SourceSpec, Path, tuple[Path, ...], tuple[Path, ...]]] = []
    for source in sources:
        integration_root = source.root / "integration_tests"
        relative_files: tuple[Path, ...] = ()
        relative_tests: tuple[Path, ...] = ()
        if integration_root.exists():
            if integration_root.is_symlink() or not integration_root.is_dir():
                _fail(
                    f"integration_tests must be a regular directory for {source.owner}: "
                    f"{integration_root}"
                )
            relative_files = _relative_files(integration_root)
            relative_tests = tuple(
                path for path in relative_files if path.match(_TEST_PATTERN)
            )
        discovered.append((source, integration_root, relative_files, relative_tests))

    pytest_paths = sorted(
        (Path(source.owner) / path).as_posix()
        for source, _, _, relative_tests in discovered
        for path in relative_tests
    )
    if not pytest_paths:
        _fail("no three-repository integration tests were found in any source")

    output_root.mkdir(parents=True, exist_ok=True)
    owner_records: list[dict[str, object]] = []
    for source, integration_root, relative_files, relative_tests in discovered:
        if relative_tests:
            destination = output_root / source.owner
            for relative_path in relative_files:
                source_path = integration_root / relative_path
                destination_path = destination / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination_path)
        owner_records.append(
            {
                "owner": source.owner,
                "test_files": [path.as_posix() for path in relative_tests],
            }
        )

    inventory: dict[str, object] = {
        "schema": INVENTORY_SCHEMA,
        "owners": owner_records,
        "total_test_files": len(pytest_paths),
    }
    path_text = "".join(f"{path}\n" for path in pytest_paths)
    inventory_text = json.dumps(
        inventory,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    _write_new_text(path_list, path_text)
    _write_new_text(inventory_path, inventory_text)
    return inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="OWNER=REPOSITORY_ROOT",
        help="repository owner label and clean source-snapshot root",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--path-list", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        sources = tuple(_parse_source(value) for value in args.source)
        inventory = stage_integration_tests(
            sources,
            output_root=args.output_root,
            path_list=args.path_list,
            inventory_path=args.inventory,
        )
    except (OSError, ValueError) as error:
        print(f"three-repository test discovery failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(inventory, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
