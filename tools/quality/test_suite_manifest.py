#!/usr/bin/env python3
"""Validate and expand the versioned CI test-suite manifest."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / ".github/quality/test-suites.json"
MANIFEST_SCHEMA = "bayesian-phystwin.ci-test-suites"
MANIFEST_VERSION = 1
_SUITE_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")


class ManifestError(ValueError):
    """Raised when CI test-suite ownership is malformed or inconsistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except OSError as error:
        raise ManifestError(
            f"cannot read test-suite manifest {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"invalid test-suite JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ManifestError("test-suite manifest root must be a JSON object")
    return payload


def _canonical_suite_name(value: object, *, name: str) -> str:
    if type(value) is not str or _SUITE_NAME.fullmatch(value) is None:
        raise ManifestError(f"{name} must be a canonical kebab-case suite name")
    return value


def _canonical_pattern(value: object, *, suite: str, index: int) -> str:
    name = f"suites[{suite!r}][{index}]"
    if type(value) is not str or not value or value.strip() != value:
        raise ManifestError(f"{name} must be a nonempty canonical path pattern")
    if "\\" in value:
        raise ManifestError(f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{name} must stay inside the repository")
    if not path.parts or path.parts[0] != "tests" or path.suffix != ".py":
        raise ManifestError(f"{name} must select Python files below tests/")
    return value


def _expand_pattern(repository_root: Path, pattern: str) -> tuple[str, ...]:
    if glob.has_magic(pattern):
        candidates = sorted(repository_root.glob(pattern))
    else:
        candidates = [repository_root / pattern]
    if not candidates:
        raise ManifestError(f"test pattern matched no files: {pattern}")

    expanded: list[str] = []
    for candidate in candidates:
        try:
            relative = candidate.relative_to(repository_root)
        except ValueError as error:
            raise ManifestError(
                f"test pattern escaped the repository: {pattern}"
            ) from error
        if candidate.is_symlink():
            raise ManifestError(f"test-suite member must not be a symlink: {relative}")
        if not candidate.is_file():
            raise ManifestError(f"test-suite member is not a file: {relative}")
        relative_posix = relative.as_posix()
        if not relative_posix.startswith("tests/") or not relative_posix.endswith(
            ".py"
        ):
            raise ManifestError(f"invalid expanded test-suite member: {relative_posix}")
        expanded.append(relative_posix)
    return tuple(expanded)


def load_test_suites(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, tuple[str, ...]]:
    """Load, expand, and cross-check every declared CI test suite."""

    payload = _load_json(manifest_path)
    expected_fields = {
        "schema",
        "schema_version",
        "suites",
        "subset_requirements",
    }
    if set(payload) != expected_fields:
        unknown = sorted(set(payload) - expected_fields)
        missing = sorted(expected_fields - set(payload))
        raise ManifestError(
            f"test-suite manifest fields changed; missing={missing}, unknown={unknown}"
        )
    if payload["schema"] != MANIFEST_SCHEMA:
        raise ManifestError("unsupported test-suite manifest schema")
    version = payload["schema_version"]
    if type(version) is not int or version != MANIFEST_VERSION:
        raise ManifestError("unsupported test-suite manifest version")

    raw_suites = payload["suites"]
    if not isinstance(raw_suites, Mapping) or not raw_suites:
        raise ManifestError("suites must be a nonempty JSON object")

    suites: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_patterns in raw_suites.items():
        suite = _canonical_suite_name(raw_name, name="suite name")
        if isinstance(raw_patterns, (str, bytes)) or not isinstance(
            raw_patterns, Sequence
        ):
            raise ManifestError(f"suite {suite!r} must contain a list of patterns")
        patterns = tuple(
            _canonical_pattern(value, suite=suite, index=index)
            for index, value in enumerate(raw_patterns)
        )
        if not patterns:
            raise ManifestError(f"suite {suite!r} must not be empty")
        if len(set(patterns)) != len(patterns):
            raise ManifestError(f"suite {suite!r} contains duplicate patterns")

        files: list[str] = []
        owners: dict[str, str] = {}
        for pattern in patterns:
            for path in _expand_pattern(repository_root, pattern):
                previous = owners.get(path)
                if previous is not None:
                    raise ManifestError(
                        f"suite {suite!r} selects {path} through both "
                        f"{previous!r} and {pattern!r}"
                    )
                owners[path] = pattern
                files.append(path)
        suites[suite] = tuple(files)

    raw_requirements = payload["subset_requirements"]
    if isinstance(raw_requirements, (str, bytes)) or not isinstance(
        raw_requirements, Sequence
    ):
        raise ManifestError("subset_requirements must be a JSON list")
    seen_requirements: set[tuple[str, str]] = set()
    for index, raw_requirement in enumerate(raw_requirements):
        if not isinstance(raw_requirement, Mapping) or set(raw_requirement) != {
            "subset",
            "superset",
        }:
            raise ManifestError(
                f"subset_requirements[{index}] must contain subset and superset"
            )
        subset = _canonical_suite_name(
            raw_requirement["subset"],
            name=f"subset_requirements[{index}].subset",
        )
        superset = _canonical_suite_name(
            raw_requirement["superset"],
            name=f"subset_requirements[{index}].superset",
        )
        requirement = (subset, superset)
        if requirement in seen_requirements:
            raise ManifestError(f"duplicate subset requirement: {requirement}")
        seen_requirements.add(requirement)
        if subset not in suites or superset not in suites:
            raise ManifestError(
                f"subset requirement references an unknown suite: {requirement}"
            )
        missing_members = sorted(set(suites[subset]) - set(suites[superset]))
        if missing_members:
            raise ManifestError(
                f"suite {subset!r} is not contained in {superset!r}: {missing_members}"
            )
    return suites


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the versioned test-suite manifest.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root used to expand test patterns.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate all suites.")
    validate.add_argument("--json", action="store_true", dest="as_json")
    list_parser = subparsers.add_parser("list", help="Print one expanded suite.")
    list_parser.add_argument("suite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.repository_root.resolve(strict=True)
        suites = load_test_suites(args.manifest, repository_root=root)
        if args.command == "validate":
            counts = {name: len(paths) for name, paths in suites.items()}
            if args.as_json:
                print(json.dumps(counts, sort_keys=True, separators=(",", ":")))
            else:
                for name, count in counts.items():
                    print(f"{name}: {count}")
            return 0
        suite = _canonical_suite_name(args.suite, name="suite")
        if suite not in suites:
            raise ManifestError(f"unknown test suite: {suite}")
        for path in suites[suite]:
            print(path)
        return 0
    except ManifestError as error:
        print(f"test-suite manifest error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
