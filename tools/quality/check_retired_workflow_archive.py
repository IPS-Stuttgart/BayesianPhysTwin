#!/usr/bin/env python3
"""Validate the exact inactive archive of retired GitHub Actions workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final, cast

ROOT: Final = Path(__file__).resolve().parents[2]
MANIFEST: Final = Path("archive/github-actions/retired-one-shot-v1/manifest.json")
SCHEMA: Final = "bayesian-phystwin.retired-github-actions"
ARCHIVE: Final = "archive/github-actions/retired-one-shot-v1/"
SHA1: Final = re.compile(r"^[0-9a-f]{40}$")
TOP: Final = {
    "schema",
    "schema_version",
    "retired_from_revision",
    "retired_workflow_count",
    "retired_workflow_bytes",
    "scientific_boundary",
    "workflows",
    "contract_tests",
}
WF: Final = {"original_path", "archived_path", "git_blob_sha1", "byte_count"}
TEST: Final = {"original_path", "archived_path", "git_blob_sha1"}


class RetiredWorkflowArchiveError(ValueError):
    """Raised when the retired workflow archive changes."""


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise RetiredWorkflowArchiveError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject(value: str) -> object:
    raise RetiredWorkflowArchiveError(f"non-finite JSON constant: {value}")


def _positive(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise RetiredWorkflowArchiveError(f"{name} must be a positive integer")
    return value


def _sha(value: object, name: str) -> str:
    if type(value) is not str or SHA1.fullmatch(value) is None:
        raise RetiredWorkflowArchiveError(f"{name} must be a Git SHA-1")
    return value


def _canonical(value: object, prefix: str, suffixes: set[str], name: str) -> str:
    if type(value) is not str or not value:
        raise RetiredWorkflowArchiveError(f"{name} must be a path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or not value.startswith(prefix)
        or path.suffix.lower() not in suffixes
    ):
        raise RetiredWorkflowArchiveError(f"{name} is noncanonical")
    return value


def load_manifest(path: Path) -> dict[str, object]:
    """Load and validate the strict archive manifest."""

    if path.is_symlink():
        raise RetiredWorkflowArchiveError("manifest must not be a symlink")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject,
        )
    except RetiredWorkflowArchiveError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetiredWorkflowArchiveError("cannot load archive manifest") from error
    if type(payload) is not dict or set(payload) != TOP:
        raise RetiredWorkflowArchiveError("manifest fields changed")
    if payload["schema"] != SCHEMA or payload["schema_version"] != 1:
        raise RetiredWorkflowArchiveError("manifest schema changed")
    _sha(payload["retired_from_revision"], "retired_from_revision")
    if type(payload["scientific_boundary"]) is not str:
        raise RetiredWorkflowArchiveError("scientific_boundary must be a string")

    workflows = payload["workflows"]
    tests = payload["contract_tests"]
    if type(workflows) is not list or type(tests) is not list:
        raise RetiredWorkflowArchiveError("archive records must be arrays")
    for index, record in enumerate(workflows):
        if type(record) is not dict or set(record) != WF:
            raise RetiredWorkflowArchiveError(f"workflows[{index}] fields changed")
        original = _canonical(
            record["original_path"],
            ".github/workflows/",
            {".yml", ".yaml"},
            f"workflows[{index}].original_path",
        )
        archived = _canonical(
            record["archived_path"],
            ARCHIVE,
            {".yml", ".yaml"},
            f"workflows[{index}].archived_path",
        )
        if (
            "/" in archived[len(ARCHIVE) :]
            or Path(original).name != Path(archived).name
        ):
            raise RetiredWorkflowArchiveError("workflow archive path changed")
        _sha(record["git_blob_sha1"], f"workflows[{index}].git_blob_sha1")
        _positive(record["byte_count"], f"workflows[{index}].byte_count")
    for index, record in enumerate(tests):
        if type(record) is not dict or set(record) != TEST:
            raise RetiredWorkflowArchiveError(f"contract_tests[{index}] fields changed")
        original = _canonical(
            record["original_path"],
            "tests/",
            {".py"},
            f"contract_tests[{index}].original_path",
        )
        archived = _canonical(
            record["archived_path"],
            ARCHIVE + "contract-tests/",
            {".py"},
            f"contract_tests[{index}].archived_path",
        )
        if (
            "/" in archived[len(ARCHIVE + "contract-tests/") :]
            or Path(original).name != Path(archived).name
        ):
            raise RetiredWorkflowArchiveError("contract-test archive path changed")
        _sha(record["git_blob_sha1"], f"contract_tests[{index}].git_blob_sha1")

    for name, records in (("workflows", workflows), ("contract_tests", tests)):
        originals = [str(record["original_path"]) for record in records]
        archives = [str(record["archived_path"]) for record in records]
        if originals != sorted(originals) or len(originals) != len(set(originals)):
            raise RetiredWorkflowArchiveError(f"{name} must be sorted and unique")
        if len(archives) != len(set(archives)):
            raise RetiredWorkflowArchiveError(f"{name} archive paths repeat")
    count = _positive(payload["retired_workflow_count"], "retired_workflow_count")
    size = _positive(payload["retired_workflow_bytes"], "retired_workflow_bytes")
    if count != len(workflows):
        raise RetiredWorkflowArchiveError("retired workflow count changed")
    if size != sum(int(record["byte_count"]) for record in workflows):
        raise RetiredWorkflowArchiveError("retired workflow byte total changed")
    return payload


def _blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _read(root: Path, relative: str) -> bytes:
    source = root / relative
    if source.is_symlink() or not source.is_file():
        raise RetiredWorkflowArchiveError(f"archived path is invalid: {relative}")
    try:
        source.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise RetiredWorkflowArchiveError(
            f"archived path escapes or is unreadable: {relative}"
        ) from error
    return source.read_bytes()


def validate_repository(
    root: Path,
    manifest_path: Path = MANIFEST,
) -> dict[str, object]:
    """Validate inactive original paths and exact archived Git blobs."""

    root = root.resolve(strict=True)
    source = manifest_path if manifest_path.is_absolute() else root / manifest_path
    manifest = load_manifest(source)
    workflows = cast(list[dict[str, object]], manifest["workflows"])
    tests = cast(list[dict[str, object]], manifest["contract_tests"])
    for record in workflows:
        original = root / str(record["original_path"])
        if original.exists() or original.is_symlink():
            raise RetiredWorkflowArchiveError(
                f"retired workflow became active again: {record['original_path']}"
            )
        archived = str(record["archived_path"])
        data = _read(root, archived)
        if len(data) != record["byte_count"] or _blob(data) != record["git_blob_sha1"]:
            raise RetiredWorkflowArchiveError(
                f"archived workflow Git blob changed: {archived}"
            )
    for record in tests:
        archived = str(record["archived_path"])
        if _blob(_read(root, archived)) != record["git_blob_sha1"]:
            raise RetiredWorkflowArchiveError(
                f"archived contract-test Git blob changed: {archived}"
            )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "retired_from_revision": manifest["retired_from_revision"],
        "retired_workflow_count": len(workflows),
        "retired_workflow_bytes": sum(
            int(record["byte_count"]) for record in workflows
        ),
        "archived_contract_test_count": len(tests),
        "active_original_path_count": 0,
        "status": "exact-inactive-archive",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = validate_repository(
            Path(arguments.repository_root),
            Path(arguments.manifest),
        )
    except RetiredWorkflowArchiveError as error:
        print(f"retired workflow archive failed: {error}")
        return 2
    if arguments.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "retired workflow archive matched: "
            f"workflows={report['retired_workflow_count']} "
            f"bytes={report['retired_workflow_bytes']} "
            f"contract-tests={report['archived_contract_test_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
