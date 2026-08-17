#!/usr/bin/env python3
"""Validate and content-address release-facing scientific claim synchronization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

CONTRACT_SCHEMA: Final = "bayesian-phystwin.release-claim-sync-contract"
REPORT_SCHEMA: Final = "bayesian-phystwin.release-claim-sync-report"
SCHEMA_VERSION: Final = 1
DEFAULT_CONTRACT_PATH: Final = Path("release/claim_contract_v1.json")
_CONTRACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "contract_name",
        "claim_boundary",
        "documents",
        "source_documents",
    }
)
_DOCUMENT_FIELDS: Final = frozenset({"path", "required_literals"})


class ReleaseClaimSyncError(ValueError):
    """Raised when release-facing claims are missing, ambiguous, or malformed."""


def _reject_constant(value: str) -> None:
    raise ReleaseClaimSyncError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseClaimSyncError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseClaimSyncError(f"cannot read UTF-8 JSON: {path}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ReleaseClaimSyncError(f"invalid JSON in {path}: {error.msg}") from error


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ReleaseClaimSyncError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReleaseClaimSyncError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ReleaseClaimSyncError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r"):
        raise ReleaseClaimSyncError(f"{name} contains a forbidden control character")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ReleaseClaimSyncError(
            f"{name} has unexpected fields; missing={missing}, extra={extra}"
        )


def _canonical_relative_path(value: object, *, name: str) -> str:
    path = _literal(value, name=name)
    if path.startswith("/") or "\\" in path:
        raise ReleaseClaimSyncError(f"{name} must be a relative POSIX path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ReleaseClaimSyncError(f"{name} is not canonical")
    if PurePosixPath(path).as_posix() != path:
        raise ReleaseClaimSyncError(f"{name} is not canonical")
    return path


def _canonical_strings(value: object, *, name: str) -> tuple[str, ...]:
    sequence = _sequence(value, name=name)
    result = tuple(
        _literal(item, name=f"{name}[{index}]") for index, item in enumerate(sequence)
    )
    if not result:
        raise ReleaseClaimSyncError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ReleaseClaimSyncError(f"{name} contains duplicate values")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _safe_file(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    path = root
    for part in PurePosixPath(relative).parts:
        path /= part
        if path.is_symlink():
            raise ReleaseClaimSyncError(
                f"required path must not traverse a symlink: {relative}"
            )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReleaseClaimSyncError(
            f"required file does not exist: {relative}"
        ) from error
    if not resolved.is_relative_to(root_resolved):
        raise ReleaseClaimSyncError(
            f"required file escapes the project root: {relative}"
        )
    if not resolved.is_file():
        raise ReleaseClaimSyncError(f"required path must be a regular file: {relative}")
    return resolved


def _file_record(path: Path, *, relative: str) -> dict[str, object]:
    try:
        value = path.read_bytes()
    except OSError as error:
        raise ReleaseClaimSyncError(f"cannot read required file: {relative}") from error
    return {
        "path": relative,
        "sha256": _sha256_bytes(value),
        "size_bytes": len(value),
    }


def _read_utf8(path: Path, *, relative: str) -> tuple[str, bytes]:
    try:
        value = path.read_bytes()
        text = value.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseClaimSyncError(
            f"cannot read UTF-8 document: {relative}"
        ) from error
    return text, value


def _normalized_whitespace(value: str) -> str:
    return " ".join(value.split())


def _validated_contract(value: object) -> dict[str, object]:
    contract = _mapping(value, name="release claim contract")
    _exact_fields(
        contract,
        expected=_CONTRACT_FIELDS,
        name="release claim contract",
    )
    if contract["schema"] != CONTRACT_SCHEMA:
        raise ReleaseClaimSyncError("unexpected release claim contract schema")
    if type(contract["schema_version"]) is not int or (
        contract["schema_version"] != SCHEMA_VERSION
    ):
        raise ReleaseClaimSyncError("unexpected release claim contract version")

    documents_value = _sequence(contract["documents"], name="documents")
    documents: list[dict[str, object]] = []
    for index, raw_document in enumerate(documents_value):
        document = _mapping(raw_document, name=f"documents[{index}]")
        _exact_fields(
            document,
            expected=_DOCUMENT_FIELDS,
            name=f"documents[{index}]",
        )
        documents.append(
            {
                "path": _canonical_relative_path(
                    document["path"],
                    name=f"documents[{index}].path",
                ),
                "required_literals": list(
                    _canonical_strings(
                        document["required_literals"],
                        name=f"documents[{index}].required_literals",
                    )
                ),
            }
        )
    if not documents:
        raise ReleaseClaimSyncError("documents must not be empty")
    document_paths = [cast(str, document["path"]) for document in documents]
    if document_paths != sorted(document_paths):
        raise ReleaseClaimSyncError("documents must be sorted by path")
    if len(document_paths) != len(set(document_paths)):
        raise ReleaseClaimSyncError("documents contains duplicate paths")

    source_documents = tuple(
        _canonical_relative_path(item, name=f"source_documents[{index}]")
        for index, item in enumerate(
            _sequence(contract["source_documents"], name="source_documents")
        )
    )
    if not source_documents:
        raise ReleaseClaimSyncError("source_documents must not be empty")
    if source_documents != tuple(sorted(source_documents)):
        raise ReleaseClaimSyncError("source_documents must be sorted")
    if len(source_documents) != len(set(source_documents)):
        raise ReleaseClaimSyncError("source_documents contains duplicate paths")

    return {
        "schema": CONTRACT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract_name": _literal(contract["contract_name"], name="contract_name"),
        "claim_boundary": _literal(contract["claim_boundary"], name="claim_boundary"),
        "documents": documents,
        "source_documents": list(source_documents),
    }


def check_release_claim_sync(
    project_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, object]:
    """Validate release-facing claim documents and return an immutable report."""

    root = project_root.resolve()
    relative_contract = _canonical_relative_path(
        contract_path.as_posix(),
        name="contract_path",
    )
    contract_file = _safe_file(root, relative_contract)
    contract_bytes = contract_file.read_bytes()
    contract = _validated_contract(_load_json(contract_file))
    contract_id = _sha256_bytes(_canonical_json(contract))

    document_records: list[dict[str, object]] = []
    for raw_document in cast(list[dict[str, object]], contract["documents"]):
        relative = cast(str, raw_document["path"])
        path = _safe_file(root, relative)
        text, value = _read_utf8(path, relative=relative)
        normalized_text = _normalized_whitespace(text)
        required = cast(list[str], raw_document["required_literals"])
        occurrences: dict[str, int] = {}
        for literal in required:
            count = normalized_text.count(_normalized_whitespace(literal))
            if count == 0:
                raise ReleaseClaimSyncError(
                    f"{relative} is missing required release-claim literal: {literal!r}"
                )
            occurrences[literal] = count
        document_records.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(value),
                "size_bytes": len(value),
                "required_literal_count": len(required),
                "matched_occurrences": occurrences,
            }
        )

    source_records = [
        _file_record(
            _safe_file(root, relative),
            relative=relative,
        )
        for relative in cast(list[str], contract["source_documents"])
    ]
    descriptor: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract_name": contract["contract_name"],
        "contract_id": contract_id,
        "claim_boundary": contract["claim_boundary"],
        "contract_file": {
            "path": relative_contract,
            "sha256": _sha256_bytes(contract_bytes),
            "size_bytes": len(contract_bytes),
        },
        "documents": document_records,
        "source_documents": source_records,
    }
    return {
        **descriptor,
        "report_id": _sha256_bytes(_canonical_json(descriptor)),
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write a report exactly once."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                report,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as error:
        raise ReleaseClaimSyncError(
            f"refusing to overwrite existing report: {path}"
        ) from error
    except OSError as error:
        raise ReleaseClaimSyncError(f"cannot write report: {path}") from error


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate release-facing claim synchronization."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = check_release_claim_sync(
            arguments.root,
            contract_path=arguments.contract,
        )
        if arguments.output is not None:
            write_report(arguments.output, report)
    except ReleaseClaimSyncError as error:
        print(f"release claim synchronization failed: {error}", file=sys.stderr)
        return 2
    print(
        "release claim synchronization passed: "
        f"contract={report['contract_id']} report={report['report_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
