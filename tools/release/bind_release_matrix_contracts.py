#!/usr/bin/env python3
"""Bind release-matrix source contracts into candidate release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

SCHEMA: Final = "bayesian-phystwin.release-candidate-evidence"
CONTRACTS: Final = {
    "numerical_environment_contract": "docs/numerical_environment_v1.md",
    "release_matrix_tool": "tools/release/build_release_matrix_evidence.py",
    "release_matrix_binder": "tools/release/bind_release_matrix_contracts.py",
    "release_build_requirements": "requirements/release-build.txt",
    "release_runtime_py310_floor_requirements": (
        "requirements/release-runtime-py310-floor.txt"
    ),
    "release_runtime_py312_requirements": "requirements/release-runtime-py312.txt",
    "release_runtime_py314_requirements": "requirements/release-runtime-py314.txt",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReleaseMatrixBindingError(ValueError):
    """Raised when source contracts cannot be bound exactly."""


def _canonical_id(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ReleaseMatrixBindingError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _load_json(path: Path) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ReleaseMatrixBindingError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ReleaseMatrixBindingError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseMatrixBindingError("cannot read base release evidence") from error
    return _mapping(value, name="base release evidence")


def _file_record(path: Path, *, relative: str) -> dict[str, object]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ReleaseMatrixBindingError(
            f"cannot read source contract {relative}"
        ) from error
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _sdist_path(dist_dir: Path, evidence: Mapping[str, Any]) -> Path:
    try:
        candidates = sorted(dist_dir.glob("*.tar.gz"))
    except OSError as error:
        raise ReleaseMatrixBindingError(
            "cannot enumerate distribution directory"
        ) from error
    if len(candidates) != 1:
        raise ReleaseMatrixBindingError("expected exactly one source distribution")
    expected = _mapping(
        _mapping(evidence.get("artifacts"), name="release artifacts").get("sdist"),
        name="source distribution record",
    )
    candidate = candidates[0]
    data = candidate.read_bytes()
    actual = {
        "path": candidate.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    if actual != {key: expected.get(key) for key in ("path", "sha256", "size_bytes")}:
        raise ReleaseMatrixBindingError(
            "source distribution does not match base release evidence"
        )
    return candidate


def bind_release_matrix_contracts(
    base_evidence_path: Path,
    *,
    dist_dir: Path,
    project_root: Path,
) -> dict[str, object]:
    """Verify sdist/source parity and return augmented release evidence."""

    evidence = _load_json(base_evidence_path)
    if evidence.get("schema") != SCHEMA or evidence.get("schema_version") != 1:
        raise ReleaseMatrixBindingError("unsupported release evidence schema")
    supplied_id = evidence.get("evidence_id")
    if type(supplied_id) is not str or _SHA256.fullmatch(supplied_id) is None:
        raise ReleaseMatrixBindingError("base release evidence ID is invalid")
    descriptor = dict(evidence)
    descriptor.pop("evidence_id")
    if _canonical_id(descriptor) != supplied_id:
        raise ReleaseMatrixBindingError(
            "base release evidence ID does not match its payload"
        )

    version = evidence.get("project_version")
    if type(version) is not str or not version:
        raise ReleaseMatrixBindingError("release evidence lacks a project version")
    archive_root = f"bayesian_phystwin-{version}"
    source_root = project_root.resolve()
    records = {
        name: _file_record(source_root / relative, relative=relative)
        for name, relative in CONTRACTS.items()
    }
    sdist = _sdist_path(dist_dir.resolve(), evidence)
    try:
        with tarfile.open(sdist, mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            for name, relative in CONTRACTS.items():
                member_name = f"{archive_root}/{relative}"
                member = members.get(member_name)
                if member is None or not member.isfile():
                    raise ReleaseMatrixBindingError(
                        f"source distribution lacks {relative}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ReleaseMatrixBindingError(
                        f"cannot read source-distribution member {relative}"
                    )
                data = stream.read()
                record = records[name]
                if (
                    hashlib.sha256(data).hexdigest() != record["sha256"]
                    or len(data) != record["size_bytes"]
                ):
                    raise ReleaseMatrixBindingError(
                        f"source-distribution member changed: {relative}"
                    )
    except (OSError, tarfile.TarError) as error:
        raise ReleaseMatrixBindingError("cannot inspect source distribution") from error

    source_contracts = dict(
        _mapping(evidence.get("source_contracts"), name="source contracts")
    )
    for name, record in records.items():
        existing = source_contracts.get(name)
        if existing is not None and existing != record:
            raise ReleaseMatrixBindingError(f"source contract already drifted: {name}")
        source_contracts[name] = record
    augmented = dict(descriptor)
    augmented["source_contracts"] = dict(sorted(source_contracts.items()))
    return {"evidence_id": _canonical_id(augmented), **augmented}


def write_evidence(path: Path, evidence: Mapping[str, Any]) -> None:
    """Publish evidence atomically without replacing an existing record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReleaseMatrixBindingError(f"refusing to overwrite {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_evidence", type=Path)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = bind_release_matrix_contracts(
            args.base_evidence,
            dist_dir=args.dist_dir,
            project_root=args.project_root,
        )
        write_evidence(args.output, evidence)
    except ReleaseMatrixBindingError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
