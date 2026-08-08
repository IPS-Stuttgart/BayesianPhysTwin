"""Strict read-only verification for paper-facing ``ClaimBundleV1`` rows."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .claim_bundle_v1 import ClaimBundleV1

COMPACT_CLAIM_TABLE_SCHEMA = "bayesian_phystwin.compact_claim_table"
COMPACT_CLAIM_TABLE_SCHEMA_VERSION = 1
CLAIM_EVIDENCE_BINDING_SCHEMA = "bayesian_phystwin.claim_evidence_bindings"
CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION = 1

_BINDING_ROOT_FIELDS = frozenset(
    {"schema_name", "schema_version", "bindings", "migration_exceptions"}
)
_BINDING_FIELDS = frozenset(
    {
        "claim_id",
        "manifest",
        "artifact_root",
        "expected_manifest_id",
        "expected_evidence_fingerprint",
        "result_artifact",
        "table_artifact",
        "table_row_id",
    }
)
_REFERENCE_FIELDS = frozenset({"name", "path", "sha256"})
_EXCEPTION_FIELDS = frozenset({"claim_id", "reason"})
_TABLE_FIELDS = frozenset({"schema_name", "schema_version", "rows"})
_TABLE_ROW_FIELDS = frozenset({"id", "claim_id", "evidence"})


class ClaimBundleArtifactLike(Protocol):
    """Structural type needed from a claim-bundle artifact."""

    name: str
    kind: str
    path: str
    sha256: str
    size_bytes: int


class ClaimBundleLike(Protocol):
    """Structural type needed from ``ClaimBundleV1``."""

    claim_ids: Sequence[str]
    artifacts: Sequence[ClaimBundleArtifactLike]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} cannot be read as UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    return _require_mapping(value, name=name)


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be literal strings")
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if not missing and not unknown:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unknown:
        details.append(f"unknown {unknown}")
    raise ValueError(f"{name} does not match schema: {', '.join(details)}")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty literal text")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return value


def _require_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    digest = _require_text(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_relative_path(value: Any, *, name: str) -> PurePosixPath:
    text = _require_text(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must use portable forward slashes")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) == "."
        or ".." in path.parts
        or path.as_posix() != text
    ):
        raise ValueError(f"{name} must be a canonical normalized relative path")
    return path


def _require_binding_root(value: Any, *, name: str) -> PurePosixPath:
    text = _require_text(value, name=name)
    if text == ".":
        return PurePosixPath()
    return _require_relative_path(text, name=name)


def _stable_regular_file_snapshot(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"paper-handoff artifact cannot be opened: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"paper-handoff artifact is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or size != after.st_size:
        raise ValueError(f"paper-handoff artifact changed while hashing: {path}")
    return digest.hexdigest(), size


def _resolved_artifact_path(root: Path, path: str) -> Path:
    relative = _require_relative_path(path, name="bundle artifact path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("paper-handoff artifacts must not use symbolic links")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"paper-handoff artifact is not a file: {candidate}"
        ) from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("paper-handoff artifact escapes artifact root") from error
    return resolved


def _verify_artifact_bytes(
    artifact: ClaimBundleArtifactLike,
    *,
    root: Path,
) -> Path:
    path = _resolved_artifact_path(root, artifact.path)
    digest, size = _stable_regular_file_snapshot(path)
    if size != artifact.size_bytes:
        raise ValueError(f"paper-handoff artifact size differs: {artifact.name}")
    if digest != artifact.sha256:
        raise ValueError(f"paper-handoff artifact digest differs: {artifact.name}")
    return path


def _binding_reference(
    value: Any,
    *,
    name: str,
    artifact_root: PurePosixPath,
) -> tuple[str, str, str]:
    reference = _require_mapping(value, name=name)
    _require_exact_fields(reference, expected=_REFERENCE_FIELDS, name=name)
    artifact_name = _require_text(reference["name"], name=f"{name}.name")
    relative = _require_relative_path(reference["path"], name=f"{name}.path")
    path = (artifact_root / relative).as_posix()
    digest = _require_sha256(reference["sha256"], name=f"{name}.sha256")
    return artifact_name, path, digest


def _bound_artifact(
    value: Any,
    *,
    name: str,
    artifact_root: PurePosixPath,
    by_path: Mapping[str, ClaimBundleArtifactLike],
    allowed_kinds: frozenset[str],
    root: Path,
) -> tuple[ClaimBundleArtifactLike, Path]:
    reference_name, reference_path, reference_digest = _binding_reference(
        value,
        name=name,
        artifact_root=artifact_root,
    )
    artifact = by_path.get(reference_path)
    if artifact is None or artifact.kind not in allowed_kinds:
        raise ValueError(f"{name} is absent from the claim bundle")
    if artifact.name != reference_name:
        raise ValueError(f"{name} selects another bundle artifact name")
    if artifact.sha256 != reference_digest:
        raise ValueError(f"{name} selects another artifact digest")
    return artifact, _verify_artifact_bytes(artifact, root=root)


def _verify_compact_table_row(
    path: Path,
    *,
    row_id: str,
    claim_id: str,
) -> None:
    table = _load_json_mapping(path, name="compact claim table")
    _require_exact_fields(table, expected=_TABLE_FIELDS, name="compact claim table")
    if table["schema_name"] != COMPACT_CLAIM_TABLE_SCHEMA:
        raise ValueError("unsupported compact-claim-table schema")
    if (
        _require_integer(
            table["schema_version"],
            name="compact claim table schema_version",
        )
        != COMPACT_CLAIM_TABLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported compact-claim-table schema version")

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_row in _require_sequence(table["rows"], name="compact claim table rows"):
        row = _require_mapping(raw_row, name="compact claim table row")
        _require_exact_fields(
            row,
            expected=_TABLE_ROW_FIELDS,
            name="compact claim table row",
        )
        candidate_id = _require_text(row["id"], name="compact claim table row id")
        if candidate_id in rows_by_id:
            raise ValueError(f"duplicate compact claim table row: {candidate_id}")
        _require_text(
            row["claim_id"],
            name=f"compact table row {candidate_id}.claim_id",
        )
        _require_sequence(
            row["evidence"],
            name=f"compact table row {candidate_id}.evidence",
        )
        rows_by_id[candidate_id] = row

    selected = rows_by_id.get(row_id)
    if selected is None:
        raise ValueError(f"compact claim table has no row {row_id!r}")
    if selected["claim_id"] != claim_id:
        raise ValueError("compact claim table row is bound to another claim")


def verify_compact_claim_table_bindings(
    bundle: ClaimBundleLike,
    *,
    root: str | Path,
) -> dict[str, object]:
    """Verify every paper binding selects one real compact-table claim row."""

    artifact_root = Path(root).resolve()
    binding_artifacts = [
        artifact for artifact in bundle.artifacts if artifact.kind == "claim_binding"
    ]
    if len(binding_artifacts) != 1:
        raise ValueError("paper handoff requires exactly one claim-binding artifact")

    by_path: dict[str, ClaimBundleArtifactLike] = {}
    for artifact in bundle.artifacts:
        if artifact.path in by_path:
            raise ValueError(f"duplicate claim-bundle artifact path: {artifact.path}")
        by_path[artifact.path] = artifact

    binding_path = _verify_artifact_bytes(binding_artifacts[0], root=artifact_root)
    payload = _load_json_mapping(binding_path, name="claim-evidence bindings")
    _require_exact_fields(
        payload,
        expected=_BINDING_ROOT_FIELDS,
        name="claim-evidence bindings",
    )
    if payload["schema_name"] != CLAIM_EVIDENCE_BINDING_SCHEMA:
        raise ValueError("unsupported claim-evidence binding schema")
    if (
        _require_integer(
            payload["schema_version"],
            name="claim-evidence binding schema_version",
        )
        != CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION
    ):
        raise ValueError("unsupported claim-evidence binding schema version")

    expected_claims = {
        _require_text(claim_id, name="bundle claim ID") for claim_id in bundle.claim_ids
    }
    if len(expected_claims) != len(bundle.claim_ids) or not expected_claims:
        raise ValueError("bundle claim IDs must be unique and nonempty")

    bound_claims: set[str] = set()
    table_paths: set[str] = set()
    for raw_binding in _require_sequence(payload["bindings"], name="bindings"):
        binding = _require_mapping(raw_binding, name="claim-evidence binding")
        _require_exact_fields(
            binding,
            expected=_BINDING_FIELDS,
            name="claim-evidence binding",
        )
        claim_id = _require_text(binding["claim_id"], name="binding claim_id")
        if claim_id in bound_claims:
            raise ValueError(f"duplicate claim-evidence binding: {claim_id}")
        bound_claims.add(claim_id)

        binding_root = _require_binding_root(
            binding["artifact_root"],
            name=f"{claim_id}.artifact_root",
        )
        _bound_artifact(
            binding["result_artifact"],
            name=f"{claim_id}.result_artifact",
            artifact_root=binding_root,
            by_path=by_path,
            allowed_kinds=frozenset({"evidence_summary", "supporting"}),
            root=artifact_root,
        )
        table_artifact, table_path = _bound_artifact(
            binding["table_artifact"],
            name=f"{claim_id}.table_artifact",
            artifact_root=binding_root,
            by_path=by_path,
            allowed_kinds=frozenset({"table_data"}),
            root=artifact_root,
        )
        table_paths.add(table_artifact.path)
        _verify_compact_table_row(
            table_path,
            row_id=_require_text(
                binding["table_row_id"],
                name=f"{claim_id}.table_row_id",
            ),
            claim_id=claim_id,
        )

    if bound_claims != expected_claims:
        missing = sorted(expected_claims - bound_claims)
        unknown = sorted(bound_claims - expected_claims)
        raise ValueError(
            "paper bindings do not match bundle claim IDs: "
            f"missing={missing}, unknown={unknown}"
        )

    exceptions: set[str] = set()
    for raw_exception in _require_sequence(
        payload["migration_exceptions"],
        name="migration_exceptions",
    ):
        exception = _require_mapping(
            raw_exception,
            name="claim-evidence migration exception",
        )
        _require_exact_fields(
            exception,
            expected=_EXCEPTION_FIELDS,
            name="claim-evidence migration exception",
        )
        claim_id = _require_text(
            exception["claim_id"],
            name="migration exception claim_id",
        )
        _require_text(exception["reason"], name=f"{claim_id}.migration reason")
        if claim_id in exceptions:
            raise ValueError(
                f"duplicate claim-evidence migration exception: {claim_id}"
            )
        exceptions.add(claim_id)
    overlap = sorted(expected_claims & exceptions)
    if overlap:
        raise ValueError(
            "paper handoff cannot rely on migration exceptions for bundle claims: "
            f"{overlap}"
        )

    return {
        "binding_claim_count": len(bound_claims),
        "compact_table_count": len(table_paths),
        "compact_table_row_count": len(bound_claims),
    }


def verify_claim_bundle_paper_handoff(
    bundle: ClaimBundleV1,
    *,
    root: str | Path,
) -> dict[str, object]:
    """Run generic bundle verification and strict paper compact-row checks."""

    from .claim_bundle_v1 import verify_claim_bundle_artifacts

    verify_claim_bundle_artifacts(bundle, root=root)
    return verify_compact_claim_table_bindings(bundle, root=root)


__all__ = [
    "CLAIM_EVIDENCE_BINDING_SCHEMA",
    "CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION",
    "COMPACT_CLAIM_TABLE_SCHEMA",
    "COMPACT_CLAIM_TABLE_SCHEMA_VERSION",
    "verify_claim_bundle_paper_handoff",
    "verify_compact_claim_table_bindings",
]
