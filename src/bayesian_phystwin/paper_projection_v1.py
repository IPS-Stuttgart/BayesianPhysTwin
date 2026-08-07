"""Strict read-only paper projections for verified claim bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .claim_bundle_v1 import (
    ClaimBundleArtifactV1,
    ClaimBundleV1,
    load_claim_bundle,
    verify_claim_bundle_artifacts,
)

PAPER_PROJECTION_SCHEMA = "bayesian_phystwin.paper_projection"
PAPER_PROJECTION_SCHEMA_VERSION = 1
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
_TABLE_FIELDS = frozenset({"schema_name", "schema_version", "rows"})
_ROW_FIELDS = frozenset({"id", "claim_id", "evidence"})
_CLAIM_FIELDS = frozenset(
    {
        "claim_id",
        "result_artifact",
        "result_sha256",
        "table_artifact",
        "table_sha256",
        "table_row_id",
        "evidence",
    }
)
_REPOSITORY_FIELDS = frozenset({"repository", "revision", "dirty", "role"})
_PROJECTION_FIELDS = frozenset(
    {
        "projection_id",
        "schema_name",
        "schema_version",
        "bundle_id",
        "run_manifest_id",
        "evidence_fingerprint",
        "run_id",
        "classification",
        "protocol_id",
        "statistical_unit",
        "claim_boundary",
        "claim_ids",
        "method_freeze_id",
        "split_id",
        "baseline_id",
        "repositories",
        "claims",
    }
)
_CLASSIFICATIONS = frozenset({"controlled", "confirmatory"})
_REPOSITORY_ROLES = frozenset(
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


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load(path: str | Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    return _mapping(value, name)


def _exact(value: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    actual = frozenset(map(str, value))
    if actual != fields:
        raise ValueError(
            f"{name} fields changed: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    digest = _text(value, name)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _git_sha(value: Any, name: str) -> str:
    revision = _text(value, name)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"{name} must be an exact lowercase Git SHA")
    return revision


def _relative(value: Any, name: str) -> str:
    text = _text(value, name)
    if "\\" in text:
        raise ValueError(f"{name} must use portable forward slashes")
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) == "." or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    return path.as_posix()


def _binding_root(value: Any, name: str) -> PurePosixPath:
    text = _text(value, name)
    return PurePosixPath() if text == "." else PurePosixPath(_relative(text, name))


def _safe(root: Path, relative: str) -> Path:
    base = root.resolve()
    unresolved = base / relative
    if unresolved.is_symlink():
        raise ValueError("paper-projection artifacts must not be symbolic links")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(base)
    except OSError as error:
        raise ValueError("paper-projection artifact is missing") from error
    except ValueError as error:
        raise ValueError("paper-projection artifact escapes artifact root") from error
    if not resolved.is_file():
        raise ValueError("paper-projection artifact must be a regular file")
    return resolved


def _finite_json(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error


def load_compact_claim_table_row(
    path: str | Path,
    *,
    row_id: str,
    claim_id: str,
) -> Any:
    """Return the evidence from one uniquely owned compact-table row."""

    table = _load(path, "compact claim table")
    _exact(table, _TABLE_FIELDS, "compact claim table")
    if (
        table["schema_name"] != COMPACT_CLAIM_TABLE_SCHEMA
        or table["schema_version"] != COMPACT_CLAIM_TABLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported compact-claim-table schema")

    selected: Mapping[str, Any] | None = None
    seen: set[str] = set()
    for raw in _sequence(table["rows"], "compact claim table rows"):
        row = _mapping(raw, "compact claim table row")
        _exact(row, _ROW_FIELDS, "compact claim table row")
        current = _text(row["id"], "compact claim table row id")
        if current in seen:
            raise ValueError(f"duplicate compact-table row ID: {current}")
        seen.add(current)
        if current == row_id:
            selected = row
    if selected is None:
        raise ValueError(f"compact table must contain exactly one row {row_id!r}")
    if _text(selected["claim_id"], "compact table claim_id") != claim_id:
        raise ValueError("compact-table row is bound to another claim")
    return _finite_json(selected["evidence"], f"{claim_id} evidence")


@dataclass(frozen=True)
class PaperProjectionClaimV1:
    claim_id: str
    result_artifact: str
    result_sha256: str
    table_artifact: str
    table_sha256: str
    table_row_id: str
    evidence: Any

    def __post_init__(self) -> None:
        for field in ("claim_id", "result_artifact", "table_artifact", "table_row_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(
            self, "result_sha256", _sha256(self.result_sha256, "result_sha256")
        )
        object.__setattr__(
            self, "table_sha256", _sha256(self.table_sha256, "table_sha256")
        )
        object.__setattr__(
            self, "evidence", _finite_json(self.evidence, f"{self.claim_id} evidence")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "result_artifact": self.result_artifact,
            "result_sha256": self.result_sha256,
            "table_artifact": self.table_artifact,
            "table_sha256": self.table_sha256,
            "table_row_id": self.table_row_id,
            "evidence": self.evidence,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PaperProjectionClaimV1:
        _exact(value, _CLAIM_FIELDS, "paper-projection claim")
        return cls(**dict(value))


@dataclass(frozen=True)
class PaperProjectionV1:
    bundle_id: str
    run_manifest_id: str
    evidence_fingerprint: str
    run_id: str
    classification: str
    protocol_id: str
    statistical_unit: str
    claim_boundary: str
    claim_ids: tuple[str, ...]
    method_freeze_id: str
    split_id: str
    baseline_id: str
    repositories: tuple[Mapping[str, Any], ...]
    claims: tuple[PaperProjectionClaimV1, ...]

    def __post_init__(self) -> None:
        for field in ("bundle_id", "run_manifest_id", "evidence_fingerprint"):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        for field in (
            "run_id",
            "protocol_id",
            "statistical_unit",
            "claim_boundary",
            "method_freeze_id",
            "split_id",
            "baseline_id",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        classification = _text(self.classification, "classification")
        if classification not in _CLASSIFICATIONS:
            raise ValueError("paper projection requires claim-bearing evidence")
        object.__setattr__(self, "classification", classification)

        claim_ids = tuple(_text(value, "claim ID") for value in self.claim_ids)
        claims = tuple(self.claims)
        by_id = {claim.claim_id: claim for claim in claims}
        if (
            not claim_ids
            or len(claim_ids) != len(set(claim_ids))
            or len(by_id) != len(claims)
            or set(by_id) != set(claim_ids)
        ):
            raise ValueError("paper-projection claims must match unique claim_ids")

        repositories: list[dict[str, Any]] = []
        names: set[str] = set()
        primary_count = 0
        for raw in self.repositories:
            repository = _mapping(raw, "paper-projection repository")
            _exact(repository, _REPOSITORY_FIELDS, "paper-projection repository")
            name = _text(repository["repository"], "repository")
            role = _text(repository["role"], f"{name}.role")
            if name in names or role not in _REPOSITORY_ROLES:
                raise ValueError("duplicate repository or unsupported repository role")
            if repository["dirty"] is not False:
                raise ValueError("paper-projection repositories must be clean")
            names.add(name)
            primary_count += role == "primary"
            repositories.append(
                {
                    "repository": name,
                    "revision": _git_sha(repository["revision"], f"{name}.revision"),
                    "dirty": False,
                    "role": role,
                }
            )
        if primary_count != 1:
            raise ValueError("paper projection requires exactly one primary repository")
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "claims", tuple(by_id[value] for value in claim_ids))
        object.__setattr__(self, "repositories", tuple(repositories))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": PAPER_PROJECTION_SCHEMA,
            "schema_version": PAPER_PROJECTION_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "run_manifest_id": self.run_manifest_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "run_id": self.run_id,
            "classification": self.classification,
            "protocol_id": self.protocol_id,
            "statistical_unit": self.statistical_unit,
            "claim_boundary": self.claim_boundary,
            "claim_ids": list(self.claim_ids),
            "method_freeze_id": self.method_freeze_id,
            "split_id": self.split_id,
            "baseline_id": self.baseline_id,
            "repositories": [dict(value) for value in self.repositories],
            "claims": [claim.as_dict() for claim in self.claims],
        }

    @property
    def projection_id(self) -> str:
        return hashlib.sha256(_canonical(self.descriptor())).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {"projection_id": self.projection_id, **self.descriptor()}


def _artifact_index(bundle: ClaimBundleV1) -> dict[str, ClaimBundleArtifactV1]:
    return {artifact.path: artifact for artifact in bundle.artifacts}


def _output_index(manifest: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for artifact in manifest.outputs:
        path = _relative(artifact.path, "manifest output path")
        if path in result:
            raise ValueError(f"duplicate run-manifest output path: {path}")
        result[path] = artifact
    return result


def _reference(
    value: Any,
    *,
    claim_id: str,
    field: str,
    root: PurePosixPath,
    artifacts: Mapping[str, ClaimBundleArtifactV1],
    outputs: Mapping[str, Any],
    kinds: frozenset[str],
) -> ClaimBundleArtifactV1:
    reference = _mapping(value, f"{claim_id}.{field}")
    _exact(reference, _REFERENCE_FIELDS, f"{claim_id}.{field}")
    name = _text(reference["name"], f"{claim_id}.{field}.name")
    relative = PurePosixPath(_relative(reference["path"], f"{claim_id}.{field}.path"))
    path = (root / relative).as_posix()
    digest = _sha256(reference["sha256"], f"{claim_id}.{field}.sha256")
    bundled = artifacts.get(path)
    if (
        bundled is None
        or bundled.name != name
        or bundled.sha256 != digest
        or bundled.kind not in kinds
    ):
        raise ValueError(f"{claim_id} {field} is absent from the bundle")
    output = outputs.get(path)
    if (
        output is None
        or output.name != name
        or output.sha256 != digest
        or output.role != "output"
    ):
        raise ValueError(f"{claim_id} {field} is absent from run-manifest outputs")
    return bundled


def _projection_claims(
    bundle: ClaimBundleV1,
    *,
    root: Path,
    outputs: Mapping[str, Any],
) -> tuple[PaperProjectionClaimV1, ...]:
    binding_artifacts = [
        artifact for artifact in bundle.artifacts if artifact.kind == "claim_binding"
    ]
    if len(binding_artifacts) != 1:
        raise ValueError("paper projection requires exactly one claim-binding artifact")
    payload = _load(_safe(root, binding_artifacts[0].path), "claim binding")
    _exact(payload, _BINDING_ROOT_FIELDS, "claim-evidence bindings")
    if (
        payload["schema_name"] != CLAIM_EVIDENCE_BINDING_SCHEMA
        or payload["schema_version"] != CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION
    ):
        raise ValueError("unsupported claim-evidence binding schema")
    if _sequence(payload["migration_exceptions"], "migration_exceptions"):
        raise ValueError("paper projection cannot contain migration exceptions")

    bindings: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(payload["bindings"], "bindings"):
        binding = _mapping(raw, "claim-evidence binding")
        _exact(binding, _BINDING_FIELDS, "claim-evidence binding")
        claim_id = _text(binding["claim_id"], "binding claim_id")
        if claim_id in bindings:
            raise ValueError(f"duplicate claim-evidence binding: {claim_id}")
        bindings[claim_id] = binding
    if set(bindings) != set(bundle.claim_ids):
        raise ValueError("claim-evidence bindings do not match bundle claim IDs")

    artifacts = _artifact_index(bundle)
    projected: list[PaperProjectionClaimV1] = []
    for claim_id in bundle.claim_ids:
        binding = bindings[claim_id]
        if (
            _sha256(binding["expected_manifest_id"], "expected_manifest_id")
            != bundle.run_manifest_id
            or _sha256(
                binding["expected_evidence_fingerprint"],
                "expected_evidence_fingerprint",
            )
            != bundle.evidence_fingerprint
        ):
            raise ValueError(f"{claim_id} binding selects another run identity")
        artifact_root = _binding_root(binding["artifact_root"], "artifact_root")
        result = _reference(
            binding["result_artifact"],
            claim_id=claim_id,
            field="result_artifact",
            root=artifact_root,
            artifacts=artifacts,
            outputs=outputs,
            kinds=frozenset({"evidence_summary", "supporting"}),
        )
        table = _reference(
            binding["table_artifact"],
            claim_id=claim_id,
            field="table_artifact",
            root=artifact_root,
            artifacts=artifacts,
            outputs=outputs,
            kinds=frozenset({"table_data"}),
        )
        if table.media_type != "application/json":
            raise ValueError(
                "paper projection requires an application/json compact table"
            )
        row_id = _text(binding["table_row_id"], f"{claim_id}.table_row_id")
        evidence = load_compact_claim_table_row(
            _safe(root, table.path), row_id=row_id, claim_id=claim_id
        )
        projected.append(
            PaperProjectionClaimV1(
                claim_id=claim_id,
                result_artifact=result.path,
                result_sha256=result.sha256,
                table_artifact=table.path,
                table_sha256=table.sha256,
                table_row_id=row_id,
                evidence=evidence,
            )
        )
    return tuple(projected)


def build_paper_projection(
    *,
    bundle_path: str | Path,
    artifact_root: str | Path,
) -> PaperProjectionV1:
    """Verify one bundle and project strict paper-facing claim rows."""

    root = Path(artifact_root).resolve()
    bundle = load_claim_bundle(bundle_path)
    manifest = verify_claim_bundle_artifacts(bundle, root=root)
    claims = _projection_claims(bundle, root=root, outputs=_output_index(manifest))
    return PaperProjectionV1(
        bundle_id=bundle.bundle_id,
        run_manifest_id=bundle.run_manifest_id,
        evidence_fingerprint=bundle.evidence_fingerprint,
        run_id=bundle.run_id,
        classification=bundle.classification,
        protocol_id=bundle.protocol_id,
        statistical_unit=bundle.statistical_unit,
        claim_boundary=bundle.claim_boundary,
        claim_ids=bundle.claim_ids,
        method_freeze_id=bundle.method_freeze_id,
        split_id=bundle.split_id,
        baseline_id=bundle.baseline_id,
        repositories=tuple(state.as_dict() for state in bundle.repositories),
        claims=claims,
    )


def _write_atomic(
    path: str | Path,
    payload: bytes,
    *,
    overwrite: bool,
    name: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(f"{name} already exists: {destination}") from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def write_paper_projection(
    path: str | Path,
    projection: PaperProjectionV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish stable projection JSON."""

    payload = (
        json.dumps(projection.as_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _write_atomic(path, payload, overwrite=overwrite, name="paper projection")


def load_paper_projection(path: str | Path) -> PaperProjectionV1:
    """Load a strict projection and reject content-address drift."""

    payload = _load(path, "paper projection")
    _exact(payload, _PROJECTION_FIELDS, "paper projection")
    if (
        payload["schema_name"] != PAPER_PROJECTION_SCHEMA
        or payload["schema_version"] != PAPER_PROJECTION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported paper-projection schema")
    expected_id = _sha256(payload["projection_id"], "projection_id")
    projection = PaperProjectionV1(
        bundle_id=payload["bundle_id"],
        run_manifest_id=payload["run_manifest_id"],
        evidence_fingerprint=payload["evidence_fingerprint"],
        run_id=payload["run_id"],
        classification=payload["classification"],
        protocol_id=payload["protocol_id"],
        statistical_unit=payload["statistical_unit"],
        claim_boundary=payload["claim_boundary"],
        claim_ids=tuple(_sequence(payload["claim_ids"], "claim_ids")),
        method_freeze_id=payload["method_freeze_id"],
        split_id=payload["split_id"],
        baseline_id=payload["baseline_id"],
        repositories=tuple(
            _mapping(value, "paper-projection repository")
            for value in _sequence(payload["repositories"], "repositories")
        ),
        claims=tuple(
            PaperProjectionClaimV1.from_mapping(
                _mapping(value, "paper-projection claim")
            )
            for value in _sequence(payload["claims"], "claims")
        ),
    )
    if projection.projection_id != expected_id:
        raise ValueError("paper-projection digest does not match its payload")
    return projection


def render_paper_projection_markdown(projection: PaperProjectionV1) -> str:
    """Render a deterministic review summary without interpreting evidence."""

    lines = [
        "# BayesianPhysTwin paper projection",
        "",
        f"- Projection ID: `{projection.projection_id}`",
        f"- Claim bundle: `{projection.bundle_id}`",
        f"- Run manifest: `{projection.run_manifest_id}`",
        f"- Evidence fingerprint: `{projection.evidence_fingerprint}`",
        f"- Classification: `{projection.classification}`",
        f"- Protocol: `{projection.protocol_id}`",
        f"- Statistical unit: {projection.statistical_unit}",
        f"- Claim boundary: {projection.claim_boundary}",
        "",
        "## Repository lock",
        "",
    ]
    for repository in projection.repositories:
        lines.append(
            f"- `{repository['repository']}` @ `{repository['revision']}` "
            f"({repository['role']})"
        )
    for claim in projection.claims:
        lines.extend(
            [
                "",
                f"## `{claim.claim_id}`",
                "",
                f"- Result: `{claim.result_artifact}` (`{claim.result_sha256}`)",
                f"- Table: `{claim.table_artifact}` (`{claim.table_sha256}`)",
                f"- Row: `{claim.table_row_id}`",
                "- Evidence:",
                "",
                "```json",
                json.dumps(claim.evidence, indent=2, sort_keys=True, allow_nan=False),
                "```",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_paper_projection_markdown(
    path: str | Path,
    projection: PaperProjectionV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish the human-reviewable projection summary."""

    _write_atomic(
        path,
        render_paper_projection_markdown(projection).encode("utf-8"),
        overwrite=overwrite,
        name="paper-projection markdown",
    )


__all__ = [
    "CLAIM_EVIDENCE_BINDING_SCHEMA",
    "CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION",
    "COMPACT_CLAIM_TABLE_SCHEMA",
    "COMPACT_CLAIM_TABLE_SCHEMA_VERSION",
    "PAPER_PROJECTION_SCHEMA",
    "PAPER_PROJECTION_SCHEMA_VERSION",
    "PaperProjectionClaimV1",
    "PaperProjectionV1",
    "build_paper_projection",
    "load_compact_claim_table_row",
    "load_paper_projection",
    "render_paper_projection_markdown",
    "write_paper_projection",
    "write_paper_projection_markdown",
]
