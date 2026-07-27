"""Evidence-bound run manifests with backward-compatible V1 loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from .repository_provenance import (
    RepositoryRole,
    RepositoryState,
    validate_revision,
)
from .run_manifest import (
    RUN_MANIFEST_SCHEMA,
    ArtifactDigest,
    RunClassification,
    RunManifestV1,
    sha256_file,
)
from .run_manifest import (
    load_run_manifest as load_run_manifest_v1,
)

RUN_MANIFEST_V2_VERSION = 2
_VALID_CLASSIFICATIONS = frozenset(
    {
        "controlled",
        "exploratory",
        "confirmatory",
        "diagnostic",
        "infrastructure",
    }
)
_ARTIFACT_FIELDS = frozenset({"name", "role", "path", "sha256", "size_bytes"})
_REPOSITORY_FIELDS = frozenset({"repository", "revision", "dirty", "role"})
_RUN_MANIFEST_V2_FIELDS = frozenset(
    {
        "manifest_id",
        "evidence_fingerprint",
        "schema_name",
        "schema_version",
        "run_id",
        "created_utc",
        "repository",
        "revision",
        "dirty",
        "related_repositories",
        "command",
        "classification",
        "statistical_unit",
        "information_boundary",
        "configuration",
        "seeds",
        "inputs",
        "outputs",
        "package_versions",
        "runtime_environment",
        "claim_ids",
        "method_freeze_id",
        "protocol_id",
        "split_id",
        "baseline_id",
        "notes",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return cast(
            dict[str, Any],
            json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False)),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(map(str, value))
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"{name} does not match schema: {', '.join(details)}")


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


@dataclass(frozen=True)
class RunManifestV2:
    """Complete cross-repository provenance for one finalized result."""

    run_id: str
    repository: str
    revision: str
    dirty: bool
    command: tuple[str, ...]
    classification: RunClassification
    statistical_unit: str
    information_boundary: Mapping[str, Any]
    configuration: Mapping[str, Any]
    seeds: tuple[int, ...] = ()
    inputs: tuple[ArtifactDigest, ...] = ()
    outputs: tuple[ArtifactDigest, ...] = ()
    package_versions: Mapping[str, str] = field(default_factory=dict)
    related_repositories: tuple[RepositoryState, ...] = ()
    runtime_environment: Mapping[str, Any] = field(default_factory=dict)
    claim_ids: tuple[str, ...] = ()
    method_freeze_id: str = ""
    protocol_id: str = ""
    split_id: str = ""
    baseline_id: str = ""
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def __post_init__(self) -> None:
        repository = str(self.repository).strip()
        parts = repository.split("/")
        if not self.run_id or len(parts) != 2 or any(not part for part in parts):
            raise ValueError(
                "run ID must be nonempty and repository must use owner/name"
            )
        if not isinstance(self.dirty, bool):
            raise ValueError("dirty must be boolean")
        if not self.command or any(not str(token) for token in self.command):
            raise ValueError("command must contain nonempty tokens")
        if self.classification not in _VALID_CLASSIFICATIONS:
            raise ValueError("unknown run classification")
        if not self.statistical_unit:
            raise ValueError("statistical_unit must be nonempty")
        try:
            created = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_utc must be an ISO-8601 timestamp") from error
        if created.tzinfo is None:
            raise ValueError("created_utc must include a timezone")

        input_artifacts = tuple(self.inputs)
        output_artifacts = tuple(self.outputs)
        names = [artifact.name for artifact in (*input_artifacts, *output_artifacts)]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique within one run")
        if any(artifact.role != "input" for artifact in input_artifacts):
            raise ValueError("inputs must contain only input artifacts")
        if any(artifact.role != "output" for artifact in output_artifacts):
            raise ValueError("outputs must contain only output artifacts")

        related = tuple(self.related_repositories)
        repository_names = [repository, *(state.repository for state in related)]
        if len(repository_names) != len(set(repository_names)):
            raise ValueError("repository states must have unique repository names")
        if any(state.role == "primary" for state in related):
            raise ValueError("related repositories cannot use the primary role")

        claims = tuple(map(str, self.claim_ids))
        if any(not claim for claim in claims) or len(claims) != len(set(claims)):
            raise ValueError("claim_ids must be unique nonempty identifiers")

        versions = {
            str(name): str(value) for name, value in self.package_versions.items()
        }
        if any(not name or not value for name, value in versions.items()):
            raise ValueError("package version entries must be nonempty")

        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "revision", validate_revision(self.revision))
        object.__setattr__(self, "command", tuple(map(str, self.command)))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "inputs", input_artifacts)
        object.__setattr__(self, "outputs", output_artifacts)
        object.__setattr__(self, "related_repositories", related)
        object.__setattr__(self, "claim_ids", claims)
        object.__setattr__(
            self,
            "information_boundary",
            _json_mapping(self.information_boundary, name="information_boundary"),
        )
        object.__setattr__(
            self,
            "configuration",
            _json_mapping(self.configuration, name="configuration"),
        )
        object.__setattr__(
            self,
            "runtime_environment",
            _json_mapping(self.runtime_environment, name="runtime_environment"),
        )
        object.__setattr__(self, "package_versions", dict(sorted(versions.items())))

    def scientific_descriptor(self) -> dict[str, object]:
        """Return the timestamp- and note-independent evidence identity."""

        return {
            "schema_name": RUN_MANIFEST_SCHEMA,
            "schema_version": RUN_MANIFEST_V2_VERSION,
            "run_id": self.run_id,
            "repository": self.repository,
            "revision": self.revision,
            "dirty": self.dirty,
            "related_repositories": [
                state.as_dict() for state in self.related_repositories
            ],
            "command": list(self.command),
            "classification": self.classification,
            "statistical_unit": self.statistical_unit,
            "information_boundary": dict(self.information_boundary),
            "configuration": dict(self.configuration),
            "seeds": list(self.seeds),
            "inputs": [artifact.as_dict() for artifact in self.inputs],
            "outputs": [artifact.as_dict() for artifact in self.outputs],
            "package_versions": dict(self.package_versions),
            "runtime_environment": dict(self.runtime_environment),
            "claim_ids": list(self.claim_ids),
            "method_freeze_id": self.method_freeze_id,
            "protocol_id": self.protocol_id,
            "split_id": self.split_id,
            "baseline_id": self.baseline_id,
        }

    @property
    def evidence_fingerprint(self) -> str:
        """Stable identity for scientifically equivalent manifest instances."""

        return hashlib.sha256(_canonical_json(self.scientific_descriptor())).hexdigest()

    def descriptor(self) -> dict[str, object]:
        return {
            **self.scientific_descriptor(),
            "evidence_fingerprint": self.evidence_fingerprint,
            "created_utc": self.created_utc,
            "notes": self.notes,
        }

    @property
    def manifest_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self.descriptor()}


RunManifest: TypeAlias = RunManifestV1 | RunManifestV2


def write_run_manifest(path: str | Path, manifest: RunManifestV2) -> None:
    """Write a stable, human-readable V2 JSON manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _artifact_from_dict(value: Mapping[str, Any]) -> ArtifactDigest:
    _require_exact_fields(value, expected=_ARTIFACT_FIELDS, name="artifact record")
    role = str(value["role"])
    if role not in {"input", "output"}:
        raise ValueError("artifact role must be 'input' or 'output'")
    return ArtifactDigest(
        name=str(value["name"]),
        role=cast(Literal["input", "output"], role),
        path=str(value["path"]),
        sha256=str(value["sha256"]),
        size_bytes=int(value["size_bytes"]),
    )


def _repository_from_dict(value: Mapping[str, Any]) -> RepositoryState:
    _require_exact_fields(value, expected=_REPOSITORY_FIELDS, name="repository record")
    role = str(value["role"])
    valid_roles = {
        "primary",
        "upstream",
        "observation",
        "downstream",
        "paper",
        "environment",
        "dependency",
    }
    if role not in valid_roles:
        raise ValueError("unknown repository role")
    return RepositoryState(
        repository=str(value["repository"]),
        revision=str(value["revision"]),
        dirty=_require_bool(value["dirty"], name="repository dirty"),
        role=cast(RepositoryRole, role),
    )


def load_run_manifest_v2(path: str | Path) -> RunManifestV2:
    """Load a V2 manifest and reject schema or content-address drift."""

    payload_value = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = _require_mapping(payload_value, name="run manifest")
    _require_exact_fields(
        payload,
        expected=_RUN_MANIFEST_V2_FIELDS,
        name="run manifest",
    )
    if payload.get("schema_name") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported run-manifest schema")
    if int(payload.get("schema_version", -1)) != RUN_MANIFEST_V2_VERSION:
        raise ValueError("unsupported run-manifest version")
    expected_manifest = _validate_sha256(
        str(payload["manifest_id"]),
        name="manifest_id",
    )
    expected_evidence = _validate_sha256(
        str(payload["evidence_fingerprint"]),
        name="evidence_fingerprint",
    )
    classification = str(payload["classification"])
    if classification not in _VALID_CLASSIFICATIONS:
        raise ValueError("unknown run classification")
    manifest = RunManifestV2(
        run_id=str(payload["run_id"]),
        repository=str(payload["repository"]),
        revision=str(payload["revision"]),
        dirty=_require_bool(payload["dirty"], name="dirty"),
        related_repositories=tuple(
            _repository_from_dict(_require_mapping(value, name="repository record"))
            for value in _require_sequence(
                payload["related_repositories"],
                name="related_repositories",
            )
        ),
        command=tuple(map(str, _require_sequence(payload["command"], name="command"))),
        classification=cast(RunClassification, classification),
        statistical_unit=str(payload["statistical_unit"]),
        information_boundary=dict(
            _require_mapping(
                payload["information_boundary"],
                name="information_boundary",
            )
        ),
        configuration=dict(
            _require_mapping(payload["configuration"], name="configuration")
        ),
        seeds=tuple(map(int, _require_sequence(payload["seeds"], name="seeds"))),
        inputs=tuple(
            _artifact_from_dict(_require_mapping(value, name="input artifact"))
            for value in _require_sequence(payload["inputs"], name="inputs")
        ),
        outputs=tuple(
            _artifact_from_dict(_require_mapping(value, name="output artifact"))
            for value in _require_sequence(payload["outputs"], name="outputs")
        ),
        package_versions=dict(
            _require_mapping(payload["package_versions"], name="package_versions")
        ),
        runtime_environment=dict(
            _require_mapping(
                payload["runtime_environment"],
                name="runtime_environment",
            )
        ),
        claim_ids=tuple(
            map(str, _require_sequence(payload["claim_ids"], name="claim_ids"))
        ),
        method_freeze_id=str(payload["method_freeze_id"]),
        protocol_id=str(payload["protocol_id"]),
        split_id=str(payload["split_id"]),
        baseline_id=str(payload["baseline_id"]),
        created_utc=str(payload["created_utc"]),
        notes=str(payload["notes"]),
    )
    if manifest.evidence_fingerprint != expected_evidence:
        raise ValueError("run manifest evidence fingerprint does not match its payload")
    if manifest.manifest_id != expected_manifest:
        raise ValueError("run manifest digest does not match its payload")
    return manifest


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load either the released V1 schema or the evidence-bound V2 schema."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run manifest must contain a JSON object")
    if payload.get("schema_name") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported run-manifest schema")
    version = int(payload.get("schema_version", -1))
    if version == 1:
        return load_run_manifest_v1(path)
    if version == RUN_MANIFEST_V2_VERSION:
        return load_run_manifest_v2(path)
    raise ValueError("unsupported run-manifest version")


def verify_run_manifest_artifacts(
    manifest: RunManifest,
    *,
    root: str | Path,
) -> None:
    """Verify every V1 or V2 artifact against one extraction or run root."""

    base = Path(root).resolve()
    for artifact in (*manifest.inputs, *manifest.outputs):
        candidate = (base / artifact.path).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as error:
            raise ValueError(
                f"artifact path escapes verification root: {artifact.path}"
            ) from error
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size != artifact.size_bytes:
            raise ValueError(f"artifact size mismatch: {artifact.name}")
        if sha256_file(candidate) != artifact.sha256:
            raise ValueError(f"artifact digest mismatch: {artifact.name}")


__all__ = [
    "RUN_MANIFEST_V2_VERSION",
    "RunManifest",
    "RunManifestV2",
    "load_run_manifest",
    "load_run_manifest_v2",
    "verify_run_manifest_artifacts",
    "write_run_manifest",
]
