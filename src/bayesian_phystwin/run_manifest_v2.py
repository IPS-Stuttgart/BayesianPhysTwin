"""Evidence-bound run manifests with backward-compatible V1 loading."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from ._canonical_contracts import (
    canonical_relative_posix_path,
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    sha256_digest,
    write_atomic_json,
)
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


def _require_exact_fields(
    value: Mapping[Any, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string fields")
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"{name} does not match schema: {', '.join(details)}")


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


def _require_literal_string(
    value: Any,
    *,
    name: str,
    allow_empty: bool,
) -> str:
    if (
        not isinstance(value, str)
        or type(value) is not str
        or (not allow_empty and not value)
    ):
        qualifier = "a literal string" if allow_empty else "a nonempty literal string"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_literal_string_sequence(
    value: Any,
    *,
    name: str,
    allow_empty_sequence: bool,
) -> tuple[str, ...]:
    sequence = _require_sequence(value, name=name)
    if not allow_empty_sequence and not sequence:
        raise ValueError(f"{name} must contain nonempty strings")
    result = tuple(
        _require_literal_string(
            item,
            name=f"{name}[{index}]",
            allow_empty=False,
        )
        for index, item in enumerate(sequence)
    )
    return result


def _validate_artifact_object(artifact: ArtifactDigest, *, name: str) -> None:
    if type(artifact) is not ArtifactDigest:
        raise ValueError(f"{name} must contain ArtifactDigest values")
    _require_literal_string(artifact.name, name=f"{name} name", allow_empty=False)
    canonical_relative_posix_path(artifact.path, name=f"{name} path")
    role = _require_literal_string(
        artifact.role,
        name=f"{name} role",
        allow_empty=False,
    )
    if role not in {"input", "output"}:
        raise ValueError("artifact role must be 'input' or 'output'")
    sha256_digest(artifact.sha256, name=f"{name} sha256")
    genuine_integer(artifact.size_bytes, name=f"{name} size_bytes", minimum=0)


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
        if type(self.run_id) is not str or not self.run_id:
            raise ValueError("run ID must be nonempty")
        if (
            type(self.repository) is not str
            or self.repository != self.repository.strip()
        ):
            raise ValueError("repository must use owner/name")
        repository = self.repository
        parts = repository.split("/")
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError("repository must use owner/name")
        dirty = _require_bool(self.dirty, name="dirty")
        command = _require_literal_string_sequence(
            self.command,
            name="command",
            allow_empty_sequence=False,
        )
        classification = _require_literal_string(
            self.classification,
            name="classification",
            allow_empty=False,
        )
        if classification not in _VALID_CLASSIFICATIONS:
            raise ValueError("unknown run classification")
        statistical_unit = _require_literal_string(
            self.statistical_unit,
            name="statistical_unit",
            allow_empty=False,
        )
        created_utc = _require_literal_string(
            self.created_utc,
            name="created_utc",
            allow_empty=False,
        )
        try:
            created = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_utc must be an ISO-8601 timestamp") from error
        if created.tzinfo is None or created.utcoffset() is None:
            raise ValueError("created_utc must include a timezone")
        # Preserve the original timezone-aware spelling so previously valid V2
        # content addresses remain stable. The default factory already emits UTC.

        input_artifacts: tuple[ArtifactDigest, ...] = tuple(
            cast(
                Sequence[ArtifactDigest],
                _require_sequence(self.inputs, name="inputs"),
            )
        )
        output_artifacts: tuple[ArtifactDigest, ...] = tuple(
            cast(
                Sequence[ArtifactDigest],
                _require_sequence(self.outputs, name="outputs"),
            )
        )
        for index, artifact in enumerate(input_artifacts):
            _validate_artifact_object(artifact, name=f"inputs[{index}]")
        for index, artifact in enumerate(output_artifacts):
            _validate_artifact_object(artifact, name=f"outputs[{index}]")
        names = [artifact.name for artifact in (*input_artifacts, *output_artifacts)]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique within one run")
        if any(artifact.role != "input" for artifact in input_artifacts):
            raise ValueError("inputs must contain only input artifacts")
        if any(artifact.role != "output" for artifact in output_artifacts):
            raise ValueError("outputs must contain only output artifacts")

        related: tuple[RepositoryState, ...] = tuple(
            cast(
                Sequence[RepositoryState],
                _require_sequence(
                    self.related_repositories,
                    name="related_repositories",
                ),
            )
        )
        if any(type(state) is not RepositoryState for state in related):
            raise ValueError("related_repositories must contain RepositoryState values")
        repository_names = [repository, *(state.repository for state in related)]
        if len(repository_names) != len(set(repository_names)):
            raise ValueError("repository states must have unique repository names")
        if any(state.role == "primary" for state in related):
            raise ValueError("related repositories cannot use the primary role")

        claims = _require_literal_string_sequence(
            self.claim_ids,
            name="claim_ids",
            allow_empty_sequence=True,
        )
        if len(claims) != len(set(claims)):
            raise ValueError("claim_ids must be unique nonempty identifiers")

        if not isinstance(self.package_versions, Mapping):
            raise ValueError("package_versions must be a mapping")
        versions: dict[str, str] = {}
        for name, value in self.package_versions.items():
            package_name = _require_literal_string(
                name,
                name="package version name",
                allow_empty=False,
            )
            package_value = _require_literal_string(
                value,
                name=f"package version {package_name}",
                allow_empty=False,
            )
            versions[package_name] = package_value

        seeds = tuple(
            genuine_integer(seed, name=f"seeds[{index}]")
            for index, seed in enumerate(_require_sequence(self.seeds, name="seeds"))
        )
        method_freeze_id = _require_literal_string(
            self.method_freeze_id,
            name="method_freeze_id",
            allow_empty=True,
        )
        protocol_id = _require_literal_string(
            self.protocol_id,
            name="protocol_id",
            allow_empty=True,
        )
        split_id = _require_literal_string(
            self.split_id,
            name="split_id",
            allow_empty=True,
        )
        baseline_id = _require_literal_string(
            self.baseline_id,
            name="baseline_id",
            allow_empty=True,
        )
        notes = _require_literal_string(self.notes, name="notes", allow_empty=True)

        object.__setattr__(self, "run_id", self.run_id)
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "revision", validate_revision(self.revision))
        object.__setattr__(self, "dirty", dirty)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "statistical_unit", statistical_unit)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "inputs", input_artifacts)
        object.__setattr__(self, "outputs", output_artifacts)
        object.__setattr__(self, "related_repositories", related)
        object.__setattr__(self, "claim_ids", claims)
        object.__setattr__(self, "method_freeze_id", method_freeze_id)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "split_id", split_id)
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "created_utc", created_utc)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(
            self,
            "information_boundary",
            frozen_finite_json_mapping(
                self.information_boundary,
                name="information_boundary",
            ),
        )
        object.__setattr__(
            self,
            "configuration",
            frozen_finite_json_mapping(self.configuration, name="configuration"),
        )
        object.__setattr__(
            self,
            "runtime_environment",
            frozen_finite_json_mapping(
                self.runtime_environment,
                name="runtime_environment",
            ),
        )
        object.__setattr__(
            self,
            "package_versions",
            frozen_finite_json_mapping(versions, name="package_versions"),
        )

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
            "information_boundary": plain_json(self.information_boundary),
            "configuration": plain_json(self.configuration),
            "seeds": list(self.seeds),
            "inputs": [artifact.as_dict() for artifact in self.inputs],
            "outputs": [artifact.as_dict() for artifact in self.outputs],
            "package_versions": plain_json(self.package_versions),
            "runtime_environment": plain_json(self.runtime_environment),
            "claim_ids": list(self.claim_ids),
            "method_freeze_id": self.method_freeze_id,
            "protocol_id": self.protocol_id,
            "split_id": self.split_id,
            "baseline_id": self.baseline_id,
        }

    @property
    def evidence_fingerprint(self) -> str:
        """Stable identity for scientifically equivalent manifest instances."""

        return content_id(self.scientific_descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            **self.scientific_descriptor(),
            "evidence_fingerprint": self.evidence_fingerprint,
            "created_utc": self.created_utc,
            "notes": self.notes,
        }

    @property
    def manifest_id(self) -> str:
        return content_id(self.descriptor())

    def as_dict(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self.descriptor()}


RunManifest: TypeAlias = RunManifestV1 | RunManifestV2


def write_run_manifest(
    path: str | Path,
    manifest: RunManifestV2,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish and verify a stable V2 JSON manifest.

    Existing evidence records are preserved by default. Callers must opt in to
    atomic replacement with ``overwrite=True``.
    """

    if type(manifest) is not RunManifestV2:
        raise TypeError("manifest must be an exact RunManifestV2 instance")
    overwrite_value = _require_bool(overwrite, name="overwrite")
    destination = Path(path)
    write_atomic_json(
        manifest.as_dict(),
        destination,
        overwrite=overwrite_value,
    )
    published = load_run_manifest_v2(destination)
    if (
        published.manifest_id != manifest.manifest_id
        or published.evidence_fingerprint != manifest.evidence_fingerprint
    ):
        raise RuntimeError("published run manifest failed post-write verification")


def _artifact_from_dict(value: Mapping[str, Any]) -> ArtifactDigest:
    _require_exact_fields(value, expected=_ARTIFACT_FIELDS, name="artifact record")
    role = _require_literal_string(
        value["role"],
        name="artifact role",
        allow_empty=False,
    )
    if role not in {"input", "output"}:
        raise ValueError("artifact role must be 'input' or 'output'")
    return ArtifactDigest(
        name=_require_literal_string(
            value["name"],
            name="artifact name",
            allow_empty=False,
        ),
        role=cast(Literal["input", "output"], role),
        path=canonical_relative_posix_path(
            value["path"],
            name="artifact path",
        ),
        sha256=sha256_digest(value["sha256"], name="artifact sha256"),
        size_bytes=genuine_integer(
            value["size_bytes"],
            name="artifact size_bytes",
            minimum=0,
        ),
    )


def _repository_from_dict(value: Mapping[str, Any]) -> RepositoryState:
    _require_exact_fields(value, expected=_REPOSITORY_FIELDS, name="repository record")
    role = _require_literal_string(
        value["role"],
        name="repository role",
        allow_empty=False,
    )
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
        repository=_require_literal_string(
            value["repository"],
            name="repository name",
            allow_empty=False,
        ),
        revision=_require_literal_string(
            value["revision"],
            name="repository revision",
            allow_empty=False,
        ),
        dirty=_require_bool(value["dirty"], name="repository dirty"),
        role=cast(RepositoryRole, role),
    )


def _manifest_from_payload(payload: Mapping[str, Any]) -> RunManifestV2:
    _require_exact_fields(
        payload,
        expected=_RUN_MANIFEST_V2_FIELDS,
        name="run manifest",
    )
    schema_name = _require_literal_string(
        payload["schema_name"],
        name="schema_name",
        allow_empty=False,
    )
    if schema_name != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported run-manifest schema")
    if (
        genuine_integer(payload["schema_version"], name="schema_version")
        != RUN_MANIFEST_V2_VERSION
    ):
        raise ValueError("unsupported run-manifest version")
    expected_manifest = sha256_digest(payload["manifest_id"], name="manifest_id")
    expected_evidence = sha256_digest(
        payload["evidence_fingerprint"],
        name="evidence_fingerprint",
    )
    classification = _require_literal_string(
        payload["classification"],
        name="classification",
        allow_empty=False,
    )
    if classification not in _VALID_CLASSIFICATIONS:
        raise ValueError("unknown run classification")
    manifest = RunManifestV2(
        run_id=_require_literal_string(
            payload["run_id"],
            name="run_id",
            allow_empty=False,
        ),
        repository=_require_literal_string(
            payload["repository"],
            name="repository",
            allow_empty=False,
        ),
        revision=_require_literal_string(
            payload["revision"],
            name="revision",
            allow_empty=False,
        ),
        dirty=_require_bool(payload["dirty"], name="dirty"),
        related_repositories=tuple(
            _repository_from_dict(_require_mapping(value, name="repository record"))
            for value in _require_sequence(
                payload["related_repositories"],
                name="related_repositories",
            )
        ),
        command=_require_literal_string_sequence(
            payload["command"],
            name="command",
            allow_empty_sequence=False,
        ),
        classification=cast(RunClassification, classification),
        statistical_unit=_require_literal_string(
            payload["statistical_unit"],
            name="statistical_unit",
            allow_empty=False,
        ),
        information_boundary=dict(
            _require_mapping(
                payload["information_boundary"],
                name="information_boundary",
            )
        ),
        configuration=dict(
            _require_mapping(payload["configuration"], name="configuration")
        ),
        seeds=tuple(
            genuine_integer(value, name=f"seeds[{index}]")
            for index, value in enumerate(
                _require_sequence(payload["seeds"], name="seeds")
            )
        ),
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
        claim_ids=_require_literal_string_sequence(
            payload["claim_ids"],
            name="claim_ids",
            allow_empty_sequence=True,
        ),
        method_freeze_id=_require_literal_string(
            payload["method_freeze_id"],
            name="method_freeze_id",
            allow_empty=True,
        ),
        protocol_id=_require_literal_string(
            payload["protocol_id"],
            name="protocol_id",
            allow_empty=True,
        ),
        split_id=_require_literal_string(
            payload["split_id"],
            name="split_id",
            allow_empty=True,
        ),
        baseline_id=_require_literal_string(
            payload["baseline_id"],
            name="baseline_id",
            allow_empty=True,
        ),
        created_utc=_require_literal_string(
            payload["created_utc"],
            name="created_utc",
            allow_empty=False,
        ),
        notes=_require_literal_string(
            payload["notes"],
            name="notes",
            allow_empty=True,
        ),
    )
    if manifest.evidence_fingerprint != expected_evidence:
        raise ValueError("run manifest evidence fingerprint does not match its payload")
    if manifest.manifest_id != expected_manifest:
        raise ValueError("run manifest digest does not match its payload")
    return manifest


def load_run_manifest_v2(path: str | Path) -> RunManifestV2:
    """Load a V2 manifest and reject schema or content-address drift."""

    return _manifest_from_payload(load_strict_json_object(path, label="run manifest"))


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load either the released V1 schema or the evidence-bound V2 schema."""

    # Preserve the released V1 dispatch behavior exactly. V2 is reparsed by the
    # strict loader before any payload value is trusted.
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
