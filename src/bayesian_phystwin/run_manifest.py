"""Content-addressed manifests for reproducible Bayesian-PhysTwin runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

RUN_MANIFEST_SCHEMA = "bayesian_phystwin.run_manifest"
RUN_MANIFEST_VERSION = 1
RunClassification = Literal[
    "controlled",
    "exploratory",
    "confirmatory",
    "diagnostic",
    "infrastructure",
]
_VALID_CLASSIFICATIONS = frozenset(
    {
        "controlled",
        "exploratory",
        "confirmatory",
        "diagnostic",
        "infrastructure",
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
        return json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def installed_package_versions(names: Sequence[str]) -> dict[str, str]:
    """Return installed versions, using ``not-installed`` for absent packages."""

    versions: dict[str, str] = {}
    for name in sorted(set(map(str, names))):
        if not name:
            raise ValueError("package names must be nonempty")
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


@dataclass(frozen=True)
class ArtifactDigest:
    """Immutable identity of one input or output artifact."""

    name: str
    role: Literal["input", "output"]
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise ValueError("artifact name and path must be nonempty")
        if self.role not in {"input", "output"}:
            raise ValueError("artifact role must be 'input' or 'output'")
        if int(self.size_bytes) < 0:
            raise ValueError("artifact size must be nonnegative")
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256, name="sha256"))
        object.__setattr__(self, "size_bytes", int(self.size_bytes))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RunManifestV1:
    """Versioned run identity and evidence-boundary record."""

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
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.run_id or not self.repository or not self.revision:
            raise ValueError("run identity fields must be nonempty")
        if not self.command or any(not token for token in self.command):
            raise ValueError("command must contain nonempty tokens")
        if self.classification not in _VALID_CLASSIFICATIONS:
            raise ValueError("unknown run classification")
        if not self.statistical_unit:
            raise ValueError("statistical_unit must be nonempty")
        try:
            datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_utc must be an ISO-8601 timestamp") from error

        seeds = tuple(int(seed) for seed in self.seeds)
        input_artifacts = tuple(self.inputs)
        output_artifacts = tuple(self.outputs)
        names = [artifact.name for artifact in (*input_artifacts, *output_artifacts)]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique within one run")
        if any(artifact.role != "input" for artifact in input_artifacts):
            raise ValueError("inputs must contain only input artifacts")
        if any(artifact.role != "output" for artifact in output_artifacts):
            raise ValueError("outputs must contain only output artifacts")

        versions = {str(name): str(value) for name, value in self.package_versions.items()}
        if any(not name or not value for name, value in versions.items()):
            raise ValueError("package version entries must be nonempty")

        object.__setattr__(self, "command", tuple(map(str, self.command)))
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "inputs", input_artifacts)
        object.__setattr__(self, "outputs", output_artifacts)
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
        object.__setattr__(self, "package_versions", dict(sorted(versions.items())))

    def descriptor(self) -> dict[str, object]:
        """Return the canonical payload covered by ``manifest_id``."""

        return {
            "schema_name": RUN_MANIFEST_SCHEMA,
            "schema_version": RUN_MANIFEST_VERSION,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "repository": self.repository,
            "revision": self.revision,
            "dirty": bool(self.dirty),
            "command": list(self.command),
            "classification": self.classification,
            "statistical_unit": self.statistical_unit,
            "information_boundary": dict(self.information_boundary),
            "configuration": dict(self.configuration),
            "seeds": list(self.seeds),
            "inputs": [artifact.as_dict() for artifact in self.inputs],
            "outputs": [artifact.as_dict() for artifact in self.outputs],
            "package_versions": dict(self.package_versions),
            "notes": self.notes,
        }

    @property
    def manifest_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self.descriptor()}


def artifact_digest(
    path: str | Path,
    *,
    name: str,
    role: Literal["input", "output"],
    root: str | Path | None = None,
) -> ArtifactDigest:
    """Hash one artifact and store a portable path when ``root`` is supplied."""

    source = Path(path)
    stored_path = source
    if root is not None:
        stored_path = source.resolve().relative_to(Path(root).resolve())
    return ArtifactDigest(
        name=name,
        role=role,
        path=stored_path.as_posix(),
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
    )


def write_run_manifest(path: str | Path, manifest: RunManifestV1) -> None:
    """Write a stable, human-readable JSON manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _artifact_from_dict(value: Mapping[str, Any]) -> ArtifactDigest:
    return ArtifactDigest(
        name=str(value["name"]),
        role=str(value["role"]),  # type: ignore[arg-type]
        path=str(value["path"]),
        sha256=str(value["sha256"]),
        size_bytes=int(value["size_bytes"]),
    )


def load_run_manifest(path: str | Path) -> RunManifestV1:
    """Load a manifest and reject schema, content-address, or field drift."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_name") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported run-manifest schema")
    if int(payload.get("schema_version", -1)) != RUN_MANIFEST_VERSION:
        raise ValueError("unsupported run-manifest version")
    expected = _validate_sha256(str(payload.get("manifest_id", "")), name="manifest_id")
    manifest = RunManifestV1(
        run_id=str(payload["run_id"]),
        repository=str(payload["repository"]),
        revision=str(payload["revision"]),
        dirty=bool(payload["dirty"]),
        command=tuple(map(str, payload["command"])),
        classification=str(payload["classification"]),  # type: ignore[arg-type]
        statistical_unit=str(payload["statistical_unit"]),
        information_boundary=dict(payload["information_boundary"]),
        configuration=dict(payload["configuration"]),
        seeds=tuple(map(int, payload.get("seeds", ()))),
        inputs=tuple(_artifact_from_dict(value) for value in payload.get("inputs", ())),
        outputs=tuple(
            _artifact_from_dict(value) for value in payload.get("outputs", ())
        ),
        package_versions=dict(payload.get("package_versions", {})),
        created_utc=str(payload["created_utc"]),
        notes=str(payload.get("notes", "")),
    )
    if manifest.manifest_id != expected:
        raise ValueError("run manifest digest does not match its payload")
    return manifest


def verify_run_manifest_artifacts(
    manifest: RunManifestV1,
    *,
    root: str | Path,
) -> None:
    """Verify every declared artifact against one extraction or run root."""

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
    "RUN_MANIFEST_SCHEMA",
    "RUN_MANIFEST_VERSION",
    "ArtifactDigest",
    "RunClassification",
    "RunManifestV1",
    "artifact_digest",
    "installed_package_versions",
    "load_run_manifest",
    "sha256_file",
    "verify_run_manifest_artifacts",
    "write_run_manifest",
]
