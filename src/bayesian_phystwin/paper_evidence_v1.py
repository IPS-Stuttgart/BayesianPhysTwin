"""Validated paper-evidence bindings layered on :mod:`run_manifest_v2`.

The profile intentionally lives inside ``RunManifestV2.information_boundary``.
That preserves the released V2 schema while making the provider, stream,
artifact, distribution, and paper-claim semantics machine-checkable and part of
the existing evidence fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .run_manifest import ArtifactDigest
from .run_manifest_v2 import RunManifestV2

PAPER_EVIDENCE_PROFILE_KEY = "paper_evidence_bindings_v1"
PAPER_EVIDENCE_SCHEMA = "bayesian_phystwin.paper_evidence_bindings"
PAPER_EVIDENCE_SCHEMA_VERSION = 1

ArtifactRole = Literal["input", "output"]
DistributionKind = Literal["wheel", "sdist"]
StreamResolution = Literal["declared", "inferred", "not_applicable"]

_PROFILE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "primary_distribution_project",
        "provider_manifest",
        "prob4d_stream_contract",
        "observation_belief",
        "twin_belief",
        "distributions",
    }
)
_ARTIFACT_BINDING_FIELDS = frozenset({"artifact_name", "artifact_id", "role"})
_STREAM_BINDING_FIELDS = frozenset({"version", "resolution"})
_DISTRIBUTION_BINDING_FIELDS = frozenset(
    {"project", "kind", "artifact_name", "artifact_id"}
)


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


def _require_nonempty(value: Any, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class ArtifactBindingV1:
    """Bind one semantic artifact to a named run-manifest artifact record."""

    artifact_name: str
    artifact_id: str
    role: ArtifactRole

    def __post_init__(self) -> None:
        name = _require_nonempty(self.artifact_name, name="artifact_name")
        artifact_id = _require_sha256(self.artifact_id, name="artifact_id")
        if self.role not in {"input", "output"}:
            raise ValueError("artifact binding role must be 'input' or 'output'")
        object.__setattr__(self, "artifact_name", name)
        object.__setattr__(self, "artifact_id", artifact_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_id": self.artifact_id,
            "role": self.role,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactBindingV1:
        _require_exact_fields(
            value,
            expected=_ARTIFACT_BINDING_FIELDS,
            name="paper-evidence artifact binding",
        )
        role = str(value["role"])
        if role not in {"input", "output"}:
            raise ValueError("artifact binding role must be 'input' or 'output'")
        return cls(
            artifact_name=str(value["artifact_name"]),
            artifact_id=str(value["artifact_id"]),
            role=cast(ArtifactRole, role),
        )


@dataclass(frozen=True)
class Prob4DStreamBindingV1:
    """Record the resolved Prob4D stream contract and how it was resolved."""

    version: int | None
    resolution: StreamResolution

    def __post_init__(self) -> None:
        if self.resolution not in {"declared", "inferred", "not_applicable"}:
            raise ValueError("unsupported Prob4D stream-contract resolution")
        if self.resolution == "not_applicable":
            if self.version is not None:
                raise ValueError(
                    "not-applicable Prob4D stream contract cannot have a version"
                )
            return
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError(
                "declared or inferred Prob4D stream contract requires an "
                "integer version"
            )
        if self.version < 1:
            raise ValueError("Prob4D stream-contract version must be positive")

    def as_dict(self) -> dict[str, object]:
        return {"version": self.version, "resolution": self.resolution}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Prob4DStreamBindingV1:
        _require_exact_fields(
            value,
            expected=_STREAM_BINDING_FIELDS,
            name="Prob4D stream binding",
        )
        raw_version = value["version"]
        if raw_version is not None and (
            isinstance(raw_version, bool) or not isinstance(raw_version, int)
        ):
            raise ValueError(
                "Prob4D stream-contract version must be an integer or null"
            )
        resolution = str(value["resolution"])
        if resolution not in {"declared", "inferred", "not_applicable"}:
            raise ValueError("unsupported Prob4D stream-contract resolution")
        return cls(
            version=cast(int | None, raw_version),
            resolution=cast(StreamResolution, resolution),
        )


@dataclass(frozen=True)
class DistributionBindingV1:
    """Bind a built wheel or source distribution to its input artifact."""

    project: str
    kind: DistributionKind
    artifact_name: str
    artifact_id: str

    def __post_init__(self) -> None:
        project = _require_nonempty(self.project, name="distribution project")
        artifact_name = _require_nonempty(
            self.artifact_name,
            name="distribution artifact_name",
        )
        artifact_id = _require_sha256(
            self.artifact_id,
            name="distribution artifact_id",
        )
        if self.kind not in {"wheel", "sdist"}:
            raise ValueError("distribution kind must be 'wheel' or 'sdist'")
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "artifact_name", artifact_name)
        object.__setattr__(self, "artifact_id", artifact_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "project": self.project,
            "kind": self.kind,
            "artifact_name": self.artifact_name,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DistributionBindingV1:
        _require_exact_fields(
            value,
            expected=_DISTRIBUTION_BINDING_FIELDS,
            name="distribution binding",
        )
        kind = str(value["kind"])
        if kind not in {"wheel", "sdist"}:
            raise ValueError("distribution kind must be 'wheel' or 'sdist'")
        return cls(
            project=str(value["project"]),
            kind=cast(DistributionKind, kind),
            artifact_name=str(value["artifact_name"]),
            artifact_id=str(value["artifact_id"]),
        )


@dataclass(frozen=True)
class PaperEvidenceBindingsV1:
    """Semantic evidence profile embedded into a ``RunManifestV2``."""

    primary_distribution_project: str
    provider_manifest: ArtifactBindingV1
    prob4d_stream_contract: Prob4DStreamBindingV1
    observation_belief: ArtifactBindingV1
    twin_belief: ArtifactBindingV1
    distributions: tuple[DistributionBindingV1, ...]

    def __post_init__(self) -> None:
        project = _require_nonempty(
            self.primary_distribution_project,
            name="primary_distribution_project",
        )
        distributions = tuple(self.distributions)
        if self.provider_manifest.role != "input":
            raise ValueError("provider manifest must be bound as an input artifact")
        if self.observation_belief.role != "input":
            raise ValueError("observation belief must be bound as an input artifact")
        if not distributions:
            raise ValueError("paper-evidence profile requires distribution artifacts")
        artifact_names = [item.artifact_name for item in distributions]
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("distribution artifact names must be unique")
        project_kinds = [(item.project, item.kind) for item in distributions]
        if len(project_kinds) != len(set(project_kinds)):
            raise ValueError("distribution project/kind pairs must be unique")
        primary_kinds = {
            item.kind for item in distributions if item.project == project
        }
        if primary_kinds != {"wheel", "sdist"}:
            raise ValueError(
                "primary distribution project must bind exactly one wheel and one sdist"
            )
        object.__setattr__(self, "primary_distribution_project", project)
        object.__setattr__(self, "distributions", distributions)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_name": PAPER_EVIDENCE_SCHEMA,
            "schema_version": PAPER_EVIDENCE_SCHEMA_VERSION,
            "primary_distribution_project": self.primary_distribution_project,
            "provider_manifest": self.provider_manifest.as_dict(),
            "prob4d_stream_contract": self.prob4d_stream_contract.as_dict(),
            "observation_belief": self.observation_belief.as_dict(),
            "twin_belief": self.twin_belief.as_dict(),
            "distributions": [item.as_dict() for item in self.distributions],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PaperEvidenceBindingsV1:
        _require_exact_fields(
            value,
            expected=_PROFILE_FIELDS,
            name="paper-evidence profile",
        )
        if value.get("schema_name") != PAPER_EVIDENCE_SCHEMA:
            raise ValueError("unsupported paper-evidence schema")
        if int(value.get("schema_version", -1)) != PAPER_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported paper-evidence schema version")
        return cls(
            primary_distribution_project=str(
                value["primary_distribution_project"]
            ),
            provider_manifest=ArtifactBindingV1.from_mapping(
                _require_mapping(
                    value["provider_manifest"],
                    name="provider_manifest",
                )
            ),
            prob4d_stream_contract=Prob4DStreamBindingV1.from_mapping(
                _require_mapping(
                    value["prob4d_stream_contract"],
                    name="prob4d_stream_contract",
                )
            ),
            observation_belief=ArtifactBindingV1.from_mapping(
                _require_mapping(
                    value["observation_belief"],
                    name="observation_belief",
                )
            ),
            twin_belief=ArtifactBindingV1.from_mapping(
                _require_mapping(value["twin_belief"], name="twin_belief")
            ),
            distributions=tuple(
                DistributionBindingV1.from_mapping(
                    _require_mapping(item, name="distribution binding")
                )
                for item in _require_sequence(
                    value["distributions"],
                    name="distributions",
                )
            ),
        )


def load_paper_evidence_bindings(
    path: str | Path,
) -> PaperEvidenceBindingsV1:
    """Load one strict paper-evidence profile from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PaperEvidenceBindingsV1.from_mapping(
        _require_mapping(payload, name="paper-evidence profile")
    )


def embed_paper_evidence_bindings(
    information_boundary: Mapping[str, Any],
    bindings: PaperEvidenceBindingsV1,
) -> dict[str, Any]:
    """Return a copy with the reserved V1 paper-evidence profile embedded."""

    result = dict(information_boundary)
    if PAPER_EVIDENCE_PROFILE_KEY in result:
        raise ValueError(
            f"information boundary already contains {PAPER_EVIDENCE_PROFILE_KEY!r}"
        )
    result[PAPER_EVIDENCE_PROFILE_KEY] = bindings.as_dict()
    return result


def paper_evidence_bindings_from_manifest(
    manifest: RunManifestV2,
) -> PaperEvidenceBindingsV1:
    """Parse the embedded profile from a V2 manifest."""

    raw = manifest.information_boundary.get(PAPER_EVIDENCE_PROFILE_KEY)
    if raw is None:
        raise ValueError("run manifest has no paper-evidence profile")
    return PaperEvidenceBindingsV1.from_mapping(
        _require_mapping(raw, name="paper-evidence profile")
    )


def _artifact_index(
    manifest: RunManifestV2,
) -> dict[tuple[ArtifactRole, str], ArtifactDigest]:
    result: dict[tuple[ArtifactRole, str], ArtifactDigest] = {}
    for artifact in manifest.inputs:
        result[("input", artifact.name)] = artifact
    for artifact in manifest.outputs:
        result[("output", artifact.name)] = artifact
    return result


def _verify_binding(
    artifacts: Mapping[tuple[ArtifactRole, str], ArtifactDigest],
    binding: ArtifactBindingV1,
    *,
    name: str,
) -> None:
    artifact = artifacts.get((binding.role, binding.artifact_name))
    if artifact is None:
        raise ValueError(
            f"{name} references missing {binding.role} artifact "
            f"{binding.artifact_name!r}"
        )
    if artifact.sha256 != binding.artifact_id:
        raise ValueError(f"{name} artifact ID disagrees with run-manifest digest")


def validate_paper_evidence_manifest(
    manifest: RunManifestV2,
) -> PaperEvidenceBindingsV1:
    """Validate the complete paper-facing evidence profile and artifact bindings."""

    if manifest.dirty or any(state.dirty for state in manifest.related_repositories):
        raise ValueError("paper-facing evidence cannot bind a dirty repository state")
    if not manifest.claim_ids:
        raise ValueError("paper-facing evidence requires at least one claim ID")
    required_identifiers = {
        "method_freeze_id": manifest.method_freeze_id,
        "protocol_id": manifest.protocol_id,
        "split_id": manifest.split_id,
        "baseline_id": manifest.baseline_id,
    }
    missing = sorted(
        name for name, value in required_identifiers.items() if not str(value).strip()
    )
    if missing:
        raise ValueError(
            "paper-facing evidence is missing required identifiers: "
            + ", ".join(missing)
        )

    bindings = paper_evidence_bindings_from_manifest(manifest)
    artifacts = _artifact_index(manifest)
    _verify_binding(
        artifacts,
        bindings.provider_manifest,
        name="provider manifest",
    )
    _verify_binding(
        artifacts,
        bindings.observation_belief,
        name="observation belief",
    )
    _verify_binding(
        artifacts,
        bindings.twin_belief,
        name="TwinBelief",
    )

    for distribution in bindings.distributions:
        binding = ArtifactBindingV1(
            artifact_name=distribution.artifact_name,
            artifact_id=distribution.artifact_id,
            role="input",
        )
        _verify_binding(
            artifacts,
            binding,
            name=f"{distribution.project} {distribution.kind}",
        )
    return bindings


__all__ = [
    "ArtifactBindingV1",
    "DistributionBindingV1",
    "PAPER_EVIDENCE_PROFILE_KEY",
    "PAPER_EVIDENCE_SCHEMA",
    "PAPER_EVIDENCE_SCHEMA_VERSION",
    "PaperEvidenceBindingsV1",
    "Prob4DStreamBindingV1",
    "embed_paper_evidence_bindings",
    "load_paper_evidence_bindings",
    "paper_evidence_bindings_from_manifest",
    "validate_paper_evidence_manifest",
]
