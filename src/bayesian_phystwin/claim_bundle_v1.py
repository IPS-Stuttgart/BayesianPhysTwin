"""Portable, content-addressed bundles for claim-bearing evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from .decisive_evidence import (
    DECISIVE_EVIDENCE_INPUT_CONTRACT,
    DECISIVE_EVIDENCE_SUMMARY_CONTRACT,
    MATCHED_COUNT_RISK_COVERAGE_CONTRACT,
    THRESHOLD_RISK_COVERAGE_CONTRACT,
)
from .paper_evidence_v1 import validate_paper_evidence_manifest
from .repository_provenance import RepositoryRole, RepositoryState
from .run_manifest import sha256_file
from .run_manifest_v2 import RunManifestV2, load_run_manifest

CLAIM_BUNDLE_SCHEMA = "bayesian_phystwin.claim_bundle"
CLAIM_BUNDLE_SCHEMA_VERSION = 1

ClaimBundleArtifactKind = Literal[
    "run_manifest",
    "evidence_summary",
    "claim_binding",
    "figure",
    "table_data",
    "supporting",
]

_VALID_ARTIFACT_KINDS = frozenset(
    {
        "run_manifest",
        "evidence_summary",
        "claim_binding",
        "figure",
        "table_data",
        "supporting",
    }
)
_CLAIM_BEARING_CLASSIFICATIONS = frozenset({"controlled", "confirmatory"})
_ARTIFACT_FIELDS = frozenset(
    {"name", "kind", "path", "sha256", "size_bytes", "media_type"}
)
_REPOSITORY_FIELDS = frozenset({"repository", "revision", "dirty", "role"})
_BUNDLE_FIELDS = frozenset(
    {
        "bundle_id",
        "schema_name",
        "schema_version",
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
        "artifacts",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "contract",
        "source_contract",
        "protocol_id",
        "statistical_unit",
        "claim_boundary",
        "reference_method",
        "analysis_configuration",
        "metrics",
    }
)
_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".npz": "application/octet-stream",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(map(str, value))
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
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_relative_path(value: Any, *, name: str) -> str:
    text = _require_text(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must use portable forward slashes")
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) == "." or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    return path.as_posix()


def _require_artifact_kind(value: Any) -> ClaimBundleArtifactKind:
    if value not in _VALID_ARTIFACT_KINDS:
        raise ValueError("unsupported claim-bundle artifact kind")
    return cast(ClaimBundleArtifactKind, value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_mapping(path: str | Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    return _require_mapping(value, name=name)


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


@dataclass(frozen=True)
class ClaimBundleArtifactV1:
    """Content identity and semantic role of one portable bundle artifact."""

    name: str
    kind: ClaimBundleArtifactKind
    path: str
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, name="artifact name"))
        object.__setattr__(self, "kind", _require_artifact_kind(self.kind))
        object.__setattr__(
            self,
            "path",
            _require_relative_path(self.path, name="artifact path"),
        )
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, name="artifact sha256"),
        )
        size = _require_integer(self.size_bytes, name="artifact size_bytes")
        if size < 0:
            raise ValueError("artifact size_bytes must be nonnegative")
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(
            self,
            "media_type",
            _require_text(self.media_type, name="artifact media_type"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClaimBundleArtifactV1:
        _require_exact_fields(
            value,
            expected=_ARTIFACT_FIELDS,
            name="claim-bundle artifact",
        )
        return cls(
            name=_require_text(value["name"], name="artifact name"),
            kind=_require_artifact_kind(value["kind"]),
            path=_require_relative_path(value["path"], name="artifact path"),
            sha256=_require_sha256(value["sha256"], name="artifact sha256"),
            size_bytes=_require_integer(value["size_bytes"], name="artifact size_bytes"),
            media_type=_require_text(value["media_type"], name="artifact media_type"),
        )


@dataclass(frozen=True)
class ClaimBundleV1:
    """One deterministic, reviewable bundle of claim-bearing evidence."""

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
    repositories: tuple[RepositoryState, ...]
    artifacts: tuple[ClaimBundleArtifactV1, ...]

    def __post_init__(self) -> None:
        classification = _require_text(self.classification, name="classification")
        if classification not in _CLAIM_BEARING_CLASSIFICATIONS:
            raise ValueError(
                "claim bundle requires a controlled or confirmatory run classification"
            )
        claims = tuple(
            _require_text(value, name="claim ID") for value in self.claim_ids
        )
        if not claims or len(claims) != len(set(claims)):
            raise ValueError("claim_ids must be unique and nonempty")

        repositories = tuple(self.repositories)
        primary = [state for state in repositories if state.role == "primary"]
        if len(primary) != 1:
            raise ValueError("claim bundle requires exactly one primary repository")
        if any(state.dirty for state in repositories):
            raise ValueError("claim bundle cannot contain a dirty repository state")
        names = [state.repository for state in repositories]
        if len(names) != len(set(names)):
            raise ValueError("claim-bundle repository names must be unique")
        normalized_repositories = (
            primary[0],
            *sorted(
                (state for state in repositories if state.role != "primary"),
                key=lambda state: (state.role, state.repository),
            ),
        )

        artifacts = tuple(
            sorted(self.artifacts, key=lambda artifact: (artifact.kind, artifact.name))
        )
        if not artifacts:
            raise ValueError("claim bundle must contain artifacts")
        artifact_names = [artifact.name for artifact in artifacts]
        artifact_paths = [artifact.path for artifact in artifacts]
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("claim-bundle artifact names must be unique")
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("claim-bundle artifact paths must be unique")
        kind_counts = {
            kind: sum(artifact.kind == kind for artifact in artifacts)
            for kind in _VALID_ARTIFACT_KINDS
        }
        if kind_counts["run_manifest"] != 1:
            raise ValueError("claim bundle requires exactly one run manifest")
        if kind_counts["evidence_summary"] != 1:
            raise ValueError("claim bundle requires exactly one evidence summary")
        if kind_counts["claim_binding"] > 1:
            raise ValueError("claim bundle permits at most one claim binding")

        object.__setattr__(
            self,
            "run_manifest_id",
            _require_sha256(self.run_manifest_id, name="run_manifest_id"),
        )
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _require_sha256(
                self.evidence_fingerprint,
                name="evidence_fingerprint",
            ),
        )
        object.__setattr__(self, "run_id", _require_text(self.run_id, name="run_id"))
        object.__setattr__(self, "classification", classification)
        object.__setattr__(
            self,
            "protocol_id",
            _require_text(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(
            self,
            "statistical_unit",
            _require_text(self.statistical_unit, name="statistical_unit"),
        )
        object.__setattr__(
            self,
            "claim_boundary",
            _require_text(self.claim_boundary, name="claim_boundary"),
        )
        object.__setattr__(self, "claim_ids", claims)
        object.__setattr__(
            self,
            "method_freeze_id",
            _require_text(self.method_freeze_id, name="method_freeze_id"),
        )
        object.__setattr__(
            self,
            "split_id",
            _require_text(self.split_id, name="split_id"),
        )
        object.__setattr__(
            self,
            "baseline_id",
            _require_text(self.baseline_id, name="baseline_id"),
        )
        object.__setattr__(self, "repositories", normalized_repositories)
        object.__setattr__(self, "artifacts", artifacts)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": CLAIM_BUNDLE_SCHEMA,
            "schema_version": CLAIM_BUNDLE_SCHEMA_VERSION,
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
            "repositories": [state.as_dict() for state in self.repositories],
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }

    @property
    def bundle_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {"bundle_id": self.bundle_id, **self.descriptor()}


def validate_decisive_evidence_summary(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the stable identity and risk-coverage contracts of one summary."""

    _require_exact_fields(
        payload,
        expected=_SUMMARY_FIELDS,
        name="decisive-evidence summary",
    )
    if _require_integer(payload["schema_version"], name="summary schema_version") != 1:
        raise ValueError("unsupported decisive-evidence summary schema version")
    if payload["contract"] != DECISIVE_EVIDENCE_SUMMARY_CONTRACT:
        raise ValueError("unsupported decisive-evidence summary contract")
    if payload["source_contract"] != DECISIVE_EVIDENCE_INPUT_CONTRACT:
        raise ValueError("unsupported decisive-evidence source contract")
    _require_text(payload["protocol_id"], name="summary protocol_id")
    _require_text(payload["statistical_unit"], name="summary statistical_unit")
    _require_text(payload["claim_boundary"], name="summary claim_boundary")
    reference = payload["reference_method"]
    if reference is not None:
        _require_text(reference, name="summary reference_method")

    configuration = _require_mapping(
        payload["analysis_configuration"],
        name="summary analysis_configuration",
    )
    if configuration.get("matched_fallback") is not True:
        raise ValueError("decisive-evidence summary must verify matched fallback")
    if (
        configuration.get("primary_risk_coverage_contract")
        != THRESHOLD_RISK_COVERAGE_CONTRACT
    ):
        raise ValueError("decisive-evidence summary has the wrong primary risk contract")
    if (
        configuration.get("secondary_risk_coverage_contract")
        != MATCHED_COUNT_RISK_COVERAGE_CONTRACT
    ):
        raise ValueError(
            "decisive-evidence summary has the wrong secondary risk contract"
        )
    if (
        configuration.get(
            "confirmatory_thresholds_must_be_source_or_calibration_frozen"
        )
        is not True
    ):
        raise ValueError("decisive-evidence summary does not freeze thresholds")

    metrics = _require_mapping(payload["metrics"], name="summary metrics")
    if not metrics:
        raise ValueError("decisive-evidence summary metrics must not be empty")
    for metric_name, raw_metric in metrics.items():
        _require_text(metric_name, name="summary metric name")
        metric = _require_mapping(raw_metric, name=f"summary metric {metric_name!r}")
        threshold = _require_mapping(
            metric.get("threshold_risk_coverage"),
            name=f"summary metric {metric_name!r} threshold_risk_coverage",
        )
        if threshold.get("contract") != THRESHOLD_RISK_COVERAGE_CONTRACT:
            raise ValueError(
                f"summary metric {metric_name!r} has the wrong threshold contract"
            )
        matched = _require_mapping(
            metric.get("matched_count_risk_coverage"),
            name=f"summary metric {metric_name!r} matched_count_risk_coverage",
        )
        if matched.get("contract") != MATCHED_COUNT_RISK_COVERAGE_CONTRACT:
            raise ValueError(
                f"summary metric {metric_name!r} has the wrong matched-count contract"
            )
    return payload


def claim_bundle_artifact(
    path: str | Path,
    *,
    name: str,
    kind: ClaimBundleArtifactKind,
    root: str | Path,
    media_type: str | None = None,
) -> ClaimBundleArtifactV1:
    """Hash one file and bind it to a portable path below ``root``."""

    artifact_root = Path(root).resolve()
    source = Path(path)
    resolved = source.resolve() if source.is_absolute() else (artifact_root / source).resolve()
    try:
        relative = resolved.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("claim-bundle artifacts must remain below artifact root") from error
    if not resolved.is_file():
        raise ValueError(f"claim-bundle artifact is not a file: {resolved}")
    return ClaimBundleArtifactV1(
        name=name,
        kind=kind,
        path=relative.as_posix(),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
        media_type=media_type or _media_type(resolved),
    )


def _manifest_repositories(manifest: RunManifestV2) -> tuple[RepositoryState, ...]:
    return (
        RepositoryState(
            repository=manifest.repository,
            revision=manifest.revision,
            dirty=manifest.dirty,
            role="primary",
        ),
        *manifest.related_repositories,
    )


def _require_v2_manifest(path: str | Path) -> RunManifestV2:
    manifest = load_run_manifest(path)
    if not isinstance(manifest, RunManifestV2):
        raise ValueError("claim bundle requires RunManifestV2")
    validate_paper_evidence_manifest(manifest)
    return manifest


def _check_manifest_summary_identity(
    manifest: RunManifestV2,
    summary: Mapping[str, Any],
) -> None:
    if summary["protocol_id"] != manifest.protocol_id:
        raise ValueError("evidence summary protocol_id differs from run manifest")
    if summary["statistical_unit"] != manifest.statistical_unit:
        raise ValueError("evidence summary statistical_unit differs from run manifest")


def build_claim_bundle(
    *,
    run_manifest_path: str | Path,
    evidence_summary_path: str | Path,
    artifact_root: str | Path,
    claim_binding_path: str | Path | None = None,
    additional_artifacts: Sequence[ClaimBundleArtifactV1] = (),
) -> ClaimBundleV1:
    """Build one bundle after validating all claim-bearing semantic bindings."""

    root = Path(artifact_root).resolve()
    manifest = _require_v2_manifest(
        Path(run_manifest_path)
        if Path(run_manifest_path).is_absolute()
        else root / run_manifest_path
    )
    summary_path = (
        Path(evidence_summary_path)
        if Path(evidence_summary_path).is_absolute()
        else root / evidence_summary_path
    )
    summary = validate_decisive_evidence_summary(
        _load_json_mapping(summary_path, name="decisive-evidence summary")
    )
    _check_manifest_summary_identity(manifest, summary)

    extras = tuple(additional_artifacts)
    if any(
        artifact.kind in {"run_manifest", "evidence_summary", "claim_binding"}
        for artifact in extras
    ):
        raise ValueError("additional artifacts must use a supporting artifact kind")

    artifacts: list[ClaimBundleArtifactV1] = [
        claim_bundle_artifact(
            run_manifest_path,
            name="run_manifest",
            kind="run_manifest",
            root=root,
            media_type="application/json",
        ),
        claim_bundle_artifact(
            evidence_summary_path,
            name="decisive_evidence_summary",
            kind="evidence_summary",
            root=root,
            media_type="application/json",
        ),
        *extras,
    ]
    if claim_binding_path is not None:
        binding_path = (
            Path(claim_binding_path)
            if Path(claim_binding_path).is_absolute()
            else root / claim_binding_path
        )
        _load_json_mapping(binding_path, name="claim binding")
        artifacts.append(
            claim_bundle_artifact(
                claim_binding_path,
                name="paper_claim_binding",
                kind="claim_binding",
                root=root,
                media_type="application/json",
            )
        )

    return ClaimBundleV1(
        run_manifest_id=manifest.manifest_id,
        evidence_fingerprint=manifest.evidence_fingerprint,
        run_id=manifest.run_id,
        classification=manifest.classification,
        protocol_id=manifest.protocol_id,
        statistical_unit=manifest.statistical_unit,
        claim_boundary=cast(str, summary["claim_boundary"]),
        claim_ids=manifest.claim_ids,
        method_freeze_id=manifest.method_freeze_id,
        split_id=manifest.split_id,
        baseline_id=manifest.baseline_id,
        repositories=_manifest_repositories(manifest),
        artifacts=tuple(artifacts),
    )


def write_claim_bundle(path: str | Path, bundle: ClaimBundleV1) -> None:
    """Write a stable, human-readable claim-bundle JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(bundle.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _repository_from_mapping(value: Mapping[str, Any]) -> RepositoryState:
    _require_exact_fields(
        value,
        expected=_REPOSITORY_FIELDS,
        name="claim-bundle repository",
    )
    role = value["role"]
    if role not in {
        "primary",
        "upstream",
        "observation",
        "downstream",
        "paper",
        "environment",
        "dependency",
    }:
        raise ValueError("unsupported claim-bundle repository role")
    dirty = value["dirty"]
    if not isinstance(dirty, bool):
        raise ValueError("claim-bundle repository dirty field must be boolean")
    return RepositoryState(
        repository=_require_text(value["repository"], name="repository"),
        revision=_require_text(value["revision"], name="repository revision"),
        dirty=dirty,
        role=cast(RepositoryRole, role),
    )


def load_claim_bundle(path: str | Path) -> ClaimBundleV1:
    """Load a strict bundle and reject schema or content-address drift."""

    payload = _load_json_mapping(path, name="claim bundle")
    _require_exact_fields(payload, expected=_BUNDLE_FIELDS, name="claim bundle")
    if payload["schema_name"] != CLAIM_BUNDLE_SCHEMA:
        raise ValueError("unsupported claim-bundle schema")
    if (
        _require_integer(payload["schema_version"], name="claim-bundle schema_version")
        != CLAIM_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported claim-bundle schema version")
    expected_id = _require_sha256(payload["bundle_id"], name="bundle_id")
    bundle = ClaimBundleV1(
        run_manifest_id=_require_sha256(
            payload["run_manifest_id"],
            name="run_manifest_id",
        ),
        evidence_fingerprint=_require_sha256(
            payload["evidence_fingerprint"],
            name="evidence_fingerprint",
        ),
        run_id=_require_text(payload["run_id"], name="run_id"),
        classification=_require_text(payload["classification"], name="classification"),
        protocol_id=_require_text(payload["protocol_id"], name="protocol_id"),
        statistical_unit=_require_text(
            payload["statistical_unit"],
            name="statistical_unit",
        ),
        claim_boundary=_require_text(
            payload["claim_boundary"],
            name="claim_boundary",
        ),
        claim_ids=tuple(
            _require_text(value, name="claim ID")
            for value in _require_sequence(payload["claim_ids"], name="claim_ids")
        ),
        method_freeze_id=_require_text(
            payload["method_freeze_id"],
            name="method_freeze_id",
        ),
        split_id=_require_text(payload["split_id"], name="split_id"),
        baseline_id=_require_text(payload["baseline_id"], name="baseline_id"),
        repositories=tuple(
            _repository_from_mapping(
                _require_mapping(value, name="claim-bundle repository")
            )
            for value in _require_sequence(
                payload["repositories"],
                name="repositories",
            )
        ),
        artifacts=tuple(
            ClaimBundleArtifactV1.from_mapping(
                _require_mapping(value, name="claim-bundle artifact")
            )
            for value in _require_sequence(payload["artifacts"], name="artifacts")
        ),
    )
    if bundle.bundle_id != expected_id:
        raise ValueError("claim-bundle digest does not match its payload")
    return bundle


def _artifact_for_kind(
    bundle: ClaimBundleV1,
    kind: ClaimBundleArtifactKind,
) -> ClaimBundleArtifactV1 | None:
    return next((artifact for artifact in bundle.artifacts if artifact.kind == kind), None)


def verify_claim_bundle_artifacts(
    bundle: ClaimBundleV1,
    *,
    root: str | Path,
) -> RunManifestV2:
    """Verify bytes and re-check manifest/summary semantics against the bundle."""

    artifact_root = Path(root).resolve()
    for artifact in bundle.artifacts:
        path = (artifact_root / artifact.path).resolve()
        try:
            path.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError("claim-bundle artifact escapes artifact root") from error
        if not path.is_file():
            raise ValueError(f"claim-bundle artifact is missing: {artifact.path}")
        if path.stat().st_size != artifact.size_bytes:
            raise ValueError(f"claim-bundle artifact size differs: {artifact.name}")
        if sha256_file(path) != artifact.sha256:
            raise ValueError(f"claim-bundle artifact digest differs: {artifact.name}")

    manifest_artifact = _artifact_for_kind(bundle, "run_manifest")
    summary_artifact = _artifact_for_kind(bundle, "evidence_summary")
    if manifest_artifact is None or summary_artifact is None:
        raise AssertionError("validated bundle lacks required artifacts")
    manifest = _require_v2_manifest(artifact_root / manifest_artifact.path)
    summary = validate_decisive_evidence_summary(
        _load_json_mapping(
            artifact_root / summary_artifact.path,
            name="decisive-evidence summary",
        )
    )
    _check_manifest_summary_identity(manifest, summary)

    expected_repositories = ClaimBundleV1(
        run_manifest_id=manifest.manifest_id,
        evidence_fingerprint=manifest.evidence_fingerprint,
        run_id=manifest.run_id,
        classification=manifest.classification,
        protocol_id=manifest.protocol_id,
        statistical_unit=manifest.statistical_unit,
        claim_boundary=cast(str, summary["claim_boundary"]),
        claim_ids=manifest.claim_ids,
        method_freeze_id=manifest.method_freeze_id,
        split_id=manifest.split_id,
        baseline_id=manifest.baseline_id,
        repositories=_manifest_repositories(manifest),
        artifacts=bundle.artifacts,
    )
    semantic_fields = (
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
    )
    for field_name in semantic_fields:
        if getattr(bundle, field_name) != getattr(expected_repositories, field_name):
            raise ValueError(f"claim-bundle {field_name} differs from bound evidence")

    binding_artifact = _artifact_for_kind(bundle, "claim_binding")
    if binding_artifact is not None:
        _load_json_mapping(
            artifact_root / binding_artifact.path,
            name="claim binding",
        )
    return manifest


__all__ = [
    "CLAIM_BUNDLE_SCHEMA",
    "CLAIM_BUNDLE_SCHEMA_VERSION",
    "ClaimBundleArtifactKind",
    "ClaimBundleArtifactV1",
    "ClaimBundleV1",
    "build_claim_bundle",
    "claim_bundle_artifact",
    "load_claim_bundle",
    "validate_decisive_evidence_summary",
    "verify_claim_bundle_artifacts",
    "write_claim_bundle",
]
