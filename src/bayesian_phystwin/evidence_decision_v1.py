"""Compact, content-addressed decisions for claim-bearing evidence.

The artifact intentionally summarizes one already-evaluated scientific gate. It
binds the human-readable decision to the exact run manifest, evidence summary,
and repository revisions without duplicating the large numerical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Literal, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from .repository_provenance import RepositoryRole, RepositoryState
from .run_manifest import RunClassification, sha256_file
from .run_manifest_v2 import RunManifestV2

EVIDENCE_DECISION_SCHEMA = "bayesian_phystwin.evidence_decision"
EVIDENCE_DECISION_SCHEMA_VERSION = 1

DecisionStatus = Literal["pass", "fail", "degraded", "inconclusive"]
_VALID_DECISION_STATUSES = frozenset({"pass", "fail", "degraded", "inconclusive"})
_METRIC_FIELDS = frozenset(
    {
        "name",
        "comparison",
        "rule",
        "observed_value",
        "threshold_value",
        "unit",
    }
)
_REPOSITORY_FIELDS = frozenset({"repository", "revision", "dirty", "role"})
_DECISION_FIELDS = frozenset(
    {
        "decision_id",
        "schema_name",
        "schema_version",
        "created_utc",
        "claim_id",
        "protocol_id",
        "status",
        "run_classification",
        "claim_authorized",
        "evidence_level",
        "metric",
        "run_manifest_id",
        "evidence_fingerprint",
        "evidence_summary_sha256",
        "repositories",
        "limitations",
        "metadata",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
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
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical nonempty text")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_integer(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_boolean(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _require_finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_created_utc(value: Any) -> str:
    text = _require_text(value, name="created_utc")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_utc must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("created_utc must be an ISO-8601 UTC timestamp")
    return text


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


def _repository_from_mapping(value: Mapping[str, Any]) -> RepositoryState:
    _require_exact_fields(
        value,
        expected=_REPOSITORY_FIELDS,
        name="evidence-decision repository",
    )
    role = value["role"]
    if type(role) is not str or role not in {
        "primary",
        "upstream",
        "observation",
        "downstream",
        "paper",
        "environment",
        "dependency",
    }:
        raise ValueError("unsupported evidence-decision repository role")
    return RepositoryState(
        repository=_require_text(value["repository"], name="repository"),
        revision=_require_text(value["revision"], name="repository revision"),
        dirty=_require_boolean(value["dirty"], name="repository dirty"),
        role=cast(RepositoryRole, role),
    )


@dataclass(frozen=True)
class DecisionMetricV1:
    """One scalar metric and the frozen rule used for the recorded decision."""

    name: str
    comparison: str
    rule: str
    observed_value: float
    threshold_value: float | None
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, name="metric name"))
        object.__setattr__(
            self,
            "comparison",
            _require_text(self.comparison, name="metric comparison"),
        )
        object.__setattr__(self, "rule", _require_text(self.rule, name="metric rule"))
        object.__setattr__(
            self,
            "observed_value",
            _require_finite_number(self.observed_value, name="observed_value"),
        )
        if self.threshold_value is not None:
            object.__setattr__(
                self,
                "threshold_value",
                _require_finite_number(self.threshold_value, name="threshold_value"),
            )
        object.__setattr__(self, "unit", _require_text(self.unit, name="metric unit"))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "comparison": self.comparison,
            "rule": self.rule,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "unit": self.unit,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DecisionMetricV1:
        _require_exact_fields(value, expected=_METRIC_FIELDS, name="decision metric")
        threshold = value["threshold_value"]
        return cls(
            name=_require_text(value["name"], name="metric name"),
            comparison=_require_text(
                value["comparison"], name="metric comparison"
            ),
            rule=_require_text(value["rule"], name="metric rule"),
            observed_value=_require_finite_number(
                value["observed_value"], name="observed_value"
            ),
            threshold_value=(
                None
                if threshold is None
                else _require_finite_number(threshold, name="threshold_value")
            ),
            unit=_require_text(value["unit"], name="metric unit"),
        )


@dataclass(frozen=True)
class EvidenceDecisionV1:
    """Small, immutable decision bound to exact multi-repository evidence."""

    claim_id: str
    protocol_id: str
    status: DecisionStatus
    run_classification: RunClassification
    claim_authorized: bool
    evidence_level: int
    metric: DecisionMetricV1
    run_manifest_id: str
    evidence_fingerprint: str
    evidence_summary_sha256: str
    repositories: tuple[RepositoryState, ...]
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        status = self.status
        if type(status) is not str or status not in _VALID_DECISION_STATUSES:
            raise ValueError("unsupported evidence decision status")
        run_classification = self.run_classification
        if type(run_classification) is not str or run_classification not in {
            "controlled",
            "exploratory",
            "confirmatory",
            "diagnostic",
            "infrastructure",
        }:
            raise ValueError("unsupported run classification")
        authorized = _require_boolean(
            self.claim_authorized,
            name="claim_authorized",
        )
        if authorized and status != "pass":
            raise ValueError("only a passing decision can authorize a claim")
        if authorized and run_classification != "confirmatory":
            raise ValueError(
                "claim authorization requires a confirmatory run classification"
            )

        evidence_level = _require_integer(self.evidence_level, name="evidence_level")
        if evidence_level not in {1, 2, 3}:
            raise ValueError("evidence_level must be one of 1, 2, or 3")

        repositories = tuple(self.repositories)
        if any(not isinstance(state, RepositoryState) for state in repositories):
            raise ValueError("repositories must contain RepositoryState values")
        primary = [state for state in repositories if state.role == "primary"]
        if len(primary) != 1:
            raise ValueError(
                "evidence decision requires exactly one primary repository"
            )
        names = [state.repository for state in repositories]
        if len(names) != len(set(names)):
            raise ValueError("evidence-decision repository names must be unique")
        if authorized and any(state.dirty for state in repositories):
            raise ValueError("an authorized claim cannot bind a dirty repository")
        normalized_repositories = (
            primary[0],
            *sorted(
                (state for state in repositories if state.role != "primary"),
                key=lambda state: (state.role, state.repository),
            ),
        )

        if isinstance(self.limitations, (str, bytes)):
            raise ValueError("limitations must be a sequence of strings")
        limitations = tuple(
            _require_text(value, name="limitation") for value in self.limitations
        )
        if len(limitations) != len(set(limitations)):
            raise ValueError("limitations must be unique")
        if status in {"degraded", "inconclusive"} and not limitations:
            raise ValueError(
                f"{status} decisions must record at least one limitation"
            )

        object.__setattr__(
            self,
            "claim_id",
            _require_text(self.claim_id, name="claim_id"),
        )
        object.__setattr__(
            self,
            "protocol_id",
            _require_text(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "run_classification", run_classification)
        object.__setattr__(self, "claim_authorized", authorized)
        object.__setattr__(self, "evidence_level", evidence_level)
        if not isinstance(self.metric, DecisionMetricV1):
            raise ValueError("metric must be a DecisionMetricV1")
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
        object.__setattr__(
            self,
            "evidence_summary_sha256",
            _require_sha256(
                self.evidence_summary_sha256,
                name="evidence_summary_sha256",
            ),
        )
        object.__setattr__(self, "repositories", normalized_repositories)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )
        object.__setattr__(self, "created_utc", _require_created_utc(self.created_utc))

    def descriptor(self) -> dict[str, object]:
        """Return the canonical payload covered by :attr:`decision_id`."""

        return {
            "schema_name": EVIDENCE_DECISION_SCHEMA,
            "schema_version": EVIDENCE_DECISION_SCHEMA_VERSION,
            "created_utc": self.created_utc,
            "claim_id": self.claim_id,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "run_classification": self.run_classification,
            "claim_authorized": self.claim_authorized,
            "evidence_level": self.evidence_level,
            "metric": self.metric.as_dict(),
            "run_manifest_id": self.run_manifest_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "evidence_summary_sha256": self.evidence_summary_sha256,
            "repositories": [state.as_dict() for state in self.repositories],
            "limitations": list(self.limitations),
            "metadata": plain_json(self.metadata),
        }

    @property
    def decision_id(self) -> str:
        """Return the SHA-256 content identity of this decision."""

        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        """Return the complete portable JSON object."""

        return {"decision_id": self.decision_id, **self.descriptor()}


def build_evidence_decision(
    *,
    manifest: RunManifestV2,
    evidence_summary_path: str | Path,
    claim_id: str,
    status: DecisionStatus,
    claim_authorized: bool,
    evidence_level: int,
    metric: DecisionMetricV1,
    limitations: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
) -> EvidenceDecisionV1:
    """Build a decision whose identities are derived from a finalized manifest.

    The evidence summary must be one of the manifest's exact output artifacts.
    Claim authorization additionally requires a clean confirmatory manifest.
    """

    if not isinstance(manifest, RunManifestV2):
        raise TypeError("manifest must be a RunManifestV2")
    normalized_claim_id = _require_text(claim_id, name="claim_id")
    if normalized_claim_id not in manifest.claim_ids:
        raise ValueError("claim_id is not declared by the run manifest")
    protocol_id = _require_text(manifest.protocol_id, name="protocol_id")
    if claim_authorized and manifest.classification != "confirmatory":
        raise ValueError(
            "claim authorization requires a confirmatory run manifest"
        )

    summary = Path(evidence_summary_path)
    if summary.is_symlink() or not summary.is_file():
        raise ValueError("evidence summary must be a regular non-symlink file")
    before = summary.stat()
    summary_sha256 = sha256_file(summary)
    after = summary.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ValueError("evidence summary changed while it was being hashed")
    summary_size = after.st_size
    matching_outputs = [
        artifact
        for artifact in manifest.outputs
        if artifact.sha256 == summary_sha256 and artifact.size_bytes == summary_size
    ]
    if not matching_outputs:
        raise ValueError(
            "evidence summary is not an exact output artifact of the run manifest"
        )

    repositories = (
        RepositoryState(
            repository=manifest.repository,
            revision=manifest.revision,
            dirty=manifest.dirty,
            role="primary",
        ),
        *manifest.related_repositories,
    )
    return EvidenceDecisionV1(
        claim_id=normalized_claim_id,
        protocol_id=protocol_id,
        status=status,
        run_classification=manifest.classification,
        claim_authorized=claim_authorized,
        evidence_level=evidence_level,
        metric=metric,
        run_manifest_id=manifest.manifest_id,
        evidence_fingerprint=manifest.evidence_fingerprint,
        evidence_summary_sha256=summary_sha256,
        repositories=repositories,
        limitations=tuple(limitations),
        metadata={} if metadata is None else metadata,
        created_utc=(
            datetime.now(timezone.utc).isoformat()
            if created_utc is None
            else created_utc
        ),
    )


def write_evidence_decision(
    path: str | Path,
    decision: EvidenceDecisionV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish stable JSON, refusing replacement unless requested."""

    if not isinstance(decision, EvidenceDecisionV1):
        raise TypeError("decision must be an EvidenceDecisionV1")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(decision.as_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
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
                raise FileExistsError(
                    f"evidence decision already exists: {destination}"
                ) from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def load_evidence_decision(path: str | Path) -> EvidenceDecisionV1:
    """Load a strict decision and reject schema or content-address drift."""

    payload = _load_json_mapping(path, name="evidence decision")
    _require_exact_fields(payload, expected=_DECISION_FIELDS, name="evidence decision")
    if payload["schema_name"] != EVIDENCE_DECISION_SCHEMA:
        raise ValueError("unsupported evidence-decision schema")
    if (
        _require_integer(payload["schema_version"], name="schema_version")
        != EVIDENCE_DECISION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported evidence-decision schema version")

    expected_id = _require_sha256(payload["decision_id"], name="decision_id")
    status = payload["status"]
    if type(status) is not str or status not in _VALID_DECISION_STATUSES:
        raise ValueError("unsupported evidence decision status")
    decision = EvidenceDecisionV1(
        claim_id=_require_text(payload["claim_id"], name="claim_id"),
        protocol_id=_require_text(payload["protocol_id"], name="protocol_id"),
        status=cast(DecisionStatus, status),
        run_classification=cast(
            RunClassification,
            _require_text(
                payload["run_classification"],
                name="run_classification",
            ),
        ),
        claim_authorized=_require_boolean(
            payload["claim_authorized"], name="claim_authorized"
        ),
        evidence_level=_require_integer(
            payload["evidence_level"], name="evidence_level"
        ),
        metric=DecisionMetricV1.from_mapping(
            _require_mapping(payload["metric"], name="decision metric")
        ),
        run_manifest_id=_require_sha256(
            payload["run_manifest_id"], name="run_manifest_id"
        ),
        evidence_fingerprint=_require_sha256(
            payload["evidence_fingerprint"], name="evidence_fingerprint"
        ),
        evidence_summary_sha256=_require_sha256(
            payload["evidence_summary_sha256"], name="evidence_summary_sha256"
        ),
        repositories=tuple(
            _repository_from_mapping(
                _require_mapping(value, name="evidence-decision repository")
            )
            for value in _require_sequence(payload["repositories"], name="repositories")
        ),
        limitations=tuple(
            _require_text(value, name="limitation")
            for value in _require_sequence(payload["limitations"], name="limitations")
        ),
        metadata=_require_mapping(payload["metadata"], name="metadata"),
        created_utc=_require_created_utc(payload["created_utc"]),
    )
    if decision.decision_id != expected_id:
        raise ValueError("evidence decision digest does not match its descriptor")
    return decision


__all__ = [
    "DecisionMetricV1",
    "DecisionStatus",
    "EVIDENCE_DECISION_SCHEMA",
    "EVIDENCE_DECISION_SCHEMA_VERSION",
    "EvidenceDecisionV1",
    "build_evidence_decision",
    "load_evidence_decision",
    "write_evidence_decision",
]
