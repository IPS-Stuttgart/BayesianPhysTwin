"""Orthogonal execution, scientific, and authorization evidence decisions.

Version 2 keeps the content-addressed provenance of :mod:`evidence_decision_v1`
while separating three concepts that must not be inferred from one overloaded
status value:

* whether the registered execution completed validly;
* whether the registered scientific rule passed or was negative; and
* whether the bound evidence authorizes advancement of the claim.

Version-1 artifacts remain immutable and loadable through their original API.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from .evidence_decision_v1 import (
    DecisionMetricV1,
    _canonical_json,
    _load_json_mapping,
    _repository_from_mapping,
    _require_created_utc,
    _require_exact_fields,
    _require_integer,
    _require_mapping,
    _require_sequence,
    _require_sha256,
    _require_text,
)
from .repository_provenance import RepositoryState
from .run_manifest import RunClassification, sha256_file
from .run_manifest_v2 import RunManifestV2

EVIDENCE_DECISION_SCHEMA = "bayesian_phystwin.evidence_decision"
EVIDENCE_DECISION_SCHEMA_VERSION = 2

ExecutionStatus = Literal[
    "completed",
    "infrastructure_failure",
    "protocol_invalid",
]
ScientificDecision = Literal["pass", "negative", "not_evaluated"]
AuthorizationDecision = Literal["advance", "stop"]
DecisionMetricV2: TypeAlias = DecisionMetricV1

_VALID_EXECUTION_STATUSES = frozenset(
    {"completed", "infrastructure_failure", "protocol_invalid"}
)
_VALID_SCIENTIFIC_DECISIONS = frozenset({"pass", "negative", "not_evaluated"})
_VALID_AUTHORIZATION_DECISIONS = frozenset({"advance", "stop"})
_VALID_RUN_CLASSIFICATIONS = frozenset(
    {"controlled", "exploratory", "confirmatory", "diagnostic", "infrastructure"}
)
_DECISION_FIELDS = frozenset(
    {
        "decision_id",
        "schema_name",
        "schema_version",
        "created_utc",
        "claim_id",
        "protocol_id",
        "execution_status",
        "scientific_decision",
        "authorization",
        "run_classification",
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


def _require_execution_status(value: Any) -> ExecutionStatus:
    if type(value) is not str or value not in _VALID_EXECUTION_STATUSES:
        raise ValueError("unsupported evidence execution status")
    return cast(ExecutionStatus, value)


def _require_scientific_decision(value: Any) -> ScientificDecision:
    if type(value) is not str or value not in _VALID_SCIENTIFIC_DECISIONS:
        raise ValueError("unsupported scientific decision")
    return cast(ScientificDecision, value)


def _require_authorization(value: Any) -> AuthorizationDecision:
    if type(value) is not str or value not in _VALID_AUTHORIZATION_DECISIONS:
        raise ValueError("unsupported authorization decision")
    return cast(AuthorizationDecision, value)


def _require_run_classification(value: Any) -> RunClassification:
    if type(value) is not str or value not in _VALID_RUN_CLASSIFICATIONS:
        raise ValueError("unsupported run classification")
    return cast(RunClassification, value)


@dataclass(frozen=True)
class EvidenceDecisionV2:
    """Content-addressed evidence decision with orthogonal outcome axes."""

    claim_id: str
    protocol_id: str
    execution_status: ExecutionStatus
    scientific_decision: ScientificDecision
    authorization: AuthorizationDecision
    run_classification: RunClassification
    evidence_level: int
    metric: DecisionMetricV2 | None
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
        execution_status = _require_execution_status(self.execution_status)
        scientific_decision = _require_scientific_decision(self.scientific_decision)
        authorization = _require_authorization(self.authorization)
        run_classification = _require_run_classification(self.run_classification)

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

        metric = self.metric
        if scientific_decision in {"pass", "negative"}:
            if not isinstance(metric, DecisionMetricV1):
                raise ValueError(
                    "evaluated scientific decisions require a DecisionMetricV2"
                )
        elif metric is not None:
            raise ValueError("not-evaluated decisions must not record a metric")

        if execution_status != "completed":
            if scientific_decision != "not_evaluated":
                raise ValueError(
                    "non-completed executions cannot record a scientific result"
                )
            if authorization != "stop":
                raise ValueError("non-completed executions must stop authorization")
            if not limitations:
                raise ValueError(
                    "non-completed executions must record at least one limitation"
                )

        if scientific_decision in {"negative", "not_evaluated"}:
            if authorization != "stop":
                raise ValueError(
                    "negative or not-evaluated evidence must stop authorization"
                )

        if scientific_decision == "not_evaluated" and not limitations:
            raise ValueError(
                "not-evaluated decisions must record at least one limitation"
            )

        if scientific_decision == "pass" and authorization == "stop":
            if not limitations:
                raise ValueError(
                    "a passing result stopped by policy must record a limitation"
                )

        if authorization == "advance":
            if execution_status != "completed" or scientific_decision != "pass":
                raise ValueError(
                    "advancement requires a completed passing scientific decision"
                )
            if run_classification != "confirmatory":
                raise ValueError(
                    "advancement requires a confirmatory run classification"
                )
            if any(state.dirty for state in normalized_repositories):
                raise ValueError("advancement cannot bind a dirty repository")

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
        object.__setattr__(self, "execution_status", execution_status)
        object.__setattr__(self, "scientific_decision", scientific_decision)
        object.__setattr__(self, "authorization", authorization)
        object.__setattr__(self, "run_classification", run_classification)
        object.__setattr__(self, "evidence_level", evidence_level)
        object.__setattr__(self, "metric", metric)
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
            "execution_status": self.execution_status,
            "scientific_decision": self.scientific_decision,
            "authorization": self.authorization,
            "run_classification": self.run_classification,
            "evidence_level": self.evidence_level,
            "metric": None if self.metric is None else self.metric.as_dict(),
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


def build_evidence_decision_v2(
    *,
    manifest: RunManifestV2,
    evidence_summary_path: str | Path,
    claim_id: str,
    execution_status: ExecutionStatus,
    scientific_decision: ScientificDecision,
    authorization: AuthorizationDecision,
    evidence_level: int,
    metric: DecisionMetricV2 | None,
    limitations: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
) -> EvidenceDecisionV2:
    """Build a v2 decision bound to a finalized manifest and output artifact."""

    if not isinstance(manifest, RunManifestV2):
        raise TypeError("manifest must be a RunManifestV2")
    normalized_claim_id = _require_text(claim_id, name="claim_id")
    if normalized_claim_id not in manifest.claim_ids:
        raise ValueError("claim_id is not declared by the run manifest")
    protocol_id = _require_text(manifest.protocol_id, name="protocol_id")

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
    matching_outputs = [
        artifact
        for artifact in manifest.outputs
        if artifact.sha256 == summary_sha256 and artifact.size_bytes == after.st_size
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
    return EvidenceDecisionV2(
        claim_id=normalized_claim_id,
        protocol_id=protocol_id,
        execution_status=execution_status,
        scientific_decision=scientific_decision,
        authorization=authorization,
        run_classification=manifest.classification,
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


def write_evidence_decision_v2(
    path: str | Path,
    decision: EvidenceDecisionV2,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish stable v2 JSON, refusing replacement by default."""

    if not isinstance(decision, EvidenceDecisionV2):
        raise TypeError("decision must be an EvidenceDecisionV2")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(decision.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def load_evidence_decision_v2(path: str | Path) -> EvidenceDecisionV2:
    """Load strict v2 JSON and reject schema or content-address drift."""

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
    metric_payload = payload["metric"]
    metric = (
        None
        if metric_payload is None
        else DecisionMetricV1.from_mapping(
            _require_mapping(metric_payload, name="decision metric")
        )
    )
    decision = EvidenceDecisionV2(
        claim_id=_require_text(payload["claim_id"], name="claim_id"),
        protocol_id=_require_text(payload["protocol_id"], name="protocol_id"),
        execution_status=_require_execution_status(payload["execution_status"]),
        scientific_decision=_require_scientific_decision(
            payload["scientific_decision"]
        ),
        authorization=_require_authorization(payload["authorization"]),
        run_classification=_require_run_classification(
            payload["run_classification"]
        ),
        evidence_level=_require_integer(
            payload["evidence_level"], name="evidence_level"
        ),
        metric=metric,
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
    "AuthorizationDecision",
    "DecisionMetricV2",
    "EVIDENCE_DECISION_SCHEMA",
    "EVIDENCE_DECISION_SCHEMA_VERSION",
    "EvidenceDecisionV2",
    "ExecutionStatus",
    "ScientificDecision",
    "build_evidence_decision_v2",
    "load_evidence_decision_v2",
    "write_evidence_decision_v2",
]
