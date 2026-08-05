"""Content-addressed accounting for raw evidence consumed across inference stages.

The BayesianPhysTwin state update is one stage in a larger Prob4D ->
BayesianPhysTwin -> Causal4D pipeline. A raw tactile, robot-state, visual, or
force factor must not be consumed in the state posterior and then multiplied
again downstream as if it were independent. This module records each use without
importing either producer or consumer packages and rejects incompatible reuse
before posterior weights are changed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)

EVIDENCE_USE_SCHEMA = "bayesian-phystwin.evidence-use"
EVIDENCE_USE_LEDGER_SCHEMA = "bayesian-phystwin.evidence-use-ledger"
EVIDENCE_USE_VERSION = 1
EVIDENCE_USE_LEDGER_VERSION = 1
EVIDENCE_USE_SEMANTICS = "one-raw-factor-one-independent-inference-path-v1"
EVIDENCE_USE_LEDGER_METADATA_KEY = "evidence_use_ledger_v1"
EVIDENCE_USE_CLAIM_BOUNDARY = (
    "Evidence accounting only. A valid ledger does not establish observation "
    "accuracy, calibration, physical-state benefit, intervention benefit, or "
    "deployment safety."
)

EvidenceInferenceRole = Literal[
    "state_update",
    "actuator_abduction",
    "contact_abduction",
    "joint_state_intervention_update",
    "calibration_only",
    "evaluation_only",
]
EVIDENCE_INFERENCE_ROLES: tuple[EvidenceInferenceRole, ...] = (
    "state_update",
    "actuator_abduction",
    "contact_abduction",
    "joint_state_intervention_update",
    "calibration_only",
    "evaluation_only",
)

_USE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "entry_id",
        "evidence_artifact_id",
        "raw_factor_id",
        "raw_factor_sha256",
        "source_repository",
        "source_revision",
        "source_artifacts",
        "sensor_family",
        "stream_id",
        "clock_id",
        "causal_frame_start",
        "causal_frame_stop",
        "correlation_group_ids",
        "inference_role",
        "metadata",
    }
)
_LEDGER_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "ledger_id",
        "protocol_id",
        "case_id",
        "causal_frame_stop",
        "entries",
        "metadata",
        "claim_boundary",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _role(value: object) -> EvidenceInferenceRole:
    if type(value) is not str or value not in EVIDENCE_INFERENCE_ROLES:
        raise ValueError("inference_role is unsupported")
    return cast(EvidenceInferenceRole, value)


def _stage(role: EvidenceInferenceRole) -> str | None:
    if role == "state_update":
        return "state"
    if role in {"actuator_abduction", "contact_abduction"}:
        return "intervention"
    if role == "joint_state_intervention_update":
        return "joint"
    return None


@dataclass(frozen=True)
class EvidenceUseV1:
    """One raw factor consumed in one declared inference role."""

    evidence_artifact_id: str
    raw_factor_id: str
    raw_factor_sha256: str
    source_repository: str
    source_revision: str
    source_artifacts: Mapping[str, str]
    sensor_family: str
    stream_id: str
    clock_id: str
    causal_frame_start: int
    causal_frame_stop: int
    correlation_group_ids: Sequence[str]
    inference_role: EvidenceInferenceRole
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        evidence_artifact_id = sha256_digest(
            self.evidence_artifact_id,
            name="evidence_artifact_id",
        )
        raw_factor_id = sha256_digest(self.raw_factor_id, name="raw_factor_id")
        raw_factor_sha256 = sha256_digest(
            self.raw_factor_sha256,
            name="raw_factor_sha256",
        )
        source_repository = repository_name(
            self.source_repository,
            name="source_repository",
        )
        source_revision = exact_revision(
            self.source_revision,
            name="source_revision",
        )
        source_artifacts = source_artifact_mapping(
            self.source_artifacts,
            name="source_artifacts",
        )
        sensor_family = nonempty_string(self.sensor_family, name="sensor_family")
        stream_id = nonempty_string(self.stream_id, name="stream_id")
        clock_id = nonempty_string(self.clock_id, name="clock_id")
        frame_start = genuine_integer(
            self.causal_frame_start,
            name="causal_frame_start",
            minimum=0,
        )
        frame_stop = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        _require(frame_start < frame_stop, "causal frame interval must be nonempty")
        groups = canonical_sorted_strings(
            self.correlation_group_ids,
            name="correlation_group_ids",
        )
        role = _role(self.inference_role)
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="evidence-use metadata",
        )

        object.__setattr__(self, "evidence_artifact_id", evidence_artifact_id)
        object.__setattr__(self, "raw_factor_id", raw_factor_id)
        object.__setattr__(self, "raw_factor_sha256", raw_factor_sha256)
        object.__setattr__(self, "source_repository", source_repository)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(self, "sensor_family", sensor_family)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "clock_id", clock_id)
        object.__setattr__(self, "causal_frame_start", frame_start)
        object.__setattr__(self, "causal_frame_stop", frame_stop)
        object.__setattr__(self, "correlation_group_ids", groups)
        object.__setattr__(self, "inference_role", role)
        object.__setattr__(self, "metadata", metadata)

    @property
    def entry_id(self) -> str:
        """Return the immutable content identity of this use record."""

        return content_id(self._descriptor())

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": EVIDENCE_USE_SCHEMA,
            "schema_version": EVIDENCE_USE_VERSION,
            "semantics": EVIDENCE_USE_SEMANTICS,
            "evidence_artifact_id": self.evidence_artifact_id,
            "raw_factor_id": self.raw_factor_id,
            "raw_factor_sha256": self.raw_factor_sha256,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_artifacts": self.source_artifacts,
            "sensor_family": self.sensor_family,
            "stream_id": self.stream_id,
            "clock_id": self.clock_id,
            "causal_frame_start": self.causal_frame_start,
            "causal_frame_stop": self.causal_frame_stop,
            "correlation_group_ids": self.correlation_group_ids,
            "inference_role": self.inference_role,
            "metadata": self.metadata,
        }

    def to_record(self) -> dict[str, object]:
        """Return the complete portable JSON record."""

        return {**self._descriptor(), "entry_id": self.entry_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "evidence use",
    ) -> EvidenceUseV1:
        """Validate and reconstruct one portable use record."""

        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_USE_FIELDS, name=name)
        if value["schema"] != EVIDENCE_USE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if (
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            != EVIDENCE_USE_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != EVIDENCE_USE_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        result = cls(
            evidence_artifact_id=value["evidence_artifact_id"],
            raw_factor_id=value["raw_factor_id"],
            raw_factor_sha256=value["raw_factor_sha256"],
            source_repository=value["source_repository"],
            source_revision=value["source_revision"],
            source_artifacts=value["source_artifacts"],
            sensor_family=value["sensor_family"],
            stream_id=value["stream_id"],
            clock_id=value["clock_id"],
            causal_frame_start=value["causal_frame_start"],
            causal_frame_stop=value["causal_frame_stop"],
            correlation_group_ids=value["correlation_group_ids"],
            inference_role=value["inference_role"],
            metadata=value["metadata"],
        )
        supplied_id = sha256_digest(value["entry_id"], name=f"{name} entry_id")
        if supplied_id != result.entry_id:
            raise ValueError(f"{name} entry_id does not match its content")
        return result


@dataclass(frozen=True)
class EvidenceUseLedgerV1:
    """Order-invariant ledger for one protocol case and causal prefix."""

    protocol_id: str
    case_id: str
    causal_frame_stop: int
    entries: Sequence[EvidenceUseV1] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        protocol_id = nonempty_string(self.protocol_id, name="protocol_id")
        case_id = nonempty_string(self.case_id, name="case_id")
        frame_stop = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        if isinstance(self.entries, (str, bytes)):
            raise ValueError("entries must be a sequence of EvidenceUseV1")
        entries = tuple(self.entries)
        if any(not isinstance(entry, EvidenceUseV1) for entry in entries):
            raise ValueError("entries must contain EvidenceUseV1 objects")
        entries = tuple(sorted(entries, key=lambda entry: entry.entry_id))

        entry_ids = [entry.entry_id for entry in entries]
        artifact_ids = [entry.evidence_artifact_id for entry in entries]
        raw_factor_ids = [entry.raw_factor_id for entry in entries]
        _require(len(set(entry_ids)) == len(entry_ids), "duplicate evidence-use entry")
        _require(
            len(set(artifact_ids)) == len(artifact_ids),
            "duplicate evidence artifact",
        )
        _require(
            len(set(raw_factor_ids)) == len(raw_factor_ids),
            "duplicate raw-factor identity",
        )

        factor_by_bytes: dict[str, str] = {}
        stages_by_group: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            _require(
                entry.causal_frame_stop <= frame_stop,
                "evidence use crosses the ledger causal prefix",
            )
            previous = factor_by_bytes.setdefault(
                entry.raw_factor_sha256,
                entry.raw_factor_id,
            )
            _require(
                previous == entry.raw_factor_id,
                "identical raw-factor bytes were relabelled",
            )
            stage = _stage(entry.inference_role)
            if stage is not None:
                for group_id in entry.correlation_group_ids:
                    stages_by_group[group_id].add(stage)

        for group_id, stages in stages_by_group.items():
            if "joint" in stages and len(stages) > 1:
                raise ValueError(
                    "joint evidence correlation group was also consumed through "
                    f"an independent path: {group_id}"
                )
            if {"state", "intervention"}.issubset(stages):
                raise ValueError(
                    "correlated evidence was consumed across state and intervention "
                    f"stages: {group_id}"
                )

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="evidence-use ledger metadata",
        )
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "causal_frame_stop", frame_stop)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "metadata", metadata)

    @property
    def ledger_id(self) -> str:
        """Return the order-invariant content identity of the ledger."""

        return content_id(self._descriptor())

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": EVIDENCE_USE_LEDGER_SCHEMA,
            "schema_version": EVIDENCE_USE_LEDGER_VERSION,
            "semantics": EVIDENCE_USE_SEMANTICS,
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "causal_frame_stop": self.causal_frame_stop,
            "entries": [entry.to_record() for entry in self.entries],
            "metadata": self.metadata,
            "claim_boundary": EVIDENCE_USE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        """Return the complete portable ledger record."""

        return {**self._descriptor(), "ledger_id": self.ledger_id}

    def summary(self) -> dict[str, object]:
        """Return compact lineage for embedding in a Bayesian belief artifact."""

        role_counts = Counter(entry.inference_role for entry in self.entries)
        groups = {
            group_id
            for entry in self.entries
            for group_id in entry.correlation_group_ids
        }
        return {
            "schema": EVIDENCE_USE_LEDGER_SCHEMA,
            "schema_version": EVIDENCE_USE_LEDGER_VERSION,
            "semantics": EVIDENCE_USE_SEMANTICS,
            "ledger_id": self.ledger_id,
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "causal_frame_stop": self.causal_frame_stop,
            "entry_count": len(self.entries),
            "correlation_group_count": len(groups),
            "role_counts": dict(sorted(role_counts.items())),
            "claim_boundary": EVIDENCE_USE_CLAIM_BOUNDARY,
        }

    def append(self, *entries: EvidenceUseV1) -> EvidenceUseLedgerV1:
        """Return a new validated ledger without mutating the current prefix."""

        return EvidenceUseLedgerV1(
            protocol_id=self.protocol_id,
            case_id=self.case_id,
            causal_frame_stop=self.causal_frame_stop,
            entries=(*self.entries, *entries),
            metadata=self.metadata,
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "evidence-use ledger",
    ) -> EvidenceUseLedgerV1:
        """Validate and reconstruct one portable ledger."""

        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_LEDGER_FIELDS, name=name)
        if value["schema"] != EVIDENCE_USE_LEDGER_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if (
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            != EVIDENCE_USE_LEDGER_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != EVIDENCE_USE_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        if value["claim_boundary"] != EVIDENCE_USE_CLAIM_BOUNDARY:
            raise ValueError(f"{name} claim boundary changed")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError(f"{name} entries must be a JSON array")
        result = cls(
            protocol_id=value["protocol_id"],
            case_id=value["case_id"],
            causal_frame_stop=value["causal_frame_stop"],
            entries=tuple(
                EvidenceUseV1.from_mapping(entry, name=f"{name} entry {index}")
                for index, entry in enumerate(raw_entries)
            ),
            metadata=value["metadata"],
        )
        supplied_id = sha256_digest(value["ledger_id"], name=f"{name} ledger_id")
        if supplied_id != result.ledger_id:
            raise ValueError(f"{name} ledger_id does not match its content")
        return result


class _MetadataBatch(Protocol):
    metadata: Mapping[str, Any]


class _Deform360ContactAnchor(Protocol):
    artifact_id: str
    object_id: str
    episode_id: int
    causal_frame_stop: int
    source_revision: str
    source_artifacts: Mapping[str, str]
    sensor_names: Sequence[str]
    correlation_group_ids: Sequence[str]


_BatchT = TypeVar("_BatchT", bound=_MetadataBatch)


def evidence_use_from_deform360_contact_anchor(
    anchor: _Deform360ContactAnchor,
    *,
    raw_factor_id: str,
    raw_factor_sha256: str,
    stream_id: str,
    clock_id: str,
    causal_frame_start: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceUseV1:
    """Describe one contact anchor as state-update evidence without raw taxel reuse."""

    lineage = plain_json(metadata or {})
    if not isinstance(lineage, dict):
        raise ValueError("contact-anchor evidence metadata must be a mapping")
    reserved = {"object_id", "episode_id", "sensor_names"}
    if reserved & set(lineage):
        raise ValueError("contact-anchor evidence metadata uses reserved fields")
    lineage.update(
        {
            "object_id": anchor.object_id,
            "episode_id": anchor.episode_id,
            "sensor_names": sorted(set(anchor.sensor_names)),
        }
    )
    return EvidenceUseV1(
        evidence_artifact_id=anchor.artifact_id,
        raw_factor_id=raw_factor_id,
        raw_factor_sha256=raw_factor_sha256,
        source_repository="brownu/deform360",
        source_revision=anchor.source_revision,
        source_artifacts=anchor.source_artifacts,
        sensor_family="tactile-proprioceptive-contact-anchor",
        stream_id=stream_id,
        clock_id=clock_id,
        causal_frame_start=causal_frame_start,
        causal_frame_stop=anchor.causal_frame_stop,
        correlation_group_ids=tuple(sorted(set(anchor.correlation_group_ids))),
        inference_role="state_update",
        metadata=lineage,
    )


def attach_evidence_use_ledger(
    batch: _BatchT,
    ledger: EvidenceUseLedgerV1,
) -> _BatchT:
    """Bind the complete ledger into a dataclass batch before inference."""

    if not isinstance(ledger, EvidenceUseLedgerV1):
        raise TypeError("ledger must be an EvidenceUseLedgerV1")
    metadata = plain_json(batch.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("batch metadata must be a mapping")
    if EVIDENCE_USE_LEDGER_METADATA_KEY in metadata:
        raise ValueError("batch already contains an evidence-use ledger")
    observed_cutoff = metadata.get("observation_causal_frame_stop")
    if observed_cutoff is not None:
        cutoff = genuine_integer(
            observed_cutoff,
            name="observation_causal_frame_stop",
            minimum=1,
        )
        if cutoff != ledger.causal_frame_stop:
            raise ValueError("batch and evidence-use ledger causal cutoffs differ")
    metadata[EVIDENCE_USE_LEDGER_METADATA_KEY] = ledger.to_record()
    return cast(_BatchT, replace(cast(Any, batch), metadata=metadata))


def save_evidence_use_ledger(
    ledger: EvidenceUseLedgerV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist one validated ledger."""

    if not isinstance(ledger, EvidenceUseLedgerV1):
        raise TypeError("ledger must be an EvidenceUseLedgerV1")
    write_atomic_json(ledger.to_record(), path, overwrite=overwrite)


def load_evidence_use_ledger(path: str | Path) -> EvidenceUseLedgerV1:
    """Load and independently revalidate one persisted ledger."""

    return EvidenceUseLedgerV1.from_mapping(
        load_strict_json_object(path, label="evidence-use ledger")
    )


__all__ = [
    "EVIDENCE_INFERENCE_ROLES",
    "EVIDENCE_USE_CLAIM_BOUNDARY",
    "EVIDENCE_USE_LEDGER_METADATA_KEY",
    "EVIDENCE_USE_LEDGER_SCHEMA",
    "EVIDENCE_USE_LEDGER_VERSION",
    "EVIDENCE_USE_SCHEMA",
    "EVIDENCE_USE_SEMANTICS",
    "EVIDENCE_USE_VERSION",
    "EvidenceInferenceRole",
    "EvidenceUseLedgerV1",
    "EvidenceUseV1",
    "attach_evidence_use_ledger",
    "evidence_use_from_deform360_contact_anchor",
    "load_evidence_use_ledger",
    "save_evidence_use_ledger",
]
