"""Diagnose the transfer scope of learned physical-model discrepancy.

The module combines two questions that should not be conflated:

1. *What registered physical explanation remains compatible with the evidence?*
2. *At what strength may the resulting correction be transported?*

The first question is supplied by the portable target-level output of the
``explain / transport / probe / abstain`` diagnosis pipeline.  The second is
answered from pre-registered transfer evidence on four domain-shift axes:

- same object and same backend;
- same object and a new backend;
- a new object and the same backend; and
- a new object and a new backend.

Exact-coefficient transfer signatures define a small, explicit scope family.
The certificate retains every compatible scope rather than forcing a unique
label.  Independently, it chooses the strongest directly supported transport
tier for the requested axis.  A diagnostic ``none_of_the_above`` outcome,
unresolved target, missing direct evidence, or procedure-only evidence returns
the caller-owned fallback.

This is a finite evidence-contract primitive.  It does not infer unrestricted
natural causes or extrapolate a transport claim to an untested shift axis.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final

DISCREPANCY_SCOPE_TRANSPORT_SCHEMA: Final = (
    "bayesian_phystwin.discrepancy_scope_transport"
)
DISCREPANCY_SCOPE_TRANSPORT_VERSION: Final = 1
DISCREPANCY_SCOPE_TRANSPORT_SEMANTICS: Final = (
    "diagnose-cause-family-and-select-directly-supported-transport-tier-v1"
)
DIAGNOSIS_SCHEMA: Final = "bayesian_phystwin.explain_transport_probe_abstain"
DIAGNOSIS_VERSION: Final = 1
DIAGNOSIS_SEMANTICS: Final = (
    "cause-family-adequacy-target-transport-and-target-directed-probing-v1"
)
DISCREPANCY_SCOPE_TRANSPORT_CLAIM_BOUNDARY: Final = (
    "The certificate is exact only for the supplied finite scope signatures, "
    "portable target diagnosis, directly observed transfer axes, registered "
    "transport tiers, frozen evidence dispositions, and caller-owned fallback. "
    "A compatible scope is an invariance diagnosis over the tested axes, not "
    "proof of a unique natural physical cause. The certificate never promotes "
    "an untested axis, never turns procedure transfer into coefficient "
    "transport, and never deploys a probe recommendation before its response "
    "is observed. It does not establish nonlinear closure, arbitrary-object or "
    "arbitrary-backend generalization, target-domain calibration, deployment "
    "safety, or state of the art."
)


class TransferAxis(str, Enum):
    """Registered relation between the source and target domains."""

    SAME_OBJECT_SAME_BACKEND = "same_object_same_backend"
    SAME_OBJECT_NEW_BACKEND = "same_object_new_backend"
    NEW_OBJECT_SAME_BACKEND = "new_object_same_backend"
    NEW_OBJECT_NEW_BACKEND = "new_object_new_backend"


class TransportTier(str, Enum):
    """Ordered reusable object, from strongest to weakest."""

    EXACT_COEFFICIENTS = "exact_coefficients"
    QUERY_EFFECT = "query_effect"
    SCALAR_AMPLITUDE = "scalar_amplitude"
    UNCERTAINTY_STRUCTURE = "uncertainty_structure"
    PROCEDURE_ONLY = "procedure_only"


TRANSPORT_TIER_ORDER: Final = (
    TransportTier.EXACT_COEFFICIENTS,
    TransportTier.QUERY_EFFECT,
    TransportTier.SCALAR_AMPLITUDE,
    TransportTier.UNCERTAINTY_STRUCTURE,
    TransportTier.PROCEDURE_ONLY,
)


class EvidenceDisposition(str, Enum):
    """Frozen result of one registered transfer experiment."""

    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class ScopeHypothesis(str, Enum):
    """Finite exact-coefficient invariance hypotheses."""

    SHARED_PHYSICS = "shared_physics"
    OBJECT_SPECIFIC_BACKEND_STABLE = "object_specific_backend_stable"
    BACKEND_SPECIFIC_OBJECT_STABLE = "backend_specific_object_stable"
    OBJECT_BACKEND_LOCAL = "object_backend_local"


class ScopeStatus(str, Enum):
    """Status of the set-valued transfer-scope diagnosis."""

    UNIQUE = "unique"
    SET_VALUED = "set_valued"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NONE_OF_THE_ABOVE = "none_of_the_above"


class OperationalDisposition(str, Enum):
    """Current operational consequence of diagnosis and transport evidence."""

    TRANSPORT_EXACT_COEFFICIENTS = "transport_exact_coefficients"
    TRANSPORT_QUERY_EFFECT = "transport_query_effect"
    TRANSPORT_SCALAR_AMPLITUDE = "transport_scalar_amplitude"
    TRANSPORT_UNCERTAINTY_STRUCTURE = "transport_uncertainty_structure"
    PROCEDURE_ONLY_REFIT_REQUIRED = "procedure_only_refit_required"
    NO_CORRECTION = "no_correction"
    PROBE_THEN_REASSESS = "probe_then_reassess"
    NONE_OF_THE_ABOVE = "none_of_the_above"
    DIAGNOSIS_UNRESOLVED = "diagnosis_unresolved"
    EVIDENCE_ONLY = "evidence_only"
    EXACT_FALLBACK = "exact_fallback"


# A ``True`` entry means exact coefficients should transport on that axis.
# These signatures are deliberately simple and falsifiable.  Mixtures or
# non-monotone patterns may be rejected as none of the registered scopes.
_SCOPE_SIGNATURES: Final[dict[ScopeHypothesis, dict[TransferAxis, bool]]] = {
    ScopeHypothesis.SHARED_PHYSICS: {
        TransferAxis.SAME_OBJECT_SAME_BACKEND: True,
        TransferAxis.SAME_OBJECT_NEW_BACKEND: True,
        TransferAxis.NEW_OBJECT_SAME_BACKEND: True,
        TransferAxis.NEW_OBJECT_NEW_BACKEND: True,
    },
    ScopeHypothesis.OBJECT_SPECIFIC_BACKEND_STABLE: {
        TransferAxis.SAME_OBJECT_SAME_BACKEND: True,
        TransferAxis.SAME_OBJECT_NEW_BACKEND: True,
        TransferAxis.NEW_OBJECT_SAME_BACKEND: False,
        TransferAxis.NEW_OBJECT_NEW_BACKEND: False,
    },
    ScopeHypothesis.BACKEND_SPECIFIC_OBJECT_STABLE: {
        TransferAxis.SAME_OBJECT_SAME_BACKEND: True,
        TransferAxis.SAME_OBJECT_NEW_BACKEND: False,
        TransferAxis.NEW_OBJECT_SAME_BACKEND: True,
        TransferAxis.NEW_OBJECT_NEW_BACKEND: False,
    },
    ScopeHypothesis.OBJECT_BACKEND_LOCAL: {
        TransferAxis.SAME_OBJECT_SAME_BACKEND: True,
        TransferAxis.SAME_OBJECT_NEW_BACKEND: False,
        TransferAxis.NEW_OBJECT_SAME_BACKEND: False,
        TransferAxis.NEW_OBJECT_NEW_BACKEND: False,
    },
}

_TRANSPORTING_DIAGNOSES: Final = frozenset(
    {
        "explain_and_transport",
        "transport_without_cause",
    }
)
_FALLBACK_DIAGNOSES: Final = frozenset(
    {
        "partial_only_fallback",
        "abstain",
    }
)
_ALL_DIAGNOSES: Final = frozenset(
    {
        "no_detectable_error",
        "none_of_the_above",
        "explain_and_transport",
        "transport_without_cause",
        "probe_then_reassess",
        "partial_only_fallback",
        "abstain",
    }
)


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _optional_finite(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number or None")
    return result


def _canonical_id(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_json(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    if type(result) is not dict:
        raise ValueError(f"{name} must be a mapping")
    return result


@dataclass(frozen=True, slots=True)
class PortableTargetDiagnosisV1:
    """Portable target-level output from the diagnosis state machine."""

    pipeline_artifact_id: str
    target_id: str
    disposition: str
    adequacy_status: str
    transport_permitted: bool
    fallback_required_now: bool
    none_of_the_above: bool
    source_record_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pipeline_artifact_id",
            _digest(self.pipeline_artifact_id, name="pipeline_artifact_id"),
        )
        object.__setattr__(
            self,
            "target_id",
            _nonempty_string(self.target_id, name="target_id"),
        )
        disposition = _nonempty_string(self.disposition, name="disposition")
        if disposition not in _ALL_DIAGNOSES:
            raise ValueError(f"unsupported diagnostic disposition: {disposition}")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "adequacy_status",
            _nonempty_string(self.adequacy_status, name="adequacy_status"),
        )
        for name in (
            "transport_permitted",
            "fallback_required_now",
            "none_of_the_above",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if disposition in _TRANSPORTING_DIAGNOSES:
            if not self.transport_permitted or self.fallback_required_now:
                raise ValueError(
                    "transporting diagnosis must permit transport without fallback"
                )
            if self.none_of_the_above:
                raise ValueError("transporting diagnosis cannot be none_of_the_above")
        elif self.transport_permitted:
            raise ValueError("nontransporting diagnosis cannot permit transport")
        if disposition == "none_of_the_above":
            if not self.none_of_the_above or not self.fallback_required_now:
                raise ValueError(
                    "none_of_the_above diagnosis must require exact fallback"
                )
        elif self.none_of_the_above:
            raise ValueError(
                "none_of_the_above flag is valid only for that disposition"
            )
        if disposition in _FALLBACK_DIAGNOSES | {"probe_then_reassess"}:
            if not self.fallback_required_now:
                raise ValueError(
                    f"{disposition} must require the current exact fallback"
                )
        if self.source_record_id is not None:
            object.__setattr__(
                self,
                "source_record_id",
                _digest(self.source_record_id, name="source_record_id"),
            )

    @classmethod
    def from_pipeline_record(
        cls,
        pipeline_record: Mapping[str, Any],
        *,
        target_id: str,
        source_record_id: str | None = None,
    ) -> PortableTargetDiagnosisV1:
        """Extract and validate one target decision from a portable pipeline record."""

        if not isinstance(pipeline_record, Mapping):
            raise TypeError("pipeline_record must be a mapping")
        if pipeline_record.get("schema") != DIAGNOSIS_SCHEMA:
            raise ValueError("pipeline_record has the wrong schema")
        if pipeline_record.get("schema_version") != DIAGNOSIS_VERSION:
            raise ValueError("pipeline_record has the wrong schema version")
        if pipeline_record.get("semantics") != DIAGNOSIS_SEMANTICS:
            raise ValueError("pipeline_record has the wrong semantics")
        artifact_id = _digest(
            pipeline_record.get("artifact_id"),
            name="pipeline_record.artifact_id",
        )
        decisions = pipeline_record.get("target_decisions")
        if not isinstance(decisions, Sequence) or isinstance(
            decisions, (str, bytes, bytearray)
        ):
            raise ValueError("pipeline_record.target_decisions must be a sequence")
        matches = [
            decision
            for decision in decisions
            if isinstance(decision, Mapping) and decision.get("target_id") == target_id
        ]
        if len(matches) != 1:
            raise ValueError("target_id must identify exactly one target decision")
        decision = matches[0]
        return cls(
            pipeline_artifact_id=artifact_id,
            target_id=target_id,
            disposition=decision.get("disposition"),
            adequacy_status=decision.get("adequacy_status"),
            transport_permitted=decision.get("transport_permitted"),
            fallback_required_now=decision.get("fallback_required_now"),
            none_of_the_above=decision.get("none_of_the_above"),
            source_record_id=source_record_id,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "pipeline_artifact_id": self.pipeline_artifact_id,
            "target_id": self.target_id,
            "disposition": self.disposition,
            "adequacy_status": self.adequacy_status,
            "transport_permitted": self.transport_permitted,
            "fallback_required_now": self.fallback_required_now,
            "none_of_the_above": self.none_of_the_above,
            "source_record_id": self.source_record_id,
        }


@dataclass(frozen=True, slots=True)
class TransferEvidenceV1:
    """One frozen transfer-tier result on one domain-shift axis."""

    axis: TransferAxis
    tier: TransportTier
    disposition: EvidenceDisposition
    evidence_id: str | None
    relative_improvement: float | None = None
    wins: int | None = None
    total: int | None = None
    frozen_before_outcome: bool = True
    target_selection_free: bool = True
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axis = self.axis
        tier = self.tier
        disposition = self.disposition
        if not isinstance(axis, TransferAxis):
            try:
                axis = TransferAxis(axis)
            except (TypeError, ValueError) as error:
                raise ValueError("axis must be a TransferAxis") from error
        if not isinstance(tier, TransportTier):
            try:
                tier = TransportTier(tier)
            except (TypeError, ValueError) as error:
                raise ValueError("tier must be a TransportTier") from error
        if not isinstance(disposition, EvidenceDisposition):
            try:
                disposition = EvidenceDisposition(disposition)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "disposition must be an EvidenceDisposition"
                ) from error
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "disposition", disposition)

        if disposition is EvidenceDisposition.UNAVAILABLE:
            if self.evidence_id is not None:
                raise ValueError("unavailable evidence cannot claim an evidence_id")
            if any(
                value is not None
                for value in (self.relative_improvement, self.wins, self.total)
            ):
                raise ValueError(
                    "unavailable evidence cannot contain numerical outcomes"
                )
        else:
            object.__setattr__(
                self,
                "evidence_id",
                _digest(self.evidence_id, name="evidence_id"),
            )

        improvement = _optional_finite(
            self.relative_improvement,
            name="relative_improvement",
        )
        object.__setattr__(self, "relative_improvement", improvement)
        if (self.wins is None) != (self.total is None):
            raise ValueError("wins and total must be supplied together")
        if self.total is not None:
            if type(self.total) is not int or self.total <= 0:
                raise ValueError("total must be a positive int")
            if type(self.wins) is not int or not 0 <= self.wins <= self.total:
                raise ValueError("wins must be an int between zero and total")
        if type(self.frozen_before_outcome) is not bool:
            raise ValueError("frozen_before_outcome must be a bool")
        if type(self.target_selection_free) is not bool:
            raise ValueError("target_selection_free must be a bool")
        object.__setattr__(
            self,
            "description",
            str(self.description),
        )
        object.__setattr__(
            self,
            "metadata",
            _frozen_json(self.metadata, name="metadata"),
        )

    @property
    def decision_admissible(self) -> bool:
        """Whether the record may authorize a transport decision."""

        return (
            self.disposition is not EvidenceDisposition.UNAVAILABLE
            and self.frozen_before_outcome
            and self.target_selection_free
        )

    def to_record(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "tier": self.tier.value,
            "disposition": self.disposition.value,
            "evidence_id": self.evidence_id,
            "relative_improvement": self.relative_improvement,
            "wins": self.wins,
            "total": self.total,
            "frozen_before_outcome": self.frozen_before_outcome,
            "target_selection_free": self.target_selection_free,
            "decision_admissible": self.decision_admissible,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class DiscrepancyScopeTransportV1:
    """Set-valued discrepancy-scope diagnosis and tiered transport decision."""

    requested_axis: TransferAxis
    fallback_id: str
    evidence: Sequence[TransferEvidenceV1]
    diagnosis: PortableTargetDiagnosisV1 | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    scope_status: ScopeStatus = field(init=False)
    compatible_scopes: tuple[ScopeHypothesis, ...] = field(init=False)
    strongest_directly_supported_tier: TransportTier | None = field(init=False)
    operational_disposition: OperationalDisposition = field(init=False)
    fallback_required_now: bool = field(init=False)
    reason: str = field(init=False)

    def __post_init__(self) -> None:
        requested_axis = self.requested_axis
        if not isinstance(requested_axis, TransferAxis):
            try:
                requested_axis = TransferAxis(requested_axis)
            except (TypeError, ValueError) as error:
                raise ValueError("requested_axis must be a TransferAxis") from error
        fallback_id = _digest(self.fallback_id, name="fallback_id")
        records = tuple(self.evidence)
        if not records:
            raise ValueError("evidence must contain at least one record")
        if any(not isinstance(record, TransferEvidenceV1) for record in records):
            raise TypeError("evidence records must be TransferEvidenceV1 instances")
        keys = [(record.axis, record.tier) for record in records]
        if len(set(keys)) != len(keys):
            raise ValueError("evidence contains duplicate axis/tier records")
        if self.diagnosis is not None and not isinstance(
            self.diagnosis,
            PortableTargetDiagnosisV1,
        ):
            raise TypeError("diagnosis must be PortableTargetDiagnosisV1 or None")
        metadata = _frozen_json(self.metadata, name="metadata")

        exact_records = {
            record.axis: record
            for record in records
            if record.tier is TransportTier.EXACT_COEFFICIENTS
            and record.decision_admissible
        }
        observed_exact = {
            axis: record.disposition is EvidenceDisposition.SUPPORTED
            for axis, record in exact_records.items()
            if record.disposition is not EvidenceDisposition.UNAVAILABLE
        }
        informative_cross_axes = set(observed_exact) - {
            TransferAxis.SAME_OBJECT_SAME_BACKEND
        }

        compatible = tuple(
            scope
            for scope, signature in _SCOPE_SIGNATURES.items()
            if all(signature[axis] == passed for axis, passed in observed_exact.items())
        )
        if not informative_cross_axes:
            scope_status = ScopeStatus.INSUFFICIENT_EVIDENCE
        elif not compatible:
            scope_status = ScopeStatus.NONE_OF_THE_ABOVE
        elif len(compatible) == 1:
            scope_status = ScopeStatus.UNIQUE
        else:
            scope_status = ScopeStatus.SET_VALUED

        direct_records = {
            record.tier: record
            for record in records
            if record.axis is requested_axis
            and record.decision_admissible
            and record.disposition is not EvidenceDisposition.UNAVAILABLE
        }
        strongest = next(
            (
                tier
                for tier in TRANSPORT_TIER_ORDER
                if tier in direct_records
                and direct_records[tier].disposition is EvidenceDisposition.SUPPORTED
            ),
            None,
        )

        disposition, fallback, reason = self._operational_decision(
            diagnosis=self.diagnosis,
            scope_status=scope_status,
            strongest=strongest,
            requested_axis=requested_axis,
        )

        object.__setattr__(self, "requested_axis", requested_axis)
        object.__setattr__(self, "fallback_id", fallback_id)
        object.__setattr__(self, "evidence", records)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "scope_status", scope_status)
        object.__setattr__(self, "compatible_scopes", compatible)
        object.__setattr__(self, "strongest_directly_supported_tier", strongest)
        object.__setattr__(self, "operational_disposition", disposition)
        object.__setattr__(self, "fallback_required_now", fallback)
        object.__setattr__(self, "reason", reason)

        expected = _canonical_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match certificate content")
        object.__setattr__(self, "artifact_id", expected)

    @staticmethod
    def _operational_decision(
        *,
        diagnosis: PortableTargetDiagnosisV1 | None,
        scope_status: ScopeStatus,
        strongest: TransportTier | None,
        requested_axis: TransferAxis,
    ) -> tuple[OperationalDisposition, bool, str]:
        if diagnosis is None:
            return (
                OperationalDisposition.EVIDENCE_ONLY,
                True,
                "transfer-scope-evidence-has-no-bound-cause-diagnosis",
            )
        if diagnosis.disposition == "no_detectable_error":
            return (
                OperationalDisposition.NO_CORRECTION,
                True,
                "diagnosis-detects-no-correction-to-transport",
            )
        if diagnosis.disposition == "none_of_the_above":
            return (
                OperationalDisposition.NONE_OF_THE_ABOVE,
                True,
                "registered-cause-family-is-inadequate",
            )
        if diagnosis.disposition == "probe_then_reassess":
            return (
                OperationalDisposition.PROBE_THEN_REASSESS,
                True,
                "diagnostic-probe-outcome-is-required-before-transport",
            )
        if diagnosis.disposition in _FALLBACK_DIAGNOSES:
            return (
                OperationalDisposition.DIAGNOSIS_UNRESOLVED,
                True,
                "diagnosis-does-not-identify-a-complete-held-intervention-target",
            )
        if diagnosis.disposition not in _TRANSPORTING_DIAGNOSES:
            return (
                OperationalDisposition.EXACT_FALLBACK,
                True,
                "unsupported-diagnostic-disposition",
            )
        if scope_status is ScopeStatus.NONE_OF_THE_ABOVE:
            return (
                OperationalDisposition.NONE_OF_THE_ABOVE,
                True,
                "transfer-pattern-is-outside-the-registered-scope-family",
            )
        if strongest is None:
            return (
                OperationalDisposition.EXACT_FALLBACK,
                True,
                f"no-directly-supported-tier-for-{requested_axis.value}",
            )
        if strongest is TransportTier.PROCEDURE_ONLY:
            return (
                OperationalDisposition.PROCEDURE_ONLY_REFIT_REQUIRED,
                True,
                "only-the-fitting-procedure-transfers;-no-correction-is-deployed",
            )
        mapping = {
            TransportTier.EXACT_COEFFICIENTS: (
                OperationalDisposition.TRANSPORT_EXACT_COEFFICIENTS
            ),
            TransportTier.QUERY_EFFECT: (OperationalDisposition.TRANSPORT_QUERY_EFFECT),
            TransportTier.SCALAR_AMPLITUDE: (
                OperationalDisposition.TRANSPORT_SCALAR_AMPLITUDE
            ),
            TransportTier.UNCERTAINTY_STRUCTURE: (
                OperationalDisposition.TRANSPORT_UNCERTAINTY_STRUCTURE
            ),
        }
        return (
            mapping[strongest],
            False,
            f"diagnosis-allows-{strongest.value}-on-{requested_axis.value}",
        )

    def evidence_for(
        self,
        axis: TransferAxis,
        tier: TransportTier,
    ) -> TransferEvidenceV1 | None:
        for record in self.evidence:
            if record.axis is axis and record.tier is tier:
                return record
        return None

    def descriptor(self) -> dict[str, object]:
        """Return content-bearing fields, excluding the derived artifact ID."""

        return {
            "schema": DISCREPANCY_SCOPE_TRANSPORT_SCHEMA,
            "schema_version": DISCREPANCY_SCOPE_TRANSPORT_VERSION,
            "semantics": DISCREPANCY_SCOPE_TRANSPORT_SEMANTICS,
            "requested_axis": self.requested_axis.value,
            "fallback_id": self.fallback_id,
            "diagnosis": None if self.diagnosis is None else self.diagnosis.to_record(),
            "evidence": [
                record.to_record()
                for record in sorted(
                    self.evidence,
                    key=lambda item: (item.axis.value, item.tier.value),
                )
            ],
            "scope_status": self.scope_status.value,
            "compatible_scopes": [scope.value for scope in self.compatible_scopes],
            "strongest_directly_supported_tier": (
                None
                if self.strongest_directly_supported_tier is None
                else self.strongest_directly_supported_tier.value
            ),
            "operational_disposition": self.operational_disposition.value,
            "fallback_required_now": self.fallback_required_now,
            "reason": self.reason,
            "metadata": self.metadata,
            "claim_boundary": DISCREPANCY_SCOPE_TRANSPORT_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        result = self.descriptor()
        result["artifact_id"] = self.artifact_id
        return result

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DiscrepancyScopeTransportV1:
        """Reconstruct and independently verify a serialized certificate."""

        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        if record.get("schema") != DISCREPANCY_SCOPE_TRANSPORT_SCHEMA:
            raise ValueError("record has the wrong schema")
        if record.get("schema_version") != DISCREPANCY_SCOPE_TRANSPORT_VERSION:
            raise ValueError("record has the wrong schema version")
        if record.get("semantics") != DISCREPANCY_SCOPE_TRANSPORT_SEMANTICS:
            raise ValueError("record has the wrong semantics")

        diagnosis_value = record.get("diagnosis")
        diagnosis = None
        if diagnosis_value is not None:
            if not isinstance(diagnosis_value, Mapping):
                raise ValueError("record.diagnosis must be a mapping or null")
            diagnosis = PortableTargetDiagnosisV1(
                pipeline_artifact_id=diagnosis_value.get("pipeline_artifact_id"),
                target_id=diagnosis_value.get("target_id"),
                disposition=diagnosis_value.get("disposition"),
                adequacy_status=diagnosis_value.get("adequacy_status"),
                transport_permitted=diagnosis_value.get("transport_permitted"),
                fallback_required_now=diagnosis_value.get("fallback_required_now"),
                none_of_the_above=diagnosis_value.get("none_of_the_above"),
                source_record_id=diagnosis_value.get("source_record_id"),
            )

        evidence_value = record.get("evidence")
        if not isinstance(evidence_value, Sequence) or isinstance(
            evidence_value, (str, bytes, bytearray)
        ):
            raise ValueError("record.evidence must be a sequence")
        evidence = []
        for item in evidence_value:
            if not isinstance(item, Mapping):
                raise ValueError("every evidence item must be a mapping")
            evidence.append(
                TransferEvidenceV1(
                    axis=item.get("axis"),
                    tier=item.get("tier"),
                    disposition=item.get("disposition"),
                    evidence_id=item.get("evidence_id"),
                    relative_improvement=item.get("relative_improvement"),
                    wins=item.get("wins"),
                    total=item.get("total"),
                    frozen_before_outcome=item.get("frozen_before_outcome"),
                    target_selection_free=item.get("target_selection_free"),
                    description=item.get("description", ""),
                    metadata=item.get("metadata", {}),
                )
            )
        result = cls(
            requested_axis=record.get("requested_axis"),
            fallback_id=record.get("fallback_id"),
            evidence=evidence,
            diagnosis=diagnosis,
            metadata=record.get("metadata", {}),
            artifact_id=record.get("artifact_id"),
        )
        if result.to_record() != dict(record):
            raise ValueError(
                "serialized certificate contains inconsistent derived fields"
            )
        return result
