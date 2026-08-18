"""Fail-closed evidence promotion for registered material backends.

The material-backend registry records integration and transport compatibility.
This module records a separate, contiguous scientific-evidence ladder.  A
backend cannot skip from an adapter or native replay directly to a target-facing
claim: every promoted status binds the exact runtime, passing numerical
qualification, source decision, fresh-object decision, and downstream decision
required by its stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from ._portable_contracts import (
    canonical_json_bytes,
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .evidence_decision_v1 import EvidenceDecisionV1
from .material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
    require_qualified_material_backend_runtime,
)
from .material_backend_v1 import (
    BackendTransportV1,
    resolve_material_backend_profile,
)

MaterialBackendEvidenceStageV1 = Literal[
    "transport-registered",
    "adapter-tested",
    "native-runtime-replayed",
    "numerically-qualified",
    "source-competent",
    "fresh-object-validated",
    "downstream-query-benefit",
]

MATERIAL_BACKEND_EVIDENCE_SCHEMA: Final = (
    "bayesian-phystwin.material-backend-evidence-status"
)
MATERIAL_BACKEND_EVIDENCE_VERSION: Final = 1
MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY: Final = (
    "A material-backend evidence status records the highest contiguous evidence "
    "stage reached by one registered producer and exact runtime. Transport, "
    "adapter, native-replay, or numerical-qualification evidence alone does not "
    "establish source competence, fresh-object transfer, calibrated deployment "
    "uncertainty, Causal4D benefit, deployment safety, or state of the art."
)


@dataclass(frozen=True, slots=True)
class MaterialBackendEvidenceStageSpecV1:
    """One immutable stage in the material-backend promotion ladder."""

    code: str
    stage: MaterialBackendEvidenceStageV1
    title: str
    required_bindings: tuple[str, ...]
    interpretation: str

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "stage": self.stage,
            "title": self.title,
            "required_bindings": list(self.required_bindings),
            "interpretation": self.interpretation,
        }


MATERIAL_BACKEND_EVIDENCE_STAGES: Final = (
    MaterialBackendEvidenceStageSpecV1(
        code="T0",
        stage="transport-registered",
        title="Transport registered",
        required_bindings=("canonical backend family", "producer transport"),
        interpretation="Schema and custody compatibility only.",
    ),
    MaterialBackendEvidenceStageSpecV1(
        code="T1",
        stage="adapter-tested",
        title="Adapter tested",
        required_bindings=("adapter evidence digest",),
        interpretation="Dependency-free adapter behavior is covered by tests.",
    ),
    MaterialBackendEvidenceStageSpecV1(
        code="T2",
        stage="native-runtime-replayed",
        title="Native runtime replayed",
        required_bindings=("runtime digest", "native replay evidence digest"),
        interpretation="A pinned native runtime produced a replayable artifact.",
    ),
    MaterialBackendEvidenceStageSpecV1(
        code="T3",
        stage="numerically-qualified",
        title="Numerically qualified",
        required_bindings=(
            "passing MaterialBackendQualificationV1",
            "source group roster",
            "exact fallback",
        ),
        interpretation=(
            "Numerical and information-order gates pass for one exact runtime."
        ),
    ),
    MaterialBackendEvidenceStageSpecV1(
        code="T4",
        stage="source-competent",
        title="Source competent",
        required_bindings=("passing source-competence EvidenceDecisionV1",),
        interpretation=(
            "The guarded candidate passed the frozen source-only competence gate."
        ),
    ),
    MaterialBackendEvidenceStageSpecV1(
        code="T5",
        stage="fresh-object-validated",
        title="Fresh-object validated",
        required_bindings=(
            "authorized confirmatory EvidenceDecisionV1",
            "disjoint target group roster",
        ),
        interpretation=(
            "The exact source-selected runtime passed an independent object or "
            "session validation."
        ),
    ),
    MaterialBackendEvidenceStageSpecV1(
        code="T6",
        stage="downstream-query-benefit",
        title="Downstream query benefit",
        required_bindings=("authorized downstream-query EvidenceDecisionV1",),
        interpretation=(
            "A separately frozen downstream physical-query or Causal4D endpoint "
            "passed."
        ),
    ),
)

_STAGE_INDEX: Final = MappingProxyType(
    {spec.stage: index for index, spec in enumerate(MATERIAL_BACKEND_EVIDENCE_STAGES)}
)
_STAGE_CODE: Final = MappingProxyType(
    {spec.stage: spec.code for spec in MATERIAL_BACKEND_EVIDENCE_STAGES}
)
_VALID_STAGES: Final = frozenset(_STAGE_INDEX)

_DESCRIPTOR_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "claim_boundary",
        "stage",
        "stage_code",
        "canonical_profile_id",
        "producer_profile_id",
        "transport",
        "adapter_evidence_id",
        "runtime_id",
        "native_replay_evidence_id",
        "qualification_artifact_id",
        "source_decision_id",
        "target_decision_id",
        "downstream_decision_id",
        "source_group_ids",
        "target_group_ids",
        "exact_fallback_verified",
        "metadata",
    }
)
_PAYLOAD_FIELDS: Final = _DESCRIPTOR_FIELDS | {"artifact_id"}


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return sha256_digest(value, name=name)


def _canonical_groups(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    groups = canonical_string_tuple(values, name=name, allow_empty=True)
    if any(value.strip() != value for value in groups):
        raise ValueError(f"{name} must contain canonical strings")
    if len(set(groups)) != len(groups):
        raise ValueError(f"{name} must contain unique groups")
    return tuple(sorted(groups))


def _stage_index(stage: object, *, name: str = "stage") -> int:
    if type(stage) is not str or stage not in _VALID_STAGES:
        raise ValueError(f"{name} is not a material-backend evidence stage")
    return _STAGE_INDEX[cast(MaterialBackendEvidenceStageV1, stage)]


def describe_material_backend_evidence_stages() -> dict[str, object]:
    """Return the canonical machine-readable promotion ladder."""

    return {
        "schema": "bayesian-phystwin.material-backend-evidence-ladder",
        "schema_version": 1,
        "claim_boundary": MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY,
        "stages": [spec.to_record() for spec in MATERIAL_BACKEND_EVIDENCE_STAGES],
        "promotion_policy": (
            "Stages are contiguous and runtime-specific. A later stage requires "
            "all earlier bindings, and failed evidence remains terminal for the "
            "exact runtime and opened cohort."
        ),
    }


@dataclass(frozen=True, slots=True)
class MaterialBackendEvidenceStatusV1:
    """Content-addressed highest contiguous stage for one backend producer."""

    canonical_profile_id: str
    producer_profile_id: str
    transport: BackendTransportV1
    adapter_evidence_id: str | None = None
    runtime_id: str | None = None
    native_replay_evidence_id: str | None = None
    qualification_artifact_id: str | None = None
    source_decision_id: str | None = None
    target_decision_id: str | None = None
    downstream_decision_id: str | None = None
    source_group_ids: Sequence[str] = ()
    target_group_ids: Sequence[str] = ()
    exact_fallback_verified: bool = False
    metadata: Mapping[str, Any] | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        resolved = resolve_material_backend_profile(self.producer_profile_id)
        if resolved.profile_id != self.canonical_profile_id:
            raise ValueError(
                "producer_profile_id does not belong to canonical_profile_id"
            )
        if self.transport != resolved.transport:
            raise ValueError("transport does not match the registered producer profile")
        object.__setattr__(self, "canonical_profile_id", resolved.profile_id)
        object.__setattr__(self, "producer_profile_id", resolved.producer_profile_id)
        object.__setattr__(self, "transport", resolved.transport)

        for name in (
            "adapter_evidence_id",
            "runtime_id",
            "native_replay_evidence_id",
            "qualification_artifact_id",
            "source_decision_id",
            "target_decision_id",
            "downstream_decision_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_digest(getattr(self, name), name=name),
            )

        if (self.runtime_id is None) != (self.native_replay_evidence_id is None):
            raise ValueError(
                "runtime_id and native_replay_evidence_id must be supplied together"
            )
        if (
            self.native_replay_evidence_id is not None
            and self.adapter_evidence_id is None
        ):
            raise ValueError("native runtime replay requires adapter evidence")
        if (
            self.qualification_artifact_id is not None
            and self.native_replay_evidence_id is None
        ):
            raise ValueError("numerical qualification requires native runtime replay")
        if (
            self.source_decision_id is not None
            and self.qualification_artifact_id is None
        ):
            raise ValueError("source competence requires numerical qualification")
        if self.target_decision_id is not None and self.source_decision_id is None:
            raise ValueError("fresh-object validation requires source competence")
        if self.downstream_decision_id is not None and self.target_decision_id is None:
            raise ValueError(
                "downstream query benefit requires fresh-object validation"
            )

        source_groups = _canonical_groups(
            self.source_group_ids,
            name="source_group_ids",
        )
        target_groups = _canonical_groups(
            self.target_group_ids,
            name="target_group_ids",
        )
        if self.qualification_artifact_id is None and source_groups:
            raise ValueError(
                "source_group_ids are admitted only with numerical qualification"
            )
        if self.qualification_artifact_id is not None and len(source_groups) < 2:
            raise ValueError(
                "numerical qualification requires at least two source groups"
            )
        if self.target_decision_id is None and target_groups:
            raise ValueError(
                "target_group_ids are admitted only with fresh-object validation"
            )
        if self.target_decision_id is not None and not target_groups:
            raise ValueError("fresh-object validation requires target_group_ids")
        overlap = sorted(set(source_groups) & set(target_groups))
        if overlap:
            raise ValueError(
                "source_group_ids and target_group_ids must be disjoint: "
                f"{overlap}"
            )
        object.__setattr__(self, "source_group_ids", source_groups)
        object.__setattr__(self, "target_group_ids", target_groups)

        fallback = genuine_boolean(
            self.exact_fallback_verified,
            name="exact_fallback_verified",
        )
        if self.qualification_artifact_id is None and fallback:
            raise ValueError(
                "exact_fallback_verified is stage-bearing only with qualification"
            )
        if self.qualification_artifact_id is not None and not fallback:
            raise ValueError("numerical qualification requires exact fallback")
        object.__setattr__(self, "exact_fallback_verified", fallback)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material backend evidence metadata",
            ),
        )

        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match evidence status content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def stage(self) -> MaterialBackendEvidenceStageV1:
        if self.downstream_decision_id is not None:
            return "downstream-query-benefit"
        if self.target_decision_id is not None:
            return "fresh-object-validated"
        if self.source_decision_id is not None:
            return "source-competent"
        if self.qualification_artifact_id is not None:
            return "numerically-qualified"
        if self.native_replay_evidence_id is not None:
            return "native-runtime-replayed"
        if self.adapter_evidence_id is not None:
            return "adapter-tested"
        return "transport-registered"

    @property
    def stage_code(self) -> str:
        return _STAGE_CODE[self.stage]

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": MATERIAL_BACKEND_EVIDENCE_SCHEMA,
            "schema_version": MATERIAL_BACKEND_EVIDENCE_VERSION,
            "claim_boundary": MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY,
            "stage": self.stage,
            "stage_code": self.stage_code,
            "canonical_profile_id": self.canonical_profile_id,
            "producer_profile_id": self.producer_profile_id,
            "transport": self.transport,
            "adapter_evidence_id": self.adapter_evidence_id,
            "runtime_id": self.runtime_id,
            "native_replay_evidence_id": self.native_replay_evidence_id,
            "qualification_artifact_id": self.qualification_artifact_id,
            "source_decision_id": self.source_decision_id,
            "target_decision_id": self.target_decision_id,
            "downstream_decision_id": self.downstream_decision_id,
            "source_group_ids": list(self.source_group_ids),
            "target_group_ids": list(self.target_group_ids),
            "exact_fallback_verified": self.exact_fallback_verified,
            "metadata": plain_json(self.metadata),
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self.descriptor()
        artifact_id = self.artifact_id
        assert artifact_id is not None
        payload["artifact_id"] = artifact_id
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> MaterialBackendEvidenceStatusV1:
        require_exact_fields(
            payload,
            expected=_PAYLOAD_FIELDS,
            name="material backend evidence status",
        )
        if payload.get("schema") != MATERIAL_BACKEND_EVIDENCE_SCHEMA:
            raise ValueError("material backend evidence status schema changed")
        if payload.get("schema_version") != MATERIAL_BACKEND_EVIDENCE_VERSION:
            raise ValueError("material backend evidence status version changed")
        if payload.get("claim_boundary") != MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY:
            raise ValueError("material backend evidence claim boundary changed")
        result = cls(
            canonical_profile_id=cast(str, payload.get("canonical_profile_id")),
            producer_profile_id=cast(str, payload.get("producer_profile_id")),
            transport=cast(BackendTransportV1, payload.get("transport")),
            adapter_evidence_id=cast(str | None, payload.get("adapter_evidence_id")),
            runtime_id=cast(str | None, payload.get("runtime_id")),
            native_replay_evidence_id=cast(
                str | None,
                payload.get("native_replay_evidence_id"),
            ),
            qualification_artifact_id=cast(
                str | None,
                payload.get("qualification_artifact_id"),
            ),
            source_decision_id=cast(str | None, payload.get("source_decision_id")),
            target_decision_id=cast(str | None, payload.get("target_decision_id")),
            downstream_decision_id=cast(
                str | None,
                payload.get("downstream_decision_id"),
            ),
            source_group_ids=cast(Sequence[str], payload.get("source_group_ids")),
            target_group_ids=cast(Sequence[str], payload.get("target_group_ids")),
            exact_fallback_verified=cast(
                bool,
                payload.get("exact_fallback_verified"),
            ),
            metadata=cast(Mapping[str, Any] | None, payload.get("metadata")),
            artifact_id=cast(str, payload.get("artifact_id")),
        )
        if canonical_json_bytes(result.to_payload()) != canonical_json_bytes(payload):
            raise ValueError("material backend evidence status does not replay")
        return result


def _metadata_binding(
    decision: EvidenceDecisionV1,
    *,
    key: str,
    expected: str,
) -> None:
    value = decision.metadata.get(key)
    if value != expected:
        raise ValueError(f"evidence decision metadata {key} does not match")


def _require_stage_decision(
    decision: EvidenceDecisionV1,
    *,
    role: str,
    canonical_profile_id: str,
    producer_profile_id: str,
    runtime_id: str,
    parent_key: str,
    parent_id: str,
    target_facing: bool,
) -> str:
    if not isinstance(decision, EvidenceDecisionV1):
        raise TypeError(f"{role} decision must be EvidenceDecisionV1")
    if decision.status != "pass":
        raise ValueError(f"{role} decision must pass")
    if target_facing:
        if decision.run_classification != "confirmatory":
            raise ValueError(f"{role} decision must be confirmatory")
        if not decision.claim_authorized:
            raise ValueError(f"{role} decision must authorize its bounded claim")
    else:
        if decision.run_classification == "infrastructure":
            raise ValueError("source-competence decision must be scientific evidence")
        if decision.claim_authorized:
            raise ValueError(
                "source-competence decision cannot authorize a target-facing claim"
            )
    _metadata_binding(decision, key="evidence_role", expected=role)
    _metadata_binding(
        decision,
        key="canonical_profile_id",
        expected=canonical_profile_id,
    )
    _metadata_binding(
        decision,
        key="producer_profile_id",
        expected=producer_profile_id,
    )
    _metadata_binding(decision, key="runtime_id", expected=runtime_id)
    _metadata_binding(decision, key=parent_key, expected=parent_id)
    return decision.decision_id


def verify_material_backend_evidence_status_v1(
    status: MaterialBackendEvidenceStatusV1,
    *,
    qualification: MaterialBackendQualificationV1 | None = None,
    source_decision: EvidenceDecisionV1 | None = None,
    target_decision: EvidenceDecisionV1 | None = None,
    downstream_decision: EvidenceDecisionV1 | None = None,
) -> MaterialBackendEvidenceStatusV1:
    """Replay all external evidence bindings claimed by ``status``."""

    if not isinstance(status, MaterialBackendEvidenceStatusV1):
        raise TypeError("status must be MaterialBackendEvidenceStatusV1")
    stage_index = _STAGE_INDEX[status.stage]

    if stage_index >= _STAGE_INDEX["numerically-qualified"]:
        if qualification is None:
            raise ValueError("qualification is required to verify this evidence stage")
        runtime_id = status.runtime_id
        assert runtime_id is not None
        require_qualified_material_backend_runtime(
            profile_id=status.canonical_profile_id,
            producer_profile_id=status.producer_profile_id,
            runtime_id=runtime_id,
            qualification=qualification,
        )
        if qualification.artifact_id != status.qualification_artifact_id:
            raise ValueError("qualification artifact does not match evidence status")
        if tuple(qualification.source_group_ids) != tuple(status.source_group_ids):
            raise ValueError("qualification source groups do not match evidence status")
        if not qualification.exact_fallback_verified:
            raise ValueError("qualification does not verify exact fallback")
    elif qualification is not None:
        raise ValueError("qualification was supplied for a pre-qualification stage")

    if stage_index >= _STAGE_INDEX["source-competent"]:
        if source_decision is None:
            raise ValueError(
                "source_decision is required to verify this evidence stage"
            )
        runtime_id = status.runtime_id
        qualification_id = status.qualification_artifact_id
        assert runtime_id is not None
        assert qualification_id is not None
        source_id = _require_stage_decision(
            source_decision,
            role="source-competence",
            canonical_profile_id=status.canonical_profile_id,
            producer_profile_id=status.producer_profile_id,
            runtime_id=runtime_id,
            parent_key="qualification_artifact_id",
            parent_id=qualification_id,
            target_facing=False,
        )
        if source_id != status.source_decision_id:
            raise ValueError("source decision does not match evidence status")
    elif source_decision is not None:
        raise ValueError("source_decision was supplied before source competence")

    if stage_index >= _STAGE_INDEX["fresh-object-validated"]:
        if target_decision is None:
            raise ValueError(
                "target_decision is required to verify this evidence stage"
            )
        runtime_id = status.runtime_id
        source_id = status.source_decision_id
        assert runtime_id is not None
        assert source_id is not None
        target_id = _require_stage_decision(
            target_decision,
            role="fresh-object-validation",
            canonical_profile_id=status.canonical_profile_id,
            producer_profile_id=status.producer_profile_id,
            runtime_id=runtime_id,
            parent_key="source_decision_id",
            parent_id=source_id,
            target_facing=True,
        )
        if target_id != status.target_decision_id:
            raise ValueError("target decision does not match evidence status")
    elif target_decision is not None:
        raise ValueError("target_decision was supplied before fresh-object validation")

    if stage_index >= _STAGE_INDEX["downstream-query-benefit"]:
        if downstream_decision is None:
            raise ValueError(
                "downstream_decision is required to verify this evidence stage"
            )
        runtime_id = status.runtime_id
        target_id = status.target_decision_id
        assert runtime_id is not None
        assert target_id is not None
        downstream_id = _require_stage_decision(
            downstream_decision,
            role="downstream-query-benefit",
            canonical_profile_id=status.canonical_profile_id,
            producer_profile_id=status.producer_profile_id,
            runtime_id=runtime_id,
            parent_key="target_decision_id",
            parent_id=target_id,
            target_facing=True,
        )
        if downstream_id != status.downstream_decision_id:
            raise ValueError("downstream decision does not match evidence status")
    elif downstream_decision is not None:
        raise ValueError(
            "downstream_decision was supplied before downstream-query benefit"
        )

    return status


def build_material_backend_evidence_status_v1(
    *,
    canonical_profile_id: str,
    producer_profile_id: str,
    adapter_evidence_id: str | None = None,
    runtime_id: str | None = None,
    native_replay_evidence_id: str | None = None,
    qualification: MaterialBackendQualificationV1 | None = None,
    source_decision: EvidenceDecisionV1 | None = None,
    target_decision: EvidenceDecisionV1 | None = None,
    downstream_decision: EvidenceDecisionV1 | None = None,
    target_group_ids: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> MaterialBackendEvidenceStatusV1:
    """Build and replay one contiguous evidence status from exact evidence."""

    resolved = resolve_material_backend_profile(producer_profile_id)
    if resolved.profile_id != canonical_profile_id:
        raise ValueError(
            "producer_profile_id does not belong to canonical_profile_id"
        )
    normalized_runtime = _optional_digest(runtime_id, name="runtime_id")
    qualification_id: str | None = None
    source_group_ids: Sequence[str] = ()
    exact_fallback_verified = False
    if qualification is not None:
        if normalized_runtime is None:
            raise ValueError("qualification requires runtime_id")
        require_qualified_material_backend_runtime(
            profile_id=resolved.profile_id,
            producer_profile_id=resolved.producer_profile_id,
            runtime_id=normalized_runtime,
            qualification=qualification,
        )
        qualification_id = qualification.artifact_id
        assert qualification_id is not None
        source_group_ids = qualification.source_group_ids
        exact_fallback_verified = qualification.exact_fallback_verified

    source_decision_id = (
        None if source_decision is None else source_decision.decision_id
    )
    target_decision_id = (
        None if target_decision is None else target_decision.decision_id
    )
    downstream_decision_id = (
        None if downstream_decision is None else downstream_decision.decision_id
    )
    result = MaterialBackendEvidenceStatusV1(
        canonical_profile_id=resolved.profile_id,
        producer_profile_id=resolved.producer_profile_id,
        transport=resolved.transport,
        adapter_evidence_id=adapter_evidence_id,
        runtime_id=normalized_runtime,
        native_replay_evidence_id=native_replay_evidence_id,
        qualification_artifact_id=qualification_id,
        source_decision_id=source_decision_id,
        target_decision_id=target_decision_id,
        downstream_decision_id=downstream_decision_id,
        source_group_ids=source_group_ids,
        target_group_ids=target_group_ids,
        exact_fallback_verified=exact_fallback_verified,
        metadata=metadata,
    )
    return verify_material_backend_evidence_status_v1(
        result,
        qualification=qualification,
        source_decision=source_decision,
        target_decision=target_decision,
        downstream_decision=downstream_decision,
    )


def require_material_backend_evidence_stage(
    status: MaterialBackendEvidenceStatusV1,
    minimum_stage: MaterialBackendEvidenceStageV1,
    *,
    qualification: MaterialBackendQualificationV1 | None = None,
    source_decision: EvidenceDecisionV1 | None = None,
    target_decision: EvidenceDecisionV1 | None = None,
    downstream_decision: EvidenceDecisionV1 | None = None,
) -> MaterialBackendEvidenceStatusV1:
    """Require at least ``minimum_stage`` and replay claim-bearing bindings."""

    if not isinstance(status, MaterialBackendEvidenceStatusV1):
        raise TypeError("status must be MaterialBackendEvidenceStatusV1")
    minimum_index = _stage_index(minimum_stage, name="minimum_stage")
    if _STAGE_INDEX[status.stage] < minimum_index:
        raise ValueError(
            f"material backend evidence stage {status.stage} is below "
            f"{minimum_stage}"
        )
    if minimum_index >= _STAGE_INDEX["numerically-qualified"]:
        verify_material_backend_evidence_status_v1(
            status,
            qualification=qualification,
            source_decision=source_decision,
            target_decision=target_decision,
            downstream_decision=downstream_decision,
        )
    return status


def save_material_backend_evidence_status_v1(
    status: MaterialBackendEvidenceStatusV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one canonical material-backend evidence status."""

    if not isinstance(status, MaterialBackendEvidenceStatusV1):
        raise TypeError("status must be MaterialBackendEvidenceStatusV1")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a literal Boolean")
    write_atomic_json(status.to_payload(), path, overwrite=overwrite)


def load_material_backend_evidence_status_v1(
    path: str | Path,
) -> MaterialBackendEvidenceStatusV1:
    """Load and replay one content-addressed status record."""

    payload = load_strict_json_object(path, label="material backend evidence status")
    return MaterialBackendEvidenceStatusV1.from_payload(payload)


__all__ = [
    "MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY",
    "MATERIAL_BACKEND_EVIDENCE_SCHEMA",
    "MATERIAL_BACKEND_EVIDENCE_STAGES",
    "MATERIAL_BACKEND_EVIDENCE_VERSION",
    "MaterialBackendEvidenceStageSpecV1",
    "MaterialBackendEvidenceStageV1",
    "MaterialBackendEvidenceStatusV1",
    "build_material_backend_evidence_status_v1",
    "describe_material_backend_evidence_stages",
    "load_material_backend_evidence_status_v1",
    "require_material_backend_evidence_stage",
    "save_material_backend_evidence_status_v1",
    "verify_material_backend_evidence_status_v1",
]
