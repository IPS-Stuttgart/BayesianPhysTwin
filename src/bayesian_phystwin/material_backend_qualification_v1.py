"""Source-only qualification records for registered material backends.

The portable backend transports establish custody and structural compatibility.
This experimental record binds the numerical and information-order gates for one
exact runtime before it can enter the broader material-backend competence gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
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
from .material_backend_v1 import (
    BackendTransportV1,
    resolve_material_backend_profile,
)

MATERIAL_BACKEND_QUALIFICATION_SCHEMA: Final = (
    "bayesian-phystwin.material-backend-qualification"
)
MATERIAL_BACKEND_QUALIFICATION_VERSION: Final = 1
MATERIAL_BACKEND_QUALIFICATION_CLAIM_BOUNDARY: Final = (
    "Source-only numerical, information-order, and incumbent-parity "
    "qualification for one exact registered material-backend runtime. Passing "
    "this record does not establish backend competence, unseen-object accuracy, "
    "calibrated uncertainty, intervention benefit, deployment safety, or state "
    "of the art."
)

_DESCRIPTOR_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "claim_boundary",
        "canonical_profile_id",
        "producer_profile_id",
        "transport",
        "runtime_id",
        "qualification_protocol_id",
        "source_evidence_id",
        "source_group_ids",
        "incumbent_runtime_id",
        "units_coordinate_entity_order_valid",
        "deterministic_replay_valid",
        "maximum_zero_action_drift_m",
        "allowed_zero_action_drift_m",
        "maximum_rigid_equivariance_error_m",
        "allowed_rigid_equivariance_error_m",
        "time_step_refinement_relative_error",
        "allowed_time_step_refinement_relative_error",
        "topology_identity_preserved",
        "physical_sanity_violations",
        "gradient_claimed",
        "maximum_jacobian_relative_error",
        "allowed_jacobian_relative_error",
        "source_query_parity_rmse_m",
        "allowed_source_query_parity_rmse_m",
        "exact_fallback_verified",
        "protocol_frozen_before_source_outcomes",
        "target_outcomes_used",
        "metadata",
        "failure_reasons",
        "qualified",
    }
)
_PAYLOAD_FIELDS: Final = _DESCRIPTOR_FIELDS | {"artifact_id"}


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _optional_jacobian_error(
    value: object,
    *,
    name: str,
    required: bool,
) -> float | None:
    if not required:
        if value is not None:
            raise ValueError(f"{name} must be null when gradient_claimed is false")
        return None
    if value is None:
        raise ValueError(f"{name} is required when gradient_claimed is true")
    return _finite_nonnegative(value, name=name)


@dataclass(frozen=True, slots=True)
class MaterialBackendQualificationV1:
    """Replayable qualification gates for one exact material-backend runtime."""

    canonical_profile_id: str
    producer_profile_id: str
    transport: BackendTransportV1
    runtime_id: str
    qualification_protocol_id: str
    source_evidence_id: str
    source_group_ids: Sequence[str]
    incumbent_runtime_id: str
    units_coordinate_entity_order_valid: bool
    deterministic_replay_valid: bool
    maximum_zero_action_drift_m: float
    allowed_zero_action_drift_m: float
    maximum_rigid_equivariance_error_m: float
    allowed_rigid_equivariance_error_m: float
    time_step_refinement_relative_error: float
    allowed_time_step_refinement_relative_error: float
    topology_identity_preserved: bool
    physical_sanity_violations: int
    gradient_claimed: bool
    maximum_jacobian_relative_error: float | None
    allowed_jacobian_relative_error: float | None
    source_query_parity_rmse_m: float
    allowed_source_query_parity_rmse_m: float
    exact_fallback_verified: bool
    protocol_frozen_before_source_outcomes: bool
    target_outcomes_used: bool
    metadata: Mapping[str, Any] | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        canonical_profile_id = _canonical_string(
            self.canonical_profile_id,
            name="canonical_profile_id",
        )
        producer_profile_id = _canonical_string(
            self.producer_profile_id,
            name="producer_profile_id",
        )
        resolved = resolve_material_backend_profile(producer_profile_id)
        if resolved.profile_id != canonical_profile_id:
            raise ValueError(
                "producer_profile_id does not belong to canonical_profile_id"
            )
        if self.transport != resolved.transport:
            raise ValueError("transport does not match the registered producer profile")
        object.__setattr__(self, "canonical_profile_id", canonical_profile_id)
        object.__setattr__(self, "producer_profile_id", producer_profile_id)
        object.__setattr__(self, "transport", resolved.transport)

        for name in (
            "runtime_id",
            "qualification_protocol_id",
            "source_evidence_id",
            "incumbent_runtime_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )

        group_ids = canonical_string_tuple(
            self.source_group_ids,
            name="source_group_ids",
            allow_empty=False,
        )
        if len(group_ids) < 2:
            raise ValueError("at least two independent source groups are required")
        if any(value.strip() != value for value in group_ids):
            raise ValueError("source_group_ids must contain canonical strings")
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("source_group_ids must be unique")
        object.__setattr__(self, "source_group_ids", tuple(sorted(group_ids)))

        for name in (
            "units_coordinate_entity_order_valid",
            "deterministic_replay_valid",
            "topology_identity_preserved",
            "gradient_claimed",
            "exact_fallback_verified",
            "protocol_frozen_before_source_outcomes",
            "target_outcomes_used",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )

        object.__setattr__(
            self,
            "physical_sanity_violations",
            genuine_integer(
                self.physical_sanity_violations,
                name="physical_sanity_violations",
                minimum=0,
            ),
        )

        for name in (
            "maximum_zero_action_drift_m",
            "allowed_zero_action_drift_m",
            "maximum_rigid_equivariance_error_m",
            "allowed_rigid_equivariance_error_m",
            "time_step_refinement_relative_error",
            "allowed_time_step_refinement_relative_error",
            "source_query_parity_rmse_m",
            "allowed_source_query_parity_rmse_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )

        object.__setattr__(
            self,
            "maximum_jacobian_relative_error",
            _optional_jacobian_error(
                self.maximum_jacobian_relative_error,
                name="maximum_jacobian_relative_error",
                required=self.gradient_claimed,
            ),
        )
        object.__setattr__(
            self,
            "allowed_jacobian_relative_error",
            _optional_jacobian_error(
                self.allowed_jacobian_relative_error,
                name="allowed_jacobian_relative_error",
                required=self.gradient_claimed,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material backend qualification metadata",
            ),
        )

        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match qualification content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.units_coordinate_entity_order_valid:
            reasons.append("units-coordinate-entity-order")
        if not self.deterministic_replay_valid:
            reasons.append("deterministic-replay")
        if self.maximum_zero_action_drift_m > self.allowed_zero_action_drift_m:
            reasons.append("zero-action-equilibrium")
        if (
            self.maximum_rigid_equivariance_error_m
            > self.allowed_rigid_equivariance_error_m
        ):
            reasons.append("rigid-transform-equivariance")
        if (
            self.time_step_refinement_relative_error
            > self.allowed_time_step_refinement_relative_error
        ):
            reasons.append("time-step-refinement")
        if not self.topology_identity_preserved:
            reasons.append("topology-identity")
        if self.physical_sanity_violations:
            reasons.append("physical-sanity")
        if self.gradient_claimed:
            maximum = self.maximum_jacobian_relative_error
            allowed = self.allowed_jacobian_relative_error
            assert maximum is not None
            assert allowed is not None
            if maximum > allowed:
                reasons.append("finite-difference-jacobian")
        if self.source_query_parity_rmse_m > self.allowed_source_query_parity_rmse_m:
            reasons.append("source-query-parity")
        if not self.exact_fallback_verified:
            reasons.append("exact-fallback")
        if not self.protocol_frozen_before_source_outcomes:
            reasons.append("protocol-not-frozen-before-source-outcomes")
        if self.target_outcomes_used:
            reasons.append("target-outcomes-used")
        return tuple(reasons)

    @property
    def qualified(self) -> bool:
        return not self.failure_reasons

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": MATERIAL_BACKEND_QUALIFICATION_SCHEMA,
            "schema_version": MATERIAL_BACKEND_QUALIFICATION_VERSION,
            "claim_boundary": MATERIAL_BACKEND_QUALIFICATION_CLAIM_BOUNDARY,
            "canonical_profile_id": self.canonical_profile_id,
            "producer_profile_id": self.producer_profile_id,
            "transport": self.transport,
            "runtime_id": self.runtime_id,
            "qualification_protocol_id": self.qualification_protocol_id,
            "source_evidence_id": self.source_evidence_id,
            "source_group_ids": list(self.source_group_ids),
            "incumbent_runtime_id": self.incumbent_runtime_id,
            "units_coordinate_entity_order_valid": (
                self.units_coordinate_entity_order_valid
            ),
            "deterministic_replay_valid": self.deterministic_replay_valid,
            "maximum_zero_action_drift_m": self.maximum_zero_action_drift_m,
            "allowed_zero_action_drift_m": self.allowed_zero_action_drift_m,
            "maximum_rigid_equivariance_error_m": (
                self.maximum_rigid_equivariance_error_m
            ),
            "allowed_rigid_equivariance_error_m": (
                self.allowed_rigid_equivariance_error_m
            ),
            "time_step_refinement_relative_error": (
                self.time_step_refinement_relative_error
            ),
            "allowed_time_step_refinement_relative_error": (
                self.allowed_time_step_refinement_relative_error
            ),
            "topology_identity_preserved": self.topology_identity_preserved,
            "physical_sanity_violations": self.physical_sanity_violations,
            "gradient_claimed": self.gradient_claimed,
            "maximum_jacobian_relative_error": (self.maximum_jacobian_relative_error),
            "allowed_jacobian_relative_error": (self.allowed_jacobian_relative_error),
            "source_query_parity_rmse_m": self.source_query_parity_rmse_m,
            "allowed_source_query_parity_rmse_m": (
                self.allowed_source_query_parity_rmse_m
            ),
            "exact_fallback_verified": self.exact_fallback_verified,
            "protocol_frozen_before_source_outcomes": (
                self.protocol_frozen_before_source_outcomes
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "failure_reasons": list(self.failure_reasons),
            "qualified": self.qualified,
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
    ) -> MaterialBackendQualificationV1:
        require_exact_fields(
            payload,
            expected=_PAYLOAD_FIELDS,
            name="material backend qualification",
        )
        if payload.get("schema") != MATERIAL_BACKEND_QUALIFICATION_SCHEMA:
            raise ValueError("material backend qualification schema changed")
        if payload.get("schema_version") != MATERIAL_BACKEND_QUALIFICATION_VERSION:
            raise ValueError("material backend qualification version changed")
        if (
            payload.get("claim_boundary")
            != MATERIAL_BACKEND_QUALIFICATION_CLAIM_BOUNDARY
        ):
            raise ValueError("material backend qualification claim boundary changed")
        result = cls(
            canonical_profile_id=cast(str, payload.get("canonical_profile_id")),
            producer_profile_id=cast(str, payload.get("producer_profile_id")),
            transport=cast(BackendTransportV1, payload.get("transport")),
            runtime_id=cast(str, payload.get("runtime_id")),
            qualification_protocol_id=cast(
                str,
                payload.get("qualification_protocol_id"),
            ),
            source_evidence_id=cast(str, payload.get("source_evidence_id")),
            source_group_ids=cast(Sequence[str], payload.get("source_group_ids")),
            incumbent_runtime_id=cast(str, payload.get("incumbent_runtime_id")),
            units_coordinate_entity_order_valid=cast(
                bool,
                payload.get("units_coordinate_entity_order_valid"),
            ),
            deterministic_replay_valid=cast(
                bool,
                payload.get("deterministic_replay_valid"),
            ),
            maximum_zero_action_drift_m=cast(
                float,
                payload.get("maximum_zero_action_drift_m"),
            ),
            allowed_zero_action_drift_m=cast(
                float,
                payload.get("allowed_zero_action_drift_m"),
            ),
            maximum_rigid_equivariance_error_m=cast(
                float,
                payload.get("maximum_rigid_equivariance_error_m"),
            ),
            allowed_rigid_equivariance_error_m=cast(
                float,
                payload.get("allowed_rigid_equivariance_error_m"),
            ),
            time_step_refinement_relative_error=cast(
                float,
                payload.get("time_step_refinement_relative_error"),
            ),
            allowed_time_step_refinement_relative_error=cast(
                float,
                payload.get("allowed_time_step_refinement_relative_error"),
            ),
            topology_identity_preserved=cast(
                bool,
                payload.get("topology_identity_preserved"),
            ),
            physical_sanity_violations=cast(
                int,
                payload.get("physical_sanity_violations"),
            ),
            gradient_claimed=cast(bool, payload.get("gradient_claimed")),
            maximum_jacobian_relative_error=cast(
                float | None,
                payload.get("maximum_jacobian_relative_error"),
            ),
            allowed_jacobian_relative_error=cast(
                float | None,
                payload.get("allowed_jacobian_relative_error"),
            ),
            source_query_parity_rmse_m=cast(
                float,
                payload.get("source_query_parity_rmse_m"),
            ),
            allowed_source_query_parity_rmse_m=cast(
                float,
                payload.get("allowed_source_query_parity_rmse_m"),
            ),
            exact_fallback_verified=cast(
                bool,
                payload.get("exact_fallback_verified"),
            ),
            protocol_frozen_before_source_outcomes=cast(
                bool,
                payload.get("protocol_frozen_before_source_outcomes"),
            ),
            target_outcomes_used=cast(bool, payload.get("target_outcomes_used")),
            metadata=cast(Mapping[str, Any] | None, payload.get("metadata")),
            artifact_id=cast(str, payload.get("artifact_id")),
        )
        if canonical_json_bytes(result.to_payload()) != canonical_json_bytes(payload):
            raise ValueError("material backend qualification payload does not replay")
        return result


def require_qualified_material_backend_runtime(
    *,
    profile_id: str,
    producer_profile_id: str,
    runtime_id: str,
    qualification: MaterialBackendQualificationV1,
) -> MaterialBackendQualificationV1:
    """Require a passing record for the exact registered runtime."""

    if not isinstance(qualification, MaterialBackendQualificationV1):
        raise TypeError("qualification must be MaterialBackendQualificationV1")
    resolved = resolve_material_backend_profile(producer_profile_id)
    if resolved.profile_id != profile_id:
        raise ValueError(
            "producer_profile_id does not belong to the requested profile_id"
        )
    if qualification.canonical_profile_id != resolved.profile_id:
        raise ValueError("qualification canonical profile does not match the runtime")
    if qualification.producer_profile_id != resolved.producer_profile_id:
        raise ValueError("qualification producer profile does not match the runtime")
    if qualification.transport != resolved.transport:
        raise ValueError("qualification transport does not match the runtime")
    canonical_runtime_id = sha256_digest(runtime_id, name="runtime_id")
    if qualification.runtime_id != canonical_runtime_id:
        raise ValueError("qualification runtime_id does not match the runtime")
    if not qualification.qualified:
        reasons = ", ".join(qualification.failure_reasons)
        raise ValueError(f"material backend runtime is not qualified: {reasons}")
    return qualification


def save_material_backend_qualification_v1(
    qualification: MaterialBackendQualificationV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one canonical qualification record."""

    if not isinstance(qualification, MaterialBackendQualificationV1):
        raise TypeError("qualification must be MaterialBackendQualificationV1")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a literal Boolean")
    write_atomic_json(qualification.to_payload(), path, overwrite=overwrite)


def load_material_backend_qualification_v1(
    path: str | Path,
) -> MaterialBackendQualificationV1:
    """Load and fully replay one qualification record."""

    payload = load_strict_json_object(
        path,
        label="material backend qualification",
    )
    return MaterialBackendQualificationV1.from_payload(payload)


__all__ = [
    "MATERIAL_BACKEND_QUALIFICATION_CLAIM_BOUNDARY",
    "MATERIAL_BACKEND_QUALIFICATION_SCHEMA",
    "MATERIAL_BACKEND_QUALIFICATION_VERSION",
    "MaterialBackendQualificationV1",
    "load_material_backend_qualification_v1",
    "require_qualified_material_backend_runtime",
    "save_material_backend_qualification_v1",
]
