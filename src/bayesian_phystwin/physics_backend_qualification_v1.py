"""Source-only qualification for external physics backend runtimes."""

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
from .physics_backend_registry_v1 import (
    PhysicsBackendProfileV1,
    profile_from_mapping,
)

PHYSICS_BACKEND_QUALIFICATION_SCHEMA: Final = (
    "bayesian-phystwin.physics-backend-qualification"
)
PHYSICS_BACKEND_QUALIFICATION_VERSION: Final = 1
PHYSICS_BACKEND_QUALIFICATION_CLAIM_BOUNDARY: Final = (
    "Source-only numerical and incumbent-parity qualification for one exact "
    "external physics runtime. Passing this record does not establish unseen-"
    "object accuracy, calibrated uncertainty, intervention benefit, deployment "
    "safety, or state of the art."
)

_DESCRIPTOR_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "claim_boundary",
        "backend_profile",
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


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


@dataclass(frozen=True, slots=True)
class PhysicsBackendQualificationV1:
    """Replayable qualification gates for one exact external backend runtime."""

    backend_profile: PhysicsBackendProfileV1
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
    maximum_jacobian_relative_error: float
    allowed_jacobian_relative_error: float
    source_query_parity_rmse_m: float
    allowed_source_query_parity_rmse_m: float
    exact_fallback_verified: bool
    protocol_frozen_before_source_outcomes: bool
    target_outcomes_used: bool
    metadata: Mapping[str, Any] | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend_profile, PhysicsBackendProfileV1):
            raise TypeError("backend_profile must be PhysicsBackendProfileV1")
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
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("source_group_ids must be unique")
        object.__setattr__(self, "source_group_ids", tuple(sorted(group_ids)))

        for name in (
            "units_coordinate_entity_order_valid",
            "deterministic_replay_valid",
            "topology_identity_preserved",
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
            "maximum_jacobian_relative_error",
            "allowed_jacobian_relative_error",
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
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="physics backend qualification metadata",
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
        if self.physical_sanity_violations != 0:
            reasons.append("physical-sanity")
        if (
            self.maximum_jacobian_relative_error
            > self.allowed_jacobian_relative_error
        ):
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
            "schema": PHYSICS_BACKEND_QUALIFICATION_SCHEMA,
            "schema_version": PHYSICS_BACKEND_QUALIFICATION_VERSION,
            "claim_boundary": PHYSICS_BACKEND_QUALIFICATION_CLAIM_BOUNDARY,
            "backend_profile": self.backend_profile.to_dict(),
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
            "maximum_jacobian_relative_error": (
                self.maximum_jacobian_relative_error
            ),
            "allowed_jacobian_relative_error": (
                self.allowed_jacobian_relative_error
            ),
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
    ) -> PhysicsBackendQualificationV1:
        require_exact_fields(
            payload,
            expected=_PAYLOAD_FIELDS,
            name="physics backend qualification",
        )
        if payload.get("schema") != PHYSICS_BACKEND_QUALIFICATION_SCHEMA:
            raise ValueError("physics backend qualification schema changed")
        if payload.get("schema_version") != PHYSICS_BACKEND_QUALIFICATION_VERSION:
            raise ValueError("physics backend qualification version changed")
        if (
            payload.get("claim_boundary")
            != PHYSICS_BACKEND_QUALIFICATION_CLAIM_BOUNDARY
        ):
            raise ValueError("physics backend qualification claim boundary changed")
        profile_raw = payload.get("backend_profile")
        if not isinstance(profile_raw, Mapping):
            raise ValueError("backend_profile must be a JSON object")
        result = cls(
            backend_profile=profile_from_mapping(
                cast(Mapping[str, Any], profile_raw)
            ),
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
            maximum_jacobian_relative_error=cast(
                float,
                payload.get("maximum_jacobian_relative_error"),
            ),
            allowed_jacobian_relative_error=cast(
                float,
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
            target_outcomes_used=cast(
                bool,
                payload.get("target_outcomes_used"),
            ),
            metadata=cast(Mapping[str, Any], payload.get("metadata")),
            artifact_id=cast(str, payload.get("artifact_id")),
        )
        if canonical_json_bytes(result.to_payload()) != canonical_json_bytes(payload):
            raise ValueError("physics backend qualification payload does not replay")
        return result


def require_qualified_backend_runtime(
    backend_profile: PhysicsBackendProfileV1,
    runtime_id: str,
    qualification: PhysicsBackendQualificationV1,
) -> PhysicsBackendQualificationV1:
    """Require that one validated runtime is covered by a passing qualification."""

    if not isinstance(backend_profile, PhysicsBackendProfileV1):
        raise TypeError("backend_profile must be PhysicsBackendProfileV1")
    if not isinstance(qualification, PhysicsBackendQualificationV1):
        raise TypeError("qualification must be PhysicsBackendQualificationV1")
    canonical_runtime_id = sha256_digest(runtime_id, name="runtime_id")
    if qualification.backend_profile != backend_profile:
        raise ValueError("qualification backend profile does not match the runtime")
    if qualification.runtime_id != canonical_runtime_id:
        raise ValueError("qualification runtime_id does not match the runtime")
    if not qualification.qualified:
        reasons = ", ".join(qualification.failure_reasons)
        raise ValueError(f"external physics runtime is not qualified: {reasons}")
    return qualification


def save_physics_backend_qualification_v1(
    qualification: PhysicsBackendQualificationV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one canonical qualification record."""

    if not isinstance(qualification, PhysicsBackendQualificationV1):
        raise TypeError("qualification must be PhysicsBackendQualificationV1")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a literal Boolean")
    write_atomic_json(qualification.to_payload(), path, overwrite=overwrite)


def load_physics_backend_qualification_v1(
    path: str | Path,
) -> PhysicsBackendQualificationV1:
    """Load and fully replay one qualification record."""

    payload = load_strict_json_object(
        path,
        label="physics backend qualification",
    )
    return PhysicsBackendQualificationV1.from_payload(payload)


__all__ = [
    "PHYSICS_BACKEND_QUALIFICATION_CLAIM_BOUNDARY",
    "PHYSICS_BACKEND_QUALIFICATION_SCHEMA",
    "PHYSICS_BACKEND_QUALIFICATION_VERSION",
    "PhysicsBackendQualificationV1",
    "load_physics_backend_qualification_v1",
    "require_qualified_backend_runtime",
    "save_physics_backend_qualification_v1",
]
