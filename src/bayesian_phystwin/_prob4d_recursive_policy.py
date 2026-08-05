"""Explicit recursive nuisance policy for Prob4D stream updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import content_id
from ._prob4d_stream_common import (
    RECURSIVE_NUISANCE_MODES,
    RECURSIVE_NUISANCE_POLICY_SCHEMA,
    RECURSIVE_NUISANCE_POLICY_VERSION,
    RecursiveNuisanceMode,
    _NUISANCE_POLICY_FIELDS,
    _sha256,
    _string_tuple,
)


@dataclass(frozen=True, slots=True)
class RecursiveNuisancePolicyV1:
    """Frozen rule preventing silent cross-update nuisance double counting."""

    mode: RecursiveNuisanceMode
    state_domain_id: str
    nuisance_family_ids: Sequence[str]
    conditional_independence_evidence_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    policy_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in RECURSIVE_NUISANCE_MODES:
            raise ValueError(
                "mode must be persistent_explicit_state or "
                "conditionally_independent_increments"
            )
        mode = cast(RecursiveNuisanceMode, self.mode)
        state_domain_id = _sha256(
            self.state_domain_id,
            name="state_domain_id",
        )
        families = _string_tuple(
            self.nuisance_family_ids,
            name="nuisance_family_ids",
        )
        evidence_id = self.conditional_independence_evidence_id
        if mode == "conditionally_independent_increments":
            evidence_id = _sha256(
                evidence_id,
                name="conditional_independence_evidence_id",
            )
        elif evidence_id is not None:
            raise ValueError(
                "conditional independence evidence is allowed only for the "
                "conditionally independent mode"
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="recursive nuisance policy metadata",
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "state_domain_id", state_domain_id)
        object.__setattr__(self, "nuisance_family_ids", families)
        object.__setattr__(
            self,
            "conditional_independence_evidence_id",
            evidence_id,
        )
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        supplied_id = self.policy_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="policy_id")
            if supplied_id != expected_id:
                raise ValueError(
                    "recursive nuisance policy_id does not match content"
                )
        object.__setattr__(self, "policy_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": RECURSIVE_NUISANCE_POLICY_SCHEMA,
            "schema_version": RECURSIVE_NUISANCE_POLICY_VERSION,
            "mode": self.mode,
            "state_domain_id": self.state_domain_id,
            "nuisance_family_ids": list(self.nuisance_family_ids),
            "conditional_independence_evidence_id": (
                self.conditional_independence_evidence_id
            ),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "policy_id": self.policy_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "recursive nuisance policy",
    ) -> RecursiveNuisancePolicyV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if set(value) != _NUISANCE_POLICY_FIELDS:
            raise ValueError(f"{name} fields changed")
        if value["schema"] != RECURSIVE_NUISANCE_POLICY_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != RECURSIVE_NUISANCE_POLICY_VERSION:
            raise ValueError(f"{name} version changed")
        return cls(
            mode=cast(RecursiveNuisanceMode, value["mode"]),
            state_domain_id=cast(str, value["state_domain_id"]),
            nuisance_family_ids=cast(
                Sequence[str],
                value["nuisance_family_ids"],
            ),
            conditional_independence_evidence_id=cast(
                str | None,
                value["conditional_independence_evidence_id"],
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            policy_id=cast(str, value["policy_id"]),
        )
