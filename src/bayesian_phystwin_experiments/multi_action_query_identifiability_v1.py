"""Multi-action query identifiability under a jointly declared nuisance model.

The module composes action-specific prospective observation designs into one
local linear problem while preserving nuisance coefficients that may be shared
across actions. It is an experimental evidence instrument rather than a stable
runtime API.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QueryIdentifiabilityCertificateV2,
    QueryIdentifiabilityStatus,
)

MULTI_ACTION_QUERY_IDENTIFIABILITY_SCHEMA: Final = (
    "bayesian_phystwin.multi_action_query_identifiability"
)
MULTI_ACTION_QUERY_IDENTIFIABILITY_VERSION: Final = 1
MULTI_ACTION_QUERY_IDENTIFIABILITY_SEMANTICS: Final = (
    "stacked-action-query-identifiability-with-joint-nuisance-v1"
)
MULTI_ACTION_QUERY_IDENTIFIABILITY_CLAIM_BOUNDARY: Final = (
    "A passing artifact establishes local linear identifiability of the stacked "
    "registered queries under the exact action-specific physical designs, "
    "transported query maps, joint nuisance design, whitening, coordinates, and "
    "numerical tolerances. It does not establish a unique physical cause, "
    "correctness of the supplied models, global nonlinear identifiability, "
    "unseen-object transfer, real-provider competence, uncertainty calibration, "
    "safe action execution, deployment safety, Causal4D benefit, or state of the art."
)


def _digest(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={64}))


def _matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _immutable(value: object) -> np.ndarray:
    return cast(np.ndarray, immutable_array(value, dtype=np.float64))


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class ActionIdentifiabilityBlockV1:
    """One action-specific physical observation and transported-query block."""

    action_id: str
    physical_response_id: str
    observation_mapping_id: str
    query_transport_id: str
    whitened_physical_design: np.ndarray
    transported_query_map: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise ValueError("action_id must be a nonempty literal string")
        for name in (
            "physical_response_id",
            "observation_mapping_id",
            "query_transport_id",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name=name),
            )
        physical = _matrix(
            self.whitened_physical_design,
            name="whitened_physical_design",
        )
        query = _matrix(self.transported_query_map, name="transported_query_map")
        if physical.shape[0] == 0 or physical.shape[1] == 0:
            raise ValueError("whitened_physical_design must have nonzero dimensions")
        if query.shape[0] == 0 or query.shape[1] != physical.shape[1]:
            raise ValueError(
                "transported_query_map must have nonzero rows and one column "
                "per latent coordinate"
            )
        object.__setattr__(self, "whitened_physical_design", _immutable(physical))
        object.__setattr__(self, "transported_query_map", _immutable(query))
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="action identifiability metadata",
            ),
        )
        expected = cast(str, content_id(self.descriptor()))
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError("action block artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def observation_dimension(self) -> int:
        return int(self.whitened_physical_design.shape[0])

    @property
    def latent_dimension(self) -> int:
        return int(self.whitened_physical_design.shape[1])

    @property
    def query_dimension(self) -> int:
        return int(self.transported_query_map.shape[0])

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": MULTI_ACTION_QUERY_IDENTIFIABILITY_SCHEMA,
            "schema_version": MULTI_ACTION_QUERY_IDENTIFIABILITY_VERSION,
            "artifact_kind": "ActionIdentifiabilityBlockV1",
            "semantics": MULTI_ACTION_QUERY_IDENTIFIABILITY_SEMANTICS,
            "action_id": self.action_id,
            "physical_response_id": self.physical_response_id,
            "observation_mapping_id": self.observation_mapping_id,
            "query_transport_id": self.query_transport_id,
            "whitened_physical_design": _array_record(self.whitened_physical_design),
            "transported_query_map": _array_record(self.transported_query_map),
            "metadata": plain_json(self.metadata),
            "claim_boundary": MULTI_ACTION_QUERY_IDENTIFIABILITY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class ActionContributionV1:
    """Leave-one-action-out diagnostic under the same stacked query."""

    action_id: str
    without_action_status: QueryIdentifiabilityStatus
    without_action_energy_fraction: float
    without_action_normalized_residual: float
    without_action_physical_rank: int
    energy_fraction_loss: float

    def to_record(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "without_action_status": self.without_action_status.value,
            "without_action_energy_fraction": self.without_action_energy_fraction,
            "without_action_normalized_residual": (
                self.without_action_normalized_residual
            ),
            "without_action_physical_rank": self.without_action_physical_rank,
            "energy_fraction_loss": self.energy_fraction_loss,
        }


@dataclass(frozen=True, slots=True)
class MultiActionQueryIdentifiabilityCertificateV1:
    """Stack action-conditioned designs and test the joint transported query."""

    latent_coordinates_id: str
    whitening_id: str
    joint_nuisance_design_id: str
    joint_query_id: str
    action_blocks: Sequence[ActionIdentifiabilityBlockV1]
    joint_whitened_nuisance_design: np.ndarray
    relative_rank_tolerance: float = 1e-10
    absolute_rank_tolerance: float = 1e-12
    identifiability_tolerance: float = 1e-8
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    stacked_physical_design: np.ndarray = field(init=False, repr=False)
    stacked_query_map: np.ndarray = field(init=False, repr=False)
    joint_certificate: QueryIdentifiabilityCertificateV2 = field(
        init=False,
        repr=False,
    )
    single_action_statuses: tuple[tuple[str, QueryIdentifiabilityStatus], ...] = field(
        init=False
    )
    action_contributions: tuple[ActionContributionV1, ...] = field(init=False)
    requires_multiple_actions: bool = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "latent_coordinates_id",
            "whitening_id",
            "joint_nuisance_design_id",
            "joint_query_id",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name=name),
            )
        if isinstance(self.action_blocks, (str, bytes)):
            raise TypeError("action_blocks must be a sequence of action blocks")
        blocks = tuple(self.action_blocks)
        if len(blocks) < 2:
            raise ValueError(
                "multi-action identifiability requires at least two actions"
            )
        if any(not isinstance(block, ActionIdentifiabilityBlockV1) for block in blocks):
            raise TypeError(
                "action_blocks must contain ActionIdentifiabilityBlockV1 values"
            )
        action_ids = tuple(block.action_id for block in blocks)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action_blocks must have unique action_id values")
        if action_ids != tuple(sorted(action_ids)):
            raise ValueError("action_blocks must be sorted by action_id")

        latent_dimension = blocks[0].latent_dimension
        if any(block.latent_dimension != latent_dimension for block in blocks):
            raise ValueError("all action blocks must share the latent dimension")
        physical = np.vstack([block.whitened_physical_design for block in blocks])
        query = np.vstack([block.transported_query_map for block in blocks])
        nuisance = _matrix(
            self.joint_whitened_nuisance_design,
            name="joint_whitened_nuisance_design",
        )
        if nuisance.shape[0] != physical.shape[0]:
            raise ValueError(
                "joint_whitened_nuisance_design must share the stacked "
                "observation row count"
            )

        physical_id = cast(
            str,
            content_id(
                {
                    "kind": "multi-action-physical-response-v1",
                    "latent_coordinates_id": self.latent_coordinates_id,
                    "block_ids": [block.artifact_id for block in blocks],
                }
            ),
        )
        observation_id = cast(
            str,
            content_id(
                {
                    "kind": "multi-action-observation-mapping-v1",
                    "whitening_id": self.whitening_id,
                    "block_ids": [block.artifact_id for block in blocks],
                }
            ),
        )
        joint = QueryIdentifiabilityCertificateV2(
            physical_response_id=physical_id,
            observation_mapping_id=observation_id,
            nuisance_design_id=self.joint_nuisance_design_id,
            query_id=self.joint_query_id,
            whitened_physical_design=physical,
            whitened_nuisance_design=nuisance,
            query_map=query,
            relative_rank_tolerance=self.relative_rank_tolerance,
            absolute_rank_tolerance=self.absolute_rank_tolerance,
            identifiability_tolerance=self.identifiability_tolerance,
            metadata={
                "action_ids": list(action_ids),
                "semantics": MULTI_ACTION_QUERY_IDENTIFIABILITY_SEMANTICS,
            },
        )

        row_offsets: list[tuple[int, int]] = []
        offset = 0
        for block in blocks:
            row_offsets.append((offset, offset + block.observation_dimension))
            offset += block.observation_dimension

        single_statuses: list[tuple[str, QueryIdentifiabilityStatus]] = []
        contributions: list[ActionContributionV1] = []
        for index, block in enumerate(blocks):
            start, stop = row_offsets[index]
            block_nuisance = nuisance[start:stop, :]
            single = QueryIdentifiabilityCertificateV2(
                physical_response_id=block.physical_response_id,
                observation_mapping_id=block.observation_mapping_id,
                nuisance_design_id=self.joint_nuisance_design_id,
                query_id=block.query_transport_id,
                whitened_physical_design=block.whitened_physical_design,
                whitened_nuisance_design=block_nuisance,
                query_map=block.transported_query_map,
                relative_rank_tolerance=self.relative_rank_tolerance,
                absolute_rank_tolerance=self.absolute_rank_tolerance,
                identifiability_tolerance=self.identifiability_tolerance,
                metadata={"action_id": block.action_id, "scope": "single-action"},
            )
            single_statuses.append((block.action_id, single.status))

            keep = np.ones(physical.shape[0], dtype=bool)
            keep[start:stop] = False
            without = QueryIdentifiabilityCertificateV2(
                physical_response_id=physical_id,
                observation_mapping_id=observation_id,
                nuisance_design_id=self.joint_nuisance_design_id,
                query_id=self.joint_query_id,
                whitened_physical_design=physical[keep, :],
                whitened_nuisance_design=nuisance[keep, :],
                query_map=query,
                relative_rank_tolerance=self.relative_rank_tolerance,
                absolute_rank_tolerance=self.absolute_rank_tolerance,
                identifiability_tolerance=self.identifiability_tolerance,
                metadata={
                    "removed_action_id": block.action_id,
                    "scope": "leave-one-action-out",
                },
            )
            contributions.append(
                ActionContributionV1(
                    action_id=block.action_id,
                    without_action_status=without.status,
                    without_action_energy_fraction=(
                        without.identifiable_query_energy_fraction
                    ),
                    without_action_normalized_residual=(
                        without.normalized_factorization_residual
                    ),
                    without_action_physical_rank=without.physical_rank,
                    energy_fraction_loss=max(
                        0.0,
                        joint.identifiable_query_energy_fraction
                        - without.identifiable_query_energy_fraction,
                    ),
                )
            )

        requires_multiple = (
            joint.status is QueryIdentifiabilityStatus.IDENTIFIABLE
            and all(
                status is not QueryIdentifiabilityStatus.IDENTIFIABLE
                for _, status in single_statuses
            )
        )

        object.__setattr__(self, "action_blocks", blocks)
        object.__setattr__(
            self,
            "relative_rank_tolerance",
            joint.relative_rank_tolerance,
        )
        object.__setattr__(
            self,
            "absolute_rank_tolerance",
            joint.absolute_rank_tolerance,
        )
        object.__setattr__(
            self,
            "identifiability_tolerance",
            joint.identifiability_tolerance,
        )
        object.__setattr__(
            self,
            "joint_whitened_nuisance_design",
            _immutable(nuisance),
        )
        object.__setattr__(self, "stacked_physical_design", _immutable(physical))
        object.__setattr__(self, "stacked_query_map", _immutable(query))
        object.__setattr__(self, "joint_certificate", joint)
        object.__setattr__(self, "single_action_statuses", tuple(single_statuses))
        object.__setattr__(self, "action_contributions", tuple(contributions))
        object.__setattr__(self, "requires_multiple_actions", requires_multiple)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="multi-action identifiability metadata",
            ),
        )
        expected = cast(str, content_id(self.descriptor()))
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError(
                    "multi-action certificate artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected)

    @property
    def status(self) -> QueryIdentifiabilityStatus:
        return self.joint_certificate.status

    @property
    def identifiable(self) -> bool:
        return self.joint_certificate.identifiable

    def arrays(self) -> Mapping[str, np.ndarray]:
        return {
            "stacked_physical_design": self.stacked_physical_design,
            "stacked_query_map": self.stacked_query_map,
            "joint_whitened_nuisance_design": (self.joint_whitened_nuisance_design),
        }

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": MULTI_ACTION_QUERY_IDENTIFIABILITY_SCHEMA,
            "schema_version": MULTI_ACTION_QUERY_IDENTIFIABILITY_VERSION,
            "artifact_kind": "MultiActionQueryIdentifiabilityCertificateV1",
            "semantics": MULTI_ACTION_QUERY_IDENTIFIABILITY_SEMANTICS,
            "latent_coordinates_id": self.latent_coordinates_id,
            "whitening_id": self.whitening_id,
            "joint_nuisance_design_id": self.joint_nuisance_design_id,
            "joint_query_id": self.joint_query_id,
            "action_blocks": [block.to_record() for block in self.action_blocks],
            "joint_whitened_nuisance_design": _array_record(
                self.joint_whitened_nuisance_design
            ),
            "stacked_physical_design": _array_record(self.stacked_physical_design),
            "stacked_query_map": _array_record(self.stacked_query_map),
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "identifiability_tolerance": self.identifiability_tolerance,
            "joint_certificate_id": self.joint_certificate.artifact_id,
            "status": self.status.value,
            "requires_multiple_actions": self.requires_multiple_actions,
            "single_action_statuses": [
                {"action_id": action_id, "status": status.value}
                for action_id, status in self.single_action_statuses
            ],
            "action_contributions": [
                contribution.to_record() for contribution in self.action_contributions
            ],
            "metadata": plain_json(self.metadata),
            "claim_boundary": MULTI_ACTION_QUERY_IDENTIFIABILITY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    def summary(self) -> dict[str, object]:
        return {
            "schema": MULTI_ACTION_QUERY_IDENTIFIABILITY_SCHEMA,
            "schema_version": MULTI_ACTION_QUERY_IDENTIFIABILITY_VERSION,
            "artifact_id": self.artifact_id,
            "action_count": len(self.action_blocks),
            "status": self.status.value,
            "requires_multiple_actions": self.requires_multiple_actions,
            "physical_rank": self.joint_certificate.physical_rank,
            "physical_nullity": self.joint_certificate.physical_nullity,
            "normalized_factorization_residual": (
                self.joint_certificate.normalized_factorization_residual
            ),
            "identifiable_query_energy_fraction": (
                self.joint_certificate.identifiable_query_energy_fraction
            ),
            "action_contributions": [
                contribution.to_record() for contribution in self.action_contributions
            ],
            "claim_boundary": MULTI_ACTION_QUERY_IDENTIFIABILITY_CLAIM_BOUNDARY,
        }
