"""Content-addressed provider/scoring source separation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._canonical_contracts import frozen_finite_json_mapping, plain_json
from .._portable_contracts import content_id
from ._common import (
    CLAIM_BOUNDARY,
    REGISTERED_SCHEMA_VERSION,
    REGISTERED_SOURCE_PROVENANCE_SCHEMA,
    _canonical_string_tuple,
    _digest_tuple,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class ResidualHistorySourceProvenanceV1:
    """Disjoint source reconstruction identity for one opened source unit."""

    source_inventory_id: str
    provider_reconstruction_id: str
    scoring_reconstruction_id: str
    provider_implementation_revision: str
    scoring_implementation_revision: str
    provider_configuration_id: str
    scoring_configuration_id: str
    provider_camera_family_ids: tuple[str, ...]
    scoring_camera_family_ids: tuple[str, ...]
    provider_input_artifact_ids: tuple[str, ...]
    scoring_input_artifact_ids: tuple[str, ...]
    provider_parent_reconstruction_ids: tuple[str, ...] = ()
    scoring_parent_reconstruction_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance_id: str | None = None

    def __post_init__(self) -> None:
        source_inventory_id = _sha256(
            self.source_inventory_id,
            name="source_inventory_id",
        )
        provider_id = _sha256(
            self.provider_reconstruction_id,
            name="provider_reconstruction_id",
        )
        scoring_id = _sha256(
            self.scoring_reconstruction_id,
            name="scoring_reconstruction_id",
        )
        if provider_id == scoring_id:
            raise ValueError("provider and scoring reconstructions must differ")
        provider_revision = _sha256(
            self.provider_implementation_revision,
            name="provider_implementation_revision",
            length=40,
        )
        scoring_revision = _sha256(
            self.scoring_implementation_revision,
            name="scoring_implementation_revision",
            length=40,
        )
        provider_config = _sha256(
            self.provider_configuration_id,
            name="provider_configuration_id",
        )
        scoring_config = _sha256(
            self.scoring_configuration_id,
            name="scoring_configuration_id",
        )
        provider_families = _canonical_string_tuple(
            self.provider_camera_family_ids,
            name="provider_camera_family_ids",
        )
        scoring_families = _canonical_string_tuple(
            self.scoring_camera_family_ids,
            name="scoring_camera_family_ids",
        )
        if set(provider_families) & set(scoring_families):
            raise ValueError("provider and scoring camera families must be disjoint")
        provider_inputs = _digest_tuple(
            self.provider_input_artifact_ids,
            name="provider_input_artifact_ids",
        )
        scoring_inputs = _digest_tuple(
            self.scoring_input_artifact_ids,
            name="scoring_input_artifact_ids",
        )
        if set(provider_inputs) & set(scoring_inputs):
            raise ValueError("provider and scoring input artifacts must be disjoint")
        provider_parents = _digest_tuple(
            self.provider_parent_reconstruction_ids,
            name="provider_parent_reconstruction_ids",
            allow_empty=True,
        )
        scoring_parents = _digest_tuple(
            self.scoring_parent_reconstruction_ids,
            name="scoring_parent_reconstruction_ids",
            allow_empty=True,
        )
        if provider_id in provider_parents or scoring_id in scoring_parents:
            raise ValueError("a reconstruction cannot be its own parent")
        provider_lineage = {provider_id, *provider_parents}
        scoring_lineage = {scoring_id, *scoring_parents}
        if provider_lineage & scoring_lineage:
            raise ValueError("provider and scoring reconstruction lineages overlap")
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        assignments = {
            "source_inventory_id": source_inventory_id,
            "provider_reconstruction_id": provider_id,
            "scoring_reconstruction_id": scoring_id,
            "provider_implementation_revision": provider_revision,
            "scoring_implementation_revision": scoring_revision,
            "provider_configuration_id": provider_config,
            "scoring_configuration_id": scoring_config,
            "provider_camera_family_ids": provider_families,
            "scoring_camera_family_ids": scoring_families,
            "provider_input_artifact_ids": provider_inputs,
            "scoring_input_artifact_ids": scoring_inputs,
            "provider_parent_reconstruction_ids": provider_parents,
            "scoring_parent_reconstruction_ids": scoring_parents,
            "metadata": metadata,
        }
        for name, value in assignments.items():
            object.__setattr__(self, name, value)
        expected = content_id(self.descriptor())
        if self.provenance_id is None:
            object.__setattr__(self, "provenance_id", expected)
        elif _sha256(self.provenance_id, name="provenance_id") != expected:
            raise ValueError("provenance_id does not match source provenance")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": REGISTERED_SOURCE_PROVENANCE_SCHEMA,
            "schema_version": REGISTERED_SCHEMA_VERSION,
            "source_inventory_id": self.source_inventory_id,
            "provider_reconstruction_id": self.provider_reconstruction_id,
            "scoring_reconstruction_id": self.scoring_reconstruction_id,
            "provider_implementation_revision": (self.provider_implementation_revision),
            "scoring_implementation_revision": (self.scoring_implementation_revision),
            "provider_configuration_id": self.provider_configuration_id,
            "scoring_configuration_id": self.scoring_configuration_id,
            "provider_camera_family_ids": list(self.provider_camera_family_ids),
            "scoring_camera_family_ids": list(self.scoring_camera_family_ids),
            "provider_input_artifact_ids": list(self.provider_input_artifact_ids),
            "scoring_input_artifact_ids": list(self.scoring_input_artifact_ids),
            "provider_parent_reconstruction_ids": list(
                self.provider_parent_reconstruction_ids
            ),
            "scoring_parent_reconstruction_ids": list(
                self.scoring_parent_reconstruction_ids
            ),
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }
