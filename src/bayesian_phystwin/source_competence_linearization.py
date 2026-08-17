"""Verified physical-linearization rebinding after source-competence refinement."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._canonical_contracts import plain_json
from .physical_linearization import (
    PhysicalLinearizationV1,
    validate_observation_linearization_alignment,
)
from .source_competence_reliability import (
    SOURCE_COMPETENCE_CLAIM_BOUNDARY,
    SourceCompetenceReliabilityUpdateV1,
)

SOURCE_COMPETENCE_LINEARIZATION_BINDING = (
    "rebind-identical-rows-and-jacobians-to-refined-observation-v1"
)


def rebind_physical_linearization_to_source_competence(
    update: SourceCompetenceReliabilityUpdateV1,
    linearization: PhysicalLinearizationV1,
) -> PhysicalLinearizationV1:
    """Bind unchanged physical Jacobians to the refined observation artifact.

    The original linearization must identify and align with the exact source
    observation. The returned artifact changes only its observation identity and
    metadata; every row identity, Jacobian, physical response, baseline belief,
    action prefix, and simulator revision remains value-identical.
    """

    if not isinstance(update, SourceCompetenceReliabilityUpdateV1):
        raise TypeError("update must be a SourceCompetenceReliabilityUpdateV1")
    if not isinstance(linearization, PhysicalLinearizationV1):
        raise TypeError("linearization must be a PhysicalLinearizationV1")

    source = update.source_observation
    refined = update.refined_observation
    validate_observation_linearization_alignment(source, linearization)

    additions = {
        "source_competence_linearization_binding": (
            SOURCE_COMPETENCE_LINEARIZATION_BINDING
        ),
        "source_competence_update_id": update.update_id,
        "source_competence_evidence_id": update.evidence.artifact_id,
        "source_competence_markov_config_id": update.config.config_id,
        "source_competence_source_observation_id": source.artifact_id,
        "source_competence_refined_observation_id": refined.artifact_id,
        "source_competence_rows_changed": False,
        "source_competence_jacobians_changed": False,
        "source_competence_physical_response_changed": False,
        "source_competence_claim_boundary": SOURCE_COMPETENCE_CLAIM_BOUNDARY,
    }
    metadata = dict(plain_json(linearization.metadata))
    for name, value in additions.items():
        if name in metadata and metadata[name] != value:
            raise ValueError(f"linearization metadata conflicts with {name}")
        metadata[name] = value

    rebound = replace(
        linearization,
        observation_artifact_id=refined.artifact_id,
        metadata=metadata,
    )
    validate_observation_linearization_alignment(refined, rebound)

    unchanged_scalars = (
        "baseline_belief_id",
        "action_prefix_id",
        "simulator_revision",
    )
    for name in unchanged_scalars:
        if getattr(rebound, name) != getattr(linearization, name):
            raise AssertionError(f"source competence changed linearization {name}")
    unchanged_arrays = (
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "state_jacobian",
        "query_state_jacobian",
        "physical_response_m",
    )
    for name in unchanged_arrays:
        if not np.array_equal(getattr(rebound, name), getattr(linearization, name)):
            raise AssertionError(f"source competence changed linearization {name}")
    return rebound


__all__ = [
    "SOURCE_COMPETENCE_LINEARIZATION_BINDING",
    "rebind_physical_linearization_to_source_competence",
]
