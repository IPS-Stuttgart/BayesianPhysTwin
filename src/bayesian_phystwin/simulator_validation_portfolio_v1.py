"""Query-local deployment portfolios over a frozen simulator validation atlas.

The atlas artifact is immutable and binds the evidence-generation module bytes.
This separate layer composes its exact-query selector without changing those
bytes or pooling heterogeneous metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

from ._canonical_contracts import genuine_boolean
from ._portable_contracts import content_id, sha256_digest
from .complete_belief_selection import ArtifactBelief
from .simulator_validation_atlas_v1 import (
    SIMULATOR_VALIDATION_ATLAS_VERSION,
    SimulatorValidationAtlasV1,
    select_prospectively_validated_candidate,
)

SIMULATOR_VALIDATION_PORTFOLIO_SELECTION_SCHEMA = (
    "bayesian_phystwin.simulator_validation_portfolio_selection"
)

BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


@dataclass(frozen=True, slots=True)
class SimulatorValidationPortfolioItemV1(Generic[BeliefT]):
    """One query-local baseline/candidate choice in a finite portfolio."""

    query_id: str
    baseline: BeliefT
    candidate: BeliefT
    inference_admissible: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_id",
            sha256_digest(self.query_id, name="query_id"),
        )
        sha256_digest(self.baseline.artifact_id, name="baseline artifact_id")
        sha256_digest(self.candidate.artifact_id, name="candidate artifact_id")
        object.__setattr__(
            self,
            "inference_admissible",
            genuine_boolean(
                self.inference_admissible,
                name="inference_admissible",
            ),
        )


def select_prospectively_validated_portfolio(
    atlas: SimulatorValidationAtlasV1,
    items: Sequence[SimulatorValidationPortfolioItemV1[BeliefT]],
) -> tuple[Mapping[str, BeliefT], dict[str, object]]:
    """Apply exact-query admission independently across a finite portfolio."""

    if not isinstance(atlas, SimulatorValidationAtlasV1):
        raise TypeError("atlas must be a SimulatorValidationAtlasV1")
    checked_items = tuple(items)
    if not checked_items:
        raise ValueError("portfolio requires at least one query-local item")
    if any(
        not isinstance(item, SimulatorValidationPortfolioItemV1)
        for item in checked_items
    ):
        raise TypeError("portfolio items must be SimulatorValidationPortfolioItemV1")
    ordered = tuple(sorted(checked_items, key=lambda item: item.query_id))
    query_ids = tuple(item.query_id for item in ordered)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("portfolio query scopes must be unique")

    selected_by_query: dict[str, BeliefT] = {}
    selection_receipts: list[dict[str, object]] = []
    selected_candidate_count = 0
    for item in ordered:
        selected, receipt = select_prospectively_validated_candidate(
            item.baseline,
            item.candidate,
            atlas,
            query_id=item.query_id,
            inference_admissible=item.inference_admissible,
        )
        selected_candidate = receipt["selected_candidate"]
        if type(selected_candidate) is not bool:
            raise AssertionError("query-local selector emitted an invalid decision")
        if selected_candidate:
            if selected is not item.candidate:
                raise AssertionError("candidate selection did not preserve object identity")
            selected_candidate_count += 1
        elif selected is not item.baseline:
            raise AssertionError("fallback did not preserve baseline object identity")
        selected_by_query[item.query_id] = selected
        selection_receipts.append(receipt)

    descriptor: dict[str, object] = {
        "schema": SIMULATOR_VALIDATION_PORTFOLIO_SELECTION_SCHEMA,
        "schema_version": SIMULATOR_VALIDATION_ATLAS_VERSION,
        "atlas_id": atlas.artifact_id,
        "query_count": len(ordered),
        "selected_candidate_count": selected_candidate_count,
        "exact_fallback_count": len(ordered) - selected_candidate_count,
        "query_local_selection": True,
        "heterogeneous_metrics_pooled": False,
        "selections": selection_receipts,
    }
    return MappingProxyType(selected_by_query), {
        **descriptor,
        "artifact_id": content_id(descriptor),
    }
