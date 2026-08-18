"""Versioned guarded complete-belief contracts for Causal4D.

This additive provider facade exposes the exact Prob4D runtime identity and the
BayesianPhysTwin candidate-construction and complete-belief selection receipts
without making their implementation modules a downstream integration boundary.
"""

from __future__ import annotations

import os
from typing import Final

from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .guarded_belief_selection_v2 import (
    CANDIDATE_CONSTRUCTION_SCHEMA,
    CANDIDATE_CONSTRUCTION_SCHEMA_VERSION,
    GUARDED_SELECTION_SCHEMA,
    GUARDED_SELECTION_SCHEMA_VERSION,
    CandidateBeliefConstructionReceiptV1,
    GuardedBeliefSelectionReceiptV2,
    bind_guarded_belief_selection_receipt,
    build_candidate_belief_construction_receipt,
)
from .provider_runtime_identity_v1 import (
    PROB4D_RUNTIME_IDENTITY_SCHEMA,
    PROB4D_RUNTIME_IDENTITY_VERSION,
    Prob4DRuntimeIdentityV1,
)

CAUSAL4D_GUARDED_BELIEF_PROVIDER_API_VERSION: Final = 1
CAUSAL4D_GUARDED_BELIEF_PROVIDER_PACKAGE_VERSION: Final = "0.4.0"
CAUSAL4D_GUARDED_BELIEF_PROVIDER_CAPABILITIES: Final = (
    "exact_prob4d_runtime_revision_identity",
    "complete_candidate_belief_construction_identity",
    "complete_belief_guard_identity",
    "exact_selected_belief_identity",
    "exact_physical_fallback_identity",
    "content_addressed_handoff_receipts",
)
CAUSAL4D_GUARDED_BELIEF_PROVIDER_ARTIFACT_SCHEMA_VERSIONS: Final = {
    "Prob4DRuntimeIdentity": PROB4D_RUNTIME_IDENTITY_VERSION,
    "CandidateBeliefConstructionReceipt": CANDIDATE_CONSTRUCTION_SCHEMA_VERSION,
    "GuardedBeliefSelectionReceipt": GUARDED_SELECTION_SCHEMA_VERSION,
}
CAUSAL4D_GUARDED_BELIEF_PROVIDER_INFERENCE_ROLE: Final = (
    "guarded complete-belief provenance and exact fallback handoff"
)
CAUSAL4D_GUARDED_BELIEF_PROVIDER_COMPATIBILITY: Final = (
    "additive provider; existing Causal4D provider and belief-provider surfaces "
    "remain unchanged"
)
CAUSAL4D_GUARDED_BELIEF_PROVIDER_CLAIM_BOUNDARY: Final = (
    "The provider establishes exact runtime, candidate-construction, guard, and "
    "selected-belief identities. It does not establish provider competence, "
    "empirical calibration, physical-query benefit, intervention benefit, "
    "deployment safety, or state of the art."
)


def causal4d_guarded_belief_provider_v1_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the guarded complete-belief handoff capability descriptor."""

    if provider_revision is not None and (
        type(provider_revision) is not str or not provider_revision
    ):
        raise ValueError("provider_revision must be a nonempty string")
    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or installed_distribution_revision("bayesian-phystwin")
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": installed_distribution_version(
            "bayesian-phystwin",
            fallback=CAUSAL4D_GUARDED_BELIEF_PROVIDER_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_GUARDED_BELIEF_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_GUARDED_BELIEF_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(
            CAUSAL4D_GUARDED_BELIEF_PROVIDER_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": (
                "bayesian_phystwin.causal4d_guarded_belief_provider_v1"
            ),
            "provider_api_version": (
                CAUSAL4D_GUARDED_BELIEF_PROVIDER_API_VERSION
            ),
            "inference_role": CAUSAL4D_GUARDED_BELIEF_PROVIDER_INFERENCE_ROLE,
            "compatibility": CAUSAL4D_GUARDED_BELIEF_PROVIDER_COMPATIBILITY,
            "runtime_identity_schema": PROB4D_RUNTIME_IDENTITY_SCHEMA,
            "candidate_construction_schema": CANDIDATE_CONSTRUCTION_SCHEMA,
            "guarded_selection_schema": GUARDED_SELECTION_SCHEMA,
            "claim_boundary": CAUSAL4D_GUARDED_BELIEF_PROVIDER_CLAIM_BOUNDARY,
        },
    }


__all__ = [
    "CAUSAL4D_GUARDED_BELIEF_PROVIDER_API_VERSION",
    "CAUSAL4D_GUARDED_BELIEF_PROVIDER_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_GUARDED_BELIEF_PROVIDER_CAPABILITIES",
    "CAUSAL4D_GUARDED_BELIEF_PROVIDER_CLAIM_BOUNDARY",
    "CAUSAL4D_GUARDED_BELIEF_PROVIDER_COMPATIBILITY",
    "CAUSAL4D_GUARDED_BELIEF_PROVIDER_INFERENCE_ROLE",
    "CAUSAL4D_GUARDED_BELIEF_PROVIDER_PACKAGE_VERSION",
    "CANDIDATE_CONSTRUCTION_SCHEMA",
    "CANDIDATE_CONSTRUCTION_SCHEMA_VERSION",
    "GUARDED_SELECTION_SCHEMA",
    "GUARDED_SELECTION_SCHEMA_VERSION",
    "PROB4D_RUNTIME_IDENTITY_SCHEMA",
    "PROB4D_RUNTIME_IDENTITY_VERSION",
    "CandidateBeliefConstructionReceiptV1",
    "GuardedBeliefSelectionReceiptV2",
    "Prob4DRuntimeIdentityV1",
    "bind_guarded_belief_selection_receipt",
    "build_candidate_belief_construction_receipt",
    "causal4d_guarded_belief_provider_v1_manifest",
]
