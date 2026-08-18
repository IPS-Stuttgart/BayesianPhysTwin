from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin import guarded_belief_selection_v2, provider_runtime_identity_v1
from bayesian_phystwin.causal4d_guarded_belief_provider_v1 import (
    CAUSAL4D_GUARDED_BELIEF_PROVIDER_API_VERSION,
    CAUSAL4D_GUARDED_BELIEF_PROVIDER_ARTIFACT_SCHEMA_VERSIONS,
    CAUSAL4D_GUARDED_BELIEF_PROVIDER_CAPABILITIES,
    CAUSAL4D_GUARDED_BELIEF_PROVIDER_CLAIM_BOUNDARY,
    CandidateBeliefConstructionReceiptV1,
    GuardedBeliefSelectionReceiptV2,
    Prob4DRuntimeIdentityV1,
    bind_guarded_belief_selection_receipt,
    build_candidate_belief_construction_receipt,
    causal4d_guarded_belief_provider_v1_manifest,
)


def test_facade_reexports_exact_contract_objects() -> None:
    assert (
        Prob4DRuntimeIdentityV1 is provider_runtime_identity_v1.Prob4DRuntimeIdentityV1
    )
    assert (
        CandidateBeliefConstructionReceiptV1
        is guarded_belief_selection_v2.CandidateBeliefConstructionReceiptV1
    )
    assert (
        GuardedBeliefSelectionReceiptV2
        is guarded_belief_selection_v2.GuardedBeliefSelectionReceiptV2
    )
    assert (
        build_candidate_belief_construction_receipt
        is guarded_belief_selection_v2.build_candidate_belief_construction_receipt
    )
    assert (
        bind_guarded_belief_selection_receipt
        is guarded_belief_selection_v2.bind_guarded_belief_selection_receipt
    )


def test_manifest_binds_versioned_contracts_without_empirical_promotion() -> None:
    manifest = causal4d_guarded_belief_provider_v1_manifest(provider_revision="a" * 40)

    assert manifest["provider_name"] == "bayesian-phystwin"
    assert manifest["provider_revision"] == "a" * 40
    assert manifest["schema_version"] == (CAUSAL4D_GUARDED_BELIEF_PROVIDER_API_VERSION)
    assert manifest["capabilities"] == list(
        CAUSAL4D_GUARDED_BELIEF_PROVIDER_CAPABILITIES
    )
    assert manifest["artifact_schema_versions"] == dict(
        CAUSAL4D_GUARDED_BELIEF_PROVIDER_ARTIFACT_SCHEMA_VERSIONS
    )
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["provider_api"] == (
        "bayesian_phystwin.causal4d_guarded_belief_provider_v1"
    )
    assert metadata["claim_boundary"] == (
        CAUSAL4D_GUARDED_BELIEF_PROVIDER_CLAIM_BOUNDARY
    )
    assert "does not establish provider competence" in (
        CAUSAL4D_GUARDED_BELIEF_PROVIDER_CLAIM_BOUNDARY
    )


@pytest.mark.parametrize("revision", ("", 7, False))
def test_manifest_rejects_malformed_explicit_revision(revision: object) -> None:
    with pytest.raises(ValueError, match="nonempty string"):
        causal4d_guarded_belief_provider_v1_manifest(  # type: ignore[arg-type]
            provider_revision=revision
        )


def test_public_lifecycle_registers_the_versioned_facade() -> None:
    root = Path(__file__).resolve().parents[1]
    lifecycle = json.loads(
        (root / "api/public-module-lifecycle-v1.json").read_text(encoding="utf-8")
    )
    assert (
        "bayesian_phystwin.causal4d_guarded_belief_provider_v1"
        in lifecycle["stable_modules"]
    )
