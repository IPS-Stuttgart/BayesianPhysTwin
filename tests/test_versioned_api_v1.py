from __future__ import annotations

import bayesian_phystwin.v1 as v1


EXPECTED_PUBLIC_API = {
    "ArtifactDigest",
    "CLAIM_BUNDLE_SCHEMA",
    "CLAIM_BUNDLE_SCHEMA_VERSION",
    "ClaimBundleArtifactV1",
    "ClaimBundleV1",
    "DecisionMetricV1",
    "DecisionStatus",
    "EVIDENCE_DECISION_SCHEMA",
    "EVIDENCE_DECISION_SCHEMA_VERSION",
    "EvidenceDecisionV1",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "ObservationBeliefV1",
    "RUN_MANIFEST_SCHEMA",
    "RUN_MANIFEST_V2_VERSION",
    "RUN_MANIFEST_VERSION",
    "RepositoryRole",
    "RepositoryState",
    "RunClassification",
    "RunManifestV1",
    "RunManifestV2",
    "build_evidence_decision",
    "load_claim_bundle",
    "load_evidence_decision",
    "load_observation_belief",
    "load_run_manifest",
    "save_observation_belief",
    "write_claim_bundle",
    "write_evidence_decision",
    "write_run_manifest",
}


def test_versioned_api_is_deliberately_small_and_frozen() -> None:
    assert set(v1.__all__) == EXPECTED_PUBLIC_API
    assert len(v1.__all__) == len(set(v1.__all__))
    for name in EXPECTED_PUBLIC_API:
        assert getattr(v1, name) is not None
