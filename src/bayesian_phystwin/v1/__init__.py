"""Deliberately small, versioned integration API for Bayesian-PhysTwin.

The legacy package root remains unchanged. New integrations should prefer this
namespace so the portable contract surface can evolve independently from the
research-oriented root API.
"""

from ..claim_bundle_v1 import (
    CLAIM_BUNDLE_SCHEMA,
    CLAIM_BUNDLE_SCHEMA_VERSION,
    ClaimBundleArtifactV1,
    ClaimBundleV1,
    load_claim_bundle,
    write_claim_bundle,
)
from ..evidence_decision_v1 import (
    EVIDENCE_DECISION_SCHEMA,
    EVIDENCE_DECISION_SCHEMA_VERSION,
    DecisionMetricV1,
    DecisionStatus,
    EvidenceDecisionV1,
    build_evidence_decision,
    load_evidence_decision,
    write_evidence_decision,
)
from ..observation_belief import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefV1,
    load_observation_belief,
    save_observation_belief,
)
from ..prob4d_api_v2 import (
    PROB4D_REQUIRED_API_VERSION,
    PROB4D_REQUIRED_FACTOR_API_VERSION,
    PROB4D_REQUIRED_PROJECT_ID,
    PROB4D_REQUIRED_PROVIDER_API_VERSION,
    Prob4DApiV2Compatibility,
    inspect_prob4d_api_v2,
    load_claim_bearing_tree_sparse_prob4d,
)
from ..repository_provenance import RepositoryRole, RepositoryState
from ..run_manifest import (
    RUN_MANIFEST_SCHEMA,
    RUN_MANIFEST_VERSION,
    ArtifactDigest,
    RunClassification,
    RunManifestV1,
)
from ..run_manifest_v2 import (
    RUN_MANIFEST_V2_VERSION,
    RunManifestV2,
    load_run_manifest,
    write_run_manifest,
)

__all__ = [
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
    "PROB4D_REQUIRED_API_VERSION",
    "PROB4D_REQUIRED_FACTOR_API_VERSION",
    "PROB4D_REQUIRED_PROJECT_ID",
    "PROB4D_REQUIRED_PROVIDER_API_VERSION",
    "Prob4DApiV2Compatibility",
    "RUN_MANIFEST_SCHEMA",
    "RUN_MANIFEST_V2_VERSION",
    "RUN_MANIFEST_VERSION",
    "RepositoryRole",
    "RepositoryState",
    "RunClassification",
    "RunManifestV1",
    "RunManifestV2",
    "build_evidence_decision",
    "inspect_prob4d_api_v2",
    "load_claim_bearing_tree_sparse_prob4d",
    "load_claim_bundle",
    "load_evidence_decision",
    "load_observation_belief",
    "load_run_manifest",
    "save_observation_belief",
    "write_claim_bundle",
    "write_evidence_decision",
    "write_run_manifest",
]
