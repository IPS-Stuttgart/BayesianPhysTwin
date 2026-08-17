"""Deliberately small, versioned integration API for Bayesian-PhysTwin.

The historical package root remains available through a lazy compatibility
shim. New integrations should prefer this namespace so the portable contract
surface can evolve independently from the research-oriented root API.
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
from ..physical_query_v1 import (
    PHYSICAL_QUERY_CLAIM_BOUNDARY,
    PHYSICAL_QUERY_SCHEMA,
    PHYSICAL_QUERY_VERSION,
    PhysicalQueryBootstrapV1,
    PhysicalQueryDecisionMarginsV1,
    PhysicalQueryV1,
    load_physical_query,
    write_physical_query,
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
    "PHYSICAL_QUERY_CLAIM_BOUNDARY",
    "PHYSICAL_QUERY_SCHEMA",
    "PHYSICAL_QUERY_VERSION",
    "PhysicalQueryBootstrapV1",
    "PhysicalQueryDecisionMarginsV1",
    "PhysicalQueryV1",
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
    "load_physical_query",
    "load_run_manifest",
    "save_observation_belief",
    "write_claim_bundle",
    "write_evidence_decision",
    "write_physical_query",
    "write_run_manifest",
]
