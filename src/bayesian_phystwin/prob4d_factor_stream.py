"""Fail-closed recursive consumption of Prob4D factor streams.

The public facade revalidates portable stream members and row identities, binds
explicit recursive nuisance handling, records posterior covariance semantics,
and routes complete beliefs with exact fallback. No Prob4D package import is
required.
"""

from ._prob4d_recursive_policy import RecursiveNuisancePolicyV1
from ._prob4d_recursive_records import (
    ClaimBearingProb4DStreamRunV1,
    ClaimBearingProb4DStreamStepV1,
)
from ._prob4d_recursive_update import (
    apply_claim_bearing_prob4d_stream_update,
    load_claim_bearing_prob4d_stream_run,
    start_claim_bearing_prob4d_stream_run,
    write_claim_bearing_prob4d_stream_run,
)
from ._prob4d_stream_binding import (
    Prob4DStreamObservationBindingV1,
    bind_prob4d_stream_observation,
    load_prob4d_observation_factor_stream,
    prob4d_observation_identity_summary,
    write_prob4d_observation_factor_stream,
)
from ._prob4d_stream_common import (
    CLAIM_BEARING_PROB4D_STREAM_RUN_SCHEMA,
    CLAIM_BEARING_PROB4D_STREAM_RUN_VERSION,
    CLAIM_BEARING_PROB4D_STREAM_STEP_SCHEMA,
    CLAIM_BEARING_PROB4D_STREAM_STEP_VERSION,
    PROB4D_CANONICAL_SOURCE_REPOSITORY,
    PROB4D_LEGACY_SOURCE_REPOSITORY,
    PROB4D_OBSERVATION_FACTOR_STREAM_SCHEMA,
    PROB4D_OBSERVATION_FACTOR_STREAM_VERSION,
    PROB4D_PROJECT_ID,
    PROB4D_SOURCE_REPOSITORY_ALIASES,
    PROB4D_STREAM_OBSERVATION_BINDING_SCHEMA,
    PROB4D_STREAM_OBSERVATION_BINDING_VERSION,
    RECURSIVE_NUISANCE_MODES,
    RECURSIVE_NUISANCE_POLICY_SCHEMA,
    RECURSIVE_NUISANCE_POLICY_VERSION,
    ArtifactBelief,
    BeliefT,
    RecursiveNuisanceMode,
)
from ._prob4d_stream_manifest import (
    Prob4DObservationFactorStreamUpdateV1,
    Prob4DObservationFactorStreamV1,
)


__all__ = [
    "ArtifactBelief",
    "BeliefT",
    "CLAIM_BEARING_PROB4D_STREAM_RUN_SCHEMA",
    "CLAIM_BEARING_PROB4D_STREAM_RUN_VERSION",
    "CLAIM_BEARING_PROB4D_STREAM_STEP_SCHEMA",
    "CLAIM_BEARING_PROB4D_STREAM_STEP_VERSION",
    "PROB4D_CANONICAL_SOURCE_REPOSITORY",
    "PROB4D_LEGACY_SOURCE_REPOSITORY",
    "PROB4D_OBSERVATION_FACTOR_STREAM_SCHEMA",
    "PROB4D_OBSERVATION_FACTOR_STREAM_VERSION",
    "PROB4D_PROJECT_ID",
    "PROB4D_SOURCE_REPOSITORY_ALIASES",
    "PROB4D_STREAM_OBSERVATION_BINDING_SCHEMA",
    "PROB4D_STREAM_OBSERVATION_BINDING_VERSION",
    "RECURSIVE_NUISANCE_MODES",
    "RECURSIVE_NUISANCE_POLICY_SCHEMA",
    "RECURSIVE_NUISANCE_POLICY_VERSION",
    "ClaimBearingProb4DStreamRunV1",
    "ClaimBearingProb4DStreamStepV1",
    "Prob4DObservationFactorStreamUpdateV1",
    "Prob4DObservationFactorStreamV1",
    "Prob4DStreamObservationBindingV1",
    "RecursiveNuisanceMode",
    "RecursiveNuisancePolicyV1",
    "apply_claim_bearing_prob4d_stream_update",
    "bind_prob4d_stream_observation",
    "load_claim_bearing_prob4d_stream_run",
    "load_prob4d_observation_factor_stream",
    "prob4d_observation_identity_summary",
    "start_claim_bearing_prob4d_stream_run",
    "write_claim_bearing_prob4d_stream_run",
    "write_prob4d_observation_factor_stream",
]
