# Prob4D provider-v2 attestation validation

Bayesian-PhysTwin accepts historical provider-v1 observations for frozen
reproduction, but new prospective Prob4D evidence must carry an explicit stream-v2
joint-gauge artifact and a calibrated, self-contained provider-v2 attestation.

The validator in `bayesian_phystwin.prob4d_provider_attestation` intentionally
does not import Prob4D. It independently checks the neutral JSON/hash contract
bound into `ObservationBeliefV1.metadata`.

## Validation order

`validate_prob4d_causal_observation_belief` performs the existing observation,
metric-anchor, causal-lineage, covariance, and stream-version checks first. When
`prob4d_provider_attestation` is present, it additionally verifies:

- attestation schema and exact declared fields;
- the embedded provider manifest's SHA-256 content address;
- provider identity, exact source revision, API version, import boundary, and
  observation schema versions;
- the capabilities and limitations required by the current claim-bearing
  provider-v2 contract;
- calibrated versus exploratory mode consistency;
- gauge and point calibration artifact IDs;
- canonical covariance-root and analytic composition-Jacobian modes; and
- matched, independently verified runtime revision evidence.

Any malformed attestation fails closed even in the ordinary compatibility mode.
An absent attestation remains acceptable for frozen provider-v1 artifacts.

## Prospective claim-bearing boundary

New Prob4D-to-Bayesian-PhysTwin experiments should use the dedicated adapter:

```python
from bayesian_phystwin import (
    build_claim_bearing_gauge_aware_batch_from_observation_belief,
)

adapted = build_claim_bearing_gauge_aware_batch_from_observation_belief(
    observation,
    physical_prediction_xyz_m=physical_prediction,
    state_jacobian=state_jacobian,
    query_state_jacobian=query_state_jacobian,
    physical_response_scale_m=physical_response_scale,
)
```

When a content-addressed `PhysicalLinearizationV1` is available, use
`build_claim_bearing_gauge_aware_batch_from_artifacts` instead. Both functions
validate the complete claim-bearing boundary before the observation innovation is
formed.

The lower-level validation entry point remains available:

```python
from bayesian_phystwin.prob4d_causal_lineage import (
    validate_claim_bearing_prob4d_observation_belief,
)

validation = validate_claim_bearing_prob4d_observation_belief(observation)
```

Claim-bearing admission now requires all of the following:

- an explicitly declared Prob4D causal stream contract version 2, not an inferred
  version and not the legacy stream-v1 representation;
- the sequential joint spanning-tree covariance model with one shared low-rank
  factor and preserved cross-window covariance;
- calibrated covariance metadata whose gauge and point artifact IDs exactly match
  the provider attestation;
- calibration on every gauge alignment;
- no uncalibrated covariance permission;
- no pointwise covariance fallback permission or recorded fallback use;
- canonical covariance roots and analytic composition Jacobians; and
- independently matched source code from VCS metadata or a clean source checkout.

This rejects provider-v1 artifacts, exploratory provider-v2 exports, attested legacy
per-window gauge marginals, inferred stream versions, fixed-lag approximations,
calibration identity drift, partial gauge calibration, covariance fallback, and
unverified runtime provenance.

The strict adapters record the validated provider-manifest ID, calibration IDs,
stream version, and runtime-evidence source in the resulting gauge-aware batch.
`build_gauge_aware_batch_from_observation_belief` remains the compatibility path for
frozen provider-v1 and explicitly labelled exploratory artifacts.

## Trust boundary

A producer attestation is not external certification. Its value is that the exact
producer statement is complete, immutable under the observation content address,
and independently checkable. Dataset independence, prospective calibration, the
baseline-relative update guard, and physical-prediction benefit remain separate
empirical gates.
