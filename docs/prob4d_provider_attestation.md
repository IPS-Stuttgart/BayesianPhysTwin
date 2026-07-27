# Prob4D provider-v2 attestation validation

Bayesian-PhysTwin accepts historical provider-v1 observations for frozen
reproduction, but new prospective Prob4D evidence should carry a calibrated,
self-contained provider-v2 attestation.

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

New Prob4D-to-Bayesian-PhysTwin experiments should call:

```python
from bayesian_phystwin.prob4d_causal_lineage import (
    validate_claim_bearing_prob4d_observation_belief,
)

validation = validate_claim_bearing_prob4d_observation_belief(observation)
```

This rejects:

- provider-v1 artifacts without an attestation;
- exploratory provider-v2 exports;
- missing or incompatible calibration artifacts;
- legacy covariance-root or composition-Jacobian modes;
- environment-only, mismatched, unavailable, or dirty runtime provenance; and
- fixed-lag products that do not satisfy the strict causal-stream contract.

The returned lineage summary records the provider manifest ID, export mode,
calibration IDs, numerical modes, and runtime-evidence source without duplicating
the complete embedded provider manifest into every downstream result.

## Trust boundary

A producer attestation is not external certification. Its value is that the exact
producer statement is complete, immutable under the observation content address,
and independently checkable. Dataset independence, prospective calibration, and
physical-prediction benefit remain separate empirical gates.
