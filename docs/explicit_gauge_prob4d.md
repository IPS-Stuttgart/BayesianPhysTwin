# Claim-bearing explicit-gauge Prob4D updates

Prob4D provider API v2 can export a neutral schema-v4 observation-factor bundle
inside a strict claim-bearing envelope. The envelope binds the manifest and NPZ
payload, causal source lineage, covariance-calibration IDs, provider manifest,
and independently verified runtime revision.

BayesianPhysTwin consumes that boundary through
`bayesian_phystwin.explicit_gauge_prob4d`. The consumer does not import Prob4D at
package-import time and independently checks the fields needed before forming a
physical innovation.

## Producer and consumer flow

```python
from prob4d.provider_v2_factors import (
    load_claim_bearing_observation_factor_bundle,
    stack_sparse_observation_factors,
)
from bayesian_phystwin.explicit_gauge_prob4d import (
    update_claim_bearing_explicit_gauge_from_artifacts,
)
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1

validated = load_claim_bearing_observation_factor_bundle(
    "outputs/case-a/factors.claim.json"
)
stacked = stack_sparse_observation_factors(validated.bundle)

linearization = PhysicalLinearizationV1(
    observation_artifact_id=validated.artifact_id,
    baseline_belief_id=baseline_belief_id,
    action_prefix_id=action_prefix_id,
    simulator_revision=simulator_revision,
    frame_ids=stacked.frame_indices,
    entity_ids=stacked.point_ids,
    view_indices=view_indices,
    window_indices=stacked.gauge_indices,
    state_jacobian=state_jacobian,
    query_state_jacobian=query_state_jacobian,
    physical_response_m=physical_response_m,
)

update = update_claim_bearing_explicit_gauge_from_artifacts(
    validated,
    stacked,
    linearization,
    physical_prediction_xyz_m=physical_prediction_xyz_m,
    maximum_dense_gauge_design_bytes=256 * 1024 * 1024,
)
```

`view_indices` must use the deterministic alphabetical order of the distinct
`stacked.view_ids`. `window_indices` must equal `stacked.gauge_indices`. The
adapter compares every row identity to the content-addressed physical
linearization before constructing the innovation.

## Covariance and probability semantics

The update consumes exactly:

```text
conditional point covariance
+ one local 3 x 7 gauge Jacobian per row
+ one row gauge index
+ the complete joint 7K x 7K gauge prior
```

It does not use `marginal_world_covariance_m2` as observation noise. Adding the
marginal covariance would count the gauge contribution twice.

The four probability-like inputs remain distinct:

- `association_probability` is a generalized-Bayes row power;
- `prior_reliability` is the source-only conditional-covariance precision scale;
- `prior_nominal_probability` is the prior mixture probability for the
  correlation group; and
- `composite_weight` is Prob4D's final provider-owned information cap.

The bridge multiplies association probability with the provider composite power
only at the final likelihood-power boundary. It leaves source reliability and
nominal-component probability unchanged and selects the provider-final weight
mode so BayesianPhysTwin does not apply another effective-sample cap.

## Provenance retained in the result

Accepted and fallback results bind the same immutable lineage:

- factor-envelope artifact ID;
- physical-linearization artifact ID;
- provider-manifest ID;
- gauge and point calibration IDs;
- runtime-revision source and independent-verification flag;
- baseline belief, action prefix, and simulator revision;
- covariance semantics and the fact that marginal point covariance was not
  consumed; and
- hashes of association, source reliability, nominal probability, provider
  composite weights, and factor-row identities.

The returned `ClaimBearingProb4DUpdateV1` therefore has the same fail-closed
identity behavior as the observation-belief path.

## Bounded compatibility bridge

Prob4D's sparse stack stores `M x 3 x 7` local Jacobians and `M` gauge indices.
The current BayesianPhysTwin solver still accepts a dense `M x 3 x 7K` nuisance
design. The adapter computes the required byte count and raises `MemoryError`
before allocation when it exceeds `maximum_dense_gauge_design_bytes`. The
default limit is exactly 268,435,456 bytes (256 MiB), and every result records
both the required allocation and the enforced limit.

This closes the strict installed-wheel interoperability path while keeping the
remaining limitation visible. A future native sparse solver should accumulate
state/gauge normal-equation blocks directly from local Jacobians and gauge
indices. That future solver must preserve exact parity with this bridge and the
complete cross-window gauge prior before replacing it.

## Causal4D boundary

Raw Prob4D factors stop at BayesianPhysTwin. Causal4D should consume only the
accepted or exact-fallback BayesianPhysTwin belief with its bound lineage. It
must not reopen the factor bundle or assimilate the same visual evidence a
second time.

Passing the contract and installed-wheel tests establishes interoperability,
causal ordering, covariance accounting, and provenance. It does not establish
held-out physical-prediction benefit, harmful-update safety, uncertainty
coverage, or improved Causal4D counterfactual prediction.
