# Prob4D material-identity marginalization

## Purpose

The cross-window development study showed that conservative source-linked tracks
can improve a BayesianPhysTwin update over the newest-window-only reference.
That study used one hard association decision. The next necessary boundary is to
retain calibrated ambiguity when several older window-local tracks could denote
the same material point.

`bayesian_phystwin.material_identity_marginalization` adds that boundary without
adding Prob4D as a runtime dependency. It independently validates the portable
Prob4D identity-mixture JSON, binds every candidate update to one common physical
state coordinate system, and marginalizes only the state posterior moments.

This is an additive development interface. It does not alter the frozen
`ObservationBeliefV1`, provider-v2, prior-aware solver, released PhysTwin
reproduction, or Causal4D provider contracts.

## Upstream contract pin

The consumer mirrors the portable contract introduced by
`IPS-Stuttgart/Prob4D` pull request `#100` at source blob
`f36726a26592397344ef113531f3c02a31878d90` on branch
`agent/material-identity-marginalization` (head revision
`f813a26288105e56c6d122123ae974be26662d3d` when this interface was written).

The accepted upstream semantics are exact:

- schema `prob4d.material-identity-mixture`, version `1`;
- source weights `source-calibrated-log-weight-v1`;
- null hypothesis `newest-window-local-reference-v1`;
- exactly one null candidate in canonical first position;
- window-local endpoints only, with linked source windows preceding the target;
- content-addressed candidate and mixture identities; and
- the upstream claim boundary reproduced byte-for-byte.

The loader rejects duplicate JSON keys, non-finite JSON constants, unknown or
missing fields, lossy integer controls, invalid revisions or digests, reordered
candidates, tampered content identities, duplicate linked endpoints, and changed
semantics. Validation is implemented independently rather than trusting an
already-instantiated Prob4D object.

## Bayesian update

Let candidate `k=0` be the newest-window null reference and let `k>0` denote
linked source hypotheses. Prob4D supplies calibrated source log weights
`log pi_k`. A separately frozen, prefix-only BayesianPhysTwin calibration
supplies candidate-aligned log likelihoods `ell_k` and a generalized-Bayes power
`lambda >= 0`:

```text
p(k | prefix) proportional to pi_k * exp(lambda * ell_k).
```

The likelihood artifact is content-addressed and records:

- the exact Prob4D mixture and candidate order;
- a common-state-domain ID;
- a calibration artifact ID;
- the likelihood power;
- the candidate log-likelihood vector digest;
- explicit `target_outcomes_used = false`; and
- the fixed `prefix-only-candidate-log-likelihood-v1` semantics.

The current prior-aware grouped-mixture solver exposes an admissible posterior,
not an exact candidate model evidence. Its IRLS objective, robust weights, or
accept/reject reason must therefore not be relabeled as `ell_k`. Candidate log
likelihoods need a separately specified and source-calibrated prefix scoring
rule.

For admissible candidate state posteriors `N(mu_k, P_k)`, BayesianPhysTwin uses
the law of total covariance:

```text
mu = sum_k p_k mu_k
P_within  = sum_k p_k P_k
P_between = sum_k p_k (mu_k - mu) (mu_k - mu)^T
P = P_within + P_between.
```

The returned object exposes both covariance terms. Omitting `P_between` would be
overconfident whenever plausible identities imply different physical states.

## Common-state-domain requirement

Moment averaging is meaningful only when every candidate uses the same state
coordinates. The caller must provide a content-addressed
`common_state_domain_id` that binds, at minimum, the physical state basis,
physical linearization, state prior, query definition, units, and causal prefix.

Before solving candidate `k`, add the required lineage fields to the candidate
batch metadata:

```python
from bayesian_phystwin.material_identity_marginalization import (
    material_identity_candidate_lineage,
)

metadata = material_identity_candidate_lineage(
    mixture,
    candidate_id=candidate_id,
    common_state_domain_id=common_state_domain_id,
    metadata={
        "physical_linearization_artifact_id": linearization.artifact_id,
    },
)
```

`marginalize_material_identity_state` rejects missing or contradictory lineage,
a changed candidate set, or mismatched state dimensions. It extracts only the
leading state block of each `GaugeAwareBeliefResult.posterior_covariance`.
Gauge, shared-bias, view-bias, anchor-bias, particle, and complete-belief moments
are not averaged across identity hypotheses by this interface.

## Exact reference fallback

Data-integrity errors raise immediately. Scientific or numerical admission
failures are routed differently:

- any inadmissible candidate result returns the exact null-candidate state mean
  and covariance;
- an all-impossible candidate likelihood vector returns the exact null result;
- a null-only mixture returns the exact null result; and
- posterior mass that is exactly one on the null returns the exact null result
  without arithmetic reconstruction.

The fallback posterior is one-hot on candidate zero, has zero between-identity
covariance, and records every candidate's inference-admissibility status. This
preserves the newest-window reference numerically. Deployment still requires the
existing nonlinear-closure check, source-frozen regret guard, and complete-belief
selection; rejection at that layer must return the exact baseline belief object.

## Example

```python
import numpy as np

from bayesian_phystwin.material_identity_marginalization import (
    MaterialIdentityLikelihoodEvidenceV1,
    load_prob4d_material_identity_mixture,
    marginalize_material_identity_state,
)

mixture = load_prob4d_material_identity_mixture(
    "prob4d-material-identity-mixture.json"
)

evidence = MaterialIdentityLikelihoodEvidenceV1(
    mixture_id=mixture.mixture_id,
    common_state_domain_id=common_state_domain_id,
    candidate_ids=mixture.candidate_ids,
    log_likelihoods=np.asarray(candidate_prefix_log_likelihoods),
    calibration_id=likelihood_calibration_artifact_id,
    likelihood_power=frozen_likelihood_power,
    target_outcomes_used=False,
)

posterior = marginalize_material_identity_state(
    mixture,
    evidence,
    candidate_results_by_id,
)
```

## Claim boundary and next experiment

This module establishes contract conformance, exact candidate alignment,
fail-closed numerical routing, and correct Gaussian state moment algebra. Its
tests are synthetic contract and numerical tests. They do not establish that
Prob4D associations are calibrated on real objects, that the candidate
likelihood is predictive, that the marginalized update improves physical state,
or that Causal4D interventions improve.

The valid scientific next step is a calibration-separated `P2m` arm in a frozen
cross-window study:

- `P0`: newest-window persistent reference;
- `P2`: hard source-linked identity from the completed development study;
- `P2m`: source-calibrated identity marginalization through this interface; and
- `P3`: evaluation-only oracle identity.

Association calibration, likelihood/power calibration, guard calibration, and
downstream evaluation must use disjoint object or simulation groups. Promotion
requires proper-score or trajectory improvement over `P0`, bounded harmful
accepted updates, and exact fallback on every rejection. A negative result is a
complete result and must not be retuned on the same target partition.
