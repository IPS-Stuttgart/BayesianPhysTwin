# Claim-bearing Prob4D coherent visual-bias updates

## Purpose

Prob4D can export a `VisualBiasNuisanceV1` sidecar for coherent errors that are
shared across rows, views, or provider scopes. The sidecar retains:

- the exact observation artifact and ordered row-identity digest;
- row-local bias Jacobians;
- ordered bias scopes and basis names;
- one complete joint covariance over every scope and basis coefficient; and
- evidence that the basis was projected out of the complete conditional-whitened
  `Sim(3)` gauge span.

BayesianPhysTwin must not replace that joint covariance with independent scalar
bias priors. It must also not add its marginal contribution to point covariance
while retaining the same bias as an explicit latent variable. The V2 consumer in
`bayesian_phystwin.prob4d_visual_bias_update` preserves those requirements
without changing the frozen V1 update or solver.

## Covariance-preserving reparameterization

Let the Prob4D sidecar define

```text
z = h(x) + B b + epsilon,
b ~ N(0, Sigma_b).
```

The existing BayesianPhysTwin solver represents shared-bias coordinates `u`
with the isotropic prior

```text
u ~ N(0, sigma_shared^2 I).
```

Using the symmetric positive-semidefinite root `L L' = Sigma_b`, the V2 adapter
sets

```text
b = L u / sigma_shared,
H_u = B L / sigma_shared.
```

Therefore

```text
H_u Cov(u) H_u' = B Sigma_b B',
```

including every cross-row and cross-scope covariance term. No approximation is
introduced by this change of coordinates. Singular covariance is allowed; null
prior directions map to zero physical bias.

## Admission boundary

`validate_prob4d_visual_bias_nuisance` independently reconstructs the producer
artifact identity from the descriptor and array bytes. It then verifies:

1. the exact `ObservationBeliefV1` artifact ID;
2. the exact ordered Prob4D observation-identity digest;
3. canonical scope and basis identities;
4. finite float64 Jacobians and a finite positive-semidefinite joint covariance;
5. the declared gauge-orthogonalization semantics and tolerance; and
6. recursively immutable metadata and bytes-backed immutable arrays.

The one-call claim-bearing path additionally recomputes the maximum
conditional-whitened projection against the gauge design actually admitted by
BayesianPhysTwin. A producer declaration that does not match the consumer's
gauge design therefore fails before inference.

## Usage

Load the sidecar through Prob4D's strict manifest and NPZ validator, then pass the
validated object into BayesianPhysTwin:

```python
from prob4d.visual_bias import load_visual_bias_nuisance

from bayesian_phystwin.prob4d_visual_bias_update import (
    update_claim_bearing_prob4d_with_visual_bias_from_artifacts,
)

visual_bias = load_visual_bias_nuisance(
    "outputs/case-a/visual-bias.json"
)
update = update_claim_bearing_prob4d_with_visual_bias_from_artifacts(
    observation_belief,
    physical_linearization,
    visual_bias_nuisance=visual_bias,
    physical_prediction_xyz_m=physical_prediction_xyz_m,
)

print(update.inference_admissible)
print(update.provider_bias_coefficients)
print(update.provider_bias_covariance)
```

The returned provider-space moments map the solver's reparameterized shared-bias
coordinates back into the ordered Prob4D coefficient space
`bias_id:basis_name`.

## Lineage and fallback

The result lineage binds:

- the visual-bias sidecar artifact ID;
- the ordered observation-identity digest;
- the covariance-root reparameterization version;
- the shared-bias prior scale used by the frozen solver;
- the independently recomputed gauge projection;
- the fact that no marginal bias covariance was added to local point covariance;
  and
- all existing claim-bearing Prob4D provider, calibration, runtime, observation,
  and physical-linearization identities.

The V2 wrapper retains the complete `ClaimBearingProb4DUpdateV1`. Numerical
rejection and later deployment rejection continue to use the existing complete
belief and exact-fallback boundaries.

## Compatibility

This feature is additive:

- `ClaimBearingProb4DUpdateV1` is unchanged;
- frozen provider-v1/provider-v2 observations are unchanged;
- schema-v4 observation-factor bundles and stream identities are unchanged;
- the prior-aware robust likelihood and point estimate are unchanged; and
- Causal4D continues to consume only the selected BayesianPhysTwin belief.

The sidecar remains a separate artifact because adding its fields silently to a
frozen observation schema would change historical identities.

## Claim boundary

A valid sidecar and a passing integration test establish only that the declared
coherent visual-bias covariance is preserved and auditable across the
Prob4D-to-BayesianPhysTwin boundary. They do not establish real provider
competence, physical-query improvement, calibrated coverage, deployment safety,
Causal4D benefit, or state of the art. Those claims still require the frozen
independent-object or acquisition-session experiment and exact fallback for every
rejection.
