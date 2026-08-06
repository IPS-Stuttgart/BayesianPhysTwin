# Persistent Prob4D visual-bias posterior

## Purpose

Prob4D can publish several causal observation updates that share one
source-calibrated coherent visual-bias latent. The existing one-shot
BayesianPhysTwin adapter correctly preserves the complete bias covariance, but
restarting that adapter for every update would insert the same source prior more
than once and make repeated evidence spuriously informative.

The additive `persistent_prob4d_visual_bias` interface carries one explicit
provider-space bias posterior across the append-only factor stream:

```text
Prob4D visual-bias stream + persistent nuisance policy
                         |
                         v
       provider bias b ~ N(mu[k-1], P[k-1])
                         |
                         v
         one causal BayesianPhysTwin update
                         |
                         v
 candidate N(mu[k], P[k]) or exact prior-moment fallback
                         |
                         v
       selected complete belief + next stream member
```

The frozen one-shot visual-bias update and all historical artifact identities
remain unchanged.

## Recursive reparameterization

For update `k`, let the provider sidecar define the row design `B[k]` and let
the carried posterior be

```text
b ~ N(mu[k-1], P[k-1]).
```

With the symmetric positive-semidefinite root
`L[k-1] L[k-1]' = P[k-1]`, the existing prior-aware solver can retain its
isotropic shared-bias coordinate `u`:

```text
u ~ N(0, sigma_shared^2 I),
b = mu[k-1] + L[k-1] u / sigma_shared.
```

The expected bias `B[k] mu[k-1]` is added to the physical prediction and the
solver receives

```text
H_u = B[k] L[k-1] / sigma_shared.
```

After the update, provider-space moments are reconstructed as

```text
mu[k] = mu[k-1] + L[k-1] E[u | y] / sigma_shared,
P[k]  = L[k-1] Cov(u | y) L[k-1]' / sigma_shared^2.
```

This preserves the complete cross-scope covariance and inserts the original
source prior exactly once. Its marginal contribution is never also added to
local point covariance.

At the first update, `mu[0] = 0` and `P[0]` is the source-calibrated Prob4D
covariance. The resulting provider-space moments are numerically equivalent to
the existing one-shot V2 adapter.

## State and lineage

`PersistentProb4DVisualBiasStateV1` is content addressed and carries:

- the exact BayesianPhysTwin stream binding, Prob4D factor stream, visual-bias
  stream, bias model, recursive nuisance policy, and nuisance-family identities;
- ordered provider coefficient names;
- the selected provider-bias mean and complete covariance;
- the exact consumed factor-stream prefix;
- the parent state and transition update;
- whether the candidate or fallback branch was selected; and
- the exact selected complete-belief ID.

The next physical linearization must bind all of the following in its metadata:

```python
{
    "recursive_nuisance_policy_id": binding.recursive_nuisance_policy_id,
    "persistent_prob4d_visual_bias_state_id": bias_state.state_id,
    "prob4d_visual_bias_stream_binding_id": binding.binding_id,
}
```

Its `baseline_belief_id` must equal
`bias_state.selected_complete_belief_id`, and its observation artifact must be
the next stream observation. Replayed, skipped, reordered, or cross-stream
updates fail before inference.

## Usage

```python
from bayesian_phystwin.persistent_prob4d_visual_bias import (
    apply_claim_bearing_prob4d_stream_update_with_persistent_visual_bias,
    initialize_persistent_prob4d_visual_bias_state,
    update_claim_bearing_prob4d_visual_bias_stream_from_artifacts,
)

bias_state = initialize_persistent_prob4d_visual_bias_state(
    visual_bias_stream_binding,
    initial_complete_belief,
)
run = start_claim_bearing_prob4d_stream_run(
    factor_stream,
    initial_complete_belief,
    nuisance_policy=nuisance_policy,
)

persistent_update = (
    update_claim_bearing_prob4d_visual_bias_stream_from_artifacts(
        visual_bias_stream_binding,
        bias_state,
        observation_belief,
        physical_linearization,
        physical_prediction_xyz_m=physical_prediction,
    )
)

selected_belief, bias_state, run, step = (
    apply_claim_bearing_prob4d_stream_update_with_persistent_visual_bias(
        factor_stream,
        run,
        visual_bias_stream_binding=visual_bias_stream_binding,
        prior_bias_state=bias_state,
        baseline=baseline_belief,
        candidate=candidate_belief,
        observation=observation_belief,
        linearization=physical_linearization,
        persistent_update=persistent_update,
        decision=complete_belief_guard_decision,
        nuisance_policy=nuisance_policy,
    )
)
```

The combined routing function verifies that physical-belief selection and bias
selection make the same decision and records the selected bias-state ID in the
claim-bearing stream step.

## Exact fallback

If numerical inference is inadmissible, selecting the candidate is forbidden.
If the frozen complete-belief guard rejects an otherwise admissible update:

- the returned physical object is the exact baseline object;
- the provider-bias mean and covariance are byte-equivalent to the prior state;
- the new state records the rejected member as consumed; and
- the next update must bind that fallback state's ID and the unchanged selected
  complete-belief ID.

Discarding the rejected update's bias learning is deliberately conservative. It
prevents later physical decisions from depending on evidence that failed the
registered complete-belief decision.

## Posterior cross-covariance

Each update exposes current physical-state-to-bias and gauge-to-bias posterior
cross-covariances in provider coordinates. They are retained as diagnostics for
the current linearization, but are not silently propagated as a physical
transition model.

The carried state is the static provider-bias marginal. A future dynamic bias
model requires a separately source-calibrated transition contract rather than
an unregistered random walk.

## Persistence

States can be stored and independently revalidated:

```python
from bayesian_phystwin.persistent_prob4d_visual_bias import (
    load_persistent_prob4d_visual_bias_state,
    write_persistent_prob4d_visual_bias_state,
)

write_persistent_prob4d_visual_bias_state(
    bias_state,
    "outputs/case-a/persistent-visual-bias-state.json",
)
restored = load_persistent_prob4d_visual_bias_state(
    "outputs/case-a/persistent-visual-bias-state.json"
)
```

Writes are atomic and non-overwriting by default. Array descriptors and the
complete state identity are recomputed during loading.

## Claim boundary

This interface establishes recursive prior accounting, exact stream and belief
lineage, covariance-preserving reparameterization, and exact fallback. It does
not establish:

- real Prob4D or another visual provider's competence;
- physical-state identifiability from camera evidence alone;
- calibrated query or deployment uncertainty;
- lower physical-query error on a fresh object or acquisition session;
- a dynamic coherent-bias transition model;
- Causal4D intervention benefit; or
- deployment safety or state of the art.

The posterior covariance remains the solver's working IRLS/Gauss--Newton
covariance. Coverage claims still require an independently grouped calibration
artifact and a frozen prospective confirmation protocol.
