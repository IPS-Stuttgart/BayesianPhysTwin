# Recursive Prob4D visual-bias streams

Prob4D can bind several causal observation-factor updates to one
source-calibrated coherent visual-bias latent. BayesianPhysTwin independently
reconstructs that producer artifact and verifies its relationship to the
admitted physical observations before any claim-bearing use.

## Why a stream contract is necessary

A one-shot `VisualBiasNuisanceV1` contains row-local bias Jacobians and one
complete joint prior covariance. Reusing that one-shot prior independently for
several recursive updates would count the same calibration prior more than once
and would remove the cross-update dependence induced by the shared latent.

The stream contract therefore represents the latent and prior once, while each
member records:

- the exact Prob4D observation-factor stream update ID;
- the exact visual-bias sidecar artifact ID;
- the exact BayesianPhysTwin observation artifact and ordered-row identity;
- the causal frame interval and contiguous row interval;
- the preceding visual-bias stream update ID; and
- the maximum residual projection onto the admitted global gauge span.

The complete bias covariance is retained once for the whole stream. It is not
added to local point covariance while the explicit nuisance remains represented.

## Independent BayesianPhysTwin validation

`validate_prob4d_visual_bias_nuisance_stream(...)` does not import or trust
Prob4D's validator. It independently:

1. reconstructs every content-addressed update, the shared bias-model ID, and
   the complete stream artifact ID;
2. checks closed schema semantics, canonical ordering, array dtypes and shapes,
   finite values, positive-semidefinite covariance, immutable retained storage,
   and the producer claim boundary;
3. rejects duplicate factor-update IDs, sidecar IDs, observation artifacts, and
   ordered-row identities;
4. verifies the append-only update chain, non-overlapping causal intervals,
   contiguous row intervals, and the exact row-to-update assignment;
5. binds each member to the corresponding
   `Prob4DObservationFactorStreamV1` update and `ObservationBeliefV1` through
   BayesianPhysTwin's existing independent factor-stream validator;
6. independently validates each one-shot sidecar and compares its bias scopes,
   basis, covariance, row mapping, Jacobians, artifact identity, and gauge
   residual with the corresponding stream slice; and
7. requires a `RecursiveNuisancePolicyV1` in `persistent_explicit_state` mode
   that names the exact model-scoped nuisance family.

The model-scoped family identifier is

```python
family_id = prob4d_visual_bias_nuisance_family_id(stream.bias_model_id)
policy = RecursiveNuisancePolicyV1(
    mode="persistent_explicit_state",
    state_domain_id=state_domain_id,
    nuisance_family_ids=("prob4d-gauge", family_id),
)
```

## Producer compatibility pin

The producer-consumer regression is bound to merged Prob4D revision
`e37c3d50d4a07a2c3760389e79d59b0ac9402dc4`, which introduced recursive
visual-bias nuisance streams. The workflow checks out that immutable revision
rather than a moving branch, installs both repositories into a fresh
environment, and verifies exact producer artifact IDs, member identities, row
slices, covariance, and execution semantics. Advancing the producer revision
requires an explicit reviewed workflow change and a fresh parity run.

## Execution paths

The existing claim-bearing V2 visual-bias solver consumes one sidecar with its
complete covariance. Its one-shot contract and artifact identities are
unchanged.

`Prob4DVisualBiasStreamConsumptionBindingV1` remains fail-closed when a caller
tries to execute a multi-update stream through that one-shot path:

- a single-update stream is compatible with the existing V2 solver;
- `binding.claim_bearing_execution_admissible` remains false for a multi-update
  stream; and
- `binding.require_claim_bearing_execution()` rejects repeated one-shot use
  because it would duplicate the prior.

Multi-update execution uses the separate persistent-state API in
`bayesian_phystwin.persistent_prob4d_visual_bias`. A
`PersistentVisualBiasRunV1`:

- instantiates the source-calibrated bias prior exactly once;
- carries one joint physical/bias posterior and its cross-covariance through
  every update and physical prediction;
- supports singular source bias covariance through a covariance-root latent;
- accumulates observation information row by row without materializing the
  complete global bias design;
- accepts only the next bound stream member;
- rejects stale candidates and replayed updates; and
- commits an accepted complete posterior or retains the exact prior belief
  object on rejection.

A rejected observation does not update either the physical state or the
persistent visual-bias state. It is nevertheless recorded as consumed so it
cannot be retried under a changed guard. See
[`persistent_prob4d_visual_bias_solver.md`](persistent_prob4d_visual_bias_solver.md)
for the model, API, prediction semantics, and contract evidence.

## Intended ecosystem route

```text
Prob4D observation factors + gauge/identity uncertainty
                      +
Prob4D persistent visual-bias stream
                      ↓
BayesianPhysTwin independent validation
                      ↓
persistent joint physical/bias candidate
                      ↓
frozen accept decision or exact complete-belief fallback
                      ↓
Causal4D abduction, intervention, and held-out prediction
```

Causal4D should receive only the selected physical belief and its lineage. Raw
Prob4D factors or visual-bias likelihoods must not be applied again downstream.

## Claim boundary

A valid binding and persistent run establish exact artifact, row, covariance,
causal-order, nuisance-policy, posterior-state, and fallback consistency. They
do not establish provider competence, target calibration, complete bias
coverage, physical-state identifiability, guarded-query improvement, Causal4D
intervention benefit, deployment safety, or state of the art.
