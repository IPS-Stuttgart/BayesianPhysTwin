# Recursive Prob4D visual-bias streams

Prob4D can bind several causal observation-factor updates to one source-calibrated coherent visual-bias latent. BayesianPhysTwin independently reconstructs that producer artifact and verifies its relationship to the admitted physical observations before any claim-bearing use.

## Why a stream contract is necessary

A one-shot `VisualBiasNuisanceV1` contains row-local bias Jacobians and one complete joint prior covariance. Reusing that one-shot prior independently for several recursive updates would count the same source-calibrated uncertainty more than once and would remove the cross-update dependence induced by the shared latent.

The stream contract therefore represents the latent and prior once, while each member records:

- the exact Prob4D observation-factor stream update ID;
- the exact visual-bias sidecar artifact ID;
- the exact BayesianPhysTwin observation artifact and ordered-row identity;
- the causal frame interval and contiguous row interval;
- the preceding visual-bias stream update ID;
- the maximum residual projection onto the admitted global gauge span.

The complete bias covariance is retained once for the whole stream. It is not added to local point covariance while the explicit nuisance remains represented.

## Independent BayesianPhysTwin validation

`validate_prob4d_visual_bias_nuisance_stream(...)` does not import or trust Prob4D's validator. It independently:

1. reconstructs every content-addressed update, the shared bias-model ID, and the complete stream artifact ID;
2. checks closed schema semantics, canonical ordering, array dtypes and shapes, finite values, positive-semidefinite covariance, immutable retained storage, and the producer claim boundary;
3. rejects duplicate factor-update IDs, sidecar IDs, observation artifacts, and ordered-row identities;
4. verifies the append-only update chain, non-overlapping causal intervals, contiguous row intervals, and the exact row-to-update assignment;
5. binds each member to the corresponding `Prob4DObservationFactorStreamV1` update and `ObservationBeliefV1` through BayesianPhysTwin's existing independent factor-stream validator;
6. independently validates each one-shot sidecar and compares its bias scopes, basis, covariance, row mapping, Jacobians, artifact identity, and gauge residual with the corresponding stream slice;
7. requires a `RecursiveNuisancePolicyV1` in `persistent_explicit_state` mode that names the exact model-scoped nuisance family.

The model-scoped family identifier is

```python
family_id = prob4d_visual_bias_nuisance_family_id(stream.bias_model_id)
policy = RecursiveNuisancePolicyV1(
    mode="persistent_explicit_state",
    state_domain_id=state_domain_id,
    nuisance_family_ids=("prob4d-gauge", family_id),
)
```

## Current execution boundary

The existing claim-bearing V2 visual-bias solver consumes one sidecar with its complete covariance. It does **not** yet propagate one shared visual-bias posterior state through several physical recursive updates.

Consequently:

- a single-update stream is compatible with the existing V2 one-shot solver;
- a multi-update stream is fully validated and bound, but `claim_bearing_execution_admissible` is false;
- `require_claim_bearing_execution()` rejects a multi-update stream with `persistent_visual_bias_state_solver_required`;
- BayesianPhysTwin never approximates a shared stream by running several independent one-shot updates.

This fail-closed boundary is intentional. A later persistent-state solver may consume the validated global design and one shared prior, but it must preserve the same model identity and posterior state across updates.

## Intended ecosystem route

```text
Prob4D observation factors + gauge/identity uncertainty
                      +
Prob4D persistent visual-bias stream
                      ↓
BayesianPhysTwin independent validation and physical update
                      ↓
accepted selected physical belief or exact fallback
                      ↓
Causal4D abduction, intervention, and held-out prediction
```

Causal4D should receive only the selected physical belief and its lineage. Raw Prob4D factors or visual-bias likelihoods must not be applied again downstream.

## Claim boundary

A valid binding establishes exact artifact, row, covariance, causal-order, and nuisance-policy consistency. It does not establish provider competence, target calibration, complete bias coverage, physical-state identifiability, guarded-query improvement, Causal4D intervention benefit, deployment safety, or state of the art.
