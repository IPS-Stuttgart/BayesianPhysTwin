# Deform360 Prob4D target-outcome authorization v1

## Purpose

The Deform360 confirmation protocol has two distinct information boundaries.

1. `Deform360ConfirmationOpeningAuthorizationV1` permits the frozen predictor-side
   confirmation inputs to be opened after Stage-1 calibration and observability
   evidence are sealed.
2. `Deform360Prob4DTargetOutcomeAuthorizationV1` is created only after the
   provider-side inputs have produced exact Prob4D target manifests. It binds
   those manifests to the frozen BayesianPhysTwin cohort while target outcomes
   remain closed.

This separation is necessary because a target-provider admission cannot exist
before the confirmation camera inputs have been processed. Requiring that
admission before any confirmation input access would be circular. Requiring it
before target-outcome access preserves the intended information order.

```text
Stage-1 calibration and observability seal
        |
        v
confirmation-opening authorization
        |
        v
open predictor-side confirmation inputs only
        |
        v
produce and metadata-admit exact Prob4D target manifests
        |
        v
Prob4D target-outcome authorization
        |
        v
open target outcomes once and retain the registered result
```

## Cross-repository identities

The authorization validates all six retained source artifacts:

- the exact BayesianPhysTwin Stage-0 selection;
- the target-blind visual-provider lock;
- the Stage-1 confirmation-opening authorization;
- the Prob4D portable cohort binding;
- the Prob4D held-out promotion lock; and
- the metadata-only target-provider admission.

The Prob4D cohort binding must reproduce the exact ten calibration and twelve
confirmation objects, episodes, strata, metadata identities, dataset revision,
processing revision, selection identities, and names/metadata-only boundary from
BayesianPhysTwin Stage 0.

The promotion lock must:

- bind that exact cohort-binding ID;
- bind the visual-provider lock as `provider_configuration`;
- use the same MotionCrafter revision and immutable model-set identity;
- retain the complete twelve-object target set;
- retain every required provider/query comparison role;
- bind the Stage-1 authorization and visual-provider lock in its metadata; and
- require the complete target cohort rather than permitting target-informed
  omission.

The target-provider admission must:

- bind the exact promotion lock, cohort, run specification, provider revision,
  model set, and target groups;
- contain one entry for every Stage-0 confirmation object with the exact episode
  and stratum;
- contain at least one causally admitted payload for every object;
- keep every payload source interval within its declared causal cutoff;
- declare `target_outcomes_used=false`; and
- bind the Stage-1 authorization, visual-provider lock, and original prediction
  producer revision in content-addressed metadata.

The required admission metadata keys are:

```json
{
  "bayesian_phystwin_confirmation_opening_authorization_id": "<SHA-256>",
  "bayesian_phystwin_visual_provider_lock_id": "<SHA-256>",
  "prediction_provider_revision": "<exact Prob4D producer commit>",
  "confirmation_provider_inputs_only": true
}
```

The promotion-lock metadata contains the first two bindings. This permits a newer
Prob4D control-plane revision to validate and evaluate artifacts while retaining
the exact older prediction-producer revision frozen before calibration access.
The scientific producer cannot drift merely because admission tooling was added
later.

## Atomic command

After predictor-side confirmation inputs have been opened and the target-provider
admission has been generated, run:

```bash
python scripts/science/authorize_deform360_prob4d_target_outcomes.py \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/\
visual-provider-lock.json \
  --confirmation-opening-authorization \
    outputs/stage1/confirmation-opening-authorization.json \
  --prob4d-cohort-binding outputs/prob4d/deform360-cohort-binding.json \
  --prob4d-promotion-lock outputs/prob4d/promotion-lock.json \
  --target-provider-admission outputs/prob4d/target-provider-admission.json \
  --confirmation-provider-inputs-opened \
  --output-dir outputs/deform360-target-outcome-authorization
```

The acknowledgement means that predictor-side confirmation inputs were opened to
produce the admitted manifests. It does **not** acknowledge target-outcome access.
The command rejects symbolic-link traversal, copies every source into a private
bundle, verifies copied bytes, constructs the authorization, writes all products,
checksums the complete directory, and publishes it atomically.

The output contains:

```text
sources/
  bayesian-phystwin/
    stage0-selection.json
    visual-provider-lock.json
    confirmation-opening-authorization.json
  prob4d/
    cohort-binding.json
    promotion-lock.json
    target-provider-admission.json
target-outcome-authorization.json
query-result-metadata.json
target-outcome-authorization-summary.json
SHA256SUMS
```

## Query-result binding

`query-result-metadata.json` is the exact metadata block to retain on the later
BayesianPhysTwin group-by-arm query-result stream. It includes:

```json
{
  "target_provider_admission_id": "<SHA-256>",
  "deform360_confirmation_opening_authorization_id": "<SHA-256>",
  "deform360_prob4d_target_outcome_authorization_id": "<SHA-256>",
  "prob4d_promotion_lock_id": "<SHA-256>",
  "prob4d_cohort_binding_id": "<SHA-256>"
}
```

Prob4D requires `target_provider_admission_id` when it seals and verifies the
held-out promotion result. The additional identities make the BayesianPhysTwin
pre-outcome authorization independently auditable without relying on filenames or
mutable execution directories.

## Claim boundary

A passing authorization establishes only information order and artifact
continuity. It does not establish provider competence, calibrated uncertainty,
physical-query benefit, tactile benefit, harmless accepted updates, Causal4D
benefit, deployment safety, or state of the art. The twelve-object outcome must be
opened once, reported whether positive or negative, and never used to retune the
same frozen protocol.
