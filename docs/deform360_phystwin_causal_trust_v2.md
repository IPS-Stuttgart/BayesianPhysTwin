# Deform360 contact-regime-gated PhysTwin trust v2

## Scope and evidence order

This is an exploratory, source-only route toward the empty PhysTwin cell in
the Deform360 multi-episode benchmark. It remains separate from the frozen
Causal4D main-paper claim. For `081-stripe-rope`, episodes 1, 3, 4, 6, 7, and
9 are source actions; episodes 0, 2, and 8 are calibration-only; episode 5 is
sealed. No calibration or target episode may be read until the source gate
passes.

Episodes 1 and 4 were used to develop the causal control variate. Episode 3
was then the first wider transfer check and falsified a global trust rule.
This v2 policy and its numeric source gates were frozen before the processed
outcomes of episodes 6, 7, and 9 were available. The full six-action result is
still exploratory because the regime policy was introduced after episode 3.

## Causal control variate

Official Warp is run from the same automatically registered frame-zero state
with the measured controller trajectory (`X_driven`) and with the controller
held fixed (`X_zero`). The predictor separates action response from autonomous
simulator drift:

```text
X_hat = X_observed,0
      + a * (X_driven - X_zero)
      + b * (X_zero - X_zero,0)
```

`a,b in {0.0, 0.1, ..., 1.0}` are selected on source training frames only.
`(0,0)` is byte-identical persistence, `(1,1)` is raw PhysTwin, and `(1,0)`
retains the full simulated intervention while removing autonomous drift.

## Why global trust was rejected

Episodes 1 and 4 are prehensile actions. Their pooled training fit selected
`(a,b)=(0.4,0.1)` and improved their untouched tails by 13.07% in track RMSE
and 6.18% in Chamfer relative to persistence.

Episode 3 is a nonprehensile push. Applying the episode-1/4 rule to its
untouched tail produced 29.585 mm track RMSE and 29.534 mm Chamfer, versus
7.891 mm and 6.460 mm for persistence. The failure is mechanistically
specific: the current PhysTwin controller is a bilateral virtual attachment
spring, whereas pushing requires unilateral contact with activation and
release. A single global trust value is therefore rejected rather than
averaged across incompatible contact regimes.

When episodes 1, 3, and 4 are pooled globally, the selected weights collapse
to `(0.1,0.0)` and the pooled tails are 10.19% worse in track RMSE and 17.85%
worse in Chamfer than persistence. The registered interim artifact has SHA-256
`8502479077e1cabceed2d5c4ab93292037b7abbfb06569047e2b0d0609e269d4`.

## Frozen v2 policy

The contact regime comes from released action metadata and is available before
the outcome:

- **Prehensile:** select causal-trust weights using only prehensile source
  training intervals. Evaluate by leave-one-prehensile-action-out transfer.
- **Nonprehensile:** use exact `(0,0)` persistence until a unilateral
  contact-transition model is implemented and independently validated.

This is a model-form trust gate, not an outcome-conditioned selector. The
nonprehensile arm receives no credit as a physics improvement; it is an exact
safe fallback. Episode 6, an unseen bimanual prehensile action, is the decisive
source transfer check for the current simulator family.

## Frozen source gates

The v2 route passes only if every condition holds:

1. every nonprehensile fold is exactly persistence;
2. both track and Chamfer beat persistence in at least two of the three
   leave-one-prehensile-action-out folds;
3. pooled six-action leave-one-out track improvement is at least 3%;
4. pooled six-action leave-one-out Chamfer improvement is at least 1%;
5. no prehensile fold degrades either metric by more than 10%;
6. all hull, registration, association, support, driven-rollout, zero-action,
   checksum, and information-boundary checks pass;
7. mutating an untouched source tail cannot change selected weights.

Passing permits uncertainty calibration on episodes 0, 2, and 8 followed by
one sealed episode-5 evaluation. Failing freezes this route as a transparent
negative result. A benchmark-wide state-of-the-art claim additionally requires
the official multi-object protocol.

## Scientific target

The published Deform360 multi-episode future-prediction leader is
ParticleFormer at 51 mm Chamfer and 79 mm track error. PhysTwin is absent from
that table because it requires per-episode registration. The intended
contribution is therefore not merely a lower number on one rope. It is an
automatic reusable physical twin that:

1. registers a new episode without manual track identities;
2. transfers object-level physics across actions;
3. detects when its contact mechanism is outside its validation domain;
4. falls back without degrading prediction; and
5. reports calibrated uncertainty after source transfer succeeds.

## Claim boundary

The causal-trust weights are source-calibrated model-form variables. They are
not material constants, measured actuator gains, or identified grasp
stiffnesses. Current Deform360 control trajectories are vision-recovered rather
than independent commanded and measured streams, so persistent actuation
variables remain unidentifiable. Future object observations are never used in
a predictive rollout, and the sealed target remains untouched.
