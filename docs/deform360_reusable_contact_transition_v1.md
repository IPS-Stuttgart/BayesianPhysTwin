# Deform360 reusable contact-transition addendum v1

## Purpose

The frozen reusable-PhysTwin protocol already uses an outcome-independent
geometry-latched contact schedule. Source diagnostics leave a small but concrete
contact-onset oracle gap, while the generic neural residual abstains. This
addendum tests the narrower mechanism suggested by that evidence: dynamic
contact realization.

The addendum does not alter the parent object panel, episode split, evaluator
boundary, trusted-physics policy, or Causal4D claim. It was locked after
preprocessing the first fit input for `004-rubber-band` and before opening any
development held outcome.

## Candidate

For each object, separate logistic onset and release hazards are trained from
the six fit episodes. The only dynamic features are gripper openness,
gripper-to-predicted-object proximity, and causal relative closing speed. The
held rollout begins from one object frame and the known robot trajectory. Its
frame-zero contact state is inferred from the onset hazard, never read from
target tactile.

The feature rollout is deliberately finite and auditable:

1. run the frozen geometry-latched reusable twin;
2. derive proximity and closing-speed features from that prediction;
3. predict the full contact schedule with the source-trained hazards;
4. rerun official Warp once with the predicted schedule;
5. apply the unchanged observable trust policy.

No future object observation or target tactile enters steps 1--5. Full future
tactile is opened only after all prediction hashes are sealed and is used as an
oracle headroom diagnostic.

## Controls and gate

The causal-transition arm must beat the frozen static-contact trusted arm, not
just persistence. Admission requires at least 2% aggregate improvement in both
future Chamfer and track error, 28 wins among 48 development held episodes, no
episode worse by more than 10%, no category-level median degradation, improved
contact Brier score, and non-degraded uncertainty metrics. Every condition is
conjunctive.

Failure leaves the parent method byte-for-byte unchanged. Unsupported or
invalid contact predictions fall back to exact persistence. Confirmatory data
remain inaccessible until this development decision is frozen.

The executable lock is
`configs/causal4d_public/deform360_reusable_contact_transition_v1.json`.
