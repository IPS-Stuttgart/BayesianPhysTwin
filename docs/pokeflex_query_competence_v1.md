# PokeFlex query-competence validation v1

## Scientific role

This study tests the controlled query-competence mechanism on public physical
measurements from PokeFlex. The registered input is the 78-action retrospective
cohort from the frozen public-transfer v6 protocol. Its causal prediction
artifacts contain frame-prefix diagnostics, a released-checkpoint fallback, a
frozen object-scale candidate, and target-mesh CD-UL1 errors.

All 116 public poking takes were used by earlier project work: 36 source actions,
two prospectively sealed references, and these 78 retrospective actions. This
study is therefore a new, outcome-blind **analysis** of already exposed data. It
can establish public real-world relevance, but it cannot establish fresh
prospective confirmation.

## Frozen split

Within each of the 18 physical objects, takes are ordered by

```text
SHA256("pokeflex-query-competence-retrospective-v1" || NUL || take_id)
```

The first take trains the risk model, the second selects its threshold, and all
remaining takes form the validation set. This yields 18 risk-training actions,
18 threshold-selection actions, and 42 validation actions. Every stage contains
all 18 objects. No outcome enters the split.

## Candidate, fallback, and features

The candidate is the frozen causal `action_local_state_relative_0.4` correction
at the object-specific scale fixed by the parent protocol. The fallback is the
byte-identical released checkpoint prediction. A rejected frame therefore has
exactly zero policy regret by construction.

The primary score is the disagreement-only mechanism supported by the controlled
ablation. It uses only candidate-versus-fallback correction magnitude and that
correction relative to predicted motion. A preregistered contextual diagnostic
also consumes update quality, force, assignment, conditioning, camera-bias, and
causal support diagnostics. Object identity is not a feature.

Every feature is read from the update for source frame `f-1`; target mesh and
target error at frame `f` are opened only to fit or score the registered stage.
Changing target outcomes without changing prefix diagnostics leaves the route
features byte-identical.

## Gates and inference

Practical harm is candidate CD-UL1 more than 1% above fallback CD-UL1, matching
the previously frozen PokeFlex regression tolerance. Threshold selection and
validation require:

- at least 25% object-balanced coverage;
- accepted frames for at least 12 of 18 objects;
- a 20,000-replicate physical-object cluster-bootstrap 95% upper bound on harm
  no greater than 10%;
- a physical-object cluster-bootstrap 95% upper bound on policy regret below
  zero; and
- exact fallback identity for every rejection.

The source command opens only the 36 risk/threshold artifacts. Validation remains
closed unless the primary disagreement arm passes. Both commands consume a
write-once attempt ledger before reading their stage artifacts, use protocol-bound
absolute output paths, and refuse retries.

## Claim boundary

A pass supports causal one-frame selective geometry prediction on the registered
PokeFlex object/action population. It does not imply an official PokeFlex score,
unseen-object transfer, a fresh prospective certificate, deployment safety, or
state of the art. SORS FEM replay remains a useful independent-backend follow-up,
but its published PokeFlex example covers only ten FoamDice poke sequences and
is not substituted for this 18-object validation.
