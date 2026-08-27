# Cross-action broken-mechanism physicality certificate v1

## Purpose

`CrossActionPhysicalityProtocolV1` is an additive, target-closed analysis layer
for the chronological `CrossActionProtocolV2` experiment. The parent experiment
asks whether a source-admitted guarded physical correction improves a different
held-out action relative to unchanged physics, discrepancy persistence, and
`last_residual`.

The physicality certificate asks a stricter secondary question:

> Does the supported guarded physical prediction also separate from controls
> that preserve the registered target query while deliberately breaking the
> claimed source-to-target physical relation?

This module implements the prospective contract and evaluator requested in
BayesianPhysTwin issue #788. It changes no Causal4D acquisition, parent transport
estimator, action order, target roster, or evidence count. Its presence in the
repository is implementation evidence only.

## Required controls

Every claim-bearing protocol contains exactly four policies and gives each one a
separate content identity.

### `wrong_source_action`

Use the sealed source-prefix evidence from another parent-scored physical
session whose registered source action differs from the current source action.
The held-out target execution and query remain unchanged.

### `wrong_object_session`

Use the sealed source-prefix evidence from another parent-scored physical
session with the same registered source action. This preserves the action
profile while breaking session identity.

### `phase_shifted_source`

Apply a predeclared nonzero temporal shift to the current session's source
prefix. The shift and period are both recorded, and a shift that is zero modulo
the period is rejected.

### `identity_permuted`

Apply a content-addressed material or graph-identity permutation to the current
source evidence. The contract records the permutation identity, dimension, and
number of moved identities. At least two identities must move.

The parent roster must contain at least two source action profiles and at least
two sessions for every source action profile. Otherwise the donor controls are
not prospectively constructible and protocol creation fails.

## Information and lineage boundary

Every `PlaceboConstructionV1` binds:

- the exact physicality protocol and parent chronological information order;
- the exact parent guarded-physical prediction and selected belief;
- the registered source and target execution and action identities;
- one policy implementation identity;
- one source-evidence identity;
- one unique concrete construction-artifact identity; and
- all donor, shift, or permutation parameters needed by that policy.

Donor controls must match the registered donor session's exact source execution,
action, and sealed source-evidence identity. Shift and permutation controls must
use the current parent prediction's exact source-evidence identity. Construction
records and predictions declare that they use only source-prefix information,
were sealed before target access, and used no target outcome.

All placebo predictions share the parent prediction batch, BayesianPhysTwin
revision, scorer, target outcome, and target-access attestation. Target-side
placebo selection is forbidden.

## Exact fallback inheritance

When the guarded physical parent rejects its candidate, each placebo must:

- use `exact_fallback`;
- select the parent's exact baseline belief;
- reuse the parent's exact prediction artifact; and
- receive the identical proper score.

Such a session contributes exactly zero contrast for all four policies. Fallback
sessions therefore cannot create positive physicality evidence or dilute the
parent accepted-update denominator.

When the guarded parent accepts its physical candidate, every placebo must
materialize and select its complete placebo candidate. A placebo cannot obtain a
favorable result by silently falling back.

## Session-level inference

The complete physical session remains the independent statistical unit. For a
registered lower-is-better proper score `S`, define

```text
C[s,p] = S[s,p] - S[s,guarded_physical].
```

Positive contrast favors the guarded physical prediction. The evaluator forms
one session-by-policy matrix and applies the registered
`paired-bonferroni-percentile-bootstrap-lower-v1` procedure:

1. resample complete physical sessions with replacement;
2. use the same sampled session indices for all four policies;
3. compute bootstrap mean contrasts;
4. use one-sided marginal level `(1-confidence)/4`; and
5. report one simultaneous lower bound per policy.

The input table is order-invariant. Frames, coordinates, actions, contacts,
posterior particles, and donor candidates do not increase the effective sample
size.

## Decision tree

`physicality_supported` is returned only when all of the following hold:

1. the exact parent result is `physical_transport_supported`;
2. every parent-scored session has all four placebo scores;
3. every parent fallback inherits identical belief, artifact, and score;
4. at least one guarded physical candidate was accepted; and
5. every simultaneous lower bound is strictly above the frozen separation
   margin.

A tie or failure for any policy returns `physicality_not_supported`. A failed
parent returns `parent_transport_not_supported`; an insufficient parent returns
`insufficient_physicality_evidence`. Those two parent decisions can be recorded
without opening or scoring placebo outcomes and cannot be rescued by placebo
behavior.

## Minimal construction sketch

```python
from bayesian_phystwin_experiments.cross_action_physicality_v1 import (
    CrossActionPhysicalityProtocolV1,
    CrossActionPhysicalityResultV1,
)

protocol = CrossActionPhysicalityProtocolV1(
    parent_protocol=parent_protocol,
    wrong_source_action_policy_id=wrong_action_policy_id,
    wrong_object_session_policy_id=wrong_session_policy_id,
    phase_shifted_source_policy_id=phase_policy_id,
    identity_permuted_policy_id=permutation_policy_id,
    prediction_batch_id=sealed_prediction_batch_id,
    commit_id=bayesian_phystwin_commit,
    scorer_id=proper_scorer_id,
    minimum_sessions=14,
    bootstrap_replicates=10_000,
    bootstrap_seed=20260828,
    confidence_level=0.95,
    minimum_placebo_separation_margin=registered_margin,
)

result = CrossActionPhysicalityResultV1(
    protocol=protocol,
    parent_result=parent_transport_result,
    score_rows=complete_placebo_score_rows,
)
```

The example values are illustrative. The final runtime protocol, policy
implementations, construction records, donor schedule, shift, permutation,
margin, seed, and replicate count must be frozen before confirmatory target
access.

## Claim boundary

A positive result supports bounded separation of the exact registered
source-admitted physical correction from four deliberately broken physical
relations on the exact chronological physical-session roster. It does not
establish a unique physical cause, arbitrary-action or unseen-object
generalization, calibrated raw covariance, real Prob4D provider competence,
Causal4D intervention benefit, deployment safety, or state of the art.
