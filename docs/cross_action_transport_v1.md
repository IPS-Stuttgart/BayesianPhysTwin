# Cross-action physical-transport evidence v1

## Purpose

A residual can improve the continuation of the action on which it was observed
without identifying physical state or material parameters. The stronger test is
**held-out action transport**:

1. use only the registered factual prefix under source action `a`;
2. construct a complete physical fallback and all frozen candidate beliefs;
3. commit every prediction before target access;
4. propagate each candidate under target action `b`; and
5. score the sealed prediction against the held-out outcome.

The off-diagonal entries `a != b` distinguish transport through the physical
model from same-action residual persistence. The v1 implementation lives in
`bayesian_phystwin_experiments.cross_action_transport_v1`. It is deliberately a
source-only research instrument outside the stable wheel and public API.

## Fixed comparison arms

`TransportArm` defines the admissibility ladder:

| Arm | Frozen interpretation |
| --- | --- |
| `physical_fallback` | Unchanged caller-owned physical belief. |
| `last_residual` | Principal matched deterministic persistence comparator. |
| `discrepancy_only` | Predictive discrepancy/readout persistence without a physical-state claim. |
| `state_only` | Source-inferred physical state transported through the target action. |
| `state_parameter` | Source-inferred physical state and parameters transported together. |
| `guarded_physical` | Only source-admitted physical components; rejected cases return exact fallback. Action-local discrepancy is not transported. |

A protocol may register a subset, but it must include `physical_fallback`, one
physical-transport arm, one discrepancy reference, and one distinct matched
comparator. The recommended first experiment registers all six arms.

## Four content-addressed artifacts

### `CrossActionProtocolV1`

The target-blind protocol binds:

- development and calibration roster identities plus the exact target-session roster;
- the action roster and exact ordered action-pair matrix;
- physical query and query Jacobian;
- proper score, grouping rule, interval method, target-access policy, and technical-failure policy;
- model stack and numerical environment;
- candidate, discrepancy, and matched-comparator arms;
- object/session bootstrap seed and replicate count;
- minimum off-diagonal benefit and contrast margins;
- maximum harmful-session fraction; and
- attestations that the method, rosters, and candidate family were frozen before
  target access.

`target_outcomes_used=True` is rejected. Input ordering does not change the
protocol identity.

### `SealedTransportPredictionV1`

Each record binds one complete prediction to:

- one object/session;
- an ordered source-to-target action pair;
- one registered arm;
- exact baseline, candidate, and selected belief identities;
- prediction artifact and source-evidence identities;
- one target-blind prediction-batch identity; and
- the exact source commit.

The selection disposition is one of:

- `baseline_reference` for the physical baseline arm;
- `candidate_selected`; or
- `exact_fallback`.

The contract rejects a fallback record that does not select the exact baseline,
a selected-candidate record that does not select the exact candidate, and any
prediction that used target outcomes.

### `TransportScoreRowV1`

A post-outcome row binds one proper score to the sealed prediction, one target
outcome, one target-access attestation, and one frozen scorer. Target-side method
selection is forbidden.

### `CrossActionTransportResultV1`

The evaluator fails closed unless:

- every object/session contains the complete registered action-pair matrix;
- every action pair contains exactly the registered arms;
- all arms for a pair score the same held-out outcome;
- all arms share the exact physical baseline for that pair;
- every prediction belongs to one target-blind prediction batch; and
- the complete table binds one target-access attestation and one frozen scorer;
- the scored and explicitly excluded sessions cover the complete target roster; and
- all predictions bind one exact BayesianPhysTwin revision.

The evaluator canonicalizes row order and produces a content-addressed result.

## Statistical unit and scores

Let `s` denote an independent physical object/session, `(a,b)` an ordered action
pair, `m` a method arm, and `S` the registered lower-is-better proper score. The
pair-level gain over the unchanged physical fallback is

```text
g[s,a,b,m] = S[s,a,b,physical_fallback] - S[s,a,b,m].
```

Positive gain is beneficial. Repeated actions and action pairs within a session
are nested observations. The evaluator first averages all off-diagonal gains
within each object/session:

```text
G[s,m] = mean over registered (a,b), a != b, of g[s,a,b,m].
```

Only the vector of independent session means `G[:,m]` enters the bootstrap.
Frames, points, tracks, views, taxels, and action pairs are never treated as
independent replicates.

The physicality contrast is

```text
C_discrepancy[s] = G[s,physical_transport_arm]
                   - G[s,discrepancy_reference_arm].
```

The matched-comparator contrast is

```text
C_matched[s] = G[s,physical_transport_arm]
               - G[s,matched_comparator_arm].
```

A harmful object/session has

```text
G[s,physical_transport_arm] < -harmful_session_margin.
```

Deterministic percentile-bootstrap intervals for gains and contrasts are
generated from complete object/session units with the frozen seed, replicate
count, and confidence level. Harmful-session frequency uses a two-sided Wilson
score interval, so zero observed harmful sessions never imply a zero upper risk
bound.

## Decision rule

`physical_transport_supported` is returned only when all registered conditions
hold:

1. the minimum number of independent off-diagonal sessions is available;
2. the lower confidence bound for physical-arm gain exceeds the frozen minimum;
3. the lower bound for the physical-versus-discrepancy contrast exceeds its
   frozen minimum;
4. the lower bound for the physical-versus-matched-comparator contrast exceeds
   its frozen minimum;
5. the upper confidence bound for the harmful-session fraction is no greater
   than the frozen maximum; and
6. at least one off-diagonal physical candidate was actually selected.

A tie, a failed contrast, harmful-session excess, or an all-fallback result gives
`physical_transport_not_supported`. Too few independent sessions gives the
separate `insufficient_off_diagonal_sessions` status. Neither negative status
may be retuned on the same opened target cohort.

## Minimal construction

```python
from bayesian_phystwin_experiments.cross_action_transport_v1 import (
    CrossActionProtocolV1,
    CrossActionTransportResultV1,
    TransportArm,
)

protocol = CrossActionProtocolV1(
    development_roster_id=development_roster_id,
    calibration_roster_id=calibration_roster_id,
    target_roster_id=target_roster_id,
    query_id=query_id,
    query_jacobian_id=query_jacobian_id,
    score_definition_id=score_definition_id,
    grouping_rule_id=object_session_grouping_id,
    interval_method_id=bootstrap_method_id,
    target_access_policy_id=target_access_policy_id,
    model_stack_id=model_stack_id,
    numerical_environment_id=numerical_environment_id,
    technical_failure_policy_id=technical_failure_policy_id,
    action_ids=(action_a_id, action_b_id),
    target_session_ids=target_session_ids,
    registered_arms=(
        TransportArm.PHYSICAL_FALLBACK,
        TransportArm.LAST_RESIDUAL,
        TransportArm.DISCREPANCY_ONLY,
        TransportArm.STATE_ONLY,
        TransportArm.STATE_PARAMETER,
        TransportArm.GUARDED_PHYSICAL,
    ),
    physical_transport_arm=TransportArm.GUARDED_PHYSICAL,
    discrepancy_reference_arm=TransportArm.DISCREPANCY_ONLY,
    matched_comparator_arm=TransportArm.LAST_RESIDUAL,
    minimum_sessions=12,
    bootstrap_replicates=10_000,
    bootstrap_seed=20260824,
    confidence_level=0.95,
    minimum_off_diagonal_gain=registered_gain_margin,
    minimum_discrepancy_contrast=registered_discrepancy_margin,
    minimum_comparator_contrast=registered_comparator_margin,
    maximum_harmful_session_fraction=registered_harm_limit,
)

# Construct SealedTransportPredictionV1 records before target access.
# After one authorized target opening, create TransportScoreRowV1 records.

evaluation = CrossActionTransportResultV1(
    protocol=protocol,
    score_rows=tuple(score_rows),
    target_accounting_id=target_accounting_id,
    excluded_session_ids=tuple(excluded_session_ids),
)
print(evaluation.summary())
```

Action labels belong in metadata. All contract identities are literal lowercase
SHA-256 strings.

## Required experimental procedure

Before target access:

1. freeze complete object/session rosters and exclusions;
2. freeze the exact action matrix, query, score, horizons, group weighting,
   bootstrap, and margins;
3. freeze the physical baseline, `last_residual`, discrepancy candidate, and at
   most one claim-bearing physical candidate;
4. bind query identifiability, estimability, and nonlinear-closure evidence for
   any physical state or parameter interpretation;
5. construct every complete prediction and publish one prediction-batch seal;
   and
6. verify that all rejected candidates select the exact baseline belief.

After the one authorized target opening:

1. score the already sealed prediction batch without changing the scorer;
2. retain every technical failure and preregistered exclusion;
3. bind every scored or preregistered excluded session in the
   target-accounting artifact;
4. construct the complete score table and evaluation artifact; and
5. report the registered positive, negative, or insufficient result without
   target-informed replacement or threshold changes.

## Relationship to Prob4D and Causal4D

Prob4D may provide uncertainty-bearing observations and shared gauge factors,
but it does not choose the physical query, the transported physical component,
or the action-transport decision. Provider competence remains a separate gate.

Causal4D owns intervention abduction and counterfactual prediction. This
BayesianPhysTwin instrument can provide a separately registered upstream
physical-transport result, but it must not alter or delay Causal4D's frozen
36-execution primary protocol. A positive downstream result cannot rescue a
failed upstream transport, provider, or calibration gate.

## Scientific boundary

A positive evaluation is bounded evidence that one exact source-inferred
physical candidate transports across the registered held-out action matrix on
the registered object/session cohort. It is not evidence of:

- a unique data-generating physical cause;
- arbitrary-action or arbitrary-object generalization;
- calibration of the operational raw posterior;
- universal safety from exact fallback;
- real Prob4D provider competence;
- completed Causal4D physical evidence;
- closed-loop control performance; or
- general deformable-object state of the art.
