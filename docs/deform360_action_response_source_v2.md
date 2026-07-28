# Deform360 action-response source v2

## Rationale

V1 safely rejected because the inherited query planner selected graph
identities before cameras were divided into disjoint evidence panels. The
resulting panel support was `4/0/0`, although a target-free audit over all
frame-zero graph nodes found abundant physically moving identities in every
panel.

V2 changes only this frame-zero planning step. It does not change AllTracker,
triangulation, covariance, admission thresholds, the source case, or any
future-information boundary.

## Frozen planner

The opt-in planner is `balanced-physical-response-v2`.

Before any RGB tracking, it:

1. identifies graph nodes whose sealed physical rollout moves by at least the
   existing `0.5 mm` physical-identifiability threshold;
2. exhaustively partitions the eight complete-prefix cameras into three
   disjoint panels;
3. maximizes the weakest panel's count of eligible nodes visible in at least
   two cameras, followed by total support and angular spread;
4. selects 16 unique, geometry-spanning graph identities while guaranteeing at
   least four eligible identities per panel.

All inputs to this planner are available at frame zero or come from the sealed
prediction-only physical rollout. It does not use an AllTracker output, future
object observation, hidden identity, outcome, or metric.

## Decision boundary

This remains an admission smoke, not an accuracy experiment.

- Rejection means the perception path still lacks enough independent,
  action-consistent response evidence and must return the exact baseline.
- Admission means only that candidate inference may begin. It does not
  authorize candidate selection, target scoring, or a state-of-the-art claim.

The next stage after admission is a source-only disjoint-hidden-identity study
with a separately frozen baseline-relative regret guard. No fresh-object
evaluation is allowed before that guard transfers across multiple already-open
source objects.
