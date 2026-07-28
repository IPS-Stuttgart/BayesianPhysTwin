# Action-Supported TAPNext++ V11 Source Protocol

## Question

V10 showed that frame-zero associations were plentiful, but its 2 mm
predicted-motion gate admitted only four of eight opened source cases. A
target-free audit subsequently showed that every case has at least eight
associated identities with sealed graph action support of at least 0.1.

V11 asks whether those identities can be tracked accurately enough to justify
a later, separately locked Bayesian state-update experiment. It does not
construct a state update or score a future prediction.

## Distinct Method

V11 is not a relaxed rerun of V10 and does not change any V10 artifact. It uses
the sealed V10 frame-zero association carrier but selects identities by:

1. association probability of at least 0.5 in at least two selected cameras;
2. sealed graph `action_support >= 0.1`;
3. action-support-weighted graph-mode information, frame-zero view support,
   and spatial diversity.

Predicted displacement has zero weight and is not an eligibility condition.
Action support localizes where an intervention can propagate; it is not a
perception-confidence cue. TAPNext++ visibility, masks, depth agreement,
reprojection, view redundancy, and assignment uncertainty alone determine
observation reliability.

Eight identities are initialized at frame zero and tracked causally through
frame 57 using the official public TAPNext++ checkpoint. Query identities are
never changed or reseeded.

## Correlation And Bias

Two-view observations are retained because strict three-view schedules have
already failed on otherwise valid sources. They are not treated as two
independent measurements:

- covariance uses the existing covariance-intersection treatment;
- a two-view result receives at least fourfold geometric and assignment
  covariance inflation;
- a 5 mm shared metric bias remains explicit;
- association-mixture spread remains in metric covariance; and
- two-view redundancy reduces prior reliability relative to three views.

The physical innovation is never used in prior reliability. This protocol
stops before any robust state likelihood could process that innovation.

## Evidence Order

```text
frozen implementation and gates
-> eight target-free query schedules
-> causal TAPNext++ and multiview prediction seals
-> completeness barrier over all eight source dispositions
-> open released prefix identities
-> one source competence result
```

The complete configuration is
`configs/sota/deform360_action_supported_tapnextpp_source_v11.json`.

## Advancement Gate

The route advances only when every registered condition passes:

- at least six provider predictions;
- at least 75% pooled endpoint support;
- at least six cases with 50% endpoint support;
- at least six scored cases;
- object-balanced RMSE and late RMSE at most 15 mm;
- at least 10% object-balanced improvement over exact persistence; and
- at least five case-level wins over persistence.

A pass authorizes only a separately locked baseline-relative guarded
state-update study. A failure closes this route without tuning queries,
tracker, lifting, covariance, cameras, or gates.

## Claim Boundary

These are already-open source objects. Even a gate pass would establish only
prefix observation-provider competence. It would not establish a
Bayesian-PhysTwin gain, future prediction, calibration, transfer,
confirmation, or state of the art.

No held-v8 artifact or process and no V1 sealed target may be accessed.
