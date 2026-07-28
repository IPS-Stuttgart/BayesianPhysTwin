# Motion-Stratified Sentinel Queries

`bayesian_phystwin.phystwin_sentinel_queries` adds an opt-in nuisance
instrument to the physics-guided active-query path. It does not change the
existing `phystwin_active_queries` planner or any frozen experiment.

## Motivation

A query planner that selects only nodes with large predicted action response
has an important blind spot. A coherent camera or reconstruction bias can make
all selected nodes appear to move in a physically plausible direction. Because
no predicted-static material identity is observed, the update cannot determine
whether it saw local object response or a shared observation shift.

The motion-stratified planner divides one fixed query budget into:

- **active queries**, selected from nodes whose physical rollout predicts
  material response; and
- **sentinel queries**, selected from nodes whose maximum physical motion over
  the allowed prefix is below a declared threshold.

Sentinels are selected for multiview support and spatial diversity. They are
not used as extra state-update evidence. Their role is to estimate an explicit
shared displacement nuisance and to force abstention when that nuisance is
inconsistent.

## Query Planning

```python
from bayesian_phystwin.phystwin_active_queries import (
    PhysicsGuidedQueryConfig,
)
from bayesian_phystwin.phystwin_sentinel_queries import (
    MotionStratifiedQueryConfig,
    plan_motion_stratified_queries,
)

plan = plan_motion_stratified_queries(
    physical_rollout_m,
    projected_pixels_xy,
    predicted_support_probability,
    mode_basis=physical_response_basis,
    tracker_support_probability=prefix_tracker_support,
    active_config=PhysicsGuidedQueryConfig(
        minimum_motion_m=0.002,
    ),
    config=MotionStratifiedQueryConfig(
        total_query_count=8,
        sentinel_query_count=2,
        sentinel_maximum_motion_m=0.0005,
    ),
)
```

The active and sentinel motion regimes must have a nonzero gap. Missing
sentinels are never replaced by extra active queries, and missing active
queries are never replaced by sentinels. A claim-bearing caller must require
`plan.initial_budget_met`; otherwise it retains its exact baseline.

`camera_queries_txy(...)` returns an explicit role for every tracker query, so
downstream code cannot silently treat sentinels as state measurements.

## Shared-Bias Estimate

For sentinel displacement residuals

```text
r_i = observed_i - predicted_i,
```

`estimate_sentinel_common_bias(...)` estimates one shared vector `b`.
Observations in the same declared correlation group are first collapsed into a
single conservative estimate. Correlation-group estimates are then fused by
equal-weight covariance intersection because their cross-correlation is
unknown.

This has three deliberate properties:

1. duplicating a correlated observation block does not reduce covariance;
2. unknown-correlation fusion is not as confident as naive independent fusion;
3. disagreement beyond the configured consistency limit makes the estimate
   unusable.

An admitted bias is removed from active displacements with
`debias_active_displacements(...)`. Its full covariance is added to every
corrected active observation before the existing robust likelihood is applied.
The state innovation is therefore still robustified once by the downstream
belief update.

## Causal And Statistical Boundary

The planner reads only the action-conditioned physical prefix, projected
support, and tracker support available up to each causal decision. It does not
read target trajectories, future observations, or evaluation outcomes.

Sentinels do not solve camera-only identifiability without assumptions. They
introduce the explicit, falsifiable assumption that their physical
displacement is small over the allowed prefix. A missed global response can
therefore be mistaken for observation bias. For that reason:

- sentinel correction remains behind the physical/action support gate;
- an unusable or incomplete sentinel channel means exact fallback;
- accepted updates still require the source-calibrated baseline-regret guard;
- source outcomes must select thresholds before any fresh-object evaluation.

## Required Source Gate

Before prospective use, compare the same raw guarded belief with and without
sentinels on already-open source objects. Advancement requires:

1. higher acceptance of genuinely beneficial updates than the current
   conservative bias-aware arm;
2. no increase in harmful accepted updates;
3. improvement in both disjoint hidden-identity RMSE and Chamfer distance;
4. non-regression on every source object under object-held-out calibration;
5. exact baseline equality whenever either query role or the bias estimate is
   rejected.

Synthetic tests establish mechanism behavior only. They do not authorize a
fresh, held, or state-of-the-art claim.
