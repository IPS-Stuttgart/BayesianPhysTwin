# Physics-Guided Active Graph Queries

`bayesian_phystwin.phystwin_active_queries` implements the target-free front end
for guarded online state updates. It converts an action-conditioned PhysTwin
rollout into graph-identity queries for a causal multiview point tracker.

The implementation deliberately stops at query planning. Bias-aware
triangulation, identifiable-state restriction, grouped robust likelihoods, and
the exact-baseline regret guard remain owned by their existing modules.

## Causal Inputs

The planner accepts only information available before or during the allowed
prefix:

- the action-conditioned physical rollout, with shape `(T, N, 3)`;
- per-camera projected query pixels and predicted support probabilities;
- an optional low-rank physical response basis, such as
  `PhysicalResponseBasis.basis`;
- the known contact position;
- optional tracker support probabilities observed up to the current prefix
  frame.

It does not accept target trajectories, future residuals, evaluation errors, or
benchmark labels. Predicted rollout and visibility may span the complete allowed
prefix because they are generated from the action and physical prior. Tracker
support is consumed sequentially: a decision at frame `t` reads no tracker
support after `t`.

## Selection Rule

A graph identity is eligible only when it:

1. moves by at least `minimum_motion_m` in the remaining physical rollout;
2. has independent support from at least `minimum_camera_support` cameras at the
   seed frame;
3. lies outside the configured contact exclusion region; and
4. has finite physical and image coordinates.

Eligible identities are selected greedily using a deterministic combination of:

- predicted motion magnitude;
- expected multiview support over the remaining prefix;
- incremental log-determinant information in the physical response modes;
- spatial distance from already active queries; and
- distance from the contact region.

Ties are resolved by the smallest graph identity. Candidate array ordering
therefore cannot change a frozen plan.

## Dynamic Seeding and Reseeding

The planner monitors active-query support causally. After
`reseed_patience_frames` consecutive frames below the hard multiview threshold,
the unsupported identity is retired. A new, previously unused graph identity is
selected at the current frame from the physical rollout. The query event records
which identity it replaces. If no safe replacement is available immediately,
that replacement lineage is retained until a later prefix frame becomes usable.

The plan exposes per-camera batches as `[seed_frame, x, y]` arrays:

```python
from bayesian_phystwin.phystwin_active_queries import (
    PhysicsGuidedQueryConfig,
    plan_physics_guided_queries,
)

plan = plan_physics_guided_queries(
    physical_rollout_m,
    projected_pixels_xy,
    predicted_support_probability,
    mode_basis=physical_response_basis.basis,
    tracker_support_probability=prefix_tracker_support,
    contact_position_m=contact_position_m,
    config=PhysicsGuidedQueryConfig(query_count=8),
)

node_ids, queries_txy, replaced_ids = plan.camera_queries_txy(camera_index=0)
```

Different cameras may receive different row counts because a query is emitted
only in independently supported views. `node_ids` preserves cross-camera graph
association. `replaced_ids == -1` denotes a new seed rather than a replacement.

## Safety Boundary

The planner never lowers `minimum_camera_support` to fill a budget. When too few
safe candidates exist, it returns a short or empty plan and
`initial_budget_met` is false. Downstream code should treat that condition as an
abstention signal and retain the unchanged PhysTwin baseline. It must not repair
the shortfall with single-view triangulation, target-derived query placement, or
weaker support thresholds.

A complete guarded path is therefore:

```text
action + PhysTwin rollout
        -> physics-guided query plan
        -> causal multiview tracker observations
        -> bias-aware identifiable-state update
        -> source-calibrated regret guard
        -> updated rollout or exact baseline
```
