# Physics-Guided Active Graph Queries

`bayesian_phystwin.phystwin_active_queries` retains the original deterministic
physics-guided planner. The versioned
`bayesian_phystwin.phystwin_active_queries_v2` surface adds a streaming causal
interface, nuisance-marginal information scoring, and a content-addressed plan
artifact without changing the historical v1 behavior.

Both planners stop at query selection. Bias-aware triangulation,
identifiable-state restriction, grouped robust likelihoods, nonlinear closure,
and the exact-baseline regret guard remain owned by their existing modules.

## Causal Inputs

The v2 planner accepts only information available before or during the allowed
prefix:

- an action-conditioned physical rollout with shape `(T, N, 3)`;
- per-camera projected pixels and predicted support probabilities;
- an optional low-rank physical response basis, such as
  `PhysicalResponseBasis.basis`;
- an optional nuisance basis for shared gauge, camera, or reconstruction modes;
- optional per-view observation precision predicted without target outcomes;
- the known contact position; and
- tracker support from the current prefix frame.

It does not accept target trajectories, future residuals, evaluation errors, or
benchmark labels. Predicted rollout and visibility may span the complete allowed
prefix because they are generated from the action and physical prior.

`PhysicsGuidedQueryPlannerV2.step(...)` accepts only one `(C, N)`
tracker-support slice and advances exactly one frame. A caller therefore cannot
provide later tracker support to an earlier reseeding decision through the
streaming API. The batch helper internally feeds the same state machine one frame
at a time.

## Eligibility Rule

A graph identity is eligible only when it:

1. moves by at least `minimum_motion_m` in the remaining physical rollout;
2. has independent support from at least `minimum_camera_support` cameras at the
   seed frame;
3. lies outside the configured contact exclusion region; and
4. has finite physical and image coordinates.

The planner never weakens these conditions to fill a query budget.

## Nuisance-Marginal Information

Eligible identities are selected greedily using a deterministic combination of:

- predicted motion magnitude;
- expected multiview support over the remaining prefix;
- incremental physical-mode information after marginalizing declared nuisance
  modes;
- spatial distance from already active queries; and
- distance from the contact region.

For selected queries, let the accumulated expected information be partitioned as

```text
I = [ A  B ]
    [ B' D ]
```

where `A` contains physical-mode information, `D` contains nuisance-mode
information, and `B` contains their coupling. Candidate information is scored by
the increase in

```text
log det(A - B D^-1 B')
```

with separate positive regularization for the physical and nuisance blocks. A
query whose apparent physical response is confounded with a declared nuisance
mode therefore contributes less marginal physical information, while an
orthogonal response remains informative.

Per-view precision and predicted support are averaged across independently
supported cameras rather than summed. This conservative aggregation prevents a
larger number of similar views from automatically multiplying information. The
full downstream likelihood still owns correlation-group and covariance
semantics.

When no nuisance basis is supplied, the criterion reduces to physical-mode
log-determinant scoring. Ties are resolved by the smallest graph identity, so
candidate-array ordering cannot change a frozen plan.

## Streaming Seeding and Reseeding

The frame-zero batch is available as `planner.initial_step`. Subsequent tracker
support is consumed causally:

```python
from bayesian_phystwin.phystwin_active_queries_v2 import (
    PhysicsGuidedQueryConfigV2,
    PhysicsGuidedQueryPlannerV2,
)

planner = PhysicsGuidedQueryPlannerV2(
    physical_rollout_m,
    projected_pixels_xy,
    predicted_support_probability,
    mode_basis=physical_response_basis.basis,
    nuisance_basis=shared_bias_basis,
    observation_precision=predicted_precision,
    contact_position_m=contact_position_m,
    config=PhysicsGuidedQueryConfigV2(query_count=8),
    source_revision=source_revision,
    support_model_id="multiview-support-v1",
)

submit_queries(planner.initial_step)
while not planner.done:
    support_now = tracker_support_for_frame(planner.next_frame)
    submit_queries(planner.step(support_now))
plan = planner.finalize()
```

After `reseed_patience_frames` consecutive frames below the hard multiview
threshold, the unsupported identity is retired. A new, previously unused graph
identity is selected at the current frame. The query event records which identity
it replaces. If no safe replacement is immediately available, that replacement
lineage is retained until a later prefix frame becomes usable.

A batch wrapper is also available for controlled compatibility paths:

```python
from bayesian_phystwin.phystwin_active_queries_v2 import (
    plan_physics_guided_queries_v2,
)

plan = plan_physics_guided_queries_v2(
    physical_rollout_m,
    projected_pixels_xy,
    predicted_support_probability,
    mode_basis=physical_response_basis.basis,
    nuisance_basis=shared_bias_basis,
    tracker_support_probability=prefix_tracker_support,
    contact_position_m=contact_position_m,
    config=PhysicsGuidedQueryConfigV2(query_count=8),
    source_revision=source_revision,
    support_model_id="multiview-support-v1",
)

node_ids, queries_txy, replaced_ids = plan.camera_queries_txy(camera_index=0)
```

Different cameras may receive different row counts because a query is emitted
only in independently supported views. `node_ids` preserves cross-camera graph
association. `replaced_ids == -1` denotes a seed that does not replace a retired
identity.

## Content-Addressed Plan Artifact

A finalized `PhysicsGuidedQueryPlanV1` binds:

- the complete v2 planner configuration;
- source revision and support-model identity;
- digests of the physical rollout, projections, predicted support, physical and
  nuisance bases, observation precision, candidate IDs, and contact positions;
- a streaming digest of every tracker-support slice actually consumed; and
- all seed, camera, score, and replacement arrays.

The artifact is non-pickled and revalidated on load:

```python
from bayesian_phystwin.phystwin_query_plan_v1 import (
    load_physics_guided_query_plan_v1,
    save_physics_guided_query_plan_v1,
)

save_physics_guided_query_plan_v1("query_plan.npz", plan)
reloaded = load_physics_guided_query_plan_v1("query_plan.npz")
assert reloaded.artifact_id == plan.artifact_id
```

Changing a consumed support slice changes the artifact identity even when the
selected events happen to remain the same. Payload tampering is rejected during
load. Claim-bearing runs should record the plan artifact ID in their
`RunManifestV2`.

## Safety Boundary

When too few safe candidates exist, the planner returns a short or empty initial
batch and `initial_budget_met` is false. Downstream code should treat that as an
abstention signal and retain the unchanged PhysTwin baseline. It must not repair
the shortfall with single-view triangulation, target-derived query placement, or
weaker support thresholds.

A complete guarded path is therefore:

```text
action + PhysTwin rollout
        -> content-addressed nuisance-aware query plan
        -> causal multiview tracker observations
        -> bias-aware identifiable-state update
        -> nonlinear closure and source-calibrated regret guard
        -> updated rollout or exact baseline
```
