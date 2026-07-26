# Action-Anchored Prefix State Source Study v1

## Question

The official-Warp Deform360 adapter currently constructs a sparse graph from
one prefix-end visual hull and initializes every node velocity to zero. This
source study asks whether two causal prefix geometries plus measured gripper
motion provide a better physical endpoint state.

The test is motivated by TrackDeform3D, but it is not an evaluation of that
RGB-D tracker. The public repository contains two materially different
behaviors:

- generic `WireTracker` uses end-effector poses to choose camera-detected leaf
  nodes and then preserves those camera positions;
- its dedicated `deform_with_hands` path hard-replaces mapped leaf nodes with
  measured end-effector positions at every frame.

Bayesian-PhysTwin implements a new, narrow adapter around the second idea. No
TrackDeform3D source is copied.

## Source Case

The smoke case is `002-rope-silk`, episode 5 (`lift sides`). It is the first
bimanual source episode of the first filament object in the locked Causal4D
replication ordering. The choice uses metadata, not prediction error.

The trusted staging step extracts visual hulls at frames 7 and 8 from the
already-open source archive. Prediction branches at frame 8 and cannot read
geometry from frame 14 onward. The full robot trajectory is known action input.
No future tactile is used.

This episode was already opened by the failed six-object Causal4D source
backend. The result is therefore post-open source development, even though the
new predictions are hashed before this run's future scoring.

## Estimator

Let adjacent ordered prefix graphs be \(x_{t-1}^{\mathrm{cam}}\) and
\(x_t^{\mathrm{cam}}\), registered contact nodes \(A\), and measured
controller velocities \(v_A^{\mathrm{ctrl}}\). The camera-topology velocity is

\[
v^{\mathrm{cam}} =
\frac{x_t^{\mathrm{cam}}-x_{t-1}^{\mathrm{cam}}}{\Delta t}.
\]

The shared observation-velocity bias is estimated only at independent action
anchors:

\[
\hat b = \operatorname{median}_{i\in A}
\left(v_i^{\mathrm{cam}}-v_i^{\mathrm{ctrl}}\right).
\]

The fused field solves

\[
\min_v
\left\|v-(v^{\mathrm{cam}}-\hat b)\right\|^2
+\lambda v^\top L v,
\qquad
v_A=v_A^{\mathrm{ctrl}}.
\]

Thus camera/hull geometry proposes free-node motion, measured grippers identify
a shared nuisance bias and fix contact-node motion, and the graph prior spreads
the correction. The physical state innovation is formed once. Rejected state
estimates fall back exactly to the existing zero-velocity physical rollout.

## Frozen Arms

1. Exact frame-8 persistence.
2. Existing official Warp with zero initial velocity.
3. Camera-topology velocity with graph smoothing.
4. Harmonic extension of measured contact velocities only.
5. Bias-aware action-anchored velocity.

All Warp arms use candidate 21, the archived leave-one-source selection made
without episode-5 fit scores. The graph, action, contact schedule, solver, and
spring parameters are otherwise identical.

## Gate

The action-anchored arm must:

- pass its target-free speed and orientation gate;
- improve mean future Chamfer by at least 5% against both exact persistence and
  zero-velocity Warp;
- degrade neither late comparator by more than 5%;
- keep p99 relative edge strain at or below 0.5.

Failure stops this route. Passing justifies a multi-episode source panel, not a
fresh target or state-of-the-art claim.

## Commands

```bash
python scripts/remote/run_deform360_action_anchored_state_source.py stage \
  --config configs/sota/deform360_action_anchored_state_source_v1.json \
  --data-root /path/to/replication-v1 \
  --output-root /path/to/action-anchored-source-v1

python scripts/remote/run_deform360_action_anchored_state_source.py predict \
  --config configs/sota/deform360_action_anchored_state_source_v1.json \
  --data-root /path/to/replication-v1 \
  --causal4d-root /path/to/Causal4D \
  --official-phystwin-repo /path/to/PhysTwin \
  --output-root /path/to/action-anchored-source-v1 \
  --device cuda:0

python scripts/remote/run_deform360_action_anchored_state_source.py seal \
  --config configs/sota/deform360_action_anchored_state_source_v1.json \
  --output-root /path/to/action-anchored-source-v1

python scripts/remote/run_deform360_action_anchored_state_source.py evaluate \
  --config configs/sota/deform360_action_anchored_state_source_v1.json \
  --data-root /path/to/replication-v1 \
  --causal4d-root /path/to/Causal4D \
  --output-root /path/to/action-anchored-source-v1 \
  --output /path/to/action-anchored-source-v1/result.json
```
