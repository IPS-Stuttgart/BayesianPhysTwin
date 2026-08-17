# PokeFlex Conservative Shrinkage Target v1

## Purpose

The source-only study selected exactly one correction:
`checkpoint_action_local_state_relative_0.4_residual_scale_0.125`. It improved
all nine opened source objects, with a 1.286% object-balanced CD_UL1 gain and no
unsupported-frame fallback mismatch. This protocol evaluates that unchanged arm
on the eight PokeFlex target objects registered before their held-out `T2` takes
were inspected.

This is the quickest credible direct test of whether Bayesian-PhysTwin can beat
the released PokeFlex Kinect checkpoint and the paper's reported metric values.
The published 6.498 mm CD_UL1 and 0.820 Jaccard values came from an internal test
split that is not reproducible from the public evaluator. A pass is therefore a
metric-reference result on our prospectively registered split, not a claim that
the samples are identical to the paper's test set.

## Causal Boundary

For every prediction at frame `f`, the method uses only Kinect frames `f-5`
through `f-1` and robot history through `f-1`. Prediction emits every contiguous
frame from 6 onward. It does not inspect force at `f` to decide whether to emit a
candidate. The frozen scorer later restricts evaluation to frames whose recorded
force-y at `f` exceeds 3 N, matching the source evaluation.

The target take's upstream-selected template mesh is an allowed task input. No
other mesh is opened during prediction. Unsupported updates return the released
checkpoint vertices byte-for-byte.

## Three-Stage Custody

1. `predict` writes one NPZ and one checksummed seal per target object. Each seal
   binds the source result, protocol, selected arm, implementation revision,
   official checkpoint bytes, template, robot record, camera parameters, and
   every causal depth input.
2. `barrier` validates all eight seals, requires one clean implementation
   revision, and emits a signed all-case barrier. No target deformed mesh is read.
3. `score` refuses to open a target mesh unless the barrier is complete. It also
   rehashes every prediction and requires the same clean Git revision.

The runner is
`scripts/held/run_pokeflex_conservative_shrinkage_target.py`. Prediction and
scoring outputs must live outside the Git checkout.

## Metrics and Gates

CD_UL1 is the mean one-sided nearest-neighbor L1 distance from 10,000
deterministic predicted surface samples to 10,000 target samples, in millimetres.
Volumetric Jaccard follows the official implementation: boolean intersection
volume divided by union volume. The `manifold` trimesh backend is locked. A
boolean failure is reported and fails the direct gate; it is never replaced by a
voxel approximation.

The direct metric-reference gate requires object-balanced CD_UL1 below 6.498 mm,
Jaccard at least 0.820, and valid candidate Jaccard on every scored frame. The
paired gate requires positive CD_UL1 improvement over the released checkpoint,
a 97.5% object-cluster bootstrap upper bound below zero, and no target-object
regression. Both are reported independently.

## Claim Boundary

The target is opened once after this implementation and protocol are committed
and pushed. No target-dependent retry, arm change, scale change, object
replacement, or gate change is permitted. This experiment is independent of
Causal4D and must not access any held-v8 artifact or process.
