# Deform360 projected-observability source v4

## Purpose

The frozen V3 view-space certificate removed dynamic triangulation from
admission but rejected on the already-open shoe case. Two cameras showed
strong directional agreement, while several selected cameras contained almost
no translation-invariant projected physical motion. V4 changes only the
target-free camera and material-query planner.

The question is:

> When every camera is queried on material identities that the sealed physical
> rollout predicts will be observable in that camera, does action-aligned RGB
> response transfer across distinct source objects?

No candidate state correction or hidden future outcome is part of this stage.

## Frozen planner

The opt-in planner is `projected-observability-v4`. It retains V2's
frame-zero/global-response construction of three spatially spread camera
panels. Before any RGB tracker is run, it then:

1. projects the sealed physical rollout at frames `0`, `19`, `38`, and `57`
   into the candidate cameras;
2. converts pixel displacement to metric camera-tangent displacement using
   frame-zero depth and focal length;
3. removes the median shared translation independently in every camera and
   frame;
4. marks a camera/material pair eligible only when its remaining RMS response
   is at least the unchanged `0.5 mm` physical-identifiability threshold;
5. selects at most eight cameras spanning all three spatial panels;
6. solves a deterministic 16-identity multicover so every selected camera has
   at least four eligible identities.

Selection maximizes camera count first, then weakest-camera support,
cross-camera identity overlap, total support, and projected response. Input
camera order and duplicate storage cannot change the content-addressed plan.
A coherent projected translation alone is a nuisance and cannot create
eligibility.

After planning, V4 uses the unchanged V3 path:

- exact causal AllTracker prefixes through each update only;
- reverse-prefix cycle error for association;
- residual-independent binary source reliability;
- metric covariance from a `2 px` standard deviation with `4x` inflation;
- one robust action-response likelihood;
- unchanged action-response admission thresholds.

## Source panel

The seven cases in
`configs/sota/deform360_projected_observability_source_v4.json` are distinct
already-open source objects with sealed physical carriers. The V3 shoe case is
excluded. Every physical manifest states `partition=source`, binds only
frame-zero object observation plus known robot action, and forbids held-v8
access.

The runner may read RGB only through frame `57`. It may not read a future
object point cloud, future material identity, target metric, tactile future,
held-v8 artifact, or sealed V1 target.

## Advancement rule

Candidate belief construction is authorized only if at least **five of seven**
source objects pass the unchanged admission certificate. Five distinct object
groups are the minimum useful support for a later object-held-out,
baseline-relative regret bound.

- Fewer than five admissions stops this planner family.
- Five or more admissions authorize only an already-open source evaluation of
  a bias-aware state update.
- Fresh-object evaluation additionally requires a frozen regret upper bound,
  disjoint hidden identities, zero target-tuned choices, and bit-exact
  fallback.

No result from this stage can support an accuracy, calibration, safety, or
state-of-the-art claim.
