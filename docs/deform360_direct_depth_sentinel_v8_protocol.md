# Direct RGB-D Sentinel V8 Source Protocol

## Status

This is a new post-open source-development arm. V7 showed that 265 of 452
physical graph nodes had depth/mask association candidates in at least three
views at both prefix endpoints, while TAPNext++ retained only four of nine
active identities and none of three sentinels. V8 tests whether the causal
RGB-D endpoint evidence is useful without a learned material-identity carrier.

V8 is not a target evaluation, confirmation, or state-of-the-art claim. The V1
sealed cohort, fresh-object locks, and every held-v8 artifact remain outside
the information boundary.

## Frozen Hypothesis

At frames 51 and 57, associate each scheduled physical graph identity directly
with the local object-mask and metric-depth candidates in the same selected
eight-camera panel. Fuse at least three supported views conservatively, form a
birth-anchored displacement, estimate one common endpoint bias from the three
low-motion sentinels, and feed only the nine debiased active measurements into
the existing robust Bayesian update.

This arm removes TAPNext++ entirely. It keeps the V7 camera panel, graph basis,
query budget, motion strata, support screen, pairwise correspondence gate,
robust mixture likelihood, RBF update, and bit-exact fallback.

## Observation and Uncertainty Contract

For each endpoint, camera, and scheduled identity:

1. project the physical identity into the selected view;
2. retain the frozen 12-pixel depth/mask candidate set;
3. use the assignment distribution to obtain a candidate pixel and pixel
   covariance;
4. propagate pixel and depth variance into world-coordinate covariance in
   square metres;
5. require at least three camera observations;
6. combine views by equal-weight covariance intersection because their
   correlation is unknown;
7. add the maximum between-view scatter as a common-mode covariance bound.

Association probability and observation reliability are distinct. Candidate
geometry may set association probability, but the residual against the
PhysTwin state does not define prior reliability. The state innovation enters
exactly once, through the existing robust mixture likelihood. Duplicating an
identical camera cannot reduce covariance.

The temporal displacement covariance is frozen as

```text
2 * (birth endpoint covariance + update endpoint covariance).
```

The multiplier is a conservative bound for unknown temporal correlation. No
dense pixel count is treated as an independent sample.

## Sentinel and Update Contract

- birth frame: 51;
- update frame: 57;
- active identities: 9;
- sentinel identities: 3;
- active physical displacement: at least 2 mm;
- sentinel physical displacement: at most 0.5 mm;
- endpoint support: at least three cameras at both endpoints;
- query reseeding: disabled;
- unused role-budget transfer: forbidden.

All three sentinels must be supported. Their displacement residuals belong to
one unknown-correlation group. The frozen four-sigma consistency rule must
admit a common bias. The bias is removed from active displacements and its full
covariance is added to each active covariance. Sentinels are never state
measurements.

The active pairwise gate still requires at least nine mutually consistent
measurements. Any incomplete schedule, endpoint support failure, sentinel
rejection, active-consensus failure, or robust update rejection yields the
bit-exact selected physical-or-persistence backbone.

## Information Boundary

The provider may use:

- sealed physical and persistence predictions through frame 57;
- calibrated camera geometry;
- object masks and metric depth at frames 51 and 57;
- the target-free complete-camera prefix certificate through frame 57.

RGB is decoded only by the pre-existing complete-camera admission certificate;
it does not enter V8 prediction. No frame after 57, future material identity,
future point cloud, manual future track, target metric, sealed V1 object, or
held-v8 artifact may be read.

## Source Execution Order

1. Build the unchanged target-free camera panel.
2. Seal the binary endpoint-support screen.
3. Seal the 9-active/3-sentinel schedule within the eligible graph nodes.
4. Build and seal direct RGB-D endpoint beliefs and metric covariance.
5. Apply the sentinel gate and produce the complete prediction artifact.
6. Stop without hidden scoring if the result is exact fallback.
7. Otherwise score only future identities disjoint from all 12 query
   identities.

The smoke case is the already-open shoe source case used by V5--V7. Its hidden
future may be opened only after the complete V8 provider and prediction
artifacts are sealed.

## Advancement Gate

V8 advances beyond the smoke case only if it:

1. fills all 12 query roles;
2. supports all three sentinels at both endpoints;
3. supports all nine active identities at both endpoints;
4. admits a coherent common-bias estimate;
5. passes the unchanged nine-inlier active pairwise gate;
6. improves disjoint hidden-identity RMSE over exact persistence;
7. improves disjoint hidden Chamfer over exact persistence;
8. improves both metrics over the selected physical-or-persistence backbone.

A tie fails. A failure closes this exact arm. No radius, endpoint, view count,
motion threshold, query count, covariance rule, reliability mapping, gate, or
assimilation setting may be changed afterward on this case.

A smoke pass permits only immutable transfer across already-open source cases.
An independent evaluation requires a separately preregistered fresh-object
cohort, a hash-only exclusion manifest, and a sealed prediction barrier.
