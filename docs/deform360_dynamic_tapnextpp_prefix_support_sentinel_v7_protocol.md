# Dynamic TAPNext++ Prefix-Support Sentinel V7 Source Protocol

## Status

This is a new post-open source-development arm. It responds to V6's
prefix-only finding that target-free projection visibility did not predict
depth/mask association or tracker survival. It is not a retuning of V6, a
target evaluation, confirmation, or state-of-the-art claim.

No hidden-future outcome from the opened smoke case has been read by V5 or V6.
The V1 sealed cohort, fresh-object locks, and all held-v8 artifacts remain
outside the information boundary.

## Frozen Hypothesis

The fixed branch interval, tracker, camera panel size, query budget, motion
strata, fusion, covariance, robust likelihood, pairwise gate, and exact
fallback remain unchanged from V6. V7 changes only the admissible graph-node
pool:

> Allocate active and sentinel queries only among nodes that have at least
> three causal depth/mask-supported association candidates at both frames 51
> and 57.

The screen is evaluated after selecting the same target-free eight-camera
panel and before allocating query identities.

## Support Screen

For every physical graph node and each endpoint:

1. project the physical node into each selected camera;
2. inspect the frozen 12-pixel local patch;
3. require at least one finite, positive-depth pixel inside the object mask;
4. count cameras with such an association candidate;
5. retain the node only if both endpoint counts are at least three.

The screen uses only the binary existence of an association candidate.
Association probability, entropy, pixel covariance, depth residual, physical
state innovation, tracker confidence, and target metrics do not influence
eligibility.

This distinction is binding:

- candidate geometry may define an assignment distribution;
- assignment probability remains separate from observation reliability;
- the state innovation is processed once by the existing robust mixture;
- endpoint screening must not be interpreted as independent-view confidence.

The complete screen, endpoint support counts, eligible identity list, source
hashes, and information boundary are checksummed before TAPNext++ execution.

## Fixed Query and Update Contract

- birth frame: 51;
- update frame: 57;
- active queries: 9;
- sentinels: 3;
- active physical displacement: at least 2 mm;
- sentinel physical displacement: at most 0.5 mm;
- predicted geometric support: at least three cameras;
- association-support screen: at least three cameras at both endpoints;
- query reseeding: disabled;
- unused role budget transfer: forbidden.

TAPNext++ consumes only frames 51--57. All three sentinels must retain accepted
birth and update support. Their shared displacement residual is fused as one
unknown-correlation group under the frozen four-sigma coherence rule. An
admitted bias is removed from active displacements and its full covariance is
added to every active covariance. Sentinel identities are not state
measurements.

Any incomplete schedule, support failure, inconsistent sentinel field,
insufficient nine-inlier active consensus, or robust-mixture rejection yields
bit-exact persistence after frame 57.

## Source Execution Order

1. Build and seal the prefix-support screen.
2. Build and seal the 9-active/3-sentinel schedule within the eligible set.
3. Run TAPNext++ and multiview fusion through frame 57.
4. Check all support, sentinel, and active-consensus gates.
5. Stop without hidden scoring if the candidate is exact persistence.
6. Otherwise seal the provider and prediction artifacts.
7. Score only future identities disjoint from all 12 query identities.

The smoke case is the already-open shoe source case. Prefix observations may
develop the provider, but hidden-future identities may be opened only once for
the frozen V7 prediction.

## Advancement Gate

V7 advances beyond the smoke case only if it:

1. fills all 12 query roles;
2. supports all three sentinels at birth and update;
3. admits a coherent shared-bias estimate;
4. passes the unchanged nine-inlier active pairwise gate;
5. improves disjoint hidden-identity RMSE over exact persistence;
6. improves disjoint hidden Chamfer over exact persistence;
7. improves both metrics over the selected physical-or-persistence backbone.

A tie fails. A failure closes this exact arm. No support radius, endpoint,
camera count, motion threshold, query count, covariance rule, reliability
mapping, gate, or assimilation setting may be modified afterward on this case.

A smoke pass permits an immutable opened-source transfer run only. Fresh-object
evaluation still requires a separately preregistered cohort, object-hash
exclusion manifest, and sealed prediction barrier.
