# Cross-Provider Guarded Field V1

## Status

This is a post-open, one-case source capacity smoke on
`single_lift_cloth`. The implementation, staged identity split, input hashes,
and gates are frozen before the prefix guard is evaluated. The future artifact
may be read only after the prediction is sealed.

A pass authorizes only a separately locked opened-source panel. It is not a
deployable method, independent confirmation, or state-of-the-art evidence.

## Question

Can a competent sparse causal tracker make a dense visual displacement field
safe enough to improve PhysTwin on identities that neither tracker observed
for selection?

This deliberately does not repeat the failed rank-4 state-reset experiment.
The proposed correction remains in readout space, where the released-case
localization audit found the strongest predictive baseline.

## Method

The frozen TAPNext++ plus CoTracker3 complement supplies four sparse provider
identities on frames `[68,88)`. CoTracker3 supplies one dense multiview
displacement block over the same interval. Their roles are different:

1. The sparse provider estimates a tracker-relative displacement gauge and
   forms a sparse-only comparator.
2. The dense block proposes a rank-4 graph-smooth field.
3. Two different manual identities on frames `[88,121)` decide whether that
   field transfers.
4. Three remaining identities are inaccessible until the sealed future score
   on frames `[121,173)`.

The three identity sets are:

| Role | Identities |
| --- | --- |
| Sparse provider | 3, 4, 6, 8 |
| Prefix validation | 1, 5 |
| Future score | 0, 2, 7 |

Validation identities are selected by finite prefix availability only. Future
values and future availability are not used. All identity-to-node associations
are fixed at frame zero and lie within 3.82 mm, below the locked 5 mm limit.

## Causal Gauge

Absolute tracker positions are not compared. For each provider identity, both
trackers are differenced from their first prior common row. The median
dense-minus-sparse displacement over the last five common rows estimates the
relative gauge. At least three provider identities must support this estimate.

This removes an arbitrary absolute offset without using the physical-state
innovation to declare a camera observation reliable.

## Correlation

Accepted dense rows form one correlated observation block. Camera-only quality,
validity, camera count, and reprojection agreement determine relative row
weights, but the total information mass is fixed to graph rank. Duplicating a
pixel block therefore cannot increase confidence.

The dense-versus-physical displacement innovation enters once through robust
graph projection. The resulting field is capped at 10 mm and held fixed in
readout space after frame 88.

## Gates

The prefix guard requires the dense field to improve vector RMSE over both:

- the unchanged selected physical baseline; and
- the sparse-provider-only graph field.

Each comparison must improve by at least 5% and 0.25 mm. Any support or
validation failure returns the selected physical baseline byte-for-byte.

After sealing, the candidate must improve both future Chamfer distance and
future manual-track error on the disjoint scoring identities. Failure closes
this arm without tuning on the case. Passing only permits a new fixed-method
source panel.

## Information Boundary

The runner has four explicit commands:

```text
stage -> predict -> seal -> score
```

`predict` loads only the dense cues, sparse provider, physical baseline, graph
basis, and prefix-validation artifact. It has no code path to the future-score
artifact. `score` refuses to run until the report and prediction archive match
the seal.

The immutable contract is
`configs/sota/phystwin_cross_provider_guarded_field_v1.json`.
