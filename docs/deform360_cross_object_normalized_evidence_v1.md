# Deform360 cross-object normalized-evidence diagnostic v1

## Evidence status

This study is **retrospective and non-fresh**. All six Deform360 objects in the
locked cohort have appeared in earlier repository work. The study uses
leave-one-object-out cross-fitting only to test whether update-count-normalized
component evidence transfers across object groups. It is not fresh independent
validation, the official action-conditioned Deform360 task, deployment
calibration, or a state-of-the-art comparison.

The metadata-only preflight in PR #115 established that the mounted cache has no
untouched fresh-object trajectory cohort. This diagnostic therefore uses only 33
explicitly enumerated packed visual-hull archives under the official-root path
`data-7fea8e2/replication-v1/observations`. It never performs generic recursive
archive selection.

## Scientific question

The full-22 PhysTwin study found that cumulative component log evidence collapses
the 15-component endpoint mixture. Strong scalar tempering restored uncertainty,
but the selected temperature sat at the registered grid boundary and slightly
worsened point prediction.

This experiment tests a lower-dimensional hypothesis:

\[
  w_k \propto \pi_k \exp\left(
    \kappa\,\frac{\ell_k}{n_{\mathrm{updates}}}
  \right),
\]

where cumulative component evidence is normalized by the number of causal prefix
updates before applying an effective-evidence multiplier `kappa`.

For each of the six target objects, `kappa` is selected from the other five
objects using equal-object mean one-step Gaussian negative log likelihood. The
target object's outcomes do not influence its own fold selection. The same
object is, however, a source object in other folds; there is no globally sealed
target cohort.

## Locked data contract

Protocol:
`protocols/deform360_cross_object_normalized_evidence_v1.json`

Canonical protocol SHA-256:
`1fea455cd61fa838d94bb4892973ba944d851b894c876087431fab6461a60ad4`

The protocol binds:

- the merged metadata-preflight commit and content inventory;
- the prior NPZ-header inventory used to enumerate compatible archives;
- six exact object IDs;
- 33 exact `sampled_hulls.npz` relative paths and expected frame counts;
- the fixed evidence-scale grid `2^-8 ... 2^8`;
- one-step rolling prediction and object-balanced aggregation; and
- the explicit non-fresh claim boundary.

Every selected archive is SHA-256 hashed when numerical evaluation begins. Any
path, frame-count, packed-offset, key, or finite-value drift fails closed.

## Comparisons

The rolling global-translation diagnostic compares:

1. zero-displacement persistence;
2. latest observed centroid displacement;
3. the historical cumulative-evidence model average;
4. update-count-normalized evidence at `kappa = 1`; and
5. the leave-one-object-out source-selected normalized mixture.

Point metrics are centroid RMSE and symmetric Chamfer RMSE after translating the
current subsampled hull by the predicted displacement. Raw uncertainty metrics
are 90% ellipsoid coverage, Gaussian negative log likelihood, and effective
component count. Episode metrics are averaged first, then objects receive equal
weight.

The point comparison against latest residual uses a paired bootstrap over the
six target objects. The bootstrap is diagnostic because the objects are not
fresh and each fold selects a potentially different `kappa`.

## Decision boundary

No automatic claim is promoted. The result is useful when it identifies whether:

- selected evidence scales lie inside the wide registered grid;
- normalization restores materially more than one effective component;
- raw coverage moves closer to 90% than cumulative evidence; and
- those uncertainty gains preserve or sacrifice point prediction relative to the
  latest-residual baseline.

A positive mechanism result would justify implementing normalized evidence as an
additive provider option and evaluating it on genuinely untouched object/session
groups. It would not by itself establish calibrated or superior deployment
performance.
