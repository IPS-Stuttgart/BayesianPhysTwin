# Deform360 source-only residual-history contract v1

This experimental contract isolates the reusable source-side part of the
covariance-only validation design. It contains **no target roster**, does not
select or replace target objects, and does not authorize target payload access,
prediction, scoring, or claim promotion.

## Frozen semantics

For each opened source object/session, the adapter stores an exact residual
history with shape `(T, N, 3)` and an explicit Boolean validity mask with shape
`(T, N)`. Invalid entries are stored as zero only; temporal, spatial,
nearest-neighbour, camera, and material-identity filling is forbidden.

The covariance-only reference mean is:

1. the caller-owned physical future;
2. plus each material identity's last **valid causal** residual in the opened
   prefix; and
3. unchanged across future horizons.

A material that is absent in the final prefix frame therefore retains its most
recent earlier valid residual. A material with no valid prefix support receives
zero residual correction.

The covariance donor remains separate from the mean and is scaled by the frozen
early/middle/late factors `[8, 16, 16]`. Provider and scoring cameras are split
by complete physical recorder family. Support or covariance rejection returns
the exact caller-owned physical future mean and covariance objects.

## Evidence boundary

Passing the contract tests establishes deterministic construction, explicit
missingness, exact-last-valid mean semantics, covariance-only composition, and
exact fallback. It is implementation evidence only. It does not establish
fresh-object calibration, target accuracy, provider competence, physical-query
benefit, intervention benefit, deployment safety, or state of the art.

The registered fresh study remains governed by issue `#461`, including its
separate source-first information order and sealed target cohort. This module
must not be used to revive the abandoned 24-target draft.
