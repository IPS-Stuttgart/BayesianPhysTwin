# Deform360 independent normalized-evidence evaluation v1

## Status

The protocol is locked before numerical payload access. The current workflow
performs only a names-and-directory inventory on `workstation2`, validates the
implementation with synthetic contracts, and writes the exact metadata needed
to commit a cohort selection. It does **not** open Deform360 arrays, videos,
images, point clouds, calibration files, or reserved target outcomes.

Protocol SHA-256:

```text
744ac7281f590c8d158c10632bb864b4141bddf370b757ae5f56f93aadfbefc9
```

The numerical calibration and confirmation stages are intentionally absent until
a selection artifact derived from that names-only inventory has been committed
and hash-bound.

## Scientific question

The released PhysTwin study showed that cumulative component log evidence treats
strongly correlated frames as too many independent observations. A scalar
temperature restored mixture uncertainty, but its source-selected value lay at
the search boundary and slightly worsened point prediction.

This experiment tests the parameter-free follow-up on physically independent
public Deform360 objects:

```text
normalized component score
    = cumulative component log evidence
      / supported prefix-observation count
```

The normalized score is used only to reweight the frozen 15-component robust
endpoint bank. No process noise, observation noise, prior, guard, or target-side
temperature is fitted.

## Cohort boundary

The historical Deform360 source, calibration, and reserved target objects remain
excluded. In particular, the twelve target objects sealed by the failed
bias-aware prospective-v2 calibration are not eligible and must remain unopened.

Only the `sheet` and `volumetric` name-only strata are evaluated. A fresh
filament claim is impossible under the historical exclusion contract: every
filament candidate except `181-belt` has already been opened or reserved. Rather
than reuse those objects or manufacture a one-object stratum, v1 makes no
filament claim.

Within each eligible stratum, SHA-256 ranking selects:

- six calibration objects;
- six separate confirmation objects;
- one metadata-ranked episode per object; and
- up to eight exact object/episode-scoped numerical archive paths.

Selection uses names and directory structure only. No object is replaced after
any numerical payload is opened.

## Staged information order

```text
commit protocol and implementation
-> run names-only inventory
-> commit exact calibration/confirmation selection
-> open calibration objects only
-> serialize and hash group scales
-> authorize confirmation only if calibration support passes
-> open confirmation objects only
-> publish object-balanced paired evidence
```

Calibration requires at least nine supported objects and at least four in each
of the two declared strata. Confirmation uses the same minimum. Unsupported
objects are retained as attrition without replacement.

## Released representation adapter

The numerical stage accepts only explicitly unit-declared NPZ contracts:

1. fixed-identity trajectories with a `(T, N, 3)` metre or millimetre key such
   as `positions_world_m`; or
2. packed visual hulls with `frame_indices`, `point_offsets`, and
   `points_world_m`.

There is no magnitude-based unit inference. Every opened archive path and file
SHA-256 is retained in the result.

## Evaluation

For horizons 1, 2, 4, and 8 frames, the causal prefix contains only historical
horizon-matched displacements ending no later than the prediction frame. The
registered methods are:

- zero-displacement persistence;
- the last supported displacement;
- the historical cumulative-evidence model average; and
- mean-log-evidence-per-observation normalization.

Object-balanced kinematic RMSE and symmetric Chamfer RMSE are reported. The two
model averages also report 3-D NEES, Gaussian negative log score, 90% coverage,
and effective component count.

Calibration objects determine monotone horizon-wise covariance multipliers in
two forms:

- an across-object conformal quantile of each object's registered 90th-percentile
  NEES ratio; and
- a stricter across-object conformal quantile of each object's maximum NEES
  ratio.

The calibration artifact is written and hash-verified before the confirmation
process starts.

## Claim boundary

A successful run supports only external object-held-out transfer of normalized
endpoint evidence on the selected public Deform360 sheet and volumetric
representations. It is not official Deform360 Table-4 parity, not a physical
simulator-state correction, not a filament result, and not a state-of-the-art
claim.
