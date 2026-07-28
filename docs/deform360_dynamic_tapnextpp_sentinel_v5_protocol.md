# Dynamic TAPNext++ Sentinel V5 Source Protocol

## Status

This is a post-open source-development study. It is not a prospective target
evaluation, confirmation result, or state-of-the-art claim. The V1 sealed
target cohort and all held-v8 artifacts remain outside the information
boundary.

## Motivation

The V3 dynamic-query provider demonstrated strong prefix tracking on one
opened source case but selected a physical continuation that was much worse
than exact persistence on disjoint hidden identities. Camera-internal
consistency cannot distinguish true coherent object motion from coherent
multiview bias. The V5 hypothesis is that a fixed subset of physically
near-static graph identities can expose this nuisance term.

## Frozen Arm

The frame-zero query budget is fixed at 12 identities:

- 9 active identities with at least 2 mm predicted prefix motion;
- 3 sentinel identities with at most 0.5 mm predicted prefix motion;
- at least three predicted visible views for every identity;
- one target-free eight-camera panel;
- no reseeding or transfer of unused budget between roles.

All identities are tracked causally from frame 0 through frame 57. Sentinel
residuals are the observed displacement minus the physical-prefix
displacement. Because every sentinel uses the same tracker and camera panel,
all sentinel identities form one unknown-correlation group. Their covariance
does not shrink as if they were independent samples.

An update is admitted only when all three sentinels have endpoint support and
their common-mode residual is coherent under the frozen four-sigma test. The
estimated bias is subtracted from active displacements, and its full covariance
is added to every active observation. Sentinel identities are never used as
state measurements. Any schedule, support, or coherence failure produces
bit-exact persistence.

## Information Boundary

Query planning reads only:

- the selected physical rollout through frame 57;
- its graph basis;
- camera calibration and image dimensions.

Provider inference reads RGB, depth, and masks only through frame 57. No
future object observation, future manual identity, target metric, sealed V1
case, or held-v8 artifact may affect a prediction.

## Source Gate

The first smoke case is the already-open source case that supported the V3
provider. Before any hidden-identity outcome is read, the implementation must:

1. fill the complete 9-active/3-sentinel schedule;
2. produce replayable, checksummed source artifacts;
3. either admit a coherent sentinel bias or retain bit-exact persistence;
4. pass all unit, lint, and static-type checks.

Advancement to additional already-open source cases requires the candidate to
improve both hidden-identity RMSE and hidden Chamfer over:

1. exact persistence; and
2. the selected physical-or-persistence backbone.

Observed query identities and all 12 sentinel/active identities are excluded
from future scoring. A smoke failure closes this fixed V5 arm. A smoke pass
permits only a frozen opened-source transfer run. Fresh-object evaluation
requires a separate preregistration and a disjoint physical-object cohort.

## Interpretation

A positive result would support a bias-aware online state update, not the
claim that TAPNext++ alone beats the physical model. A rejection would show
that three low-motion sentinels are insufficient under this query budget or
that the remaining error is not well represented by one shared displacement
bias. Either outcome leaves the historical V3 and V4 evidence unchanged.

## Runtime Amendment 1

The first source execution reached the rejection assertion before writing any
result artifact or reading any hidden-identity outcome. The assertion compared
the full candidate trajectory with persistence. The unchanged assimilation
backend intentionally preserves its physical prefix and guarantees exact
persistence only after the causal branch frame. The assertion was therefore
narrowed to frames after frame 57, and a regression test now verifies that
forecast-only identity. No query, covariance, support, bias, assimilation, or
evaluation setting changed.
