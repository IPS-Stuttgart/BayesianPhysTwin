# MatPhys surface-UQ source protocol v1

## Question

This source-only study asks whether target-excluded disagreement among official
MatPhys material proposals contains useful uncertainty information when the
point prediction itself remains the frozen DEFORM result. It does not ask
whether MatPhys replaces DEFORM as the best mean predictor.

The candidate and every comparator therefore use the exact same `prediction_m`
bytes. Only the predictive covariance changes. This protects the existing
DEFORM result and makes an NLL, coverage, or selective-risk improvement
attributable to uncertainty rather than to a hidden mean change.

## Source denominator

The complete denominator is the ten previously opened Deform360 development
cases in
[`matphys_surface_uq_source_v1.json`](../configs/sota/matphys_surface_uq_source_v1.json).
No case may be replaced. Ordinary predictions, exact-mean fallbacks, retained
technical failures, and unscorable outcomes are reported separately.

These cases are outside the eleven-object MatPhys training universe. Every
MatPhys checkpoint may therefore contribute without training on the evaluated
physical object. The ensemble is replayed through the official PhysTwin Warp
backend, and repeated same-field replays measure numerical simulator noise
separately from between-checkpoint spread.

## Observation boundary

Camera names alone define two disjoint panels before source suffix decoding.
Frame-zero DINO part features use the lexical complement of the twelve scoring
cameras. Future source surfaces are reconstructed only from the scoring panel.
No RGB, mask, depth, Splatfacto, or tracking artifact is shared between the two
panels.

The source endpoint contains frames 58 through 75 of the registered 81-frame
action window. It is a custom disjoint-camera Deform360 reconstruction, not an
official processed annotation or an official Deform360 benchmark score.

## Estimand and comparators

For each future frame, scoring-camera depth is backprojected into the world
frame. Each registered DEFORM mean node is associated with its nearest surface
point, subject to the frozen 50 mm metric limit. The event residual is a 3-D
point-to-surface displacement in metres.

The MatPhys covariance is the target-excluded between-model plus within-replay
covariance for the same graph node. A scalar multiplier is fit on the other
nine source cases, with a fixed 5 mm observation/reconstruction floor. The two
comparators are:

1. a leave-one-case-out isotropic Gaussian around the identical DEFORM mean;
2. a leave-one-case-out radial split-conformal sphere around that same mean.

Metrics are first averaged within each case and then equally across cases.
Coordinates, nodes, cameras, and frames are not treated as independent
experimental replicates.

## Advancement rule

The MatPhys covariance advances only if at least eight cases are scorable, each
scored case retains at least half of its attempted events, and all registered
gates pass: at least six case-level NLL wins against the isotropic comparator,
at least 0.05 nats/event equal-case NLL improvement, 90% coverage between 80%
and 98%, and at least 5% lower mean 90% ellipsoid volume than the conformal
sphere. A source failure leaves DEFORM unchanged and forbids a fresh target run.

Passing this source gate authorizes only a separately locked, genuinely fresh
public-object calibration evaluation. It does not authorize held-v8, DLO4,
DLO5, any revoked covariance cohort, or a SOTA claim.
