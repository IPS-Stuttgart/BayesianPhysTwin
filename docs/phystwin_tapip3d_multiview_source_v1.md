# Multiview TAPIP3D source competence protocol

## Purpose

This study tests a narrow missing capability in Bayesian-PhysTwin: automatic,
material-attached sparse 3D identity observations with enough support and accuracy
to approach the existing manual-prefix capacity ceiling. It does not alter the
frozen Causal4D claim, inspect held-v8, or claim forecasting or state-of-the-art
performance.

The prior official TAPIP3D control used one RGB-D camera on the already-open
`single_lift_cloth` interaction. It retained 85.49% prefix support but had 10.17 mm
displacement RMSE and therefore failed its frozen competence gate. The new arm is
genuinely distinct: the same frame-zero identities are tracked independently in
three calibrated views and fused in world coordinates.

## Information boundary

- The development case is the already-open `single_lift_cloth` interaction.
- Each prediction may use RGB-D frames `[0, 121)` only.
- The nine released manual identities contribute frame-zero world coordinates
  only when constructing model queries.
- Later manual coordinates are unavailable until every per-view prediction and
  the fused carrier have been sealed and hashed.
- No RGB-D frame after frame 120 is permitted.
- No held-v8 runtime, target, query, score, barrier, or outcome artifact is
  permitted.
- A passing result authorizes only a locked automatic frame-zero graph-query
  study on opened source data. It does not authorize a fresh target.

## View admission

A query is supplied to a camera only when its frame-zero world point projects
inside that camera, has positive sensor depth, and differs from projected depth
by at most 10 mm. This is target-free input QA. Ineligible identities are omitted
from that view rather than replaced using later trajectory evidence.

All views use official TAPIP3D commit
`4cb7e69a1687f67d56ec3e506768f51f2c581b46`, checkpoint SHA-256
`3a9514d526559838e6158360af2b857da596d064b06a94fae7fd3b85134b2b1e`,
native image size, resolution factor 1, six iterations, support grid size 16,
eight threads, and visibility threshold 0.9.

## Fusion

Each camera trajectory is re-anchored at the exact supplied frame-zero identity:

```text
y'_v(t) = q(0) + y_v(t) - y_v(0).
```

For each identity-frame pair, fusion requires at least two valid views. The point
estimate is the deterministic geometric median. Pairs with more than 20 mm
maximum cross-view disagreement are unsupported and retain exact downstream
fallback.

The covariance is deliberately conservative:

```text
Sigma = (0.005^2 + r_max^2) I  [m^2],
```

where `r_max` is the largest view-to-consensus distance. The 5 mm term is a shared
camera-bias floor. It is not divided by view count: duplicating a correlated
camera cannot make the observation more confident. The provider validity mask is
kept separate from the baseline fallback, so unsupported rows cannot inflate
competence metrics.

## Frozen gate

All conditions must pass on the sealed prefix trajectory:

- overall support at least 70%;
- displacement RMSE at most 5 mm;
- frame-zero anchor RMSE at most 2 mm;
- late-third support at least 50%;
- late-third displacement RMSE at most 10 mm;
- at least 20% displacement-RMSE improvement over the sealed camera-0 result on
  their shared identity-frame support.

NEES and 90% ellipsoid coverage are diagnostic only. No covariance calibration is
fit on this one opened trajectory.

## Decision rule

If the conjunction passes, freeze the result and design an automatic graph-query
source study with the existing physical/action-supported guarded update and exact
fallback. If it fails, stop this multiview TAPIP3D feeder without changing view
admission, fusion, or gates on the opened case.
