# Calibrated capability half-spaces

Let the exact model-side pairwise gap envelope for action `a` against benchmark `b` over a registered affine task family be represented by witness half-spaces

```text
normal[r] @ theta <= offset[r].
```

Let `q_plus >= 0` be a split-conformal correction computed from one scalar nonconformity score per independent calibration object or trajectory. The statistically corrected action region is

```text
normal[r] @ theta <= offset[r] - q_plus
```

for every witness constraint. Thus calibration is an exact inward shift of the model-side polyhedron.

For a centered objective-uncertainty set `U`, the combined condition at center `theta_bar` is

```text
normal[r] @ theta_bar + support_U(normal[r]) + q_plus <= offset[r].
```

For a norm ball of radius `rho`, `support_U(normal) = rho * dual_norm(normal)`. This exposes three additive terms: registered latent-state ambiguity, declared objective ambiguity, and data-derived model discrepancy.

The correction is deliberately clipped below at zero so target-data calibration cannot enlarge a model-side capability region.
