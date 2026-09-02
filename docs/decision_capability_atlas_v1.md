# Decision capability atlas v1

## Purpose

A physical-twin update may determine quotient-class masses while leaving the
complete state ambiguous inside every class. The existing finite-action
certificate answers whether one registered loss admits an exact or
bounded-regret action over every compatible complete belief.

`bayesian_phystwin.decision_capability_atlas_v1` extends that certificate to an
affine family of registered tasks. It answers a larger question:

> Which downstream tasks is this incomplete physical twin qualified to decide?

## Affine task family

For hypothesis `i`, action `a`, and task parameter `theta`, register

```text
L_theta(i, a) = beta[i, a] + phi[i, a] @ theta.
```

The common example

```text
L_(target,risk)(i,a)
  = (predicted_displacement[i,a] - target)^2
    + risk * physical_risk[i,a]
```

is affine in `(target, risk)` after dropping the action-independent `target^2`
term.

At every supplied task point, the atlas computes the exact classwise support
function and exact worst-case regret over all complete beliefs compatible with
the registered quotient masses and positive prior support. No latent
representative, interpolation, or learned task classifier is used.

## Continuous capability region

For action `a` and benchmark `b`, write

```text
g_iab(theta)
  = L_theta(i,a) - L_theta(i,b).
```

The action is epsilon-capable exactly when

```text
sum_c lambda[c] max_{i in c, p[i] > 0} g_iab(theta) <= epsilon
```

for every benchmark action `b`. Because every `g_iab` is affine, each action's
capability region is a convex polyhedron. An exact task-space half-space system
is obtained by choosing one supported witness hypothesis in every
posterior-supported quotient class and imposing the corresponding affine
inequality for every benchmark action. The implementation enumerates that
system for small supports and fails closed before a caller-supplied constraint
limit is exceeded.

For two-dimensional task families, `capability_polygon_2d` clips a registered
rectangular domain by these exact half-spaces. The union over actions is the
decision capability atlas; points outside the union require the caller-owned
fallback.

## API sketch

```python
import numpy as np

from bayesian_phystwin.decision_capability_atlas_v1 import (
    affine_capability_halfspaces,
    affine_decision_capability_atlas,
    capability_polygon_2d,
)

atlas = affine_decision_capability_atlas(
    prior_weights,
    quotient_weights,
    class_index,
    loss_intercepts,
    loss_coefficients,
    task_grid,
    regret_tolerance=0.0,
)

left_region = capability_polygon_2d(
    affine_capability_halfspaces(
        prior_weights,
        quotient_weights,
        class_index,
        loss_intercepts,
        loss_coefficients,
        action_index=0,
    ),
    task_bounds=np.array([[-1.5, 1.5], [0.0, 4.0]]),
)
```

## Claim boundary

The atlas is exact only for the supplied finite hypotheses, positive prior
support, registered quotient masses, affine task-loss family, action set, and
regret tolerance. It does not validate the quotient or task family, identify a
physical state, justify the loss or tolerance, establish out-of-support
validity, calibrate uncertainty, certify safety, or authorize deployment.
