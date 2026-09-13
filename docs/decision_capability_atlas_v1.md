# Decision capability atlas v1

## Purpose

A physical-twin update may determine quotient-class masses while leaving the
complete state ambiguous inside every class. The existing finite-action
certificate answers whether one registered loss admits an exact or
bounded-regret action over every compatible complete belief.

`bayesian_phystwin.decision_capability_atlas_v1` extends that certificate to an
affine family of registered tasks. It answers a larger question:

> Which downstream tasks is this incomplete physical twin qualified to decide?

`bayesian_phystwin.decision_capability_task_uncertainty_v1` adds a second
ambiguity layer. It answers whether the same action remains certified when the
task objective itself is not known exactly—for example, when a target location,
force penalty, damage weight, or user preference lies in a registered set.

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
fallback. Overlaps are retained as multi-action admissible regions rather than
resolved by an unregistered tie break.

## Uncertain task objectives

Let one action's exact capability region be

```text
T_a = {theta: n[j] @ theta <= b[j] for every j}.
```

Suppose the task is only known to lie in `center + U`, where `U` is a centered
compact convex uncertainty set. The complete task set is inside `T_a` exactly
when

```text
n[j] @ center + sigma_U(n[j]) <= b[j] for every j,
```

where `sigma_U(v) = sup_{u in U} v @ u` is the support function. This test is
simultaneously robust to every complete physical belief represented by the
quotient and every task objective in the declared uncertainty set.

Two useful specializations are closed form:

```text
axis-aligned box: U = {u: abs(u) <= r}
                  sigma_U(n) = abs(n) @ r

ellipsoid:        U = {G z: ||z||_2 <= 1}
                  sigma_U(n) = ||G.T @ n||_2
```

Subtracting these support values from the half-space offsets gives the exact
polyhedron of task-set centers whose complete translated box or ellipsoid is
certified. This is geometric erosion of the capability region, not grid
sampling.

For a nominal capable task, `norm_ball_capability_margin` returns the largest
radius `rho` such that every task in `||theta-center|| <= rho` remains in the
action region. For task norm `||.||`,

```text
rho = min_j (b[j] - n[j] @ center) / ||n[j]||_*,
```

where `||.||_*` is the dual norm. Outside the capability region, the returned
negative normalized constraint margin is a violation diagnostic; it is not
advertised as distance to the polyhedron.

## API sketch

```python
import numpy as np

from bayesian_phystwin.decision_capability_atlas_v1 import (
    affine_capability_halfspaces,
    affine_decision_capability_atlas,
    capability_polygon_2d,
)
from bayesian_phystwin.decision_capability_task_uncertainty_v1 import (
    box_robust_center_halfspaces,
    box_task_set_capability,
    ellipsoid_task_set_capability,
    norm_ball_capability_margin,
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

left_halfspaces = affine_capability_halfspaces(
    prior_weights,
    quotient_weights,
    class_index,
    loss_intercepts,
    loss_coefficients,
    action_index=0,
)

left_region = capability_polygon_2d(
    left_halfspaces,
    task_bounds=np.array([[-1.5, 1.5], [0.0, 4.0]]),
)

# Every target/risk objective in this box must support the same action.
box_report = box_task_set_capability(
    left_halfspaces,
    task_centers=[[-1.2, 0.2]],
    half_widths=[[0.2, 0.2]],
)

# Equivalently, construct the exact region of admissible box centers.
robust_center_region = capability_polygon_2d(
    box_robust_center_halfspaces(left_halfspaces, [0.2, 0.2]),
    task_bounds=np.array([[-1.3, 1.3], [0.2, 3.8]]),
)

ellipsoid_report = ellipsoid_task_set_capability(
    left_halfspaces,
    task_centers=[[-1.2, 0.2]],
    generators=np.diag([0.1, 0.2]),
)

margin = norm_ball_capability_margin(
    left_halfspaces,
    task_centers=[[-1.2, 0.2]],
    task_norm="l2",
)
```

## Controlled result

The deterministic study keeps the original two-dimensional atlas and adds a
fixed task-objective box with half-width `(0.1, 0.2)`. On the center domain for
which every box remains inside the registered task rectangle, the nominal
certified union covers 83.18% of task centers. Requiring one action to remain
valid for the complete objective box reduces this to 66.33% and expands exact
fallback to 33.67%.

The study also contains a strictness witness: `pull_left` is exactly certified at
the nominal task `(-0.6, 0.1)`, but no action is certified for the surrounding
box with half-width `(0.04, 0.05)`. A point or nominal-only atlas would therefore
commit despite unresolved task preference.

These are deterministic mechanism results. They demonstrate simultaneous
robustness to latent-state ambiguity and declared task-objective uncertainty;
they are not real-provider, model-misspecification, safety, or deployment
evidence.

## Claim boundary

The atlas is exact only for the supplied finite hypotheses, positive prior
support, registered quotient masses, affine task-loss family, action set, and
regret tolerance. The task-set extension is exact only for the supplied
half-spaces and declared box or ellipsoid. It does not validate the quotient,
task family, task-uncertainty set, loss, or tolerance; establish out-of-support
validity; cover physical-model misspecification; calibrate uncertainty; certify
safety; or authorize deployment.
