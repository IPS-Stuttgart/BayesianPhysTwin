# Trajectory-conformal support-robust decision certificate v1

## Purpose

The exact query-decision certificate is exact over a registered finite physical
support. A new physical trajectory may not be represented by that support. This
module enlarges the registered regret bound with a split-conformal trajectory
radius and returns the exact caller-owned fallback when the enlarged bound is
too large.

This is the first open-world layer above the finite-support certificate. It does
not replace that certificate: it quantifies how much realized decision regret
has exceeded the registered support-wise bound on untouched calibration
trajectories.

## Calibration score

Fix the complete base selection policy before calibration. For calibration
trajectory `j` and decision `d`, let

- `a_jd` be the base action;
- `B_jd(a_jd)` be its finite-support worst-case-regret bound;
- `R_jd(a_jd)` be its realized regret against the best registered action on the
  observed trajectory.

The complete-trajectory nonconformity score is

```text
S_j = max over base nonfallback decisions d
      max(0, R_jd(a_jd) - B_jd(a_jd)).
```

A trajectory containing only fallback decisions has score zero. Frames and
windows are not treated as independent calibration units.

## Split-conformal radius

For `n` exchangeable calibration trajectories and miscoverage `alpha`, define

```text
k = ceil((n + 1) * (1 - alpha)).
```

If `k <= n`, the radius is the `k`th smallest calibration score. If `k > n`,
the radius is positive infinity. The latter is not clipped to the largest
observed score: the operational result is mandatory fallback.

For one new complete trajectory exchangeable with the calibration
trajectories,

```text
P(S_new <= radius) >= 1 - alpha.
```

No independence between decisions or frames inside a trajectory is required.

## Operational rule

For a base nonfallback action with finite-support bound `B`, conformal radius
`q`, and declared operational regret tolerance `epsilon`, execute the base
action only when

```text
B + q <= epsilon.
```

Otherwise return the exact fallback. The wrapper does not select a different
action after calibration. Such reselection would require an all-action score or
new calibration.

On the conformal event, every executed action in the new trajectory has
realized regret no larger than `epsilon`. Therefore,

```text
P(any executed nonfallback decision in the new trajectory has
  realized regret > epsilon) <= alpha.
```

This is a marginal complete-trajectory guarantee for a fixed policy. It is not
pointwise conditional validity.

## Claim boundary

The guarantee requires exchangeability of the new complete trajectory with the
calibration trajectories. It does not establish validity under an unseen
material, object, action family, intervention distribution, or arbitrary domain
shift. It does not validate the finite physical support, quotient, action set,
loss matrix, provider, or fallback. It is not predictive-distribution
calibration, an individual safety guarantee, deployment authorization, or a
replacement for prospective robotic evaluation.

The DEFORM DLO4/DLO5 audit is retrospective because those evaluation outcomes
were already opened by the parent finite-action study. A future cross-object
or DOT protocol must freeze the envelope and action portfolio before opening
its target trajectories to support prospective confirmation.

## API

```python
from bayesian_phystwin.support_robust_decision_certificate_v1 import (
    split_conformal_trajectory_envelope,
    support_robust_action_decision,
    trajectory_policy_regret_excess,
)

score = trajectory_policy_regret_excess(
    realized_loss_by_decision_action,
    finite_support_regret_by_decision_action,
    selected_action_index,
    fallback_action_index=0,
)

envelope = split_conformal_trajectory_envelope(
    calibration_scores,
    miscoverage=0.2,
)

decision = support_robust_action_decision(
    base_selected_action_index=base_action,
    fallback_action_index=0,
    action_count=3,
    finite_support_regret_bound=base_bound,
    conformal_radius=envelope.radius,
    operational_regret_tolerance=declared_epsilon,
)
```
