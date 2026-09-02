# Trajectory-level conformal regret envelope v1

## Purpose

The finite query-quotient certificate is exact over a registered finite physical
support.  The DEFORM gate audit showed that this support can understate realized
held-trajectory regret.  This module adds a separate, data-calibrated envelope
for that support-misspecification error; it does not reinterpret the original
certificate as an open-world guarantee.

For calibration trajectory `j`, decision `t`, and action `a`, let

- `B[j,t,a]` be the registered support-wise worst-case regret bound;
- `R[j,t,a]` be the realized action regret.

The trajectory score is

```text
S[j] = max_{t,a in registered roster} (R[j,t,a] - B[j,t,a]).
```

Given `n` complete exchangeable calibration trajectories and miscoverage
`alpha`, the radius is the order statistic at

```text
ceil((n + 1) * (1 - alpha)).
```

If this rank exceeds `n`, the radius is infinite and the operational rule falls
back.  The radius is clipped below at zero, so finite-data calibration never
makes the registered certificate less conservative.

For a future trajectory, define

```text
U[t,a] = B[t,a] + radius.
```

A nonfallback action is executed only when it is the unique minimum-regret
action and its inflated bound does not exceed a separately declared regret
budget.  Otherwise the exact caller-owned fallback is returned.

## Guarantee and boundary

Under exchangeability of complete calibration and future trajectories, with
marginal probability at least `1 - alpha`, every registered action at every
registered decision of one future trajectory satisfies

```text
realized regret <= inflated regret bound.
```

Consequently, every nonfallback action emitted under that event has realized
regret no larger than the declared budget.  The simultaneous unit is one
complete trajectory, not a frame or action coordinate.

This is not pointwise conditional validity.  It does not validate
exchangeability, establish unseen-object transport, justify the loss or regret
budget, calibrate a probabilistic state estimate, authorize deployment, or
certify safety.  An object-, material-, or action-shift claim requires a
separately registered calibration design for that shift.

## API

```python
from bayesian_phystwin.conformal_regret_envelope_v1 import (
    support_robust_decision,
    trajectory_conformal_regret_envelope,
)

envelope = trajectory_conformal_regret_envelope(
    realized_regret_by_trajectory,
    registered_upper_bound_by_trajectory,
    miscoverage=0.20,
)

decision = support_robust_decision(
    registered_worst_case_regret,
    conformal_radius=envelope.radius,
    regret_tolerance=0.25,
    fallback_action_index=0,
)
```

The DEFORM companion experiment uses complete trajectories for calibration,
freezes all budgets before opening evaluation outcomes, and reports the entire
utility--coverage--violation frontier rather than choosing a target-favorable
operating point.
