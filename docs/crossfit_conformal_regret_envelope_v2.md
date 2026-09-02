# Routed trajectory-conformal regret envelopes

## Purpose

The finite-support decision certificate bounds regret over the complete beliefs
represented by a registered quotient. It does not guarantee that a future
physical trajectory lies inside that support. The v1 conformal extension
calibrated the excess of realized regret over the registered support bound, but
only eight untouched source-test trajectories were available per DLO. Ordinary
split conformal calibration therefore had no finite 90% operating point.

Version 2 increases the calibration count without promoting correlated windows
to independent samples. For each DLO it defines two complementary routes:

- route 0 fits and tunes on source half A and calibrates on source half B;
- route 1 fits and tunes on source half B and calibrates on source half A.

Each half contains 28 complete trajectories. Within a training half, model
selection is nested again into 21 fit and 7 tune trajectories. A held trajectory
is assigned to exactly one route by an outcome-independent hash of its DLO and
file name before its numeric payload is read.

## Guarantee for one route

Fix route `r`, its model, registered candidate actions, and complete-trajectory
calibration set of size `n`. Let

```text
S_j = max over registered decisions t and candidate actions a
      [realized_regret(j,t,a) - registered_support_bound(j,t,a)]
```

for calibration trajectory `j`. For requested miscoverage `alpha`, define

```text
k = ceil((n + 1) * (1 - alpha))
```

and let `rho_r` be the `k`th smallest calibration score. If `k > n`, the radius
is positive infinity and the policy returns fallback for every finite budget.
Otherwise, under exchangeability of the complete calibration trajectories and
one future trajectory assigned to route `r`,

```text
P[for every registered t,a on the future trajectory:
  realized_regret(t,a) <= registered_support_bound(t,a) + rho_r]
  >= 1 - alpha.
```

The finite-sample lower bound is actually `k / (n + 1)`, which can be larger
than the requested nominal level because of discreteness. With `n = 28`, the
95%, 90%, and 80% requests use ranks 28, 27, and 24, corresponding to lower
bounds 28/29, 27/29, and 24/29.

If the policy emits a nonfallback action only when its inflated bound is no
larger than a declared budget `epsilon`, the same coverage event implies that
no emitted action on that trajectory exceeds `epsilon`. Consequently,

```text
P[there exists a nonfallback decision on the future trajectory
  whose realized regret exceeds epsilon] <= alpha.
```

This controls a trajectory-level joint exceedance event, not conditional risk
among selected decisions.

## Metadata-only routing

Let `R` be the route assigned to a held trajectory. If route assignment is fixed
without inspecting its numeric payload or outcome and complete trajectories are
exchangeable within each declared route/DLO stratum, then

```text
P[coverage failure] = sum_r P[R=r] P[coverage failure | R=r] <= alpha.
```

The routing construction therefore reuses all source trajectories across the
two deployed procedures while preserving an ordinary split-conformal argument
for the single route actually used by each target.

This statement relies on the declared exchangeability assumption. File names or
collection order may encode experimental structure; the code records the route
of every held trajectory so that this assumption can be audited rather than
silently asserted.

## Real-data result

The source-sealed DEFORM run is retained at
`results/science/deform_dlo45_crossfit_conformal_regret_envelope_v2/`.
All four route radii are finite at the 95%, 90%, and 80% requested levels.
Empirical simultaneous held-trajectory coverage is 14/14 for both DLOs at 95%
and 13/14 for both at 90%.

At the preregistered 90% level and regret budget 0.50, the policy emits 89 of
532 nonfallback decisions, reduces aggregate RMSE by 2.97%, and has zero held
regret-budget exceeds across all 28 trajectories. Two emitted updates are worse
than physical fallback while remaining inside the declared regret budget.

The complete predeclared frontier is part of the result. At the 90% level and
budget 0.75, 477 of 532 decisions are nonfallback and aggregate RMSE decreases
by 18.79%, with zero budget exceeds. The larger budget is not a safety
threshold, and this operating point was not selected using held outcomes.

## Claim boundary

The guarantee is marginal over a future complete trajectory and simultaneous
only over the registered decisions and action portfolio. It does not establish
pointwise conditional validity, exchangeability, unseen-object or
cross-material transport, arbitrary-action safety, calibrated state
uncertainty, online robotic success, or deployment authorization. The DEFORM
evaluation is retrospective because these held trajectories were opened by
previous studies, although this protocol uses no held outcome for routing,
model selection, calibration, budget choice, target tuning, or retries.
