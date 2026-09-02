# DEFORM DLO4/DLO5 conformal regret envelope v1

## Status

Completed retrospective, source-sealed real-data evaluation on the official
DEFORM DLO4/DLO5 trajectories.  This result extends the finite-support decision
certificate with a split-conformal trajectory-level envelope for support
misspecification.  It does not modify or reinterpret the earlier certificate.

- Workflow run: `33594490034`
- Runner label: `gpuserver4090`
- Scientific implementation revision before evidence publication:
  `95d7cc96cb0ab3eabf1bbadd3de2ced420a27428`
- Parent decision workflow: `33473378340`
- Dataset revision: `b73b8b8ecc033caefa693fab7898741d4e6dbeff`
- Target tuning: none
- Target retries: none

## Design

For each DLO, the original metadata-independent split is retained:

| Partition | Trajectories | Use |
|---|---:|---|
| Fit | 39 | Fit response pool |
| Model-selection calibration | 9 | Retain the historical selected settings |
| Source test | 8 | Calibrate the new conformal support envelope |
| Official evaluation | 14 | Held trajectory audit |

Each trajectory contains 19 nonoverlapping decisions.  One conformity score is
computed per complete source-test trajectory,

```text
S_j = max_(decision, registered nonfallback action)
      (realized regret - registered support regret bound).
```

The split-conformal radius uses rank
`ceil((n + 1) * (1 - alpha))`.  It is then added to the registered support-wise
regret of every candidate nonfallback action.  A nonfallback action is emitted
only when it is the unique minimum-regret action and its inflated bound is no
larger than the declared regret budget.  Otherwise the exact physical fallback
is returned.

The complete trajectory, not a frame, node, action, or decision window, is the
calibration and evaluation unit.

## Source calibration

With eight source-test trajectories per DLO, nominal 90% trajectory coverage is
not attainable by ordinary split conformal calibration: the required rank is 9.
The implementation therefore records an explicit `infinite` radius and returns
fallback for every finite budget.  It does not remove or silently relax that
operating point.

The finite per-DLO radii are:

| Nominal trajectory coverage | DLO4 radius | DLO5 radius |
|---:|---:|---:|
| 80% | 0.635917 | 0.280324 |
| 70% | 0.267313 | 0.272595 |

The pooled balanced-DLO sensitivity radius is 0.272595 at both 80% and 70%, but
its validity requires exchangeability under that declared DLO mixture.  The
per-DLO envelopes are primary.

## Held trajectory coverage

For the primary 80% per-DLO envelope, empirical simultaneous trajectory
coverage of the original certificate-selected actions is:

| DLO | Covered trajectories | Empirical coverage |
|---|---:|---:|
| DLO4 | 13/14 | 92.86% |
| DLO5 | 10/14 | 71.43% |
| Descriptive combined count | 23/28 | 82.14% |

The combined count is descriptive; the formal interpretation remains
within-DLO trajectory-marginal validity under exchangeability.

## Predeclared 80%-coverage regret frontier

The protocol reports the complete predeclared budget frontier.  The table below
uses the simultaneous envelope over both nonfallback actions and allows the
minimum-regret action even when the original support-only tolerance would have
returned fallback.

| Regret budget | Nonfallback | Held RMSE reduction | Harmful nonfallback | Budget exceeds | Trajectories with exceed | Mean trajectory gain [95% bootstrap] |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 18/532 | 1.18% | 0/18 | 2/18 | 2/28 | 1.04% [0.51%, 1.62%] |
| 0.50 | 202/532 | 11.42% | 2/202 | 1/202 | 1/28 | 10.61% [6.50%, 14.92%] |
| 0.75 | 334/532 | 16.65% | 9/334 | 2/334 | 2/28 | 16.13% [12.70%, 19.76%] |
| 1.00 | 508/532 | 23.71% | 12/508 | 1/508 | 1/28 | 24.36% [22.16%, 26.65%] |

At budget 1.00, every complete held trajectory improves relative to fallback;
the worst trajectory RMSE ratio is 0.8711.  This is a high regret budget and
must not be presented as a safety threshold.  The full curve, rather than one
post-outcome selected point, is the scientific result.

The preregistered primary point is the 80%-coverage, budget-0.30 envelope applied
only to actions selected by the original support certificate.  It emits 18
nonfallback decisions, reduces aggregate RMSE by 1.18%, has no harmful update,
and has two realized budget exceeds on two held trajectories.  It is deliberately
conservative and emits no nonfallback action on DLO4 because the source-calibrated
DLO4 radius exceeds the primary budget.

## Interpretation

The earlier finite-support certificate answered:

> Is an action admissible for every complete belief represented by the declared
> finite quotient support?

This extension answers a different finite-data question:

> After calibrating the observed support-misspecification error on complete
> trajectories, which actions remain below a declared realized-regret budget?

The result turns the previous support-mismatch failure into an explicit
risk--coverage--utility frontier.  It also demonstrates why the fallback is
necessary: demanding 90% coverage with only eight calibration trajectories
produces an infinite radius and therefore no licensed departure from fallback.

## Claim boundary

Under exchangeability of complete source-test and future trajectories within the
same DLO stratum, the per-DLO envelope has trajectory-marginal simultaneous
coverage at its nominal level over the registered decisions and actions.  On the
coverage event, every emitted nonfallback action has realized regret no larger
than the declared budget.

This is not pointwise conditional validity.  It does not establish exchangeability,
unseen-object or cross-material transport, arbitrary-action safety, calibrated
state uncertainty, online robot performance, or deployment authorization.  The
held evaluation is retrospective method-development evidence because DLO4/DLO5
outcomes were used by earlier studies, although no held outcome is used by this
protocol for calibration, budget selection, target tuning, or retries.

## Next decisive experiment

The next major step is object-disjoint calibration and confirmation: calibrate a
loss envelope over several physical objects or material groups, freeze a risk
budget and probe policy, and evaluate both decision risk and physical probe cost
on previously unused objects.  That experiment—not the present within-DLO
analysis—is required for an open-world physical-twin claim.
