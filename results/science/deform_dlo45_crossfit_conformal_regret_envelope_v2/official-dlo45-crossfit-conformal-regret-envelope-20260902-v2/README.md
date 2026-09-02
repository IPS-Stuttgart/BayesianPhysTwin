# Cross-fitted conformal regret envelope v2

Two complementary routes per DLO each train and tune on 28 source
trajectories and calibrate on the disjoint 28. A held trajectory is
assigned to exactly one route by an outcome-independent filename hash.
Complete trajectories, not windows, are the calibration units.

## Source calibration

| DLO | Route | Selected settings | 95% radius | 90% radius | 80% radius |
|---|---:|---|---:|---:|---:|
| DLO4 | 0 | `k=16, C=16, T=1.0, eps=0.05` | 0.654699 | 0.520794 | 0.419715 |
| DLO4 | 1 | `k=16, C=16, T=1.0, eps=0.05` | 0.484803 | 0.326836 | 0.297627 |
| DLO5 | 0 | `k=64, C=16, T=1.0, eps=0.0` | 0.657835 | 0.373061 | 0.105589 |
| DLO5 | 1 | `k=16, C=16, T=1.0, eps=0.05` | 1.214556 | 0.537759 | 0.292876 |

All requested radii are finite. Calibration trajectories are never
used to fit or tune their associated route model.

## Held simultaneous trajectory coverage

| Nominal | DLO4 | DLO5 | Descriptive combined |
|---:|---:|---:|---:|
| 95% | 14/14 | 14/14 | 28/28 |
| 90% | 13/14 | 13/14 | 26/28 |
| 80% | 13/14 | 12/14 | 25/28 |

## Selected predeclared frontier points

| Nominal | Budget | Nonfallback | RMSE reduction | Harmful vs fallback | Budget exceeds | Trajectories with exceed |
|---:|---:|---:|---:|---:|---:|---:|
| 95% | 0.75 | 170/532 | 5.38% | 7/170 | 0/170 | 0/28 |
| 90% | 0.50 | 89/532 | 2.97% | 2/89 | 0/89 | 0/28 |
| 90% | 0.75 | 477/532 | 18.79% | 18/477 | 0/477 | 0/28 |
| 80% | 0.50 | 359/532 | 14.40% | 13/359 | 0/359 | 0/28 |

Primary (90%, budget 0.50): 89/532 nonfallback decisions, 2.97%
RMSE reduction, two updates worse than fallback, and zero realized
regret-budget exceeds across all 28 held trajectories.

Predeclared 90%, budget 0.75: 477/532 nonfallback decisions, 18.79%
RMSE reduction, and zero regret-budget exceeds. The larger budget is
not a safety threshold and this point was not selected using targets.

At 95% nominal coverage, both DLOs cover 14/14 held trajectories; at
90%, both cover 13/14. These are descriptive checks of a retrospective
experiment, not proof of exchangeability or unseen-object transport.

## Claim boundary

This retrospective cross-fitted analysis assigns each held trajectory to one metadata-only route whose model is trained and tuned on 28 disjoint source trajectories and whose split-conformal envelope is calibrated on the complementary 28 complete source trajectories. Under within-DLO trajectory exchangeability, each routed envelope has trajectory-marginal simultaneous coverage over the registered decisions and actions. It does not provide pointwise conditional validity, unseen-object or cross-material transport, arbitrary-action safety, calibrated state uncertainty, online robot performance, or deployment authorization.

Workflow run: `33603986580` on `gpuserver4090` (`workstation1`).
Scientific revision: `f8febe3b75fc5933a4ea0c6d51656dcf8003109c`.
Target tuning and target retries: none.
