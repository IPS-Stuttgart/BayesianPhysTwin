# Completed heavy-tail control study — 2026-09-07

## Conclusion

**The small-fitting-set improvement survives three empirical Student-t controls, but the incremental benefit of posterior integration is not established against a matched plug-in Student-t.** At 32 fitting trajectories, none of the new comparisons establishes a posterior advantage. This narrows the interpretation of the earlier result rather than replacing its unsuccessful primary decision.

The matched plug-in Student-t still uses the parent's regularized noise-covariance estimate. It is therefore an integration/distribution-family control, not evidence that every Bayesian covariance-estimation component is unnecessary. The three fully empirical uncertainty controls estimate their covariance from retained source residuals instead.

## Execution and immutable inputs

- Successful Actions run: `34134095482`; job `101780858294`, runner `workstation1` through `[self-hosted, gpuserver4090]`.
- Request-file-only trigger: `74b77dceaf340dad51f9da6dec3adb1ba8c290b3`.
- Evaluator Git blob: `a14f34cf6cf4ad0e6dbed38ac4004107c9d8ccdc`.
- Output artifact: `10023268308`, `deform-query-heavy-tail-controls-v1-34134095482`, 10,649,543 bytes.
- Output artifact SHA-256: `a7f80d8e694f08bba95f0a8b6409c8e528e95efe81de3da1dda75edf37a49263`.
- Original output `result.json` SHA-256: `b7d9d8ccea96ea2ac1b2ded7ffc404bf6ef7625c0c758388e2fe8aaad60cd019`.
- Parent run: `33985128557`, artifact `9974939619`; all parent artifact/seal identities verified.
- Tests on the successful runner: **28 passed**.

Two infrastructure failures are retained: run `34133643524` stopped at pytest's duplicate-module-name collection check; run `34133853774` passed the tests but found no installed `gh` executable. Neither started experiment-data evaluation. Repairs changed only pytest import mode and the artifact transport. The statistical evaluator and its protocol blob remained unchanged through all three workflow attempts. There was one completed new-control real-data evaluation and no post-outcome tuning.

## Eight fitting trajectories: the secondary finding

Each object additionally supplies **12 calibration trajectories**. All methods share exactly the original predictive mean, and the same 24 source-test recordings are scored. Smaller NLL and CRPS are better; NLL uses metre-valued densities. These are distribution scores, not trajectory RMSE.

| Predictive distribution | NLL | CRPS (mm) | Nominal-90% coverage | Full interval width (mm) |
|---|---:|---:|---:|---:|
| Original posterior Student-t | -2.002208 | 22.69677 | 89.421% | 133.531 |
| Original plug-in Gaussian | -1.970590 | 22.91505 | 91.829% | 149.448 |
| Empirical Student-t, global | -1.964517 | 22.90998 | 88.750% | 137.901 |
| Empirical Student-t, conditional | -1.976325 | 22.85801 | 89.676% | 140.225 |
| Conditional Student-t + frequentist ridge sampling covariance | -1.970937 | 22.86732 | 89.051% | 138.544 |
| Matched plug-in Student-t | -1.998617 | 22.72909 | 90.579% | 139.885 |

Posterior-minus-control differences with pointwise 95% complete-trajectory bootstrap intervals:

| New comparator | NLL difference [95% interval] | CRPS difference in mm [95% interval] |
|---|---:|---:|
| Empirical global Student-t | -0.037691 [-0.054287, -0.020897] | -0.213209 [-0.292202, -0.135800] |
| Empirical conditional Student-t | -0.025883 [-0.038519, -0.013534] | -0.161247 [-0.220508, -0.101894] |
| Frequentist sampling-covariance Student-t | -0.031271 [-0.047437, -0.015586] | -0.170554 [-0.250103, -0.094527] |
| Matched plug-in Student-t | -0.003591 [-0.016023, +0.010787] | -0.032324 [-0.107006, +0.045606] |

The first three comparisons favor the posterior on both scores, with favorable object-mean differences on both DLO4 and DLO5. Their CRPS improvements are approximately 0.93%, 0.71%, and 0.75% respectively. The closest matched-family comparison is only a 0.14% average CRPS difference and is inconclusive on both scores.

Replacing the original plug-in Gaussian with a source-calibrated plug-in Student-t closes **85.2% of its CRPS gap and 88.6% of its NLL gap** to the posterior on this panel. These are descriptive gap reductions, not proof of statistical equivalence or a formal causal attribution of all effects to heavy tails.

## Primary 32-fit condition and size sensitivity

All rows use the same twelve additional calibration recordings per object.

| Fitting trajectories | Posterior CRPS (mm) | Matched plug-in Student-t CRPS (mm) | Posterior minus matched Student-t, mm [95% interval] |
|---|---:|---:|---:|
| 8 — secondary | 22.69677 | 22.72909 | -0.032324 [-0.107006, +0.045606] |
| 16 — secondary | 17.48433 | 17.46242 | +0.021908 [-0.024674, +0.067279] |
| 32 — primary | 15.23560 | 15.23518 | +0.000419 [-0.060663, +0.062899] |

At 32 fitting trajectories the new NLL values are posterior `-2.345080`, global empirical Student-t `-2.348663`, conditional empirical Student-t `-2.345310`, frequentist sampling-covariance Student-t `-2.346005`, and matched plug-in Student-t `-2.346163`. All four posterior-minus-new-control NLL intervals cross zero, as do all four CRPS intervals. The follow-up primary superiority condition therefore fails. At 16 fitting trajectories, NLL favors the posterior against the three empirical controls, but the CRPS intervals remain inconclusive.

The source-size flag in the machine output is true at eight only for the explicitly declared empirical-conditional and sandwich controls. It must not be read as success against the matched plug-in Student-t or as promotion of eight to the primary condition.

## Verification

The complete artifact and both generations of prediction seals were independently verified. The 7,560 retained parent arm/context score rows are **exactly unchanged**. All 1,080 prediction contexts copy the parent mean exactly, and the runner's replay of the original Bayesian score has zero discrepancy.

Independent pandas aggregation of all 11,880 score rows agrees with `result.json` to `8.881784197001252e-16`. Independent covariance-to-query projection checks agree to maximum relative spread `1.9000659489747885e-15`. This verifies the saved numerical relationships; it is not a separate real-data replication.

## Scientific interpretation and limits

This is a retrospective follow-up designed after the earlier outcomes were known, on the same two objects and 24 source-test trajectories. New control parameters are selected solely from the original development queries and calibration records, but this does not turn the reused evaluation set into independent confirmation. Intervals are pointwise and descriptive, with complete-recording resampling stratified within the two fixed objects; they are not multiplicity-adjusted unseen-object confidence statements.

The results support a modest small-fit advantage of the evaluated posterior construction over the tested empirical covariance models. They **do not demonstrate an incremental benefit of integrating over the posterior once the plug-in predictor is given a matched heavy-tailed family and the same regularized noise-covariance estimate**. Absence of a resolved difference is not proof that the models are equivalent.

A working explanation is that regularized covariance estimation and a suitable heavy-tailed predictive family are doing much of the useful work. The present comparisons do not separately prove the optimality or novelty of either ingredient. No point-prediction improvement is possible here because means are deliberately fixed.

The common mean remains the original compact action-conditioned ridge surrogate, not a full DEFORM simulator, Prob4D perception model, or Causal4D intervention pipeline. No official evaluation files, new objects, new robot actions, active measurements, or new physical acquisitions were used. The original result and its primary failure remain immutable. No manuscript claim is automatically promoted.
