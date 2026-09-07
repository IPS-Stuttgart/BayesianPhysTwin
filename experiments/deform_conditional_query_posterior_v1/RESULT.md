# Completed real-data result: conditional posterior query transfer v1

## Execution and provenance

- Workflow run: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33985128557
- Job: `101357122553`; execution conclusion: **success**.
- Runner: `workstation1`, selected by `[self-hosted, gpuserver4090]`.
- Frozen implementation: `0e13401eab716f9dda9b578f89d844214b55e39e`.
- Request-file-only trigger: `70283945d068834e6f2bd211d56cc7a65ce42fef`.
- Artifact: `9974939619`, `deform-conditional-query-posterior-v1-33985128557`, 9,330,318 bytes.
- Artifact SHA-256: `e7e182e009ac234fb611afb6ca03eb1006ee65da2688ec2a084105667fa4e591`.
- Original `result.json` SHA-256: `5ded16a91bd0e12563379d8a380428b873d706d9b0b9af74a07feffda4fa4d4b`.
- Ten numerical and information-exclusion tests passed on the actual runner.

The original result, complete predictions, fitted models, input hashes, prediction seal, source configuration, and per-trajectory/panel scores are retained in the workflow artifact. This document summarizes that immutable output; it does not replace it. No model, split, calibration setting, or scientific decision rule was changed after scoring.

## Primary outcome: not established

The pre-specified primary condition used **32 fitting trajectories plus 12 calibration trajectories per object**. Both scores had to improve over the plug-in model and all three empirical baselines, with paired bootstrap upper limits below zero. That conjunction **did not pass**.

All arms use exactly the same trajectory mean. Primary results average the 24 complete source-test trajectories, with equal counts for DLO4 and DLO5. NLL uses metre-valued densities; smaller NLL and CRPS are better.

| Arm | NLL | CRPS (mm) | 90% coverage | Full interval width (mm) | Brier |
|---|---:|---:|---:|---:|---:|
| Posterior Student-t | -2.345080 | 15.2356 | 90.648% | 93.517 | 0.194026 |
| Matched plug-in Gaussian | -2.328102 | 15.2578 | 89.560% | 88.817 | 0.194581 |
| Independently calibrated Gaussian posterior covariance | -2.334167 | 15.2638 | 91.319% | 95.717 | 0.195010 |
| Exactly covariance-matched Gaussian | -2.337108 | 15.2460 | 90.718% | 93.760 | 0.194397 |
| Empirical shrinkage covariance | -2.331482 | 15.3167 | 92.454% | 100.372 | 0.196488 |
| Global whole-vector residual bootstrap | -2.326834 | 15.3250 | 92.315% | 99.945 | 0.196563 |
| Local whole-vector residual bootstrap | -2.251124 | 15.6169 | 93.009% | 108.908 | 0.200920 |

Posterior-minus-comparator differences and stratified complete-trajectory bootstrap 95% intervals:

| Comparator | NLL difference [95% interval] | CRPS difference in mm [95% interval] |
|---|---:|---:|
| Matched plug-in | -0.016978 [-0.033256, -0.003428] | -0.022215 [-0.068265, +0.026620] |
| Independently calibrated Gaussian posterior covariance | -0.010913 [-0.017586, -0.004074] | -0.028189 [-0.060939, +0.007155] |
| Exactly covariance-matched Gaussian | -0.007971 [-0.011126, -0.005120] | -0.010423 [-0.017822, -0.003181] |
| Empirical shrinkage | -0.013597 [-0.030528, +0.004655] | -0.081125 [-0.163089, -0.000724] |
| Global residual bootstrap | -0.018246 [-0.035248, -0.000613] | -0.089367 [-0.167638, -0.014139] |
| Local residual bootstrap | -0.093956 [-0.129663, -0.062939] | -0.381347 [-0.511060, -0.256758] |

The primary gate fails specifically because plug-in CRPS and empirical-shrinkage NLL remain inconclusive. It is not correct to call the entire experiment negative: both bootstrap comparisons and the covariance-matched shape comparison improve on both scores. The absolute CRPS advantages are small, particularly against the plug-in and moment-matched Gaussian.

## Pre-specified source-size analyses

Every size below uses the same additional **12 calibration trajectories per object**. The smaller sizes are nested fitting subsets, not replacements selected after observing test performance. In particular, the first row is not an eight-total-recording experiment.

| Fitting trajectories per object | Posterior CRPS (mm) | Plug-in CRPS (mm) | Difference in mm [95% interval] | NLL difference [95% interval] |
|---|---:|---:|---:|---:|
| 8, secondary | 22.69677 | 22.91505 | -0.218285 [-0.382699, -0.063154] | -0.031618 [-0.046607, -0.017729] |
| 16, secondary | 17.48433 | 17.49837 | -0.014044 [-0.090918, +0.065296] | -0.025022 [-0.053394, -0.004111] |
| 32, primary | 15.23560 | 15.25781 | -0.022215 [-0.068265, +0.026620] | -0.016978 [-0.033256, -0.003428] |

At eight fitting trajectories, posterior-minus-comparator NLL and CRPS intervals are below zero against **all six** tested controls. CRPS improves about **0.95%** versus the plug-in model, **3.03%** versus global bootstrap, and **3.62%** versus local bootstrap. Both DLOs have favorable mean NLL and CRPS against the plug-in and empirical baselines at this size. This is a bounded positive secondary small-fitting-set result, not a replacement for the failed primary conjunction.

Against an exactly covariance-matched Gaussian, eight-fit CRPS improves only `0.042914 mm` with interval `[-0.082883, -0.003370]`. This isolates a small distribution-shape benefit but does not show that Bayesian inference is uniquely necessary: an independently developed heavy-tailed frequentist predictive model was not tested.

## Independent artifact checks

The downloaded ZIP digest and all four prediction-seal hashes verified. Source fitting, calibration, and source-test partitions have disjoint file-content hashes within each DLO. Independent pandas aggregation of the retained score rows agrees with the result to maximum absolute error `8.881784197001252e-16`.

There are 90 fitted models, 1,080 prediction contexts, and 7,560 arm/context score rows. These are not independent statistical units. The statistical units remain 24 complete recordings within two fixed objects. Bootstrap distributions have mean roundoff at most `1.734723475976807e-18 m`; the exact covariance comparator matches to `1.0408340855860843e-17 m^2`.

## Scientific boundary

This is a **retrospective real-data test of a compact action-conditioned ridge surrogate**, not a rerun of DEFORM's full simulator, prior BayesianPhysTwin benchmark means, Prob4D perception, or Causal4D. The forecasts condition on recorded future clamped-node positions as exogenous inputs. Priors, heteroscedastic scales, preprocessing, and temperatures are source-fitted empirical-Bayes quantities, not fully integrated hyperparameters.

No official evaluation files were opened, no robot action or measurement was selected, and no new physical data were acquired. The evidence supports a modest small-fitting-set probabilistic benefit and improved log scores in this setting. It does not establish a generally more accurate physical twin or broad calibrated uncertainty across unseen objects.
