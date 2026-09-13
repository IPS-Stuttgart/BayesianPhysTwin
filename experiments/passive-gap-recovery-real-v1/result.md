# Completed passive gap-and-recovery result

## Execution and verification

- Pull request: https://github.com/IPS-Stuttgart/BayesianPhysTwin/pull/937
- Workflow: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33984801057
- Event: `push`, triggered by the change to `experiments/passive-gap-recovery-real-v1/request.json`.
- Executed commit: `178a418c7b656958730fc2441d7122a1470642ee`.
- Scientific module blob: `dac16cdcb8fe476a86e14bf66f87733fd8107425`.
- Runner: `workstation1`, labels `[self-hosted, Linux, X64, gpuserver4090]`.
- Job and every step completed successfully. Scientific computation elapsed: 77.60 seconds.
- Artifact: `9974846454`, `passive-gap-recovery-real-v1-33984801057-1`.
- Archive SHA-256: `18518d0d4940be5b794aa86dcf15900f72911eeed5d4498f933826e6ae386f23`.
- Result JSON SHA-256: `9245ddfbfca057b03010720f5e987883d0ff1ca80096ca17e76c75a7bf8962d9`.
- Scores CSV SHA-256: `c5d4b04819084b840320475dcc1d5e91a05325f8eddd3abbdd101bded0b144f0`.
- Source-fit seal: `acf69c3e0b1959890d5e5f9e34d2d8f11ac0c835a8b46ed91567c407ea939526`.
- Complete prediction seal: `9d386be4293f1b33485a9686c135b5fccc89e35a1e11edf4885edc37bae412ae`.
- Protocol: `a8cc959bc03bae47a48ebdc49b9a8c966a54a28fe921f6ae70deb4b4d007d054`.

The downloaded artifact archive hash and all eight file checksums were verified independently after execution. A separate pandas aggregation reproduced every headline table value from all 3,520 score rows, verified the 32-recording roster and absence of duplicate score keys, and independently recomputed the source-fit, protocol, and prediction-seal hashes. No target-driven changes or replacement scientific runs followed the result.

## Decision

**The predeclared overall-superiority criterion failed.** Recursive covariance helps within this particular physical-residual architecture, but a tuned deterministic filter acting directly on observed positions is better overall.

This is a completed scientific comparison, not a technical failure. Workflow success must not be confused with hypothesis acceptance.

## Setup

Public Tracking Cloth Deformation motion-capture trajectories: 32 shaking recordings for fitting and 32 twisting recordings for evaluation, over eight material/size specimens. The already-open free-hanging recordings were reused for a distinct retrospective question. All 56 collision recordings remained numerically unopened by this experiment.

The existing limited spring-mesh pilot supplies the physical prior. It is not released PhysTwin, DEFORM, or a real RGB-D provider. All free-marker observations are hidden for fixed 0.1, 0.3, or 0.6-second intervals, with three gaps and 0.3-second recovery windows per clip. Driven corners remain available as current-time boundary observations and are not scored. All forecasts are scored before assimilating the current free-marker observation.

The internal `clean` label means the **native, unmodified observation stream**, not guaranteed complete or noiseless measurements. Its target scoring window contains 66,917 valid out of 67,200 nominal marker samples; 283 reference samples are naturally missing and are excluded consistently across methods. Those native gaps also affect inference. Imposed masks are additional to native missingness.

## Equal-specimen mean 3-D RMSE

Values are in millimetres. Each recording and each imposed gap length has equal weight within a specimen.

| Method | Gap + recovery | Gap only | Recovery only | Native observations |
|---|---:|---:|---:|---:|
| Tuned raw-position alpha-beta | 25.6598 | 28.3637 | 20.6282 | 3.8906 |
| Raw-position persistence | 28.5212 | 33.4157 | 19.5214 | 6.5542 |
| Source-selected deterministic reference | 30.0843 | 33.5284 | 23.7814 | 4.0412 |
| Recursive Bayesian residual | 34.3381 | 39.3778 | 24.5816 | 4.0691 |
| Same-model Gaussian MAP | 34.3381 | 39.3778 | 24.5816 | 4.0691 |
| Tuned physical-residual alpha-beta | 37.2300 | 41.6022 | 29.2499 | 5.0906 |
| Last residual / source-selected exponential | 44.4577 | 52.2231 | 30.5880 | 10.8405 |
| Raw constant velocity | 46.4146 | 48.7813 | 42.5260 | 5.7186 |
| Same-model stationary gain | 47.9785 | 47.9893 | 47.0165 | 7.2734 |
| Covariance reset on return | 47.9785 | 47.9893 | 47.0165 | 7.2734 |
| Spring prior without correction | 124.0827 | 119.8832 | 127.9280 | 128.9286 |

The source-selected reference was raw-position alpha-beta for cotton A2 and polyester A2, and physical-residual alpha-beta for the other six specimens. Every method's parameters and this selection were fixed using shaking recordings, before twisting predictions were made.

## Primary contrast and stronger simple baseline

Bayesian minus source-selected reference:

- Difference: **+4.2538 mm**, or **14.14% worse** in mean gap/recovery RMSE.
- Paired specimen-bootstrap 95% interval: **[-2.8129, +13.4004] mm**.
- Material-cluster sensitivity interval: **[-2.8761, +11.3837] mm**.
- Bayesian wins: 6/8 specimens; the two larger regressions prevent an average improvement.
- Native-observation cost: +0.0279 mm, interval [-0.9556, +1.2839] mm; the predeclared 0.5 mm upper-limit condition is not established.

The separately source-tuned raw-position alpha-beta arm is even stronger: Bayesian RMSE is **33.82% worse**, a difference of **+8.6783 mm**, with specimen-bootstrap interval **[+1.2697, +16.6908] mm**. Bayesian wins only 1/8 specimens in that comparison. Raw-position persistence also beats the Bayesian residual model: 28.5212 versus 34.3381 mm.

## Positive, narrower mechanism comparisons

At the same spring prior:

- Against last residual: **22.76% lower** RMSE, difference **-10.1197 mm**, interval **[-12.5889, -8.2169] mm**, 8/8 specimen wins.
- Against independently source-tuned physical-residual alpha-beta: **7.77% lower** RMSE, difference **-2.8920 mm**, interval **[-4.5343, -1.5071] mm**, 8/8 wins. Material-cluster interval: [-3.7700, -2.0139] mm.
- Against the same model's stationary gain: **28.43% lower** RMSE, difference **-13.6404 mm**, interval **[-26.2769, -4.3619] mm**, 8/8 wins.
- Resetting covariance on measurement return gives the same performance as stationary gain to numerical precision, and is likewise worse than propagation.

Gaussian information-form MAP matches the Bayesian mean with maximum absolute deviation **1.9984e-14 m**. Thus the result does not show superiority over equivalent Gaussian point estimation. The stationary-gain and reset arms share the Bayesian-selected dynamics/noise parameters; they are mechanism ablations, not independently tuned competitors. The independently tuned alpha-beta comparisons supply the stronger controls.

## Imposed-gap-duration breakdown

These are gap-plus-recovery RMSE, again in millimetres, independently aggregated from the CSV without retuning.

| Gap duration | Bayesian residual | Physical-residual alpha-beta | Last residual | Raw-position alpha-beta |
|---|---:|---:|---:|---:|
| 0.1 s | 8.7033 | 10.0004 | 17.7407 | 7.1732 |
| 0.3 s | 29.5738 | 32.6075 | 41.4809 | 22.2205 |
| 0.6 s | 64.7372 | 69.0822 | 74.1516 | 47.5856 |

## Interpretation

Supported: maintaining covariance improves the recursive discrepancy correction relative to the specified fixed-gain and residual-persistence controls on this real-trajectory panel.

Not supported: the present spring-residual Bayesian twin is the most accurate predictor, beats strong model-free temporal baselines, or is superior to same-model MAP. The results suggest that the weak physical prior/model-discrepancy interface can outweigh the benefit of covariance propagation; they do not uniquely identify the cause of the remaining error.

These are retrospective same-specimen/new-motion results with artificially hidden real measurements. They do not establish natural RGB-D occlusion robustness, calibrated uncertainty, arbitrary-object generalization, a released-PhysTwin benchmark win, or robot-control benefit. The cohort and result are retained without outcome-based tuning, omission of the raw-position controls, or reopening protected datasets.
