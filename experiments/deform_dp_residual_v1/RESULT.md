# Completed DP residual-expert source screen v1

**Execution succeeded. The prespecified DP superiority criterion failed.**

The tested truncated variational DP mixture provided no established advantage over the matched finite mixture and was worse than both the single-expert and nonlinear RBF controls. The result does not support centering a paper contribution on this DP residual variant. It does identify a useful nonlinear residual-control result, without promoting it to independent confirmation or full physical-twin superiority.

## Real-data comparison

DEFORM DLO4/DLO5, exactly 32 fitting +12 validation +12 held-source recordings per fixed object. Five forecast origins and horizons 5/20/50 produce 360 forecast contexts, evaluated as 24 complete trajectories. Every method conditions on the same observed prefix and recorded future clamped-node positions. No robot action is selected and no new observations are collected. Only pinned public `train` files were accessed; official evaluation was neither enumerated nor downloaded.

The reference is the unchanged action-conditioned ridge **surrogate** from PR #940. This is NOT the official physical DEFORM checkpoint, not the DLO2 7.8606-mm physical-plus-local-residual predictor, and not an evaluation of Prob4D or Causal4D.

| Method | Coordinate L1 (mm) | Coordinate RMSE (mm) |
|---|---:|---:|
| Persistence | 60.191056 | 88.508527 |
| Boundary interpolation | 40.442170 | 63.167344 |
| Unchanged reference ridge | 23.757780 | 35.405069 |
| One residual expert | 23.246430 | 34.621776 |
| Finite Bayesian mixture | 23.510908 | 35.121794 |
| Truncated DP mixture | 23.534710 | 35.086492 |
| Nonlinear RBF residual | **22.593794** | **33.416905** |

## Frozen primary test

The criterion required at least 1% mean L1 improvement over ridge AND favorable paired 95% bootstrap upper bounds against ridge, finite mixture and RBF. Resampling uses complete trajectories, stratified within the two fixed objects, with 10,000 replicates. This is not an unseen-object population interval.

Differences below are **DP minus comparator**; negative favors DP.

| Comparator | Difference (mm) | 95% interval (mm) | DP wins /24 |
|---|---:|---:|---:|
| Reference ridge | -0.223070 | [-0.460966, +0.020159] | 15 |
| Finite mixture | +0.023803 | [-0.028843, +0.075973] | 10 |
| One expert | +0.288280 | [+0.025899, +0.568351] | 8 |
| RBF residual | +0.940916 | [+0.625439, +1.285238] | 3 |

The DP mean gain over ridge was **0.9389%**, with an interval crossing zero. The failure is not merely missing the arbitrary 1% threshold: DP also failed to establish superiority over the finite mixture and clearly lost to the RBF comparator on this panel.

## Useful secondary observation

The prespecified RBF comparator reduced mean L1 by **4.8994%** versus ridge. It was better on **23/24** held-source trajectories and reduced both fixed-object means:

| Object | Ridge L1 (mm) | DP L1 (mm) | RBF L1 (mm) |
|---|---:|---:|---:|
| DLO4 | 22.411253 | 22.184711 | 21.299916 |
| DLO5 | 25.104307 | 24.884709 | 23.887672 |

A descriptive post-run reaggregation gives RBF-minus-ridge -1.163986 mm with paired 95% interval [-1.531605, -0.826311]. This additional comparison is not substituted for the failed DP primary hypothesis. RBF was already in the frozen comparator bank; no new model was fitted after source-test scoring.

Interpretation: for this representation and data budget, smooth nonlinear residual regression is more promising than the tested mixture specialization. This does not prove that every DP, temporal switching model, physical-checkpoint residual, or uncertainty-aware regime formulation is unhelpful. No density-calibration or physical-regime-identification claim is evaluated here.

## Execution and independent audit

- Scientific implementation: `bfadd35a303e2c9340daddd17e64e7c7d06f0ca7`.
- Hosted execution trigger: `ae3bcef907dc273c68d66c26c5d3697233110ab9`.
- Successful [run 34053230315](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34053230315), job `101540449982`, Ubuntu 24.04 hosted CPU runner.
- Artifact `9995192962`, ZIP SHA-256 `0568dc58790d384b03d532ea254992b0e144322acd1b0cbf1f319c7ffa7d3577`.
- Scientific files remained byte-identical during the execution-host change. The initial self-hosted run `34053040058` was cancelled while queued, with no steps executed; it produced no scientific outcome.
- All 112 downloaded source-file hashes match the previous on-server manifest. Manifest SHA-256 `9a1ef8d257f4150df842feecc2d45b579eb49a1276e0b023d670221084c0646f`.
- Runtime: Python 3.12.14, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.8.0; one BLAS/OpenMP thread.
- Ten numerical/information-boundary tests passed on the actual execution runner.
- No candidate structure failed. Validation selected nonzero corrections for every family/panel; the negative DP result is not an all-fallback artifact.
- Downloaded ZIP digest and every code/model/prediction seal hash verified.
- All **8,640 reference predicted coordinates are exactly equal** to the 32-fitting-recording predictions in the previous PR #940 artifact; maximum difference is zero.
- Independent score reaggregation reproduced all reported L1/RMSE aggregates and DP bootstrap intervals. All selected configurations minimize their recorded validation scores.

The independent audit checks artifacts, baseline parity and score reaggregation; it is not a second local rescore from raw trajectories. Raw predictions, fitted parameters, validation records, protocol, source identities, runtime and tests are in the retained Actions artifact. No estimator, grid, partition or decision rule was changed after outcomes were scored.

## Status

Retain this as a completed **negative DP-specific development result**, alongside the nonlinear-control improvement. Do not merge it as a proven production enhancement or rewrite any frozen manuscript claim. All source recordings were historically exposed, so the study is retrospective held-source evaluation on two fixed objects, not prospective confirmation.
