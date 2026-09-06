# Completed raw-data reproduction: posterior query transfer

## Execution

The owner-requested GitHub retry completed successfully on `workstation1`, selected by `[self-hosted, gpuserver4090]`.

- Successful run: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34052497288
- Job: `101538486027`.
- Execution/request-file-only commit: `0dff23d43f231dc79498832486c08ee95e2be89a`.
- Original experiment: PR #940, run `33985128557`, scientific revision `0e13401eab716f9dda9b578f89d844214b55e39e`.
- Reproduction PR: https://github.com/IPS-Stuttgart/BayesianPhysTwin/pull/948
- Full artifact: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34052497288/artifacts/9994976937
- Artifact ZIP SHA-256: `0ebd4ba48441d8e1f161f7b974282c8b6db7eb1c489bf7a8da57b0f3ea36ac58`.
- New result.json SHA-256: `c1ccb037ca538ef8cae9fea46f6b31155dd3af65f2c3008eb4785e157f67641a`.
- New prediction_seal.json SHA-256: `fdaa1144407fc6f96dba5dc7e1d4ac2775022c9b29d44adb028318e7e838ba0a`.

This execution reran fitting, calibration, prediction sealing, scoring, and trajectory-level confidence intervals from the real recordings. It was **not merely a reaggregation of retained scores**. The historical artifact was downloaded only for the final reproduction comparison.

## Verification

All ten original numerical and future-information-exclusion tests passed on the runner. The request-only-change check and all four code/workflow blob checks passed. Input-recording hashes match the historical experiment. The verifier checked the new protocol, fitted-model, prediction, and input-manifest seals.

All **505 checked numerical values** in the configuration, accounting, scores, paired intervals, and decision-related result fields reproduced exactly: maximum absolute difference `0.0`. Separately implemented aggregation of **7,560 score rows** reconstructed **105 metrics**, with maximum absolute difference `8.881784197001252e-16`.

The original failed primary scientific gate remains failed. Workflow success means technical execution and reproduction, not scientific superiority.

## Data and comparison

The fixed experiment uses the 56 real source recordings per object in the canonical DLO4/DLO5 `train` directories under `/mnt/seagate10tb/florianpfaff/datasets/deform/data_set`. Each object has 32 fitting, 12 calibration, and 12 source-test recordings. Nested eight- and sixteen-fitting subsets are prespecified secondary analyses; they each still use all twelve calibration recordings.

Seven distributions share the exact same point mean: posterior Student-t, plug-in Gaussian, independently calibrated posterior-covariance Gaussian, exactly covariance-matched Gaussian, empirical shrinkage covariance, global residual bootstrap, and context-local residual bootstrap. Six calibration projections and twelve different evaluation projections are used. There are five origins and three forecast horizons per recording.

The empirical prediction controls resample **whole 24-dimensional future-shape residual vectors**, preserving spatial dependence. They do not define a single joint distribution over the entire temporal trajectory. Statistical confidence intervals separately resample complete test recordings within the two fixed objects.

## Primary result: superiority not established

Primary setting: 32 fitting plus 12 calibration recordings per object. Lower NLL and CRPS are better. NLL refers to metre-valued densities; CRPS is reported in millimetres.

| Method | NLL | CRPS (mm) | Nominal-90% coverage | Full interval width (mm) |
|---|---:|---:|---:|---:|
| Posterior Student-t | -2.345080 | 15.23560 | 90.648% | 93.517 |
| Matched plug-in Gaussian | -2.328102 | 15.25781 | 89.560% | 88.817 |
| Independently calibrated posterior-covariance Gaussian | -2.334167 | 15.26379 | 91.319% | 95.717 |
| Exactly covariance-matched Gaussian | -2.337108 | 15.24602 | 90.718% | 93.760 |
| Empirical shrinkage | -2.331482 | 15.31672 | 92.454% | 100.372 |
| Global residual bootstrap | -2.326834 | 15.32497 | 92.315% | 99.945 |
| Local residual bootstrap | -2.251124 | 15.61694 | 93.009% | 108.908 |

The posterior has the best average scores, but the predeclared conjunction fails. Posterior-minus-plug-in CRPS is **-0.022215 mm**, with 95% interval **[-0.068265, +0.026620]**. The NLL comparison with empirical shrinkage is also inconclusive. Posterior-minus-plug-in NLL is **-0.016978**, interval **[-0.033256, -0.003428]**.

## Prespecified source-size analysis

Every row uses twelve additional calibration recordings per object. These are not total-data counts.

| Fitting recordings per object | Posterior CRPS (mm) | Plug-in CRPS (mm) | Posterior minus plug-in (mm), 95% interval |
|---|---:|---:|---|
| 8, secondary | 22.69677 | 22.91505 | -0.218285 [-0.382699, -0.063154] |
| 16, secondary | 17.48433 | 17.49837 | -0.014044 [-0.090918, +0.065296] |
| 32, primary | 15.23560 | 15.25781 | -0.022215 [-0.068265, +0.026620] |

The reproduced eight-fit result has favorable NLL and CRPS intervals against all six tested controls. CRPS gains are approximately 0.95% against plug-in, 3.03% against global bootstrap, and 3.62% against local bootstrap. It is a modest positive secondary small-fitting-set result, not a replacement for the failed primary criterion. The sixteen-fit result does not establish the same advantage.

## Scope

This is a compact action-conditioned ridge-surrogate uncertainty test, **not the DEFORM physical-simulator benchmark or the previously reported BayesianPhysTwin mean predictor**. The posterior integrates regression and covariance uncertainty conditionally on source-fitted empirical-Bayes choices; hyperparameters and calibration temperatures are not fully integrated. Recorded future clamped-node positions are exogenous inputs.

No new physical measurements, robot actions, active probing, official evaluation files, or reserved cohorts were used. No dataset or original evidence was mutated. Reusing these historically exposed source recordings provides reproducibility, not new independent confirmation or unseen-object generalization. Two physical objects, not 24 independent objects, underlie the experiment. An independently fitted heavy-tailed frequentist comparator was not tested. No uniquely Bayesian necessity or broad full-twin superiority is established.

## Retained technical failure

The initial new run `34052441452` stopped at the verifier-blob receipt check before dataset contents, historical artifact download, tests, fitting, or scoring. Its request erroneously recorded the local verifier hash rather than the committed verifier hash. The unchanged committed verifier was fetched and inspected, and the request was corrected to `ba85dc58d7332c86832cbb22e51a813c55f3e439`. The correction touched only the request file and produced the successful run above. No scientific source or decision rule changed.

The full logs and artifacts are retained in Actions. This documentation commit does not trigger another experiment. The reproduction PR remains draft and unmerged; no manuscript claim is automatically promoted.
