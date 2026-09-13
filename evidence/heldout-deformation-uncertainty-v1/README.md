# Completed held-out deformation uncertainty experiment

## Decision

**The primary hypothesis was not established.** The joint Bayesian predictive distribution substantially outperforms independently calibrated diagonal uncertainty, but it does not conclusively beat the centered whole-trajectory bootstrap. The symmetric trajectory bootstrap has a better Brier score.

This is a completed scientific result, not a job failure. No scientific retuning or retry followed scoring.

- [Completed Actions run 33985033404](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33985033404)
- [Complete result artifact 9974910819](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33985033404/artifacts/9974910819)
- [Implementation and frozen protocol](../../experiments/heldout_deformation_uncertainty_v1/README.md)
- [Compact machine-readable summary](result_summary.json)

Execution was triggered by the request-only commit `ecbe195997dcc00419555c9c853a4a7fcc73f557`, using `gpuserver4090` on runner `workstation1`.

## What was tested

DEFORM DLO4 and DLO5 each retain the original 39 fit / 9 calibration / 8 source-test trajectory split. The existing source-qualified hybrid physical predictor and BayesianPhysTwin residual mean were replayed without retraining. Saved source-test mean predictions were reproduced with maximum absolute discrepancy **0.0 m on both DLOs**.

A new covariance-only empirical-Bayes inverse-Wishart/Student predictive layer was fitted around that mean. This is not a test of the original stored BPT covariance. The comparison includes calibrated diagonal Student uncertainty, centered whole-trajectory bootstrap uncertainty, a symmetric whole-trajectory bootstrap, and a matched-covariance Gaussian.

All five methods retain exactly the same mean. Each receives its own leave-one-calibration-trajectory-out scale fit using centroid quantities only. The primary test equally weights three held-out families: relative displacement, local second differences, and temporal centroid displacement. Recorded event labels indicate whether the absolute quantity exceeds 5, 10, or 20 mm.

## Results

| Uncertainty representation | Brier, lower better | CRPS, mm, lower better | Nominal-90% coverage | Mean interval width, mm |
|---|---:|---:|---:|---:|
| Joint Bayesian Student | 0.093091 | 9.9230 | 90.51% | 60.825 |
| Calibrated diagonal Student | 0.132384 | 15.0180 | 94.10% | 152.770 |
| Centered trajectory bootstrap | 0.094991 | 10.3755 | 70.20% | 34.051 |
| Symmetric trajectory bootstrap | 0.087242 | 9.7658 | 82.73% | 46.314 |
| Joint covariance-matched Gaussian | 0.094090 | 9.9734 | 90.51% | 61.416 |

The joint Student model reduces Brier by **29.68%** and CRPS by **33.93%** relative to calibrated diagonal uncertainty. However, its Brier is **6.70% worse** than the symmetric trajectory bootstrap.

The two primary paired Brier differences (joint minus comparator) are:

| Comparator | Difference | 97.5% paired-bootstrap interval | Frozen decision |
|---|---:|---:|---|
| Calibrated diagonal | -0.039293 | [-0.046451, -0.033495] | Pass |
| Centered trajectory bootstrap | -0.001900 | [-0.006550, +0.003125] | Fail |

The second comparison also changes sign across DLOs: -0.006567 on DLO4 and +0.002766 on DLO5. The protocol required a negative contrast on each DLO as well as the corrected interval criterion against both primary controls.

## Where the diagonal advantage originates

| Held-out family | Joint Student Brier | Diagonal Brier | Centered bootstrap Brier | Symmetric bootstrap Brier |
|---|---:|---:|---:|---:|
| Relative displacement | 0.062729 | 0.063833 | 0.071952 | 0.061562 |
| Local second difference | 0.161774 | 0.276664 | 0.149234 | 0.142207 |
| Temporal displacement | 0.054769 | 0.056654 | 0.063786 | 0.057956 |

The large gain over diagonal uncertainty comes mainly from second-difference queries. Calibrating diagonal noise to centroid uncertainty does not preserve cancellation in relative/bending quantities. Empirical trajectory distributions preserve such dependence too, explaining why the diagonal result alone is not a Bayesian-specific superiority claim.

The overall 90.51% coverage does not prove calibration: family coverage is 86.46%, 97.57%, and 87.50%, respectively, and the cohort is small. Interval width and proper scores must remain visible.

## Verification and reproducibility

Downloaded artifact SHA256: `fc69cc01f014a7591ae2d3de3dc9a1d8acb76ca06c76a902cb06937b827c457c`.

Canonical result SHA256: `1c051bc95d9d19e1209ff3a2dc0c71b741340be7c5044f5718673e290dd1a77e`.

An independent compact-score verifier recomputed all Brier aggregates and paired bootstrap intervals from **23,760 method-event rows**, checked agreement of the **4,752 recorded event labels**, verified the result content hash, and reproduced the failed primary decision. This is score-level verification, not another raw simulator or source-fit rerun. The verification script is `experiments/heldout_deformation_uncertainty_v1/verify_scores.py`.

A prior execution, run 33984874637, stopped before trajectory loading/scoring because Git rejected the shared checkout's ownership. Its failure artifact 9974850196 is retained. A process-local trust setting for that exact caller-authorized path repaired execution without weakening revision checks or changing the scorer/protocol.

## Claim boundary

This is retrospective source-only evidence from **16 trajectories on two fixed objects**. Bootstrap intervals resample whole trajectories within each DLO; they are not independent-object population intervals. Neither official evaluation partitions nor unrelated reserved data were opened. There were no new physical measurements, probes, or robot actions.

Supported: joint uncertainty is useful relative to calibrated diagonal uncertainty for the evaluated output-family transfer.

Not established: a Bayesian-specific advantage over credible empirical trajectory distributions, fresh confirmation, unseen-object generalization, or calibrated deployment safety.
