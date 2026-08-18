# DEFORM DLO2 local-residual official v7

This is the single-use official DLO2 evaluation of the source-confirmed all-56
method. Before enumerating an evaluation file, the runner verifies the all-train
protocol, result, final physical checkpoint, local-residual NPZ, window schedule,
runtime, and fixed arm, then writes an authorization record with
`official_eval_read=false`.

The evaluator reads all 14 sorted official cases exactly once and also reports
the canonical with-replacement draw needed to compare with the published DEFORM
DLO2 value of 9.7 mm. The candidate is compared against the identically trained
6,400-update physical checkpoint. The locked claim requires:

- candidate mean below 9.7 mm on all 14 unique cases;
- candidate mean below 9.7 mm on the canonical reference draw;
- at least 1% improvement over the identical physical checkpoint;
- at least 8 of 14 paired wins;
- no case ratio above 1.10; and
- all 14 expected cases present.

The all-train covariance is reused unchanged, in metric units of square metres.
Target selection, calibration, retries, and case replacement are forbidden. A
runtime failure after target opening is sealed and cannot authorize a retry.

Protocol: `configs/sota/deform_dlo2_local_residual_official_v7.json`.

This public document defines the executable protocol only. The final compact
outcome, published-operator audit, custody record, and interpretation are
canonical in the private
[BayesianPhysTwin-Paper record](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/deform_dlo2_local_residual_official_v7_result.md).
