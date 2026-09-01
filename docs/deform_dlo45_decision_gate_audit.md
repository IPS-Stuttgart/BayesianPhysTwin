# DEFORM decision-gate audit: utility versus support admissibility

This audit is a frozen follow-up to the completed finite-action DEFORM DLO4/DLO5 experiment. It tests whether the exact query-quotient certificate is empirically necessary relative to inexpensive confidence gates.

## Question

For a candidate Jeffrey correction action, can a scalar gate reproduce the certificate's held-data benefit at comparable update coverage?

The audit compares nine source-calibrated scores: quotient concentration, maximum quotient mass, maximum kernel weight, expected fallback advantage, expected action gap, hypothesis-action agreement, negative residual disagreement, negative unsupported specificity, and a deterministic random control. Thresholds are fitted only on the previously registered source-test partitions. No held internal-node outcome is used for fitting, thresholding, coverage matching, or retries.

## Result

The exact certificate selects 82 of 532 actions, reduces aggregate RMSE by 4.27%, and has three harmful nonfallback decisions on held data. All 82 selected actions satisfy the registered finite-support regret tolerance of `0.05`; their maximum registered worst-case regret is `0.0498734`.

The best source-calibrated heuristic is expected fallback advantage. It selects 63 actions, reduces aggregate RMSE by 7.13%, and has one harmful held update. However, 51 of its 63 actions violate the same registered support contract, its p95 registered worst-case regret is `0.2589`, and its maximum is `0.3135`. At exactly 82 selected target-covariate decisions, the heuristic reduces RMSE by 8.70% but violates the support contract on 66 actions.

Thus, the certificate is not the empirical utility optimum on this cohort. It supplies a different property: exact admissibility over the declared finite support. Simple heuristics can trade away that support-wise guarantee for higher average held utility.

## Interpretation

The paper should present a two-axis comparison:

1. **Realized utility:** held RMSE, observed harmful updates, and trajectory-level effects.
2. **Registered admissibility:** support-violation rate and worst-case regret over every complete belief represented by the quotient.

Neither axis subsumes the other. In particular, zero observed harm in 532 decisions does not imply support-wise admissibility, and support-wise admissibility does not imply zero harm under a held trajectory outside the finite support.

## Support mismatch

For the exact certificate, held realized regret exceeds the registered source-support bound on 54.88% of selected decisions. This is not a theorem violation: the guarantee is conditional on the registered finite support. Two of the three held harmful decisions show large realized-regret excess. Nearest source-residual distance does not identify those failures reliably.

Consequently, the next method extension should be an explicit **support-adequacy gate** or a conformal/out-of-support envelope, not a claim that the current finite-support certificate provides arbitrary real-world safety.

## Submission claim

A bounded ICRA-level claim is:

> We separate empirical utility from decision admissibility under partial physical identification. An exact query-quotient certificate enforces a registered worst-case-regret bound and exact fallback, whereas source-calibrated confidence gates may improve average held accuracy by selecting actions that are not admissible under the same represented physical support.

Do not claim that the certificate dominates heuristic gates in RMSE, held harm rate, unseen-object generalization, calibration, or deployment safety.
