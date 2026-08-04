# Controlled Prob4D-to-BayesianPhysTwin decision study

Decision: **PASS**

| Method | Deployed RMSE | Improvement | Accept | Harmful accepted |
| --- | ---: | ---: | ---: | ---: |
| B0_physical_fallback | 6.166 mm | +0.00% | 0.0% | 0 |
| B1_naive_last_frame_state | 6.166 mm | +0.00% | 0.0% | 0 |
| P1_marginal_gauge_persistent | 1.231 mm | +80.04% | 96.9% | 8 |
| P2_explicit_gauge_framewise | 0.719 mm | +88.34% | 97.7% | 0 |
| P3_explicit_gauge_persistent | 0.534 mm | +91.33% | 97.1% | 0 |
| P4_explicit_gauge_persistent_metric_anchor | 0.551 mm | +91.06% | 96.6% | 0 |

## Registered criteria

- PASS: `mean_improvement_at_least_registered`
- PASS: `paired_upper_bound_below_zero`
- PASS: `harmful_accepted_rate_at_most_registered`
- PASS: `worst_scenario_regression_at_most_registered`
- PASS: `all_rejections_exact_fallback`
- PASS: `explicit_persistent_noninferior_to_marginal`

## Claim boundary

Controlled calibration/target-separated synthetic evidence only. A pass authorizes a fresh physical-object or acquisition-session experiment; it does not establish real-world Prob4D provider competence, calibrated deployment uncertainty, BayesianPhysTwin physical benefit on an independent cohort, or Causal4D intervention benefit.
