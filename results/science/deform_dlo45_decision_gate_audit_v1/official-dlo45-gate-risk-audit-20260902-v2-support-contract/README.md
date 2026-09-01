# DEFORM DLO4/DLO5 matched-coverage and support-contract audit

**Status:** complete, source-frozen retrospective follow-up to workflow run `33473378340`.

Audit workflow run: `33540426989`  
Audit artifact: `9813375113` (`sha256:d8a758dd4102b0d44c9567e3bf4c7f0ded4436a6c2bb180b9d4cf44d826fcd37`)  
Dataset: `roahmlab/DEFORM@b73b8b8ecc033caefa693fab7898741d4e6dbeff`

The primary certificate and its 532 held decisions are unchanged. Every heuristic threshold was fitted only on the pre-existing official-training source-test partition. The held prefixes and future endpoint-action carriers may be used as covariates, but held internal-node outcomes never determine a threshold or coverage level.

## Primary result: utility versus registered-support admissibility

| Policy | Nonfallback | Held RMSE reduction | Harmful nonfallback | Registered-support violations | p95 registered worst-case regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact finite-support certificate | 82 / 532 | 4.27% | 3 / 82 | **0 / 82** | **0.0468** |
| Best source-calibrated heuristic: expected fallback advantage | 63 / 532 | **7.13%** | **1 / 63** | 51 / 63 | 0.2589 |
| Same heuristic, exactly matched target-covariate coverage | 82 / 532 | **8.70%** | **1 / 82** | 66 / 82 | 0.2572 |

The registered regret tolerance is `0.05`. The certificate's maximum registered worst-case regret among its nonfallback decisions is `0.0498734`; all 82 satisfy the contract. The best source-calibrated heuristic has maximum registered regret `0.313480`, and 80.95% of its nonfallback decisions violate the same contract.

The certificate therefore does **not** dominate heuristics in realized held-data utility. Its empirical role is different: it is the only compared policy constrained to satisfy the registered finite-support regret contract. The result is a utility--admissibility Pareto separation, not an empirical safety or accuracy dominance claim.

## Source-calibrated heuristic results

| Gate | Nonfallback | RMSE reduction | Harm | Support violations |
| --- | ---: | ---: | ---: | ---: |
| Expected fallback advantage | 63 | 7.13% | 1 | 51 |
| Hypothesis-action agreement | 65 | 5.57% | 2 | 50 |
| Maximum kernel weight | 90 | 4.86% | 0 | 76 |
| Exact certificate | 82 | 4.27% | 3 | 0 |
| Quotient concentration | 73 | 3.97% | 1 | 62 |
| Deterministic random | 72 | 3.97% | 0 | 63 |
| Expected action gap | 65 | 3.88% | 0 | 38 |
| Maximum quotient mass | 68 | 3.86% | 1 | 64 |
| Negative residual disagreement | 99 | 3.20% | 5 | 88 |
| Negative unsupported specificity | 89 | 2.92% | 5 | 74 |

Observed harmful-update counts alone are insufficient to establish support-wise admissibility: several gates happened to have zero held harms while violating the registered support contract on most selected updates.

## Registered-support mismatch audit

The certificate controls regret only over the registered source-supported complete beliefs. Held realized regret exceeded that source-support bound in 54.88% of its 82 nonfallback decisions. Three certified decisions were worse than fallback on the held outcome; two had large positive realized-regret excess. Nearest-support residual distance was not a useful detector of those harms (`AUC=0.401`), while realized regret excess was more diagnostic (`AUC=0.751`) but is unavailable before the held outcome.

This distinction is essential:

- `0/82` registered-support violations means every certificate action obeyed the finite-support theorem and tolerance;
- `3/82` held harms and frequent bound exceedance show that the registered finite support is not an arbitrary real-world safety envelope.

## Claim boundary

A defensible claim is:

> Exact decision certification enforces a preregistered worst-case-regret contract over every complete belief represented by the finite query quotient. On held DEFORM trajectories this contract trades empirical utility for admissibility: source-calibrated heuristic gates improve average RMSE more, but violate the same registered support bound on most selected updates.

The audit remains within-DLO held-trajectory evidence. It does not establish unseen-object generalization, calibrated probabilities, arbitrary-action safety, deployment authorization, or empirical dominance of the certificate.

## Files

- `compact_result.json`: principal comparison and claim boundary;
- `target_audit.json`: complete method, trajectory, support and sensitivity summaries;
- `per_decision.jsonl`: outcome-free gate scores, registered action regrets and support masks for all 532 decisions;
- `thresholds.json`: source-frozen scalar thresholds;
- `source_audit.json` and `source_seal.json`: source-only comparison and content binding;
- `provenance.json`: workflow, dataset and no-retry/no-tuning record.
