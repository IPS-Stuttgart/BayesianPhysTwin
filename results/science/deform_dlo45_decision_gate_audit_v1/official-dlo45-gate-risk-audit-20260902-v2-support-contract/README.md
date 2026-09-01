# DEFORM DLO4/DLO5 matched-coverage gate and support audit

The primary certificate is unchanged. Heuristic thresholds are fitted
only on the pre-existing source-test partitions.

## Frozen certificate

- nonfallback: **82 / 532**
- RMSE reduction: **4.27%**
- harmful nonfallback: **3**
- p95 normalized regret: **0.8544**

## Source-calibrated gates

| Gate | Coverage | RMSE reduction | Harm/nonfallback | p95 regret |
| --- | ---: | ---: | ---: | ---: |
| `quotient_concentration` | 13.7% | 3.97% | 1/73 | 0.8605 |
| `maximum_quotient_mass` | 12.8% | 3.86% | 1/68 | 0.8605 |
| `maximum_kernel_weight` | 16.9% | 4.86% | 0/90 | 0.8515 |
| `expected_fallback_advantage` | 11.8% | 7.13% | 1/63 | 0.8401 |
| `expected_action_gap` | 12.2% | 3.88% | 0/65 | 0.8435 |
| `hypothesis_action_agreement` | 12.2% | 5.57% | 2/65 | 0.8590 |
| `negative_residual_disagreement` | 18.6% | 3.20% | 5/99 | 0.8563 |
| `negative_unsupported_specificity` | 16.7% | 2.92% | 5/89 | 0.8635 |
| `deterministic_random` | 13.5% | 3.97% | 0/72 | 0.8544 |

## Exact target-covariate coverage match (outcome-free secondary)

| Gate | RMSE reduction | Harm/nonfallback | p95 regret |
| --- | ---: | ---: | ---: |
| `quotient_concentration` | 4.24% | 1/82 | 0.8590 |
| `maximum_quotient_mass` | 4.19% | 3/82 | 0.8562 |
| `maximum_kernel_weight` | 4.23% | 2/82 | 0.8403 |
| `expected_fallback_advantage` | 8.70% | 1/82 | 0.8266 |
| `expected_action_gap` | 4.87% | 1/82 | 0.8424 |
| `hypothesis_action_agreement` | 6.64% | 3/82 | 0.8590 |
| `negative_residual_disagreement` | 2.50% | 5/82 | 0.8580 |
| `negative_unsupported_specificity` | 2.95% | 4/82 | 0.8627 |
| `deterministic_random` | 4.60% | 0/82 | 0.8555 |

## Registered-support audit

- harmful certified decisions: **3**
- safe certified decisions: **79**
- support-distance AUC for harm: **0.4008438818565401**
- regret-excess AUC for harm: **0.7510548523206751**

## Claim boundary

This retrospective follow-up compares the already frozen certificate with source-calibrated heuristic gates. Primary thresholds use only official DEFORM training trajectories assigned to the pre-existing source-test partition. The target-covariate matched analysis uses held prefixes/actions and scores but never held internal-node outcomes to choose coverage. Results remain within-DLO held-trajectory evidence and do not establish unseen-object generalization, calibrated probabilities, arbitrary-action safety, or deployment authorization.
