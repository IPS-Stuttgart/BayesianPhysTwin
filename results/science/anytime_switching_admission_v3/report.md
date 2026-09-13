# Switching-null anytime admission v3

- Decision: **switching-null-robustness-supported**
- Mechanism gate: **True**
- First-epoch alpha: **0.025**
- E-value threshold: **40.0**

## Controlled scenarios

| Scenario | Method | Crossing probability | Wilson 95% | Median crossing |
|---|---|---:|---:|---:|
| `stable_gain_boundary` | `latched_shared_alpha_iut` | 0.0080 | [0.0068, 0.0093] | 237.0 |
| `stable_gain_boundary` | `switching_union_min_score` | 0.0053 | [0.0044, 0.0064] | 210.0 |
| `stable_harm_boundary` | `latched_shared_alpha_iut` | 0.0148 | [0.0132, 0.0166] | 94.0 |
| `stable_harm_boundary` | `switching_union_min_score` | 0.0112 | [0.0099, 0.0128] | 150.0 |
| `switching_invalidity` | `latched_shared_alpha_iut` | 1.0000 | [0.9998, 1.0000] | 229.0 |
| `switching_invalidity` | `switching_union_min_score` | 0.0000 | [0.0000, 0.0002] | None |
| `moderate_safe_benefit` | `latched_shared_alpha_iut` | 0.8947 | [0.8885, 0.9006] | 214.0 |
| `moderate_safe_benefit` | `switching_union_min_score` | 0.7359 | [0.7272, 0.7444] | 252.0 |
| `strong_safe_benefit` | `latched_shared_alpha_iut` | 1.0000 | [0.9996, 1.0000] | 73.0 |
| `strong_safe_benefit` | `switching_union_min_score` | 1.0000 | [0.9996, 1.0000] | 108.0 |

## Assumption stress test

- Latched IUT crossing under switching invalidity: **1.0000**.
- Robust minimum-score crossing under the same stream: **0.0000**.
- Maximum robust null Wilson upper bound: **0.0128**.
- Moderate safe-benefit robust power: **0.7359**.
- Strong safe-benefit robust power: **1.0000**.

## Interpretation

The efficient latched intersection--union rule is valid only when one component null holds throughout an epoch. In the registered switching stream, gain evidence is accumulated while harm is excessive and harm evidence is accumulated later while mean gain is negative. Latching can therefore authorize outside its theorem boundary. The minimum-score process remains valid because its score is no larger than whichever component has nonpositive conditional expectation at each reveal.

Controlled stress evidence for the switching-null certificate and for the necessity of the stable-null assumption behind the more efficient latched intersection--union rule. It is not fresh real-world validation or a physical-safety guarantee.
