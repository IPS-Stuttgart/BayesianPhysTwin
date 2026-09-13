# Shared-alpha joint anytime admission v2

- Decision: **shared-alpha-iut-efficiency-supported**
- Mechanism gate: **True**
- Shared first-epoch alpha: **0.025**
- Shared component threshold: **40.0**
- Split component threshold: **80.0**

## Controlled scenarios

| Scenario | Method | Crossing probability | Wilson 95% | Median crossing |
|---|---|---:|---:|---:|
| `gain_boundary_low_harm` | `shared_alpha_iut` | 0.0083 | [0.0072, 0.0097] | 237.0 |
| `gain_boundary_low_harm` | `bonferroni_split` | 0.0036 | [0.0029, 0.0045] | 257.0 |
| `harm_boundary_positive_gain` | `shared_alpha_iut` | 0.0148 | [0.0132, 0.0166] | 94.0 |
| `harm_boundary_positive_gain` | `bonferroni_split` | 0.0070 | [0.0060, 0.0083] | 103.0 |
| `moderate_safe_benefit` | `shared_alpha_iut` | 0.8999 | [0.8939, 0.9056] | 214.0 |
| `moderate_safe_benefit` | `bonferroni_split` | 0.8501 | [0.8430, 0.8570] | 242.0 |
| `strong_safe_benefit` | `shared_alpha_iut` | 1.0000 | [0.9996, 1.0000] | 73.0 |
| `strong_safe_benefit` | `bonferroni_split` | 1.0000 | [0.9996, 1.0000] | 81.0 |

## Efficiency comparison

- Maximum shared-IUT null Wilson upper bound: **0.0166**.
- Moderate-alternative power gain, shared minus split: **+0.0498**.
- Moderate-alternative median crossing ratio, shared over split: **0.8843**.

## Theorem boundary

The invalid-candidate null is the union of insufficient mean gain and excessive harm rate. Joint admission requires both latched component e-processes to cross the same epoch-wise alpha boundary. If either fixed component null holds throughout the epoch, false admission is bounded by that alpha without a Bonferroni split.

Controlled Monte Carlo evidence for the shared-alpha intersection--union admission mechanism. It is not fresh real-world validation, a universal power guarantee, or a physical-safety certificate.
