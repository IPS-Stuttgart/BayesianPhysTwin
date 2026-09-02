# Independently tuned factor-envelope admission v4

- Decision: **factor-envelope-efficiency-supported**
- Confirmation gate: **True**
- First-epoch alpha: **0.025**
- E-value threshold: **40.0**
- Cartesian factor components: **30**

## Confirmation scenarios

| Scenario | Method | Crossing probability | Wilson 95% | Median crossing |
|---|---|---:|---:|---:|
| `stable_gain_boundary` | `switching_union_min_score_v3` | 0.0046 | [0.0035, 0.0061] | 210.0 |
| `stable_gain_boundary` | `switching_union_factor_envelope_v4` | 0.0028 | [0.0019, 0.0040] | 226.0 |
| `stable_harm_boundary` | `switching_union_min_score_v3` | 0.0128 | [0.0108, 0.0152] | 116.0 |
| `stable_harm_boundary` | `switching_union_factor_envelope_v4` | 0.0082 | [0.0066, 0.0102] | 124.0 |
| `switching_invalidity` | `switching_union_min_score_v3` | 0.0000 | [0.0000, 0.0004] | None |
| `switching_invalidity` | `switching_union_factor_envelope_v4` | 0.0000 | [0.0000, 0.0004] | None |
| `moderate_safe_benefit` | `switching_union_min_score_v3` | 0.7430 | [0.7307, 0.7549] | 252.0 |
| `moderate_safe_benefit` | `switching_union_factor_envelope_v4` | 0.8334 | [0.8228, 0.8435] | 280.0 |
| `strong_safe_benefit` | `switching_union_min_score_v3` | 1.0000 | [0.9992, 1.0000] | 108.0 |
| `strong_safe_benefit` | `switching_union_factor_envelope_v4` | 1.0000 | [0.9992, 1.0000] | 122.0 |

## Power and robustness

- Maximum factor-envelope null Wilson upper bound: **0.0102**.
- Switching-null factor-envelope crossing probability: **0.0000**.
- Moderate factor-envelope power: **0.8334**.
- Moderate minimum-score power: **0.7430**.
- Moderate power gain, envelope minus minimum score: **+0.0904**.
- Median crossing ratio, envelope over minimum score: **1.1111**.
- Strong factor-envelope power: **1.0000**.

## Theorem

For every fixed tuple of component parameters, the update factor is the minimum of the registered component e-factors. Under the pointwise union null, at least one component factor has conditional expectation at most one at every reveal. The minimum is dominated by that active valid factor, even when the active component changes. Products over time and an outcome-independent mixture over fixed tuples therefore remain anytime-valid.

Controlled confirmation evidence for the independently tuned factor-envelope construction. Scenario families were inherited from the already observed version-3 controlled study. A separate version-4 pilot roster was used only to choose the frozen gates; the retained confirmation seed roster was not opened before this protocol was committed. No real outcomes are used.
