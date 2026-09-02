# Controlled Transport4D tier separation

Decision: **controlled-tier-separation-passed**

| Case | Expected | Selected | Action | Exact fallback |
|---|---|---|---|---:|
| `same-object-cross-backend` | `exact_coefficients` | `exact_coefficients` | `execute` | false |
| `known-coordinate-pushforward` | `query_identifiable_effect` | `query_identifiable_effect` | `execute` | false |
| `amplitude-recalibration` | `low_dimensional_correction` | `low_dimensional_correction` | `execute` | false |
| `mean-unsupported-dependence-retained` | `uncertainty_only` | `uncertainty_only` | `hold` | true |
| `cross-object-coefficient-failure` | `procedure_only` | `procedure_only` | `hold` | true |
| `query-conditional-descent` | `query_identifiable_effect` | `query_identifiable_effect` | `execute` | false |
| `no-supported-transport` | `None` | `None` | `hold` | true |

The strict hierarchy is exact coefficients, a query-identifiable
effect, low-dimensional correction, uncertainty only, and
procedure only. Deterministic mean tiers additionally need a unique
finite-action decision inside the registered robust-regret budget.

The certificate is exact only for the supplied tier candidates, evidence checks, query effects, Euclidean error radii, affine action losses, and regret tolerance. It does not validate a physical transformation, infer an error radius, establish exchangeability, prove nonlinear closure, authorize target-data access, certify deployment safety, or establish state of the art.
