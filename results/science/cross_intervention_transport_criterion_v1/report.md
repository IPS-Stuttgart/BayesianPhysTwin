# Cross-intervention transport criterion V1

## Decision

**criterion-useful-but-requires-declared-nuisance-and-conservative-guard**

The criterion is useful as a falsification test for source-local discrepancy, but it is not sufficient without declared nuisance and identifiability checks.

## Controlled claim rates

| Regime | Source-only | Transport-only | Full protocol |
|---|---:|---:|---:|
| `transportable_physical` | 100.0% | 100.0% | 38.4% |
| `source_local_discrepancy` | 100.0% | 1.0% | 0.0% |
| `correlated_local_discrepancy` | 100.0% | 0.9% | 0.0% |
| `shared_action_independent_bias` | 100.0% | 0.2% | 0.0% |
| `action_aligned_undeclared_nuisance` | 100.0% | 98.0% | 28.4% |
| `action_aligned_declared_nuisance` | 100.0% | 0.0% | 0.0% |
| `physical_conservative_transport` | 100.0% | 100.0% | 78.5% |
| `physical_sign_error` | 100.0% | 0.0% | 0.0% |
| `physical_local_mixture` | 100.0% | 41.2% | 0.4% |

## Registered checks

- PASS — `source_fit_falsely_accepts_local_discrepancy`
- PASS — `transport_rejects_source_local_discrepancy`
- PASS — `full_protocol_rejects_local_and_shared_bias`
- PASS — `transport_detects_registered_physical_mechanism`
- PASS — `undeclared_action_aligned_nuisance_can_fool_transport`
- PASS — `declared_nonidentifiability_fails_closed`
- PASS — `simulator_sign_error_is_not_misreported_as_transport`
- PASS — `conservative_transport_improves_full_protocol_power`

## Interpretation

1. Same-action source improvement is non-diagnostic: it accepts the source-local discrepancy null in nearly every trial.
2. Held-out intervention transport sharply reduces that false physical attribution while retaining high sensitivity to the registered shared physical coefficient.
3. An undeclared nuisance with exactly the physical action signature can still fool transport. Declaring the competing nuisance makes the query nonidentifiable and the full method correctly returns exact fallback.
4. The complete protocol is substantially more conservative than the transport endpoint. In this finite-session design, conservative correction magnitude improves the probability of satisfying the harmful-update gate.

## Boundary

Controlled local-linear mechanism evidence only. A positive result shows that held-out intervention transport can reject source-local discrepancy under the registered simulation assumptions. It does not establish a unique physical cause, simulator adequacy, real-object transfer, real calibration, provider competence, Causal4D physical benefit, deployment safety, or state of the art.
