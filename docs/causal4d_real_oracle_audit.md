# Causal4D Real Oracle-Gap Audit

Status: completed on `single_lift_sloth` on 2026-07-12.

This is a diagnostic result, not a new estimator. It freezes the Causal4D
architecture and asks which subsystem limits the real held-out prediction:

```text
inference gap = Causal4D posterior - current-bank oracle
proposal gap  = current-bank oracle - expanded-bank oracle
model gap     = expanded-bank oracle - labeled discrepancy ceiling
```

Every oracle selects or fits on the held-out labels and is marked
`diagnostic_only`, `deployable=false`. The six-frame `O+` prefix is the only
post-intervention evidence available to BPT or Causal4D prediction.

## Protocol

- case: `single_lift_sloth`
- endpoint: released frame 58
- abduction evidence: six `O+` frames
- untouched evaluation: rollout frames `[7, 27)`, 68,867 valid point-frames
- current bank: 9 interventions x 4 physical particles
- expanded bank: the complete 108-intervention grid x 4 particles
- deterministic official Warp rollout: 667 substeps per video frame
- oracle selection metric: mean Euclidean track error

The audit verifies that all nine current hypotheses occur at expanded-bank
indices 0 through 8 and that their trajectories are bit-identical: maximum
absolute difference `0 m`.

## Predictor and oracle ladder

| Predictor | Coordinate RMSE | Track error | Label use for prediction |
| --- | ---: | ---: | --- |
| Released nominal PhysTwin | 25.465 mm | 36.660 mm | none |
| Bayesian-PhysTwin, nominal `z` | 22.494 mm | 32.130 mm | six-frame prefix only |
| Current Causal4D posterior | 22.260 mm | 31.694 mm | six-frame prefix only |
| Current 36-component oracle | 20.492 mm | 29.378 mm | holdout diagnostic |
| Expanded 432-component oracle | 20.311 mm | 29.071 mm | holdout diagnostic |
| Expanded + global translation oracle | 18.084 mm | 25.027 mm | holdout diagnostic |
| Expanded + capped 10 mm point field | 15.834 mm | 20.633 mm | holdout diagnostic |
| Expanded + uncapped point field | 6.700 mm | 8.399 mm | holdout diagnostic |

Bayesian-PhysTwin removes `4.531 mm` of track error from nominal PhysTwin.
Causal4D intervention marginalization removes another `0.436 mm` (`1.36%`)
from the Bayesian-PhysTwin predictor.

The expanded winner uses the same `grid_3_4` physical particle, zero attachment
shift, zero delay, zero slip, and `-8 degree` controller-frame rotation as the
current-bank winner. The only new winning variable is gain `0.85` instead of
`1.0`. The larger bank therefore finds a small adjustment rather than a missing
causal mode.

## Gap result

| Gap | Coordinate RMSE | Track error | Track share of total headroom |
| --- | ---: | ---: | ---: |
| Inference | 1.768 mm | 2.316 mm | 9.94% |
| Proposal | 0.181 mm | 0.307 mm | 1.32% |
| Model | 13.611 mm | 20.672 mm | **88.74%** |

The uncapped point field is an intentionally optimistic in-sample ceiling. The
conclusion survives the physically conservative 10 mm cap: inference accounts
for 20.94%, proposal expansion 2.77%, and model discrepancy **76.29%** of capped
headroom.

Global translation recovers only `4.043 mm` of the expanded-oracle-to-ceiling
track gap. Across the 4,147 nodes visible at least once, the uncapped winning
field has mean point norm `33.40 mm`, 95th percentile `79.60 mm`, and 91.3% of
points above the 10 mm cap. It is evidence for a large, spatially structured,
quasi-static mismatch, not a correction that may be deployed.

## Variance audit

The current posterior's coordinate-wise predictive variance is allocated with
a weighted Shapley decomposition of conditional means. This handles posterior
dependence between `theta`, persistent `phi`, and event-specific `kappa`.
Discrepancy mean, conditional discrepancy variance, the state-discrepancy cross
term, and the configured variance floor remain explicit.

| Source | Variance share | Root-equivalent scale |
| --- | ---: | ---: |
| Conditional discrepancy | 60.66% | 8.134 mm |
| Configured conditional noise floor | 22.92% | 5.000 mm |
| Event intervention `kappa` | 10.97% | 3.459 mm |
| Physical parameters `theta` | 3.82% | 2.042 mm |
| Persistent actuation `phi` | 2.15% | 1.531 mm |
| Discrepancy-mean epistemic | 0.11% | 0.352 mm |
| State-discrepancy covariance | -0.63% | -0.832 mm signed root |

The ledger closes to below `5e-21 m^2`. Total predictive standard deviation is
`10.44 mm`, but empirical coordinate RMSE is `22.26 mm`; residual MSE is 4.54
times predictive variance. The ratio worsens from 2.69 early to 4.64 middle and
6.14 late. The posterior broadens with horizon, but much more slowly than the
real error grows.

The noise floor is a configured combined simulator/observation proxy. This
audit does not identify replay and observation noise separately.

## Decision

The primary real-data limitation is model/state discrepancy. It is not the
width of the handcrafted intervention bank, and simply retaining more
`theta` particles is unlikely to close the observed gap.

The next model milestone is:

1. fit frame/gravity and graph-smooth rest-configuration or rest-length
   corrections using `O-` plus the declared `O+` prefix only;
2. inject each correction into the PhysTwin state/model and rerun Warp;
3. compare global registration, output correction, state correction, and
   physical rest-geometry correction on untouched future actions;
4. localize remaining error at support/contact regions;
5. then run the same-object multi-action real protocol and revisit intervention
   inference.

Do not tune the 108-state bank further before this model-side test. The
proposal oracle improved track error by only `0.307 mm`.

## Reproduction

The CLI writes a provenance-complete JSON summary and one row per component:

```bash
causal4d-audit-real-oracle-gap \
  current.bank.npz expanded108.bank.npz belief.npz physical.npz \
  CASE/final_data.pkl CASE/inference.pkl \
  real_oracle_audit.json real_oracle_components.csv
```

`scripts/remote/run_causal4d_real_oracle_audit.sh` builds the expanded bank and
runs this audit with configurable artifact roots.
