# Causal4D Real Undercoverage Audit

Status: parameter-support, graph-discrepancy, and locked source-transfer
diagnostics completed on 2026-07-12; calibration remains below its
independent-trial claim gate.

This work starts only after the locked real oracle-gap audit. That audit assigns
`88.74%` of uncapped predictive headroom to model/state discrepancy, `9.94%` to
inference, and `1.32%` to intervention proposals. Consequently, this milestone
does not tune the intervention beam and does not inflate target covariance in
place.

## Information boundary

- `O-` fits the graph discrepancy basis and coefficient dynamics.
- The first six `O+` frames estimate the target discrepancy coefficients.
- Later `O+` frames are read only by diagnostic metrics.
- Parameter support is selected from the parameter posterior; stabilization is
  measured against full-support predictive moments, without target labels.
- Affine variance parameters use `double_lift_sloth` as fit source and
  `double_stretch_sloth` as a disjoint calibration source.
- `single_lift_sloth` remains the target and cannot tune `a`, `b`, rank, support,
  or likelihood temperature.

Real point-frames are spatially and temporally correlated. They are not counted
as independent calibration trials. The calibration artifact therefore carries
an explicit independent-execution count and remains `claim_ready=false` until
at least ten held-out calibration executions are available.

## Parameter-support convergence

The complete `9 x 9` profile was replayed once, producing 81 particle-specific
endpoint beliefs and `81 x 9 = 729` deterministic Warp rollouts. The rollout
bank is the common reference for every reduction.

| Support | Direct mass | Mean delta to full | Variance delta | Track error | Coverage | NEES |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 4 | 42.33% | 1.435 mm | 7.9% | 31.694 mm | 50.59% | 7.23 |
| Top 8 | 65.26% | 1.129 mm | 5.8% | 31.360 mm | 51.02% | 6.71 |
| Top 16 | 89.30% | 0.512 mm | 3.6% | 31.216 mm | 52.89% | 6.28 |
| Top 32 | 99.00% | 0.050 mm | 0.4% | 30.829 mm | 54.84% | 5.72 |
| Full 81 | 100% | 0 | 0 | 30.792 mm | 55.05% | 5.66 |
| Coreset 16 | represents 100% | 0.396 mm | 3.2% | 31.139 mm | 53.19% | 6.20 |
| Coreset 32 | represents 100% | 0.015 mm | 0.2% | 30.803 mm | 54.99% | 5.68 |

The label-free stability gate is mean RMSE at most `0.5 mm` and relative
marginal-variance error at most `5%` against all 81 particles. Top-mass support
first passes at `K=32`; deterministic weighted-medoids support first passes at
`K=16`. Full support improves track error by `0.902 mm` and coverage by only
`4.46` percentage points over top-4. Truncation is measurable but cannot explain
the real undercoverage.

The 81-particle endpoint plus rollout build took approximately 16 minutes on
`gpuserver6000`; the saved rollout bank is 1.1 GB. Once that common bank exists,
support inference takes `0.18 s` at `K=4`, `0.44 s` at coreset `K=16`, `0.76 s`
at `K=32`, and `1.74 s` at full support.

## Graph-temporal discrepancy

The discrepancy model uses low-frequency eigenvectors `U_r` of the symmetric
normalized spring-graph Laplacian. O-minus residual coefficients are projected
onto the basis and fit with a stable linear model:

```text
delta_t = U_r a_t
a_(t+1) = A a_t + eta_t
```

Rank is chosen only on an O-minus validation suffix. On
`single_lift_sloth`, rank 16 wins (`5.785 mm` one-step validation RMSE); the
unconstrained transition has spectral radius `1.118` and is clipped to `0.995`.
A constant-coefficient persistence control uses the same selected basis and
prefix projection.

| Method | Track error | Coordinate RMSE | Coverage | NEES | NLL | Energy score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current random-walk readout | 31.694 mm | 22.260 mm | 50.59% | 7.23 | -0.344 | 26.092 mm |
| State only | 35.124 mm | 24.327 mm | 42.93% | 12.67 | 2.192 | 30.093 mm |
| Graph-temporal AR | 26.847 mm | 20.204 mm | 60.50% | 6.35 | -0.846 | 22.494 mm |
| **Graph persistence** | **23.105 mm** | **17.702 mm** | **67.78%** | **4.99** | **-1.513** | **18.571 mm** |

The result repeats on the independent `double_lift_sloth` source: graph
persistence changes track error from `25.714` to `17.451 mm` and coverage from
`49.99%` to `76.56%`. Learned AR dynamics is worse there (`33.293 mm`, `50.35%`).
The supported conclusion is therefore a smooth persistent discrepancy field,
not successful learned residual dynamics.

### Where coverage fails

On the target, the current posterior already covers the direct attachment and
two-hop contact neighborhood at `97.13%` and `97.73%`. Its far-graph region has
only `48.33%` coverage and `32.824 mm` track error. Graph persistence raises
far-graph coverage to `66.59%` and lowers its error to `23.722 mm`.

Coverage still deteriorates strongly with horizon:

| Method | Early | Middle | Late | Worst group |
| --- | ---: | ---: | ---: | ---: |
| Current readout | 60.41% | 48.06% | 44.68% | 44.68% |
| Graph persistence | 87.06% | 65.88% | 53.11% | 53.11% |

This localizes the remaining problem away from the attachment itself and toward
a spatially broad model/rest-state mismatch whose forecast uncertainty grows
too slowly.

## Source-only affine calibration

The implemented transform is:

```text
Sigma_cal = a * Sigma_raw + b I
```

`a` and `b` first minimize equal-trial Gaussian NLL on fit executions. A
disjoint source-calibration split then estimates a held-out inflation factor.
Every artifact records fit IDs, calibration IDs, coordinate count, independent
trial count, dependence warning, and a checksum. Target IDs are rejected if
they overlap either source split.

The currently locked source assignment is:

| Role | Interaction | Method |
| --- | --- | --- |
| Affine fit | `double_lift_sloth` | graph persistence |
| Held-out source calibration | `double_stretch_sloth` | graph persistence |
| Untouched target | `single_lift_sloth` | graph persistence |

This two-source exercise is a transfer diagnostic, not a calibration claim.
The claim gate requires the same-object multi-action protocol and at least ten
independent calibration executions.

The locked result is negative. Graph persistence is undercovered on the fit
source (`76.56%`) but overcovered on the disjoint calibration source (`98.06%`).
The NLL fit gives `a=2.1544` and `b=1.899e-5 m^2`; the held-out source then
selects inflation `0.09963`, yielding final `a=0.21464` and
`b=1.892e-6 m^2`. Transferred unchanged to `single_lift_sloth`:

| Target graph-persistence uncertainty | Coverage | Worst group | NEES | NLL | Width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 67.78% | 53.11% | 4.99 | -1.513 | 24.25 mm |
| Source-calibrated | **43.03%** | **30.40%** | 20.11 | 5.358 | 12.13 mm |

The calibration source contains 626,427 valid coordinates but only one
independent execution. Its harmful transfer is retained as evidence that a
single global scale is action-dependent and not identified by the current
three-interaction dataset. No reverse split or target-tuned alternative is
selected after observing this result.

## Reproduction

Full parameter-support audit:

```bash
causal4d-audit-parameter-support \
  full81/known.bank.npz full81/known.twin_belief.npz CASE/final_data.pkl \
  parameter_support_audit.json parameter_support_audit.csv \
  --counts 4 8 16 32 81 \
  --methods top_mass weighted_coreset
```

Graph discrepancy comparison:

```bash
causal4d-evaluate-graph-temporal-discrepancy \
  known_action.physical.npz CASE/final_data.pkl CASE/optimal_params.pkl \
  parameter_profile.npz graph_temporal_discrepancy.json \
  --rank-candidates 4 8 16 32
```

Source-only calibration:

```bash
causal4d-real-calibration fit \
  configs/causal4d/sloth_real_calibration_sources_v1.json \
  affine_calibration.json

causal4d-real-calibration evaluate \
  affine_calibration.json \
  configs/causal4d/sloth_real_calibration_target_v1.json \
  target_calibration_evaluation.json
```

## Decision

1. Replace top-4 with the 16-point weighted coreset for future real protocols;
   retain top-32 as the conservative non-aggregating control.
2. Carry graph persistence as the discrepancy baseline. Keep learned AR dynamics
   as a negative control until it transfers across actions.
3. Do not call the current real posterior calibrated. Neither full parameter
   support nor graph discrepancy reaches nominal coverage.
4. Do not deploy the current affine transform: its locked source transfer is
   harmful. Refit affine or regime-specific calibration only under the locked
   same-object multi-action folds. Report action, contact, horizon,
   graph-region, calibration-curve, NLL, energy-score, interval-width, NEES,
   and worst-group results.
5. Do not begin physical closed-loop execution while nominal 90% coverage is
   67.78% raw or 43.03% after the rejected source transform. The first hardware
   pilot is gated by `configs/causal4d/hardware_execution_gate_v1.json`; the
   existing real-artifact replay remains a software result only.
