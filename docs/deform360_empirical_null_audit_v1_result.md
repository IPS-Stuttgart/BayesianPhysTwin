# Completed Deform360 empirical-null and matched-acceptance result

## Execution and provenance

- GitHub Actions run: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33984701676
- Pull request: https://github.com/IPS-Stuttgart/BayesianPhysTwin/pull/935
- Run conclusion: success; all evaluation and evidence-upload steps completed.
- Trigger: push changing only `.github/requests/deform360-empirical-null-audit-20260906.json`.
- Executed commit: `f03075b9179f9048e760e0c54350bd01b764e9bd`.
- Requested runner labels: `[self-hosted, Linux, X64, gpuserver4090]`; actual runner name: `workstation1`.
- Canonical data: `/mnt/seagate10tb/florianpfaff/datasets/deform360`.
- Artifact ID: `9974821795`, name `deform360-empirical-null-audit-33984701676-1`.
- Artifact ZIP SHA256: `59d133e33eeb445071d23e1b6f23f0eadb72f6d5bfe0a06e9f554fddc5bc85ab`.
- `result.json` file SHA256: `cac1aaaac188e174aebed922eab45894e73e48ef92d466675c9c164f5739ec86`.
- All 100 file hashes in the downloaded artifact ledger verified.

The experiment completed all 92 previously evaluated real objects and all five original tactile-field queries, totaling 3,904 forecast windows and 19,520 query forecasts. The inferential unit remains the physical object, not the window or query. This is a retrospective real-data audit, not fresh confirmation, and not dense visual 4-D reconstruction.

The original predictive mean hashes and registered point results reproduced exactly. Every reference covariance rank was 8; every conventional rank-matched covariance also had rank 8. The maximum coordinate-marginal mismatch for that comparator was `1.1102230246251565e-16`. No new holdouts, robot interactions, geometry or camera payloads were opened.

## Main results

Lower Brier score and fixed-cost loss are better. The fixed-cost rule accepts when predicted adverse-event probability is at most 0.1, with unit cost for an adverse accepted event and cost 0.1 for flagging.

| Model | Brier score | Fixed-cost loss | Acceptance | Query NLL | Nominal 90% coverage |
|---|---:|---:|---:|---:|---:|
| Original full low-rank | 0.163336873 | 0.107558958 | 37.939% | 1.425873 | 71.703% |
| Original diagonal | 0.174260239 | 0.119535575 | 53.817% | 15.776467 | 51.973% |
| Original scrambled | 0.174659692 | 0.119815670 | 53.946% | 17.359854 | 51.486% |
| Conventional rank-matched empirical covariance | 0.163806281 | 0.108675372 | 39.061% | 1.654019 | 70.735% |
| Direct empirical scalar-query Gaussian | 0.163336873 | 0.107558958 | 37.939% | 1.425873 | 71.703% |
| Independently query-recalibrated diagonal | 0.163336873 | 0.107558958 | 37.939% | 1.425873 | 71.703% |

The conventional covariance uses source residual PCA, fixed 10% diagonal shrinkage, matched rank and rescaling to the original coordinate marginal variances. It is not a target-tuned or claimed-optimal empirical estimator.

The direct empirical scalar-query model estimates the five required query-error variances directly from the same source residuals. It need not preserve the original full-field coordinate marginals; it is a same-mean query-output comparator, not a marginal-matched full-field model.

## Primary matched-acceptance result

At exactly 40% acceptance within every object, all methods select the same number of forecasts by ascending predicted adverse-event probability; outcome labels are not used in selection. Query/windows are pooled within each object for ranking. Reported risks are then averaged equally over objects.

| Model | Adverse-event rate among accepted forecasts |
|---|---:|
| Original full low-rank | 14.422618% |
| Original diagonal | 15.069486% |
| Original scrambled | 15.185858% |
| Conventional rank-matched empirical covariance | 14.419394% |
| Direct empirical scalar-query Gaussian | 14.422618% |
| Independently query-recalibrated diagonal | 14.422618% |

Paired full-minus-comparator risk differences, in percentage points, with 95% intervals from 10,000 physical-object bootstrap resamples:

| Comparator | Difference [percentage points] | 95% interval [percentage points] |
|---|---:|---:|
| Original diagonal | -0.646868 | [-1.178614, -0.177129] |
| Original scrambled | -0.763239 | [-1.253957, -0.307108] |
| Conventional rank-matched covariance | +0.003225 | [-0.064643, +0.076202] |
| Direct empirical scalar-query Gaussian | 0 | [0, 0] |
| Independently query-recalibrated diagonal | 0 | [0, 0] |

Thus the original representation still beats dependence-destruction controls at matched acceptance, but it does not establish an advantage over conventional correlated residual modeling at the primary 40% endpoint. The rank-matched comparison has 5 full-model object wins, 83 ties and 4 losses at this endpoint.

There is a small proper-score advantage over the fixed rank-matched covariance comparator: the Brier difference is `-0.0004694078`, with 95% object-bootstrap interval `[-0.0006423162, -0.0003044827]`. Fixed-cost loss difference is `-0.0011164139`, interval `[-0.0019338141, -0.0004077595]`. Neither advantage persists against the direct empirical-query null, whose scores and decisions tie to numerical precision.

## Explanation of the empirical null

The historical v6 full arm uses, per object and registered query:

```
scale = source_query_error_MSE / raw_projected_covariance
calibrated_query_variance = raw_projected_covariance * scale
                         = source_query_error_MSE
```

Consequently the source empirical Gaussian has the same mean, variance and Gaussian event probabilities. Independently recalibrating the diagonal arm per query yields the same result. This equality is specific to the scalar-query calibration scheme; it does not prove equality of joint trajectory distributions or arbitrary new query predictions.

The run measured a maximum full/direct-empirical probability difference of `2.7755575615628914e-16`. An independent verifier that imports neither BayesianPhysTwin nor the audit runner recomputed event thresholds, empirical variances and Gaussian probabilities from saved arrays for all 92 objects / 460 object-query pairs. It passed, with maximum probability difference `7.549516567451064e-15`, maximum variance difference `6.938893903907228e-18`, identical 40%-acceptance risks on every object, and equal-object Brier score `0.1633368729989292`.

## Scientific decision

**The stronger hypothesis of an additional Bayesian advantage over conventional source-fitted uncertainty is not supported for this protocol.** The registered scalar-query predictions are reproducible by a non-Bayesian empirical-error model. The narrower dependence-destruction comparison remains positive.

The full query distribution remains undercovered: nominal-90% intervals achieve only 71.703% coverage. Also, always flagging has fixed cost exactly 0.1, lower than the full model's 0.107559. Improvements over selected baselines therefore must not be described as overall optimal utility or calibrated uncertainty.

These results do not refute Bayesian state estimation, nonlinear posterior marginalization, uncertainty-aware data fusion, joint-event prediction or transfer to queries not calibrated individually. Those hypotheses were not tested in this run. No experiment was selected or rerun to conceal this null result.
