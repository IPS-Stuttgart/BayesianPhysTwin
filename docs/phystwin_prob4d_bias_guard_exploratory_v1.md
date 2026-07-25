# PhysTwin Prob4D Bias-Aware Guard: Exploratory V1

## Status

This is post-open method development on the previously examined 19-case
PhysTwin cohort. The protocol was locked before this new arm's future scores
were computed, and all 19 predictions were sealed before evaluation. The
result can reject or motivate a method family; it cannot establish independent
confirmation, calibration, or state of the art.

## Method

The arm combines three existing pieces without reading future Prob4D frames:

1. the released Bayesian validation-selected trajectory is the baseline;
2. the released raw PhysTwin prefix defines a rank-4 physically reachable
   response basis;
3. the Prob4D Arm-C direct position-flow associations provide metric prefix
   observations, residual-independent reliability, and covariance.

At 75% of the released training interval, the estimator decomposes the
innovation into reachable state, shared spatial bias, and global camera bias.
The state innovation enters one Student-t mixture likelihood. The inferred
state correction is held fixed, then evaluated on the untouched final 25% of
the training interval using dense released pseudo-observation Chamfer and
identity error. It is deployed only when both metrics do not regress and their
mean relative ratio improves by at least 0.1%. Every rejection is a byte-exact
fallback to the selected Bayesian trajectory.

Prefix materialization, prediction, and future evaluation are separate
commands. The predictor consumes a truncated prefix artifact and cannot load
future Prob4D, future object observations, or manual tracks.

## Result

The candidate was admitted on 2/19 cases and fell back exactly on 17/19.

| Method | Future CD | Future track | Late CD | Late track |
| --- | ---: | ---: | ---: | ---: |
| Selected Bayesian baseline | 9.815 mm | 19.531 mm | 12.293 mm | 22.958 mm |
| Raw bias-aware candidate | 10.554 mm | 19.997 mm | 12.843 mm | 23.313 mm |
| Prefix-guarded candidate | **9.673 mm** | **19.385 mm** | **12.094 mm** | **22.646 mm** |

The guard improves the case-balanced mean by 1.46% CD and 0.74% track error,
but the physical-object-cluster intervals cross zero:

| Metric difference, guarded minus baseline | Mean | 95% cluster interval |
| --- | ---: | ---: |
| CD | -0.008 mm | [-0.359, +0.336] mm |
| Track | +0.041 mm | [-0.437, +0.559] mm |

The discrepancy between the case-balanced and cluster-balanced means reflects
repeated interactions from the same physical objects; the cluster analysis is
the inferentially relevant one.

### Accepted cases

| Case | Future CD change | Future track change | Outcome |
| --- | ---: | ---: | --- |
| `single_lift_cloth_1` | -3.948 mm | -4.808 mm | large two-metric win |
| `single_push_sloth` | +1.232 mm | +2.048 mm | harmful false acceptance |

`single_push_sloth` is the previously documented non-reproducible replay case,
but it was included by the locked exploratory protocol and cannot be removed
after scoring.

## Gate Decision

The transfer gate fails:

- only 2 cases were accepted; at least 3 were required;
- one accepted case was harmful; zero were allowed;
- both cluster-bootstrap upper bounds exceed zero;
- all 17 rejections were nevertheless byte-exact;
- 18/19 cases improve or tie both metrics because rejected cases tie exactly.

Decision: **do not start an independent evaluation of this static correction
family**.

## Scientific Conclusion

The experiment finds one substantial, causally selected gain and confirms that
bias-aware physical support prevents the broad regressions of unguarded Prob4D
assimilation. It also reproduces the central limitation of endpoint
persistence: a correction can improve a disjoint prefix and still fail after
the action boundary. Static Prob4D correction is therefore closed as the next
predictive method.

The next admissible observation-driven family must predict time-varying
low-rank discrepancy coefficients from the known physical/action trajectory,
carry action-divergence uncertainty, and preserve the same exact fallback. It
must be developed on a newly declared source panel. Reusing this opened future
to relax the gate or special-case `single_push_sloth` would not be credible.

## Evidence

- Protocol:
  `configs/sota/phystwin_prob4d_bias_guard_exploratory_v1.json`
- Result:
  `results/sota/diagnostics/phystwin_prob4d_bias_guard_exploratory_v1/result.json`
  (canonical SHA-256
  `9f0185fa2b62a65ee9dd2ede098124702fe5f84da68e4c6a90dd116e0241c0fe`)
- Prediction seal:
  `results/sota/diagnostics/phystwin_prob4d_bias_guard_exploratory_v1/prediction_cohort_seal.json`
- Compact target-free decision ledger:
  `results/sota/diagnostics/phystwin_prob4d_bias_guard_exploratory_v1/target_free_decision_ledger.json`
  (canonical SHA-256
  `f8a539f6c63b533743d9c6d8b5021334fee2799004281d006d29a664e89e5521`)
- Implementation:
  `src/bayesian_phystwin/phystwin_prob4d_bias_guard.py`
- Runner:
  `scripts/remote/run_phystwin_prob4d_bias_guard_exploratory.py`

The complete prefix and trajectory artifacts remain at
`gpuserver6000:/mnt/corsair/florianpfaff/bpt-prob4d-bias-guard-exploratory-v1-r3`.
That remote tree also retains every full target-free prediction report.
