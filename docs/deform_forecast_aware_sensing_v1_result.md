# Public DEFORM: Forecast-aware Sparse Sensing

This local/private evidence note accompanies
`results/sota/deform_forecast_aware_sensing_v1/`. Full prediction arrays, logs,
plots, and paths relative to `evidence/` below are preserved at:

```text
/mnt/c/users/emper/documents/codex/2026-08-25/where-are-we-with-cut3r/deform-forecast-aware-sensing-v1
```

This result is not automatically authorized for public outcome publication.

## Conclusion

**Completed, verified, and not promoted.** The frozen eight-observation planning
gate failed: forecast-aware selection did not beat the equally informed uniform
schedule. The original DEFORM predictor and previous positive state-update result
remain byte unchanged.

There is useful new evidence for the paper: propagating sparse pose/velocity
information through the native dynamics beats every tested fixed temporal
readout control. The stronger claim that this forecast-variance planner chooses
better observations is not supported at the primary budget.

## Scope

- Public released data only; no additional recordings or robot execution.
- All 30 registered trajectories produced predictions; 29 enter analysis because
  the original DLO2 design trajectory remains excluded. No failures or replacements.
- DLO1/DLO3 provide 16 already-open transfer trajectories; DLO2 has 13 analyzed
  discovery-reference trajectories. The physical parameters remain object-specific.
- All methods share two full initialization frames, known future clamped inputs,
  and released material identities. After initialization, only selected prefix
  points are observed; future scoring identities are disjoint from those queries.
- These are opened-object development results, not fresh confirmation, automatic
  perception performance, calibrated uncertainty, or an official SOTA comparison.
- No DLO4/DLO5, official DLO3 evaluation, held-v8, or protected target was opened.

## Main Results

Hidden-point Euclidean RMSE in mm; lower is better. Each object is averaged over
its trajectories. The transfer column equally weights DLO1 and DLO3. These are
not Chamfer distances or the PhysTwin 22-case manual-track metric.

| Method | DLO1 (8) | DLO2 (13) | DLO3 (8) | Transfer mean |
|---|---:|---:|---:|---:|
| Unchanged incumbent | 24.702 | 25.614 | 21.435 | 23.068 |
| Previous paired physical update, 8 observations | 21.754 | 23.066 | 19.389 | 20.572 |
| New joint state/bias update, uniform 8 | 22.653 | 23.667 | 20.206 | 21.430 |
| New joint state/bias update, forecast-aware 8 | 22.911 | 23.755 | 20.289 | 21.600 |
| Fixed 100 ms decaying temporal correction | 23.582 | 24.831 | 20.862 | 22.222 |

Mean of the two objects' RMSE percentage changes versus the incumbent:
previous paired **-10.74%**, uniform **-7.01%**, forecast-aware **-6.30%**,
and the 100 ms temporal control **-3.60%**. Their joint L1/RMSE trajectory wins
are respectively 14/16, 14/16, 13/16, and 14/16. Win count alone does not measure
improvement magnitude.

Forecast-aware selection is 1.14% worse than uniform on DLO1 and 0.41% worse on
DLO3, instead of the required at least 2% improvement on each. It improves both
metrics over the incumbent, beats every fixed temporal control, and passes the
late-horizon and five-of-eight-win criteria. The uniform-comparison criteria
nevertheless fail, so no larger independent evaluation of this planner is
recommended by the registered decision.

## Temporal Controls

All eight controls use the same original eight observations. Time constants were
fixed before outcomes; the best-looking row is not a newly selected method.
Equal-object DLO1/DLO3 means:

| Control | Coordinate L1 (mm) | Point RMSE (mm) |
|---|---:|---:|
| Static residual | 11.823 | 27.173 |
| Constant residual velocity | 43.169 | 109.855 |
| Damped velocity, 100 ms | 14.641 | 33.884 |
| Damped velocity, 300 ms | 21.175 | 50.074 |
| Damped velocity, 1000 ms | 32.122 | 78.820 |
| Decaying pose/velocity, 100 ms | 9.435 | 22.222 |
| Decaying pose/velocity, 300 ms | 10.595 | 24.850 |
| Decaying pose/velocity, 1000 ms | 21.806 | 51.805 |
| Previous paired physical update | **8.711** | **20.572** |
| New forecast-aware physical update | 9.193 | 21.600 |

The previous physical update also beats all eight controls on each of the three
objects separately. This narrows the explanation of its benefit beyond simple
temporal persistence or the tested extrapolation rules. It does not establish
superiority over every possible learned temporal model.

## Horizon and Budget

Late-horizon RMSE changes relative to the incumbent:

| Method | DLO1 | DLO2 | DLO3 |
|---|---:|---:|---:|
| Previous paired update | +4.68% | -1.58% | -3.94% |
| New forecast-aware 8 | -2.30% | -1.29% | -1.51% |

The new method removes the previous DLO1 late regression on this cohort, but
gives up average accuracy. That tradeoff does not override the failed gate.

The frozen secondary budget comparison, equal-object transfer RMSE:

| Additional 3D observations | Uniform | Forecast-aware |
|---|---:|---:|
| 4 | 22.532 | 21.653 |
| 8 | 21.430 | 21.600 |
| 12 | 21.721 | 21.698 |
| 16 | 21.652 | 21.652 |

Four forecast-aware queries nearly match eight uniform queries. This is a
secondary lead, not a substitute primary result: four-query random and
current-shape controls were not included, and no new budget is selected for
confirmation. The full-budget plans and predictions are identical by construction
and verification. All four frozen random eight-query arms are retained in the
complete result rather than ensembled or selectively reported.

## Simulated Sensor Bias

Eight fixed repetitions per trajectory, averaged within trajectory before
aggregation. Equal-object transfer RMSE in mm:

| Method | Clean | 1 mm independent noise | Same noise + 5 mm shared bias |
|---|---:|---:|---:|
| Incumbent | 23.068 | 23.068 | 23.068 |
| Previous paired update | 20.572 | 20.586 | 20.712 |
| Uniform joint state/bias update | 21.430 | 21.443 | 21.441 |
| Forecast-aware joint state/bias update | 21.600 | 21.599 | 21.600 |

The new joint model is insensitive to this specified common-offset stress, but
still does not beat the previous paired update in absolute RMSE. There is no
matched nuisance-disabled arm isolating why, and these simulated errors do not
validate real camera noise or posterior coverage. Planning covariance remains an
uncalibrated inference/design model.

## Verification and Preservation

- One empirical prediction run; all schedules sealed before revealing queries,
  and all 30 trajectories sealed before scoring. CPU-only runtime: 866 seconds.
- 283 relevant unit/regression tests pass, including 45 sensing/checker tests;
  Ruff, focused MyPy, and diff checks pass. This was not the full repository suite.
- Separate information-form planning and batch-posterior calculations verify
  schedules, coefficients, measurement values, and temporal controls.
- All **6,450** case/arm/noise forecast metric records were recomputed, including
  horizon summaries, 10,000 whole-trajectory bootstrap intervals, and the decision.
- All **60** clean primary native forecasts were replayed and matched numerically.
- All 30 incumbent and previous paired forecasts retain the registered original
  bytes. No method, metric, threshold, source roster, or old result was changed.
- All 43 transferred evidence/log files match fresh server-side SHA-256 values.
- The numerical verifier needed three disclosed checker-only fixes: contiguous
  native input buffers, the standalone metric import path, and float64 noise
  arithmetic. Exact comparison tolerances were not relaxed. All three failed
  checker logs are retained; predictions were not rerun or edited.
- This is independent analysis code, not an independent person or empirical
  replication. Bootstrap intervals are conditional on these already-open objects.

## Reproducibility

Frozen prediction source: `f62e4ae496df1f9bcd73ef0d61e379f0e08968b3`.
Final checker source: `15b8702e9aa92773dabb28eed3737087694ce4c2`.
Both are preserved locally, separately from the unchanged incumbent.

```text
source receipt:
d69f7b96010864357bce54090c323346f7b69eac7421ad9c668e0918491e4231
prediction barrier:
fbf6c97f35eb1ce34d08f7ad231cab206ca608b085ece55cad060e5f67c3bd4c
full result:
c58c9b61cf862954475d7d57f27f1e377ad86dd5089fc913b538fded4121104f
successful numerical verification:
77d0dc00cd28c59566a4a42f04b65b605159698ff74116b031ed78ef956fc539
```

Full predictions and metrics are in `evidence/run-v1/`; the successful audit is
`evidence/verification_v1_2.json`. Run `sha256sum -c SHA256SUMS` inside `evidence`
to verify the transported files. `results.png` and `results.pdf` summarize the
comparison. `README.md` records the code and execution boundaries.

The useful paper addition is the matched temporal ablation and the observed
accuracy/late-horizon tradeoff. Do not claim successful active sensing, retune
the planner on this result, or open a protected cohort to rescue it.
