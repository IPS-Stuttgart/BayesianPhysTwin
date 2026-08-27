# Weak-constraint belief: frozen development result

## Decision

**FAIL. Do not promote the new mean or its structured covariance.** The previous
eight-observation paired physical update remains the strongest point predictor.
The current experiment does not authorize a fresh evaluation, protected target,
official DEFORM claim, or changes to the incumbent.

One frozen CPU-only prediction run completed in 594.38 seconds: 30/30 ordinary
successful trajectories, zero technical failures, zero unsealable trajectories.
The existing DLO2 design trajectory is excluded from analysis, leaving 13 opened
DLO2 calibration trajectories and 16 opened DLO1/DLO3 transfer trajectories.
All nine arms were sealed before calibration; calibration was sealed before
transfer scoring. There was no empirical rerun or parameter change.

## Point results

These are equal-object means over DLO1/DLO3, not pooled coordinate errors or
official benchmark numbers. All arms share the same two initial full states,
object-specific frozen model/readout, and prescribed clamp trajectory. Query
counts below are additional three-dimensional prefix point observations.

| Arm | Queries | Coordinate L1 (mm) | Hidden point RMSE (mm) | Joint wins /16 versus incumbent |
|---|---:|---:|---:|---:|
| Unchanged incumbent | 0 | 9.859 | 23.068 | 0 |
| Previous paired physical update | 8 | **8.711** | **20.572** | 14 |
| OLS endpoint + physical propagation | 16 | 9.192 | 21.512 | 14 |
| OLS readout extrapolation | 16 | 27.679 | 63.857 | 0 |
| DEFORM-style periodic position correction | 16 | 9.312 | 21.914 | 12 |
| Strong-constraint belief | 8 | 9.109 | 21.430 | 14 |
| Weak-constraint belief, secondary | 8 | 9.028 | 21.236 | 15 |
| Strong-constraint belief | 16 | 9.224 | 21.652 | 13 |
| Weak-constraint belief, primary | 16 | 9.200 | 21.591 | 15 |

The primary weak_16 modestly improves on strong_16 in aggregate, but not on DLO3
individually. It fails the required 2% gains over both the previous paired arm
and matched OLS, and does not beat every matched control on both objects. The
secondary eight-query result cannot rescue the frozen primary gate.

The previous paired update is also better than the new DEFORM-style periodic
position control despite using half as many additional observations. This is a
descriptive comparison of the frozen arms, not a fresh-confirmation claim. The
periodic control is an adaptation to this exact identity/readout contract, not a
reproduction of DEFORM's complete camera-tracking pipeline.

| Object | Incumbent late RMSE | Previous paired late RMSE | Weak_16 late RMSE |
|---|---:|---:|---:|
| DLO1 | 23.698 | 24.808 | 23.348 |
| DLO3 | 21.939 | 21.074 | 21.646 |

The new method improves the DLO1 late-horizon issue but trades away mean accuracy.
This is not permission to select different methods per object or horizon.

## Uncertainty results

Calibration uses only the 13 non-design DLO2 trajectories. These are marginal 3D
Gaussian scores on the already-opened transfer trajectories. NLL is in nats with
coordinates in metres; lower is better. Width is the geometric-mean full axis
diameter of the 90% ellipsoid, not a coordinatewise interval width.

| Mean and covariance | Calibration | NLL | Coverage | Width (mm) |
|---|---|---:|---:|---:|
| Previous paired + isotropic | Moment | **-8.839** | 91.08% | **69.31** |
| Weak_16 + isotropic | Moment | -8.645 | 89.93% | 72.59 |
| Weak_16 + native tangent covariance | Moment | -8.650 | 90.07% | 72.67 |
| Previous paired + isotropic | Conformal | -7.840 | 98.58% | 134.24 |
| Weak_16 + isotropic | Conformal | -7.602 | 98.76% | 143.18 |
| Weak_16 + native tangent covariance | Conformal | -7.616 | 98.70% | 142.33 |

The same-mean shaped-minus-isotropic NLL difference has trajectory-bootstrap 95%
intervals [-0.02880, 0.01683] on DLO1 and [-0.00958, 0.00059] on DLO3. Both include
zero. Shaped covariance also increases volume on DLO1. The primary uncertainty
gate fails. The cheap isotropic comparator is not displaced.

The previous mean with source-fitted isotropic uncertainty reaches 89.71% coverage
on DLO1 and 92.45% on DLO3, but only 85.06% on its DLO2 fitting trajectories. The
91.08% aggregate is therefore useful development evidence, not proof of a
calibrated posterior or a distribution-free cross-object guarantee. The declared
conformal score uses rank 13/13 and is much wider. No calibrator is selected from
these transfer outcomes.

## Scope of the negative

This rejects the fixed bounded, locally linear weak-constraint construction, not
Bayesian inference, all process-error models, or state uncertainty generally.
A post-score, nonselective diagnostic of the sealed covariance artifacts shows
that the primary mean guard was active on all 30 predictions. Minimum gains were
0.116/0.245/0.217 for DLO1/DLO2/DLO3. The median fraction of marginal covariance
trace supplied by the fixed isotropic floor was 98.63%/99.74%/99.56%, respectively.
Thus this particular guarded covariance carried very little non-isotropic shape.
These observations do not authorize changing the caps/floor and rescoring this run.

The useful next question is how to report uncertainty about the point forecast
actually deployed when a state posterior is guarded or its mean is rejected.
That requires a separately frozen mathematical/statistical contract, not a new
claim from this failed arm. Preserve the current point mean and compare any such
uncertainty layer with the isotropic and conformal controls above.

## Verification and provenance

- Prediction/source commit: `b87eea1e9477ad2b5c4445691f8b1af1b353a701`.
- Checker-only amendment: `1abe066fe3bd68de1542e523a0abc45305365d6d`.
- Original readiness: 419 relevant tests; 44 focused tests in the exact remote
  NumPy 1.24.3 / Torch 2.0.1 CPU runtime. Final checker revision: 420 relevant
  tests, 45 focused tests; Ruff and focused MyPy pass.
- All 952 frozen source files verified. All 30 incumbent and previous paired
  means remain byte-identical. No original result file was edited.
- Independent batch posterior, physical increments, native covariance, calibration,
  point metrics, bootstrap summaries, and decision rechecked.
- 270 case/arm forecast records, 180 native continuations, and 83,520 marginal
  uncertainty events verified. The local delivery rechecks 12 NPZs/135 arrays.
- One initial checker failure at a floating-point conformal membership boundary
  is preserved. The amended checker reuses the registered binary comparison order
  only at that boundary; continuous scores remain independently factored. No
  prediction, calibration, metric, gate, or scientific tolerance changed.

| Artifact | SHA-256 |
|---|---|
| Source receipt | `33430d3f79f27c11634ab1221383b50b8e75a439ad1dcec31e66ed20168cf849` |
| Protocol | `8a825b4f6d7ca127f3329ec21d0767e62204e6bc3d6c880a2c6763f3c25d6197` |
| Response barrier | `cfc71f0bb43b2f6723cb62a429f7b7455fc1fd32915181a25f940362a871d42f` |
| Prediction barrier | `1fe14811ffc26731584e0c601003bacb9e5634a52cd456250621c6f260700045` |
| Calibration | `c54e4ff9ecdb6917585bcc16c949b0eada9f73a7ddf902135a39089fe83eb0a8` |
| Result | `7208eed54efa0118e5352a38df6289c87b7e91a9f580cce035ec3c40696a05ca` |
| Independent verification | `82190bec5ee0b442a0c66af5b19a7d627ff55d09ffad045f588f67c79ee0b4d9` |

Full evidence remains in the source-only server run and local user archive.
This result is local/private-paper evidence only; it was not pushed or merged.
