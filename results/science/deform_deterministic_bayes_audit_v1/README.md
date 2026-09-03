# DEFORM deterministic/Bayesian point-equivalence audit v1

Status: **passed**.

This read-only audit asks whether the DEFORM DLO2--DLO5 point-accuracy gain is
uniquely caused by the Bayesian formulation. It reuses the existing sealed
predictions, local-residual models, and evaluation manifests. It independently
reconstructs the deterministic ridge point mean from the stored feature
normalization and coefficient arrays without reading any covariance array.

No physical model was retrained, no residual model was refitted, no target was
selected or calibrated, and no original artifact was modified.

## Result

| Panel | Matched DEFORM base | Deterministic ridge mean | Gain | Wins | Deterministic vs Bayesian |
|---|---:|---:|---:|---:|---:|
| DLO2 | 8.7470 mm | 7.8606 mm | 10.13% | 14/14 | exact |
| DLO3 | 7.0518 mm | 6.3467 mm | 10.00% | 14/14 | exact |
| DLO4 | 9.9549 mm | 8.9532 mm | 10.06% | 14/14 | exact |
| DLO5 | 8.0501 mm | 7.8268 mm | 2.77% | 14/14 | exact |
| Equal-DLO | 8.4509 mm | 7.7468 mm | 8.33% | 56/56 | exact |

The maximum pointwise difference between the independently reconstructed
deterministic mean and the archived Bayesian point mean is `0.0 m`. Every early,
middle, and late horizon third also improves over the corresponding base.

## Scientific decision

The point-accuracy result belongs to the **residual adapter**, not uniquely to
Bayesian inference. A matched deterministic ridge comparator obtains exactly the
same mean and therefore exactly the same 56/56 trajectory wins. The defensible
Bayesian contribution must be attributed to the predictive covariance and joint
dependence, evaluated separately through proper scores, calibration diagnostics,
and downstream decision loss.

## Execution receipt

- Workflow run: `33749453799`
- Workflow source revision: `9c126bf3a0429076dee26ef84245ac2428a812b3`
- Required runner label: `gpuserver4090`
- Runner name: `workstation1`
- Artifact ID: `9890938371`
- Artifact digest: `sha256:8fb75fddc7a335b633edfc624ecaa718f7760612259ded8dc8f0a707a0a67832`
- Runtime: Python 3.10.12, NumPy 1.24.3

The complete content-addressed inputs, horizon metrics, timings, and decision are
in `result.json`; all 56 trajectory rows are in `trajectory_metrics.csv`.
