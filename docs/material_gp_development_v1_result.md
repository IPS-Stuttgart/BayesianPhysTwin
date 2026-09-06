# Material-sharing GP residual pilot: completed development result

## Decision

**Mixed result. Do not replace the existing ridge residual or pivot the paper to a GP contribution on this evidence.** Material sharing improves the GP variant, and the material GP gives the lowest coordinate RMSE, but it does not beat the existing conservative ridge correction on the primary coordinate L1 metric. It also has a substantially worse individual trajectory.

This is a completed real-data comparison, not a synthetic demonstration or a queued experiment. The nine exact-head algebra/interface tests passed. No robot, additional data collection, backbone retraining, source-test read or official evaluation read was involved.

## Execution

- Code review: [PR #963](https://github.com/IPS-Stuttgart/BayesianPhysTwin/pull/963).
- Completed [Actions run 34053302178](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34053302178), on `gpuserver6000`.
- Execution commit: `dabe56561a3188a737b29feb19cae9a7d77f1bde`.
- Exact GP comparison blob: `9b9399c236c40f4902735c312e452d3fc8eca022`.
- [Machine-readable compact evidence](../results/experiments/material_gp_development_v1/summary.json).
- [Complete artifact](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34053302178/artifacts/9995216697), including the original result JSON, full candidate bank, prediction seal, all mean predictions and marginal variances, source-cache receipts, and test log.
- Artifact ZIP SHA-256: `b78765131d7ac3da1bc0762b0c34afd2bc6abb5493f32c53b6946b54ec2c0a9b`.

The comparison reused the same checksum-bound DLO2 fit/validation forecasts already cached by a sibling experiment on the second server. The importer checked both receipts, checkpoint and source-manifest hashes, every recording identity/order, finite arrays, complete 498-step forecasts and baseline reproduction. The validation baseline differed from its historical reference by only `3.70e-10 m`. Only `fit.npz` and `validation.npz` were read; the sibling source-test cache was not opened. Cache reuse changes neither the fixed GP code nor its model-selection protocol.

The initial native replay attempt failed at Git ownership verification before trajectory access. Its subsequent queued retry on `gpuserver4090` was cancelled by this experiment after the second-server cache passed verification. No unrelated run was cancelled. The cached comparison, including inner model selection and all final fits/scores, took `32.52 s`; that excludes the prior simulator export and environment setup.

## Matched setup

The baseline is the frozen update-6400 **DEFORM physics-plus-GCN hybrid**, not bare rod physics. Forty original fitting trajectories supply an inner 30/10 whole-trajectory split for residual hyperparameter selection. The selected models are refitted to all 40 and scored on the eight historical development-validation trajectories. No exact causal-query duplicates or cross-split duplicate collisions were found. Each prediction covers 498 steps, with 12 nodes of which eight are free. Point metrics follow the original full-node coordinate contract, including the exactly preserved clamped nodes.

Both GP arms use the unchanged causal feature builder and a linear-plus-Matérn-3/2 deformation kernel. The material arm additionally shares through a Matérn-3/2 kernel over normalized material-node index. Both have 768 inducing locations in total. Native ridge uses its usual full fitting frames; GPs fit 24 time samples per trajectory but predict the full horizon. See the [method specification](material_gp_development_v1.md).

## Accuracy

Lower is better for both errors. Wins are against the unchanged DEFORM hybrid, not against the existing ridge.

| Method | Coordinate L1 (mm), primary | Coordinate RMSE (mm) | Trajectory wins vs hybrid |
|---|---:|---:|---:|
| Unchanged DEFORM hybrid | 7.9120 | 14.1166 | — |
| Existing ridge, ridge 1 / shrinkage 0.25 | **7.2403** | 12.9431 | **7/8** |
| Inner-selected ridge, ridge 10 / shrinkage 1 | 7.5678 | 13.5044 | 4/8 |
| Independent per-node GP | 7.5373 | 13.3333 | 5/8 |
| Material-sharing GP | 7.3312 | **12.6220** | 5/8 |

Relative to the existing ridge, the material GP is **1.26% worse in primary L1** and **2.48% better in RMSE**. It wins L1 on only four of eight trajectories. The paired mean L1 difference is `+0.0909 mm`, with a descriptive whole-trajectory bootstrap 95% interval of `[-0.5811, +0.8070] mm`.

Relative to the independently inner-selected ridge, it is 3.13% better in L1, but the paired interval still crosses zero: mean `-0.2365 mm`, interval `[-0.6396, +0.1611] mm`. Existing ridge has historical validation-selection advantage; the fair inner-selected comparison is useful, but neither turns this reused, eight-trajectory panel into independent confirmation.

Material sharing reduces L1 by 2.73% relative to independent GPs and wins seven of eight trajectories in that within-GP comparison. This is a useful development observation, not proof that the GP family beats the actual existing method.

## Individual-trajectory risk

| Trajectory | Hybrid L1 (mm) | Existing ridge L1 (mm) | Material GP L1 (mm) |
|---|---:|---:|---:|
| 102.pkl | 5.4660 | 5.7285 | 7.6000 |
| 106.pkl | 5.7526 | 5.4176 | 6.3379 |
| 19.pkl | 5.3224 | 4.5981 | 4.4610 |
| 25.pkl | 13.7200 | 12.3685 | 11.4787 |
| 3.pkl | 10.4601 | 9.1709 | 7.9437 |
| 50.pkl | 8.3739 | 7.6824 | 6.7855 |
| 56.pkl | 6.8835 | 6.1436 | 6.9069 |
| 6.pkl | 7.3177 | 6.8127 | 7.1360 |

On `102.pkl`, material-GP L1 is 39.04% above the unchanged hybrid and 32.67% above the existing ridge. The existing ridge's worst trajectory is only 4.80% above the hybrid. Therefore, quoting only the material GP's lower aggregate RMSE would omit an important regression.

## Uncertainty diagnostics

All values below use a single covariance multiplier fitted from the inner tuning trajectories only. No covariance scaling uses the eight outer validation outcomes. Coverage is nominal-90% **coordinate-marginal** coverage on free nodes; width is the mean full interval width. NLL is marginal Gaussian negative log density in metre units, with lower being better.

| Method | Coverage | Mean full width (mm) | Normalized coordinate NEES | Marginal NLL |
|---|---:|---:|---:|---:|
| Existing ridge | 93.02% | 55.44 | 0.789 | -2.9284 |
| Inner-selected ridge | 91.09% | 50.65 | 0.970 | **-2.9396** |
| Independent GP | 90.62% | 49.51 | 0.985 | -2.9091 |
| Material-sharing GP | 90.93% | 50.17 | 0.962 | -2.7925 |

The scaled GP marginals have approximately nominal coverage on this panel, but the material GP does not win the proper marginal score. Its unscaled intervals were very conservative: 98.08% coverage and 84.02 mm full width. These observations do not establish calibrated physical-parameter uncertainty, a useful full spatiotemporal posterior, or a downstream Bayesian advantage.

## Scientific interpretation

This pilot tested a concrete improvement rather than a new hardware protocol. It found a limited benefit from material-sharing GP structure, but no primary-metric replacement case against the existing conservative correction. Preserve the positive RMSE result and the negative L1/trajectory/proper-score results together.

No model was retuned after these validation outcomes. No learning curve, matched neural residual, different-object transfer, or official evaluation was run. The present conclusion is **mixed development evidence**, not a definitive rejection of every GP construction and not a confirmed new paper contribution.
