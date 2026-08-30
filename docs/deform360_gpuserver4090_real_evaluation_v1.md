# Deform360 real-data development evaluation

- Status: **complete**
- Dataset root: `/mnt/seagate10tb/florianpfaff/datasets/deform360`
- Git revision: `6ba943fbeb8b18929c734bfbc8012fe5c0522ef6`
- Runner: `workstation1` with required label `gpuserver4090`
- Primary released carrier: **tactile_response**
- Physical objects in the primary aggregate: **16**
- Completed source/target groups: **64**
- Rejected groups retained in the audit: **0**

## Object-balanced point prediction

Primary metric: `tactile_field_rmse`. Lower is better.

| Method | Value |
|---|---:|
| `persistence` | 0.012961312 |
| `last_residual` | 0.037575217 |
| `map_motion` | 0.012961312 |
| `bayesian_motion` | 0.013447951 |
| `guarded_bayesian_motion` | 0.012961312 |

## Joint uncertainty diagnostics

| Diagnostic | Value |
|---|---:|
| `coordinate_nll_standardized` | 9.828299e+09 |
| `energy_score_standardized` | 8.2333043 |
| `joint_90_ellipsoid_coverage` | 0.68388195 |
| `joint_nanees` | 1.9656598e+10 |
| `marginal_90_coverage` | 0.9050311 |
| `mean_marginal_90_width_standardized` | 7.1690498 |

## Paired object-level contrasts

Negative guarded-minus-comparator values favor the guarded Bayesian method.

| Comparator | Mean difference | 95% object bootstrap | W/T/L | Worst regret |
|---|---:|---:|---:|---:|
| `persistence` | 0 | [0, 0] | 0/16/0 | 0 |
| `last_residual` | -0.024613905 | [-0.029681689, -0.020330342] | 16/0/0 | -0.013935125 |
| `map_motion` | 0 | [0, 0] | 0/16/0 | 0 |

## Evidence boundary

This is a retrospective public-real-data **development evaluation**. The
target recording in every group was selected from path identity and metadata
before target numeric loading; source model selection and covariance calibration
were frozen first. Reserved confirmation objects were never numerically opened.
Frames and sensor streams are averaged within physical objects and are not
treated as independent inferential units.

A tactile primary result validates real contact-response forecasting and joint
uncertainty, but it does not by itself validate dense 4-D geometry or a strict
counterfactual intervention claim. A geometry primary result uses only released
processed carriers discovered beneath the frozen dataset root.

No raw recording, point cloud, tactile tensor, or private trajectory is retained
in this evidence bundle. `paper_claim_authorized` and
`fresh_confirmation_authorized` remain false.
