# Completed partial-observation belief mechanism test

## Execution and provenance

- Successful Actions run: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33984667329
- Real-data job: `101355923547`, runner `workstation1`, requested label `gpuserver4090`.
- Evaluated source revision: `8e00a95fa3b36d937b0f95dfc3433971e5192f68`.
- Protocol SHA-256: `04c9dbf1294ba42f11fd3dca05526f500d930e623518f91def1717a22e70d1eb`.
- Artifact: `partial-observation-belief-v1-33984667329`, ID `9974784792`, 17,366,659 bytes.
- Artifact SHA-256: `e5d17aab3d047823c350ef678c49069e6f03e2a0e09ba96b0979fa06040ba5d1`.
- Artifact download: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33984667329/artifacts/9974784792

Both the contract job and real-data job completed successfully. The eight numerical/noninterference checks passed. The earlier workflow-context failures occurred before any jobs were created; the scientific protocol was unchanged. This report transcribes the completed job's retained results and does not represent a second data run.

## Scientific result

**Joint conditioning improves hidden-region reconstruction and subsequent prediction relative to the unchanged prior and tuned interpolation. A distinctive low-rank/Bayesian advantage over all strong alternatives is not established.** The frozen `hypothesis_supported_against_all_controls` field is `false`.

The shared mean is source-fitted ridge plus endpoint-transported damped velocity. **This is NOT the released DEFORM physics/GCN hybrid, not a native simulator rerun, and not a test of an existing BayesianPhysTwin checkpoint.** It tests joint space-time residual conditioning on public measured trajectories with imposed fixed missingness.

## Cohort and information

Only the 56 public `train/` trajectories per object were used. Each DLO has 32 fit, 12 calibration, and 12 source-held test trajectories. All windows of a trajectory remain together. There are 24 test trajectories across two objects, 20 windows per trajectory, nine mask/budget cells, eight arms, and 34,560 scored rows. Rows, windows, and coordinates are not independent objects.

Two full initialization frames and the recorded boundary positions are supplied to every arm. At initialization +8 frames, fixed masks expose 1, 2, or 4 of eight internal nodes (12.5%, 25%, 50%). Hidden-current coordinate RMSE and all-free-node coordinate RMSE +16 frames later are co-primary; +4 is secondary. Official `eval/` files remain unopened. No new physical interaction or adaptive observation selection occurs. This is retrospective recorded-data replay, not real-camera occlusion validation.

## Point metrics

Values are millimetres, lower is better. Each row averages window RMSE, masks, and complete test trajectories with the declared equal weights.

| Arm | DLO4 hidden now | DLO4 future +16 | DLO5 hidden now | DLO5 future +16 |
|---|---:|---:|---:|---:|
| Prior / independent coordinates | 8.3698 | 22.9089 | 7.6491 | 19.9221 |
| Joint low-rank conditioning | 7.0917 | 20.0884 | 6.3040 | 17.8926 |
| Empirical shrinkage covariance | 7.0817 | 20.0702 | 6.3228 | 17.9012 |
| Tuned interpolation | 7.4555 | 21.3754 | 6.5584 | 18.6265 |
| Mask-aware conditional ridge | 6.8948 | 19.9789 | 6.2350 | 18.1575 |
| Marginal-preserving scrambled dependence | 10.5389 | 26.2184 | 9.5349 | 22.6037 |
| Deterministic MAP equivalent | 7.0917 | 20.0884 | 6.3040 | 17.8926 |

Joint conditioning reduces hidden-region RMSE versus prior by 15.27% on DLO4 and 17.58% on DLO5; future +16 RMSE improves by 12.31% and 10.19%. It wins 12/12 test trajectories on each co-primary metric in each object, after averaging the fixed mask/window cells. Equal-object mean reductions are 16.38% for hidden reconstruction and 11.32% for future prediction.

Both objects select rank 32 from the predeclared source-calibration grid. The deterministic precision-space MAP solution agrees numerically, as expected; its floating-point differences are not counted as scientific wins.

## Paired trajectory-bootstrap contrasts

Differences are joint low-rank minus comparator, in mm. Negative is favorable. Intervals resample 12 complete test trajectories within the indicated object; these are not unseen-object confidence intervals.

| Object / comparator | Hidden-current difference [95% CI] | Future +16 difference [95% CI] |
|---|---:|---:|
| DLO4 / prior | -1.2781 [-1.7185, -0.9402] | -2.8205 [-4.1055, -1.9601] |
| DLO5 / prior | -1.3451 [-1.5673, -1.1261] | -2.0295 [-2.6534, -1.4301] |
| DLO4 / interpolation | -0.3638 [-0.5160, -0.2165] | -1.2870 [-2.1296, -0.7585] |
| DLO5 / interpolation | -0.2544 [-0.3314, -0.1831] | -0.7339 [-1.1179, -0.3628] |
| DLO4 / empirical covariance | +0.0100 [-0.0091, +0.0337] | +0.0182 [-0.0082, +0.0488] |
| DLO5 / empirical covariance | -0.0188 [-0.0312, -0.0068] | -0.0086 [-0.0455, +0.0259] |
| DLO4 / masked ridge | +0.1969 [+0.0542, +0.3506] | +0.1096 [-0.1980, +0.4402] |
| DLO5 / masked ridge | +0.0691 [-0.2050, +0.2858] | -0.2649 [-0.9036, +0.2748] |

Empirical covariance is practically near-tied; the small DLO5 hidden-current advantage does not establish superiority on both endpoints and objects. Mask-aware ridge is significantly better for DLO4 hidden-current reconstruction. Neither object's future comparison against masked ridge excludes zero.

## Interpretation

The narrow result supports retaining and using cross-region/cross-time information instead of discarding it or replacing it with independent coordinate uncertainty. It does not establish that Bayesian treatment is uniquely responsible for a gain unavailable to empirical covariance or deterministic conditional prediction. The full all-control criterion fails and must not be relabeled as positive.

The artifact retains all per-case errors, available marginal coverage/NLL/width diagnostics, by-mask tables, source calibration choices, data hashes/splits, and prediction arrays. No claim of calibrated uncertainty is made from the point-accuracy results. No native physical-twin, material-parameter, real-camera, counterfactual, or general object-class claim follows from this pilot.
