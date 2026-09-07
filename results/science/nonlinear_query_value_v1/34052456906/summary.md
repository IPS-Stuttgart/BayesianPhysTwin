# Nonlinear deformation-query posterior test

Completed retrospective DEFORM DLO4/DLO5 replay: 28 evaluation trajectories, 498 forecast frames each, 27 quadratic queries.

All readouts use exactly the same coordinate mean. RMSE below is dimensionless, normalized using source-only family scales. Positive improvement means lower RMSE.

| Panel / query | Readout | Normalized RMSE | Improvement vs point |
|---|---|---:|---:|
| all_queries/squared_distance | point_plugin | 0.071240 | +0.00% |
| all_queries/squared_distance | native_block_posterior | 0.071333 | -0.13% |
| all_queries/squared_distance | source_coupled_posterior_extension | 0.068785 | +3.45% |
| all_queries/squared_distance | empirical_global_centered | 0.068764 | +3.48% |
| all_queries/squared_distance | empirical_horizon_centered | 0.069414 | +2.56% |
| all_queries/squared_distance | empirical_symmetric_residual | 0.068445 | +3.92% |
| all_queries/squared_distance | source_query_bias | 0.063614 | +10.70% |
| all_queries/squared_distance | source_query_horizon_bias | 0.068365 | +4.04% |
| all_queries/squared_distance | source_query_ridge | 0.065869 | +7.54% |
| all_queries/bending_statistic | point_plugin | 0.480027 | +0.00% |
| all_queries/bending_statistic | native_block_posterior | 4.985942 | -938.68% |
| all_queries/bending_statistic | source_coupled_posterior_extension | 2.706584 | -463.84% |
| all_queries/bending_statistic | empirical_global_centered | 0.510727 | -6.40% |
| all_queries/bending_statistic | empirical_horizon_centered | 0.507545 | -5.73% |
| all_queries/bending_statistic | empirical_symmetric_residual | 0.520440 | -8.42% |
| all_queries/bending_statistic | source_query_bias | 0.420644 | +12.37% |
| all_queries/bending_statistic | source_query_horizon_bias | 0.448650 | +6.54% |
| all_queries/bending_statistic | source_query_ridge | 0.415288 | +13.49% |
| heldout_queries/squared_distance | point_plugin | 0.202189 | +0.00% |
| heldout_queries/squared_distance | native_block_posterior | 0.192224 | +4.93% |
| heldout_queries/squared_distance | source_coupled_posterior_extension | 0.192160 | +4.96% |
| heldout_queries/squared_distance | empirical_global_centered | 0.196469 | +2.83% |
| heldout_queries/squared_distance | empirical_horizon_centered | 0.197883 | +2.13% |
| heldout_queries/squared_distance | empirical_symmetric_residual | 0.195563 | +3.28% |
| heldout_queries/squared_distance | source_query_bias | 0.190907 | +5.58% |
| heldout_queries/squared_distance | source_query_horizon_bias | 0.191153 | +5.46% |
| heldout_queries/squared_distance | source_query_ridge | 0.199034 | +1.56% |
| heldout_queries/bending_statistic | point_plugin | 0.764477 | +0.00% |
| heldout_queries/bending_statistic | native_block_posterior | 7.338668 | -859.96% |
| heldout_queries/bending_statistic | source_coupled_posterior_extension | 4.053362 | -430.21% |
| heldout_queries/bending_statistic | empirical_global_centered | 0.850535 | -11.26% |
| heldout_queries/bending_statistic | empirical_horizon_centered | 0.837990 | -9.62% |
| heldout_queries/bending_statistic | empirical_symmetric_residual | 0.865752 | -13.25% |
| heldout_queries/bending_statistic | source_query_bias | 0.767807 | -0.44% |
| heldout_queries/bending_statistic | source_query_horizon_bias | 0.769513 | -0.66% |
| heldout_queries/bending_statistic | source_query_ridge | 0.689004 | +9.87% |

## Frozen decision rules

{
  "native_block_posterior": {
    "all_queries": false,
    "heldout_queries": false,
    "strong_control_superiority": false
  },
  "source_coupled_posterior_extension": {
    "all_queries": false,
    "heldout_queries": false,
    "strong_control_superiority": false
  }
}

Both query families must improve by at least 1% in each DLO and have negative upper paired-bootstrap MSE-difference bounds against the point readout. Strong-control superiority additionally requires beating every declared empirical and deterministic control.

The native posterior contains no cross-marker covariance. The separately named coupled extension is source-estimated, not a retained full Bayesian posterior. Source controls and target means inherit the parent 39-fit/all-56-refit difference. Trajectories, not frames or marker pairs, are the resampling units. Results describe two fixed objects, not unseen-object generalization.
