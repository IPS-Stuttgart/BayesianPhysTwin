# Full-22 Bayesian uncertainty-value result

## Main result

The deterministic `last_residual` predictor remains the strongest point-mean reference, but it is not the strongest predictive distribution. On the already-open 22-object cohort, three Bayesian candidates improved the registered Gaussian negative log score after simultaneous correction over four candidates and four time aggregations. The graph-dynamic candidate moved in the opposite direction.

All effects below are candidate minus `last_residual`; lower is better. Intervals are case-clustered max-t simultaneous 95% confidence intervals.

| Candidate | Raw Gaussian NLL effect | Simultaneous 95% CI | Decision | Better / tie / worse objects |
| --- | ---: | ---: | --- | ---: |
| `independent_endpoint_v1` | -5.545 | [-9.485, -1.606] | candidate better | 18 / 0 / 4 |
| `dynamic_endpoint_v2` | -6.907 | [-12.819, -0.994] | candidate better | 16 / 0 / 6 |
| `structured_kernel_rank4_v1` | -10.351 | [-18.510, -2.191] | candidate better | 18 / 0 / 4 |
| `graph_dynamic_kernel_rank4_v1` | 9.658 | [2.992, 16.323] | candidate worse | 2 / 0 / 20 |

## Time localization

The uncertainty advantage is not present immediately after the causal cutoff. It appears as the forecast horizon grows, exactly where the zero-covariance last-residual forecast becomes increasingly overconfident.

| Candidate | Early NLL effect | Middle NLL effect | Late NLL effect |
| --- | ---: | ---: | ---: |
| `independent_endpoint_v1` | -0.336 | -4.982† | -11.318† |
| `dynamic_endpoint_v2` | -0.199 | -6.346† | -14.175† |
| `structured_kernel_rank4_v1` | -0.089 | -10.378† | -20.585† |
| `graph_dynamic_kernel_rank4_v1` | 12.909‡ | 9.709‡ | 6.355 |

† familywise better; ‡ familywise worse.

The absolute mean NLL of `last_residual` rose from −7.662 (early) to 4.336 (middle) and 16.167 (late). The structured candidate remained near the reference early (−7.752) but reached −6.042 in the middle and −4.418 late, producing the largest late-horizon gain (−20.585).

## Point-mean trade-off

The NLL gain is not explained by better point predictions. It is evidence that the Bayesian candidates encode forecast uncertainty that the deterministic persistence baseline omits.

| Candidate | Track-error effect (mm) | Simultaneous 95% CI (mm) | Chamfer effect (mm) | Simultaneous 95% CI (mm) |
| --- | ---: | ---: | ---: | ---: |
| `independent_endpoint_v1` | 0.144 | [-0.0910, 0.3787] | 0.137 | [-0.0091, 0.2829] |
| `dynamic_endpoint_v2` | 0.432 | [-0.0782, 0.9412] | 0.237 | [0.0005, 0.4726] |
| `structured_kernel_rank4_v1` | 1.695 | [0.4487, 2.9417] | 0.919 | [0.4908, 1.3479] |
| `graph_dynamic_kernel_rank4_v1` | 2.898 | [1.3987, 4.3978] | 1.515 | [0.7482, 2.2821] |

- `independent_endpoint_v1` is point-wise indistinguishable from `last_residual` on both metrics while improving NLL.
- `dynamic_endpoint_v2` improves NLL, is inconclusive on track error, and is slightly but familywise worse on Chamfer (+0.237 mm).
- `structured_kernel_rank4_v1` gives the strongest NLL gain but worsens track error by 1.695 mm and Chamfer by 0.919 mm.
- `graph_dynamic_kernel_rank4_v1` is worse on NLL and both point metrics.

## Exploratory action-regime localization

The following object-family means are descriptive only because the groups are small and were inspected after the aggregate result. They nevertheless give a concrete hypothesis for a fresh study: the useful uncertainty signal is concentrated in lifting and stretching, not pushing.

| Family | Objects | Independent NLL effect | Dynamic NLL effect | Structured NLL effect | Graph-dynamic NLL effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_lift` | 8 | -10.312 | -14.596 | -15.029 | 16.470 |
| `double_lift` | 4 | -5.405 | -4.787 | -24.483 | 6.002 |
| `double_stretch` | 2 | -6.221 | -5.473 | -5.268 | -3.605 |
| `single_push` | 4 | -0.490 | -0.341 | 1.053 | 5.080 |
| `single_clift` | 2 | -0.836 | -1.029 | -2.869 | 3.224 |
| `rope_double_hand` | 1 | -1.850 | -1.239 | -0.428 | 1.684 |
| `weird_package` | 1 | 0.038 | -0.422 | 2.942 | 35.460 |

Across all 66 object–horizon units, the NLL gain became more negative as the reference track error grew (Spearman correlations −0.475, −0.560, and −0.393 for the independent, dynamic, and structured candidates). The graph-dynamic effect instead correlated positively with reference error (+0.565), consistent with failure under the harder cases.

## Deployment-policy finding

The existing source-only guard does not preserve the raw uncertainty-score gains. Its accepted-object counts are 8/22 for `last_residual`, 9/22 for the independent and dynamic candidates, 7/22 for the structured candidate, and 5/22 for the graph-dynamic candidate.

| Candidate | Deployed NLL effect | Simultaneous 95% CI | Deployed decision |
| --- | ---: | ---: | --- |
| `independent_endpoint_v1` | -3.782 | [-8.099, 0.536] | inconclusive |
| `dynamic_endpoint_v2` | -4.678 | [-10.819, 1.463] | inconclusive |
| `structured_kernel_rank4_v1` | -0.408 | [-7.909, 7.094] | inconclusive |
| `graph_dynamic_kernel_rank4_v1` | 7.348 | [0.057, 14.639] | candidate worse |

The guard renders the independent, dynamic, and structured NLL advantages inconclusive; the graph-dynamic candidate remains familywise worse. Thus the current deployment policy is calibrated to point-loss protection, not to harvesting distributional value.

## Scientific interpretation

1. **There is a clear retrospective Bayesian uncertainty-value signal, not a mean-correction win.** This resolves the apparent contradiction between a point-prediction near-tie and useful Bayesian modeling.
2. **The value is horizon dependent.** A common 5 mm observation floor is adequate early, but persistence becomes overconfident at middle and late horizons.
3. **Low-dimensional independent or dynamic beliefs are preferable to the graph-dynamic model on this cohort.** Added graph dynamics degrade every primary endpoint.
4. **The structured model overtrades mean accuracy for uncertainty fit.** It is scientifically informative but not the best current deployment candidate.
5. **The sharp prospective hypothesis is covariance-only.** Preserve the `last_residual` mean exactly and import a frozen, horizon-conditioned covariance from the independent or dynamic Bayesian belief. This would test uncertainty value without allowing a hidden point-prediction change.

## Reproducibility and boundary

- Source evidence: workflow run `31410594302`, artifact `9074451004`, digest `sha256:22984bd34992ef7693c7577045c7496f8de2990641c3d2592ce230b9fbc97220`.
- Analysis: workflow run `31456300622`, artifact `9088165631`, digest `sha256:7b7c433db139842d2272d8ed92ba7d27151c30a18a250f6e0271516b80256ca0`.
- Report ID: `75f02ffdfde2588ceb05843f82b4092faae60294a943b1f076b19318566304cf`.
- The analysis is retrospective, source-only, and secondary. It authorizes neither model selection nor a confirmatory claim.
