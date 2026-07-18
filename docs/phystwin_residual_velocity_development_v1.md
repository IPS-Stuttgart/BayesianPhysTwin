# PhysTwin residual-velocity development result

Run date: 2026-07-18

Status: recurrent per-case and shared linear residual-velocity families rejected
on the three already-open sloth development interactions. The exploratory
19-case cohort remained closed.

## Per-case recurrent model

The model fits a low-rank residual state and recursively integrates predicted
residual velocity from controller and baseline-physics features. Hyperparameters
are selected on the untouched interval between `fit_end_frame` and
`train_end_frame`; endpoint persistence is the exact fallback.

| Case | Validation CD | Validation track | Future CD | Future track |
| --- | ---: | ---: | ---: | ---: |
| `single_lift_sloth` | -8.212% | -6.897% | +2.282% | +1.234% |
| `double_lift_sloth` | -0.918% | +0.822% | fallback | fallback |
| `double_stretch_sloth` | -10.766% | -7.515% | +2.071% | +3.304% |

The two accepted validation models both reverse sign on the longer future.
This is a direct selection-overfit result, not evidence that residual dynamics
are absent.

## Cross-episode shared linear model

The shared model fits a per-node world-frame velocity law with leave-one-action-
out folds and causal local adaptation. Its best candidate uses smoothing
`0.25` and local prior strength `10`. Relative to persistence, aggregate
validation CD worsens by `4.139%`, track error improves by only `0.250%`, no
fold passes its individual gate, and there are `0/3` two-metric wins. The
balanced improvement is `-1.944%` under the implementation's positive-is-better
convention. Future metrics remain unopened.

## Conclusion

Neither per-case recursive regression nor a pooled linear velocity law is a
credible route past endpoint persistence. The failure pattern motivates a
nonlinear spatial-temporal residual model trained across many PhysTwin source
rollouts, with cross-action long-horizon selection. It does not justify opening
the 19-case future cohort for either implemented family.

Machine evidence is archived under
`results/sota/diagnostics/residual_velocity_v1/` and
`results/sota/diagnostics/shared_residual_velocity_v1/`.
