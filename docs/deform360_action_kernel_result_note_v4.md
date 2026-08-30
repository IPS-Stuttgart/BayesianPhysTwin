# Deform360 action-kernel result note

The source-fitted nonlinear Bayesian action ensemble improves 32-frame real
tactile-response prediction over persistence in both evaluated cohorts:

- development: **5.75% lower** active-field RMSE, **14/14** object wins,
  95% object-bootstrap interval `[-0.04728, -0.02771]`;
- reserved confirmation: **4.11% lower** active-field RMSE, **4/4** object wins,
  interval `[-0.06871, -0.00336]`.

Permuting the future robot action degrades the ensemble in all 18 objects across
both cohorts. This supports an action-conditioned forecasting interpretation,
not merely temporal persistence.

The result remains tactile rather than dense 4-D geometry. Joint covariance is
miscalibrated, and the guarded deployment arm is not supported because its
source-selected fallback underperforms persistence on two reserved objects.
