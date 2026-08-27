# Predictive Measurement Value: DEFORM Development Experiment

## Contribution Being Tested

The previous one-case sparse-budget experiment found that an assumed
graph-persistent covariance could shrink while actual errors did not. This
experiment tests three concrete changes before proposing a larger study:

1. Learn the relationship between prefix residuals and hidden future residuals
   from other complete trajectories.
2. Retain a component of future model discrepancy that observations cannot
   eliminate.
3. Use nested source validation to decline an empirically harmful mean update.

The prospective contribution is **validated predictive value of measurements
under model discrepancy**, not a claim to have invented active sensing,
Gaussian conditioning, covariance shrinkage, or information gain.

Relevant prior work includes [Krause et al.](https://jmlr.org/papers/v9/krause08a.html)
on GP sensor selection, [Caccamo et al.](https://arxiv.org/abs/1802.04691) on
probabilistic active deformability estimation, and
[JIGGLE](https://arxiv.org/abs/2405.09743) on active boundary estimation with a
soft-body simulator. A recent [sparse-touch mesh estimator](https://arxiv.org/abs/2607.13479)
also uses learned uncertainty to select touches. These establish why a generic
"Bayesian active sensing" description is insufficient as a novelty claim.
The present test concerns hidden **future** prediction with an already good
fixed dynamics predictor, while explicitly checking misspecification and
declining updates. Any distinctiveness still requires experimental evidence.

## Data and Information Boundary

The only input is the already-open DLO2 v7 archive, SHA-256
`431c778022bfb7b602512e5e6c2132a3f42e5959c959368e5203059bd2ce223b`.
It contains 14 trajectories from one DLO object. The previous design case,
`103.pkl`, can contribute to training but is excluded from reported outer
holdouts. The other 13 cases each serve once as a whole-trajectory holdout.
This is exploratory cross-validation of previously opened data, not a fresh
test set or official benchmark rerun.

For each outer forecast:

- all observations from that trajectory are excluded from covariance fitting;
- the predictor receives its frozen reference, marginal variance, and only the
  permitted prefix, never its own future truth;
- the other 13 trajectories supply training residuals;
- nested guard selection leaves out each eligible inner trajectory in full;
- the design case is never used as a scored inner or outer validation case;
- all outer predictions are written before outer metric aggregation.

Other folds' future outcomes are necessarily used as cross-validation training
data. Do not misdescribe this as "no future outcomes opened before prediction."
The invariant is that a forecast cannot use its **own** future in fitting,
selection, or guard choice. Frames, coordinates and repeated random schedules
are not independent physical trials. There is one object, not 13 objects.

The successful DEFORM/local-residual predictor, original evaluation artifacts,
and previous sparse-budget result remain unchanged. No DLO4, DLO5, held-v8,
Causal4D acquisition, sealed target, robot, or new perception provider is used.

## Fixed Measurement Contract

Reuse the previous declared prefix times, candidate identities, disjoint hidden
identities, horizon, 1 mm observation noise assumption, and budgets 0/1/2/4/8.
The native released coordinates are the only empirical observation condition.
Known initial states and future clamped actions remain allowed by DEFORM.

Compare random (32 fixed orders), spatial/temporal spread, maximum variance,
global information, future-query variance reduction, and a simple
`latest_uniform` schedule. The latter first observes graph-spread free nodes at
the latest available prefix frame, before spending budget on earlier frames.
Schedules depend on reference geometry, fitted covariance and fixed seeds,
never held-trajectory measurement values. Zero-information ties are filled in
index order so all policies expend the same point-frame budget.

Include a last-residual control on `latest_uniform`: for each observed free
identity use its latest available causal residual, interpolate along the chain
with zero residual at controlled boundaries, and persist that spatial field.
It retains baseline covariance and is a point-prediction control, not a new
uncertainty method.

## Learned Coupling and Floor

Express each source residual trajectory in an initial action frame computed
from its **reference prediction**, not its true shape. Source trajectories have
equal weight. Use a zero-centered second moment, preserving a zero correction
prior and the unchanged baseline mean at budget zero.

For source residual matrix `E` with one row per trajectory, use the low-rank
factor `F = sqrt((1-a)/n) E.T`. Retain the per-point 3x3 second-moment blocks
`D = a mean(e e.T)` as independent unresolved discrepancy. Then:

```text
r_observed = F_observed z + eta_observed + measurement_noise
r_future   = F_future   z + eta_future
z ~ N(0, I)
```

The future floor is never conditioned away. It is a shrinkage assumption,
not proof that the unresolved component is truly independent in space/time.
Full per-point 3x3 covariances are rotated back into the held reference frame.
Compare five fixed methods:

- `graph_persistence`: exact previous graph assumption and update.
- `empirical_no_floor`: source-learned coupling, `a=0`.
- `empirical_floor`: source-learned coupling, fixed `a=0.5`.
- `permuted_floor`: matched-complexity placebo, with future source residuals
  cyclically permuted relative to their prefix rows, `a=0.5`.
- `source_guarded_floor`: the floor model with nested source-only mean shrinkage.

The no-floor and floor arms have the same pointwise prior second moments.
Their distinction is which component can be reduced by measurements. Budget
zero preserves the exact baseline **mean** in all arms; learned prior
covariances can differ from the original archived covariance. Only the
unchanged baseline and a rejected guarded update promise exact covariance
fallback as well.

## Source Guard

For each outer fold, use its 12 non-design inner validations to evaluate blends
`0, 0.25, 0.5, 1`. Random-policy validation uses eight fixed orders per inner
trajectory, averaged before counting that trajectory. Choose a nonzero blend
only when it improves both mean coordinate-L1 and mean point-RMSE ratios by at
least 1%, jointly improves at least two-thirds of inner trajectories, and has
no inner metric ratio above 1.10. Among eligible blends, choose the smallest
balanced mean ratio, with smaller blend breaking exact ties. Otherwise use zero.

This is an **accuracy guard**, not a calibrated risk certificate. No finite
sample safety guarantee is claimed. Conditional uncertainty uses the first two
moments of a mixture between the unchanged and corrected forecasts, including
between-component spread. The score uses the resulting moment-matched
Gaussian, not the exact mixture density. A zero blend returns byte-identical
baseline mean and covariance.

## Metrics and Interpretation

Score disjoint hidden future identities with coordinate L1, 3D point RMSE,
early/middle/late coordinate L1, full-3D marginal Gaussian NLL, point NEES,
nominal 90% point-ellipsoid coverage, and ellipsoid volume. Add the same fixed
`0.001^2 I` score-noise floor to every method. These are per-point marginals,
not a joint trajectory distribution.

Average random orders within trajectory, then weight all 13 trajectories
equally. A fixed 10,000-replicate paired trajectory bootstrap is descriptive:
fold training sets overlap and all trajectories belong to the same object.
It is not an independent-object confidence interval or a multiplicity-adjusted
confirmation. Report every frozen arm and budget, including the placebo.

Three separate questions must be answered:

1. Does learned coupling outperform both unchanged DEFORM and last-residual
   interpolation on held trajectories?
2. Does retaining an unresolved floor improve predictive scoring/coverage,
   rather than merely narrowing the reported covariance?
3. Does future-query selection outperform the strongest simple selection policy
   at matched budget? A gain from coupling alone is not a selection novelty.

The positive synthetic control uses a known shared prefix/future latent field;
the zero-coupling control must trigger exact guarded fallback. Tests must also
prove whole-trajectory exclusion, candidate-specific prefix use, no dependence
on the held future, covariance PSD, and immutable original model/evidence.
Freeze and commit implementation/configuration before the empirical run. No
method, floor, guard, case, horizon or budget may be revised from its outcomes.
