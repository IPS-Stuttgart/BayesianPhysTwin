# Full-22 covariance-only hybrid v1

## Scientific question

The full-22 uncertainty-value diagnostic found that Bayesian endpoint models
improved Gaussian negative log score even though their predictive means did not
improve physical point error. This experiment isolates the suggested mechanism:

> Does Bayesian covariance improve the predictive distribution when the
> `last_residual` predictive mean is preserved exactly?

The hybrid accepts the exact caller-owned `last_residual` mean array and returns
that same array object. It imports only a covariance from one of two frozen
Bayesian endpoint models. The hybrid cannot change track error or Chamfer distance
because those scores depend only on the unchanged point trajectory.

## Registered development experiment

The exact source is the already-sealed full-22 tournament artifact from workflow
run `31410594302`, artifact `9074451004`, with digest
`sha256:22984bd34992ef7693c7577045c7496f8de2990641c3d2592ce230b9fbc97220`.
The 22 physical object sessions are independent units with equal weight.

Two covariance donors are registered in advance:

- `independent_endpoint_v1`;
- `dynamic_endpoint_v2`.

For each of 22 outer leave-one-object-out folds, the other 21 objects select one
positive covariance scale separately for the early, middle, and late horizon.
The same 21 objects then select one donor by equal-object, equal-horizon Gaussian
negative log score. The held object cannot tune its own donor or scale.

The finite scale grid is
`[0.25, 0.5, 1, 2, 4, 8, 16]`. No isotropic variance is outcome-fitted. The
observation model remains the registered common 5 mm standard deviation and
`1e-12 m²` eigenvalue floor. The primary comparison is the cross-fitted hybrid
minus zero-covariance `last_residual`; lower is better.

The primary interval is a 100,000-replicate case-clustered max-t 95% confidence
interval simultaneous over overall, early, middle, and late effects. Raw donor
covariances and donor-specific cross-fitted scales are secondary diagnostics.
Marginal 90% coverage and full interval width are reported descriptively.

## Exact mean boundary

`bayesian_phystwin.covariance_only_hybrid` enforces all of the following:

- the reference mean is a finite, C-contiguous `float64` NumPy array;
- the returned mean is the exact same Python object;
- donor covariance has matching dimensions and is finite, symmetric, and
  positive semidefinite;
- the scale schedule is finite and strictly positive;
- the output covariance is immutable; and
- a content-addressed record binds the reference mean, donor covariance, scale
  schedule, output covariance, and the invariant `point_prediction_changed=false`.

Observation noise is not silently added by the hybrid. It belongs to the frozen
scoring or deployment contract.

## Historical source-lineage boundary

The evaluator consumes the exact split sealed by the historical prefix manifest.
Before opening scoring arrays, it independently verifies all of the following:

- the prefix-manifest content identity and information boundary;
- every prefix-case archive and its sealed split scalars;
- each prediction manifest's content identity and prefix-manifest binding; and
- the SHA-256 of `final_data.pkl`, `inference.pkl`, `gt_track_3d.pkl`, and
  `split.json` for every case.

It never recomputes the historical split with current helper code.

## Sealed retrospective result

The registered execution completed successfully in workflow run `31461910994`,
attempt 1, on analyzer revision
`9ac5344036331a40e9029a1aa5814f601c0eaf15`. Its compact evidence is recorded in
[`results/science/full22_covariance_only_hybrid_v1/`](../results/science/full22_covariance_only_hybrid_v1/).

The primary cross-fitted selected/scaled covariance arm changed Gaussian NLL by
`-9.136`, with simultaneous 95% CI `[-13.961, -4.312]`. It was better in `17/22`
complete object-session units and worse in `5/22`. The simultaneous early,
middle, and late effects were respectively:

- `-1.539 [-2.874, -0.203]`;
- `-8.233 [-13.222, -3.243]`; and
- `-17.638 [-26.324, -8.952]`.

Exact mean identity held in all 22 cases, so track and Chamfer differences are
exactly zero. Marginal 90% coverage increased from `0.706` to `0.910`, while mean
full interval width increased from `0.01645 m` to `0.05094 m`, a `3.10×` ratio.
The result therefore identifies useful uncertainty with a material width cost,
not an improved point trajectory.

The outer selector chose `independent_endpoint_v1` in `21/22` folds and
`dynamic_endpoint_v2` in `1/22`. The frozen full-source candidate for one
separate fresh study is the exact `last_residual` mean, covariance donor
`independent_endpoint_v1`, and early/middle/late scales `[8.0, 16.0, 16.0]`.

## Scientific boundary

This experiment is stronger than an in-sample covariance transplant because the
scored object cannot tune its own donor or scale. It is nevertheless
retrospective development evidence: the released full-22 outcomes were already
open before this secondary hypothesis was formulated.

The full-source donor and horizon scales may be used only to freeze a separate
fresh-object protocol. They cannot themselves authorize selection, promotion,
deployment, or a scientific claim. The existing Deform360 v6 study is not
modified because its candidate roster and target boundary were frozen earlier.
