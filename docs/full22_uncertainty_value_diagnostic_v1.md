# Full-22 Bayesian uncertainty-value diagnostic v1

## Scientific question

The sealed full-22 discrepancy tournament found no point-prediction winner over
`last_residual`. That result leaves a narrower scientific question open:

> Does any Bayesian candidate improve the registered predictive-distribution
> score even when its physical point prediction does not improve over the
> deterministic last-residual reference?

This diagnostic tests that question directly on the immutable future-score table
from workflow run `31410594302`, artifact
`bpt-full22-discrepancy-31410594302-1`. The artifact is bound by digest
`sha256:22984bd34992ef7693c7577045c7496f8de2990641c3d2592ce230b9fbc97220`.

## Comparisons

The deterministic `last_residual` method is the reference. The four registered
Bayesian candidates were analyzed without selecting one in advance:

- `independent_endpoint_v1`;
- `dynamic_endpoint_v2`;
- `structured_kernel_rank4_v1`; and
- `graph_dynamic_kernel_rank4_v1`.

The physical fallback was retained only to verify exact fallback values. It was
not part of the Bayesian-candidate hypothesis family.

## Outcomes and statistical design

For every physical object session and each of the `early`, `middle`, and `late`
horizons, the analysis compares candidate-minus-reference differences in:

- the registered regularized Gaussian negative log score;
- official track error; and
- official Chamfer distance.

Raw candidate behavior and guarded deployed behavior are analyzed separately.
Lower values are better for every endpoint.

The 22 complete physical object sessions are the independent resampling units.
The overall effect gives every object equal weight and averages the three
horizons within object before pooling across objects. A deterministic 100,000-
replicate case-clustered bootstrap reports ordinary 95% intervals and max-t
simultaneous 95% intervals over the four candidates and four time aggregations,
separately for every endpoint and raw/deployed stream.

The Gaussian score is
`mean-valid-track-gaussian-nll-common-5mm-floor-v1`: each candidate's raw
predictive covariance is augmented by the same 5 mm observation-noise standard
deviation and a `1e-12 m²` eigenvalue floor. A candidate is called familywise
better only when the upper simultaneous bound is below zero. No practical
margin was estimated from these already-open outcomes.

## Completed result

The exact analysis completed successfully in workflow run `31456300622`, attempt
`1`. The compact five-file result artifact is
`full22-uncertainty-value-v1-31456300622-1` (`9088165631`), size `43,473`
bytes, with digest
`sha256:7b7c433db139842d2272d8ed92ba7d27151c30a18a250f6e0271516b80256ca0`.
The deterministic report ID is
`75f02ffdfde2588ceb05843f82b4092faae60294a943b1f076b19318566304cf`.
The repository record is in
[`results/science/full22_uncertainty_value_v1/summary.json`](../results/science/full22_uncertainty_value_v1/summary.json).

All effects below are candidate minus `last_residual`; lower is better.
Intervals are simultaneous across all four Bayesian candidates and the
`overall`/`early`/`middle`/`late` aggregations for the corresponding endpoint.

| Raw candidate | Gaussian NLL effect | Simultaneous 95% CI | Track effect | Chamfer effect |
| --- | ---: | ---: | ---: | ---: |
| `independent_endpoint_v1` | **-5.545** | **[-9.485, -1.606]** | +0.144 mm, inconclusive | +0.137 mm, inconclusive |
| `dynamic_endpoint_v2` | **-6.907** | **[-12.819, -0.994]** | +0.432 mm, inconclusive | **+0.237 mm, worse** |
| `structured_kernel_rank4_v1` | **-10.351** | **[-18.510, -2.191]** | **+1.695 mm, worse** | **+0.919 mm, worse** |
| `graph_dynamic_kernel_rank4_v1` | **+9.658** | **[+2.992, +16.323]** | **+2.898 mm, worse** | **+1.515 mm, worse** |

Three raw Bayesian distributions therefore carry familywise-supported proper-
score value despite no point-mean advantage. The strongest observed raw NLL
signal is the structured rank-4 candidate, but it also has a clear point-error
penalty and is not a promotion candidate. The graph-dynamic candidate is
familywise worse in raw NLL, track error, and Chamfer distance.

The useful uncertainty signal is horizon-dependent. For the independent,
dynamic, and structured candidates, the early-horizon NLL comparisons are
inconclusive, while the middle and late effects are familywise better. Their
late-horizon effects are respectively:

- `independent_endpoint_v1`: `-11.318`, CI `[-18.624, -4.013]`;
- `dynamic_endpoint_v2`: `-14.175`, CI `[-24.193, -4.157]`; and
- `structured_kernel_rank4_v1`: `-20.585`, CI `[-33.370, -7.800]`.

This localizes the raw Bayesian value to growing forecast uncertainty rather than
to the immediate post-prefix point prediction.

## Deployment consequence

The current admission/fallback layer does **not** preserve the positive raw NLL
result at familywise level:

| Guarded deployment | NLL effect | Simultaneous 95% CI | Decision |
| --- | ---: | ---: | --- |
| `independent_endpoint_v1` | -3.782 | [-8.099, +0.536] | inconclusive |
| `dynamic_endpoint_v2` | -4.678 | [-10.819, +1.463] | inconclusive |
| `structured_kernel_rank4_v1` | -0.408 | [-7.909, +7.094] | inconclusive |
| `graph_dynamic_kernel_rank4_v1` | +7.348 | [+0.057, +14.639] | worse |

Thus the defensible interpretation is narrower than “the Bayesian method wins.”
The raw endpoint distributions contain retrospective proper-score information,
but the current prefix-validation guard and exact physical fallback dilute that
value. The next method work should target covariance calibration and an
uncertainty-aware, harm-bounded admission rule, not another discrepancy family.

The independent endpoint candidate is the least damaging starting point for
that work: it has a familywise raw NLL benefit while both point-error intervals
still cross zero. This is an engineering prioritization only, not model selection
or a scientific claim.

## Information boundary

This is retrospective source-only scientific localization. The original
candidate forecasts and admission decisions were sealed before the future was
scored, but this secondary question was formulated after the aggregate
full-cohort result was known. Consequently, every generated report fixes
`claim_authorized=false`, `promotion_authorized=false`, and
`selection_authorized=false`.

The result establishes neither calibrated raw covariance nor safe deployment.
It does not authorize model selection, fresh-object transfer, physical-state
identification, practical equivalence, or a state-of-the-art claim. The proper
score is tied to the registered 5 mm observation model; sensitivity to a
prospectively justified observation-noise model belongs in a separately frozen
experiment.
