# Full-22 Bayesian uncertainty-value diagnostic v1

## Scientific question

The sealed full-22 discrepancy tournament found no point-prediction winner over
`last_residual`. That result leaves a narrower scientific question open:

> Does any Bayesian candidate improve the registered predictive-distribution
> score even when its physical point prediction is indistinguishable from the
> deterministic last-residual reference?

This diagnostic tests that question directly on the immutable future-score table
from workflow run `31410594302`, artifact
`bpt-full22-discrepancy-31410594302-1`. The artifact is bound by digest
`sha256:22984bd34992ef7693c7577045c7496f8de2990641c3d2592ce230b9fbc97220`.

## Comparisons

The deterministic `last_residual` method is the reference. The four registered
Bayesian candidates are analyzed without selecting one in advance:

- `independent_endpoint_v1`;
- `dynamic_endpoint_v2`;
- `structured_kernel_rank4_v1`; and
- `graph_dynamic_kernel_rank4_v1`.

The physical fallback is retained only to verify exact fallback values. It is not
part of the Bayesian-candidate hypothesis family.

## Outcomes

For every physical object session and each of the `early`, `middle`, and `late`
horizons, the analysis compares candidate-minus-reference differences in:

- the registered regularized Gaussian negative log score;
- official track error; and
- official Chamfer distance.

Raw candidate behavior and guarded deployed behavior are analyzed separately.
Lower values are better for every endpoint.

## Statistical design

The 22 complete physical object sessions are the independent resampling units.
The overall effect gives every object equal weight and averages the three
horizons within object before pooling across objects. A deterministic
case-clustered bootstrap reports ordinary 95% intervals and max-t simultaneous
95% intervals over the four candidates and four time aggregations, separately
for every endpoint and raw/deployed stream.

The report also includes paired sign counts, exact sign-test probabilities, Holm
adjustment for the four overall candidate comparisons, worst and best object
effects, and leave-one-object-out sign stability.

A candidate is called familywise better only when the upper simultaneous bound
is below zero. No practical margin is estimated from these already-open
outcomes.

## Information boundary

This is retrospective source-only scientific localization. The original
candidate forecasts and admission decisions were sealed before the future was
scored, but this secondary question was formulated after the aggregate
full-cohort result was known. Consequently, every generated report fixes
`claim_authorized=false`, `promotion_authorized=false`, and
`selection_authorized=false`.

The result can show whether Bayesian structure contributes predictive-
distribution value rather than point-mean value. It cannot authorize model
selection, fresh-object transfer, calibrated deployment, physical-state
identification, or a state-of-the-art claim.
