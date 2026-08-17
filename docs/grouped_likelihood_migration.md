# Grouped likelihood migration

The historical public operation
`grouped_student_t_mixture_likelihood` remains available and preserves its
numerical interpretation. It is a covariance-marginalized diagnostic: declared
low-rank factors are folded into the observation covariance and row-level
`prior_reliability` is not used.

New analyses that need the grouped objective used by
`update_prior_aware_gauge_belief` should call
`conditional_grouped_student_t_mixture_objective`. Its prediction argument must
already contain the evaluated physical-state and nuisance contributions. The
operation then applies conditional local covariance, row reliability, the shared
nominal/outlier Student-t mixture kernel, and the declared group information
power.

Record the result's `semantics` property in manifests and paper-facing evidence:

```text
covariance-marginalized-student-t-score-v1
conditional-reliability-weighted-student-t-objective-v1
```

Do not compare the absolute values of the two scores as though they were the
same likelihood. They answer different questions and may use different residual
representations and information powers.

For frozen reproduction, keep the historical operation and exact source
revision. For new claim-bearing experiments, declare the intended operation,
composite-weight mode, group cap, degrees of freedom, outlier multiplier, and
conditional prediction construction before target outcomes are opened.
