# Deform360 exact-query-marginal copula audit v1

## Question

The completed v6 dependence study preserved the frozen point prediction and all
field-coordinate marginals. Its linear physical queries nevertheless had
different marginal variances, because those variances are induced by the field
covariance. This retrospective audit asks a stricter question:

> Does the coupling among physical queries improve joint-event probabilities
> and an offline execute-versus-fallback decision when each complete empirical
> univariate query distribution is exactly the same?

## Frozen construction

The workflow reruns the exact 92-object bound-carrier v6 implementation and
requires its complete scientific projection to reproduce the retained result
from workflow `33528032875`. It then projects the full covariance to the five
previously registered physical queries.

For each physical object, the three audit arms are:

1. `full_query_copula`: the Gaussian copula induced by the projected covariance;
2. `independent_query_copula`: independent row permutations of each query
   sample column; and
3. `scrambled_query_copula`: a fixed, target-independent derangement of the
   full-copula rank carriers with fixed signs.

Every control is produced only by permuting values within each query column.
Consequently, every arm has the same predictive query mean, the same sorted
residual samples for every query, and the same probability for every
single-query event at every target window. Only cross-query coupling differs.

## Endpoints

No target event pair is selected. All ten unordered pairs of the five existing
queries are evaluated as both conjunctions and disjunctions, giving twenty
joint events per object. The primary object-balanced endpoints are:

- joint-event Brier score; and
- realized loss of the existing execute-versus-fallback rule.

The complete result is retained whether positive, mixed, negative, or
inconclusive. A positive audit requires the object-bootstrap 95% upper bound for
`full minus control` to be below zero for both endpoints and both controls.

## Evidence boundary

The 92-object target cohort was already opened before this protocol was
designed. This is therefore retrospective mechanism evidence, not fresh
confirmation. It opens no camera pixels, geometry, point clouds, additional
objects, or unbound numeric carriers and collects no new measurement.
