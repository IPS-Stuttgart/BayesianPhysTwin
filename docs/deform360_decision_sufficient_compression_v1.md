# Decision-sufficient covariance compression on Deform360

## Contribution

The completed 92-object same-mean study establishes that the learned shared
cross-output dependence improves offline execute-versus-fallback decisions even
when the predictive mean and every coordinate marginal are fixed.  This
extension asks a stronger structural question:

> How much of that dependence must a physical twin retain for a fixed set of
> robot-relevant queries and decisions?

Let the frozen Gaussian field belief be

\[
X \sim \mathcal N(\mu, D + U U^\top),
\]

and let the registered physical-query portfolio be \(q = QX\).  Consider the
fixed orthogonal factor-projection family

\[
U \mapsto U V, \qquad V^\top V = I.
\]

The complete query distribution is preserved exactly if and only if

\[
\operatorname{range}(U^\top Q^\top)
\subseteq
\operatorname{range}(V).
\]

The minimum retained rank within this family is therefore

\[
r_\star = \operatorname{rank}(QU).
\]

For a scalar query, at most one shared factor is necessary.  For the five-query
Deform360 portfolio, at most five are necessary.  Since the mean, diagonal
block, and joint Gaussian distribution of \(QX\) are unchanged, every score,
event probability, execute/fallback decision, and Bayes risk depending only on
the registered queries is unchanged as well.

## Real-data evaluation

The experiment reuses the exact frozen predictor and bound 92-object Deform360
cohort from workflow `33528032875`.  It compares:

1. the complete low-rank factor;
2. the minimum portfolio-sufficient factor;
3. a separately compiled minimum factor for each scalar query; and
4. a matched-rank spectral factor that retains globally energetic modes without
   using the query portfolio.

The first three representations use the same predictive mean, diagonal block,
source-only query calibration, and decision rule.  The sufficient projections
are functions only of the frozen covariance factor and registered query matrix;
target outcomes are not inputs.

The primary exactness endpoints are:

- maximum error in the five-dimensional query covariance;
- maximum scalar-query variance error;
- maximum difference in registered NLL, coverage, Brier score, decision loss,
  acceptance, and harmful-acceptance metrics;
- retained factor rank and factor-entry reduction.

The spectral comparator tests whether the result follows merely from reducing
the rank rather than retaining the query-visible subspace.

## Claim boundary

This experiment can establish exact representation sufficiency for the fixed
query portfolio within the supplied decomposition and orthogonal projection
family.  It does not preserve the unrestricted field covariance, arbitrary
future queries, or observation likelihoods not represented by the registered
queries.  It does not repair the underdispersion observed in the parent study,
and it is an offline logged-data decision experiment rather than closed-loop
robot execution.
