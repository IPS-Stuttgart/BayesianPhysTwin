# Quantitative interventional cause attribution v1

## Purpose

`interventional_cause_identifiability_v1` answers whether a registered
cause-specific query is determined by the available intervention responses after
all competing causes and declared nuisance directions are projected out. This
module makes the result quantitative. It returns:

- the identifiable component of each cause query;
- its minimum-covariance linear unbiased estimate under the declared whitened
  noise model;
- exact-model covariance and marginal confidence intervals;
- a deterministic finite-noise error radius when a residual-norm bound is
  supplied;
- the unresolved query map when the cause is only partially identifiable; and
- a minimum-cost diagnostic intervention portfolio.

The method never turns an unresolved component into a zero estimate. An interval
is marked as valid for the complete cause query only when the complete registered
query is identifiable.

## Model

For a registered cause `c`, stack the whitened observations produced by a finite
intervention set:

\[
r = S_c\beta_c + S_{-c}\beta_{-c} + N\nu + e.
\]

Let `C_c=[N,S_{-c}]` and let the columns of `Q_c` be an orthonormal basis for the
orthogonal complement of `col(C_c)`. The competitor-free response is

\[
y_c=Q_c^\top r,
\qquad
A_c=Q_c^\top S_c.
\]

For a registered cause query

\[
q_c=B_c\beta_c,
\]

define

\[
M_c=B_cA_c^\dagger.
\]

The estimated identifiable component is

\[
\widehat q_c=M_cy_c,
\qquad
B_c^{\mathrm{id}}=M_cA_c=B_cA_c^\dagger A_c.
\]

The unresolved map is

\[
B_c^{\mathrm{un}}=B_c-B_c^{\mathrm{id}}.
\]

It is exposed in every result. The full cause query is identifiable precisely
when `B_c^{un}=0` up to the frozen numerical tolerance.

## Theorem 1: minimum-covariance linear unbiased attribution

Assume the exact local model and whitened noise

\[
e\sim\mathcal N(0,\sigma^2I).
\]

If `ker(A_c) subseteq ker(B_c)`, then

\[
\widehat q_c=B_cA_c^\dagger Q_c^\top r
\]

is unbiased for `B_c beta_c`. Among every linear estimator `K Q_c^T r`
satisfying `K A_c=B_c`, it has minimum covariance in positive-semidefinite
order:

\[
\operatorname{Cov}(\widehat q_c)
=
\sigma^2B_cA_c^\dagger(A_c^\dagger)^\top B_c^\top.
\]

### Proof

Because `ker(A_c) subseteq ker(B_c)`, every row of `B_c` lies in the row space of
`A_c`, hence `B_cA_c^dagger A_c=B_c`. The displayed estimator is therefore
unbiased. Every other linear unbiased factor can be written as

\[
K=B_cA_c^\dagger+Z(I-A_cA_c^\dagger).
\]

The row spaces of the two summands are orthogonal. Consequently,

\[
KK^\top
=
B_cA_c^\dagger(A_c^\dagger)^\top B_c^\top
+
Z(I-A_cA_c^\dagger)Z^\top,
\]

and the second term is positive semidefinite.

For a partially identifiable query, the same construction is the BLUE for
`B_c^{id} beta_c`; it is not an estimator of the unresolved component.

## Theorem 2: deterministic finite-noise stability

When the projected whitened error satisfies `||Q_c^T e||_2 <= rho`,

\[
\|\widehat q_c-B_c^{\mathrm{id}}\beta_c\|_2
\le
\|B_cA_c^\dagger\|_2\rho.
\]

The implementation reports the amplification factor and, when `rho` is supplied,
the complete right-hand side.

## Theorem 3: additional registered interventions cannot worsen BLUE covariance

Suppose a second intervention portfolio contains every row of a first portfolio
and appends new intervention rows without changing the registered cause family or
noise whitening. Any unbiased estimator on the first portfolio remains a feasible
unbiased estimator on the enlarged portfolio by assigning zero weight to the new
rows. The BLUE on the enlarged portfolio therefore has covariance no larger in
positive-semidefinite order.

This is an exact-model statement. A changed or misspecified response relation can
still degrade real prediction, which is why the controlled study includes a
wrong-action placebo.

## Budgeted diagnostic-intervention planning

For a finite intervention roster `U`, registered costs `c(u)`, mandatory source
interventions `U_0`, and required cause set `C_0`, the exact planner enumerates
all feasible portfolios and first solves

\[
\min_{V\subseteq U}
\sum_{u\in V}c(u)
\quad\text{subject to}\quad
U_0\subseteq V,
\quad
B_c^{\mathrm{un}}(V)=0\;\forall c\in C_0.
\]

Ties are resolved by worst cause-query variance, worst noise amplification,
portfolio size, and canonical intervention order. If no feasible portfolio fully
identifies the registered target, the planner returns a
`budget_limited_partial_identification` result. Its lexicographic objective
maximizes full cause coverage, partial cause coverage, and identifiable query
energy before minimizing cost. It does not represent a partial portfolio as a
complete diagnosis.

Exact enumeration is intentionally limited to 15 registered interventions. Larger
rosters require a separately versioned approximate planner and approximation
claim.

## Controlled result

The controlled study uses five causes and five interventions. All cause signatures
are identical under the factual source action. Three diagnostic interventions
rotate the physical, contact, gauge, and discrepancy signatures differently; a
cheap fifth intervention is deliberately redundant.

The registered minimum-cost plan is:

```text
action-0-source
action-1-view-change
action-3-control-change
```

Its total cost is 1.8, compared with 3.5 for the full intervention roster. In
10,000 frozen trials at noise standard deviation 0.05, the retained run reports:

- 100% resolved coverage;
- 100% cause accuracy;
- 0% false physical promotion;
- 95.08% nominal-95% interval coverage;
- cause-query RMSE 0.0770, versus 0.0573 for the full roster;
- 49.92% resolved coverage for an equal-count random portfolio;
- zero resolved coverage for the factual action plus the cheap redundant probe;
- 57.02% accuracy and RMSE 0.7925 after swapping the diagnostic-action relation.

A factual-only forced physical label is at chance, with 20% accuracy, and promotes
every nonphysical case into a physical explanation. The source-action certificate
instead returns unresolved attribution.

## Role in “Why Is the Twin Wrong?”

The complete method stack is now:

1. **identifiability:** determine which registered cause-query components are
   distinguishable under the intervention portfolio;
2. **estimation:** estimate only those components with their exact-model
   covariance and finite-noise amplification;
3. **intervention design:** find the cheapest portfolio that resolves the pending
   attribution target;
4. **nonlinear closure:** verify that the local response signatures reproduce
   source-frozen nonlinear replays;
5. **held-intervention transport:** test the inferred correction under changed
   physical actions; and
6. **relation-breaking falsification:** require the result to deteriorate when
   action, contact, timing, object identity, or correspondence is deliberately
   broken.

The first three steps are formal and executable in this module family. Steps four
to six remain empirical admission requirements for a physical-cause claim.

## Scientific boundary

A passing estimate or plan is conditional on the exact finite cause family,
response signatures, nuisance design, whitening, query maps, and local linear
model. Omitted causes can imitate a registered cause. Exact-model confidence
intervals are not real-data calibration. The method does not establish a unique
data-generating mechanism, global nonlinear identifiability, unseen-object or
arbitrary-action transfer, online robotic control, deployment safety, or state of
the art.
