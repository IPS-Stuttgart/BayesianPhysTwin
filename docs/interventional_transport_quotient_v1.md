# Held-intervention transport over a cause ambiguity set

## Motivation

A physical cause may remain ambiguous even after the registered cause family has
passed its adequacy test. That does not imply that every downstream correction is
ambiguous. Several compatible causes can induce the same response for a
particular held intervention and physical query.

This certificate therefore separates two questions:

1. **Why is the twin wrong?** Which coefficient in the registered cause family
   generated the residual?
2. **What transports?** Which held-intervention query is invariant across every
   compatible cause explanation?

The second question can have a unique answer when the first does not.

## Affine cause ambiguity

For adequate stacked cause design `S` and residual `r`, the compatible local
coefficient set is

\[
\mathcal B(r)=\beta_0+\ker(S),
\qquad
\beta_0=S^\dagger r.
\]

Let `T` map cause coefficients to a registered target-intervention query. The set
of compatible target effects is

\[
\mathcal T(r)=T\beta_0+T\ker(S).
\]

The complete target correction is identifiable exactly when

\[
T\ker(S)=\{0\},
\]

or equivalently

\[
\ker(S)\subseteq\ker(T).
\]

This is also equivalent to the existence of an operator `M` such that

\[
T=MS.
\]

In that case the target correction can be computed directly from the residual:

\[
T\beta=M r=T S^\dagger r,
\]

without choosing a unique cause coefficient.

## Partial target identifiability

For vector queries, let `N` be a basis of `ker(S)` and let

\[
K=TN.
\]

`range(K)` is the target-output ambiguity subspace. If `P_id` projects onto its
orthogonal complement, then

\[
P_{id}T(\beta_0+Nz)=P_{id}T\beta_0
\quad\text{for every }z.
\]

The certificate returns this invariant component and the explicit ambiguity
projector. It never treats the representative component inside `range(K)` as a
supported target effect.

## Stability

Let the whitened residual perturbation obey `||e|| <= rho`. On the identifiable
target subspace,

\[
\left\|\Delta q_{id}\right\|_2
\le
\left\|P_{id}T S^\dagger\right\|_2\rho.
\]

The operator norm and resulting error bound are reported for every target query.
This is a local deterministic perturbation bound, not a calibrated probability
statement.

## Fail-closed order

```text
cause-family adequacy
        | fail -> unmodeled cause / exact fallback
        v
cause coefficient ambiguity set
        |
        v
target query invariant over ambiguity?
        | yes -> full target correction
        | partly -> identified target projection only
        | no -> diagnostic intervention or exact fallback
        v
nonlinear closure and empirical held-intervention transport
```

A `fully_identifiable` record authorizes only the linear target candidate. It
still requires nonlinear closure, source-group value evidence, and a sealed
held-intervention evaluation before a paper-facing physical benefit claim.

## Scientific implication

The hierarchy is strict:

\[
\text{unique cause identification}
\Longrightarrow
\text{target transport identification},
\]

but the converse is false. A robot may know how an observed error changes its
pending action without knowing whether the latent explanation should be called
state, gauge, contact, or material.

This prevents a false dichotomy between forcing a physical-cause label and
rejecting all useful information.
