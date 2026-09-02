# Interventional cause-family adequacy v1

## Why identifiability is not enough

A rank test can distinguish causes only inside the family supplied to it. If the
true mechanism is missing, a forced classifier can still assign the residual to
state, material, contact, gauge, or discrepancy. Good same-action fit then
becomes a false physical interpretation.

This certificate inserts a logically prior question:

> Is the observed residual compatible with **any** registered combination of
> cause signatures at the declared noise scale?

Only an adequate family may reach the cause-identifiability layer.

## Model

For stacked whitened residual `r`, registered cause signatures `S_c`, and total
signature matrix

\[
S=[S_1\;S_2\;\cdots\;S_C],
\]

the certificate computes the orthogonal decomposition

\[
r=S\hat\beta+e_\perp,
\qquad
\hat\beta=S^\dagger r,
\qquad
 e_\perp=(I-SS^\dagger)r.
\]

For registered deterministic noise radius \(\rho\), the family is adequate only
when

\[
\|e_\perp\|_2\le\rho.
\]

If this fails, the only permitted semantic outcome is `unmodeled_cause`. No
registered physical label is promoted.

## Set-valued attribution theorem

When the family is adequate, every coefficient vector that gives the same fitted
response is

\[
\mathcal B(r)=\hat\beta+\ker(S).
\]

Thus:

1. the complete coefficient attribution is unique iff \(\ker(S)=\{0\}\);
2. cause block \(c\) is identifiable iff the projection of \(\ker(S)\) onto that
   block is zero; and
3. equivalently, with all other cause signatures treated as nuisance,

\[
\operatorname{rank}\!\left(P^\perp_{S_{-c}}S_c\right)=\dim(\beta_c).
\]

The implementation retains the minimum-norm point and the entire nullspace. A
consumer may inspect the identifiable block dimensions, but it may not replace a
set-valued attribution by an arbitrary point label.

## Stability

Let \(\sigma_r\) be the smallest nonzero singular value of `S`. On the
identifiable coefficient subspace, a whitened residual perturbation bounded by
\(\rho\) obeys

\[
\|\Delta\beta_{\mathrm{id}}\|_2\le\rho/\sigma_r.
\]

This is a local finite-family statement. A small bound does not validate the
signature model or prove that the registered family is physically complete.

## Statuses

- `no_detectable_error`: the complete residual is within the noise radius;
- `unmodeled_cause`: the residual contains too much energy outside the family;
- `adequate_unique`: the family explains the residual and coefficients are unique;
- `adequate_set_valued`: the family explains the residual but an affine ambiguity
  remains.

Individual cause blocks are reported as identifiable, partially identifiable,
or confounded.

## Relationship to interventional cause identifiability

The intended order is:

```text
residual detected?
        |
        v
registered cause family adequate?
        | no -> unmodeled_cause / exact fallback
        v
action-stacked cause identifiable modulo competitors?
        | no -> set-valued attribution / exact fallback
        v
nonlinear closure and held-intervention transport
        |
        v
bounded physical interpretation
```

The adequacy certificate does not replace nonlinear closure, source-group proper
score evidence, held-intervention transport, relation-breaking placebos, or
independent-object validation.

## Claim boundary

A pass means only that one supplied local linear cause family can explain one
supplied whitened residual within one supplied deterministic radius. It does
not establish a unique data-generating cause, completeness of the cause family,
correct simulator physics, unseen-object transfer, deployment safety, or state
of the art.
