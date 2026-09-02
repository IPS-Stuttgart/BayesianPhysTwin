# Target-directed diagnostic intervention design

## Why full cause identification is unnecessarily expensive

After a source observation, the twin may retain several compatible error causes.
One response is to probe until every cause coefficient is identified. That can
waste physical interactions when the pending held-intervention query depends
only on a quotient of those coefficients.

For current stacked response design `S_0`, candidate diagnostic designs `D_u`,
and target map `T`, a subset `U` yields

\[
S_U=
\begin{bmatrix}
S_0\\
D_u,\ u\in U
\end{bmatrix}.
\]

The target is identifiable after `U` exactly when

\[
\ker(S_U)\subseteq\ker(T).
\]

Full cause identification instead requires the stronger condition

\[
\ker(S_U)=\{0\}.
\]

The first condition may require no probe or a much cheaper probe set.

## Exact finite-roster optimization

For a finite registered intervention roster and additive nonnegative cost, the
certificate enumerates every subset and solves

\[
U^*\in
\arg\min_U\sum_{u\in U}c(u)
\quad\text{subject to}\quad
\ker(S_U)\subseteq\ker(T).
\]

Ties are resolved only after retaining the complete set of equally optimal
subsets. The frozen order is:

1. minimum total intervention cost;
2. minimum number of interventions;
3. minimum target noise-amplification gain
   `||T S_U^dagger||_2`; and
4. canonical intervention-ID order.

The implementation separately reports the minimum cost of identifying every
cause coefficient. This quantifies the interaction cost saved by solving only
the pending target problem.

The exact search is intentionally bounded to at most twelve candidate
interventions. Larger portfolios require a separately validated optimization
method rather than silently replacing exactness with a heuristic.

## Partial and impossible portfolios

When no subset fully identifies the target, the certificate reports the maximum
identifiable target-output dimension. It may return `partial_improvement`, but
this does not authorize a complete target correction. If no subset improves the
current target quotient, the status is `unresolvable` and the caller retains its
fallback.

## Monotonicity

Appending correctly registered intervention rows can only shrink the coefficient
nullspace:

\[
\ker(S_{U\cup\{v\}})\subseteq\ker(S_U).
\]

Consequently, the target ambiguity space `T ker(S_U)` cannot grow and the target
identifiable dimension cannot decrease. This is an algebraic statement about the
supplied local designs, not a guarantee that a physical probe will match its
modeled response.

## Relationship to cause attribution

```text
cause-family adequate?
        | no -> none of the above
        v
pending target already invariant over cause ambiguity?
        | yes -> no diagnostic intervention
        v
minimum-cost target-identifying intervention subset
        | none -> partial report or exact fallback
        v
updated cause/transport certificate
        |
        v
nonlinear closure and held-intervention validation
```

The selected intervention need not uniquely identify state, material, contact,
gauge, and discrepancy. It is sufficient when it identifies the physical
quantity needed for the next decision.

## Claim boundary

The result is exact only for the supplied finite roster, additive costs, local
linear response designs, target query, coordinates, and tolerances. It does not
validate the response models, guarantee safe physical execution, establish
nonlinear closure, or prove real-world target improvement.
