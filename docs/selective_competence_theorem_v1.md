# Finite-sample selective simulator competence

## Scope

A simulator is compared with a registered fallback for a finite set of physical
query contexts. A context contains the object domain, action, horizon, query,
loss, and all pre-outcome features used by the policy. The method does not claim
that a simulator is globally valid.

Let the baseline-relative group loss in context `c` be

\[
D_{c,g}=L_{c,g}(a_c)-L_{c,g}(a_0),
\]

where `a_0` is the exact fallback. Negative values favor the candidate. The
policy is frozen before confirmation outcomes and may depend only on pre-outcome
information.

## Theorem 1: exact fallback

Define

\[
\pi(c)=\begin{cases}
 a_c,&c\in\mathcal A,\\
 a_0,&c\notin\mathcal A.
\end{cases}
\]

If the implementation returns the same immutable fallback object when
`c` is rejected, then for every rejected context and every outcome,

\[
L(\pi(c),Y)=L(a_0,Y).
\]

**Proof.** For rejected contexts, `pi(c)` and `a_0` are identical objects; the
loss receives identical arguments. No probabilistic assumption is required.

## Theorem 2: simultaneous source-side competence

Assume that, for every fixed context `c` in a finite family of size `K`, the
independent group losses satisfy `D_{c,g} in [a,b]`. Let `n_c` be the number of
source groups and `Dbar_c` their mean. Define

\[
U_c=\bar D_c+(b-a)\sqrt{\frac{\log(K/\alpha)}{2n_c}}.
\]

Then, with probability at least `1-alpha`, simultaneously for all contexts,

\[
E[D_c]\le U_c.
\]

Consequently, accepting only contexts with `U_c <= 0` implies that every
accepted context has nonpositive expected baseline-relative loss on the
registered source population, with confidence at least `1-alpha`.

**Proof.** One-sided Hoeffding gives

\[
P(E[D_c]>U_c)\le \alpha/K.
\]

A union bound over the `K` fixed contexts gives failure probability at most
`alpha`.

The theorem is only as relevant as its bounded-loss normalization, group
independence, and population definition. It is not a deployment-safety theorem.

## Theorem 3: exact harmful-use confirmation bound

Let a frozen policy generate `n` accepted confirmation uses. Suppose their
harm indicators are exchangeable Bernoulli variables under the registered
confirmation population, and let `k` be harmful uses. Let
`U_CP(k,n;alpha)` be the one-sided Clopper--Pearson endpoint. Then

\[
P\{p_{harm}\le U_{CP}(K,n;\alpha)\}\ge 1-\alpha
\]

for every true harmful-use probability `p_harm`.

This statement remains valid after arbitrary source-side model and policy
selection only because the policy is frozen independently of confirmation
outcomes. The statistical unit cannot be changed after seeing the result.

## Corollary: selective-twin admission

If a frozen policy satisfies all of the following on its registered evidence:

1. a nonempty accepted set;
2. a simultaneous source regret endpoint no greater than zero;
3. a confirmation harmful-use endpoint no greater than the registered tolerance;
4. exact fallback on every rejection;

then the policy has a finite-sample competence certificate for that exact
simulator, query family, fallback, loss, and population. The certificate does
not transfer automatically to new objects, actions, simulators, or queries.
