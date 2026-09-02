# Switching-null anytime admission v3

## Motivation

The shared-alpha intersection--union rule in version 2 is efficient: mean-gain
and harm-rate evidence each use the full epoch alpha, and admission requires
both component boundaries to have been crossed. Its guarantee has an important
condition, however: **at least one fixed component null must hold throughout the
whole admission epoch**.

That condition is appropriate when a correction is consistently unhelpful or
consistently too harmful. It does not cover an epoch in which the reason for
rejecting the correction changes. For example, a candidate could initially be
useful but harm too many cases, then later become harmless but systematically
worse than fallback. Latching the two component crossings can combine evidence
collected in those incompatible regimes.

Version 3 supplies a conservative certificate for this stronger setting.

## Pointwise switching-union null

Let \(G_t\in[-1,1]\) denote the registered bounded gain score. Under insufficient
conditional mean gain,

\[
\mathbb E[G_t\mid\mathcal F_{t-1}]\le 0.
\]

Let \(H_t\in\{0,1\}\) denote the registered materially harmful-outcome indicator
and let \(\rho\in(0,1)\) be the maximum tolerated conditional harm probability.
Define

\[
S_t=\frac{\rho-H_t}{\max(\rho,1-\rho)}.
\]

Then \(S_t\in[-1,1]\), and whenever

\[
\Pr(H_t=1\mid\mathcal F_{t-1})\ge\rho,
\]

we have

\[
\mathbb E[S_t\mid\mathcal F_{t-1}]\le0.
\]

The switching-union null requires that at every reveal at least one of these two
conditional inequalities holds. The active component may change arbitrarily
with time.

## Robust joint score

Define

\[
Z_t=\min(G_t,S_t).
\]

Because \(Z_t\le G_t\) and \(Z_t\le S_t\), whichever component null is active at
time \(t\) implies

\[
\mathbb E[Z_t\mid\mathcal F_{t-1}]\le0.
\]

Therefore, for every fixed \(\lambda\in(0,1)\),

\[
M_t(\lambda)=\prod_{i=1}^{t}(1+\lambda Z_i)
\]

is a nonnegative supermartingale under the pointwise switching-union null. A
frozen convex mixture over betting fractions is an e-process, so

\[
\Pr\!\left(\sup_t E_t\ge1/\alpha_j\right)\le\alpha_j
\]

for admission epoch \(j\). Geometric alpha spending extends the bound over an
unbounded number of externally declared epochs.

The pointwise minimum is the largest scalar score that is guaranteed not to
exceed either component score. The certificate is consequently robust but can
be substantially less powerful than the stable-component intersection--union
rule.

## Controlled assumption stress test

The frozen protocol `protocols/anytime_switching_admission_v3.json` includes a
500-observation switching-null process:

1. for 50 observations, the correction has strongly positive mean gain but a
   20% harm rate against a 10% ceiling;
2. for 450 observations, it causes no materially harmful outcomes but has
   negative mean gain.

At every reveal one admissibility requirement fails, and the aggregate mean gain
is also negative. Nevertheless, the efficient latched version-2 rule can first
collect gain evidence in phase one and then collect low-harm evidence in phase
two. This is outside its theorem boundary and is retained deliberately as an
assumption counterexample.

The version-3 process uses the same first-epoch alpha and betting grid but updates
only with \(Z_t\). The protocol additionally contains two fixed-null controls and
two genuinely beneficial low-harm alternatives. It reports crossing
probabilities, Wilson intervals, and time-to-crossing for both controllers.

## Operational contract

Every controller binds by SHA-256:

- the candidate correction;
- the exact physical fallback;
- the gain score;
- the harm definition and ceiling;
- the available information set;
- the reveal policy; and
- the statistical configuration.

Trials are registered before their outcomes mature. Outcomes from an earlier
epoch remain in the audit trail but cannot update a later epoch. Selection
verifies candidate and fallback identifiers and returns the exact registered
object.

## Recommended hierarchy

Use version 2 when the application can justify an epoch with one stable
invalidity mode; it is more efficient and separately certifies mean utility and
harm rate. Use version 3 when the invalidity reason itself may change inside the
epoch and a pointwise robust certificate is required.

This gives a transparent robustness ladder:

\[
\begin{array}{ll}
\text{stable component null} &\rightarrow \text{shared-alpha IUT},\\
\text{switching component null} &\rightarrow \text{minimum-score certificate},\\
\text{contract or support change} &\rightarrow \text{new epoch / exact fallback}.
\end{array}
\]

## Claim boundary

The theorem concerns the frozen bounded gain and binary harm definitions under
predictable trial registration and paired delayed outcomes. It does not imply
physical safety, zero harmful deployments, validity under hidden changes to the
candidate or score, arbitrary-object transfer, causal identification, or
unclipped-loss control. The controlled experiment is not fresh real-world
validation.
