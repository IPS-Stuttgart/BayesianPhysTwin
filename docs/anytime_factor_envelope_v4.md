# Factor-enveloped anytime admission v4

## Motivation

Version 2 gives the efficient rule when one fixed reason for rejecting a
candidate remains valid throughout an admission epoch. Version 3 removes that
stable-null assumption by replacing utility and harm with the single bounded
score

\[
Z_t=\min(G_t,S_t).
\]

That construction is valid, but it forces all constraints through one shared
betting fraction. A utility factor may prefer aggressive betting while a rare
harm indicator may prefer a different likelihood-ratio alternative. The shared
scalarization can therefore lose power for a purely technical reason.

Version 4 removes that coupling.

## General lower-envelope theorem

Let a conjunctive decision have component nulls
\(\mathcal H_{1,t},\ldots,\mathcal H_{K,t}\). For each component \(k\) and fixed
parameter \(\theta_k\), suppose

\[
F_{t,k,\theta_k}\ge0
\]

is measurable after reveal \(t\) and satisfies

\[
\mathbb E[F_{t,k,\theta_k}\mid\mathcal F_{t-1}]\le1
\]

whenever \(\mathcal H_{k,t}\) holds.

Assume only the pointwise union null:

\[
\forall t,\qquad
\mathcal H_{1,t}\ \lor\cdots\lor\ \mathcal H_{K,t}.
\]

The active component may change arbitrarily as a function of the past. For a
fixed parameter tuple
\(\theta=(\theta_1,\ldots,\theta_K)\), define

\[
L_{t,\theta}
=
\min_{k=1,\ldots,K}F_{t,k,\theta_k}.
\]

At every reveal there is at least one active component \(k^\star_t\). Since

\[
L_{t,\theta}\le F_{t,k^\star_t,\theta_{k^\star_t}},
\]

we have

\[
\mathbb E[L_{t,\theta}\mid\mathcal F_{t-1}]\le1.
\]

Consequently,

\[
E_{n,\theta}=\prod_{t=1}^{n}L_{t,\theta}
\]

is a nonnegative test supermartingale. For any outcome-independent prior
\(w_\theta\),

\[
E_n=\sum_\theta w_\theta E_{n,\theta}
\]

is an e-process. Ville's inequality gives

\[
\Pr\!\left(\sup_n E_n\ge\frac1\alpha\right)\le\alpha.
\]

The proof uses neither independence between constraints nor a stable identity
of the active null.

## Physical-twin specialization

For the registered bounded gain score \(G_t\in[-1,1]\), use

\[
F^{\mathrm{gain}}_{t,\lambda}
=
1+\lambda G_t,
\qquad 0<\lambda<1.
\]

Under insufficient conditional mean gain,
\(\mathbb E[G_t\mid\mathcal F_{t-1}]\le0\), this factor has conditional
expectation at most one.

For the harmful-update indicator \(H_t\in\{0,1\}\), harm ceiling \(\rho\), and
fixed alternative \(q<\rho\), use

\[
F^{\mathrm{harm}}_{t,q}
=
\left(\frac q\rho\right)^{H_t}
\left(\frac{1-q}{1-\rho}\right)^{1-H_t}.
\]

If the conditional harmful-update probability is at least \(\rho\), the
conditional expectation of this factor is at most one. The version-4 factor is

\[
L_{t,\lambda,q}
=
\min\!\left(
F^{\mathrm{gain}}_{t,\lambda},
F^{\mathrm{harm}}_{t,q}
\right).
\]

The implementation mixes over the Cartesian product of a frozen gain-bet grid
and a frozen harm-alternative grid.

## Relation to version 3

For binary harm, the version-3 linear harm score satisfies

\[
1+\lambda S_t
=
F^{\mathrm{harm}}_{t,q(\lambda)}
\]

for the corresponding Bernoulli alternative \(q(\lambda)\). Therefore

\[
1+\lambda\min(G_t,S_t)
=
\min(1+\lambda G_t,1+\lambda S_t).
\]

Version 3 is thus a diagonal restriction of the factor-envelope construction:
the same \(\lambda\) is used for utility and harm. Version 4 permits
\(\lambda_{\mathrm{gain}}\) and \(q_{\mathrm{harm}}\) to vary independently
while preserving the same switching-union theorem.

This separates two ideas:

1. **validity**, supplied by the lower-envelope domination argument; and
2. **power adaptation**, supplied by an outcome-independent mixture over
   independently tuned component factors.

## Extension beyond two constraints

The theorem applies without change to additional registered constraints, for
example:

- support or out-of-distribution violations;
- calibration failure;
- latency or compute-budget overruns;
- structural-consistency failures; and
- action-specific risk limits.

Each constraint needs a nonnegative component e-factor valid under its own
conditional null. The update factor is their pointwise minimum. This yields a
single anytime-valid certificate for a conjunction of requirements even when
the active failure mode changes over time.

## Confirmation design

The retained protocol is
`protocols/anytime_factor_envelope_v4.json`.

Scientific provenance is explicit:

- the version-3 results and scenario families were already observed;
- a disjoint pilot roster with seed base `2026090400` was used only to choose
  the version-4 gates;
- the retained confirmation roster uses seed base `2026091400`;
- the confirmation roster, thresholds, and gates were frozen before that
  roster was opened; and
- no real outcomes are used.

The comparison uses identical simulated outcomes for the version-3
minimum-score process and the version-4 factor envelope. The registered gates
require null control, control under switching invalidity, at least 0.78
moderate-case power, at least five percentage points of moderate-case power
gain, no more than a 15% increase in method-specific median crossing time, and
at least 0.99 strong-case power.

## Sealed confirmation result

The content-addressed result is
`results/science/anytime_factor_envelope_v4/result.json`; its protocol digest is
`8d7a406a944e0a5c38cdd2b4670a10a9f6424b0aad0e6b775194e3901492b026`.
All preregistered gates passed.

| Quantity | Registered result |
|---|---:|
| Maximum factor-envelope null Wilson upper bound | **0.010166** |
| Switching-invalidity crossing probability | **0.0000** |
| Moderate v3 minimum-score power | **0.7430** |
| Moderate v4 factor-envelope power | **0.8334** |
| Moderate power gain | **+0.0904** |
| Moderate median crossing, v3 | **252** |
| Moderate median crossing, v4 | **280** |
| Median crossing ratio, v4/v3 | **1.1111** |
| Strong v4 power | **1.0000** |

Thus the independent factor grid recovers 9.04 percentage points of
moderate-case power while retaining pointwise switching-null control. The gain
is not obtained by hiding a materially slower test: the registered median-time
ratio remains below the frozen 1.15 ceiling.

The evidence workflow additionally verified that every registered null phase
has maximum expected lower-envelope factor no larger than one, that the result
is bound to the frozen protocol and seed roster, that CSV artifacts use
canonical LF serialization, and that every retained artifact matches
`SHA256SUMS`.

## Integrated admission lifecycle

The theorem is not left as a stand-alone numerical primitive. The deployable
implementation is
`src/bayesian_phystwin/anytime_factor_envelope_controller_v4.py`, containing:

- `FactorEnvelopeAdmissionContractV4`;
- `FactorEnvelopeAdmissionConfigV4`;
- `FactorEnvelopePendingTrialV4`;
- `FactorEnvelopeResolvedTrialV4`;
- `FactorEnvelopeAdmissionSnapshotV4`; and
- `FactorEnvelopeAdmissionControllerV4`.

The controller adds the operational invariants required for a defensible
physical-twin admission claim:

- the candidate, exact fallback, score, harm definition, information set,
  reveal policy, factor family, and parameter grids are hashed into one
  decision-contract identity;
- trials are registered before their paired outcomes mature;
- a malformed reveal does not consume the pending trial and can be corrected
  exactly once;
- delayed outcomes from a closed epoch remain auditable but cannot update the
  new epoch;
- geometric alpha spending bounds an unbounded sequence of externally declared
  epochs;
- admission requires both the minimum resolved-trial count and an anytime
  boundary crossing; and
- selection returns the exact caller-owned fallback object unless the current
  epoch is authorized.

The integrated validation passed 55 focused tests spanning versions 1--4,
contract hashing, delayed-outcome order, reveal atomicity, switching invalidity,
exact object identity, and stable-suite registration.

## Claim boundary

The confirmation supports the lower-envelope composition mechanism and the
usefulness of independent factor tuning under the registered controlled
distributions. It does not establish fresh real-world validity, physical
safety, causal identification, universal power, or validity after
outcome-dependent redesign of the factor grid, candidate, fallback, score, or
reveal policy.

## Paper-facing contribution

A defensible paper statement is:

> We introduce a lower-envelope composition rule for anytime-valid
> multi-constraint admission. Under a pointwise union of conditional nulls, the
> minimum of independently tuned component e-factors is dominated by whichever
> component is currently valid. Products and fixed mixtures therefore remain
> e-processes even when the active failure mode changes over time. Applied to
> physical twins, this yields a single continuously monitored certificate for
> utility and harmful-update rate without requiring a stable invalidity mode or
> a common scalar betting parameter. In a frozen controlled confirmation,
> independent tuning increased moderate-case admission power from 74.30% to
> 83.34% while the registered null-control gates remained satisfied.
