# Query-conditional trust certificate v1

## The question

A robot does not need a simulator that is globally accurate. It needs a
fail-closed answer to a narrower question:

> Is this simulator trustworthy enough to use for this action, horizon, and
> physical query, and is the available physical belief precise enough to make
> the resulting action choice?

Those are two different statements. A locally accurate prediction may still be
decision-insufficient when unresolved physical hypotheses prefer different
actions. Conversely, a robust action may be identifiable even when the full
state and some physical parameters remain ambiguous.

`bayesian_phystwin.query_conditional_trust_v1` composes two independently
registered certificates and returns the caller-owned fallback plan exactly when
either one fails.

## Certificate A: local simulator competence

For context

```text
x = (object, action, horizon, query),
```

the simulator-competence certificate requires a source-qualified backend,
fixed runtime and interfaces, a pre-outcome risk score, a frozen selective
threshold, and a finite-group bound on harmful accepted predictions. Its
decision is

```text
C_sim(x) = 1[r(x) <= tau and every registered scope check passes].
```

This certificate asks whether the candidate simulator may replace the exact
prediction fallback for the registered query. It does not establish that the
latent physical state is identified or that an action choice is robust.

## Certificate B: decision sufficiency

Let `p_i > 0` define supported physical hypotheses, `lambda_c` be the observed
posterior mass of registered query-equivalence class `c`, and `L(i,a)` be the
registered loss of action `a` under hypothesis `i`. For any action pair `a,b`,
the exact worst-case loss gap over all complete beliefs compatible with the
observed quotient posterior is

```text
Delta(a,b)
  = sum_c lambda_c max_{i in c: p_i > 0} (L(i,a) - L(i,b)).
```

The exact worst-case regret is

```text
Reg(a) = max_b Delta(a,b).
```

The decision certificate admits the candidate action only when
`Reg(a) <= epsilon`, where `epsilon` is registered before target outcomes. It
does not choose one convenient within-class physical explanation.

## Composed rule

The plan is authorized iff both statements hold:

```text
C_trust(x,a) = C_sim(x) and 1[Reg(a) <= epsilon].
```

The implementation additionally binds the exact backend/runtime, object and
action domains, query functional, loss, hypothesis support, quotient
registration and posterior, finite action set, loss model, candidate action,
and candidate/fallback plan identities. Cross-query, cross-action, or
cross-loss substitution fails closed.

If the conjunction is false, the returned object is the same fallback plan
object supplied by the caller. No reconstructed or approximately equivalent
fallback is permitted.

## Why this is a stronger contribution

This reframes simulator validation as a decision interface rather than a global
leaderboard:

1. **Prediction validity is local.** Competence is conditioned on the exact
   action, horizon, and queried physical functional.
2. **State identification is not required.** The exact regret certificate
   considers every supported complete belief compatible with the observed
   quotient posterior.
3. **Decision validity is separate from prediction validity.** Either
   certificate can veto simulator-backed action selection.
4. **Abstention is operational.** Rejection preserves the incumbent plan
   exactly, making non-regression testable at the software boundary.

## Evidence boundary

The prospective DLO-Lab Slingshot reward-aligned study already supports the
first half of this story: on 288 fresh evaluation worlds, a frozen local
policy-gain certificate updated 36 worlds, improved mean reward by `0.003457`
with paired 95% interval `[0.001514, 0.005711]`, and kept the one-sided harmful
world upper bound at `0.04070`, below the registered `0.05` budget. It also beat
an equal-data simultaneous-regret guard by `0.004338`, with paired interval
`[0.001935, 0.006973]`.

That closed result must not be retrospectively relabeled as a validation of the
new exact-regret composition. The composition is presently an exact software
and mathematical contract with adversarial tests. A claim that both
certificates jointly improve a policy requires a separately frozen, fresh-world
experiment.

The immutable Slingshot result identity is
`2882809b7265714a93be2d3f1455eeac527adbe681cc990cde762777fcaf3a85`.
Its compact summary has SHA-256
`cfbab2f371ec606fdbcf844cc8484f543a57829780f893f1b9bf3359dbae2564`;
the full 1.60 GB tree contains 13,947 files and is bound by canonical tree
SHA-256
`1172e89dc795952ed1358c86974569ea5b8744bbe0d1810b66470803b587ed85`.
On 2026-09-01 the registered verifier rehashed that tree, revalidated all 3,328
ordinary action seals and 416 world qualifications, reconstructed the sealed
decisions, and recomputed the complete score successfully. The evidence remains
on the immutable `experiment/dlolab-slingshot-independent-native-v3` line; this
document does not rewrite it.

## Claim boundary

Passing the composed certificate does not establish universal simulator
validity, identify a unique physical cause, validate the observation provider
or loss model, prove distribution shift robustness, or certify real-robot
safety. It certifies only the registered finite decision problem and local
simulator context. Evidence from public simulators is mechanism evidence, not a
physical deployment guarantee or an official state-of-the-art claim.
