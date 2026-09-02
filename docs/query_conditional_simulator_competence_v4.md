# Paper synthesis v4: prospective query-portfolio certification

## Contribution

The deployable object is not a globally trusted simulator. It is a finite,
registered portfolio of decision queries. Each query fixes the simulator,
world distribution, observation policy, action bank, reward, statistical unit,
Bayesian decision rule, harm margin, and exact fallback before evaluation.

The central prospective claim is:

> Bayesian uncertainty has decision value across a finite portfolio of
> deformable-control queries when it is allowed to change the incumbent only
> where a query-specific certificate passes, with simultaneous control of
> positive-value and baseline-relative-harm claims.

This is stronger than the earlier single-task certificate and the post-hoc
atlas synthesis. The portfolio membership, fresh world seeds, 320-world
denominators, familywise allocation, and joint decision rule were frozen before
either portfolio outcome was opened.

## Certificate theorem

Let `Q` be the registered finite query set. For each query `q`, let `G_q` be
the world-level reward gain of the guarded policy over its exact fallback, and
let `H_q = 1{G_q < -epsilon}` denote harm beyond the registered numerical
margin. Let `L_q` be a one-sided lower confidence bound for `E[G_q]` with
miscoverage `alpha_gain,q`, and let `U_q` be a one-sided upper confidence bound
for `P(H_q=1)` with miscoverage `alpha_harm,q`.

If the evaluation worlds are exchangeable within each registered query and
the component bounds have their stated marginal coverage, then

```text
P(
  for every q in Q:
    E[G_q] >= L_q and P(H_q=1) <= U_q
)
>= 1 - sum_q(alpha_gain,q + alpha_harm,q).
```

This follows directly from the union bound and does not require independence
between queries, between the value and harm statistics, or between their
confidence procedures. Under the frozen allocation,

```text
sum_q alpha_gain,q = 0.01
sum_q alpha_harm,q = 0.04
```

so the complete simultaneous statement has confidence at least `0.95`. The
portfolio passes only when every registered query has `L_q > 0` and
`U_q <= 0.05`. Missing worlds, partial results, technical replacements, or a
failed query make the joint claim fail; they cannot silently reduce the family.

## Exact-fallback semantics

For each world, the guard either deploys its registered Bayesian action or
returns the incumbent action exactly. Thus abstention has zero policy regret
relative to the registered fallback by construction. The certificate does not
claim that the fallback is optimal or physically safe. It certifies only the
incremental decision authority granted to Bayesian uncertainty over that
fallback on the registered query distribution.

This distinction matters. A high-accuracy simulator can still be harmful for
a particular decision, while a globally imperfect simulator can be useful for
a query whose action ordering is locally reliable. The method therefore tests
reward-relevant competence instead of converting trajectory error into a
universal trust score.

## Prospective public-simulator study

The frozen replication contains two distinct DLO-Lab decision queries:

- Wrapping, using the registered posterior 97.5% guard and its existing source
  calibration certificate;
- Slingshot, using a fresh 128-world calibration partition and the registered
  policy-gain guard.

Each query has 320 fresh evaluation worlds. Rewards are never pooled across
tasks: all means, bootstrap bounds, harm counts, and gates are query-specific.
The only cross-query operation is the outcome-independent familywise error
allocation. Both complete component artifacts must reproduce from their sealed
world-level decisions and rewards before the joint assembler can run.

At the time this document is frozen, both empirical runs are still active and
no portfolio result is asserted. The result section must report either the full
two-query certificate or the exact terminal failure; a favorable component
cannot be promoted alone under this protocol.

## Main paper structure

1. Show why simulator-wide competence is the wrong unit using retained positive
   and negative backend/query evidence.
2. Define a decision query and the exact-fallback guarded policy.
3. State the finite-portfolio theorem and error allocation.
4. Present the prospective two-query study with complete failure accounting.
5. Report point prediction only as mechanism context, not as the main claim.
6. Discuss extension to new queries as a new preregistered family or as an
   alpha-spending sequence, never as outcome-selected portfolio expansion.

## Claim boundary

The study can establish simultaneous decision value on two registered public
simulator queries. It cannot establish backend-wide competence, physical robot
safety, arbitrary-distribution validity, point-prediction state of the art, or
automatic transfer to a new task. Those are deliberately outside the theorem.

A suitable title is:

> **Do Not Trust a Simulator Everywhere: Familywise Bayesian Competence
> Certificates for Deformable Manipulation**
