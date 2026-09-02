# Paper synthesis v5: familywise simulator competence certificates

## Recommended paper claim

> We introduce query-conditional simulator competence certificates: a Bayesian
> decision layer that grants a simulator authority only when it can certify
> positive value for a registered decision query, otherwise returning an
> incumbent policy exactly. In a prospective two-query public-simulator study,
> both certificates improve reward with simultaneous 95% confidence while
> controlling baseline-relative harm below 5%.

This is a decision-value and selective-reliance paper, not a trajectory-SOTA
paper. Its novelty is the unit being certified: a simulator is neither accepted
nor rejected globally. Competence is attached to a concrete distribution,
observation policy, action bank, reward, and fallback.

## Contributions

1. **Query-conditional competence.** Define the smallest deployable trust unit
   for a physical simulator and separate reward-relevant competence from global
   state-prediction accuracy.
2. **Exact-fallback Bayesian guard.** Use posterior uncertainty to modify an
   incumbent only when the registered query-specific value certificate passes;
   abstention is byte-for-byte the incumbent action.
3. **Finite-portfolio guarantee.** Give a familywise construction that controls
   simultaneous positive-value and harm statements over a preregistered query
   family without assuming independence across queries.
4. **Prospective public evidence.** Evaluate 640 fresh worlds across Wrapping
   and Slingshot, plus 128 separate Slingshot calibration worlds, with complete
   denominators, no technical replacement, and no cross-task reward pooling.

## Main result

| Query | Mean gain | Adjusted gain lower | Harm | Adjusted harm upper | Decision |
|---|---:|---:|---:|---:|---|
| Wrapping | +0.00590771 | +0.00480501 | 1/320 | 0.01809391 | pass |
| Slingshot | +0.00633863 | +0.00382284 | 1/320 | 0.01809391 | pass |

The joint portfolio gate passes at simultaneous confidence at least `0.95`.
The guarded policies deploy on 335/640 worlds and exactly fall back on 305/640.

The matched permissive controls explain why uncertainty matters. In Slingshot,
always deploying the posterior action gives larger average gain but harms
67/320 worlds. In Wrapping, the continuous Bayesian policy harms 16/320 worlds.
The registered guards reduce this to one harmful world in each query while
retaining positive, familywise-certified reward gain. Thus the result is not
merely that a simulator can choose useful actions; it is that posterior
uncertainty determines when those actions should receive decision authority.

## Suggested narrative

1. Modern deformable simulators are heterogeneous: the same backend may rank
   actions correctly for one query and fail badly for another.
2. Global validation scores therefore answer the wrong deployment question.
   A controller needs to know whether this simulator is useful for this action
   choice under this observation and loss.
3. Bayesian posterior uncertainty supplies a local competence statistic, but a
   single successful query does not justify broad trust.
4. Register a finite portfolio and allocate confidence across both value and
   harm families before evaluation.
5. Show prospectively that both distinct queries pass, while matched unguarded
   Bayesian policies expose substantial downside.
6. Conclude that calibrated selective authority, rather than universal simulator
   fidelity, is a viable route to dependable simulator-assisted decisions.

## Reviewer-facing distinction

This is stronger than selective prediction on state error. The guard is trained
and tested on downstream policy value, compares against an unchanged incumbent,
and certifies both a positive expected gain and a bounded probability of harmful
regret. It is also stronger than reporting two independent confidence intervals:
the query family and its error allocation were fixed prospectively, and the
claim is made only by the complete fail-closed joint assembler.

## Scope

The evidence is entirely public and simulated. It does not establish robot
safety or universal backend validity. The right next extension is a third,
mechanically distinct query under a new prospective family or an alpha-spending
sequence, followed by a robot study. Neither is required for the current public
simulator contribution, but both are explicit paths to broader validity.

The recommended title remains:

> **Do Not Trust a Simulator Everywhere: Familywise Bayesian Competence
> Certificates for Deformable Manipulation**

The authoritative numerical result and provenance are in
`docs/query_portfolio_replication_v5_result.md`.
