# Prospective two-query portfolio replication v1

## Purpose

This experiment prospectively tests whether the same uncertainty-gated design
has decision value on two distinct public deformable-simulator queries. It
removes two limitations of the existing portfolio synthesis: the portfolio is
declared before its fresh outcomes exist, and positive value plus bounded harm
are controlled jointly at 95% rather than reported as separate 95% families.

## Frozen design

- Queries: DLO-Lab Wrapping with the frozen v9 guard and DLO-Lab Slingshot with
  the frozen reward-aligned v4 guard.
- Evaluation: 320 fresh worlds per query; no world replacement.
- Mean: the existing frozen query policy. No refitting or threshold changes.
- Fallback: the exact registered incumbent action.
- Harm: reward gain below `-0.002` in one world.
- Overall error budget: 0.05.
- Gain family: 0.01, Bonferroni 0.005 per query.
- Harm family: 0.04, Bonferroni 0.02 per query.
- Gain bound: 100,000-replicate world bootstrap lower percentile.
- Harm bound: one-sided Clopper-Pearson upper bound.
- Promotion: both gain lower bounds exceed zero and both harm upper bounds are
  at most 0.05.

The reward scales are task-specific and are never pooled across Wrapping and
Slingshot. The joint statement is an intersection of four registered claims,
not an average reward claim.

## Feasibility evidence, not target evidence

Replaying only the already frozen 288-world source outcomes gives 99.5%
one-sided bootstrap gain lower bounds of approximately `0.00366` for Wrapping
and `0.00099` for Slingshot. At 320 worlds, the adjusted harm test permits at
most eight harmful worlds. These calculations selected the sample size before
fresh-world generation; they are not part of the prospective result.

## Claim boundary

A pass supports a prospective public-simulator claim: Bayesian uncertainty can
certify query-specific simulator competence with exact fallback across two
different decision tasks while jointly controlling positive-value and harm
statements. It is not evidence of physical-robot safety, universal backend
competence, or point-prediction state of the art.
