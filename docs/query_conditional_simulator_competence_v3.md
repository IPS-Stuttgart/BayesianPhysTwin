# Paper synthesis v3: certified competence is a query portfolio

## Central claim

**Bayesian uncertainty has measurable decision value when it certifies where a
deformable simulator may influence an exact query, while every uncertified
query retains the baseline exactly.**

The contribution is not a stronger global simulator label. It is a staged
competence atlas plus a simultaneous deployment certificate:

1. define a query by simulator, task, observation policy, action bank, metric,
   world distribution, and statistical unit;
2. require native validity, action headroom, source transfer, and prospective
   risk evidence in that order;
3. deploy a Bayesian update only for a fully certified exact query;
4. use the exact baseline for every rejected or unknown query;
5. correct over the complete family that reached final risk evaluation.

## Why this is larger than one guarded-controller result

The same public deformable simulator produces two certified queries and four
rejections. Even within Slingshot, v2 is rejected while the reward-aligned v4
query is certified. A backend-wide label cannot represent that evidence.

Across the complete six-query atlas, three policies reached final prospective
risk evaluation. A familywise 95% correction still certifies Wrapping and
reward-aligned Slingshot below a 5% harm budget, with adjusted harm upper bounds
of `0.020813` and `0.047069`. Their adjusted gain lower bounds remain positive
at `0.003830` and `0.001359`. The remaining four queries use exact fallback.

This converts isolated task results into a reusable runtime object: a finite
atlas in which a registered query either earns decision authority or cannot
change the incumbent.

## Main figure

The paper's central figure should show a six-row stage diagram. Rows terminate
at native validity, headroom/transfer, or prospective risk. The two passing
rows continue to a familywise certificate box; all four failing rows route to
the same exact-fallback node. A side panel should show 84 unguarded versus 7
guarded harmful worlds over the two deployed 288-world evaluations, labeled as
a descriptive equal-query aggregate.

## Positioning against the closest ideas

This is not the first work to learn where a model applies. Model-precondition
methods use predicted transition error to switch among dynamics models, and
Learn-Then-Test or conformal-risk-control methods provide finite-sample risk
control for selected predictors. The defensible novelty is their combination
at a different unit:

- competence is attached to an exact deformable-manipulation **decision query**,
  not to state-transition error or a simulator name;
- the certificate targets downstream reward and a baseline-relative harm event,
  not pointwise dynamics accuracy alone;
- qualification failures and statistical rejections share one exact fallback
  contract;
- the complete finite query family receives simultaneous multiplicity
  accounting; and
- two prospectively evaluated deformable tasks retain positive decision value
  under that accounting.

The closest conceptual references are LaGrassa and Kroemer's *Learning Model
Preconditions for Planning with Multiple Models* (2022), LaGrassa, Lee, and
Kroemer's *Task-Oriented Active Learning of Model Preconditions for Inaccurate
Dynamics Models* (2024), and the Learn-Then-Test / risk-controlling-prediction
line. The paper should cite these directly and avoid claiming invention of
model applicability or abstention in general.

## Evaluation story

- **Prospective components:** 272 calibration and 576 evaluation worlds on the
  public DLO-Lab simulator.
- **Decision value:** positive query-specific gain for both deployed tasks
  after multiplicity correction.
- **Downside:** familywise harm bounds below the registered 5% budget.
- **Selectivity:** two certified queries, one final-stage rejection, and three
  earlier-stage rejections, all retained.
- **Comparator:** unguarded Bayesian policies gain more on average but produce
  84 harmed worlds; guarded policies retain smaller positive gains with 7.
- **External mechanism evidence:** the public-real retrospective audit shows
  query-rank reversal and exact-context transfer failures, explaining why a
  global trust score is not defensible.

## Honest positioning

This is strongest as a decision-theoretic and evaluation contribution, not as
a point-prediction paper. The portfolio synthesis is post hoc even though its
component trials were prospective. Separate familywise 95% statements are
made for positive value and harm; their joint confidence lower bound is 90%.
No physical safety, official benchmark, backend-wide competence, or arbitrary
out-of-distribution guarantee is claimed.

A suitable paper title is:

> **Do Not Trust the Simulator Everywhere: Query-Conditional Bayesian
> Certificates for Deformable Manipulation**
