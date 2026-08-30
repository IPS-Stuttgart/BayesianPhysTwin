# DLO-Lab Slingshot certified-guard replication v2

## Purpose

This is a prospective public-simulator replication of an exact-fallback Bayesian
decision guard on a second deformable-object task. It asks whether the fixed
mean-regret guard previously developed on DLO-Lab Slingshot can retain positive
decision value while providing a finite-sample upper bound on the probability
of harming a fresh world relative to the unchanged incumbent action.

The study is intended to complement the separately frozen DLO-Lab Wrapping v9
certificate. It is not an official benchmark or SOTA experiment, does not use
recorded physical data, and makes no real-robot safety claim.

## Honest development boundary

The parent belief/control study failed its registered primary joint-guard gate.
Its predeclared `mean_regret_guard` control nevertheless improved mean native
reward by `0.0022203773260116577` on 32 opened worlds, made three nonfallback
decisions, and caused zero harms beyond the `0.002` numerical margin. Those
opened outcomes are used to select the candidate for this new study and are not
counted as independent evidence.

The candidate is frozen without alteration:

- the exact 27-particle model bank;
- the exact 19-world split-conformal mean-regret offset
  `0.7285524030751176`;
- the exact source-selected action bank and incumbent action 5;
- the original shared-bias likelihood and metric noise model;
- strict nonworsening admission and byte-identical fallback.

## Fresh panel and estimand

The new panel contains 288 deterministic draws from the parent continuous
world distribution:

- `x_offset_m ~ Uniform[-0.02, 0.02]`;
- `bending_E ~ LogUniform[50000, 200000]`;
- `stretching_K ~ LogUniform[400000, 1600000]`.

The roster is checked to be disjoint from all registered Slingshot source,
particle, calibration, evaluation, and active-Bayes development rosters. Each
world receives 4096 independently generated sensor draws with a 5 mm shared xyz
bias and 2 mm independent point noise. Native rewards for all seven actions are
generated once per world. The statistical unit is one fresh world after
averaging the policy's reward over its registered sensor draws.

A harm is a world-level mean reward more than `0.002` below the incumbent. The
reported risk certificate is the exact one-sided 95% Clopper-Pearson upper
confidence bound over the 288 fresh worlds.

## Information boundary

The order is fixed:

1. run and qualify all 36 causal-prefix batches;
2. generate all noisy observations and seal all policy decisions;
3. require the registered pre-future support gate;
4. generate all 288 all-action futures;
5. score once.

No future native reward may exist before the complete decision barrier. The
output root is write-once, retries and replacements are forbidden, and any
technical failure remains in the denominator and terminates the study.

## Registered gates

Before futures, the guard must update at least 1% of all sensor decisions,
update at least 32 worlds, differ from the unguarded posterior mean on at least
1% of decisions, use at least two actions, and pass every native prefix check.

The final gate requires all 288 native futures and QA checks, at least two oracle
actions, guard gain of at least `0.001`, a paired bootstrap 95% lower bound above
zero, and a one-sided 95% harm-risk upper bound no greater than `0.05`. It also
requires at least ten harmed worlds from the unguarded posterior mean, at least
five harms removed by the guard, at least 75% mean-downside reduction, at least
10% retention of the posterior mean's gain, and at least 5% of oracle headroom.

Passing would support a scoped cross-task claim: on two public deformable-object
simulators, a baseline-relative Bayesian policy can expose useful decisions
while retaining an exact incumbent fallback and an empirical finite-sample harm
certificate. It would not establish distribution-free real-world safety.
