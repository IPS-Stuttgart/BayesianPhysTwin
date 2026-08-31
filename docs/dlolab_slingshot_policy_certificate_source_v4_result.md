# DLO-Lab Slingshot reward-aligned policy certificate v4 result

## Decision

**The prospective source gate passed on the complete fresh denominator.** A
query-local Bayesian policy-gain certificate improved native Slingshot reward
while satisfying its registered coverage, harm, transfer, and matched-comparator
gates. This is the first positive Slingshot policy-value result in the project;
v1-v3 remain immutable negative evidence for their different estimands.

## Prospective evidence

- Frozen source revision: `e446285660991e4f2b83422c64788f4c410e0c97`.
- Public simulator only: 128 fresh calibration worlds and 288 further fresh
  evaluation worlds; no recordings or protected datasets.
- Native execution: 3,328/3,328 one-world/one-action fresh processes ordinary,
  zero failures, retries, or replacements.
- Information boundary: 36 accepted and 252 fallback evaluation decisions were
  sealed before any evaluation future was simulated or read.
- Policy-gain guard: 36/288 worlds updated; mean reward gain `0.0034568` with
  paired bootstrap 95% interval `[0.0015144, 0.0057113]`.
- Harm: 6 harmful guarded worlds versus 69 for the unguarded posterior mean;
  the guard's one-sided 95% harm-probability upper bound is `0.04070`, below the
  registered `0.05` budget.
- Calibration: marginal selected-policy gain coverage `0.8958`; simultaneous
  action coverage `0.8924`.
- Matched comparator: the simultaneous mean-regret guard changed 30 worlds and
  had mean gain `-0.0008813`. The policy-gain guard beat it by `0.0043380`, with
  paired 95% interval `[0.0019350, 0.0069734]`.
- The selective guard retains `24.48%` of unguarded posterior gain, captures
  `13.80%` of oracle headroom, and removes 63 of the posterior policy's 69
  harmful worlds.

The exact result is
`4c4cea8632a7fbc00eafc909519d6decb5b63ff270a81778ab64192a3bbae942`.
The compact evidence is
`results/source/dlolab_slingshot_policy_certificate_source_v4/summary.json`;
the complete 1.60 GB raw tree is hash-bound and independently replayable.

## Why the reward-aligned estimand matters

V3 stopped because one duplicate native trajectory differed by 0.864 mm even
though its decision reward differed by less than 0.001. V4 prospectively treats
that late state bifurcation as simulator-process variability while retaining
reward repeatability as a hard admission gate. It does not relax or rescore v3.

The new panel again observed one calibration world above the old 0.5 mm state
tolerance (`0.537 mm`), while all 416 worlds passed reward repeatability and the
maximum evaluation duplicate reward difference was `0.0001125`. This supports
the revised statistical unit: expected policy value over fresh world, sensor,
and native-process draws. The incumbent reward averages its two independently
executed duplicate slots, so a single baseline process draw cannot determine
the comparison.

## Contribution

The result is not another point-prediction benchmark. It is prospective evidence
that posterior uncertainty can create **decision value**: a frozen local
certificate identifies a small subset of queries where changing the action has
positive expected value, controls the upper confidence bound on harm, and beats
an equal-data simultaneous-action uncertainty guard. Exact fallback handles the
other 87.5% of worlds.

This strengthens the public-data paper around a query-conditional simulator
competence certificate: uncertainty is useful when it decides whether the
simulator-backed policy should be trusted, not merely when it draws wider error
bars around every forecast.

## Claim boundary

This is a complete prospective result for the registered DLO-Lab Slingshot
public-simulator distribution. It is not an official benchmark or SOTA claim,
does not establish physical-robot safety or real-world transfer, and does not
authorize tuning on these 416 worlds. Held-v8, DLO4, DLO5, protected targets,
and new recordings were not accessed. The roster, thresholds, and one-attempt
execution are closed.
