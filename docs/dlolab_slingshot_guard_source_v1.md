# Slingshot exact-fallback guard source study v1

## Question

The fresh-world active-Bayes v2 study showed that posterior integration materially
improves over plug-in MAP, but active Bayes still lost to the unchanged blind prior.
This source-development study asks whether the Bayesian update can be admitted only
where its own posterior reward distribution supplies robust evidence of benefit.

This is not a retry or reinterpretation of v2. It uses the 19 already-open parent
calibration worlds, a new active-prefix observation episode, and the unchanged public
DLO-Lab Slingshot simulator. A fresh-world study is not authorized automatically.

## Frozen method

For each noisy active-prefix observation, the 27-particle posterior supplies a reward
difference between the posterior-Bayes action and the blind-prior action. Candidate
guards use

```text
posterior mean gain - lambda * posterior gain standard deviation >= margin
```

with `lambda in {0, 0.5, 1, 2}` and
`margin in {0, 0.001, 0.002}`. A thirteenth candidate is the byte-exact blind-prior
fallback. The candidate bank, sensor draws, and every candidate decision are sealed
before the existing parent calibration reward artifact is parsed.

Candidate selection is leave-one-world-out. On each 18-world fit set a non-fallback
candidate must:

- update at least 10% of sensor decisions and at least 20% of worlds;
- retain a delete-one minimum mean gain of at least `0.00025`;
- harm no more than 20% of fit worlds beyond the `0.002` numerical margin.

The admissible candidate with the largest delete-one minimum gain is selected, with
predeclared tie-breakers. If no candidate is admissible, that fold uses the exact
blind-prior action. The held-out world is then scored without changing the choice.

## Information boundary

The only new simulator work is three 300-step active-prefix batches. They contain no
task future and no reward. All 19 x 8 noisy observations, posterior fields, active MAP
and Bayes actions, and 13 candidate decisions must seal and pass the pre-outcome gate
before the runner may read the hash-bound, already-open parent `calibrator.json`.

The parent calibration futures are not rerun. Protected targets, held-v8, DLO4/DLO5,
official DLO3 evaluation, GPUs, robots, and new recordings remain outside scope.

## Advancement gate

A fresh-world guarded study is justified only if the cross-fitted source result has:

- at least 16 non-fallback sensor decisions across at least four worlds;
- at least `0.001` mean gain over blind with a positive paired 95% world-bootstrap
  lower bound;
- at least `0.001` mean gain over unguarded active Bayes;
- fewer harmed worlds than unguarded active Bayes and no more than three total;
- a non-fallback candidate selected on all 19 source worlds.

The source result remains source-development evidence even if every gate passes.
It carries no SOTA, official benchmark, real-world, perception, material-identification,
or physical-safety claim.
