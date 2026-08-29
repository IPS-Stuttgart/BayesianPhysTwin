# Slingshot Active Bayesian Identification v2 Result

## Decision

The sole v2 attempt completed all 32 fresh continuous worlds with zero technical
failures, but the frozen source gate failed. No method is promoted and no retry
or threshold change is authorized.

## Primary Result

The active posterior-integrated decision beats the active plug-in MAP decision:

| Arm | Mean native reward | Gain over blind | Worlds harmed beyond 0.002 |
|---|---:|---:|---:|
| Blind prior | 7.010289 | 0.000000 | 0 |
| Passive MAP | 7.007607 | -0.002681 | 15 |
| Passive Bayes | **7.010782** | +0.000494 | 1 |
| Active MAP | 6.997652 | -0.012636 | 20 |
| Active Bayes | 7.006907 | -0.003381 | 15 |

Active Bayes improves on active MAP by `0.009255`, with paired world-bootstrap
95% CI `[0.003948, 0.015259]`. It removes 73.24% of active MAP's mean loss
relative to blind and harms five fewer worlds. This is positive controlled
source evidence for posterior integration over plug-in MAP under this finite
particle approximation.

It is not evidence that the active identification maneuver has task value.
Active Bayes remains `0.003381` below blind (95% CI
`[-0.007914, 0.001220]`) and `0.003875` below passive Bayes (95% CI
`[-0.008347, 0.000720]`). Passive Bayes is the best tested arm, but its small
gain over blind is not significant under the frozen interval. The primary v2
hypothesis and advancement gate therefore fail.

## Gate Accounting

The prefix-only gate passed before any task future existed: all eight native
prefix batches qualified; active Bayes differed from blind on 95 of 256 sensor
decisions, differed from active MAP on 63, and selected three distinct actions.

The complete source run passes native QA, the 32-world denominator, six-action
oracle diversity, active-Bayes improvement and positive interval versus active
MAP, and the harm-count comparison. It fails gain and interval checks versus
blind and passive Bayes, the absolute gain threshold, and the oracle-headroom
fraction. Its headroom fraction is negative because active Bayes underperforms
blind.

## Interpretation And Boundaries

The fresh-world result refines the earlier finite-particle observation:
posterior integration is materially safer than committing to one MAP material
particle, but the 70% frontloaded probe drives both active policies away from a
strong prior action often enough to erase task value. This could reflect probe
likelihood mismatch, finite-bank approximation, shared sensor bias, or simply
insufficient value of information for this task. The frozen experiment does not
distinguish those explanations and does not authorize tuning another probe on
these worlds.

This is public-simulator source evidence only. It is not an official benchmark,
SOTA, real-robot, perception, calibrated-UQ, material-identification, or safety
claim. It uses no protected data, held-v8, DLO4/DLO5, GPU, or new recording and
does not alter any successful DEFORM result.
