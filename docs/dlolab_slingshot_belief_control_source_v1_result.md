# Native Slingshot Belief-Control: Frozen Negative Result

The primary source gate **FAILED**. All 19 calibration worlds and all 32 new
continuous evaluation worlds completed native QA; there were no technical
failures or replacements. Every evaluation decision preceded every evaluation
future. The primary returned the byte-identical incumbent in 32/32 worlds.
No controller is promoted and no protected target is authorized.

| Arm | Mean reward | Mean gain | Paired 95% gain interval | Updates | Harm > 0.002 |
|---|---:|---:|---|---:|---:|
| Incumbent | 6.967336 | 0 | [0, 0] | 0/32 | 0/32 |
| Nominal point | 6.967730 | 0.000395 | [-0.003536, 0.004029] | 32/32 | 6/32 |
| Prior predictive mean | 6.976812 | 0.009477 | [0.001377, 0.017185] | 32/32 | 9/32 |
| MAP point | 6.971409 | 0.004074 | [-0.010431, 0.015132] | 32/32 | 8/32 |
| Posterior predictive mean | 6.975834 | 0.008498 | [0.000661, 0.016121] | 32/32 | 9/32 |
| Posterior ignoring shared bias | 6.953714 | -0.013622 | [-0.039714, 0.008749] | 32/32 | 10/32 |
| Calibrated mean guard | 6.969556 | 0.002220 | [0, 0.004937] | 3/32 | 0/32 |
| Calibrated independent guard | 6.967336 | 0 | [0, 0] | 0/32 | 0/32 |
| Primary calibrated joint guard | 6.967336 | 0 | [0, 0] | 0/32 | 0/32 |

Intervals use the frozen 10,000 world-level paired bootstrap. The 95% harm
upper bound for zero harms among all 32 decisions is 8.94%, not a conditional
bound among accepted updates. Joint action-bound coverage was 100%, independent
96.875%, mean 90.625%. Full coverage through total abstention does not establish
useful uncertainty-aware control. The primary fails update, gain, and matched
control comparisons. The prior mean outperforms the posterior mean, so an
improvement over the weaker original incumbent does not establish Bayesian value.

## Interpretation

The finite model bank has weak material identifiability in the permitted
prefix and poor continuous-world reward prediction. Descriptive posterior
reward RMSE is 0.125421 and action-regret RMSE 0.112471, much larger than the
0.020579 average oracle gain over the incumbent. Joint coupling reduces regret
variance relative to independent coupling in 84.375% of candidate/world pairs,
but this does not yield accepted decisions after calibration. In 28 worlds an
alternative improves reward by more than 0.002; the primary misses all of them.

This closes this finite-bank, observation-budget, action-bank study. It does
not reject Bayesian control generally, justify lowering the frozen thresholds,
or establish published Slingshot parity. Known 3D identities with synthetic
noise are assumed. Native parameters are simulator parameters, not identified
material constants. The 64-candidate backbone is not the published CMA-ES
training budget. No new recording or robot execution was used.

## Provenance And Verification

- Frozen implementation: `8ed240a060ccbbbe1a30271db738baebad522e96`.
- Lock ID: `015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25`.
- Result ID: `9b8ff0817744392e0584c9b59936dd1b0e9331d3b0fa2d021f5a361947d32ee9`.
- Result file SHA-256: `1df6afe4832a9c35bc65543255f5ce2c5830e6d58cfaa23d1140f8c867767e0b`.
- 143 relevant pre-run tests passed; changed-file Ruff, focused MyPy and diff
  checks passed. The separate archive verifier rehashed every input/array and
  recomputed the posterior, decisions, calibration, native QA, metrics and gate.
  This is second-implementation verification by us, not independent human review.

Compact evidence is in `results/source/dlolab_slingshot_belief_control_v1/`.
The complete native archive is retained locally at
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1`.
No existing DEFORM implementation, successful result, or frozen failure changed.
This record remains on an isolated local branch; nothing is pushed or merged.
