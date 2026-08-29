# DLO-Lab wrapping chance-guard replication v8 result

## Decision

The complete 144-world public-simulator replication is a **strict source-gate
failure with positive prospective risk-value evidence**. The registered 0.975
posterior chance guard improves mean native reward over the fixed action by
`0.005042` (paired 95% CI `[0.003726, 0.006409]`) and reduces the number of
worlds harmed beyond the numerical margin from 10 under unguarded continuous
Bayes to 2. Mean downside below the fixed action falls by 94.29%.

The result must not be reclassified as a gate pass. Four locked requirements
fail: zero harmed worlds, retention of at least half the continuous-Bayes gain,
loss to continuous Bayes no greater than `0.012`, and capture of at least 20%
of oracle headroom. No threshold was changed after outcomes were generated.

## Registered execution

- Frozen source revision: `5c6a163d664114126bacba5865f8b1597cf8684b`
- Public native simulator: DLO-Lab wrapping through the v7-qualified native
  Linux CPU/OSMesa runtime
- Fresh worlds: 144, disjoint from the source bank and all v1-v4 study worlds
- Prefix batches: 16/16 ordinary successes
- All-action futures: 144/144 ordinary successes
- Technical failures: 0
- Replacements or retries: 0
- New recordings, protected data, held-v8, DLO4/DLO5, or official DLO3: none

The pre-future gate passed before any future was simulated. The primary guard
used three actions, made 164,521 nonfixed sensor decisions, and differed from
continuous Bayes on 350,839 decisions. The minimum posterior improvement
probability among its nonfixed decisions was `0.975000224`.

## Results

| Arm | Mean reward | Gain over fixed | Harmed worlds | Mean downside |
| --- | ---: | ---: | ---: | ---: |
| Fixed action | 0.886734 | 0.000000 | 0 | 0.000000 |
| Continuous Bayes | 0.907778 | 0.021044 | 10 | 0.005538 |
| 0.90 chance guard | 0.895237 | 0.008503 | 5 | 0.001507 |
| **0.975 chance guard** | **0.891776** | **0.005042** | **2** | **0.000316** |
| 0.99 chance guard | 0.890453 | 0.003719 | 1 | 0.000047 |
| Finite-particle Bayes | 0.911209 | 0.024475 | 13 | 0.010077 |
| Continuous MAP | 0.906431 | 0.019697 | 31 | 0.011501 |

The primary guard loses `0.016002` mean reward to unguarded continuous Bayes
(95% CI `[-0.022170, -0.009103]`). It retains 23.96% of the continuous-Bayes
gain and captures 9.47% of oracle headroom. In return, it removes 8 of the 10
continuous-Bayes harms and lowers mean downside from `0.005538` to `0.000316`.
The paired downside-reduction CI is `[0.000953, 0.010779]`.

The registered threshold sensitivity is coherent but descriptive: relaxing the
guard to 0.90 raises gain and harms; tightening it to 0.99 lowers both. These
arms were frozen before the run and are not post-outcome alternatives to the
primary arm.

## Gate accounting

The following locked checks pass:

- complete 144-world denominator and all native QA;
- at least two oracle actions;
- gain over fixed at least `0.005`;
- positive paired 95% CI versus fixed;
- continuous Bayes harms at least two worlds;
- at least two harms removed by the guard;
- at least 50% mean-downside reduction.

The following locked checks fail:

- zero guard-harmed worlds: 2 observed;
- retain at least 50% of continuous-Bayes gain: 23.96% observed;
- mean loss to continuous Bayes at most `0.012`: `0.016002` observed;
- capture at least 20% of oracle headroom: 9.47% observed.

## Verification

The read-only verifier enumerates the exact 647-file, 4,936,658,734-byte result
tree. It validates every write-once record and native bundle, reconstructs all
prefix observations from the frozen source bank, regenerates the complete
decision bundle, re-derives the pre-future barrier, checks all 144 native
futures and prefix parity, reconstructs the reward matrix, and recomputes the
score and paired bootstrap independently. Verification passed; it is an
arithmetic/custody check, not an independent human review.

Key identities:

- attempt: `8f20c4c8ca03d56daed26519d31b34f7409b939ce02edcbd31c7476a9acf67a5`
- lock: `1a99a36263f2613ba3af61ccb382a678f0a0df52997c0af667a2c8e0ea7e04e2`
- decision: `25a7ce96aa82bb475df753b3cf4d89b85dc2e665e1eceae0ea47304fe4e86301`
- barrier: `3740dcc207310b7c3696df03c283fefa34daf21f61505036e293fc34d90b599e`
- generation: `2dc04f43d60ec1f44660f9242ac32b1a6c6fa90c6c87c5cb2e0c81508af9fc8f`
- result: `61c97165b9277790ac0e4e5374659fb9eab598dc0ff96ef3503859d6198ae5b2`
- verified tree manifest: `21ad861aa73e8c76bcbdc196258e22c40c3ea925e25d77dd4dc39219130215df`

## Claim boundary

This is prospective evidence in a public simulator that baseline-relative
Bayesian chance constraints expose a reproducible reward/downside tradeoff. It
does not establish zero regret, physical-robot safety, official benchmark
superiority, point-prediction SOTA, real-world calibration, or material
identification. The strict source gate remains failed, and these 144 worlds are
closed to further threshold or method selection.

