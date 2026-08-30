# DLO-Lab Slingshot certified-guard v2 result

## Decision

The complete 288-world prospective public-simulator replication **fails the
registered source gate**. The frozen mean-regret guard reduces the unguarded
posterior controller's mean downside by `92.65%` and reduces harmed worlds from
`62` to `14`, but it does not meet the preregistered finite-sample risk or value
requirements.

The guard's mean gain over the incumbent is only `0.000220` (paired 95%
bootstrap CI `[-0.000111, 0.000530]`). Its exact one-sided 95% Clopper-Pearson
upper bound on the registered world-level harm probability is `0.074952`, above
the frozen `0.05` budget. The controller is not promoted, and no threshold or
gate is retuned on these opened worlds.

## Registered execution

- Frozen source revision: `7da610e3c321f605be29682d1360357496693c7e`
- Public simulator: DLO-Lab Slingshot under the parent-qualified native Linux
  CPU/OSMesa runtime
- Fresh continuous worlds: 288, disjoint from every registered Slingshot source
  and development roster used to select the guard
- Prefix batches: 36/36 ordinary successes
- All-action futures: 288/288 ordinary successes
- Technical failures: 0
- Replacements or task retries: 0
- Sensor draws per world: 4096
- New recordings, protected data, held-v8, DLO4/DLO5, or official DLO3: none

The original desktop PTY disappeared after 24 ordinary prefix seals. A
write-once continuation receipt revalidated those seals, proved that prefix
tasks 24-35 and every future were unattempted, and resumed only the unattempted
tasks. No task was retried. The pre-future barrier then passed before any future
simulation, with 60,132 nonfallback sensor decisions across 88 worlds.

## Results

| Arm | Mean reward | Gain over incumbent | Paired 95% gain CI | Harmed worlds | Harm upper 95% | Mean downside |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 6.988024 | 0.000000 | [0, 0] | 0 | 0.010348 | 0.000000 |
| Posterior predictive mean | 7.004012 | 0.015988 | [0.012051, 0.019972] | 62 | 0.258963 | 0.005367 |
| **Frozen mean-regret guard** | **6.988244** | **0.000220** | **[-0.000111, 0.000530]** | **14** | **0.074952** | **0.000394** |

The unguarded posterior mean has substantial average value, but it harms 62 of
288 worlds by more than the frozen `0.002` reward margin. The guard removes 48
of those harm events and sharply reduces average downside. It also discards
almost all useful value: it retains `1.38%` of posterior-mean gain and captures
only `0.86%` of oracle headroom. Five preregistered checks fail: minimum gain,
positive paired interval, harm-risk budget, retained posterior gain, and oracle
headroom.

## Cross-task interpretation

This result complements, rather than erases, the positive 288-world DLO-Lab
wrapping v9 result. A prospectively fixed baseline-relative Bayesian guard can
retain useful value and certify low harm on wrapping, yet the earlier Slingshot
calibration does not transfer to fresh continuous Slingshot worlds at the same
risk budget. The evidence therefore supports **task- and query-conditional
simulator competence certificates**, not a backend-wide or method-wide safety
label.

This is useful negative evidence for the paper: exact fallback and prospective
gating prevented a weakly transferring controller from being promoted, while
the unguarded arm exposes real decision headroom that a better calibrated guard
could capture in a genuinely new protocol. These 288 worlds are now closed to
controller, threshold, calibrator, or gate selection.

## Verification

The read-only verifier enumerates the exact 1,305-file, 677,043,088-byte result
tree. It rehashes every record and numeric bundle, verifies frozen source blobs,
reconstructs all 288 causal prefix observations, independently regenerates all
1,179,648 sensor-level decisions, recomputes native rewards from the sealed cube
trajectories, checks prefix/future replay and native QA, and independently
recomputes the world-level bootstrap intervals, exact binomial bounds, and gate.
Verification passed. This is a second implementation and custody replay by us,
not independent human review.

Key identities:

- compact summary: `f7947a626d6bf941704b532aebf6cde5447a7cde25e00a587ffe20a566f21086`
- verification: `c5206613f48bfb68c682f3c63289046252432d5cb4a144ef4e48e2345bd4ce94`
- lock: `7008acbe9ab7fd805832df4e97794f5c6924d00153bb25b6a5b6a2aa9abd54ef`
- continuation receipt: `70b14a0ba4b2b3b9449be954675e94b62617864483be066f542816951b97d5d2`
- decision: `46504a73df7f77e46b0d252657b6948ca23a301884a48a6e0e108e4f4243f490`
- barrier: `d2c1893755f86e6cca4210ef362ad1cb46b0ad5defc90a6c91c6a083df895432`
- generation: `0514d18e98bac0904d258916da5c24d929926497dfb1e92183ffcb4860922fa1`
- result: `35388657b9d3e162a5dcadeb003f6943123b3f19a9d8ac04b2eccd1cdec32ba1`
- continuation complete: `bf549331975d60a982f4efda17749fc9c1ff9ebe0ace9f261c670b7b40e48086`
- verified tree: `9c66f1a3f241465966d1ba37e0de8fe91622a9d7f87d910436d18c021803424f`
- summary file SHA-256: `85ea08c0a0f9bac17f39e40fa60f2734dd7a5fdcf7d63b30f23285ad353eaad3`
- verification file SHA-256: `f1f4639c196dac8618084cac059f2901a8c1db6fcf9bb725b7459dd67ea5011a`

The complete native archive remains local at
`/home/fpfaff/source-only/dlolab-slingshot-certified-guard-source-v2`.

## Claim boundary

This is prospective public-simulator evidence about one frozen Slingshot guard
under one registered world distribution. It does not establish robot safety,
real-world calibration, official benchmark superiority, point-prediction SOTA,
material identification, or performance on arbitrary out-of-distribution
worlds. The parent 32-world result remains development evidence only and is not
reclassified as a passing source study.
