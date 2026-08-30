# DLO-Lab wrapping finite-sample certified guard v9 result

## Decision

The complete 288-world prospective public-simulator replication **passes the
registered source gate**. The unchanged 0.975 posterior chance guard improves
mean native reward over the fixed action by `0.004721` (paired 95% bootstrap
CI `[0.003894, 0.005597]`) while reducing worlds harmed by unguarded continuous
Bayes from 15 to 1. Its exact one-sided 95% Clopper-Pearson upper bound on the
registered world-level harm probability is `0.016365`, below the frozen `0.05`
budget.

This result is the fresh replication evidence, not a reclassification of v8.
V8 remains a failure under its original strict gate and supplies only the
post-open calibration certificate that admitted the already-fixed controller.
The controller and its `0.975` threshold were unchanged after v8.

## Registered execution

- Frozen source revision: `1b66630b939852547798a1d421a728b429cd7d88`
- Public native simulator: DLO-Lab wrapping through the v7-qualified native
  Linux CPU/OSMesa runtime
- Fresh worlds: 288, disjoint from the source bank, v1-v4 studies, and all 144
  v8 calibration worlds
- Prefix batches: 32/32 ordinary successes
- All-action futures: 288/288 ordinary successes
- Technical failures: 0
- Replacements or retries: 0
- Sensor draws per world: 4096
- New recordings, protected data, held-v8, DLO4/DLO5, or official DLO3: none

The pre-future barrier passed only after the compact v8 calibration certificate
was revalidated. Failure of the certificate or any custody check would have
selected the exact fixed fallback rather than the guarded controller.

## Results

| Arm | Mean reward | Gain over fixed | Harmed worlds | Mean downside |
| --- | ---: | ---: | ---: | ---: |
| Fixed action | 0.887122 | 0.000000 | 0 | 0.000000 |
| Continuous Bayes | 0.911422 | 0.024299 | 15 | 0.001588 |
| 0.90 chance guard | 0.896294 | 0.009171 | 4 | 0.000200 |
| **0.975 chance guard** | **0.891844** | **0.004721** | **1** | **0.000036** |
| 0.99 chance guard | 0.890327 | 0.003204 | 0 | 0.000013 |
| Finite-particle Bayes | 0.916081 | 0.028958 | 28 | 0.005866 |
| Continuous MAP | 0.909049 | 0.021927 | 58 | 0.009514 |

The primary guard retains 19.43% of the continuous-Bayes gain and captures
8.70% of oracle headroom. It loses `0.019578` reward to continuous Bayes
(paired 95% CI `[-0.022978, -0.016379]`) while removing 14 of its 15 harmed
worlds. Mean downside falls from `0.001588` to `0.000036`, a 97.72% reduction;
the paired downside-reduction CI is `[0.000303, 0.003290]` in native reward.

The 0.90 and 0.99 thresholds and the finite-particle and MAP arms were frozen
before execution and remain descriptive sensitivity analyses. They do not
replace the registered 0.975 primary arm.

## Calibration-to-replication stability

| Quantity | V8 calibration (144 worlds) | V9 fresh replication (288 worlds) |
| --- | ---: | ---: |
| Guard gain over fixed | 0.005042 | 0.004721 |
| Harmed worlds | 2 | 1 |
| One-sided 95% harm-risk upper bound | 0.043073 | 0.016365 |
| Continuous-Bayes harmed worlds | 10 | 15 |
| Mean-downside reduction | 94.29% | 97.72% |

The fresh gain is close to the calibration estimate while the larger fresh
denominator tightens the exact harm-risk bound. This table is descriptive;
only the V9 column is evidence for the prospectively registered V9 gate.

## Finite-sample risk statement

The registered harm event is a world's mean reward over 4096 sensor draws more
than `0.002` below the exact fixed action. One event occurred among 288 fresh
worlds: an observed fraction of `0.003472` and an exact one-sided 95% binomial
upper bound of `0.016365`. This statement assumes exchangeable worlds from the
registered simulator stress distribution. It is not a guarantee for every
sensor draw, every possible simulated world, or any physical system.

## Gate accounting

Every locked check passes:

- complete 288-world denominator and all native QA;
- compact calibration certificate revalidated before future access;
- at least two distinct oracle actions;
- gain over fixed at least `0.003` with positive paired 95% CI;
- one-sided 95% harm-risk upper bound at most `0.05`;
- at least 10 continuous-Bayes harmed worlds and at least five removed;
- at least 75% mean-downside reduction;
- at least 15% of continuous-Bayes gain retained;
- at least 5% of oracle headroom captured.

## Verification

The read-only verifier enumerates the exact 1,287-file,
9,891,781,089-byte result tree. It validates every write-once record and native
bundle, verifies the frozen source blobs, reconstructs all prefix observations,
regenerates the decision bundle and pre-future barrier, checks all native
futures and prefix parity, reconstructs reward and scoring, and independently
recomputes the paired bootstrap intervals and exact Clopper-Pearson bound.
Verification passed. It is an arithmetic and custody check, not independent
human review. The verifier requires byte-identical recorded worlds and actions,
while allowing at most four ULPs when independently regenerating material
parameters across NumPy runtimes and a tightly bounded floating-point tolerance
for posterior diagnostics; the selected actions must remain exactly identical.

Key identities:

- compact summary: `d3c577ce1ec215c6d56c4d405e7f9d886f38b7e6d021bb6d62f37da6bd4784b9`
- attempt: `1b4430b526178d7247a0639ce1e662b3d84e1ac13a9c070cfafbc48882a4733a`
- lock: `2f96bb2e52501a5e137e44faec4ed699b81dd828b00a98e3654f53e588e798ce`
- calibration certificate: `790bc9facc33838bf32b64e232e10f8012035cd53f1f8880d629f7fdf06714c2`
- decision: `d8374787a58805c145aa429fb8994a86b6cbb9206197b3c4f6490cff89eb5365`
- barrier: `fc2a29ceb932a1bc8dbffd00f5355922603d2d1d08456e8457af2e053fd33fde`
- generation: `7b2bb3180b56456343a871b458d24a64fb51923f4adc6ec6a2f9b506f1b4fe86`
- result: `50801d4da518238ffc2e2d1995d7467f97286cb4535eff60633a5f3d0112b32d`
- verified tree: `ffca78f511c882446b95d283458aebb469942bc7d98dd66fb7b5844bcdecef5c`
- verification file SHA-256:
  `973826ab96141c3efd7178daddfbeed75e5b3a0870d56570441a55e1c540c6d5`

## Claim boundary

This is prospective evidence in a public deformable-object simulator that a
baseline-relative Bayesian chance guard can retain positive decision value
while sharply reducing downside relative to an unguarded Bayesian controller,
with an exact finite-sample bound on a preregistered world-level harm event.
It does not establish physical-robot safety, real-world calibration, official
benchmark superiority, point-prediction SOTA, material identification, or
performance on arbitrary out-of-distribution worlds. These 288 worlds are now
closed to controller, threshold, or gate selection.
