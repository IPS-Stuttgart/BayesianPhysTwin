# Cloth Sim2Real Prob4D Covariance Attribution v1

## Status

The five-treatment retrospective real-Cloth covariance attribution completed on
`workstation2`. The full-joint treatment did **not** establish a point-accuracy
advantage. Its bounded positive signal is better raw uncertainty behavior: it
achieved the highest raw nominal-90% dynamic coverage and required the smallest
calibration inflation. Raw coverage remained far below 90%, calibrated intervals
remained broad, and all treatments failed on the quasi-static task.

This is controlled retrospective attribution on previously opened repeat-2
outcomes, not fresh confirmation. `claim_authorized` is `false`.

## Evidence identity

- Workflow run: `31384479427`
- Source revision: `7019d4e6effa88addba40cf6faeaeabc6d285c02`
- Artifact ID: `9061934079`
- Artifact SHA-256:
  `04f98659a45f5b6b64d7b9feb865c29085c7be18dd99b1684a7a99a439d18955`
- Result ID:
  `e4fb7b7bd7f7455cf5e97def84db6f82da1360f5d00c344ca0a55c8d37219670`
- Report ID:
  `bbed9a588ed1f5d2ba076131c2ec9f89bee1f4624b5eaa902ee5fa16a9ca3ff1`
- Dataset SHA-256:
  `268d07d94396f6f4ca277b6da0e8acf43512747fea6d40327eb33166da972c7f`
- Dataset size: `3,762,021,195` bytes
- Frozen campaign revision:
  `94fe5d0012e66b93d73bf9474df2b28f2a83d153`
- Benchmark revision:
  `178a9b9722191c51cf0dcbc3cf0dc03701b09eb3`
- Simulator-utilities revision:
  `e4d107965c09d7a31da33b5b3e14a2bdbbfe79a4`

The exact dataset passed byte-count and SHA-256 validation. All twelve registered
MuJoCo baselines were verified or reproduced before scoring. For each case, all
five prediction seals were written before the newly performed future read.

## Primary dynamic result

All five covariance treatments retained essentially the same conditional-mean
improvement over the physical rollout.

| Treatment | Symmetric L1 CD | Improvement | Raw 90% coverage | Calibration multiplier | Calibrated coverage | Calibrated width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full joint | 78.6001 mm | 7.4207% | 44.36% | 4.8163 | 91.15% | 150.85 mm |
| Shared uncertainty underreported | 78.6001 mm | 7.4207% | 42.51% | 4.9413 | 90.13% | 145.03 mm |
| Block diagonal | 78.6210 mm | 7.3987% | 40.55% | 5.4034 | 90.53% | 148.23 mm |
| Independent rows | 78.5952 mm | 7.4260% | 40.43% | 5.4587 | 90.64% | 149.12 mm |
| Shared uncertainty removed | 78.6001 mm | 7.4207% | 40.29% | 5.4670 | 90.57% | 148.69 mm |

The independent-row treatment is lower than full joint by only `0.00491 mm` in
object-balanced dynamic symmetric L1 Chamfer. Shared-uncertainty removal and
underreporting alter the mean by only a few nanometres. The experiment therefore
provides no practically meaningful point-loss preference for full joint.

Full joint raises raw dynamic coverage by `1.85` to `4.07` percentage points
relative to the four ablations and reduces the fitted calibration multiplier by
`2.53%` to `11.90%`. This is evidence that retaining shared uncertainty reduces
underdispersion. It is not evidence of sharp calibration: raw coverage is only
`44.36%`, and the calibrated mean interval width is `150.85 mm`.

## Quasi-static negative result

The physical quasi-static baseline has object-balanced symmetric L1 Chamfer of
`56.0134 mm`. Full joint produces `60.1285 mm`, an `8.1334%` regression. The
other treatments regress by `8.13%` to `8.46%`.

Thus, covariance treatment does not repair the mean-model mismatch in
contact-rich quasi-static evolution. On the symmetric metric, full joint admits
two harmful target updates out of six total trials; block diagonal admits three.
The task-level source/calibration evidence already distinguishes this negative
domain from dynamic continuation.

## Pairwise attribution

The complete six-trial report gives nearly identical full-joint and
shared-uncertainty-removed means. Full joint differs from independent rows by
`+0.00120 mm` in mean symmetric L1 Chamfer, so independent rows is microscopically
better on this metric. Full joint is `0.10715 mm` better than block diagonal over
all six trials, driven mainly by the quasi-static cases.

`pairwise_diagnostic.json` adds an exact paired percentile bootstrap over every
`n^n` resample. It is a post-outcome retrospective diagnostic, not a new
confirmatory analysis. With only three trials in each task stratum, inferential
resolution remains weak even when a numerical interval excludes zero.

## Calibration-derived domain guard diagnostic

A separate diagnostic applies one rule to the calibration repeat only:

- authorize a task stratum when mean improvement is at least 5%;
- require at least two wins among three cloth trials;
- reject a stratum if any trial regresses by more than 5%;
- use exact physical fallback for a rejected stratum.

For full joint, calibration authorizes dynamic continuation (`6.35%` mean
improvement, two wins, no regression) and rejects quasi-static continuation
(`-7.41%` mean improvement, one win, worst regression `-18.60%`). Applied to the
already-open target as a post-outcome diagnostic, this yields three accepted
dynamic trials, three exact fallbacks, zero harmful accepted updates, and an
overall six-trial improvement of `4.48%`.

This demonstrates a plausible safer deployment policy, but it was specified
after target outcomes were already open. It requires a new independent cohort
before supporting a prospective claim.

## Scientific conclusion

The real-Cloth study narrows the role of Prob4D covariance modelling:

1. The dominant dynamic accuracy gain comes from the prefix-conditioned mean
   correction, not the covariance representation.
2. Full joint covariance reduces raw underdispersion and calibration inflation,
   but does not make uncertainty sharp.
3. Covariance modelling cannot compensate for a misspecified conditional mean in
   quasi-static contact-rich motion.
4. The highest-value next experiment is a fresh dynamic-cloth cohort with a
   calibration-frozen task or regime guard, plus a proper multivariate score and
   sufficiently many independent calibration trials for finite-sample coverage.

The present result does not establish fresh-object transfer, general uncertainty
calibration, Causal4D intervention benefit, deployment safety, or state of the
art.
