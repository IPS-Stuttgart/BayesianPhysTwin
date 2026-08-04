# Full-22 source-tuned tempered model-average experiment v1

> **Evidence status:** retrospective and non-claim-bearing. The released
> PhysTwin cohort and its frozen three-case development / nineteen-case
> confirmation split already informed BayesianPhysTwin development. These
> results are not fresh independent validation, deployment calibration, or a
> new state-of-the-art claim.

## Locked selection and execution

- Protocol SHA-256: `a351cf37ba19130feca4dcfb87b1e7ab9a2e601d22edeed3a39a00c904ecbbe3`
- Selection ID: `d204e656894c007644c4b04cbe1b529da0e6aca0081a823dc4d5ef602bbb56fa`
- Selection JSON SHA-256: `5dac806ab91a5709215bd3572b04361b5a541cfe81987c9385455f8ba7420b5f`
- Readout SHA-256: `b110e0ef65f90a982ee0b2968ded4fa6ea3650342eedceb4e11ffa3bcabfdb69`
- Per-case CSV SHA-256: `439d3a91bfe071d9e1455435134e86d33ad6e7da0c4e6fc95f9213a8429fdb34`
- Final self-hosted run: `30897954787`
- Final self-hosted artifact: `8887837678`
- Final self-hosted artifact SHA-256: `91af2dca7ada3c8dbdb606af9b872dcd0d7ec16703f317bd2d4f7e3824e241c0`

The evaluator wrote and hash-bound `selection.json` before opening confirmation
outcomes. Temperature, guard threshold, and covariance scales depend only on the
frozen development cases `single_lift_sloth`, `double_lift_sloth`, and
`double_stretch_sloth`.

A contract-only correction removed an un-reweighted nominal-probability field
from the internal tempered object. The corrected `workstation2` execution
reproduced the exact same selection, readout, and per-case bytes as the preceding
run; only the recorded Git merge revision and revision-dependent summary hash
changed.

## Source-only choices

The locked temperature grid was `[1, 2, 4, 8, 16, 32, 64, 128]`. Selection by
equal-development-case updated-only future Gaussian negative log likelihood
chose the largest candidate:

- selected evidence temperature: **128**;
- model-average-specific guard threshold: **4.611 mm** endpoint disagreement;
- covariance multipliers: **2.742 early**, **5.049 middle**, **8.261 late**.

The development objective improved monotonically at the upper end: mean NLL was
`621.377` at temperature 1, `-6.569` at 64, and `-7.974` at 128. Therefore the
locked grid did **not** bracket an interior optimum. Temperature 128 is evidence
that cumulative component log evidence is severely over-scaled, not a universal
recommended constant.

The guard accepted only `double_stretch_sloth` among the three development
cases. Its source objective improved from the fallback by 1.21% in the combined
mean of Chamfer and track ratios, while the other two cases failed the no-positive-
source-regret requirement.

## Confirmation point accuracy

Equal-case means on the frozen 19-case confirmation partition:

| Method | Chamfer (mm) | Track (mm) |
| --- | ---: | ---: |
| released PhysTwin | 11.122 | 22.189 |
| selected Bayesian anchor | 9.828 | 19.523 |
| temperature-1 model average | 9.713 | 19.335 |
| temperature-128 model average | 9.764 | 19.429 |
| temperature-128 guarded model average | 9.614 | 19.234 |
| last-supported residual | **9.581** | **19.188** |

Strong tempering improves uncertainty but slightly degrades the unguarded point
predictor relative to temperature 1:

- Chamfer: `+0.050 mm`, locked 95% paired-bootstrap interval
  `[-0.009, +0.117] mm`;
- track: `+0.094 mm`, interval `[+0.008, +0.191] mm`.

The guard accepts 5/19 confirmation cases and falls back exactly on the other 14.
Its remaining difference from last residual is small and inconclusive:

- Chamfer: `+0.033 mm`, interval `[-0.015, +0.101] mm`;
- track: `+0.046 mm`, interval `[-0.015, +0.116] mm`.

It improves over the selected Bayesian anchor by `0.214 mm` Chamfer, with locked
interval `[-0.448, -0.022] mm`, and by `0.289 mm` track error, with interval
`[-0.690, +0.031] mm`. This mainly reflects conservative fallback to the stronger
last-residual baseline, not successful selection of model-average wins.

### Post-hoc guard diagnosis

This diagnosis is descriptive and was not a selection criterion. Of the five
accepted confirmation cases, only one improves the combined Chamfer/track ratio
over fallback; accepted cases average **1.94% positive combined regret**. The
Spearman association between disagreement score and confirmation combined regret
is `-0.18`. The source-tuned disagreement score therefore does not transfer as a
useful regret ranking. The guard is safe chiefly because it rarely accepts.

## Mixture uncertainty

On confirmation cases, strong tempering reverses the component-collapse failure:

| Diagnostic | Temperature 1 | Temperature 128 |
| --- | ---: | ---: |
| mean component entropy | 0.117 nats | 2.462 nats |
| mean median effective components | 1.005 / 15 | 10.281 / 15 |
| mean median between-model covariance fraction | 0.00021 | 0.342 |

Raw predictive diagnostics on updated tracks:

| Posterior | Pred. std (mm) | 90% coverage | NEES / 3 | Mean NLL |
| --- | ---: | ---: | ---: | ---: |
| selected Bayesian anchor | 7.583 | 0.519 | 691.129 | 1022.053 |
| temperature-1 model average | 3.667 | 0.380 | 2529.798 | 3779.117 |
| temperature-128 model average | 8.796 | 0.796 | 3.037 | -7.306 |
| temperature-128 + case-blocked inflation | 20.696 | 0.972 | 0.636 | -8.584 |

Tempering is the decisive positive result: it raises raw 90% coverage from
38.0% to 79.6%, reduces NEES/3 by more than three orders of magnitude, and
restores material between-model covariance. The raw tempered posterior is nearly
on target early (`0.908` coverage), but remains under-dispersed in the middle
(`0.746`) and late (`0.711`).

The worst-development-case horizon multipliers overcorrect. Confirmation 90%
coverage becomes `0.981` early, `0.962` middle, and `0.970` late. Overall 97.2%
coverage and NEES/3 of 0.636 are conservative rather than calibrated. One
confirmation case still reaches only 89.0% coverage, showing that high aggregate
coverage does not establish groupwise validity.

## Scientific decision

1. **Do not replace the point predictor with the unguarded tempered mean.** It
   loses a small but measurable amount of track accuracy relative to temperature
   1 and remains worse than last residual.
2. **Do not promote the current disagreement guard.** It suppresses large regret
   through fallback but does not identify confirmation improvements.
3. **Do not promote the worst-case horizon inflation as calibrated uncertainty.**
   It is markedly overconservative and is based on only three previously used
   development groups.
4. **Preserve evidence tempering as the promising result.** It fixes the central
   mixture-collapse pathology and substantially improves uncertainty quality
   without changing the component bank.

The next claim-bearing study should estimate an evidence scale or effective
sample size on genuinely independent object/session groups, use an interior-
bracketing temperature grid or normalized per-observation evidence, and calibrate
a smooth horizon-dependent scale on independent groups. Point prediction should
remain last-residual or temperature-1 unless an independently trained guard shows
actual regret-ranking skill.
