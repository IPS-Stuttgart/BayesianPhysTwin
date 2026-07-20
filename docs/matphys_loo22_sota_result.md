# Object-Disjoint MatPhys LOO22 Result

## Status

This is the frozen result of protocol `matphys-guarded-bayesian-loo22-v2`.
The official PhysTwin benchmark informed method development, so this is a
retrospective benchmark evaluation with an object-disjoint training boundary and a
sealed within-run future opening. It is not an untouched external confirmation.

No future observation or metric was used to train the eleven leave-one-object-out
models, choose a proposal strength, or select a within-family correction. The permitted
past validation interval did include released manual 3D tracks, so the selector is
online-supervised rather than label-free.

## Frozen Inputs

- Cohort: 22 interactions from 11 physical objects.
- MatPhys initialization: generic frozen `MCG-NJU/videomae-base`; fresh trainable
  parameters in every fold; no PhysTwin-trained checkpoint.
- Proposal strengths: exact incumbent plus `0.25`, `0.50`, `0.75`, and `1.00` log-space
  spring interpolation.
- Per-case gate: at least 1% past-validation improvement, no CD or track regression,
  and at most 1 mm identity-replay coordinate RMSE.
- CoTracker3 cues: 104,871 tracks across 22 cases; all cue entries at or after the
  released training boundary are neutral and unavailable.
- Evaluation code: Bayesian-PhysTwin commit
  `e59ec8fe85493213e33ce023a6b0a3f893355368`.
- Merged spring-field manifest SHA-256:
  `c5301b6c0597e3e1db5cdd70007468956b4bfc7c7fc39f8c2c78875eccd0986e`.

## Sealed Selection

The source-only selector chose the exact incumbent for 14 cases and a MatPhys proposal
for 8 cases:

| Strength | Selected cases |
| ---: | ---: |
| `0.00` | 14 |
| `0.25` | 1 |
| `0.50` | 1 |
| `0.75` | 1 |
| `1.00` | 5 |

Within the selected families, the existing overlay selector chose the raw backbone in
3 cases, the Bayesian anchor in 14, and the last-residual baseline in 5. The sealed
selection SHA-256 is
`5eedb6cb5a747b856c0af696c5029038a8022f00828f43295f201578a4494890`.

## Future Result

All values are equal-case means over the complete 22-case cohort.

| Method | Future CD (mm) | Future manual-track error (mm) |
| --- | ---: | ---: |
| Exact incumbent family | 10.849 | 19.482 |
| Guarded LOO MatPhys selection | **10.242** | **19.059** |
| Published rounded MatPhys reference | 8.000 | 15.000 |

The guarded method improves the incumbent equal-case means by 5.59% CD and 2.17%
track error, but misses the preregistered SOTA thresholds by 2.242 mm CD and 4.059 mm
track error. The SOTA point-estimate gate therefore fails on both metrics.

The object-clustered paired bootstrap gives:

| Metric | Equal-object change | 95% interval | Improvement probability |
| --- | ---: | ---: | ---: |
| Future CD | -3.16% | [-5.75%, -0.84%] | 0.998 |
| Future track | -1.51% | [-3.26%, +0.68%] | 0.928 |

CD improves across early, middle, and late horizons with object-clustered intervals
excluding zero. Track improvement is supported in the early third, but its middle and
late object-clustered intervals include zero. Seven cases improve in CD, eight improve
in track error, one case regresses in CD by 1.01%, and the identity fallback makes all
other case-level changes exact ties.

Every unguarded global proposal strength is worse than the incumbent on both full-cohort
means. Full strength reaches 12.281 mm CD and 24.868 mm track error. The positive result
is therefore the guarded selective transfer, not a generally superior spring field.

Predictive calibration is not established by this deterministic family report. The
paired intervals quantify point-estimate change only; they are not NEES or predictive
coverage results.

## Decision

Do not advance this spring-only arm to an independent preregistered SOTA claim. Preserve
it as a useful object-disjoint component and as evidence that conservative fallback is
essential. Do not tune it against the opened LOO22 futures.

The next independent candidate should target the remaining online state/discrepancy
error rather than add more spring strength. The open Deform360 diagnostics motivate a
prospective belief update that requires both physical-prior/action support and
structurally redundant observation evidence, models common-mode camera bias, and falls
back exactly when innovation support is weak. That method must be frozen on open source
objects and evaluated on fresh objects; the present LOO22 future cannot be reused for
selection.

## Evidence

- `results/sota/matphys_loo22_v2/loo_strength_sweep.json`
- `results/sota/matphys_loo22_v2/backbone_family_selection.json`
- `results/sota/matphys_loo22_v2/backbone_family_future.json`
- `results/sota/matphys_loo22_v2/matphys_loo_sota_report.json`

Evidence SHA-256 values, in the same order, are:

```text
ba40c04c81c49be3152d4a7207da3a9de43a2da3a3bae62d15107736c4124788
5eedb6cb5a747b856c0af696c5029038a8022f00828f43295f201578a4494890
6560317dbaebaf99b46328e526febf4a276d6183163284bc56b0d473dfa5b9d9
edfcb2991ee1e710c4b35c56bb3d598e005daf7dfa1f15f90412ea5a82c6496f
```
