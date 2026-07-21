# Frozen Prob4D bias guard on the additional cloth source

## Status

This is an unchanged-method diagnostic on the already-open 11-case additional
cloth cohort. It is source evidence only. It is not a confirmatory evaluation,
calibration result, or state-of-the-art comparison.

The candidate, validation split, thresholds, and exact-fallback behavior are
identical to the frozen 19-case exploratory experiment. The only metric available
for this cohort is point-cloud Chamfer distance; no manual-track result is inferred.

## Information boundary

- The Prob4D update reads only frames before the released training boundary.
- The candidate is fitted on the first 75% of the training interval.
- Admission uses the remaining 25% of the training interval.
- Future point clouds are read only after the guarded trajectory is fixed.
- The fixed Bayesian anchor trajectory is the selected physical baseline.
- Rejected updates return that baseline bit for bit.
- Cases are clustered by physical garment, not treated as independent actions.

The cohort had already been examined by earlier work, so the split is an audit
boundary rather than a prospective seal.

## Result

| Arm | Future CD (mm) | Late CD (mm) | Future change |
| --- | ---: | ---: | ---: |
| Selected Bayesian anchor | 5.547 | 6.379 | reference |
| Raw static Prob4D candidate | 6.897 | 7.498 | +24.34% |
| Guarded candidate | 5.529 | 6.358 | -0.33% |

The candidate was available in all 11 cases. The frozen rule admitted one case,
`cloth_shirt_fold`, where future CD improved by 0.203 mm and late CD improved by
0.231 mm. The other ten cases were exact fallbacks. There were no harmful admitted
cases.

The six-garment cluster bootstrap for guarded minus selected future CD is
-0.0169 mm with a 95% interval of [-0.0507, 0.0000] mm. The upper endpoint is zero
because five of six garment clusters are exact fallback ties.

## Decision

The diagnostic validates the conservative fallback but does not establish static
Prob4D discrepancy transfer. Across the 19-case exploratory cohort and this
separate cloth source, the static family produces isolated gains rather than broad
support, and it has already admitted one harmful replay-noisy case. It therefore
does not advance to an independent preregistered evaluation.

The next admissible source-only family is an action-conditioned temporal
discrepancy model. It must predict correction evolution from the observed physical
response, retain disjoint prefix validation, and preserve exact fallback. No
threshold may be changed using this result.

## Reproduction

The runner is
`scripts/remote/diagnose_phystwin_prob4d_bias_guard_additional.py`. The archived
result is
`results/sota/diagnostics/phystwin_prob4d_bias_guard_additional_v1/result.json`
with canonical result SHA-256
`5a01c24d131219e3897ce4af0081ba1c4ddbba4ac9585e79fe64c89b946872c6`.
