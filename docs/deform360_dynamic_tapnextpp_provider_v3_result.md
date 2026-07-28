# Dynamic TAPNext++ Set-Valued Association V3 Result

## Status

V3 is a post-open two-case source mechanism study. It is not a transfer,
confirmation, calibration, or state-of-the-art result. The V1 target cohort,
fresh objects, held-v8 artifacts, future camera observations, and tactile data
remained unopened.

Prediction code was frozen at
`4b8be3010924c62177b4e61e3133ee36b9db0474`; the source evaluator and
advancement rule were frozen at
`08a2936032716e8288a7ce4ac43a54e882538e69`.

## Mechanical Result

The set-valued treatment fixed the confidence-collapse mechanism on the source
case that already had valid multiview endpoints:

| Diagnostic | V2 | V3 |
| --- | ---: | ---: |
| Accepted update intervals | 0/3 | **1/3** |
| Available centres at frame 57 | 0 | **15** |
| Mean prior reliability at frame 57 | below 0.02 | **0.8926** |
| Pairwise-consensus inliers | 0 | **15/15** |
| Mean posterior inlier probability | 0 | **0.9998** |

The second case retained zero birth-to-update support and therefore remained
an exact fallback. No view-count, reliability, correspondence, or regret
threshold was lowered.

## Hidden-Identity Result

On the one nonzero case, V3 improved the globally selected physical/persistence
backbone:

| Metric | Selected backbone | V3 | Relative |
| --- | ---: | ---: | ---: |
| Hidden identity RMSE | 6.382 mm | **5.575 mm** | **-12.65%** |
| Hidden symmetric Chamfer | 4.666 mm | **4.318 mm** | **-7.47%** |
| Late hidden identity RMSE | 10.599 mm | **9.266 mm** | **-12.57%** |

However, pure persistence was `0.088` mm identity RMSE and `0.056` mm Chamfer
on those hidden identities. V3 therefore regressed by more than sixtyfold
against the correct fallback. The frozen advancement gate failed because the
nonzero case did not jointly beat persistence.

The unsupported second case was an exact tie with its selected persistence
backbone. Both cases were nonregressing relative to the selected backbone, but
this is insufficient for advancement.

## Diagnosis

The result falsifies the idea that confidence double counting was the only
barrier. V3 fixed that barrier and produced a useful correction relative to
the chosen physical continuation, yet the global backbone decision remained
wrong.

The query schedule deliberately selects graph nodes with at least 5 mm of
predicted motion. At the supported endpoint, those observed identities moved
enough that persistence had `6.525` mm direct endpoint RMSE and the physical
backbone won the sparse current-observation selector. The disjoint hidden
identities were instead almost static, making persistence nearly exact.

The sparse sample is therefore informative about the high-motion region but
not representative of the whole object. One global physical-versus-persistence
choice transfers this selection bias to hundreds of hidden identities.

## Decision

Do not evaluate V3 on a fresh cohort. Preserve it as evidence that:

1. set-valued assignment uncertainty is the correct interface for dense local
   candidate patches;
2. the guarded Bayesian update can improve an admitted physical continuation;
3. motion-targeted observations cannot select one global object-wide
   continuation.

The next source method must separate two observation roles:

- **active queries** near predicted response regions estimate local dynamic
  corrections;
- **sentinel queries** sampled across low-, middle-, and high-motion graph
  strata determine whether motion is local or global and protect static
  regions.

Backbone weights and residual corrections must then be decoded as local fields
with an explicit persistence prior away from supported centres. A normalized
RBF whose tails eventually cover every node is not an adequate fallback.

## Artifacts

- Compact result:
  `results/sota/diagnostics/deform360_dynamic_tapnextpp_provider_v3/source_development_result.json`
- Per-case source reports:
  `results/sota/diagnostics/deform360_dynamic_tapnextpp_provider_v3/source_reports/`
- Result payload SHA-256:
  `0afa1ca157bb439d97a421bbbfc29b919cc55a8003620dc21b4646438acb7481`
