# Deform360 MVTracker Privileged-Depth Competence Result v1

Date: 2026-07-26

Status: competence gate failed; the MVTracker-to-Deform360 route is stopped.

## Question

This one-case, outcome-open source control asked whether a learned multiview
3D tracker could recover useful material-point motion when supplied with
favorable rendered depth. The depth was generated from the released
full-sequence splat and is privileged reconstruction evidence, so even a pass
would only have authorized a later causal learned-depth study.

The source case, cameras, 16 query identities, tracker revision, checkpoint,
prefix, anchoring rule, and three acceptance gates were frozen before running
MVTracker. Prediction arrays were hashed and sealed before the authorized
source target was transferred for scoring.

## Result

All 304 eligible identity-frames were supported, but the accuracy gates failed
decisively.

| Predictor | Identity RMSE |
| --- | ---: |
| Exact persistence | 1.165 mm |
| MVTracker | 11.984 mm |
| Sealed physical prior | 12.696 mm |

MVTracker is 5.61% better than the physical prior, but its error is 10.29
times persistence, a 928.56% regression. The frozen gate required at least a
10% gain over the better baseline and at most 10 mm absolute RMSE. Only the
75% support gate passed.

The exact registered decision is:

```text
stop-mvtracker-deform360-route
```

Do not build or evaluate the proposed prefix-only learned-depth arm on this
evidence. Do not tune MVTracker, its anchoring, visibility threshold, camera
panel, or depth processing against this opened source outcome.

## Interpretation

The negative result is informative because favorable multiview depth did not
rescue identity tracking on an action window where the sealed physical model
predicted substantial motion. MVTracker retained complete visibility and had
only 0.153 mm mean query-frame anchor correction, so the failure is not
explained by missing support or a large initial query offset.

The control does not establish that MVTracker is generally inaccurate. It
rejects this frozen combination of MVTracker, privileged Deform360 splat
depth, the 16 registered material identities, and the selected squirrel
source interaction. It also does not validate persistence as a general
state-of-the-art method; persistence is unusually strong in these short,
low-motion Deform360 action windows.

## Provenance Note

The sealed prediction report contains a runtime label saying scene
normalization used "frame-zero rendered depth and calibrated cameras only."
Inspection of the hashed runner shows that
`compute_auto_scene_normalization` received all authorized prefix depths and
calibrations from frames `0..19`. The report separately and correctly records
those exact rendered-depth indices.

This wording error does not cross the information boundary: all frames were
inside the frozen prefix, and the arm was already explicitly privileged
because every rendered depth came from a full-sequence splat. The sealed
artifacts are preserved unchanged. The runner wording was corrected only for
future provenance; no prediction was rerun.

## Information Boundary

- RGB and rendered-depth frames used during prediction: `0..19`.
- Rendered depth status: privileged full-sequence reconstruction evidence.
- Source target and outcome were absent until after prediction sealing.
- No fresh object, held-v8 artifact, or sealed target was accessed.
- This is a one-case source competence result, not confirmation, a
  Bayesian-PhysTwin gain, or a state-of-the-art result.

## Provenance

- prediction implementation commit:
  `fb380504dae3c1dcbcf64334619ee1d404733b3e`
- evaluation compatibility commit:
  `0db3188b13f76055fc8125b8c8f914a08c004aff`
- MVTracker revision:
  `ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072`
- prediction archive SHA-256:
  `55112b2557d40a1bac0744944291042bf225a66cb9542d46fb7ae249282a6eda`
- prediction report SHA-256:
  `86e18ffe462bebee0a0949ac28e73ee6f81b7c7e67b04b0981d0772d6f0dad2b`
- prediction seal SHA-256:
  `3719875b54f4928140ba361df48ae37a4a1e17b12e8ab7591ab96572ec109697`
- evaluation SHA-256:
  `bbcaf4c791d31184869102a16c58b4e61b24721bc00e24825c10a0659a929511`
- evaluation canonical result SHA-256:
  `000da21dbef486172294763632004fd73aa911caac87fb002daa5ca768cf4c37`
- archived evidence:
  `results/sota/diagnostics/deform360_mvtracker_privileged_depth_competence_v1/`
