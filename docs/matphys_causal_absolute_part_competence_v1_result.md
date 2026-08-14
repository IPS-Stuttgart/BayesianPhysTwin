# Causal absolute part-aware MatPhys competence result

## Decision

The frozen one-case causal competence gate failed. Do not run the proposed
five-case panel, tune this family on `single_lift_zebra`, or present the result
as evidence for the published MatPhys method.

This result is development-only and uses one already-open released PhysTwin
case. The prediction was sealed before its released future interval was
scored. No fresh target, confirmation, or held-v8 artifact was opened.

## Result

The 200-epoch terminal fit used only 16 RGB frames ending at frame 33. All
6,600 optimizer steps were accepted, no step was rejected, and the model and
optimizer checkpoint were finite. The exact export and official evaluator were
then sealed before frames 46--65 were scored.

| Method | Future CD (mm) | Future track (mm) |
|---|---:|---:|
| Released PhysTwin | 15.451 | 24.304 |
| Causal absolute graph-part field | 31.057 | 34.789 |
| Error change | +101.01% | +43.14% |

The required improvement was at least 10% in both metrics. The candidate
instead regressed on both, so the metric gate and overall competence gate
failed. It also missed the non-claiming `8/15 mm` headroom diagnostic.

The physical validity checks passed only in the literal protocol sense: all
54,989 springs were finite and positive, and five part geometric means were
numerically distinct. Their maximum/minimum ratio was just
`1.000000000035`, however, so the learned field was effectively spatially
uniform and saturated near `100,000`. This is not meaningful part recovery.

## Interpretation

Together with the positive all-frame reconstruction control, the result
separates representational capacity from causal identification. The DINO
graph-part adapter can express a useful spatial field when future frames are
available, but this prefix-only endpoint objective does not identify one that
transfers to the future. More epochs drove the field toward an almost uniform
upper-bound solution and worsened continuation.

The direct per-case absolute-prefix family is therefore closed without outcome
tuning. A future MatPhys use in BayesianPhysTwin should be source-trained and
baseline-relative: treat physical fields as competing prior mechanisms, admit
them only through held-out transfer and calibration gates, and preserve exact
PhysTwin fallback. It should not replace the guarded online-belief program or
be evaluated on a fresh target from this failed gate.

## Provenance

The compact evidence is in
`results/sota/matphys_causal_absolute_part_competence_v1/`. Its key files are:

- `prediction_seal.json`, sealed before future scoring, SHA-256
  `a39cbc7f61a714b7da10f59d73b63fa93e9482a83fe3594be5d72ebcf425be3a`;
- `decision.json`, SHA-256
  `cc692cac69b3c8b835d001d8469138571095da0f4232effac53949aa04e9edd5`;
- candidate and released-PhysTwin evaluator outputs, SHA-256
  `89a05f59d36e4a0d0e42e36c72d54a5229f1caaf22adf54d7a291712807e3664`
  and
  `7903bc2292749edb2cf07ef6244b6655cc9d527e7ac6bb7d25861a8559163220`.

The implementation is pinned at
`4bef7408390ccd5b6533e490873ae5929366b270`. Prob4D was unused and
MolmoMotion weight was zero.
