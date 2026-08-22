# RGBench MatPhys selective-risk cohort v1

This lock creates a fresh public-data route for a MatPhys-backed probabilistic
forecast and abstention study without reusing the exhausted PhysTwin-22 cases or
touching the protected Deform360 cohorts.

## Scope

The candidate is the native Warp spring-mass family released with MatPhys,
adapted to RGBench fixed-point episodes. It will fit an ensemble from a causal
point-cloud prefix and forecast the untouched continuation under recorded
end-effector motion. Ensemble disagreement is a candidate risk signal; it is not
a safety certificate.

RGBench does not provide the RGB video required by MatPhys's published visual
parameter predictor. This is therefore a **native MatPhys simulator study**, not
a reproduction of the published visual MatPhys method. It is also not an
official RGBench SOTA claim, a Deform360 result, or a PhysTwin-22 result.

## Frozen cohort boundary

The public Git and Hugging Face revisions are pinned in
`protocols/locks/rgbench_matphys_selective_risk_v1.json`. Garments are split by
published manifold status using metadata only:

| Role | Manifold garments | Non-manifold garments | Cells |
| --- | ---: | ---: | ---: |
| Source/development | 4 | 1 | 15 |
| Untouched target | 3 | 1 | 12 |

One capture is selected per garment and action (`fling`, `fold`, `grasp`) using
the minimum salted SHA-256 over the metadata identity. Every selected identity
and its selection key are content-bound. No difficult or failed case may be
replaced.

Only source payloads may now be downloaded and decoded. Target download,
decode, execution, and outcome access remain forbidden. Source results may be
used to build the adapter and freeze the eventual target method.

## Required advancement evidence

Before target access, a separate source-gate artifact must demonstrate:

- deterministic source replay and causal-prefix custody;
- a non-degenerate MatPhys ensemble;
- leave-one-garment-out future-mean non-regression;
- risk-coverage ordering that beats a prefix-residual comparator;
- exact fallback or abstention when the provider or gate fails; and
- complete ordinary-success, technical-failure, and unsealable accounting.

The 15 source cells are development groups, not enough independent evidence for
a deployment-harm guarantee. Passing the source gate would authorize only a
separately frozen untouched-garment evaluation.
