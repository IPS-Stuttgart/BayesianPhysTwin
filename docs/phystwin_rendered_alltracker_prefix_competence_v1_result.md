# PhysTwin Rendered-AllTracker Prefix Competence v1 Result

## Decision

**Rejected.** The direct PhysTwin-render-to-real AllTracker interface failed
all five frozen source-competence gates on `single_lift_cloth`. No automatic
query or Bayesian-assimilation follow-up is admitted for this interface.

This is an opened-source association-oracle control, not an independent
evaluation. Prediction used only the nine frame-zero manual world positions
and RGB/masks through frame 120. The later manual identity trajectory and
strict CoTracker3 comparator were opened only after the prediction archive and
report were hash-sealed.

## Frozen Result

| Quantity | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Candidate support over finite target point-frames | at least 50% | 5/38 = 13.16% | no |
| Candidate position RMSE | at most 5 mm | 67.43 mm | no |
| Candidate final-frame RMSE | at most 8 mm | 62.90 mm | no |
| Gain over physical carrier on shared support | at least 10% | -106.10% | no |
| Gain over strict CoTracker3 on shared support | at least 20% | undefined; zero shared support | no |

The physical carrier has 63.37 mm RMSE over all 38 finite target point-frames.
On the five point-frames admitted by the candidate, its RMSE is 32.72 mm,
versus 67.43 mm for the rendered-AllTracker observation. Thus the candidate
more than doubles error on exactly matched support.

The candidate covariance diagnostic is also negative but very small-sample:
mean NEES is 7.57 and empirical 90% ellipsoid coverage is 0/5. This is not a
calibration estimate.

## Support Audit

The failure is not primarily caused by a total lack of tracker responses:

- 189 camera/identity/frame queries were attempted.
- 189 had physical render support.
- 174 landed inside the target object mask.
- 157 passed the 0.5 tracker-quality threshold.
- 163 passed the 5-pixel forward/reverse cycle threshold.
- 148 passed all per-view checks jointly.
- 57 identity-frames retained at least two distinct cameras.
- Only 5 survived the frozen 3-pixel multiview reprojection threshold.

For those 57 multiview candidates, reprojection error has median 3.779 pixels,
90th percentile 5.128 pixels, and maximum 5.479 pixels. All five accepted
observations belong to one of the nine identities. Their prior reliability is
only 0.125 to 0.158 even before any physical innovation is considered.

The direct pairwise tracker therefore produces many locally plausible
correspondences that do not form a sufficiently consistent metric multiview
observation. The few observations admitted by strict geometry are still
substantially less accurate than the physical carrier. This is consistent with
the broader camera-only evidence: internal tracker confidence and reprojection
consistency cannot by themselves rule out coherent correspondence or frame
bias.

## Scorer Amendment

The first score invocation validated the sealed prediction and then opened the
hash-locked CoTracker3 archive, but stopped before computing or writing any
metric. Its quality tensor has the intended prefix-only shape
`(3, 121, 7793)`, whereas the packed triangulated trajectory has full shape
`(173, 7793, 3)`. The frozen parser had incorrectly required equal frame
dimensions.

The amendment at commit
`5e3f0b0173b6f903ad59d48d78862e26a6f093ca` accepts a prefix-only quality
tensor when it covers every scored frame and its track dimension matches.
It changes no prediction, identity, comparator, support rule, threshold,
metric, covariance, or gate. Twelve focused tests pass.

## Provenance

Prediction artifacts:

```text
prediction_report.json  4d8424d2f9e62ddfd768852d5511265892ee1fe58204c1ec881cac0f1953b516
prediction.npz          61ab6c7ae07d96cb104cb4211a07b6f5eb08222be502755edd98654c47ff637f
PREDICTION_SEAL         339bfff37115d2aacc7633726b18f6b88178158ca32395510abc30495e7fc552
score.json              076fb5b2f55dc0be3e0a260b87c9d15eb42fdced17c78bbc70c762f41a3ec4ee
```

The frozen information-boundary flags record:

- no RGB frame after 120;
- no later manual identity before the prediction seal;
- no CoTracker3 comparator before the prediction seal;
- no held-v8 access.

## Consequence

Do not tune this result by relaxing reprojection, cycle, quality, camera-count,
frame, identity, or covariance settings. The experiment already grants oracle
frame-zero material associations; failure at this easier level is evidence
against spending more effort on automatic query selection for the same
render-to-real AllTracker interface.

The next Bayesian-PhysTwin work should stay with the physically and
action-supported guarded online-belief route: preserve the unchanged physical
fallback, admit camera updates only under independently frozen support and
baseline-relative regret control, and model shared camera/frame bias
explicitly. Camera-only persistent fields, background gauges, first-order
photometric correction, and this direct rendered correspondence have now each
failed their locked source gates.
