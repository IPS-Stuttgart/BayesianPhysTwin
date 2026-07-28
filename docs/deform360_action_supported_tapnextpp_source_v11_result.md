# Action-Supported TAPNext++ V11 Source Result

## Decision

The frozen source gate **failed**. Do not construct the proposed state update,
tune this provider on the eight opened cases, or evaluate it on fresh objects.

| Source result | Frozen value | Required | Pass |
| --- | ---: | ---: | :---: |
| Sealed provider predictions | 8 / 8 | at least 6 | yes |
| Supported endpoint identities | 5 / 64 (7.81%) | at least 75% | no |
| Cases with at least 50% support | 0 / 8 | at least 6 | no |
| Scored cases | 2 / 8 | at least 6 | no |
| Object-balanced provider RMSE | 6.567 mm | at most 15 mm | no |
| Object-balanced late RMSE | 6.618 mm | at most 15 mm | no |
| Exact-persistence RMSE | 0.762 mm | comparator | -- |
| Relative gain over persistence | -761.96% | at least +10% | no |
| Case wins over persistence | 1 / 8 | at least 5 | no |

The absolute RMSE conditions are marked failed because the preregistered
minimum of six scored cases was not met, even though the two observed means are
below 15 mm.

## What V11 Established

The action-support selector repaired the mechanical V10 query-budget failure:
all eight cases produced complete eight-query schedules without predicted
displacement. That did not translate into a competent observation provider.

Endpoint support by locked case was:

```text
0, 0, 0, 0, 2, 3, 0, 0 out of 8 queries
```

Only `059-shoe` and `061-cup` could be scored. On `061-cup`, TAPNext++ was
slightly better than persistence (0.632 versus 0.780 mm). On `059-shoe`, it
regressed sharply (12.502 versus 0.744 mm). The panel-wide result therefore
combines two distinct blockers:

1. frame-zero action-supported identities rarely retain accepted multiview
   support through frame 57; and
2. where support survives, the selected action-only rows can be so static that
   exact persistence is already submillimetric.

Action support is useful intervention localization, but it is not evidence
that an identity will move enough to justify a visual state update.

## Uncertainty

The raw covariance covered all five admitted endpoint residuals at the nominal
90% chi-squared threshold, with mean NEES 1.317. Five correlated rows from two
objects cannot establish calibration. This is reported only as a diagnostic;
no source calibration or state likelihood was fitted.

## Evidence Boundary

Implementation and gates were frozen and pushed at
`e1640bdf58306420f9ee6489c1a2c80eb3e49d3d`. A clean native checkout passed
1,312 tests with 28 skips, and changed-file Ruff passed.

All eight tracker/multiview predictions were sealed before the source identity
operator ran. The prediction barrier has SHA-256
`06767567f48fe9b10c21eeaa9fe52bd6ab4d15be402b18834128a59b39a48280`.
The canonical result SHA-256 is
`9321bcc81e8833c6c80904f1e984dd89b44026540571764d78fb12a4ae39b0a6`.

Three environment attempts on the smoke case stopped before identity scoring:
the first lacked OpenCV, the second lacked HDF5 bindings, and the third lacked
TorchVision. The existing Deform360 processing environment contained the
already validated Torch/TorchVision/OpenCV/HDF5 stack and produced the complete
frozen run without a code or method change. The failed directories remain on
the server as provenance.

No state update, future prediction metric, V1 sealed target, or held-v8
artifact or process was read or modified.

## Consequence

V11 closes action-support-only, frame-zero, fixed-identity TAPNext++ as the
next Bayesian-PhysTwin observation feeder. It does not reject TAPNext++ on
high-motion prefixes; the earlier one-case result remains accurate but
under-supported. It shows that action support alone cannot bridge the gap from
that curated high-motion result to a general source provider.

The Bayesian-PhysTwin SOTA effort should therefore keep the currently
independent guarded online-belief evaluation as the primary evidence path.
Any later tracker experiment must introduce genuinely new information, such
as a causally observed motion trigger or an independent modality, and must use
newly locked objects rather than tuning these eight opened cases.

Exact prediction and evaluation carriers are archived under
`results/sota/diagnostics/deform360_action_supported_tapnextpp_source_v11/`.
