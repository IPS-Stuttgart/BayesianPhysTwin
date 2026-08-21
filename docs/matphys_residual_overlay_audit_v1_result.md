# MatPhys Residual-Overlay Audit

## Status

This is a matched, post-open exploratory audit of artifacts that were already frozen by
the object-disjoint MatPhys LOO22 experiment. It answers a narrow question:

> Does the causal residual-correction layer that helped PhysTwin, DEFORM, and
> PyElastica also improve trajectories selected from the MatPhys proposal family?

The answer is **yes**, with an important qualification: the Bayesian anchor does not
beat the simple last-residual point control on aggregate point error.

This is not an independent reproduction of the published MatPhys system. In this
repository, MatPhys proposes an object-disjoint spring field and official PhysTwin/Warp
produces the physical trajectory. The audit changes neither the family selector nor
the future predictions. It is not a fresh confirmation, calibration result, or SOTA
claim.

## Matched Design

The primary comparison contains the eight interactions where the frozen selector chose
a nonzero MatPhys spring proposal. The remaining 14 LOO22 cases selected the exact
`alpha_0000` PhysTwin family and are reported only in the secondary 22-case analysis.

Every primary trajectory was evaluated under four already frozen conditions:

| Name | Frozen condition |
| --- | --- |
| Raw backbone | Prefix-selected MatPhys spring proposal replayed by official Warp |
| Bayesian anchor | Frozen causal Bayesian residual anchor on that same trajectory |
| Last residual | Frozen causal last-prefix-residual point control |
| Operational selector | Existing prefix-only choice among the three conditions |

The family selection was object-disjoint and made before future opening. The overlay
used permitted prefix manual 3D tracks, so it is online-supervised. No future frame or
metric was used to change the family, overlay, rank, or selector for this audit.

## Primary Result

Equal-case means on the eight nonzero-MatPhys cases are:

| Method | CD (mm) | Change | Track (mm) | Change | Joint wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw MatPhys/Warp backbone | 10.689 | - | 18.601 | - | - |
| Bayesian anchor | 9.598 | **-10.21%** | 16.322 | **-12.25%** | 6/8 |
| Last residual | **9.469** | **-11.41%** | **16.216** | **-12.82%** | 6/8 |
| Operational selector | 9.594 | -10.24% | 16.325 | -12.23% | 6/8 |

The Bayesian anchor improves CD in 7/8 cases and track error in 7/8 cases. Its worst
case changes are +0.81% CD and +2.85% track. The last-residual control has slightly
better means, but only 6/8 track wins and a worse maximum track regression of +7.98%.

The object-cluster and frame-block bootstrap is limited by only five physical-object
clusters. For the Bayesian anchor, its 95% interval is `[-12.74%, -1.31%]` for CD and
`[-14.24%, +0.39%]` for track error. The CD interval excludes zero; the track interval
narrowly crosses zero.

The improvement persists through the forecast, while becoming smaller at long
horizons:

| Horizon | Raw CD | Bayesian CD | Raw track | Bayesian track |
| --- | ---: | ---: | ---: | ---: |
| Early | 7.873 mm | 6.225 mm | 13.933 mm | 11.585 mm |
| Middle | 10.388 mm | 9.372 mm | 19.372 mm | 17.100 mm |
| Late | 14.164 mm | 13.585 mm | 22.838 mm | 20.619 mm |

## Secondary Result

Across all 22 cases, including the 14 exact PhysTwin fallbacks, the raw selected-family
stack is 11.389 mm CD / 21.300 mm track. The Bayesian anchor reaches 10.245 / 19.057
mm, while last residual reaches 10.195 / 18.936 mm. These full-cohort values show the
operational fallback stack, not a direct MatPhys-only comparison.

## Interpretation

The residual layer is usefully backend-portable: it materially improves the physical
trajectories on which the MatPhys spring proposal was actually accepted. This closes
the immediate transfer question positively.

It does **not** establish Bayesian point-mean novelty. Last residual remains the stronger
aggregate point control, while the Bayesian anchor is more consistent case by case. A
credible next experiment must therefore test what the Bayesian method is meant to add:
calibrated uncertainty, guarded fallback, and decision quality, alongside point error.

The next clean test is a newly frozen accepted-MatPhys cohort comparing raw MatPhys,
last residual, Bayesian anchor, and conformal or calibrated predictive distributions.
This opened LOO22 cohort must not be used for further method selection.

## Evidence

- Configuration: `configs/sota/matphys_residual_overlay_audit_v1.json`
- Result: `results/sota/diagnostics/matphys_residual_overlay_audit_v1/result.json`
- Result SHA-256:
  `db5d7ce2bd78aa0a8150c10c026a4ab63bf9a730d94aa70bb758e6101e5e2388`
- Frozen family selection SHA-256:
  `5eedb6cb5a747b856c0af696c5029038a8022f00828f43295f201578a4494890`
- Frozen future summary SHA-256:
  `6560317dbaebaf99b46328e526febf4a276d6183163284bc56b0d473dfa5b9d9`

No held-v8 or fresh target artifact was accessed.
