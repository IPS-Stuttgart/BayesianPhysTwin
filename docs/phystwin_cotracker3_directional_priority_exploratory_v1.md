# CoTracker3 directional endpoint update

Run date: 2026-07-24

Status: post-open track improvement; joint transfer gate rejected.

## Motivation

The frozen hard three-view routing result improved manual identity error but
slightly worsened Chamfer distance. The first follow-up retained source-depth
motion along a local surface normal and admitted the multiview correction only
in the tangent plane. It also required both channels to be valid and preserved
the source validity mask exactly.

That point-fusion arm failed on the already-open `single_lift_cloth` smoke:

| Development arm | Future CD | Future manual track |
| --- | ---: | ---: |
| Source RGB-D | 11.984 mm | 50.730 mm |
| Joint-valid tangent point fusion | 11.984 mm | 50.730 mm |

Only one identity had joint source and multiview validity at the final prefix
frame. Running all 19 cases for that arm was therefore not justified.

## Directional posterior

The successor treats the two channels as measurements of different geometric
subspaces instead of constructing another pseudo-3D point:

- nonpriority identities retain the existing full 3D source update;
- priority identities receive a one-dimensional source-normal innovation;
- the same identities receive a two-dimensional redundant-view tangent
  innovation;
- each innovation enters one robust mixture likelihood;
- the graph smoother receives the largest endpoint-covariance eigenvalue in
  square metres.

The priority threshold (40%), three-camera requirement, 16-neighbor initial
surface basis, filter parameters, physical backbone, causal selector, case
order, and advancement gates were checksum-locked after the one development
case and before the other 18 directional outcomes were scored.

The directional development smoke improved both metrics:

| Development arm | Future CD | Future manual track |
| --- | ---: | ---: |
| Source RGB-D | 11.984 mm | 50.730 mm |
| Directional endpoint | 11.893 mm | 49.445 mm |
| Hard three-view routing | 11.683 mm | 49.308 mm |

## Frozen PhysTwin-19 result

The co-primary arm remains `causal_selected_dense_relative_cap`.

| Observation path | Future CD | Future manual track |
| --- | ---: | ---: |
| Source RGB-D | 8.160 mm | 20.192 mm |
| Hard three-view priority | 8.251 mm | 19.331 mm |
| Directional endpoint | 8.192 mm | 19.492 mm |
| Directional vs source | +0.40% | -3.47% |
| Directional vs hard priority | -0.71% | +0.83% |

Against source depth, the directional arm improves or ties CD in 10/19 cases
and track error in 14/19, but both metrics in only 7/19. Equal-physical-object
cluster bootstrap intervals for candidate minus source are:

- CD: `[-0.022, +0.056] mm`;
- manual track: `[-1.038, -0.049] mm`.

Thus the track improvement transfers across object clusters, while the small
CD regression remains unresolved. The fixed graph-smoothed 60 mm diagnostic
has the same tradeoff: `+0.95%` CD and `-0.96%` track.

The locked advancement gate fails all three required parts. The method must not
be sent to a fresh confirmatory cohort in this form. At `8.192/19.492 mm`, it
also does not beat the published `8/15 mm` operating point.

## Corrected interpretation

The protocol's provisional development interpretation expected directional
updates to remain available when source depth was missing. The completed audit
shows that this did not occur: across all 19 cases,
`multiview_tangent_updates_without_source_count` is zero. Three-view validity
is a strict subset of source-channel validity for these extracted cues.

The improvement therefore comes from assigning source and multiview
innovations to orthogonal likelihood subspaces and propagating directional
covariance, not from increasing endpoint support. This is still useful:
directional filtering reduces the hard-routing CD penalty by roughly two
thirds while retaining most of its identity gain. It does not solve the
remaining geometry-identity tradeoff.

## Decision

Stop threshold and fusion development on PhysTwin-19. Preserve the directional
filter as an observation-model component, but require a source-calibrated,
baseline-relative regret guard with exact fallback before prospective use.
That guard must be calibrated on independent source objects, not tuned against
these opened outcomes.

The broader Prob4D/Deform360 conclusion remains unchanged: camera-internal
agreement cannot identify coherent common-mode bias. A genuinely stronger
Bayesian-PhysTwin observation update needs physical/action support and,
preferably, an independent depth, tactile, or other gauge modality.

Machine-readable evidence is under
`results/sota/diagnostics/phystwin_cotracker3_directional_priority_exploratory_v1/`.
