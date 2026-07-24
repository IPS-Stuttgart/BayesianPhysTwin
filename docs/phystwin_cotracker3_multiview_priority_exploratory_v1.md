# CoTracker3 redundant-view identity routing

Run date: 2026-07-24

Status: post-open identity improvement; two-metric transfer gate rejected.

## Question

The Prob4D and Deform360 diagnostics showed that dense camera evidence is
strongly correlated and that two calibrated views can agree on a coherent but
wrong 3D point. This experiment asks whether requiring persistent support from
three camera views can improve the causal CoTracker3 observation path without
using a PhysTwin residual as prior reliability.

The fixed candidate starts from source-camera RGB-D tracks. A graph identity
uses only three-view triangulation when valid three-view observations cover at
least 40% of its released prefix. Every other identity retains source depth.
The threshold was selected on the already-open `single_lift_cloth` diagnostic
and then held fixed. It is not an independent or confirmatory experiment.

## Information boundary

- CoTracker3 decodes only frames in `[0, train_end)`.
- Future cue rows are neutral and all 19 transferred cue hashes pass the
  leakage checks in the archived cue manifest.
- Exact archived graph identities define association.
- Cue quality, cycle consistency, mask support, reprojection error, and
  persistent three-view availability are residual-independent.
- The physical innovation is handled once by the existing robust endpoint
  filter.
- Released manual tracks are evaluation-only.
- The object-disjoint MatPhys spring-family replay is fixed before this
  observation experiment.

The protocol and analysis were checksum-locked before the other 16 cases were
scored. The linked analyzer refuses a changed threshold, case order, primary
arm, or shared comparator setting.

## Frozen PhysTwin-19 result

The co-primary comparison uses the causal relative-cap arm selected from
prefix-only point-cloud Chamfer.

| Observation path | Future CD | Future manual track |
| --- | ---: | ---: |
| CoTracker3 source RGB-D | 8.160 mm | 20.192 mm |
| Three-view identity priority | 8.251 mm | 19.331 mm |
| Change | +1.12% | -4.26% |

The candidate improves or ties CD in 7/19 cases and manual-track error in
13/19, but improves or ties both in only 4/19. Equal-physical-object cluster
bootstrap intervals for candidate minus source are:

- CD: `[-0.061, +0.141] mm`;
- manual track: `[-1.233, -0.004] mm`.

Thus the identity gain transfers across object clusters, but the dense-geometry
cost does not clear zero. The fixed graph-smoothed 60 mm diagnostic has the
same tradeoff: CD worsens by 1.22% while track error improves by 3.47%.

The locked advancement gate fails. At `8.251/19.331 mm`, this candidate also
does not beat the published `8/15 mm` operating point. It must not be sent to a
fresh confirmatory cohort in its present form.

## Interpretation

Three-view redundancy is useful identity evidence, not a complete observation
replacement. The largest track gains occur on cloth interactions, while hard
replacement can move the reconstructed surface enough to hurt Chamfer.
Two-view RGB-D agreement was explicitly rejected during development because it
retained the catastrophic sign error on the diagnostic landmark.

The next low-capacity hypothesis is therefore readout-aware fusion:

1. retain source-depth motion in the local surface-normal direction;
2. admit only the three-view correction component tangent to the initial
   material surface;
3. require both channels at an update and otherwise fall back exactly to source
   depth;
4. keep the same 40% identity-availability rule and run no threshold sweep.

This is post-open mechanism development. Even a positive result will require a
genuinely fresh-object preregistration before any state-of-the-art claim.

Machine-readable evidence is under
`results/sota/diagnostics/phystwin_cotracker3_multiview_priority_exploratory_v1/`.
