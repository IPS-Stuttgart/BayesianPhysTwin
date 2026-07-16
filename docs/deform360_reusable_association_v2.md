# Reusable Deform360 material association v2

## Motivation

The frozen contact-trust v1 method failed before dynamics scoring. Independent
per-view masks produced 26,402--31,861 Gaussians where source reconstructions
contained 514--1,364, and one equal-cardinality export contained an identity
jump above the fixed 30 mm gate. Target episode 1 stayed sealed.

## Method boundary

V2 separates identity from geometric feasibility:

1. Source-reference appearance, shape, area, predicted IoU, and SAM stability
   rank at most four candidates per view.
2. Candidates below the old hard appearance threshold remain weak options;
   neither simulator state nor simulator residual participates in their prior.
3. If the top-appearance set forms a valid calibrated 3D consensus, it is kept
   exactly. Geometry cannot replace it with a different, easier-to-intersect
   object.
4. A deterministic coarse-grid search is used only when that set fails, and the
   selected masks must still pass the unchanged full-resolution visual-hull
   audit.
5. Reconstruction cardinality is gated against the source object before any
   physical fit is run.

Temporal identities use stable export order as a warm-start hypothesis, not an
axiom. Pairs moving at most 30 mm retain their indices with distance-based
confidence. Outliers are sparsely rematched one-to-one. Ambiguous rematches
carry assignment variance rather than creating false certainty, and an
unmatched point does not invalidate every other identity.

## Source evidence

Seven source-only cases across filament, sheet, and volumetric strata retain
10--12 calibrated views. The representative montages are archived in
`milestones/deform360-reusable-association-v2-source/artifacts`. Five existing
stripe-rope Splatfacto prefixes achieve 100% match fraction and at least 97.09%
effective reliable support.

This is a source competence result, not an independent result and not a SOTA
claim. The frozen calibration gate reads only first-frame and six-frame-prefix
geometry. Future CD, track error, and all target media remain inaccessible
until that conjunction passes.
