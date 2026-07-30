# RGBench Isotropic Dynamic v2

## Status

This is a new prospective protocol. It does not revive or modify the closed
RGBench v1 study, and it has no relationship to held-v8, V12, V13, or V14.

RGBench v1 closed as `predictions_incomplete` because the released 31,521-node
brown-coat mesh repeatedly crashed the native PyBullet soft-body backend. The
failure occurred before calibration or target outcomes were opened. A separate
target-free feasibility study found a deterministic, physically admissible
backbone by isotropically remeshing the released surface while preserving the
released fling-contact coordinates.

## Hypothesis

The first scientific hypothesis is not that remeshing improves accuracy. It is
that a source-derived, contact-preserving mesh can make the public physical
backend executable and deterministic without reading real object outcomes.

Only after this physical gate passes will the method test the substantive
hypothesis:

> A source-trained, action-conditioned dynamic graph discrepancy can improve
> online continuation beyond a static endpoint correction and the physical
> baseline.

The frozen static v1 method remains a predecessor and control.

## Target-Free Mesh Selection

For an original mesh with at most 13,000 vertices, v2 retains the exact source
OBJ if it passes all geometry gates. Larger or invalid meshes are remeshed with
PyMeshLab 2025.7.post1 using five iterations of explicit isotropic remeshing.
Candidate target edge lengths form a fixed 8.00--20.00 mm grid in 0.25 mm
increments. The first candidate satisfying every gate is selected.

If every remeshed candidate fails a geometry gate, a clean released source OBJ
may be retained as a fallback only when it has at most 21,000 vertices. This
exception was added after the target-free pleated-skirt audit: the original
20,279-node mesh had zero self-intersections and produced two byte-identical
physical replays, while all under-12k isotropic candidates introduced
self-intersections. The fallback does not bypass the full physical gate.

The gates require:

- 128--13,000 vertices;
- one connected component;
- edge- and vertex-manifold topology;
- orientability;
- no degenerate or duplicate triangles;
- no selected self-intersection faces;
- source-vertex to derived-triangle surface mean, 99th percentile, and maximum
  distances no larger than 5, 10, and 15 mm.

The surface metric is computed against triangles, not only against derived
vertices. This distinction was locked after a target-free cake-skirt audit:
nearest-vertex distance incorrectly reported 10.5 mm at the 99th percentile
for a 12,986-node candidate whose true point-to-triangle value was 2.54 mm.
The 13,000-node cap is the smallest round cap admitting that first clean
candidate; it remains subject to deterministic PyBullet preflight.

For fling actions, two derived vertices are snapped to the exact released
source-pin coordinates. Their indices are recomputed from the no-UV raw OBJ and
bound in the artifact. Grasp and fold retain RGBench's upstream automatic
nearest-contact rule.

Selection may read released mesh coordinates and cloth-parameter metadata. It
may not read robot outcomes, segmented point coordinates, or evaluation
metrics.

## Physical Gate

The gate runs sample 01 for every combination of seven primary garments and
three actions. Each of the 21 cases is replayed twice in PyBullet DIRECT mode.
All 42 runs must complete, and every replay pair must be byte-identical.

The runner may read point-cloud filenames to reproduce the published timestamps
and may use the released future actuator trajectory as a known intervention. It
must not parse any real point-cloud coordinate. Consequently, this gate tests
physical-backend admissibility and replay determinism, not predictive accuracy.

## Source Development

If the physical gate passes, all model-family work remains confined to the 27
source captures from `white_cakeskirt`, `brown_coat`, and `green_tshirt`.

The dynamic discrepancy bank starts from the graph-smoothed prefix correction
and adds a slope term with frozen shrinkages
`{0, 0.1, 0.25, 0.5, 0.75, 1}`. Settings are selected per action by
leave-one-garment-out source cross-fitting. Unsupported or rejected updates
fall back exactly to the physical trajectory.

Calibration opens only if the cross-fitted source result achieves at least 5%
object/action-balanced improvement over the physical baseline, does not regress
any source garment, improves at least six of nine garment/action cells, beats
the published GarmentDynamics aggregate, and beats its published cell in at
least six of nine source cells. Target garments remain sealed unless the
separately frozen calibration gate also passes.

## Comparison Boundary

The v2 candidate is an online continuation method: it observes an early prefix
of the real object trajectory. The published GarmentDynamics numbers are an
open-loop RGBench reference and therefore use less information. A lower v2
number is useful evidence of online-state-update headroom, but it is not an
equal-information SOTA comparison. Every table and claim must state this
asymmetry.
