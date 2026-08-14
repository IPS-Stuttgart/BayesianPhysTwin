# Volumetric Newton MPM source gate v2 technical result

## Decision

**TECHNICAL FAIL / no outcome scoring.**

All eight frozen v2 candidates were retained as technical failures before fit,
validation, or future object outcomes were supplied to the scorer. The
prediction runtime successfully generated each first driven trajectory, then
rejected its static query map because the protocol expected the map maximum
computed on float64 preprocessing particles rather than the production
runtime's float32 particles.

The expected value was `0.025976117725161844 m`; the production value was
`0.0259761295949988 m`, an absolute representation difference of
`1.1869836957084656e-8 m`. This is not model-performance evidence and does not
authorize inspecting any object outcome.

## Disposition

The v2 grid remains immutable. Protocol v2.1 changes only the protocol ID and
the expected query-map maximum to the value produced by the already-frozen
float32 runtime. It leaves the source files, split, parameter bank, contact and
mass model, selection rule, gate thresholds, denominator, and exact-fallback
policy unchanged.

Compact evidence:

- `source-custody.json`: SHA-256
  `a7067fefada1ac09c39953c21666bade5d758c39d34a7866b2a19e859ce3f6c6`;
- `newton-grid.json`: SHA-256
  `a9bf297f5ae9b1774fc51eef6a5986f9945526fd48abbd715422876c2bc75eae`;
- successful candidates: `0/8`;
- technical failures: `8/8`;
- `object_outcome_artifact_read=false`; and
- `target_or_held_out_artifact_read=false`.

No v2 prefix result or future result exists.
