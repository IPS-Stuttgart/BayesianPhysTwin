# Dynamic Direct-Depth Admission V9

## Question

Can direct metric-depth endpoints identify a genuinely action-aligned object
response before a sparse state update is constructed?

V8 established that direct depth can support all scheduled identities, but its
single late endpoint was almost exactly static and the nonzero correction was
harmful. V4 established that requiring projected camera-tangent response before
measurement is too restrictive. V9 keeps direct metric depth and replaces the
fixed endpoint with a causal stopping rule.

## Causal Stopping Rule

The registered endpoint pairs are:

```text
(13, 19) -> (32, 38) -> (51, 57)
```

At each pair, the runner:

1. certifies and decodes cameras only through the proposed update frame;
2. selects nine physically active and three near-static sentinel identities;
3. requires at least three depth views at both endpoints;
4. estimates one shared endpoint bias from all three sentinels;
5. removes that nuisance from the active endpoint displacements;
6. tests action alignment in three balanced spatial groups;
7. stops at the first admitted pair.

If a pair rejects, the process may wait for the next registered update. It may
not inspect that later prefix before making the earlier decision.

## Admission

An endpoint requires:

- all sentinel identities to support a usable common-bias estimate;
- at least six of nine active identities after reliability and association
  gates;
- at least 2 mm of measured actuator displacement;
- at least two of three spatial groups to pass;
- within a passing group, at least two supported identities, at least 2 mm
  physical response RMS, at least 1 mm observed response RMS, response gain in
  `[0.1, 3.0]`, a conservative gain lower bound of at least `0.05`, direction
  cosine at least `0.5`, and at least two-thirds positive directional support.

Dense pixels and views are not counted independently. Direct-depth views use
covariance intersection plus between-view scatter, and each spatial group has
an effective-information cap of three.

## Source Gate

The seven cases and physical archive hashes are frozen in
`configs/sota/deform360_dynamic_direct_depth_admission_source_v9.json`. At
least four cases must admit. During this gate:

- no state update is constructed;
- no hidden identity is loaded;
- no future metric or object observation is read;
- no target, V1 sealed, or held-v8 artifact is touched.

If fewer than four cases admit, this family stops. Thresholds, endpoint pairs,
and support requirements may not be relaxed on these cases.

If the gate passes, a separate commit must freeze candidate construction,
baseline-relative regret control, exact fallback, disjoint hidden scoring, and
the source outcome protocol before any future outcome is opened.
