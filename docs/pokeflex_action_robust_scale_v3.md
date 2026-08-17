# PokeFlex Repeated-Action Robust Scale V3

## Question

Can repeated interactions identify a transferable magnitude for the guarded
Bayesian state correction, beyond the globally validated `0.125` scale?

The first source interaction suggested large object-specific differences. A
fixed per-object map improved the next twelve-take cohort by `0.61233%` over the
global correction on average, but it regressed `3dPrintedPyramid` by `1.1402%`.
The post-open scale audit showed why: Pyramid preferred multiplier `0.5` on one
action and `4.0` on the next. One action was insufficient to identify a
transferable magnitude.

## Frozen method

For every physical object, v3 evaluates the same multiplier bank

```text
0.5, 1.0, 1.5, 2.0, 3.0, 4.0
```

on two opened source interactions. It chooses the multiplier that maximizes the
minimum relative improvement over multiplier `1.0` across both actions. Mean
gain, distance to multiplier one, and smaller magnitude break exact ties. Since
multiplier one is in the bank, the selected source lower envelope cannot be
negative. Conflicting evidence therefore returns the validated global scale
exactly.

| Physical object | Multiplier | Effective scale | Selection |
| --- | ---: | ---: | --- |
| `3dPrintedCylinder` | 3.0 | 0.3750 | maximin |
| `3dPrintedPizza` | 0.5 | 0.0625 | maximin |
| `3dPrintedPyramid` | 1.0 | 0.1250 | global fallback |
| `Beanbag` | 4.0 | 0.5000 | maximin |
| `FoamCylinder` | 3.0 | 0.3750 | maximin |
| `FoamHalfSphere` | 2.0 | 0.2500 | maximin |
| `Pillow` | 2.0 | 0.2500 | maximin |
| `PlushDice` | 4.0 | 0.5000 | maximin |
| `PlushMoon` | 4.0 | 0.5000 | maximin |
| `PlushTurtle` | 4.0 | 0.5000 | maximin |
| `PlushVolleyball` | 1.0 | 0.1250 | global fallback |
| `Sponge` | 1.5 | 0.1875 | maximin |

The 24 source actions contain zero regressions relative to the global scale and
have `1.28878%` mean relative improvement. Synthetic controls pass 12/12
consistent-signal detections and produce 0/12 non-default selections under
matched conflicting placebos.

The calibration canonical digest is
`78d3c74e4246ec6b69cbcfe113ed04324bf1a9f49d543194df8a7a87d7f09157`;
its file SHA-256 is
`96fe0046d15dfdd150b3f2f695b678a5b2b8a6acd790978624b120f6fa6408b0`.

## Prospective panel

After excluding both earlier fresh12 campaigns, eight untouched public takes
remain across six objects. The frozen v3 salt selects one take per object. The
freshness audit binds the current 379-ref Git snapshot, both server provenance
scans, and every selected ZIP hash. It found no unregistered exact-ID exposure.

Prediction emits three arms from identical checkpoint and registration inputs:

1. released PokeFlex checkpoint;
2. globally validated correction scale `0.125`;
3. repeated-action robust per-object scale.

All three use only Kinect and robot history through frame `f-1`. Unsupported
updates return the released checkpoint byte for byte. Target meshes remain
unread until all six prediction seals pass one barrier. No failed take may be
replaced.

The v3 candidate advances only if it improves the released checkpoint and the
global arm, has a paired object-bootstrap upper difference below zero, and has
no object-level regression versus either reference. The published `6.498 mm`
PokeFlex result remains contextual because this six-take panel is not the
published eighteen-object validation split.

The target protocol canonical digest is
`3e85d8fc89b16cdc2aceb13e9bf49d0c9e47f0a2761550323ec13b9b1bda8157`;
its file SHA-256 is
`173434fe5916c57dd4e8809f098152b096c6cf09e7efdf37190b488ee5cc7263`.

## Boundaries

- Both source interactions per object are opened development data.
- The third-panel exact takes are prospective, but the physical objects are not
  unseen.
- This experiment changes only correction magnitude. It does not validate a new
  observation model, uncertainty calibration claim, or direct table-SOTA claim.
- No held-v8 artifact, process, identity, query, barrier, or outcome is part of
  this protocol.
