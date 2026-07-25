# Deform360 depth-supported frame-zero initializer source result

## Scope

This source-development result uses the same 12 previously opened and
exhausted Deform360 objects as the frozen visual-hull audit, balanced across
filament, sheet, and volumetric strata. It reads frame zero only. No future
RGB, dense future geometry, particle tracks, outcome metrics, or reserved
targets are used.

The candidate is revision `d98e39e` with config digest
`3b44741985052512147d33a5b5878130eba12695e59b20765f356b7e16e0a21a`.
The audit is archived at
`results/sota/deform360_frame_zero_depth_initializer_source_v2/source_audit.json`
(file SHA-256
`30c8f3ac4ddb4864fa1b8386d35f9299c996d08e811550a81bbc5a9e59124c1b`,
canonical result SHA-256
`43ea9365acd1af2873d99fa6f073522152e2f3898c8392de65f563afb6625c19`).

## Candidate

The selector preserves every admitted original Splatfacto material cloud
byte-identically. Only a sub-128-point failure activates the new fallback:

1. construct the frozen v1 strict multiview visual-hull surface;
2. project each surface node into every frame-zero rendered-depth map;
3. retain nodes whose camera-space depth agrees within 50 mm in at least one
   view;
4. require at least eight informative depth cameras and 128 retained nodes;
5. reject the fallback if these requirements are not met.

The depth maps are Splatfacto expected-depth renders. They are correlated with
the original Splat reconstruction and are used as geometric anchors only.
They are not counted as an independent modality, independent likelihood, or
new uncertainty evidence.

## Result

| Source diagnostic | Result |
| --- | ---: |
| Cases / physical objects | 12 / 12 |
| Exact admitted-path parity | 9 / 9 |
| Natural sub-128 failures recovered | 3 / 3 |
| Forced depth-supported constructions | 12 / 12 |
| Minimum retained point count | 144 |
| Paired distance improvements vs v1 hull | 11 / 12 |
| Median paired improvement | 6.35 mm |
| Mean paired improvement | 7.27 mm |
| Maximum paired regression | 0.26 mm |

All predeclared source gates pass. The result supports depth pruning over the
broad visual hull on this open development panel. Distance to the original
Splat cloud remains a diagnostic rather than ground-truth geometry accuracy,
especially in the three cases where the original cloud itself is sparse.

## Decision

Freeze v2 before applying it to the four already opened quality failures. It
must recover all four cases and pass the unchanged official-Warp stability
limits before promotion. A positive post-open result would justify a new
fresh-object protocol; it would not retroactively confirm the candidate on
the source or known-failure cases. Admitted original twins, the failed
bias-aware prospective result, and all reserved target seals remain unchanged.
