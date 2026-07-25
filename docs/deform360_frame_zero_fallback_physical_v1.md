# Deform360 frame-zero fallback physical result

## Scope

This milestone evaluates the frozen frame-zero visual-hull fallback after its
12-object source audit. The first stage applies it to four already opened
Deform360 calibration cases whose original Splatfacto clouds failed the
128-point admission check. The second stage constructs automatic twins and
runs the official Warp backend. No future RGB, dense future geometry, particle
tracks, target metrics, or reserved target cases are used.

The post-open geometry audit is archived at
`results/sota/deform360_frame_zero_postopen_failures_v1/postopen_audit.json`
(file SHA-256
`1092eee61bbebb59eec1954fecedb289a7913882b48d36c28ee8a2d87bc1b305`,
canonical result SHA-256
`f819e37b2ccac3959349298ba3824a82d342902c9d357ddc53d5f548278d06df`).
The physical audit is archived at
`results/sota/deform360_frame_zero_fallback_physical_v1/physical_audit.json`
(file SHA-256
`9fb90afe842cf5ed220b7d82cb025f8d4e212b6a487ae47bb89203f3b36a2b25`,
canonical result SHA-256
`858e03515ab8a45bd2ffbe6159ffbd3c7e042f00eed333ecdab53a0575d3d3ff`).

## Geometry recovery

| Case | Original points | Fallback points | Recovered |
| --- | ---: | ---: | ---: |
| `015-airbag-cloth-ep0006` | 48 | 3,495 | yes |
| `100-puppet-ep0009` | 63 | 3,479 | yes |
| `160-hose-ep0001` | 25 | 2,675 | yes |
| `174-chain-ep0001` | 103 | 264 | yes |

The source-frozen fallback recovers all four missing material clouds without a
new fit or threshold choice. This establishes operational geometry coverage,
not physical accuracy.

## Physical admission

The locked physical gate requires all four cases to produce an automatic
`warp_twin`, preserve frame-zero material identity exactly, remain finite, and
stay below 0.25 m p99 and 0.50 m maximum zero-action displacement.

| Case | Graph vertices | Zero-action p99 | Zero-action max | Passed |
| --- | ---: | ---: | ---: | ---: |
| `015-airbag-cloth-ep0006` | 1,024 | 297.35 mm | 439.89 mm | no |
| `100-puppet-ep0009` | 1,024 | 300.76 mm | 317.11 mm | no |
| `160-hose-ep0001` | 1,024 | 223.62 mm | 253.00 mm | yes |
| `174-chain-ep0001` | 264 | 106.42 mm | 111.43 mm | yes |

All four automatic twins are constructed, preserve exact material identity,
and complete finite official-Warp rollouts. Only two of four pass the locked
stability thresholds, so the physical gate fails.

## Decision

The strict silhouette hull is retained as a frozen geometry-coverage result
but is not promoted as a physical initializer. The broad hull surface can
produce materially unstable twins even when point count, connectivity, and
multiview silhouette support pass.

Any successor must be a separately versioned source-developed method. The
next candidate may use only frame-zero metric depth to prune the silhouette
surface, must leave admitted original clouds byte-identical, and must pass a
new physical gate before any fresh-object or reserved-target use. The failed
bias-aware prospective result and its target seals remain unchanged.
