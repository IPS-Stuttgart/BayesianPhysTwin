# Deform360 depth-supported frame-zero post-open result

## Scope

This milestone applies the source-frozen v2 depth-supported initializer to the
four already opened frame-zero failures and then runs an explicitly
diagnostic-only official-Warp test on the three recovered cases. These cases
are not independent confirmation. No future object RGB, dense future
geometry, tracks, target metric, or reserved target object is read.

The post-open audit is archived at
`results/sota/deform360_frame_zero_depth_postopen_v2/postopen_audit.json`
(file SHA-256
`449b7238305656554bff20a990c4ec0e5d3d622f04e029079fbd54176f3b7eaa`,
canonical result SHA-256
`4a3d5fbb3a4a87c6e9a5dcd3254add8e7deff5e6ac096517229b7fb47626d9c5`).
The physical diagnostic is archived at
`results/sota/deform360_frame_zero_depth_physical_v2/physical_audit.json`
(file SHA-256
`f4f70ca1af4b034930651308c4bfc4380352b0620b4078fc02a57a826dcb9675`,
canonical result SHA-256
`baac9016acf0555a87252b4b5163b88713fa4968d63cef936b445c7d1298e614`).

## Coverage gate

| Case | V1 hull points | V2 depth-supported points | Recovered |
| --- | ---: | ---: | ---: |
| `015-airbag-cloth-ep0006` | 3,495 | 1,109 | yes |
| `100-puppet-ep0009` | 3,479 | 1,334 | yes |
| `160-hose-ep0001` | 2,675 | 1,193 | yes |
| `174-chain-ep0001` | 264 | below 128 | no |

The locked gate requires 4/4 recoveries. V2 recovers 3/4, so its post-open
coverage gate fails. The 50 mm depth tolerance, one-view support rule, and
128-node floor are not changed after seeing this result.

## Physical diagnostic

The three recovered geometries all construct automatic `warp_twin` backends,
preserve frame-zero material identity exactly, and complete finite rollouts.
The unchanged gate requires zero-action p99 displacement no larger than
250 mm and maximum displacement no larger than 500 mm.

| Case | V1 p99 | V2 p99 | V2 max | V2 passed |
| --- | ---: | ---: | ---: | ---: |
| `015-airbag-cloth-ep0006` | 297.35 mm | 332.53 mm | 359.30 mm | no |
| `100-puppet-ep0009` | 300.76 mm | 310.90 mm | 327.76 mm | no |
| `160-hose-ep0001` | 223.62 mm | 183.44 mm | 201.92 mm | yes |

Only one of three cases passes. Depth pruning improves `hose` but worsens the
two v1-unstable cases. The physical diagnostic therefore also fails.

## Decision

Do not promote the depth-supported initializer and do not open a fresh-object
cohort for it. The combined v1 and v2 evidence rejects the hypothesis that a
broad silhouette shell is the main cause of the unstable automatic twins.
The v2 implementation and source-positive result remain useful negative
evidence: better agreement with a frame-zero reconstruction does not imply a
better physical equilibrium.

Stop frame-zero geometry development on these opened cases. Keep admitted
original Splatfacto twins byte-identical and route failed automatic twins to
the unchanged fallback behavior. The next method effort belongs in guarded
state/discrepancy belief updates with explicit model-form uncertainty, not in
post-hoc geometry thresholds or stronger spring-family searches.
