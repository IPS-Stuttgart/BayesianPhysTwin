# Deform360 reusable-twin source admission v5

This note records the source-only admission result before any fresh held episode
is opened. All rows use frame-zero RGB, calibration, and the known robot action.
No post-initial object frame, tactile stream, simulator residual, or object outcome
was read.

| Object | Ep. | Accepted views | Hull points | Min. action contact | Result |
|---|---:|---:|---:|---:|---|
| cable | 1 | 4 | 2697 | 0.67 mm | pass |
| cable | 3 | 4 | 629 | 0.39 mm | pass |
| cable | 4 | 3 | 560 | 0.29 mm | **reject** |
| cable | 6 | 3 | 2961 | 0.08 mm | **reject** |
| cable | 7 | 4 | 846 | 0.18 mm | pass |
| cable | 9 | 4 | 585 | 0.30 mm | pass |
| scarf | 1 | 7 | 1549 | 9.64 mm | pass |
| scarf | 3 | 7 | 7532 | 17.71 mm | pass |
| scarf | 4 | 7 | 3729 | 21.87 mm | pass |
| scarf | 6 | 8 | 6238 | 32.31 mm | **reject** |
| scarf | 7 | 6 | 1020 | 16.67 mm | pass |
| scarf | 9 | 6 | 623 | 21.72 mm | pass |
| penguin | 1 | 9 | 1352 | 0.46 mm | pass |
| penguin | 3 | 9 | 1458 | 0.11 mm | pass |
| penguin | 4 | 6 | 626 | 0.78 mm | pass |
| penguin | 6 | 8 | 1006 | 2.90 mm | pass |
| penguin | 7 | 9 | 1447 | 0.42 mm | pass |
| penguin | 9 | 5 | 553 | 0.57 mm | pass |

The cable arm is rejected because its source-selected camera panel transfers in
only four of six fit episodes. The scarf arm is rejected because episode 6 never
enters the frozen 3 cm contact envelope. Neither arm may be retuned; both return
exact persistence if included in a later aggregate. Only penguin proceeds to
shared-physics fitting.

## Penguin reference smoke

Reference episode 1 produced 1,264 finite frame-zero Gaussians, a 1,143-point
prediction input, and one geometry-conditioned controller group with onset at
frame 18. The shared graph has 384 nodes and 2,598 object springs.

Both the driven and zero-action official Warp controls completed 76/76 finite
frames. The driven rollout had 21.08% maximum and 4.47% p99 relative object-edge
strain. Its final mean displacement from the zero-action rollout was 28.40 mm
(44.08 mm p95; 51.87 mm maximum), with 21.16 mm future RMS separation. This is a
material source-only response and passes the reference smoke gate. It is not an
accuracy result because no source future outcome was used in this diagnostic.

The next allowed operation is to build checksum-bound predictions for every
penguin fit episode, seal the 18-candidate source grid, and only then open source
future outcomes for shared-parameter selection. Held episodes remain sealed.
