# Deform360 reusable-twin state admission v1

The penguin arm passed the frozen rigid-state admission before any source future
object outcome was opened. Cable and scarf remain rejected by the preceding v5
source-mask transfer gate and were not revived or retuned.

## Frame-zero state gate

| Episode | Symmetric CD (mm) | Readout RMSE (mm) | Effective reliability | Supported target fraction | p99 initial edge strain |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.404 | 1.812 | 0.746 | 1.000 | 5.42e-8 |
| 3 | 6.750 | 4.078 | 0.715 | 0.990 | 5.42e-8 |
| 4 | 11.192 | 3.931 | 0.716 | 1.000 | 5.42e-8 |
| 6 | 8.641 | 4.158 | 0.710 | 0.998 | 5.42e-8 |
| 7 | 7.209 | 3.969 | 0.714 | 0.994 | 5.42e-8 |
| 9 | 12.237 | 3.694 | 0.718 | 1.000 | 5.42e-8 |

The shared graph has 384 nodes and 2,598 object springs. Episode placement is a
proper rigid transform, so the residual strain is only float32 roundoff. The
target geometry remains external to the simulator through the frame-zero readout.

## Reference Warp gate

All fixed-reference (`Y=10000`, drag `10`, dashpot `100`) official Warp rollouts
were finite for 76/76 frames and built the frozen dynamic controller attachment.

| Episode | Controller springs | Contact onset frame(s) | p99 rollout strain | Maximum rollout strain |
|---:|---:|---|---:|---:|
| 1 | 1 | 18 | 0.0446 | 0.2108 |
| 3 | 16 | 0 | 0.2303 | 0.8291 |
| 4 | 16 | 0 | 0.2948 | 0.6776 |
| 6 | 10 | 50, 34 | 0.1569 | 0.4470 |
| 7 | 5 | 25, 20 | 0.1242 | 0.5657 |
| 9 | 16 | 0 | 0.2420 | 0.7003 |

The admission criterion concerns finite execution and p99 strain at or below
0.50. Maximum single-edge excursions are retained as diagnostics and are not
hidden by the p99 gate.

## Outcome-blind physical grids

All 18 frozen parameter tuples have matched driven and zero-action response
archives for every source episode. Every stored array was checksum-validated.

| Episode | Response count | Fit-grid result SHA-256 |
|---:|---:|---|
| 1 | 18 | `6205bc88096fb311952bf37fa2242814e67f47c21efccb96ca03b19fcdabe2ec` |
| 3 | 18 | `cdd8b4100884512757b55a08d98a18c5c7250909349c2f4a69d4a90975d2b81f` |
| 4 | 18 | `8e18dda3b48d73a8bdc7e8be648b3b6d5b1ccf79c9b516ddcf839f00e315ba6e` |
| 6 | 18 | `21425a0dab23925e1e5af1d127c399fd8eeefa3b5a5533053fd54ee06b63da98` |
| 7 | 18 | `e82ddcdb5b2aa397c7a488e07ecfa51c7c05e72d77f3042e82b1a44612c0cfd1` |
| 9 | 18 | `470a8ec50a68d88d884c4c5ac72768092965b323681c5b4f081dce2d402d3ffd` |

The state addendum hash bound into every response is
`33e9171bd85f8a646b8cfebf76f150398450c84676b4f1f0a4c358dbf119978f`.
No post-initial object frame, future tactile signal, simulator residual, source
future outcome, held media, or held outcome was used in state construction or
response generation.

This is an admission result, not an accuracy result. It permits the source-only
physical selection and pooling controls; it does not permit held-episode access
or a state-of-the-art claim.
